"""YAML 설정 로딩, `--set a.b=value` 덮어쓰기, 설정 해시.

설정은 중첩 dict 하나로 다루고 점 표기 키로 읽고 쓴다. checkpoint 재개가 같은
설정에서만 일어나도록 학습 의미에 영향을 주는 키만 모아 `config_hash`를 만든다.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import runtime

#: 모든 설정 파일이 공유하는 기본값. YAML은 이 위에 덮어쓴다.
DEFAULTS: dict[str, Any] = {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "revision": None,
    "model_path": None,  # 로컬 디렉터리를 직접 지정할 때
    "precision": "bf16",  # bf16 | nf4 | int8
    "attn_implementation": "sdpa",
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1024 * 28 * 28,
    "prompt_id": "v1_image_only",
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "targets": "auto",
    },
    "train": {
        "lr": 1e-4,
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 8,
        "warmup_ratio": 0.03,
        "scheduler": "linear",  # linear | cosine | constant
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 42,
        "shuffle_options": True,
        "save_every_steps": 50,
        "save_every_minutes": 15,
        "keep_last": 2,
        # 계약에 없는 보조 키. 기본값이 있으므로 YAML에서 생략해도 된다.
        "heartbeat_every_steps": 5,
        "log_every_steps": 1,
        "num_workers": 0,
        "adam_betas": [0.9, 0.95],
    },
    "infer": {
        "batch_size": 1,
        "save_every_items": 50,
        "num_workers": 0,
    },
    "runtime": {
        # Ampere 이상에서 fp32 matmul을 TF32로 돌린다. 수치가 조금 달라지지만 학습
        # 의미를 바꾸는 설정이 아니므로 config_hash에서는 제외한다.
        "tf32": True,
    },
}

#: config_hash 계산에서 제외하는 점 표기 키. 학습 결과를 바꾸지 않는 운영 설정이다.
HASH_EXCLUDED_KEYS: tuple[str, ...] = (
    "infer",
    "train.save_every_steps",
    "train.save_every_minutes",
    "train.keep_last",
    "train.heartbeat_every_steps",
    "train.log_every_steps",
    "train.num_workers",
    "model_path",
    "attn_implementation",
    # TF32는 속도 설정이다. 이 값 때문에 checkpoint 재개가 거부되면 안 된다.
    "runtime.tf32",
)


class ConfigError(ValueError):
    """설정 파일이나 덮어쓰기 값이 잘못되었을 때 발생한다."""


# --------------------------------------------------------------------------
# dict 도우미
# --------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """중첩 dict를 재귀적으로 합친다. override가 이긴다."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_path(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """`train.lr` 같은 점 표기 키를 읽는다."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """점 표기 키에 값을 넣는다. 중간 dict는 만들어 준다."""
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def drop_path(config: dict[str, Any], dotted: str) -> None:
    """점 표기 키를 지운다. 없으면 아무 일도 하지 않는다."""
    parts = dotted.split(".")
    node: Any = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def parse_scalar(text: str) -> Any:
    """`--set` 우변 문자열을 파이썬 값으로 바꾼다."""
    lowered = text.strip()
    if lowered.lower() in {"none", "null", "~"}:
        return None
    if lowered.lower() in {"true", "yes", "on"}:
        return True
    if lowered.lower() in {"false", "no", "off"}:
        return False
    try:
        return json.loads(lowered)
    except (ValueError, TypeError):
        pass
    try:
        if lowered.lstrip("+-").isdigit():
            return int(lowered)
        return float(lowered)
    except ValueError:
        return text


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """`a.b=value` 형식 문자열들을 설정에 반영한다."""
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set 형식이 잘못되었습니다 (a.b=value): {item!r}")
        key, raw = item.split("=", 1)
        set_path(config, key.strip(), parse_scalar(raw))
    return config


# --------------------------------------------------------------------------
# 로딩과 검증
# --------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8-sig") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: 최상위가 매핑이 아닙니다.")
    return loaded


def load_config(
    path: str | Path | None,
    overrides: Iterable[str] | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """설정 파일을 기본값 위에 얹고 `--set`을 적용한다."""
    raw = load_yaml(path) if path is not None else {}
    config = deep_merge(DEFAULTS, raw)
    apply_overrides(config, overrides)
    if path is not None:
        config["_config_path"] = str(Path(path).resolve())
    if validate:
        validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """값의 범위와 형식을 확인한다. 실패하면 ConfigError."""
    precision = str(get_path(config, "precision", "")).lower()
    if precision not in {"bf16", "nf4", "int8"}:
        raise ConfigError(f"precision은 bf16 | nf4 | int8 중 하나여야 합니다: {precision!r}")
    scheduler = str(get_path(config, "train.scheduler", "")).lower()
    if scheduler not in {"linear", "cosine", "constant"}:
        raise ConfigError(f"train.scheduler는 linear | cosine | constant 중 하나여야 합니다: {scheduler!r}")
    for key in ("train.micro_batch", "train.grad_accum", "train.epochs", "infer.batch_size"):
        value = get_path(config, key)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"{key}는 1 이상의 정수여야 합니다: {value!r}")
    for key in ("min_pixels", "max_pixels"):
        value = get_path(config, key)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"{key}는 양의 정수여야 합니다: {value!r}")
    if get_path(config, "min_pixels") > get_path(config, "max_pixels"):
        raise ConfigError("min_pixels가 max_pixels보다 큽니다.")
    warmup = float(get_path(config, "train.warmup_ratio", 0.0))
    if not 0.0 <= warmup < 1.0:
        raise ConfigError(f"train.warmup_ratio는 [0, 1) 범위여야 합니다: {warmup}")
    targets = get_path(config, "lora.targets")
    if not (targets == "auto" or isinstance(targets, list)):
        raise ConfigError("lora.targets는 'auto' 또는 모듈 이름 리스트여야 합니다.")


def hashable_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """config_hash 대상 키만 남긴 사본."""
    view = copy.deepcopy({k: v for k, v in config.items() if not k.startswith("_")})
    for key in HASH_EXCLUDED_KEYS:
        drop_path(view, key)
    return view


def config_hash(config: Mapping[str, Any]) -> str:
    """학습 의미에 영향을 주는 키만으로 만든 sha256."""
    return runtime.sha256_json(hashable_view(config))


def model_source(config: Mapping[str, Any]) -> str:
    """from_pretrained에 넘길 경로 또는 repo id.

    우선순위: config.model_path > MODEL_ROOT/<model_id 마지막 조각> > model_id
    """
    explicit = get_path(config, "model_path")
    if explicit:
        return str(Path(str(explicit)).expanduser())
    model_id = str(get_path(config, "model_id"))
    root = runtime.model_root()
    if root is not None:
        for candidate in (root / model_id.replace("/", "--"), root / model_id.split("/")[-1], root / model_id):
            if candidate.exists():
                return str(candidate)
    return model_id


# --------------------------------------------------------------------------
# argparse 연동
# --------------------------------------------------------------------------


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """모든 CLI가 공유하는 설정 인자."""
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="설정 덮어쓰기. 점 표기 키를 쓴다. 여러 번 지정할 수 있다.",
    )
    parser.add_argument("--data-root", default=None, help="VQA_DATA_ROOT 덮어쓰기")
    parser.add_argument("--run-root", default=None, help="VQA_RUN_ROOT 덮어쓰기")
    parser.add_argument("--local-scratch", default=None, help="VQA_LOCAL_SCRATCH 덮어쓰기")


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, getattr(args, "overrides", None))
