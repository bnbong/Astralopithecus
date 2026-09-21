"""모델·processor 로딩과 LoRA target 자동 탐지.

Qwen2.5-VL(transformers 4.x)과 Qwen3.5(transformers 5.x)를 같은 코드로 다룬다.
로더는 AutoModelForMultimodalLM -> AutoModelForImageTextToText -> AutoModelForVision2Seq
순서로 시도한다. 정밀도는 bf16 / nf4 / int8을 지원한다.

LoRA target은 language backbone 안의 Linear만 고른다. vision tower와 merger에도
`q_proj` 같은 같은 이름의 모듈이 있으므로 leaf 이름만으로 고르면 vision까지 학습하게
된다. 그래서 전체 경로로 걸러 내고, PEFT에는 전체 이름 정규식을 넘긴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import config as config_mod
from . import runtime

#: 후보 leaf 이름. Qwen3.5의 linear-attention projection까지 포함한다.
LORA_LEAF_CANDIDATES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: 경로 조각에 이 단어가 들어가면 vision 계열로 보고 제외한다.
VISION_PATH_MARKERS: tuple[str, ...] = (
    "visual",
    "vision",
    "vision_tower",
    "vision_model",
    "image_encoder",
    "image_tower",
    "img_encoder",
    "merger",
    "patch_embed",
    "pixel_shuffle",
    "audio",
    "speech",
)

#: language backbone의 루트로 인정하는 속성 이름.
LANGUAGE_ROOT_NAMES: tuple[str, ...] = (
    "language_model",
    "text_model",
    "text_decoder",
    "llm",
)


class ModelLoadError(RuntimeError):
    """모델/processor 로딩이 모든 경로에서 실패했을 때."""


# --------------------------------------------------------------------------
# 버전 호환
# --------------------------------------------------------------------------


def transformers_major() -> int:
    try:
        import transformers

        return int(str(transformers.__version__).split(".")[0])
    except Exception:
        return 4


def _dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """transformers 4.x는 torch_dtype, 5.x는 dtype을 쓴다."""
    return {"dtype": dtype} if transformers_major() >= 5 else {"torch_dtype": dtype}


# --------------------------------------------------------------------------
# processor
# --------------------------------------------------------------------------


def load_processor(cfg: dict[str, Any], *, source: str | None = None) -> Any:
    """AutoProcessor를 config의 pixel budget으로 로드한다."""
    from transformers import AutoProcessor

    src = source or config_mod.model_source(cfg)
    kwargs: dict[str, Any] = {
        "min_pixels": int(config_mod.get_path(cfg, "min_pixels")),
        "max_pixels": int(config_mod.get_path(cfg, "max_pixels")),
    }
    revision = config_mod.get_path(cfg, "revision")
    if revision:
        kwargs["revision"] = revision
    if runtime.is_offline():
        kwargs["local_files_only"] = True

    try:
        processor = AutoProcessor.from_pretrained(src, **kwargs)
    except TypeError:
        # min/max_pixels를 받지 않는 processor도 있다.
        kwargs.pop("min_pixels", None)
        kwargs.pop("max_pixels", None)
        processor = AutoProcessor.from_pretrained(src, **kwargs)

    set_padding_side(processor, "right")
    return processor


def set_padding_side(processor: Any, side: str) -> None:
    """tokenizer와 processor 양쪽의 padding_side를 맞춘다."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = side
    if hasattr(processor, "padding_side"):
        try:
            processor.padding_side = side
        except Exception:
            pass


# --------------------------------------------------------------------------
# base model
# --------------------------------------------------------------------------


def _auto_classes() -> list[tuple[str, Any]]:
    import transformers

    out = []
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            out.append((name, cls))
    if not out:
        raise ModelLoadError("transformers에서 사용할 수 있는 multimodal Auto 클래스를 찾지 못했습니다.")
    return out


#: OOM일 때 붙이는 안내. Auto 클래스를 바꿔 봐야 결과가 같으므로 곧바로 멈춘다.
OOM_GUIDE = (
    "GPU 여유 메모리가 부족합니다. 다른 프로세스가 GPU를 점유하고 있는지 "
    "`nvidia-smi`와 `python -m vqa.jobs list`로 확인하세요."
)


def is_oom_error(exc: BaseException) -> bool:
    """CUDA 메모리 부족 예외인지. torch가 없는 환경에서는 메시지만으로 판별한다."""
    try:
        import torch

        candidates = (
            getattr(torch.cuda, "OutOfMemoryError", None),
            getattr(torch, "OutOfMemoryError", None),
        )
        oom_types = tuple(cls for cls in candidates if isinstance(cls, type))
        if oom_types and isinstance(exc, oom_types):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


def quantization_config(precision: str) -> Any | None:
    """precision 문자열에 맞는 BitsAndBytesConfig. bf16이면 None."""
    precision = precision.lower()
    if precision == "bf16":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if precision == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"알 수 없는 precision: {precision!r}")


def load_base_model(cfg: dict[str, Any], *, source: str | None = None, for_training: bool = True) -> tuple[Any, str]:
    """base 모델을 로드한다. (model, 사용한 Auto 클래스 이름)을 돌려준다."""
    import torch

    src = source or config_mod.model_source(cfg)
    precision = str(config_mod.get_path(cfg, "precision", "bf16")).lower()

    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": config_mod.get_path(cfg, "attn_implementation", "sdpa"),
    }
    kwargs.update(_dtype_kwarg(torch.bfloat16))
    revision = config_mod.get_path(cfg, "revision")
    if revision:
        kwargs["revision"] = revision
    if runtime.is_offline():
        kwargs["local_files_only"] = True
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    quant = quantization_config(precision)
    if quant is not None:
        kwargs["quantization_config"] = quant

    errors: list[str] = []
    for name, cls in _auto_classes():
        try:
            model = cls.from_pretrained(src, **kwargs)
        except Exception as exc:  # 클래스별 미지원은 다음 후보로 넘어간다
            if is_oom_error(exc):
                # 메모리가 모자란 것은 Auto 클래스 문제가 아니다. 다음 후보도 똑같이 실패한다.
                raise ModelLoadError(f"{OOM_GUIDE}\n{name}: {type(exc).__name__}: {exc}") from exc
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        prepare_for_training(model, precision, enable=for_training)
        return model, name

    raise ModelLoadError("모델 로딩에 모두 실패했습니다:\n" + "\n".join(errors))


def prepare_for_training(model: Any, precision: str, *, enable: bool = True) -> None:
    """gradient checkpointing, input require grads, use_cache를 학습용으로 맞춘다."""
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = not enable
    if not enable:
        return

    if precision.lower() in {"nf4", "int8"}:
        try:
            from peft import prepare_model_for_kbit_training

            prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


# --------------------------------------------------------------------------
# LoRA target 탐지
# --------------------------------------------------------------------------


@dataclass
class LoraTargets:
    """탐지 결과. run_manifest에 그대로 기록한다."""

    names: list[str] = field(default_factory=list)
    counts_by_leaf: dict[str, int] = field(default_factory=dict)
    excluded_vision: list[str] = field(default_factory=list)
    language_prefix: str | None = None

    @property
    def leaf_names(self) -> list[str]:
        return sorted(self.counts_by_leaf)

    def regex(self) -> str:
        """PEFT에 넘길 전체 경로 정규식."""
        if not self.names:
            raise ValueError("LoRA target이 비어 있습니다.")
        return "^(?:" + "|".join(re.escape(name) for name in self.names) + ")$"

    def as_dict(self) -> dict[str, Any]:
        return {
            "language_prefix": self.language_prefix,
            "leaf_names": self.leaf_names,
            "counts_by_leaf": dict(sorted(self.counts_by_leaf.items())),
            "num_modules": len(self.names),
            "num_excluded_vision": len(self.excluded_vision),
            "excluded_vision_sample": self.excluded_vision[:8],
            "modules_sample": self.names[:8],
            "modules": self.names,
        }


def _is_vision_path(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in VISION_PATH_MARKERS)


def find_language_prefix(model: Any) -> str | None:
    """language backbone 루트 모듈의 전체 경로를 찾는다. 없으면 None."""
    candidates: list[str] = []
    for name, _module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in LANGUAGE_ROOT_NAMES and not _is_vision_path(name):
            candidates.append(name)
    if not candidates:
        return None
    # 가장 얕은 경로를 쓴다.
    return sorted(candidates, key=lambda n: (n.count("."), len(n)))[0]


def detect_lora_targets(
    model: Any,
    *,
    leaf_candidates: Sequence[str] = LORA_LEAF_CANDIDATES,
    explicit_leaves: Iterable[str] | None = None,
) -> LoraTargets:
    """language 쪽 Linear만 골라 LoRA target 전체 경로 목록을 만든다.

    `explicit_leaves`를 주면 (config의 lora.targets 리스트) 그 leaf 이름만 대상으로
    삼되 language 한정 필터는 그대로 적용한다.
    """
    import torch

    wanted = tuple(explicit_leaves) if explicit_leaves else tuple(leaf_candidates)
    prefix = find_language_prefix(model)
    result = LoraTargets(language_prefix=prefix)

    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in wanted:
            continue
        if not _is_linear(module, torch):
            continue
        if _is_vision_path(name):
            result.excluded_vision.append(name)
            continue
        if prefix is not None and not (name == prefix or name.startswith(prefix + ".")):
            # language 루트를 찾았는데 그 밖에 있는 projection은 제외한다.
            result.excluded_vision.append(name)
            continue
        result.names.append(name)
        result.counts_by_leaf[leaf] = result.counts_by_leaf.get(leaf, 0) + 1

    return result


def _is_linear(module: Any, torch: Any) -> bool:
    """nn.Linear 또는 bitsandbytes의 양자화 Linear인지."""
    if isinstance(module, torch.nn.Linear):
        return True
    cls_name = type(module).__name__
    return cls_name in {"Linear4bit", "Linear8bitLt", "LinearNF4", "Params4bit", "Linear8bitLtThreshold"}


def build_lora_model(model: Any, cfg: dict[str, Any], targets: LoraTargets | None = None) -> tuple[Any, LoraTargets]:
    """탐지한 target으로 PEFT LoRA를 붙인다."""
    from peft import LoraConfig, get_peft_model

    configured = config_mod.get_path(cfg, "lora.targets", "auto")
    if targets is None:
        explicit = None if configured == "auto" else list(configured)
        targets = detect_lora_targets(model, explicit_leaves=explicit)
    if not targets.names:
        raise ModelLoadError(
            "language backbone에서 LoRA target을 찾지 못했습니다. "
            "모델 구조가 예상과 다르니 named_modules를 확인하세요."
        )

    lora_config = LoraConfig(
        r=int(config_mod.get_path(cfg, "lora.r", 16)),
        lora_alpha=int(config_mod.get_path(cfg, "lora.alpha", 32)),
        lora_dropout=float(config_mod.get_path(cfg, "lora.dropout", 0.05)),
        bias="none",
        # 전체 경로 정규식을 넘겨 vision 쪽 동명 모듈이 절대 잡히지 않게 한다.
        target_modules=targets.regex(),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, targets


def count_parameters(model: Any) -> dict[str, int]:
    """전체/학습 파라미터 수."""
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def trainable_parameters(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def verify_no_vision_adapters(model: Any) -> list[str]:
    """LoRA가 붙은 모듈 중 vision 경로에 있는 것이 있으면 그 이름을 돌려준다."""
    bad = []
    for name, module in model.named_modules():
        if "lora_A" in name or "lora_B" in name:
            if _is_vision_path(name):
                bad.append(name)
    return sorted(set(bad))


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cublas_status_alloc_failed" in msg


def cleanup(model: Any = None) -> None:
    """모델을 버리고 CUDA 캐시를 비운다."""
    import gc

    if model is not None:
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
