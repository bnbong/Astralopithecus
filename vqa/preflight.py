"""본 학습 전에 환경과 설정을 점검한다.

여기서 걸러야 하는 것은 몇 시간 뒤에야 드러나는 실패다. a/b/c/d가 단일 토큰이 아닌
tokenizer, thinking이 켜진 채로 나가는 chat template, vision까지 잡아 버린 LoRA target,
엉뚱한 위치에 걸린 label, 4-bit 커널이 없는 GPU 같은 것들이다.

사용 예:
    python -m vqa.preflight --config configs/lab_qwen25vl_3b.yaml --n-train-steps 20 --n-infer 50
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import collator as collator_mod
from . import config as config_mod
from . import prompt as prompt_mod
from . import runtime

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"


@dataclass
class CheckResult:
    name: str
    status: str
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "reason": self.reason, "detail": self.detail}


class Preflight:
    """검사를 순서대로 돌리고 결과를 모은다."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def run(self, name: str, fn: Callable[[], CheckResult]) -> CheckResult:
        try:
            result = fn()
        except Exception as exc:
            result = CheckResult(
                name=name,
                status=FAIL,
                reason=f"{type(exc).__name__}: {exc}",
                detail={"traceback": traceback.format_exc(limit=8)},
            )
        self.results.append(result)
        marker = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP", WARN: "WARN"}[result.status]
        print(f"[{marker}] {result.name}" + (f" — {result.reason}" if result.reason else ""), flush=True)
        return result

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        header = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines = ["", "=" * 68, f"PREFLIGHT {'FAIL' if self.failed else 'PASS'}  ({header})", "=" * 68]
        for result in self.results:
            lines.append(f"  {result.status:4s} {result.name}" + (f" — {result.reason}" if result.reason else ""))
        if self.failed:
            lines.append("")
            lines.append("실패한 항목:")
            for result in self.failed:
                lines.append(f"  - {result.name}: {result.reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 개별 검사
# --------------------------------------------------------------------------


def check_environment(cfg: dict[str, Any], paths: dict[str, Any]) -> CheckResult:
    report = runtime.environment_report(paths)
    torch_info = report.get("torch", {})
    if not torch_info.get("cuda_available"):
        return CheckResult(
            "environment", WARN, "CUDA를 쓸 수 없습니다. 이후 GPU 검사는 건너뜁니다.", report
        )
    gpu = report.get("gpu", {})
    reason = f"{gpu.get('name')} {gpu.get('total_gb')}GB, torch {torch_info.get('version')}"
    return CheckResult("environment", PASS, reason, report)


def check_bnb_4bit() -> CheckResult:
    """현재 GPU에서 bitsandbytes 4-bit forward/backward가 되는지 본다."""
    import torch

    if not torch.cuda.is_available():
        return CheckResult("bitsandbytes_4bit", SKIP, "CUDA 없음")
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        return CheckResult("bitsandbytes_4bit", FAIL, f"bitsandbytes import 실패: {exc}")

    layer = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
        compress_statistics=True,
    ).cuda()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    ok = x.grad is not None and bool(torch.isfinite(x.grad).all())
    detail = {
        "bitsandbytes": getattr(bnb, "__version__", None),
        "loss": float(loss.detach()),
        "grad_finite": ok,
    }
    if not ok:
        return CheckResult("bitsandbytes_4bit", FAIL, "4-bit backward에서 grad가 유한하지 않습니다.", detail)
    return CheckResult("bitsandbytes_4bit", PASS, "4-bit forward+backward 정상", detail)


def check_single_tokens(processor: Any) -> CheckResult:
    result = prompt_mod.check_single_tokens(processor)
    if not result["ok"]:
        bad = [k for k, v in result["letters"].items() if not v["bare_single"]]
        return CheckResult("choice_single_token", FAIL, f"단일 토큰이 아닌 letter: {bad}", result)
    return CheckResult("choice_single_token", PASS, "a/b/c/d 모두 단일 토큰", result)


def check_template(processor: Any, spec: prompt_mod.PromptSpec, image: Any) -> CheckResult:
    text = prompt_mod.full_generation_prompt(processor, spec, image)
    tail = text[-220:]
    supported = prompt_mod.template_supports_thinking_flag(processor)
    disabled = prompt_mod.thinking_disabled(tail)
    detail = {
        "tail": tail,
        "enable_thinking_kwarg_supported": supported,
        "thinking_disabled": disabled,
        "prompt_id": spec.prompt_id,
        "prompt_hash": spec.hash(),
    }
    if not disabled:
        return CheckResult("chat_template_tail", FAIL, "generation prompt 꼬리에 비어 있지 않은 <think>가 있습니다.", detail)
    reason = "thinking 비활성 확인" + ("" if supported else " (템플릿이 enable_thinking 인자를 받지 않아 기본 경로 사용)")
    return CheckResult("chat_template_tail", PASS, reason, detail)


def check_choice_tokens(processor: Any, spec: prompt_mod.PromptSpec, image: Any) -> CheckResult:
    text = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, text)
    detail = tokens.as_dict()
    if tokens.source != "tail":
        return CheckResult(
            "choice_token_after_tail",
            WARN,
            "템플릿 꼬리 뒤 토큰을 확정하지 못해 letter 단독 토큰 id를 씁니다.",
            detail,
        )
    if tokens.ids != [tokens.bare_ids[letter][0] for letter in prompt_mod.LETTERS]:
        return CheckResult(
            "choice_token_after_tail",
            WARN,
            "꼬리 뒤 토큰 id가 letter 단독 id와 다릅니다. scoring은 꼬리 기준 id를 씁니다.",
            detail,
        )
    return CheckResult("choice_token_after_tail", PASS, f"scoring token ids={tokens.ids}", detail)


#: processor가 min/max_pixels를 실제로 적용했는지 볼 때 허용하는 상대 오차.
#: smart_resize가 patch 배수로 올림하거나 내림하므로 정확히 경계에 붙지 않는다.
PIXEL_LIMIT_TOLERANCE = 0.02

#: 크기 표본을 훑을 때 기본으로 열어 보는 이미지 개수.
DEFAULT_PIXEL_SAMPLE = 200


def largest_image_records(
    records: Sequence[collator_mod.VQARecord],
    data_root: Path,
    *,
    n: int = 2,
    sample: int = DEFAULT_PIXEL_SAMPLE,
) -> list[collator_mod.VQARecord]:
    """표본에서 원본 픽셀 수가 가장 큰 문항을 n개 고른다.

    forward_backward가 OOM을 보려고 고르는 "가장 긴 표본"은 `approx_length()`, 즉 질문과
    선택지의 글자 수만 본다. 해상도를 올린 설정에서는 메모리를 지배하는 쪽이 글자가 아니라
    시각 토큰이므로, 가장 큰 이미지가 표본에 한 장도 들어가지 않으면 preflight가 OOM을
    놓치고 본 학습에서 몇 시간 뒤에 죽는다. 그래서 이미지 크기 기준으로도 따로 고른다.
    이미지를 디코딩하지 않고 헤더의 크기만 읽으므로 비용은 거의 들지 않는다.
    """
    if int(n) <= 0:
        return []
    from PIL import Image

    sized: list[tuple[int, collator_mod.VQARecord]] = []
    for record in records[: max(int(sample), 1)]:
        if record.answer_idx is None:
            continue
        path = collator_mod.resolve_image_path(record.path, data_root)
        if not path.exists():
            continue
        try:
            with Image.open(path) as handle:
                width, height = handle.size
        except Exception:
            continue
        sized.append((width * height, record))
    # key로만 정렬하므로 넓이가 같아도 record끼리 비교하지 않는다.
    sized.sort(key=lambda item: -item[0])
    return [record for _, record in sized[: int(n)]]


def _visual_tokens_from_grid(grid: Any, merge_size: int) -> int | None:
    """image_grid_thw 한 행에서 merge 뒤의 시각 토큰 수를 계산한다.

    grid는 [t, h, w]를 patch 단위로 담고 있다. merge_size는 가로와 세로를 각각
    merge_size배로 묶으므로 토큰 수는 t*h*w를 merge_size의 제곱으로 나눈 값이다.
    """
    try:
        values = [int(x) for x in list(grid)]
    except Exception:
        return None
    if not values:
        return None
    product = 1
    for value in values:
        product *= value
    divisor = max(int(merge_size), 1) ** 2
    return product // divisor


class ImageProbeError(RuntimeError):
    """image_processor를 어떤 호출 형태로도 부르지 못했을 때 쓴다."""


def _call_image_processor(image_processor: Any, image: Any) -> Any:
    """버전마다 다른 호출 형태를 차례로 시도한다.

    transformers 버전에 따라 image_processor가 받는 인자 이름이 달라진다. 어느 형태로도
    부르지 못하면 ImageProbeError를 올려서, 검사 자체를 FAIL이 아니라 WARN으로 떨어뜨린다.
    해상도 설정이 틀렸다는 뜻이 아니라 확인하지 못했다는 뜻이기 때문이다.
    """
    attempts = (
        lambda: image_processor(images=[image], return_tensors="pt"),
        lambda: image_processor(images=[image]),
        lambda: image_processor([image], return_tensors="pt"),
    )
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise ImageProbeError("; ".join(errors))


def _probe_image(image_processor: Any, image: Any, merge_size: int) -> tuple[int | None, list[str]]:
    """이미지 한 장을 실제로 전처리해서 시각 토큰 수를 구한다."""
    encoded = _call_image_processor(image_processor, image)
    keys = sorted(encoded.keys()) if hasattr(encoded, "keys") else []
    grid = encoded.get("image_grid_thw") if hasattr(encoded, "get") else None
    if grid is None:
        return None, keys
    row = grid[0] if len(grid) else grid
    return _visual_tokens_from_grid(row, merge_size), keys


def check_pixel_limits(
    processor: Any,
    cfg: dict[str, Any],
    records: Sequence[collator_mod.VQARecord],
    data_root: Path,
    *,
    sample: int = DEFAULT_PIXEL_SAMPLE,
) -> CheckResult:
    """설정한 min/max_pixels가 processor에 실제로 먹었는지 이미지로 직접 확인한다.

    모델은 필요 없다. 표본 중에서 가장 큰 이미지와 가장 작은 이미지를 골라 image_processor에
    그대로 넣고, image_grid_thw에서 시각 토큰 수를 센 다음, 설정값에서 기대되는 범위와
    비교한다. patch 크기와 merge 크기는 하드코딩하지 않고 image_processor 속성에서 읽는다.
    """
    min_pixels = config_mod.get_path(cfg, "min_pixels")
    max_pixels = config_mod.get_path(cfg, "max_pixels")
    if not min_pixels or not max_pixels:
        return CheckResult("pixel_limits", SKIP, "config에 min_pixels 또는 max_pixels가 없습니다.")
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return CheckResult(
            "pixel_limits", SKIP, "processor에 image_processor가 없어서 확인할 수 없습니다."
        )

    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not patch_size or not merge_size:
        return CheckResult(
            "pixel_limits",
            WARN,
            "image_processor에서 patch_size 또는 merge_size를 읽지 못해 기대 범위를 세울 수 "
            "없습니다. InternVL처럼 448 타일 개수로 해상도를 정하는 processor가 여기에 "
            "해당합니다. 그런 모델에서는 min/max_pixels가 아니라 설정의 processor_kwargs "
            "(max_patches, min_patches)가 해상도를 정합니다.",
            {"patch_size": patch_size, "merge_size": merge_size,
             "image_processor_class": type(image_processor).__name__},
        )
    patch_size = int(patch_size)
    merge_size = int(merge_size)

    # 시각 토큰 하나가 덮는 원본 픽셀 넓이다. patch를 merge_size배로 묶기 때문이다.
    pixels_per_token = (patch_size * merge_size) ** 2
    tokens_upper = max_pixels / pixels_per_token
    tokens_lower = min_pixels / pixels_per_token

    # 이미지를 디코딩하지 않고 크기만 읽어서 가장 큰 것과 가장 작은 것을 고른다.
    from PIL import Image

    largest = None
    smallest = None
    scanned = 0
    for record in records[: max(int(sample), 1)]:
        path = collator_mod.resolve_image_path(record.path, data_root)
        if not path.exists():
            continue
        try:
            with Image.open(path) as handle:
                width, height = handle.size
        except Exception:
            continue
        scanned += 1
        entry = {"id": record.id, "path": record.path, "width": width, "height": height,
                 "pixels": width * height}
        if largest is None or entry["pixels"] > largest["pixels"]:
            largest = entry
        if smallest is None or entry["pixels"] < smallest["pixels"]:
            smallest = entry

    if largest is None or smallest is None:
        return CheckResult("pixel_limits", SKIP, "열어 볼 수 있는 이미지를 찾지 못했습니다.")

    detail: dict[str, Any] = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "pixels_per_token": pixels_per_token,
        "expected_tokens_upper": tokens_upper,
        "expected_tokens_lower": tokens_lower,
        "scanned_images": scanned,
        "largest": dict(largest),
        "smallest": dict(smallest),
    }
    # 이후에는 detail 안의 사본을 그대로 채워 나간다.
    largest = detail["largest"]
    smallest = detail["smallest"]

    for label, entry in (("largest", largest), ("smallest", smallest)):
        image = collator_mod.load_image(entry["path"], data_root)
        try:
            tokens, keys = _probe_image(image_processor, image, merge_size)
        except ImageProbeError as exc:
            detail["probe_errors"] = str(exc)
            return CheckResult(
                "pixel_limits",
                WARN,
                "image_processor를 직접 호출하지 못해 해상도 적용 여부를 확인하지 못했습니다. "
                "자세한 내용은 detail의 probe_errors에 있습니다.",
                detail,
            )
        if tokens is None:
            detail["encoded_keys"] = keys
            return CheckResult(
                "pixel_limits",
                WARN,
                "image_processor 결과에 image_grid_thw가 없어서 시각 토큰 수를 셀 수 없습니다. "
                "타일 방식 processor는 이 키 대신 num_patches를 돌려줍니다.",
                detail,
            )
        detail[label]["visual_tokens"] = tokens

    # 원본이 상한(하한)에 걸릴 만큼 크지(작지) 않으면 그 방향은 시험한 것이 아니다.
    upper_binding = largest["pixels"] > max_pixels
    lower_binding = smallest["pixels"] < min_pixels
    detail["upper_binding"] = upper_binding
    detail["lower_binding"] = lower_binding

    upper_allowed = tokens_upper * (1.0 + PIXEL_LIMIT_TOLERANCE) + 1
    lower_allowed = tokens_lower * (1.0 - PIXEL_LIMIT_TOLERANCE) - 1

    if largest["visual_tokens"] > upper_allowed:
        return CheckResult(
            "pixel_limits",
            FAIL,
            "가장 큰 이미지 %s 가 시각 토큰 %d개를 만들어 기대 상한 %.0f개를 넘었습니다. "
            "max_pixels=%d 가 processor에 적용되지 않은 것입니다."
            % (largest["id"], largest["visual_tokens"], tokens_upper, max_pixels),
            detail,
        )
    if lower_binding and smallest["visual_tokens"] < lower_allowed:
        return CheckResult(
            "pixel_limits",
            FAIL,
            "가장 작은 이미지 %s 가 시각 토큰 %d개뿐이어서 기대 하한 %.0f개에 못 미칩니다. "
            "min_pixels=%d 가 processor에 적용되지 않은 것입니다."
            % (smallest["id"], smallest["visual_tokens"], tokens_lower, min_pixels),
            detail,
        )

    reason = (
        "최대 %dx%d -> 시각 토큰 %d개, 최소 %dx%d -> 시각 토큰 %d개 "
        "(기대 상한 %.0f개, 기대 하한 %.0f개)"
        % (
            largest["width"], largest["height"], largest["visual_tokens"],
            smallest["width"], smallest["height"], smallest["visual_tokens"],
            tokens_upper, tokens_lower,
        )
    )
    if not upper_binding:
        return CheckResult(
            "pixel_limits",
            WARN,
            reason + ". 다만 표본에서 가장 큰 이미지도 원본이 max_pixels보다 작아서 상한이 "
            "실제로 걸리는지는 확인하지 못했습니다.",
            detail,
        )
    return CheckResult("pixel_limits", PASS, reason, detail)


def check_label_mask(
    processor: Any,
    spec: prompt_mod.PromptSpec,
    records: Sequence[collator_mod.VQARecord],
    data_root: Path,
    tokens: Any,
    n: int = 4,
) -> CheckResult:
    import torch

    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)
    tokenizer = getattr(processor, "tokenizer", processor)
    samples = []
    for record in records[:n]:
        if record.answer_idx is None:
            continue
        samples.append(
            collator_mod.build_sample_view(
                record, data_root=data_root, seed=0, epoch=0, shuffle_options=False, with_gold=True
            )
        )
    if not samples:
        return CheckResult("label_mask", SKIP, "정답이 있는 표본이 없습니다.")

    enc = collate(samples)
    labels = enc["labels"]
    supervised = (labels != -100).sum(dim=1).tolist()
    positions = []
    for b in range(labels.shape[0]):
        index = int(torch.where(labels[b] != -100)[0][0])
        token_id = int(labels[b, index])
        last = int(collator_mod.last_nonpad_index(enc["attention_mask"][b : b + 1])[0])
        positions.append(
            {
                "sample": samples[b].id,
                "position": index,
                "last_valid": last,
                "distance_from_end": last - index,
                "decoded": tokenizer.decode([token_id]),
                "gold": samples[b].gold_letter,
                "seq_len": int(labels.shape[1]),
            }
        )
    detail = {"supervised_per_sample": supervised, "positions": positions}
    if any(count != 1 for count in supervised):
        return CheckResult("label_mask", FAIL, f"감독 토큰 수가 1이 아닙니다: {supervised}", detail)
    return CheckResult("label_mask", PASS, "표본당 정답 토큰 1개만 감독", detail)


def check_prefix_match(
    processor: Any,
    spec: prompt_mod.PromptSpec,
    records: Sequence[collator_mod.VQARecord],
    data_root: Path,
    tokens: Any,
    n: int = 4,
) -> CheckResult:
    """학습 텍스트의 감독 위치 앞 토큰이 추론 input_ids와 같은지 표본 여러 개로 본다."""
    import torch

    collate = collator_mod.AnswerOnlyCollator(
        processor=processor, spec=spec, choice_tokens=tokens, verify_prefix=False
    )
    samples = []
    for record in records:
        if record.answer_idx is None:
            continue
        samples.append(
            collator_mod.build_sample_view(
                record, data_root=data_root, seed=0, epoch=0, shuffle_options=False, with_gold=True
            )
        )
        if len(samples) >= max(n, 1):
            break
    if not samples:
        return CheckResult("train_infer_prefix_match", SKIP, "정답이 있는 표본이 없습니다.")

    enc = collate(samples)
    labels = enc["labels"]
    reports = []
    for b, view in enumerate(samples):
        position = int(torch.where(labels[b] != -100)[0][0])
        reports.append(
            collator_mod.compare_prefix_with_inference(
                processor, spec, view, enc["input_ids"][b], position
            )
        )
    bad = [r for r in reports if not r["match"]]
    detail = {
        "checked": len(reports),
        "end_of_turn_suffix": prompt_mod.end_of_turn_suffix(processor),
        "reports": reports,
    }
    if bad:
        return CheckResult(
            "train_infer_prefix_match",
            FAIL,
            f"{len(bad)}/{len(reports)} 표본에서 학습 prefix가 추론 prefix와 다릅니다.",
            detail,
        )
    return CheckResult(
        "train_infer_prefix_match",
        PASS,
        f"{len(reports)}개 표본 모두 추론 prefix와 토큰 단위로 일치",
        detail,
    )


def attached_lora_modules(model: Any) -> list[str]:
    """PEFT로 감싼 모델에서 실제로 LoRA가 붙은 모듈의 전체 경로를 돌려준다.

    `lora_A` 속성을 가진 모듈이 adapter가 붙은 자리다. 감싼 뒤의 leaf 클래스는
    `torch.nn.Linear`가 아니라 `peft.tuners.lora.layer.Linear`(bf16)나
    `Linear4bit`(nf4)이므로 클래스 이름으로 판정하지 않는다.
    """
    names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if hasattr(module, "lora_A"):
            names.append(name)
    return names


def check_lora_targets(model: Any, cfg: dict[str, Any], targets: Any = None) -> tuple[CheckResult, Any]:
    """LoRA target 점검.

    target 목록은 **PEFT로 감싸기 전에** 탐지한 값(`backend.targets`)을 쓴다. 감싼 뒤의
    모델에서 다시 탐지하면 leaf가 `peft.tuners.lora.layer.Linear`로 바뀌어 있어
    bf16에서는 0개가 나온다. 그 다음 감싼 모델에서 실제로 adapter가 붙은 개수가
    target 수와 같은지, vision 경로에 붙은 것이 없는지를 확인한다.
    """
    from . import modeling

    configured = config_mod.get_path(cfg, "lora.targets", "auto")
    explicit = None if configured == "auto" else list(configured)

    source = "backend.targets (감싸기 전)"
    if targets is None or not getattr(targets, "names", None):
        # backend가 값을 넘겨 주지 않은 경우에만 직접 탐지한다. 감싸지 않은 base 모델이어야 한다.
        targets = modeling.detect_lora_targets(model, explicit_leaves=explicit)
        source = "감싸지 않은 base 모델에서 재탐지"

    detail = targets.as_dict()
    detail["targets_source"] = source
    if not targets.names:
        return CheckResult("lora_targets", FAIL, "language backbone에서 target을 찾지 못했습니다.", detail), targets

    attached = attached_lora_modules(model)
    vision_attached = sorted({name for name in attached if modeling._is_vision_path(name)})
    detail["num_attached"] = len(attached)
    detail["attached_sample"] = attached[:8]
    detail["attached_vision"] = vision_attached

    if vision_attached:
        return (
            CheckResult(
                "lora_targets",
                FAIL,
                f"vision 경로에 LoRA가 붙었습니다: {vision_attached[:3]}",
                detail,
            ),
            targets,
        )
    if len(attached) != len(targets.names):
        return (
            CheckResult(
                "lora_targets",
                FAIL,
                f"LoRA가 붙은 모듈 {len(attached)}개가 target {len(targets.names)}개와 다릅니다.",
                detail,
            ),
            targets,
        )
    if len(targets.leaf_names) < 4:
        return (
            CheckResult(
                "lora_targets",
                WARN,
                f"찾은 leaf 종류가 적습니다: {targets.leaf_names}. 모델 구조를 확인하세요.",
                detail,
            ),
            targets,
        )
    reason = (
        f"language 모듈 {len(targets.names)}개, leaf {targets.leaf_names}, "
        f"vision 제외 {len(targets.excluded_vision)}개, 실제 부착 {len(attached)}개"
    )
    return CheckResult("lora_targets", PASS, reason, detail), targets


#: 시간 측정에 쓰는 microbatch 개수의 기본 상한.
DEFAULT_MAX_TIMING_MICROBATCHES = 64

#: --tf32-ab에서 한쪽(off 또는 on)마다 재는 microbatch 개수와 그 앞에 버리는 warm-up 개수.
TF32_AB_MICROBATCHES = 8
TF32_AB_WARMUPS = 2


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def check_forward_backward(
    backend: Any,
    records: Sequence[collator_mod.VQARecord],
    n_steps: int,
    *,
    grad_accum: int = 1,
    max_timing_microbatches: int = DEFAULT_MAX_TIMING_MICROBATCHES,
    n_longest: int = 4,
    largest_images: Sequence[collator_mod.VQARecord] = (),
    seed: int = 0,
    tf32_ab: bool = False,
    tf32_enabled: bool = True,
) -> CheckResult:
    """OOM은 가장 긴 표본으로, 속도는 무작위 표본으로 따로 본다.

    한 번의 forward+backward는 **microbatch 하나**다. optimizer step 하나는
    grad_accum개의 microbatch이므로 예산 계산에는 `sec_per_optimizer_step`을 써야
    한다. 가장 긴 표본만으로 잰 시간은 상한이라 평균 대신 쓰면 안 된다.

    `largest_images`를 주면 그 문항들을 OOM 표본 뒤에 덧붙여서 한 번씩 더 돌린다. 글자
    수는 짧아도 이미지가 커서 시각 토큰이 많은 문항을 반드시 통과시키기 위한 것이며,
    고를 때에는 `largest_image_records`를 쓴다.

    `n_steps`가 0 이하이면 이 검사 자체를 건너뛰고 SKIP을 돌려준다. 학습을 하지 않는
    가상 머신에서는 역전파 활성값 때문에 이 검사 하나가 out of memory를 일으키는데,
    그 GPU에서도 추론 점검은 그대로 수행해야 하기 때문이다.

    시간은 CPU 준비 구간(make_micro_batch: 이미지 디코딩, processor, 템플릿)과 GPU
    구간(forward+backward)으로 나누어 잰다. 둘 중 어느 쪽이 병목인지 모르면 속도
    차이를 GPU 설정 탓으로만 돌리기 쉽다. `sec_per_microbatch_mean`은 예전과 같이
    두 구간의 합이다.
    """
    import random

    import torch

    if int(n_steps) <= 0:
        return CheckResult(
            "forward_backward", SKIP, "학습 점검을 건너뛰었습니다 (--n-train-steps 0)."
        )

    labeled = [r for r in records if r.answer_idx is not None]
    if not labeled:
        return CheckResult("forward_backward", SKIP, "학습 표본이 없습니다.")

    grad_accum = max(int(grad_accum), 1)
    ordered = sorted(labeled, key=lambda r: -r.approx_length())
    longest = ordered[: max(min(int(n_longest), len(ordered)), 1)]

    # OOM 표본 = 글자가 가장 긴 문항 + 이미지가 가장 큰 문항. 뒤쪽은 호출자가 골라서 준다.
    oom_group = list(longest)
    oom_ids = {record.id for record in oom_group}
    for record in largest_images:
        if record.id in oom_ids:
            continue
        oom_group.append(record)
        oom_ids.add(record.id)

    n_timing = min(max(int(n_steps), 1) * grad_accum, max(int(max_timing_microbatches), 1))
    n_timing = min(n_timing, len(labeled))
    sampled = random.Random(int(seed)).sample(labeled, n_timing)

    params = backend.trainable_parameters()

    def run_one(record: collator_mod.VQARecord) -> tuple[float, float, float]:
        """(준비 초, 계산 초, loss). CUDA가 있으면 계산 구간 끝에서 동기화한다."""
        started = time.time()
        batch = backend.make_micro_batch([record.id], 0)
        prepared = time.time()
        loss = backend.compute_loss(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{record.id}: loss가 유한하지 않습니다.")
        loss.backward()
        runtime.cuda_sync()
        computed = time.time()
        for param in params:
            param.grad = None
        return prepared - started, computed - prepared, float(loss.detach())

    runtime.reset_peak_vram()

    # (a) 가장 긴 표본과 가장 큰 이미지로 OOM을 본다. 첫 회는 warm-up이라 cold로 따로 기록한다.
    longest_timings: list[float] = []
    losses: list[float] = []
    try:
        for record in oom_group:
            prepare, compute, loss_value = run_one(record)
            longest_timings.append(prepare + compute)
            losses.append(loss_value)
    except RuntimeError as exc:
        return CheckResult("forward_backward", FAIL, str(exc))
    peak_after_longest = runtime.peak_vram_gb()

    # (b) 무작위 표본으로 대표 속도를 잰다.
    timing_seconds: list[float] = []
    prepare_seconds: list[float] = []
    compute_seconds: list[float] = []
    try:
        for record in sampled:
            prepare, compute, loss_value = run_one(record)
            timing_seconds.append(prepare + compute)
            prepare_seconds.append(prepare)
            compute_seconds.append(compute)
            losses.append(loss_value)
    except RuntimeError as exc:
        return CheckResult("forward_backward", FAIL, str(exc))

    sec_per_microbatch_mean = _mean(timing_seconds)
    warm_longest = longest_timings[1:] or longest_timings
    sec_per_microbatch_longest = _mean(warm_longest)
    sec_per_optimizer_step = sec_per_microbatch_mean * grad_accum

    detail = {
        "steps": len(longest_timings) + len(timing_seconds),
        "cold_seconds": longest_timings[0],
        "grad_accum": grad_accum,
        "timing_microbatches": len(timing_seconds),
        "timing_seed": int(seed),
        "max_timing_microbatches": int(max_timing_microbatches),
        "sec_per_microbatch_mean": sec_per_microbatch_mean,
        "sec_prepare_mean": _mean(prepare_seconds),
        "sec_compute_mean": _mean(compute_seconds),
        "sec_per_microbatch_longest": sec_per_microbatch_longest,
        "sec_per_optimizer_step": sec_per_optimizer_step,
        # 예전 키. sec_per_microbatch_mean의 별칭이며 optimizer step 시간이 아니다.
        "warm_seconds_mean": sec_per_microbatch_mean,
        "longest_samples": [r.id for r in longest],
        "longest_sample": longest[0].id,
        "largest_image_samples": [r.id for r in oom_group[len(longest):]],
        "longest_approx_length": longest[0].approx_length(),
        "loss_mean": _mean(losses),
        "peak_vram_gb": runtime.peak_vram_gb(),
        "peak_vram_gb_longest": peak_after_longest,
        "tf32": runtime.tf32_state(),
    }
    reason = (
        f"cold {detail['cold_seconds']:.2f}s/microbatch, "
        f"평균 {sec_per_microbatch_mean:.2f}s/microbatch (무작위 {len(timing_seconds)}개), "
        f"최장 {sec_per_microbatch_longest:.2f}s/microbatch, "
        f"grad_accum={grad_accum} -> {sec_per_optimizer_step:.2f}s/optimizer step, "
        f"peak VRAM {detail['peak_vram_gb']} GB"
    )
    if detail["largest_image_samples"]:
        reason += f", 가장 큰 이미지 {len(detail['largest_image_samples'])}장 포함"

    if tf32_ab:
        try:
            detail["tf32_ab"] = _run_tf32_ab(run_one, sampled)
        except RuntimeError as exc:
            return CheckResult("forward_backward", FAIL, str(exc), detail)
        finally:
            # A/B가 어떻게 끝나든 설정값으로 되돌려 놓는다.
            detail["tf32"] = runtime.apply_tf32(bool(tf32_enabled))
        off = detail["tf32_ab"]["off"]["sec_per_microbatch_mean"]
        on = detail["tf32_ab"]["on"]["sec_per_microbatch_mean"]
        reason += f", TF32 off {off:.2f}s / on {on:.2f}s per microbatch"

    return CheckResult("forward_backward", PASS, reason, detail)


def _run_tf32_ab(
    run_one: Callable[[Any], tuple[float, float, float]],
    sampled: Sequence[collator_mod.VQARecord],
    *,
    n_microbatches: int = TF32_AB_MICROBATCHES,
    n_warmups: int = TF32_AB_WARMUPS,
) -> dict[str, Any]:
    """같은 표본으로 TF32를 끄고 한 번, 켜고 한 번 재서 두 평균을 돌려준다."""
    needed = max(int(n_warmups), 0) + max(int(n_microbatches), 1)
    if not sampled:
        return {}
    chosen = [sampled[i % len(sampled)] for i in range(needed)]

    def one_pass(enabled: bool) -> dict[str, Any]:
        flags = runtime.apply_tf32(enabled)
        for record in chosen[:n_warmups]:
            run_one(record)
        prepare: list[float] = []
        compute: list[float] = []
        for record in chosen[n_warmups:]:
            prep, comp, _ = run_one(record)
            prepare.append(prep)
            compute.append(comp)
        return {
            "tf32": flags,
            "microbatches": len(compute),
            "warmups": min(int(n_warmups), len(chosen)),
            "sec_per_microbatch_mean": _mean([p + c for p, c in zip(prepare, compute)]),
            "sec_prepare_mean": _mean(prepare),
            "sec_compute_mean": _mean(compute),
        }

    return {"off": one_pass(False), "on": one_pass(True)}


def check_inference(
    scorer: Any,
    records: Sequence[collator_mod.VQARecord],
    n_infer: int,
    batch_size: int,
) -> CheckResult:
    """추론 경로와 확률 형식, 항목당 시간을 본다.

    학습 쪽과 같은 이유로 CPU 준비 구간(encode: 이미지 디코딩, processor, 템플릿)과
    GPU 구간(scoring forward)을 따로 잰다. `seconds`와 `seconds_per_item`의 의미는
    예전 그대로이며 두 구간의 합이다.
    """
    import numpy as np

    chosen = list(records[: max(n_infer, 1)])
    started = time.time()
    correct = 0
    scored = 0
    prepare_total = 0.0
    compute_total = 0.0
    for start in range(0, len(chosen), batch_size):
        chunk = chosen[start : start + batch_size]
        chunk_started = time.time()
        enc = scorer.encode(chunk, (0, 1, 2, 3))
        prepared = time.time()
        probs, _ = scorer.score(enc)
        runtime.cuda_sync()
        prepare_total += prepared - chunk_started
        compute_total += time.time() - prepared
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
            return CheckResult("inference", FAIL, "확률 합이 1이 아닙니다.", {"sums": probs.sum(axis=1).tolist()})
        for i, record in enumerate(chunk):
            if record.answer_idx is not None:
                scored += 1
                correct += int(int(probs[i].argmax()) == record.answer_idx)
    elapsed = time.time() - started
    items = max(len(chosen), 1)
    detail = {
        "items": len(chosen),
        "seconds": elapsed,
        "seconds_per_item": elapsed / items,
        "sec_prepare_mean": prepare_total / items,
        "sec_compute_mean": compute_total / items,
        "accuracy_on_labeled": (correct / scored) if scored else None,
        "labeled_items": scored,
        "peak_vram_gb": runtime.peak_vram_gb(),
        "tf32": runtime.tf32_state(),
    }
    return CheckResult(
        "inference",
        PASS,
        f"{detail['seconds_per_item']:.2f}s/item, 라벨 있는 {scored}건 정확도={detail['accuracy_on_labeled']}",
        detail,
    )


# --------------------------------------------------------------------------
# 커널 fallback 경고 수집
# --------------------------------------------------------------------------

#: transformers가 최적화된 커널 대신 느린 경로를 쓸 때 남기는 문구들.
FALLBACK_PATTERNS: tuple[str, ...] = ("falling back", "reference pytorch", "fast path")


class KernelFallbackCollector(logging.Handler):
    """transformers 로거에서 커널 fallback 경고만 골라 중복 없이 모은다."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:
            return
        lowered = text.lower()
        if not any(pattern in lowered for pattern in FALLBACK_PATTERNS):
            return
        if text not in self.messages:
            self.messages.append(text)


@contextlib.contextmanager
def capture_kernel_fallbacks(collector: KernelFallbackCollector) -> Iterator[KernelFallbackCollector]:
    """모델 로딩과 첫 forward 동안 transformers 로거에 핸들러를 붙인다.

    경고를 하나도 잡지 못해도 정상이다. 어떤 이유로든 로거를 건드리지 못하면 조용히
    아무것도 수집하지 않는다.
    """
    logger = None
    previous_level = None
    try:
        logger = logging.getLogger("transformers")
        previous_level = logger.level
        if previous_level > logging.WARNING or previous_level == logging.NOTSET:
            logger.setLevel(logging.WARNING)
        logger.addHandler(collector)
    except Exception:
        logger = None
    try:
        yield collector
    finally:
        if logger is not None:
            try:
                logger.removeHandler(collector)
                if previous_level is not None:
                    logger.setLevel(previous_level)
            except Exception:
                pass


#: Gated DeltaNet 커널이 참조 구현으로 내려갔을 때 경고에 들어가는 낱말들.
GATED_DELTA_HINTS: tuple[str, ...] = ("gated_delta", "gated delta", "deltanet", "delta_rule")

#: causal-conv1d가 없을 때 나오는 경고에 들어가는 낱말들. 속도 영향이 작아서 따로 본다.
CAUSAL_CONV_HINTS: tuple[str, ...] = ("causal_conv1d", "causal-conv1d", "short convolution")

#: fla 가속 커널을 설치하는 명령. --no-deps가 아니면 torch와 triton이 함께 바뀐다.
FLA_INSTALL_COMMAND = (
    "pip install --no-deps fla-core flash-linear-attention  그리고  pip install einops"
)


def classify_fallbacks(messages: Sequence[str]) -> dict[str, list[str]]:
    """수집한 fallback 경고를 연산 종류별로 나눈다."""
    groups: dict[str, list[str]] = {"gated_delta": [], "causal_conv": [], "other": []}
    for message in messages:
        lowered = message.lower()
        if any(hint in lowered for hint in GATED_DELTA_HINTS):
            groups["gated_delta"].append(message)
        elif any(hint in lowered for hint in CAUSAL_CONV_HINTS):
            groups["causal_conv"].append(message)
        else:
            groups["other"].append(message)
    return groups


def check_kernel_fast_path(
    messages: Sequence[str], *, model_loaded: bool = True
) -> CheckResult:
    """학습과 추론이 가속 커널 경로로 돌고 있는지 본다.

    Gated DeltaNet이 PyTorch 참조 구현으로 내려가면 학습이 약 두 배 느려진다. 그래도
    학습 자체는 되므로 FAIL이 아니라 WARN으로 표시하고 설치 명령을 함께 알려 준다.
    causal-conv1d만 없는 경우는 속도 영향이 작아서 정보로만 남긴다.
    """
    if not model_loaded:
        return CheckResult(
            "kernel_fast_path", SKIP, "모델을 올리지 않아 커널 경고를 수집하지 못했습니다."
        )
    groups = classify_fallbacks(messages)
    detail = {
        "messages": list(messages),
        "gated_delta": groups["gated_delta"],
        "causal_conv": groups["causal_conv"],
        "other": groups["other"],
        "install_command": FLA_INSTALL_COMMAND,
    }
    if groups["gated_delta"]:
        return CheckResult(
            "kernel_fast_path",
            WARN,
            "Gated DeltaNet이 PyTorch 참조 구현으로 내려갔습니다. 학습이 약 2배 느려집니다. "
            "다음 명령으로 가속 커널을 설치한 뒤 런타임을 재시작하십시오: " + FLA_INSTALL_COMMAND,
            detail,
        )
    if groups["causal_conv"]:
        return CheckResult(
            "kernel_fast_path",
            PASS,
            "Gated DeltaNet은 가속 커널을 씁니다. causal-conv1d만 참조 구현이며 속도 영향은 작습니다.",
            detail,
        )
    if groups["other"]:
        return CheckResult(
            "kernel_fast_path",
            PASS,
            "Gated DeltaNet fallback은 없습니다. 분류하지 못한 경고 %d건은 detail에 남겼습니다."
            % len(groups["other"]),
            detail,
        )
    return CheckResult(
        "kernel_fast_path", PASS, "커널 fallback 경고가 없습니다.", detail
    )


def fallback_summary(messages: Sequence[str]) -> str:
    """수집한 커널 fallback 경고를 요약 문단으로 만든다. 없으면 빈 문자열이다."""
    if not messages:
        return ""
    lines = ["", "커널 fallback 경고 (느린 경로로 내려간 연산)"]
    for message in messages:
        lines.append(f"  - {message.strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def timing_summary(results: Sequence[CheckResult]) -> str:
    """측정값을 단위와 함께 다시 정리한다. 예산 계산에서 단위를 혼동하지 않기 위해서다."""
    lines: list[str] = []
    for result in results:
        detail = result.detail or {}
        if result.name == "forward_backward" and "sec_per_microbatch_mean" in detail:
            lines.append("")
            lines.append("학습 속도 (microbatch 1개 = 표본 micro_batch개의 forward+backward 1회)")
            lines.append(f"  sec_per_microbatch_mean    : {detail['sec_per_microbatch_mean']:.3f} 초/microbatch (준비+계산 합)")
            if "sec_prepare_mean" in detail:
                lines.append(f"  sec_prepare_mean           : {detail['sec_prepare_mean']:.3f} 초/microbatch (CPU 배치 준비)")
            if "sec_compute_mean" in detail:
                lines.append(f"  sec_compute_mean           : {detail['sec_compute_mean']:.3f} 초/microbatch (GPU forward+backward)")
            lines.append(f"  sec_per_microbatch_longest : {detail['sec_per_microbatch_longest']:.3f} 초/microbatch (최장 표본)")
            lines.append(f"  grad_accum                 : {detail['grad_accum']} microbatch/optimizer step")
            lines.append(f"  sec_per_optimizer_step     : {detail['sec_per_optimizer_step']:.3f} 초/optimizer step  <- 예산은 이 값을 씁니다")
            lines.append(f"  peak_vram_gb               : {detail['peak_vram_gb']} GB")
            ab = detail.get("tf32_ab") or {}
            if ab.get("off") and ab.get("on"):
                lines.append("")
                lines.append(f"TF32 A/B (한쪽마다 warm-up {ab['on'].get('warmups')}회 뒤 {ab['on'].get('microbatches')} microbatch)")
                for label, key in (("off", "off"), ("on", "on")):
                    side = ab[key]
                    lines.append(
                        f"  TF32 {label:3s}                   : "
                        f"{side['sec_per_microbatch_mean']:.3f} 초/microbatch "
                        f"(준비 {side['sec_prepare_mean']:.3f} 초, 계산 {side['sec_compute_mean']:.3f} 초)"
                    )
                off_mean = ab["off"]["sec_per_microbatch_mean"]
                on_mean = ab["on"]["sec_per_microbatch_mean"]
                if on_mean > 0:
                    lines.append(f"  TF32 속도 배수             : {off_mean / on_mean:.2f}배 (off 기준 on이 몇 배 빠른지)")
        if result.name == "inference" and "seconds_per_item" in detail:
            lines.append("")
            lines.append("추론 속도")
            lines.append(f"  seconds_per_item           : {detail['seconds_per_item']:.3f} 초/문항 (준비+계산 합)")
            if "sec_prepare_mean" in detail:
                lines.append(f"  sec_prepare_mean           : {detail['sec_prepare_mean']:.3f} 초/문항 (CPU 배치 준비)")
            if "sec_compute_mean" in detail:
                lines.append(f"  sec_compute_mean           : {detail['sec_compute_mean']:.3f} 초/문항 (GPU scoring forward)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vqa.preflight", description="학습 전 환경 점검")
    config_mod.add_config_arguments(parser)
    parser.add_argument(
        "--n-train-steps",
        type=int,
        default=20,
        help="학습 점검에 쓸 optimizer step 수. 0을 주면 forward_backward 검사를 건너뛴다",
    )
    parser.add_argument("--n-infer", type=int, default=50)
    parser.add_argument(
        "--max-timing-microbatches",
        type=int,
        default=DEFAULT_MAX_TIMING_MICROBATCHES,
        help="속도 측정에 쓰는 microbatch 개수 상한 (n-train-steps * grad_accum을 넘지 않게 자른다)",
    )
    parser.add_argument("--csv", default=None, help="점검에 쓸 CSV. 기본값 DATA_ROOT/train.csv")
    parser.add_argument(
        "--pixel-sample",
        type=int,
        default=DEFAULT_PIXEL_SAMPLE,
        help="pixel_limits 검사에서 크기를 훑어 볼 이미지 개수 (기본: %d)" % DEFAULT_PIXEL_SAMPLE,
    )
    parser.add_argument(
        "--largest-image-samples",
        type=int,
        default=0,
        help=(
            "forward_backward의 OOM 표본에 원본 픽셀 수가 가장 큰 이미지를 이만큼 강제로 "
            "넣는다 (기본: 0, 넣지 않음). 무작위 표본과 글자 길이 기준 표본만으로는 큰 "
            "이미지가 빠질 수 있어서, 해상도를 올린 설정에서 OOM을 놓치지 않으려면 켜야 한다."
        ),
    )
    parser.add_argument("--skip-model", action="store_true", help="모델 로딩 없이 토큰/템플릿만 점검")
    parser.add_argument(
        "--tf32-ab",
        action="store_true",
        help=(
            f"무작위 표본 일부로 TF32를 끄고 한 번, 켜고 한 번 속도를 잰다 "
            f"(한쪽마다 warm-up {TF32_AB_WARMUPS}회 뒤 {TF32_AB_MICROBATCHES} microbatch)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.config_from_args(args)

    tf32_enabled = bool(config_mod.get_path(cfg, "runtime.tf32", True))
    tf32_info = runtime.apply_tf32(tf32_enabled)
    print(runtime.tf32_line(tf32_info))

    data_root = runtime.data_root(args.data_root)
    root = runtime.run_root(args.run_root)
    scratch = runtime.local_scratch(args.local_scratch)
    out_dir = runtime.ensure_dir(root / "preflight")

    flight = Preflight()
    paths = {"data_root": data_root, "run_root": root, "local_scratch": scratch}
    flight.run("environment", lambda: check_environment(cfg, paths))
    flight.run("bitsandbytes_4bit", check_bnb_4bit)

    csv_path = Path(args.csv) if args.csv else data_root / "train.csv"
    records: list[collator_mod.VQARecord] = []
    processor = None
    spec = prompt_mod.get_prompt(str(config_mod.get_path(cfg, "prompt_id")))
    tokens = None

    def load_data() -> CheckResult:
        nonlocal records
        records = collator_mod.load_records(csv_path, data_root=data_root)
        labeled = [r for r in records if r.answer_idx is not None]
        missing = []
        for record in records[:50]:
            path = collator_mod.resolve_image_path(record.path, data_root)
            if not path.exists():
                missing.append(str(path))
        detail = {"csv": str(csv_path), "rows": len(records), "labeled": len(labeled), "missing_sample": missing[:3]}
        if missing:
            return CheckResult("data", FAIL, f"앞 50행 중 {len(missing)}개 이미지를 찾지 못했습니다.", detail)
        return CheckResult("data", PASS, f"{len(records)}행, 라벨 {len(labeled)}건", detail)

    flight.run("data", load_data)

    def load_processor_check() -> CheckResult:
        nonlocal processor
        from . import modeling

        started = time.time()
        processor = modeling.load_processor(cfg)
        return CheckResult(
            "processor",
            PASS,
            f"{type(processor).__name__} ({time.time() - started:.1f}s)",
            {
                "class": type(processor).__name__,
                "min_pixels": config_mod.get_path(cfg, "min_pixels"),
                "max_pixels": config_mod.get_path(cfg, "max_pixels"),
                "padding_side": getattr(getattr(processor, "tokenizer", None), "padding_side", None),
                "source": config_mod.model_source(cfg),
            },
        )

    if flight.run("processor", load_processor_check).status == PASS and records:
        image = collator_mod.load_image(records[0].path, data_root)
        flight.run("choice_single_token", lambda: check_single_tokens(processor))
        flight.run("chat_template_tail", lambda: check_template(processor, spec, image))

        def choice_check() -> CheckResult:
            nonlocal tokens
            text = prompt_mod.full_generation_prompt(processor, spec, image)
            tokens = prompt_mod.resolve_choice_tokens(processor, text)
            return check_choice_tokens(processor, spec, image)

        flight.run("choice_token_after_tail", choice_check)
        flight.run(
            "pixel_limits",
            lambda: check_pixel_limits(processor, cfg, records, data_root, sample=args.pixel_sample),
        )
        flight.run("label_mask", lambda: check_label_mask(processor, spec, records, data_root, tokens))
        flight.run(
            "train_infer_prefix_match",
            lambda: check_prefix_match(processor, spec, records, data_root, tokens),
        )

    fallback_collector = KernelFallbackCollector()
    model_loaded = False

    if not args.skip_model and records:
        backend = None
        scorer = None

        def build_model() -> CheckResult:
            nonlocal backend
            from . import train as train_mod
            from . import modeling

            by_id = {r.id: r for r in records}
            ids = [r.id for r in records if r.answer_idx is not None][: max(args.n_train_steps, 1) * 4]
            started = time.time()
            backend = train_mod.VLMBackend(cfg=cfg, data_root=data_root, records=by_id, ids=ids).build()
            counts = modeling.count_parameters(backend.model)
            bad = modeling.verify_no_vision_adapters(backend.model)
            detail = {
                "load_seconds": time.time() - started,
                "loader_class": backend.loader_name,
                "parameters": counts,
                "lora_targets": backend.targets.as_dict(),
                "vision_adapters": bad,
            }
            if bad:
                return CheckResult("model_load", FAIL, f"vision 경로에 LoRA가 붙었습니다: {bad[:3]}", detail)
            return CheckResult(
                "model_load",
                PASS,
                f"{backend.loader_name}, trainable={counts['trainable']:,} / {counts['total']:,}",
                detail,
            )

        # 모델 로딩과 첫 forward 동안 transformers의 커널 fallback 경고를 모은다.
        with capture_kernel_fallbacks(fallback_collector):
            model_loaded = flight.run("model_load", build_model).status == PASS
            if model_loaded:
                # target 목록은 backend가 감싸기 전에 탐지해 둔 값을 그대로 쓴다.
                flight.run(
                    "lora_targets",
                    lambda: check_lora_targets(backend.model, cfg, getattr(backend, "targets", None))[0],
                )
                # 이미지가 가장 큰 문항은 전체 표본을 훑어서 고른다. 헤더만 읽으므로 빠르다.
                big_images = largest_image_records(
                    records, data_root, n=args.largest_image_samples, sample=len(records)
                )
                if big_images:
                    print(
                        "forward_backward의 OOM 표본에 가장 큰 이미지 %d장을 넣습니다: %s"
                        % (len(big_images), ", ".join(r.id for r in big_images)),
                        flush=True,
                    )
                flight.run(
                    "forward_backward",
                    lambda: check_forward_backward(
                        backend,
                        records,
                        args.n_train_steps,
                        grad_accum=int(config_mod.get_path(cfg, "train.grad_accum", 1) or 1),
                        largest_images=big_images,
                        max_timing_microbatches=args.max_timing_microbatches,
                        seed=int(config_mod.get_path(cfg, "train.seed", 42) or 0),
                        tf32_ab=bool(args.tf32_ab),
                        tf32_enabled=tf32_enabled,
                    ),
                )

        if model_loaded:

            def inference_check() -> CheckResult:
                nonlocal scorer
                from . import predict as predict_mod

                scorer = predict_mod.Scorer(cfg=cfg, data_root=data_root)
                scorer.processor = backend.processor
                scorer.spec = backend.spec
                scorer.choice_tokens = backend.choice_tokens
                scorer.model = backend.model
                scorer.device = backend.device
                scorer.adapter_path = None
                backend.model.eval()
                try:
                    return check_inference(
                        scorer, records, args.n_infer, int(config_mod.get_path(cfg, "infer.batch_size", 1))
                    )
                finally:
                    backend.model.train()

            flight.run("inference", inference_check)

    # 커널 경고는 모델 로딩과 첫 forward에서 나오므로 그 뒤에 판정한다.
    flight.run(
        "kernel_fast_path",
        lambda: check_kernel_fast_path(fallback_collector.messages, model_loaded=model_loaded),
    )

    payload = {
        "created": runtime.now_iso(),
        "config": cfg,
        "config_hash": config_mod.config_hash(cfg),
        "args": vars(args),
        "overall": FAIL if flight.failed else PASS,
        "results": [r.as_dict() for r in flight.results],
        "tf32": tf32_info,
        "kernel_fallback_warnings": list(fallback_collector.messages),
    }
    out_path = out_dir / f"{runtime.timestamp_tag()}.json"
    runtime.atomic_write_json(out_path, payload)

    print(flight.summary())
    print(timing_summary(flight.results))
    fallbacks = fallback_summary(fallback_collector.messages)
    if fallbacks:
        print(fallbacks)
    print(f"\n기록: {out_path}")
    return 1 if flight.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
