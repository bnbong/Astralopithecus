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
    seed: int = 0,
    tf32_ab: bool = False,
    tf32_enabled: bool = True,
) -> CheckResult:
    """OOM은 가장 긴 표본으로, 속도는 무작위 표본으로 따로 본다.

    한 번의 forward+backward는 **microbatch 하나**다. optimizer step 하나는
    grad_accum개의 microbatch이므로 예산 계산에는 `sec_per_optimizer_step`을 써야
    한다. 가장 긴 표본만으로 잰 시간은 상한이라 평균 대신 쓰면 안 된다.

    시간은 CPU 준비 구간(make_micro_batch: 이미지 디코딩, processor, 템플릿)과 GPU
    구간(forward+backward)으로 나누어 잰다. 둘 중 어느 쪽이 병목인지 모르면 속도
    차이를 GPU 설정 탓으로만 돌리기 쉽다. `sec_per_microbatch_mean`은 예전과 같이
    두 구간의 합이다.
    """
    import random

    import torch

    labeled = [r for r in records if r.answer_idx is not None]
    if not labeled:
        return CheckResult("forward_backward", SKIP, "학습 표본이 없습니다.")

    grad_accum = max(int(grad_accum), 1)
    ordered = sorted(labeled, key=lambda r: -r.approx_length())
    longest = ordered[: max(min(int(n_longest), len(ordered)), 1)]

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

    # (a) 가장 긴 표본으로 OOM을 본다. 첫 회는 warm-up이라 cold로 따로 기록한다.
    longest_timings: list[float] = []
    losses: list[float] = []
    try:
        for record in longest:
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
    parser.add_argument("--n-train-steps", type=int, default=20)
    parser.add_argument("--n-infer", type=int, default=50)
    parser.add_argument(
        "--max-timing-microbatches",
        type=int,
        default=DEFAULT_MAX_TIMING_MICROBATCHES,
        help="속도 측정에 쓰는 microbatch 개수 상한 (n-train-steps * grad_accum을 넘지 않게 자른다)",
    )
    parser.add_argument("--csv", default=None, help="점검에 쓸 CSV. 기본값 DATA_ROOT/train.csv")
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
        flight.run("label_mask", lambda: check_label_mask(processor, spec, records, data_root, tokens))
        flight.run(
            "train_infer_prefix_match",
            lambda: check_prefix_match(processor, spec, records, data_root, tokens),
        )

    fallback_collector = KernelFallbackCollector()

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
                flight.run(
                    "forward_backward",
                    lambda: check_forward_backward(
                        backend,
                        records,
                        args.n_train_steps,
                        grad_accum=int(config_mod.get_path(cfg, "train.grad_accum", 1) or 1),
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
