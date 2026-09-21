"""preflight 측정값의 단위와 표본 선택을 확인한다.

여기서 지키려는 것은 한 가지다. forward+backward 한 번은 **microbatch 한 개**이고,
optimizer step 한 개는 그 grad_accum배다. 이 둘을 섞으면 학습 시간이 grad_accum배만큼
적게 잡힌다. 모델은 필요 없으므로 가짜 backend로 루프만 돌린다.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from vqa import preflight as preflight_mod  # noqa: E402
from vqa.collator import VQARecord  # noqa: E402


def make_records(n: int = 40) -> list[VQARecord]:
    records = []
    for i in range(1, n + 1):
        records.append(
            VQARecord(
                id=f"train_{i:04d}.jpg",
                path=f"train/train_{i:04d}.jpg",
                question="질" * i,  # 길이가 제각각이라 "가장 긴 표본"이 정해진다
                options=["가", "나", "다", "라"],
                answer_idx=i % 4,
            )
        )
    return records


class FakeBackend:
    """forward+backward를 흉내내며 어떤 id가 몇 번 들어왔는지 기록한다."""

    def __init__(self, prepare_sleep: float = 0.0, compute_sleep: float = 0.0) -> None:
        self.param = torch.nn.Parameter(torch.zeros(2))
        self.seen: list[str] = []
        self.prepare_sleep = prepare_sleep
        self.compute_sleep = compute_sleep

    def trainable_parameters(self):
        return [self.param]

    def make_micro_batch(self, ids, epoch):  # noqa: ARG002
        self.seen.extend(ids)
        if self.prepare_sleep:
            time.sleep(self.prepare_sleep)
        return list(ids)

    def compute_loss(self, batch):  # noqa: ARG002
        if self.compute_sleep:
            time.sleep(self.compute_sleep)
        return (self.param + 1.0).pow(2).sum()


def run(records, backend=None, **kwargs):
    backend = backend or FakeBackend()
    result = preflight_mod.check_forward_backward(backend, records, kwargs.pop("n_steps", 4), **kwargs)
    return backend, result


def test_optimizer_step_seconds_is_microbatch_seconds_times_grad_accum():
    backend, result = run(make_records(), n_steps=4, grad_accum=8)
    assert result.status == preflight_mod.PASS
    detail = result.detail
    assert detail["grad_accum"] == 8
    assert detail["sec_per_optimizer_step"] == pytest.approx(
        detail["sec_per_microbatch_mean"] * 8
    )
    # 예전 키는 microbatch 값의 별칭이다. optimizer step 시간이 아니다.
    assert detail["warm_seconds_mean"] == pytest.approx(detail["sec_per_microbatch_mean"])
    assert "s/microbatch" in result.reason and "s/optimizer step" in result.reason


def test_timing_uses_a_random_sample_not_only_the_longest():
    records = make_records(40)
    backend, result = run(records, n_steps=4, grad_accum=8, max_timing_microbatches=16)
    longest_ids = set(result.detail["longest_samples"])
    timed = backend.seen[len(longest_ids) :]
    assert result.detail["timing_microbatches"] == 16
    assert len(timed) == 16
    # 무작위 표본이므로 가장 긴 표본들만 다시 도는 일은 없어야 한다.
    assert not set(timed) <= longest_ids


def test_timing_count_is_capped_by_max_timing_microbatches():
    records = make_records(200)
    _, capped = run(records, n_steps=20, grad_accum=8, max_timing_microbatches=64)
    assert capped.detail["timing_microbatches"] == 64
    _, small = run(records, n_steps=2, grad_accum=2, max_timing_microbatches=64)
    assert small.detail["timing_microbatches"] == 4


def test_timing_sample_is_reproducible_for_the_same_seed():
    records = make_records(60)
    first, _ = run(records, n_steps=3, grad_accum=4, seed=7)
    second, _ = run(records, n_steps=3, grad_accum=4, seed=7)
    third, _ = run(records, n_steps=3, grad_accum=4, seed=8)
    assert first.seen == second.seen
    assert first.seen != third.seen


def test_longest_samples_are_checked_first_for_oom():
    records = make_records(40)
    backend, result = run(records, n_steps=2, grad_accum=2)
    longest = result.detail["longest_samples"]
    assert backend.seen[: len(longest)] == longest
    assert result.detail["longest_sample"] == "train_0040.jpg"


def test_timing_summary_states_the_units():
    _, result = run(make_records(), n_steps=3, grad_accum=8)
    text = preflight_mod.timing_summary([result])
    assert "초/microbatch" in text
    assert "초/optimizer step" in text
    assert "sec_per_optimizer_step" in text


def test_no_labeled_records_is_skipped():
    records = make_records(3)
    for record in records:
        record.answer_idx = None
    _, result = run(records)
    assert result.status == preflight_mod.SKIP


# --------------------------------------------------------------------------
# 준비 시간과 계산 시간의 분리
# --------------------------------------------------------------------------


def test_prepare_and_compute_are_measured_separately():
    """CPU 배치 준비와 GPU 계산을 나누어 재고, 합은 예전 키의 의미를 유지한다."""
    backend = FakeBackend(prepare_sleep=0.01, compute_sleep=0.003)
    _, result = run(make_records(), backend=backend, n_steps=1, grad_accum=4)
    detail = result.detail
    assert detail["sec_prepare_mean"] > detail["sec_compute_mean"]
    assert detail["sec_per_microbatch_mean"] == pytest.approx(
        detail["sec_prepare_mean"] + detail["sec_compute_mean"], rel=0.05
    )
    # 예산에 쓰는 값의 의미는 그대로다.
    assert detail["sec_per_optimizer_step"] == pytest.approx(detail["sec_per_microbatch_mean"] * 4)


def test_timing_summary_reports_the_prepare_compute_split():
    backend = FakeBackend(prepare_sleep=0.002, compute_sleep=0.002)
    _, result = run(make_records(), backend=backend, n_steps=1, grad_accum=4)
    text = preflight_mod.timing_summary([result])
    assert "sec_prepare_mean" in text
    assert "sec_compute_mean" in text
    assert "CPU 배치 준비" in text
    assert "GPU forward+backward" in text


# --------------------------------------------------------------------------
# TF32 A/B
# --------------------------------------------------------------------------


@pytest.fixture
def keep_tf32_flags():
    """A/B 측정은 전역 torch 플래그를 건드리므로 테스트가 끝나면 되돌려 놓는다."""
    before = preflight_mod.runtime.tf32_state()
    yield before
    preflight_mod.runtime.apply_tf32(bool(before.get("matmul")))


def test_tf32_ab_runs_both_sides_and_restores_the_configured_flag(keep_tf32_flags):
    backend = FakeBackend()
    _, result = run(backend=backend, records=make_records(40), n_steps=4, grad_accum=8, tf32_ab=True, tf32_enabled=True)
    ab = result.detail["tf32_ab"]
    assert set(ab) == {"off", "on"}
    for side in ab.values():
        assert side["microbatches"] == preflight_mod.TF32_AB_MICROBATCHES
        assert side["warmups"] == preflight_mod.TF32_AB_WARMUPS
        assert side["sec_per_microbatch_mean"] == pytest.approx(
            side["sec_prepare_mean"] + side["sec_compute_mean"], abs=1e-6
        )
    assert ab["off"]["tf32"]["requested"] is False
    assert ab["on"]["tf32"]["requested"] is True
    # 측정이 끝나면 설정값으로 되돌아온다.
    assert result.detail["tf32"]["requested"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert "TF32 off" in result.reason


def test_tf32_ab_restores_the_flag_when_it_is_configured_off(keep_tf32_flags):
    _, result = run(make_records(40), n_steps=4, grad_accum=8, tf32_ab=True, tf32_enabled=False)
    assert result.detail["tf32"]["requested"] is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_tf32_ab_is_not_run_by_default():
    _, result = run(make_records(40), n_steps=4, grad_accum=8)
    assert "tf32_ab" not in result.detail
    # A/B를 돌리지 않아도 현재 플래그 상태는 기록한다.
    assert set(result.detail["tf32"]) >= {"matmul", "cudnn", "float32_matmul_precision"}


def test_tf32_ab_summary_prints_both_means(keep_tf32_flags):
    _, result = run(make_records(40), n_steps=4, grad_accum=8, tf32_ab=True, tf32_enabled=True)
    text = preflight_mod.timing_summary([result])
    assert "TF32 A/B" in text
    assert "TF32 off" in text and "TF32 on" in text
    assert "TF32 속도 배수" in text


def test_tf32_ab_cli_flag_defaults_to_off():
    args = preflight_mod.build_parser().parse_args(["--config", "configs/lab_qwen35_9b.yaml"])
    assert args.tf32_ab is False
    args = preflight_mod.build_parser().parse_args(["--config", "configs/lab_qwen35_9b.yaml", "--tf32-ab"])
    assert args.tf32_ab is True


# --------------------------------------------------------------------------
# 커널 fallback 경고 수집
# --------------------------------------------------------------------------


def test_kernel_fallback_warnings_are_collected_without_duplicates():
    collector = preflight_mod.KernelFallbackCollector()
    logger = logging.getLogger("transformers")
    with preflight_mod.capture_kernel_fallbacks(collector):
        logger.warning("Falling back to the reference PyTorch implementation of gated delta rule.")
        logger.warning("Falling back to the reference PyTorch implementation of gated delta rule.")
        logger.warning("The fast path is not available because ...")
        logger.warning("모델을 불러왔습니다.")
    assert len(collector.messages) == 2
    assert any("reference PyTorch" in m for m in collector.messages)
    assert any("fast path" in m for m in collector.messages)
    # 핸들러는 블록이 끝나면 떨어진다.
    assert collector not in logger.handlers


def test_kernel_fallback_summary_is_empty_when_nothing_is_captured():
    collector = preflight_mod.KernelFallbackCollector()
    with preflight_mod.capture_kernel_fallbacks(collector):
        logging.getLogger("transformers").warning("평범한 메시지입니다.")
    assert collector.messages == []
    assert preflight_mod.fallback_summary(collector.messages) == ""
    assert "커널 fallback" in preflight_mod.fallback_summary(["Falling back to eager attention."])


# --------------------------------------------------------------------------
# 추론 시간의 분리
# --------------------------------------------------------------------------


class FakeScorer:
    """encode에서 CPU 준비를, score에서 GPU 계산을 흉내낸다."""

    def __init__(self, prepare_sleep: float = 0.004, compute_sleep: float = 0.001) -> None:
        self.prepare_sleep = prepare_sleep
        self.compute_sleep = compute_sleep

    def encode(self, records, perm):  # noqa: ARG002
        time.sleep(self.prepare_sleep)
        return list(records)

    def score(self, enc, *, also_base: bool = False):  # noqa: ARG002
        import numpy as np

        time.sleep(self.compute_sleep)
        return np.full((len(enc), 4), 0.25), None


def test_inference_splits_prepare_and_compute_time():
    result = preflight_mod.check_inference(FakeScorer(), make_records(6), 6, 1)
    assert result.status == preflight_mod.PASS
    detail = result.detail
    assert detail["sec_prepare_mean"] > detail["sec_compute_mean"]
    assert detail["seconds_per_item"] == pytest.approx(
        detail["sec_prepare_mean"] + detail["sec_compute_mean"], rel=0.2
    )
    assert detail["items"] == 6
    text = preflight_mod.timing_summary([result])
    assert "초/문항 (CPU 배치 준비)" in text
    assert "초/문항 (GPU scoring forward)" in text
