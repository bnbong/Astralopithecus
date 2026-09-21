"""preflight 측정값의 단위와 표본 선택을 확인한다.

여기서 지키려는 것은 한 가지다. forward+backward 한 번은 **microbatch 한 개**이고,
optimizer step 한 개는 그 grad_accum배다. 이 둘을 섞으면 학습 시간이 grad_accum배만큼
적게 잡힌다. 모델은 필요 없으므로 가짜 backend로 루프만 돌린다.
"""

from __future__ import annotations

import sys
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

    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.zeros(2))
        self.seen: list[str] = []

    def trainable_parameters(self):
        return [self.param]

    def make_micro_batch(self, ids, epoch):  # noqa: ARG002
        self.seen.extend(ids)
        return list(ids)

    def compute_loss(self, batch):  # noqa: ARG002
        return (self.param + 1.0).pow(2).sum()


def run(records, **kwargs):
    backend = FakeBackend()
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
