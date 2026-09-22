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


def test_largest_images_are_added_to_the_oom_group():
    records = make_records(40)
    # 글자 수가 짧아서 "가장 긴 표본"에는 절대 들어가지 않는 문항을 큰 이미지로 지정한다.
    big = [records[0], records[1]]
    backend, result = run(records, n_steps=2, grad_accum=2, largest_images=big)
    assert result.detail["largest_image_samples"] == [big[0].id, big[1].id]
    longest = result.detail["longest_samples"]
    assert backend.seen[: len(longest)] == longest
    assert backend.seen[len(longest) : len(longest) + 2] == [big[0].id, big[1].id]
    assert "가장 큰 이미지 2장 포함" in result.reason


def test_largest_images_are_not_run_twice_when_already_longest():
    records = make_records(40)
    longest_record = records[-1]  # 글자 수가 가장 긴 문항이다
    _, result = run(records, n_steps=2, grad_accum=2, largest_images=[longest_record])
    assert result.detail["largest_image_samples"] == []
    assert longest_record.id in result.detail["longest_samples"]


def test_largest_images_are_off_by_default():
    _, result = run(make_records(20), n_steps=2, grad_accum=2)
    assert result.detail["largest_image_samples"] == []
    assert "가장 큰 이미지" not in result.reason


def test_zero_train_steps_skips_the_check_without_running_a_batch():
    # 학습을 하지 않는 GPU에서는 이 검사만으로 OOM이 나므로 모델을 한 번도 돌리지 않아야 한다.
    backend, result = run(make_records(20), n_steps=0, grad_accum=8)
    assert result.status == preflight_mod.SKIP
    assert backend.seen == []


def test_negative_train_steps_also_skips_the_check():
    _, result = run(make_records(20), n_steps=-1, grad_accum=8)
    assert result.status == preflight_mod.SKIP


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


# --------------------------------------------------------------------------
# kernel_fast_path: 가속 커널 경로 판정
# --------------------------------------------------------------------------

GATED_DELTA_WARNING = (
    "`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation "
    "because the fla package is not installed."
)
CAUSAL_CONV_WARNING = (
    "`causal_conv1d` is falling back to its reference PyTorch implementation "
    "because causal-conv1d is not installed."
)


def test_kernel_fast_path_warns_when_gated_delta_falls_back():
    result = preflight_mod.check_kernel_fast_path([GATED_DELTA_WARNING])
    assert result.status == preflight_mod.WARN
    assert "2배" in result.reason
    assert "--no-deps" in result.reason
    assert result.detail["gated_delta"] == [GATED_DELTA_WARNING]
    assert result.detail["causal_conv"] == []


def test_kernel_fast_path_treats_causal_conv_only_as_information():
    result = preflight_mod.check_kernel_fast_path([CAUSAL_CONV_WARNING])
    assert result.status == preflight_mod.PASS
    assert "causal-conv1d" in result.reason
    assert result.detail["causal_conv"] == [CAUSAL_CONV_WARNING]


def test_kernel_fast_path_passes_without_any_warning():
    result = preflight_mod.check_kernel_fast_path([])
    assert result.status == preflight_mod.PASS
    assert result.detail["messages"] == []


def test_kernel_fast_path_is_skipped_without_a_model():
    result = preflight_mod.check_kernel_fast_path([], model_loaded=False)
    assert result.status == preflight_mod.SKIP


def test_kernel_fast_path_warns_even_when_other_warnings_are_mixed_in():
    messages = ["Falling back to eager attention.", GATED_DELTA_WARNING, CAUSAL_CONV_WARNING]
    result = preflight_mod.check_kernel_fast_path(messages)
    assert result.status == preflight_mod.WARN
    assert result.detail["other"] == ["Falling back to eager attention."]


def test_kernel_fast_path_warning_does_not_fail_the_preflight():
    """WARN은 종료 코드를 바꾸지 않는다. 학습 자체는 가능하기 때문이다."""
    flight = preflight_mod.Preflight()
    flight.run("kernel_fast_path", lambda: preflight_mod.check_kernel_fast_path([GATED_DELTA_WARNING]))
    assert flight.failed == []


# --------------------------------------------------------------------------
# pixel_limits: min/max_pixels가 processor에 실제로 먹었는지 확인
# --------------------------------------------------------------------------

PATCH_SIZE = 16
MERGE_SIZE = 2
PIXELS_PER_TOKEN = (PATCH_SIZE * MERGE_SIZE) ** 2  # 1024


class FakeImageProcessor:
    """min/max_pixels를 적용할지 말지를 골라서 흉내내는 가짜 image_processor다."""

    def __init__(self, min_pixels, max_pixels, *, enforce=True,
                 patch_size=PATCH_SIZE, merge_size=MERGE_SIZE):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.enforce = enforce
        self.patch_size = patch_size
        self.merge_size = merge_size

    def __call__(self, images, return_tensors=None):  # noqa: ARG002
        grids = []
        for image in images:
            width, height = image.size
            pixels = width * height
            if self.enforce:
                pixels = min(max(pixels, self.min_pixels), self.max_pixels)
            tokens = max(int(pixels // PIXELS_PER_TOKEN), 1)
            # [t, h, w]를 patch 단위로 돌려준다. merge로 나누면 tokens가 된다.
            grids.append([1, tokens * self.merge_size, self.merge_size])
        return {"image_grid_thw": grids}


class FakeProcessor:
    def __init__(self, image_processor):
        self.image_processor = image_processor


def make_pixel_fixture(tmp_path, sizes):
    """지정한 크기의 이미지를 만들고 (records, data_root)를 돌려준다."""
    from PIL import Image

    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    records = []
    for index, (width, height) in enumerate(sizes, start=1):
        name = "train_%04d.jpg" % index
        Image.new("RGB", (width, height), (128, 128, 128)).save(tmp_path / "train" / name)
        records.append(
            VQARecord(
                id=name,
                path="train/" + name,
                question="질문",
                options=["가", "나", "다", "라"],
                answer_idx=0,
            )
        )
    return records, tmp_path


PIXEL_CFG = {"min_pixels": 262144, "max_pixels": 1048576}


def test_pixel_limits_passes_when_the_processor_applies_the_budget(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536), (320, 240)])
    processor = FakeProcessor(FakeImageProcessor(262144, 1048576, enforce=True))
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.PASS
    detail = result.detail
    assert detail["largest"]["width"] == 2048 and detail["largest"]["height"] == 1536
    assert detail["smallest"]["width"] == 320 and detail["smallest"]["height"] == 240
    assert detail["expected_tokens_upper"] == 1024.0
    assert detail["largest"]["visual_tokens"] <= 1024
    assert detail["upper_binding"] is True
    assert "시각 토큰" in result.reason


def test_pixel_limits_fails_when_max_pixels_is_ignored(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(4000, 3000), (320, 240)])
    processor = FakeProcessor(FakeImageProcessor(262144, 1048576, enforce=False))
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.FAIL
    assert "max_pixels" in result.reason
    assert result.detail["largest"]["visual_tokens"] > result.detail["expected_tokens_upper"]


def test_pixel_limits_fails_when_min_pixels_is_ignored(tmp_path):
    # 가장 큰 이미지는 상한 안에 들어오고, 가장 작은 이미지만 하한에 못 미치게 만든다.
    records, data_root = make_pixel_fixture(tmp_path, [(1024, 1024), (64, 64)])
    processor = FakeProcessor(FakeImageProcessor(262144, 1048576, enforce=False))
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.FAIL
    assert "min_pixels" in result.reason
    assert result.detail["lower_binding"] is True


def test_pixel_limits_warns_when_no_image_reaches_the_upper_bound(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(800, 600), (640, 480)])
    processor = FakeProcessor(FakeImageProcessor(262144, 1048576, enforce=True))
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.WARN
    assert result.detail["upper_binding"] is False
    assert "확인하지 못했습니다" in result.reason


def test_pixel_limits_warns_without_patch_or_merge_size(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536)])
    processor = FakeProcessor(FakeImageProcessor(262144, 1048576, patch_size=None, merge_size=None))
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.WARN
    assert "patch_size" in result.reason


def test_pixel_limits_reads_patch_and_merge_size_from_the_processor(tmp_path):
    """patch/merge 크기를 하드코딩하지 않고 image_processor에서 읽는다."""
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536), (320, 240)])
    processor = FakeProcessor(
        FakeImageProcessor(262144, 1048576, enforce=True, patch_size=14, merge_size=2)
    )
    result = preflight_mod.check_pixel_limits(processor, PIXEL_CFG, records, data_root)
    assert result.detail["patch_size"] == 14
    assert result.detail["merge_size"] == 2
    assert result.detail["pixels_per_token"] == (14 * 2) ** 2
    assert result.detail["expected_tokens_upper"] == 1048576 / ((14 * 2) ** 2)


def test_pixel_limits_skips_without_an_image_processor(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(800, 600)])

    class Bare:
        pass

    result = preflight_mod.check_pixel_limits(Bare(), PIXEL_CFG, records, data_root)
    assert result.status == preflight_mod.SKIP


def test_pixel_limits_warns_without_image_grid_thw(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536)])

    class NoGrid(FakeImageProcessor):
        def __call__(self, images, return_tensors=None):  # noqa: ARG002
            return {"pixel_values": [0]}

    result = preflight_mod.check_pixel_limits(
        FakeProcessor(NoGrid(262144, 1048576)), PIXEL_CFG, records, data_root
    )
    assert result.status == preflight_mod.WARN
    assert "image_grid_thw" in result.reason


def test_pixel_limits_accepts_a_processor_without_the_images_keyword(tmp_path):
    """transformers 버전에 따라 호출 형태가 달라도 검사가 그대로 동작해야 한다."""
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536), (320, 240)])

    class PositionalOnly(FakeImageProcessor):
        def __call__(self, images, return_tensors=None):  # noqa: ARG002
            if return_tensors is not None:
                raise TypeError("return_tensors를 받지 않습니다.")
            return FakeImageProcessor.__call__(self, images)

    result = preflight_mod.check_pixel_limits(
        FakeProcessor(PositionalOnly(262144, 1048576, enforce=True)), PIXEL_CFG, records, data_root
    )
    assert result.status == preflight_mod.PASS


def test_pixel_limits_warns_when_the_processor_cannot_be_called(tmp_path):
    """호출 자체가 막히면 해상도가 틀렸다는 뜻이 아니므로 FAIL이 아니라 WARN이어야 한다."""
    records, data_root = make_pixel_fixture(tmp_path, [(2048, 1536), (320, 240)])

    class Broken(FakeImageProcessor):
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            raise TypeError("이 버전은 다른 인자를 받습니다.")

    result = preflight_mod.check_pixel_limits(
        FakeProcessor(Broken(262144, 1048576)), PIXEL_CFG, records, data_root
    )
    assert result.status == preflight_mod.WARN
    assert "확인하지 못했습니다" in result.reason
    assert "TypeError" in result.detail["probe_errors"]


def test_largest_image_records_picks_the_biggest_first(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(320, 240), (2048, 1536), (800, 600)])
    picked = preflight_mod.largest_image_records(records, data_root, n=2)
    assert [r.id for r in picked] == ["train_0002.jpg", "train_0003.jpg"]


def test_largest_image_records_returns_nothing_when_disabled(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(320, 240), (2048, 1536)])
    assert preflight_mod.largest_image_records(records, data_root, n=0) == []


def test_largest_image_records_skips_unlabeled_and_missing_files(tmp_path):
    records, data_root = make_pixel_fixture(tmp_path, [(320, 240), (2048, 1536)])
    records[1].answer_idx = None  # 가장 큰 이미지지만 학습 표본이 아니다
    records.append(
        VQARecord(id="train_9999.jpg", path="train/train_9999.jpg", question="질문",
                  options=["가", "나", "다", "라"], answer_idx=0)
    )
    picked = preflight_mod.largest_image_records(records, data_root, n=3)
    assert [r.id for r in picked] == ["train_0001.jpg"]


def test_largest_image_cli_flag_defaults_to_off():
    parser = preflight_mod.build_parser()
    args = parser.parse_args(["--config", "configs/lab_qwen35_9b.yaml"])
    assert args.largest_image_samples == 0
    args = parser.parse_args(
        ["--config", "configs/lab_qwen35_9b.yaml", "--largest-image-samples", "2"]
    )
    assert args.largest_image_samples == 2


def test_pixel_limits_cli_sample_flag():
    parser = preflight_mod.build_parser()
    args = parser.parse_args(["--config", "configs/lab_qwen35_9b.yaml"])
    assert args.pixel_sample == preflight_mod.DEFAULT_PIXEL_SAMPLE
    args = parser.parse_args(["--config", "configs/lab_qwen35_9b.yaml", "--pixel-sample", "10"])
    assert args.pixel_sample == 10
