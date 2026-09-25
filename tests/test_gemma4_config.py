"""Gemma-4-31B 비교 실험 설정과 그 설정이 건드리는 파이프라인 지점 테스트.

Gemma-4는 Qwen 계열과도 InternVL 계열과도 계통이 다르다. 확인해야 하는 것은 세 가지다.
설정 파일이 노트북과 preflight 가 읽는 키를 모두 들고 있는지, run-id 앞머리가 27B run 과
갈라지는지, 그리고 이미지 목록을 표본마다 중첩해서 받아야 하는 processor 에서도 배치
인코딩이 되는지다. 마지막 항목은 실제 가중치 없이 확인해야 하므로 가짜 processor 를 쓴다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("yaml")

from vqa import collator as collator_mod  # noqa: E402
from vqa import config as config_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEMMA4_CONFIG = ROOT / "configs" / "colab_gemma4_31b_bf16.yaml"

#: Gemma4ImageProcessor 가 받는 시각 토큰 예산. 이 다섯 개 밖의 값은 거부된다.
SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def test_config_pins_what_the_notebook_and_preflight_read():
    cfg = config_mod.load_config(GEMMA4_CONFIG)
    assert config_mod.get_path(cfg, "model_id") == "google/gemma-4-31B-it"
    assert len(str(config_mod.get_path(cfg, "revision"))) == 40
    assert config_mod.get_path(cfg, "precision") == "bf16"
    assert config_mod.get_path(cfg, "attn_implementation") == "sdpa"
    assert config_mod.get_path(cfg, "prompt_id") == "v2_read_text"
    assert config_mod.get_path(cfg, "weights_gb") == 62
    assert config_mod.get_path(cfg, "lora.targets") == "auto"


def test_visual_token_budget_is_one_of_the_allowed_values():
    """max_soft_tokens 가 다섯 개 값 밖으로 나가면 processor 가 거부한다."""
    cfg = config_mod.load_config(GEMMA4_CONFIG)
    budget = config_mod.get_path(cfg, "processor_kwargs.max_soft_tokens")
    assert budget in SOFT_TOKEN_BUDGETS
    assert budget == max(SOFT_TOKEN_BUDGETS)


def test_bf16_inference_fits_in_eighty_gigabytes_but_training_does_not():
    """21번 노트북 5번 절의 하한 계산을 그대로 옮겨 적은 것이다."""
    weights_gb = float(config_mod.get_path(config_mod.load_config(GEMMA4_CONFIG), "weights_gb"))
    infer_min = round(weights_gb + 12.0, 1)
    train_min = round(max(70.0 * (weights_gb / 54.0), weights_gb + 20.0), 1)
    assert infer_min <= 80.0
    assert train_min > 80.0


def test_run_id_prefix_is_separate_from_the_27b_runs():
    cfg = config_mod.load_config(GEMMA4_CONFIG)
    prefix = config_mod.run_id_prefix(config_mod.get_path(cfg, "model_id"))
    assert prefix == "gemma4_31b"
    assert prefix != config_mod.DEFAULT_RUN_ID_PREFIX
    # base 판을 지정해도 같은 폴더로 떨어져야 한다.
    assert config_mod.run_id_prefix("google/gemma-4-31B") == "gemma4_31b"


def test_config_hash_reacts_to_the_visual_token_budget():
    """예산을 바꾼 run 이 같은 checkpoint 를 이어받으면 안 된다."""
    base = config_mod.load_config(GEMMA4_CONFIG)
    changed = config_mod.load_config(GEMMA4_CONFIG, ["processor_kwargs.max_soft_tokens=560"])
    assert config_mod.config_hash(base) != config_mod.config_hash(changed)


# --------------------------------------------------------------------------
# 이미지 목록 중첩을 요구하는 processor
# --------------------------------------------------------------------------


class _NestedImagesProcessor:
    """Gemma4Processor 처럼 표본마다 이미지 목록이 중첩되어 있기를 요구하는 가짜 processor."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def __call__(self, *, text, images, **kwargs):
        self.calls.append(images)
        if any(not isinstance(item, (list, tuple)) for item in images):
            raise ValueError(
                f"Received inconsistently sized batches of images (1) and text ({len(text)})."
            )
        return {"text": list(text), "images": [list(item) for item in images], "kwargs": kwargs}


class _FlatImagesProcessor:
    """Qwen 과 InternVL 처럼 평평한 목록을 그대로 받는 가짜 processor."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def __call__(self, *, text, images, **kwargs):
        self.calls.append(images)
        return {"text": list(text), "images": list(images), "kwargs": kwargs}


def test_call_processor_retries_with_nested_images():
    processor = _NestedImagesProcessor()
    out = collator_mod.call_processor(
        processor, dict(text=["t0", "t1"], images=["i0", "i1"]), padding_side="right"
    )
    assert out["images"] == [["i0"], ["i1"]]
    assert out["kwargs"]["padding_side"] == "right"
    # 평평한 호출을 먼저 시도하고 실패한 뒤에만 중첩한다.
    assert processor.calls == [["i0", "i1"], [["i0"], ["i1"]]]


def test_call_processor_leaves_flat_processors_alone():
    """평평한 호출이 성공하면 중첩 경로에 들어가지 않아야 한다."""
    processor = _FlatImagesProcessor()
    out = collator_mod.call_processor(processor, dict(text=["t0", "t1"], images=["i0", "i1"]))
    assert out["images"] == ["i0", "i1"]
    assert processor.calls == [["i0", "i1"]]


def test_call_processor_reraises_when_images_are_already_nested():
    """중첩된 목록으로도 실패하면 원래 오류를 그대로 올려야 한다."""

    class _AlwaysFails(_NestedImagesProcessor):
        def __call__(self, *, text, images, **kwargs):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        collator_mod.call_processor(_AlwaysFails(), dict(text=["t0"], images=[["i0"]]))
