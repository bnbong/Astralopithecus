"""InternVL3.5-38B NF4 QLoRA 설정 테스트.

38B 는 결합에 넣을 네 번째 계통이다. 8B 판과 같은 파일 이름 규칙을 쓰기 때문에, 설정이
조용히 8B 쪽 값으로 되돌아가거나 run-id 앞머리가 8B 와 겹치면 두 모델의 예측이 한 run
폴더에 섞인다. 그래서 확인하는 것은 네 가지다. 설정 파일이 노트북과 preflight 가 읽는 키를
모두 들고 있는지, 80GB A100 에서 학습이 허용되는 크기인지, run-id 가 8B 와 27B 와 갈라지는지,
그리고 해상도를 정하는 키가 processor_kwargs.max_patches 하나인지다.

가중치를 받지 않고 확인할 수 있는 범위만 다룬다. LoRA target 448개와 최상위 모듈 이름은
meta device 로 따로 확인했고 그 결과를 설정 파일 주석에 적어 두었다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("yaml")

from vqa import config as config_mod  # noqa: E402
from vqa import modeling as modeling_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERNVL8_CONFIG = ROOT / "configs" / "colab_internvl35_8b_bf16.yaml"
INTERNVL38_CONFIG = ROOT / "configs" / "colab_internvl35_38b_nf4_lora.yaml"
GEMMA4_QLORA_CONFIG = ROOT / "configs" / "colab_gemma4_31b_nf4_lora.yaml"


def test_config_pins_what_the_notebook_and_preflight_read():
    cfg = config_mod.load_config(INTERNVL38_CONFIG)
    assert config_mod.get_path(cfg, "model_id") == "OpenGVLab/InternVL3_5-38B-HF"
    # revision 이 비어 있으면 5번 절이 main 을 쓰고, 나중에 같은 가중치를 다시 받을 수 없다.
    assert config_mod.get_path(cfg, "revision") == "7c830fc25e874f9a861087e3c70587f2b9b555f3"
    assert config_mod.get_path(cfg, "precision") == "nf4"
    assert config_mod.get_path(cfg, "attn_implementation") == "sdpa"
    assert config_mod.get_path(cfg, "prompt_id") == "v2_read_text"
    assert config_mod.get_path(cfg, "weights_gb") == 45
    assert config_mod.get_path(cfg, "lora.targets") == "auto"


def test_training_settings_match_the_gemma_qlora_run():
    """LoRA 와 train 설정이 다르면 Gemma 결과와 견줄 수 없다."""
    gemma = config_mod.load_config(GEMMA4_QLORA_CONFIG)
    cfg = config_mod.load_config(INTERNVL38_CONFIG)
    for key in (
        "precision",
        "attn_implementation",
        "prompt_id",
        "lora.r",
        "lora.alpha",
        "lora.dropout",
        "lora.targets",
        "train.lr",
        "train.epochs",
        "train.micro_batch",
        "train.grad_accum",
        "train.warmup_ratio",
        "train.scheduler",
        "train.weight_decay",
        "train.max_grad_norm",
        "train.seed",
        "train.shuffle_options",
        "infer.batch_size",
    ):
        assert config_mod.get_path(cfg, key) == config_mod.get_path(gemma, key), key


def test_resolution_comes_only_from_max_patches():
    """InternVL 의 image processor 는 min/max_pixels 를 받고도 무시한다.

    그래서 해상도를 바꾸는 키는 processor_kwargs.max_patches 하나뿐이다. min/max_pixels 는
    스키마를 맞추기 위한 값이며 8B 설정과 같아야 한다.
    """
    cfg = config_mod.load_config(INTERNVL38_CONFIG)
    extra = config_mod.get_path(cfg, "processor_kwargs")
    assert extra == {"max_patches": 4}
    eight = config_mod.load_config(INTERNVL8_CONFIG)
    for key in ("min_pixels", "max_pixels"):
        assert config_mod.get_path(cfg, key) == config_mod.get_path(eight, key), key


def test_visual_token_budget_stays_under_the_eight_b_default():
    """max_patches 4 는 타일 5장, 시각 토큰 1,280개다.

    기본값 12(타일 13장, 3,328개)로 되돌아가면 670 step 학습이 하룻밤에 끝나지 않는다.
    타일 수는 max_patches 에 썸네일 한 장을 더한 값이고 타일마다 image_seq_length 가 256이다.
    """
    budget = config_mod.get_path(config_mod.load_config(INTERNVL38_CONFIG), "processor_kwargs.max_patches")
    tiles = budget + 1
    assert tiles == 5
    assert tiles * 256 == 1280
    assert tiles * 256 < 13 * 256


def test_nf4_training_fits_in_eighty_gigabytes_only():
    """21번 노트북 5번 절의 하한 계산을 그대로 옮겨 적은 것이다."""
    weights_gb = float(config_mod.get_path(config_mod.load_config(INTERNVL38_CONFIG), "weights_gb"))
    scale = weights_gb / 54.0
    train_min = round(max(70.0 * scale, weights_gb + 20.0), 1)
    assert train_min == 65.0
    assert train_min <= 80.0
    # 40GB A100 에서는 막혀야 한다. 가중치만 45GB 라서 활성값이 들어가지 않는다.
    assert train_min > 40.0
    # 추론 하한도 80GB 안이어야 한다. 넘으면 5번 절이 정밀도를 한 번 더 내리려 든다.
    assert round(weights_gb + 12.0, 1) <= 80.0
    # 최소 추론 하한은 40GB A100 도 통과한다. 그 가상 머신에서는 추론만 돌린다.
    assert round(20.0 * scale, 1) <= 40.0


def test_quantization_skips_the_vision_side_and_the_output_layer():
    """InternViT-6B 를 4bit 로 내리면 이미지 특징이 먼저 뭉개진다.

    이름은 meta device 로 확인한 최상위 모듈 이름과 같아야 한다. 오타가 나면 조용히
    양자화되고, 그 사실은 fold 0 정확도가 낮게 나온 다음에야 드러난다.
    """
    cfg = config_mod.load_config(INTERNVL38_CONFIG)
    skip = config_mod.get_path(cfg, "quant_skip_modules")
    assert skip == ["model.vision_tower", "model.multi_modal_projector", "lm_head"]
    # quantizer 에 넘어가는 목록에 lm_head 가 빠지면 출력 계층까지 4bit 로 내려간다.
    resolved = modeling_mod.resolve_quant_skip_modules(skip)
    assert "lm_head" in resolved
    for name in skip:
        assert name in resolved


def test_run_id_prefix_is_separate_from_the_eight_b_and_the_27b_runs():
    cfg = config_mod.load_config(INTERNVL38_CONFIG)
    prefix = config_mod.run_id_prefix(config_mod.get_path(cfg, "model_id"))
    assert prefix == "internvl35_38b"
    assert prefix != config_mod.DEFAULT_RUN_ID_PREFIX
    assert prefix != config_mod.run_id_prefix("OpenGVLab/InternVL3_5-8B-HF")


def test_config_hash_reacts_to_the_visual_token_budget():
    """예산을 바꾼 run 이 같은 checkpoint 를 이어받으면 안 된다."""
    base = config_mod.load_config(INTERNVL38_CONFIG)
    changed = config_mod.load_config(INTERNVL38_CONFIG, ["processor_kwargs.max_patches=2"])
    assert config_mod.config_hash(base) != config_mod.config_hash(changed)
    # 양자화 제외 목록도 학습 결과를 바꾸므로 해시에 들어가야 한다.
    quantized_vision = config_mod.load_config(
        INTERNVL38_CONFIG, ['quant_skip_modules=["lm_head"]']
    )
    assert config_mod.config_hash(base) != config_mod.config_hash(quantized_vision)


def test_config_hash_differs_from_the_eight_b_config():
    """8B 와 38B 의 checkpoint 가 서로 이어지면 안 된다."""
    assert config_mod.config_hash(config_mod.load_config(INTERNVL38_CONFIG)) != config_mod.config_hash(
        config_mod.load_config(INTERNVL8_CONFIG)
    )
