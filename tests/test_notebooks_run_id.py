"""21번 노트북의 파라미터와 run-id 파생 규칙 테스트.

run-id 파생은 노트북 셀 안에서만 돌아가는 코드라 직접 부를 수 없다. 그래서 두 갈래로 지킨다.
한쪽은 규칙을 그대로 옮겨 적은 아래의 `derive_run_ids` 로 3.5와 3.8의 이름이 실제로 갈라지는지
확인하고, 다른 한쪽은 생성된 노트북 원본에 그 규칙을 적용하는 문장이 남아 있는지 확인한다.
둘 중 하나만 바뀌면 테스트가 깨지므로 규칙과 노트북이 말없이 어긋나지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("yaml")

from vqa import config as config_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB_21 = ROOT / "notebooks" / "21_colab_27b_run.ipynb"

QWEN38_UPSCALE_CONFIG = "configs/colab_qwen38_27b_bf16_upscale.yaml"
UPSCALE_CONFIG = "configs/colab_qwen35_27b_bf16_upscale.yaml"
BF16_CONFIG = "configs/colab_qwen35_27b_bf16.yaml"


def derive_run_ids(config_path, prompt_id="v1_image_only", train_scope="fold0",
                   precision_suffix="", base_stem="colab27b_base_bf16",
                   ft_stem="colab27b_ft_bf16"):
    """21번 노트북 5번 절의 run-id 파생 규칙을 그대로 옮겨 적은 것이다."""
    cfg = config_mod.load_config(ROOT / config_path)
    prefix = config_mod.run_id_prefix(config_mod.get_path(cfg, "model_id"))
    default = config_mod.DEFAULT_RUN_ID_PREFIX
    if prefix != default:
        if base_stem.startswith(default):
            base_stem = prefix + base_stem[len(default):]
        if ft_stem.startswith(default):
            ft_stem = prefix + ft_stem[len(default):]

    prompt_suffix = "" if prompt_id == "v1_image_only" else "_" + prompt_id
    res_suffix = "_upscale" if "upscale" in Path(config_path).stem else ""
    scope_suffix = "_full" if train_scope == "full" else ""

    def resolve(stem):
        return stem + prompt_suffix + res_suffix + precision_suffix

    return resolve(base_stem), resolve(ft_stem + scope_suffix)


def test_qwen35_run_ids_are_unchanged():
    """기존 run 폴더와 예측 파일이 그대로 이어져야 한다."""
    assert derive_run_ids(BF16_CONFIG) == ("colab27b_base_bf16", "colab27b_ft_bf16")
    assert derive_run_ids(UPSCALE_CONFIG, prompt_id="v2_read_text") == (
        "colab27b_base_bf16_v2_read_text_upscale",
        "colab27b_ft_bf16_v2_read_text_upscale",
    )
    assert derive_run_ids(UPSCALE_CONFIG, train_scope="full")[1] == "colab27b_ft_bf16_full_upscale"


def test_qwen38_run_ids_get_their_own_prefix():
    assert derive_run_ids(QWEN38_UPSCALE_CONFIG, prompt_id="v2_read_text") == (
        "colab38_27b_base_bf16_v2_read_text_upscale",
        "colab38_27b_ft_bf16_v2_read_text_upscale",
    )
    assert derive_run_ids(QWEN38_UPSCALE_CONFIG, prompt_id="v2_read_text", train_scope="full") == (
        "colab38_27b_base_bf16_v2_read_text_upscale",
        "colab38_27b_ft_bf16_full_v2_read_text_upscale",
    )


def test_two_models_never_share_a_run_folder():
    for scope in ("fold0", "full"):
        for prompt_id in ("v1_image_only", "v2_read_text"):
            qwen35 = derive_run_ids(UPSCALE_CONFIG, prompt_id=prompt_id, train_scope=scope)
            qwen38 = derive_run_ids(QWEN38_UPSCALE_CONFIG, prompt_id=prompt_id, train_scope=scope)
            assert not set(qwen35) & set(qwen38), (scope, prompt_id)


def test_hand_edited_run_id_keeps_its_prefix():
    """파라미터 셀에서 이름을 직접 고쳤다면 앞머리를 바꾸지 않는다. 접미사 규칙은 그대로다."""
    base, ft = derive_run_ids(QWEN38_UPSCALE_CONFIG, base_stem="내_이름_base", ft_stem="내_이름_ft")
    assert (base, ft) == ("내_이름_base_upscale", "내_이름_ft_upscale")


# --------------------------------------------------------------------------
# 생성된 노트북 쪽
# --------------------------------------------------------------------------


def notebook_source():
    if not NB_21.exists():
        pytest.skip("노트북이 아직 생성되지 않았습니다. scripts/build_notebooks.py를 실행하십시오.")
    nb = json.loads(NB_21.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


def test_notebook_offers_the_qwen38_config():
    source = notebook_source()
    assert 'QWEN38_UPSCALE_CONFIG = "%s"' % QWEN38_UPSCALE_CONFIG in source


def test_notebook_derives_the_run_id_prefix_from_the_config():
    source = notebook_source()
    assert "_config_mod.run_id_prefix(MODEL_ID)" in source
    assert "_config_mod.DEFAULT_RUN_ID_PREFIX" in source


def test_notebook_reads_the_model_id_from_the_config():
    """MODEL_ID를 파라미터 셀에 하드코딩해 두면 CONFIG와 어긋난 가중치를 받게 된다."""
    source = notebook_source()
    assert '_cfg_model_id = str(_config_mod.get_path(_cfg_preview, "model_id"))' in source
    assert '_cfg_revision = _config_mod.get_path(_cfg_preview, "revision")' in source
    assert "MODEL_ID = _cfg_model_id" in source


# --------------------------------------------------------------------------
# VAL_FOLD / EPOCH_ADAPTER / Qwen3-VL-32B
# --------------------------------------------------------------------------

QWEN3VL_CONFIG = "configs/colab_qwen3vl_32b_bf16_upscale.yaml"


def test_qwen3vl_32b_gets_its_own_run_id_prefix():
    """27B run 폴더에 32B 예측이 섞이면 결합할 때 구분할 수 없다."""
    base, ft = derive_run_ids(QWEN3VL_CONFIG, prompt_id="v2_read_text")
    assert base == "qwen3vl_32b_base_bf16_v2_read_text_upscale"
    assert ft == "qwen3vl_32b_ft_bf16_v2_read_text_upscale"
    for other in (BF16_CONFIG, UPSCALE_CONFIG, QWEN38_UPSCALE_CONFIG):
        assert not set(derive_run_ids(other, prompt_id="v2_read_text")) & {base, ft}


def test_qwen3vl_32b_config_pins_what_the_notebook_reads():
    cfg = config_mod.load_config(ROOT / QWEN3VL_CONFIG)
    assert config_mod.get_path(cfg, "model_id") == "Qwen/Qwen3-VL-32B-Instruct"
    assert len(str(config_mod.get_path(cfg, "revision"))) == 40
    assert config_mod.get_path(cfg, "precision") == "bf16"
    assert config_mod.get_path(cfg, "attn_implementation") == "sdpa"
    assert config_mod.get_path(cfg, "prompt_id") == "v2_read_text"
    assert config_mod.get_path(cfg, "weights_gb") == 66
    assert config_mod.get_path(cfg, "lora.targets") == "auto"
    # 해상도는 27B 확대본과 같아야 모델 하나만 다른 비교가 된다.
    upscale = config_mod.load_config(ROOT / UPSCALE_CONFIG)
    for key in ("min_pixels", "max_pixels"):
        assert config_mod.get_path(cfg, key) == config_mod.get_path(upscale, key)


def test_notebook_offers_the_qwen3vl_config():
    assert 'QWEN3VL_CONFIG = "%s"' % QWEN3VL_CONFIG in notebook_source()


def test_notebook_splits_the_bf16_inference_and_training_thresholds():
    """가중치 66GB에서 80GB A100의 bf16 추론은 되고 학습은 막혀야 한다."""
    source = notebook_source()
    assert "VRAM_BF16_INFER_GB = round(WEIGHTS_GB + 12.0, 1)" in source
    assert "VRAM_BF16_GB = round(max(70.0 * _weights_scale, WEIGHTS_GB + 20.0), 1)" in source
    assert "WANT_NF4 = VRAM_GB < VRAM_BF16_INFER_GB" in source

    def thresholds(weights_gb):
        scale = weights_gb / 54.0
        return round(weights_gb + 12.0, 1), round(max(70.0 * scale, weights_gb + 20.0), 1)

    # 27B 54GB: 80GB A100에서 추론도 학습도 그대로 허용된다.
    infer_27b, train_27b = thresholds(54.0)
    assert 80.0 >= infer_27b and 80.0 >= train_27b
    # 32B 66GB: 추론은 허용, 학습은 weights_gb + 20 = 86GB라서 막힌다.
    infer_32b, train_32b = thresholds(66.0)
    assert 80.0 >= infer_32b
    assert train_32b == 86.0 and 80.0 < train_32b


def test_notebook_parameter_cell_offers_val_fold_and_epoch_adapter():
    source = notebook_source()
    assert "VAL_FOLD = 0" in source
    assert 'EPOCH_ADAPTER = ""' in source
    # fold 0은 지금까지 쓰던 분할과 검증 id를 그대로 쓴다.
    assert "SPLIT_FILE, SCOPE_SUFFIX = SPLIT_FILE_FOLD0, \"\"" in source
    # 다른 fold는 split_fold{N}.json 과 _fold{N} 접미사를 쓴다.
    assert 'SPLIT_FILE = "artifacts/data/split_fold%d.json" % VAL_FOLD' in source
    assert 'SCOPE_SUFFIX = "_fold%d" % VAL_FOLD' in source
    # 검증 id는 그 분할 파일의 val_ids 에서 읽는다.
    assert '_val_ids = [str(i) for i in _split_payload.get("val_ids") or []]' in source
    # epoch adapter는 final_adapter 자리에 그대로 들어간다.
    assert '"epoch_adapters"' in source
    assert 'EPOCH_SUFFIX = "_epoch%d" % EPOCH_NUMBER' in source
