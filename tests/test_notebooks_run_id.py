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
