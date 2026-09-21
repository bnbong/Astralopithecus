"""LoRA target 자동 탐지 테스트.

vision tower와 language backbone에 같은 leaf 이름(q_proj 등)을 가진 가짜 모델을 만들어,
탐지기가 language 쪽만 고르는지 본다. 실제 모델 가중치는 필요 없다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
nn = torch.nn

from vqa import modeling  # noqa: E402


class AttentionBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class LinearAttentionBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.in_proj_qkv = nn.Linear(dim, dim * 3)
        self.in_proj_z = nn.Linear(dim, dim)
        self.in_proj_a = nn.Linear(dim, dim)
        self.in_proj_b = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


class MLP(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)


class LanguageLayer(nn.Module):
    def __init__(self, linear_attention=False):
        super().__init__()
        self.self_attn = LinearAttentionBlock() if linear_attention else AttentionBlock()
        self.mlp = MLP()


class VisionLayer(nn.Module):
    """language와 같은 leaf 이름을 가진 vision block."""

    def __init__(self):
        super().__init__()
        self.attn = AttentionBlock()
        self.mlp = MLP()


class FakeQwen35(nn.Module):
    def __init__(self, n_language=4, n_vision=3):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([VisionLayer() for _ in range(n_vision)])
        self.model.visual.merger = MLP()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [LanguageLayer(linear_attention=(i % 2 == 1)) for i in range(n_language)]
        )
        self.lm_head = nn.Linear(8, 16)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """PEFT의 CAUSAL_LM 래퍼가 존재를 확인하는 메서드 자리."""
        return kwargs


class FakeNoLanguageRoot(nn.Module):
    """language_model 속성이 없는 구조. vision 이름 필터만으로 걸러야 한다."""

    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.layers = nn.ModuleList([VisionLayer() for _ in range(2)])
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([LanguageLayer() for _ in range(2)])


def test_detects_only_language_modules():
    model = FakeQwen35()
    targets = modeling.detect_lora_targets(model)

    assert targets.language_prefix == "model.language_model"
    assert targets.names, "language target을 찾지 못했습니다."
    for name in targets.names:
        assert name.startswith("model.language_model.")
        assert "visual" not in name and "merger" not in name
    assert targets.excluded_vision, "vision 모듈이 제외 목록에 있어야 합니다."
    for name in targets.excluded_vision:
        assert "visual" in name or "merger" in name


def test_counts_per_leaf_type():
    model = FakeQwen35(n_language=4)
    targets = modeling.detect_lora_targets(model)
    counts = targets.counts_by_leaf
    # 4개 layer 중 짝수 index 2개가 full attention, 홀수 2개가 linear attention.
    assert counts["q_proj"] == 2
    assert counts["k_proj"] == 2
    assert counts["v_proj"] == 2
    assert counts["o_proj"] == 2
    assert counts["in_proj_qkv"] == 2
    assert counts["in_proj_z"] == 2
    assert counts["out_proj"] == 2
    assert counts["gate_proj"] == 4
    assert counts["up_proj"] == 4
    assert counts["down_proj"] == 4
    assert sum(counts.values()) == len(targets.names)


def test_regex_matches_exactly_the_targets():
    model = FakeQwen35()
    targets = modeling.detect_lora_targets(model)
    pattern = targets.regex()
    matched = [name for name, _ in model.named_modules() if name and re.fullmatch(pattern, name)]
    assert sorted(matched) == sorted(targets.names)
    # vision 쪽 동명 모듈은 하나도 걸리지 않아야 한다.
    for name, _ in model.named_modules():
        if "visual" in name or "merger" in name:
            assert re.fullmatch(pattern, name) is None


def test_vision_filter_without_language_root():
    model = FakeNoLanguageRoot()
    targets = modeling.detect_lora_targets(model)
    assert targets.language_prefix is None
    assert targets.names
    for name in targets.names:
        assert name.startswith("decoder.")
    assert all("vision_tower" in name for name in targets.excluded_vision)


def test_explicit_leaf_list_is_respected():
    model = FakeQwen35()
    targets = modeling.detect_lora_targets(model, explicit_leaves=["q_proj", "v_proj"])
    assert set(targets.counts_by_leaf) == {"q_proj", "v_proj"}


def test_as_dict_has_manifest_fields():
    targets = modeling.detect_lora_targets(FakeQwen35())
    payload = targets.as_dict()
    for key in ("language_prefix", "leaf_names", "counts_by_leaf", "num_modules", "modules"):
        assert key in payload
    assert payload["num_modules"] == len(targets.names)


def test_count_parameters_splits_trainable():
    model = FakeQwen35(n_language=1, n_vision=1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.language_model.parameters():
        param.requires_grad = True
    counts = modeling.count_parameters(model)
    assert 0 < counts["trainable"] < counts["total"]


def test_build_lora_model_wraps_only_language(tmp_path):
    """PEFT를 실제로 붙여 vision 쪽에 adapter가 생기지 않는지 확인한다."""
    peft = pytest.importorskip("peft")
    model = FakeQwen35(n_language=2, n_vision=2)
    cfg = {"lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": "auto"}}
    wrapped, targets = modeling.build_lora_model(model, cfg)
    assert modeling.verify_no_vision_adapters(wrapped) == []

    adapted = {
        name.split(".lora_A")[0]
        for name, _ in wrapped.named_modules()
        if name.endswith(".lora_A")
    }
    assert adapted, "LoRA가 하나도 붙지 않았습니다."
    for name in adapted:
        assert "language_model" in name
        assert "visual" not in name

    counts = modeling.count_parameters(wrapped)
    assert 0 < counts["trainable"] < counts["total"]
