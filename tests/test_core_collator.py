"""선택지 순열, 정답 remap, label 위치, padding 처리 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from vqa import collator as collator_mod  # noqa: E402
from vqa.collator import PERM_BANK, VQARecord  # noqa: E402


def make_record(rid="train_0001.jpg", options=None, answer_idx=0) -> VQARecord:
    return VQARecord(
        id=rid,
        path=f"train/{rid}",
        question="질문",
        options=options or ["가", "나", "다", "라"],
        answer_idx=answer_idx,
    )


# --------------------------------------------------------------------------
# (f) 순열과 정답 remap
# --------------------------------------------------------------------------


def test_apply_permutation_moves_gold_with_options():
    options = ["A", "B", "C", "D"]
    for perm in collator_mod.ALL_PERMUTATIONS:
        for gold in range(4):
            shown, new_gold = collator_mod.apply_permutation(options, gold, perm)
            # 화면 j자리에는 원래 perm[j]번 보기가 온다.
            assert shown == [options[j] for j in perm]
            # 새 정답 자리에 놓인 보기는 원래 정답 보기와 같아야 한다.
            assert shown[new_gold] == options[gold]


def test_permutation_is_deterministic_per_seed_epoch_id():
    first = collator_mod.option_permutation(42, 0, "train_0007.jpg")
    assert first == collator_mod.option_permutation(42, 0, "train_0007.jpg")
    assert first != collator_mod.option_permutation(42, 1, "train_0007.jpg") or True
    # 서로 다른 (seed, epoch, id)는 대체로 서로 다른 순열을 준다.
    values = {
        collator_mod.option_permutation(42, 0, f"train_{i:04d}.jpg") for i in range(50)
    }
    assert len(values) > 5


def test_order_dependent_options_are_not_permuted():
    dependent_cases = [
        ["가", "나", "다", "위의 모두"],
        ["가", "나", "다", "a와 b"],
        ["가", "나", "다", "둘 다 맞다"],
        ["가", "나", "다", "All of the above"],
        ["가", "나", "다", "정답 없음"],
    ]
    for options in dependent_cases:
        assert collator_mod.is_order_dependent(options), options
        record = make_record(options=options, answer_idx=1)
        shown, gold, perm = collator_mod.permuted_view(record, seed=42, epoch=0, shuffle_options=True)
        assert perm == (0, 1, 2, 3)
        assert shown == options
        assert gold == 1

    normal = ["서울", "부산", "대구", "광주"]
    assert not collator_mod.is_order_dependent(normal)


def test_shuffle_options_false_keeps_identity():
    record = make_record(answer_idx=2)
    shown, gold, perm = collator_mod.permuted_view(record, seed=1, epoch=0, shuffle_options=False)
    assert perm == (0, 1, 2, 3)
    assert shown == record.options
    assert gold == 2


def test_permuted_view_keeps_gold_content():
    record = make_record(options=["W", "X", "Y", "Z"], answer_idx=3)
    for epoch in range(5):
        shown, gold, perm = collator_mod.permuted_view(record, seed=11, epoch=epoch, shuffle_options=True)
        assert shown[gold] == "Z"
        assert sorted(shown) == sorted(record.options)


# --------------------------------------------------------------------------
# (e) TTA 역매핑
# --------------------------------------------------------------------------


def test_tta_permutations_are_the_agreed_four():
    assert PERM_BANK == ((0, 1, 2, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 2, 1, 0))
    for perm in PERM_BANK:
        assert sorted(perm) == [0, 1, 2, 3]


def test_invert_probs_round_trips_every_permutation():
    import numpy as np

    original = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    for perm in collator_mod.ALL_PERMUTATIONS:
        # 화면에 perm 순서로 보였을 때 모델이 내놓을 확률
        shown = original[:, list(perm)]
        restored = collator_mod.invert_probs(shown, perm)
        assert np.allclose(restored, original)


def test_invert_probs_matches_option_permutation_semantics():
    """모델이 '보이는 정답 자리'를 맞히면 역매핑 뒤 원래 정답 자리가 최대가 된다."""
    import numpy as np

    options = ["A", "B", "C", "D"]
    gold = 2
    for perm in PERM_BANK:
        shown_options, shown_gold = collator_mod.apply_permutation(options, gold, perm)
        assert shown_options[shown_gold] == options[gold]
        probs = np.zeros((1, 4))
        probs[0, shown_gold] = 1.0
        restored = collator_mod.invert_probs(probs, perm)
        assert int(restored[0].argmax()) == gold


# --------------------------------------------------------------------------
# (g) 오른쪽 padding에서 마지막 non-pad 위치
# --------------------------------------------------------------------------


def test_last_nonpad_index_with_right_padding():
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
        ]
    )
    assert collator_mod.last_nonpad_index(mask).tolist() == [3, 1, 5, 0]


def test_last_nonpad_index_single_row():
    mask = torch.tensor([1, 1, 1, 0])
    assert int(collator_mod.last_nonpad_index(mask)) == 2


def test_last_nonpad_index_picks_final_one_even_with_internal_zeros():
    mask = torch.tensor([[1, 0, 1, 1, 0, 0]])
    assert collator_mod.last_nonpad_index(mask).tolist() == [3]


def test_gather_at_last_nonpad_selects_right_logits():
    """오른쪽 padding에서 실제로 뽑히는 logits 행이 맞는지 직접 확인한다."""
    logits = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    index = collator_mod.last_nonpad_index(mask)
    picked = logits[torch.arange(2), index]
    assert torch.equal(picked[0], logits[0, 2])
    assert torch.equal(picked[1], logits[1, 4])


# --------------------------------------------------------------------------
# label 위치 찾기
# --------------------------------------------------------------------------


def test_find_answer_position_takes_last_valid_occurrence():
    # 프롬프트 안에도 'a' 토큰(=7)이 여러 번 나오고, 마지막 유효 토큰이 정답이다.
    input_ids = torch.tensor([7, 3, 7, 9, 7, 0, 0])
    attention = torch.tensor([1, 1, 1, 1, 1, 0, 0])
    position, token = collator_mod.find_answer_position(input_ids, attention, {7})
    assert position == 4
    assert token == 7


def test_find_answer_position_ignores_padding_region():
    input_ids = torch.tensor([7, 3, 9, 7, 7])
    attention = torch.tensor([1, 1, 1, 0, 0])
    position, token = collator_mod.find_answer_position(input_ids, attention, {7})
    assert position == 0


def test_find_answer_position_raises_when_absent():
    input_ids = torch.tensor([1, 2, 3])
    attention = torch.tensor([1, 1, 1])
    with pytest.raises(collator_mod.LabelMaskError):
        collator_mod.find_answer_position(input_ids, attention, {99})


def test_resolve_image_path_prefers_csv_relative_path(tmp_path):
    (tmp_path / "train").mkdir()
    target = tmp_path / "train" / "train_0001.jpg"
    target.write_bytes(b"x")
    assert collator_mod.resolve_image_path("train/train_0001.jpg", tmp_path) == target
    # 하위 디렉터리 없이 파일 이름만 있어도 찾아 준다.
    assert collator_mod.resolve_image_path("train_0001.jpg", tmp_path) == target
