"""pad를 왼쪽에 붙이는 processor에서도 label mask와 prefix 비교가 맞는지 본다.

InternVL 계열 processor는 tokenizer.padding_side와 무관하게 자기 기본값으로 왼쪽
padding을 한다. 그러면 감독 위치 앞 prefix의 머리에 pad가 섞여 추론 prefix와 어긋난다.
가짜 processor로 그 상황을 만들어, (1) collator가 오른쪽 padding을 명시해 pad가 오른쪽에
붙는지, (2) padding_side를 아예 무시하는 processor에서도 pad 자리를 뺀 prefix 비교가
통과하는지 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

from vqa import collator as collator_mod  # noqa: E402
from vqa import preflight as preflight_mod  # noqa: E402
from vqa import prompt as prompt_mod  # noqa: E402

#: 테스트용 end-of-turn 꼬리. a~d 글자가 들어가지 않아야 감독 위치 탐색이 흔들리지 않는다.
SUFFIX = "<|eot|>"
PAD_ID = 0


class FakeTokenizer:
    """글자 하나를 토큰 하나로 보는 tokenizer. id는 글자의 코드포인트다."""

    def __init__(self) -> None:
        self.padding_side = "right"
        self.eos_token = SUFFIX
        self.pad_token_id = PAD_ID

    def encode(self, text, add_special_tokens=False):  # noqa: D401 - transformers 시그니처
        return [ord(ch) for ch in str(text)]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class FakeProcessor:
    """텍스트를 글자 단위로 인코딩하는 processor.

    accepts_padding_side가 False면 padding_side 인자를 받지 않는 예전 processor를,
    always_left가 True면 padding_side를 받고도 무시하고 왼쪽에 붙이는 processor를 흉내낸다.
    """

    def __init__(self, *, default_side="left", accepts_padding_side=True, always_left=False) -> None:
        self.tokenizer = FakeTokenizer()
        self.default_side = default_side
        self.accepts_padding_side = accepts_padding_side
        self.always_left = always_left
        self.seen_padding_side: list[object] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
            else:
                for chunk in content:
                    parts.append(chunk["text"] if chunk.get("type") == "text" else "<image>")
            parts.append(SUFFIX)
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text=None, images=None, padding=False, return_tensors=None, **kwargs):
        if "padding_side" in kwargs and not self.accepts_padding_side:
            raise TypeError("padding_side를 받지 않는 processor입니다.")
        side = kwargs.pop("padding_side", None)
        self.seen_padding_side.append(side)
        if self.always_left:
            side = "left"
        elif side is None:
            side = self.default_side

        rows = [self.tokenizer.encode(t) for t in list(text)]
        width = max(len(r) for r in rows) if padding else max(len(r) for r in rows)
        ids = []
        masks = []
        for row in rows:
            gap = width - len(row)
            if side == "left":
                ids.append([PAD_ID] * gap + row)
                masks.append([0] * gap + [1] * len(row))
            else:
                ids.append(row + [PAD_ID] * gap)
                masks.append([1] * len(row) + [0] * gap)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


@pytest.fixture
def spec():
    return prompt_mod.get_prompt("v1_image_only")


@pytest.fixture
def image():
    from PIL import Image

    return Image.new("RGB", (8, 8), "white")


def make_views(image):
    """길이가 서로 다른 세 표본. padding이 실제로 생긴다."""
    return [
        collator_mod.SampleView(
            id="train_0001.jpg",
            image=image,
            question="짧은 질문",
            options=["가", "나", "다", "라"],
            gold_letter="a",
        ),
        collator_mod.SampleView(
            id="train_0002.jpg",
            image=image,
            question="조금 더 긴 질문 " * 5,
            options=["가" * 10, "나" * 10, "다" * 10, "라" * 10],
            gold_letter="c",
        ),
        collator_mod.SampleView(
            id="train_0003.jpg",
            image=image,
            question="아주 아주 긴 질문 " * 20,
            options=["가" * 30, "나" * 30, "다" * 30, "라" * 30],
            gold_letter="d",
        ),
    ]


def test_collator_forces_right_padding_on_a_left_padding_processor(spec, image):
    processor = FakeProcessor(default_side="left")
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec)
    views = make_views(image)

    enc = collate(views)

    # processor를 부를 때 오른쪽을 명시했다.
    assert processor.seen_padding_side[0] == "right"
    masks = enc["attention_mask"]
    # 짧은 표본의 pad는 뒤쪽에 있다.
    assert int(masks[0][0]) == 1
    assert int(masks[0][-1]) == 0
    assert int(masks[-1].sum()) == masks.shape[1]
    # 표본마다 감독 토큰은 하나이고, 마지막 실제 토큰에서 꼬리 길이만큼 떨어져 있다.
    labels = enc["labels"]
    assert (labels != -100).sum(dim=1).tolist() == [1, 1, 1]
    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        assert processor.tokenizer.decode([int(labels[b, position])]) == view.gold_letter
        last = int(collator_mod.last_nonpad_index(masks[b : b + 1])[0])
        assert last - position == len(SUFFIX)


def test_collator_prefix_verification_passes_with_a_left_padding_processor(spec, image):
    """verify_prefix가 켜진 기본 경로가 왼쪽 padding processor에서도 통과해야 한다."""
    processor = FakeProcessor(default_side="left")
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, verify_prefix=True)
    collate(make_views(image))  # LabelMaskError가 나지 않아야 한다.


def test_collator_falls_back_when_processor_rejects_padding_side(spec, image):
    """padding_side 인자를 받지 않는 processor에서는 tokenizer 설정을 그대로 쓴다."""
    processor = FakeProcessor(default_side="right", accepts_padding_side=False)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec)

    enc = collate(make_views(image))

    assert int(enc["attention_mask"][0][0]) == 1
    assert (enc["labels"] != -100).sum(dim=1).tolist() == [1, 1, 1]


def test_prefix_comparison_ignores_left_pad_positions(spec, image):
    """padding_side를 무시하고 늘 왼쪽에 붙이는 processor에서도 prefix 비교가 맞아야 한다."""
    processor = FakeProcessor(always_left=True)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, verify_prefix=False)
    views = make_views(image)
    enc = collate(views)

    # 앞쪽에 pad가 실제로 붙었다.
    assert int(enc["attention_mask"][0][0]) == 0

    for b, view in enumerate(views):
        position = int(torch.where(enc["labels"][b] != -100)[0][0])
        naive = collator_mod.compare_prefix_with_inference(
            processor, spec, view, enc["input_ids"][b], position
        )
        masked = collator_mod.compare_prefix_with_inference(
            processor,
            spec,
            view,
            enc["input_ids"][b],
            position,
            attention_mask=enc["attention_mask"][b],
        )
        assert masked["match"], masked
        if int(enc["attention_mask"][b][0]) == 0:
            # mask를 주지 않으면 머리의 pad 때문에 첫 토큰부터 어긋난다.
            assert not naive["match"]
            assert naive["first_diff_index"] == 0


def test_strip_pad_positions_leaves_right_padded_rows_unchanged():
    ids = torch.tensor([7, 8, 9, PAD_ID, PAD_ID])
    mask = torch.tensor([1, 1, 1, 0, 0])
    assert collator_mod.strip_pad_positions(ids, mask) == [7, 8, 9]
    assert collator_mod.strip_pad_positions(ids) == [7, 8, 9, PAD_ID, PAD_ID]
    left_ids = torch.tensor([PAD_ID, PAD_ID, 7, 8, 9])
    left_mask = torch.tensor([0, 0, 1, 1, 1])
    assert collator_mod.strip_pad_positions(left_ids, left_mask) == [7, 8, 9]


def test_preflight_prefix_check_passes_with_a_left_padding_processor(spec, tmp_path):
    """preflight의 배치 경로(여러 표본을 한 번에 인코딩)에서도 통과해야 한다."""
    from PIL import Image

    data_root = tmp_path
    (data_root / "train").mkdir()
    records = []
    for i in range(3):
        name = f"train_{i:04d}.jpg"
        Image.new("RGB", (8 + i, 8), "white").save(data_root / "train" / name)
        records.append(
            collator_mod.VQARecord(
                id=name,
                path=f"train/{name}",
                question="질문 " * (i + 1) * 3,
                options=[f"보기{j}" * (i + 1) for j in range(4)],
                answer_idx=i,
            )
        )

    processor = FakeProcessor(default_side="left")
    result = preflight_mod.check_prefix_match(processor, spec, records, data_root, None, n=3)
    assert result.status == preflight_mod.PASS, result.as_dict()
    assert result.detail["checked"] == 3

    mask_result = preflight_mod.check_label_mask(processor, spec, records, data_root, None, n=3)
    assert mask_result.status == preflight_mod.PASS, mask_result.as_dict()
    assert all(p["distance_from_end"] == len(SUFFIX) for p in mask_result.detail["positions"])
