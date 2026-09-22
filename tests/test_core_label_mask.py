"""실제 Qwen2.5-VL processor로 label 마스킹 위치를 확인한다.

모델 가중치는 받지 않는다. tokenizer/processor 설정 파일만 있으면 되므로 몇 MB로 끝난다.
받을 수 없는 환경(오프라인 등)에서는 건너뛴다. 이 검사는 GPU 환경에서
`python -m vqa.preflight`의 label_mask 항목이 같은 내용을 다시 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL")

from vqa import collator as collator_mod  # noqa: E402
from vqa import prompt as prompt_mod  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    except Exception as exc:  # 네트워크 없음, 캐시 없음 등
        pytest.skip(f"{MODEL_ID} processor를 받을 수 없습니다: {exc}")
    proc.tokenizer.padding_side = "right"
    return proc


@pytest.fixture(scope="module")
def image():
    from PIL import Image

    return Image.new("RGB", (224, 168), "white")


def make_views(image, spec):
    """길이가 다른 두 표본. 오른쪽 padding이 실제로 생기게 한다."""
    return [
        collator_mod.SampleView(
            id="train_0001.jpg",
            image=image,
            question="사진 속 선반에 적힌 문구는 무엇인가요?",
            options=["SUNDAY BURGER LOVE", "MONDAY SALAD TIME", "FRIDAY PIZZA PARTY", "WEDNESDAY SOUP DAY"],
            gold_letter="a",
        ),
        collator_mod.SampleView(
            id="train_0002.jpg",
            image=image,
            question="이 " + "아주 긴 질문 " * 20 + "의 답은 무엇인가요?",
            options=["가" * 40, "나" * 40, "다" * 40, "라" * 40],
            gold_letter="c",
        ),
    ]


def test_choice_letters_are_single_tokens(processor):
    result = prompt_mod.check_single_tokens(processor)
    assert result["ok"], result
    ids = [result["letters"][letter]["bare"][0] for letter in "abcd"]
    assert len(set(ids)) == 4


def test_generation_prompt_tail_has_no_open_thinking(processor, image):
    spec = prompt_mod.get_prompt("v1_image_only")
    text = prompt_mod.full_generation_prompt(processor, spec, image)
    assert text.rstrip().endswith("assistant") or "assistant" in text[-40:]
    assert prompt_mod.thinking_disabled(text[-220:])


def test_choice_tokens_after_template_tail(processor, image):
    spec = prompt_mod.get_prompt("v1_image_only")
    text = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, text)
    assert tokens.source == "tail"
    assert len(tokens.ids) == 4 and len(set(tokens.ids)) == 4
    tokenizer = processor.tokenizer
    for letter, token_id in zip("abcd", tokens.ids):
        assert tokenizer.decode([token_id]).strip() == letter


def test_end_of_turn_suffix_is_derived_from_the_template(processor):
    """꼬리를 하드코딩하지 않고 템플릿에서 뽑아낸다. Qwen 계열이면 <|im_end|>\\n이다."""
    suffix = prompt_mod.end_of_turn_suffix(processor)
    assert suffix, "end-of-turn 꼬리를 얻지 못했습니다."
    assert suffix.startswith("<|im_end|>")
    # 꼬리만으로는 어떤 letter 토큰도 생기지 않아야 한다.
    assert len(processor.tokenizer.encode(suffix, add_special_tokens=False)) <= 4


def test_training_text_is_generation_prompt_plus_letter_plus_suffix(processor, image):
    """학습 텍스트의 앞부분은 추론 프롬프트와 문자열 단위로 같아야 한다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    question = "사진 속 선반에 적힌 문구는 무엇인가요?"
    options = ["가", "나", "다", "라"]
    gen = prompt_mod.generation_text(processor, spec, image, question, options)
    train = prompt_mod.training_text(processor, spec, image, question, options, "c")
    assert train.startswith(gen)
    assert train == gen + "c" + prompt_mod.end_of_turn_suffix(processor)


def test_train_prefix_tokens_equal_inference_tokens(processor, image):
    """감독 위치 앞의 토큰 열이 추론 input_ids와 토큰 단위로 같아야 한다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    tail = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, tail)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)

    views = make_views(image, spec)
    enc = collate(views)
    labels = enc["labels"]
    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        report = collator_mod.compare_prefix_with_inference(
            processor, spec, view, enc["input_ids"][b], position
        )
        assert report["match"], report
        assert report["train_prefix_len"] == report["infer_len"]


def test_supervised_token_is_the_scoring_token(processor, image):
    """감독하는 토큰 id는 resolve_choice_tokens가 고른 scoring id와 같아야 한다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    tail = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, tail)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)
    views = make_views(image, spec)
    enc = collate(views)
    labels = enc["labels"]
    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        expected = tokens.ids["abcd".index(view.gold_letter)]
        assert int(labels[b, position]) == expected


def test_collator_rejects_a_prefix_that_does_not_match_inference(processor, image, monkeypatch):
    """학습 prefix와 추론 prefix가 어긋나면 첫 배치에서 바로 걸린다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec)

    original = collator_mod.inference_text

    def drifted(proc, spec_, view):
        # 추론 쪽 프롬프트만 한 글자 바꾼다. 학습 텍스트는 그대로다.
        return original(proc, spec_, view).replace("질문:", "문제:", 1)

    monkeypatch.setattr(collator_mod, "inference_text", drifted)
    with pytest.raises(collator_mod.LabelMaskError, match="추론 prefix"):
        collate(make_views(image, spec))


def test_label_mask_supervises_exactly_one_token_after_image_expansion(processor, image):
    spec = prompt_mod.get_prompt("v1_image_only")
    tail = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, tail)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)

    views = make_views(image, spec)
    enc = collate(views)
    labels = enc["labels"]
    input_ids = enc["input_ids"]
    attention = enc["attention_mask"]

    # 두 표본의 길이가 달라 오른쪽 padding이 생겨야 한다.
    lengths = attention.sum(dim=1).tolist()
    assert lengths[0] != lengths[1]
    assert int(attention[0, -1]) == 0 or int(attention[1, -1]) == 0

    supervised = (labels != -100).sum(dim=1)
    assert supervised.tolist() == [1, 1]

    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        token_id = int(labels[b, position])
        assert processor.tokenizer.decode([token_id]).strip() == view.gold_letter
        assert int(input_ids[b, position]) == token_id

        # 이미지 토큰 확장 뒤의 위치여야 한다: 이미지 placeholder 구간보다 뒤에 있다.
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_positions = torch.where(input_ids[b] == image_token_id)[0]
        assert len(image_positions) > 1, "processor가 이미지 토큰을 확장하지 않았습니다."
        assert position > int(image_positions[-1])

        # assistant turn의 끝 근처여야 한다.
        last_valid = int(collator_mod.last_nonpad_index(attention[b : b + 1])[0])
        assert 0 <= last_valid - position <= 8

        # padding 구간은 감독되지 않는다.
        assert int((labels[b][attention[b] == 0] != -100).sum()) == 0


def test_label_mask_detects_wrong_gold(processor, image):
    """gold와 다른 letter가 감독되면 오류를 낸다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec)
    views = make_views(image, spec)
    views[0].gold_letter = None
    with pytest.raises(collator_mod.LabelMaskError):
        collate(views)


def test_permuted_training_sample_supervises_the_remapped_letter(processor, image):
    """선택지를 섞은 뒤에도 감독 토큰은 새 정답 자리 letter여야 한다."""
    spec = prompt_mod.get_prompt("v1_image_only")
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec)
    record = collator_mod.VQARecord(
        id="train_0003.jpg",
        path="train/train_0003.jpg",
        question="정답은 세 번째입니다",
        options=["W", "X", "Y", "Z"],
        answer_idx=2,
    )
    shown, gold_idx, perm = collator_mod.permuted_view(record, seed=42, epoch=0, shuffle_options=True)
    assert shown[gold_idx] == "Y"
    view = collator_mod.SampleView(
        id=record.id,
        image=image,
        question=record.question,
        options=shown,
        gold_letter="abcd"[gold_idx],
        perm=perm,
    )
    enc = collate([view])
    labels = enc["labels"]
    position = int(torch.where(labels[0] != -100)[0][0])
    assert processor.tokenizer.decode([int(labels[0, position])]).strip() == "abcd"[gold_idx]


def test_ocr_prompt_reaches_the_tokenized_text(processor, image):
    spec = prompt_mod.get_prompt("v1_ocr")
    messages = prompt_mod.build_messages(
        spec, image, "질문", ["가", "나", "다", "라"], gold=None, ocr="SUNDAY BURGER LOVE"
    )
    text = prompt_mod.apply_template(processor, messages, add_generation_prompt=True)
    assert "SUNDAY BURGER LOVE" in text
    assert "틀릴 수 있습니다" in text


# --------------------------------------------------------------------------
# v2_read_text (팀원 스크립트 p2 재현)
# --------------------------------------------------------------------------

#: infer_val.py의 PROMPTS["p2"]를 그대로 옮긴 템플릿. 테스트가 원본과 비교하는 기준이다.
TEAMMATE_P2_TEMPLATE = (
    "이미지를 자세히 보고, 이미지에 적힌 글자를 정확히 읽어서 질문에 답하세요.\n\n"
    "질문: {question}\n\n"
    "a. {a}\nb. {b}\nc. {c}\nd. {d}\n\n"
    "정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요."
)


def test_teammate_template_constant_matches_infer_val_source():
    """infer_val.py를 직접 읽어서 위 상수가 원본과 같은지 확인한다."""
    import ast

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "experiments" / "eval_tools" / "infer_val.py",
        repo_root / "infer_val.py",
        repo_root / "scripts" / "infer_val.py",
    ]
    infer_val_path = next((p for p in candidates if p.exists()), None)
    assert infer_val_path is not None, (
        "infer_val.py를 찾지 못했습니다 (확인한 경로: "
        + ", ".join(str(p) for p in candidates)
        + "). 템플릿 일치 검증을 건너뛸 수 없습니다."
    )
    source = infer_val_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    prompts = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROMPTS" for t in node.targets
        ):
            prompts = ast.literal_eval(node.value)
    assert prompts is not None, "infer_val.py에서 PROMPTS를 찾지 못했습니다."
    assert prompts["p2"] == TEAMMATE_P2_TEMPLATE


def test_v2_read_text_user_text_equals_the_teammate_prompt():
    spec = prompt_mod.get_prompt("v2_read_text")
    question = "사진 속 선반에 적힌 문구는 무엇인가요?"
    options = ["SUNDAY BURGER LOVE", "MONDAY SALAD TIME", "FRIDAY PIZZA PARTY", "WEDNESDAY SOUP DAY"]
    expected = TEAMMATE_P2_TEMPLATE.format(
        question=question, a=options[0], b=options[1], c=options[2], d=options[3]
    )
    assert spec.user_text(question, options) == expected


def test_v2_read_text_has_no_system_turn_like_infer_val():
    """infer_val.py는 user turn 하나만 넣고 이미지를 텍스트 앞에 둔다."""
    spec = prompt_mod.get_prompt("v2_read_text")
    messages = prompt_mod.build_messages(spec, "IMG", "질문", ["가", "나", "다", "라"])
    assert [m["role"] for m in messages] == ["user"]
    assert [c["type"] for c in messages[0]["content"]] == ["image", "text"]


def test_v2_read_text_prefix_and_label_mask_still_hold(processor, image):
    """프롬프트를 바꿔도 감독 토큰은 하나이고 추론 prefix와 일치해야 한다."""
    spec = prompt_mod.get_prompt("v2_read_text")
    tail = prompt_mod.full_generation_prompt(processor, spec, image)
    tokens = prompt_mod.resolve_choice_tokens(processor, tail)
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)

    views = make_views(image, spec)
    enc = collate(views)
    labels = enc["labels"]
    assert (labels != -100).sum(dim=1).tolist() == [1, 1]
    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        assert int(labels[b, position]) == tokens.ids["abcd".index(view.gold_letter)]
        report = collator_mod.compare_prefix_with_inference(
            processor, spec, view, enc["input_ids"][b], position
        )
        assert report["match"], report


def test_v2_read_text_ocr_adds_the_reference_block():
    spec = prompt_mod.get_prompt("v2_read_text_ocr")
    text = spec.user_text("질문", ["가", "나", "다", "라"], "SUNDAY BURGER LOVE")
    assert text.startswith("이미지를 자세히 보고")
    assert "SUNDAY BURGER LOVE" in text
    assert text.endswith("정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요.")
