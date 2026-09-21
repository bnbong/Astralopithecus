"""데이터 레코드, 선택지 순열, answer-only collator.

학습 loss는 assistant turn의 정답 letter 토큰 **한 개**에만 걸린다. processor가
이미지 토큰을 확장한 뒤의 실제 위치를 찾아야 하므로 뒤에서부터 탐색하고, 표본마다
정확히 한 개만 감독되는지와 그 토큰이 정답 letter로 디코딩되는지를 확인한다.

학습 텍스트는 `추론 프롬프트 + 정답 letter + end-of-turn 꼬리`로 만든다. chat
template에 assistant turn을 통째로 넣어 렌더링하면 템플릿 버전에 따라 추론 쪽
prefix와 한 글자씩 어긋날 수 있어서다. 구성만으로 같다고 믿지 않고, 첫 배치의 첫
표본에 대해서는 감독 위치 앞의 토큰 열이 추론 쪽 input_ids와 같은지 실제로 비교한다.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import prompt as prompt_mod
from .prompt import LETTERS, ChoiceTokens, PromptSpec

#: 4개 선택지의 모든 순열. perm[j] = 화면 j번째 자리에 놓인 원래 선택지 index.
ALL_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))

#: TTA=4에서 쓰는 고정 순열. 계약에 적힌 (0123),(1302),(2031),(3210).
PERM_BANK: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
    (3, 2, 1, 0),
)

#: 이 문구가 들어간 선택지가 있으면 순서에 의존하므로 순열을 적용하지 않는다.
ORDER_DEPENDENT_MARKERS: tuple[str, ...] = (
    "위의",
    "위 항목",
    "위에서",
    "모두",
    "둘 다",
    "둘다",
    "a와",
    "b와",
    "c와",
    "d와",
    "all of the above",
    "none of the above",
    "both a",
    "both b",
    "정답 없음",
    "답 없음",
    "해당 없음",
    "위 보기",
)


class LabelMaskError(RuntimeError):
    """정답 토큰 위치를 확정할 수 없을 때."""


# --------------------------------------------------------------------------
# 레코드
# --------------------------------------------------------------------------


@dataclass
class VQARecord:
    """한 문항. answer_idx는 정답이 없는 test에서는 None."""

    id: str
    path: str
    question: str
    options: list[str]
    answer_idx: int | None = None
    ocr: str | None = None

    @property
    def answer_letter(self) -> str | None:
        return None if self.answer_idx is None else LETTERS[self.answer_idx]

    def approx_length(self) -> int:
        """배치 정렬에 쓰는 대략적인 텍스트 길이."""
        return len(self.question) + sum(len(o) for o in self.options) + len(self.ocr or "")


def load_records(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
    ocr_map: dict[str, str] | None = None,
) -> list[VQARecord]:
    """CSV를 VQARecord 목록으로 읽는다.

    vqa.data.load_csv가 있으면 그것을 쓰고, 없으면 같은 규칙의 내부 fallback을 쓴다.
    """
    frame = _read_frame(csv_path)
    records: list[VQARecord] = []
    has_answer = "answer" in frame.columns
    for row in frame.to_dict("records"):
        answer_idx = None
        if has_answer:
            raw = str(row.get("answer", "")).strip().lower()
            if raw in LETTERS:
                answer_idx = LETTERS.index(raw)
        rid = str(row["id"])
        records.append(
            VQARecord(
                id=rid,
                path=str(row["path"]),
                question=str(row["question"]),
                options=[str(row[letter]) for letter in LETTERS],
                answer_idx=answer_idx,
                ocr=(ocr_map or {}).get(rid),
            )
        )
    return records


def _read_frame(csv_path: str | Path):
    try:
        from . import data as data_mod  # 다른 워커가 소유한 모듈

        loader = getattr(data_mod, "load_csv", None)
        if loader is not None:
            return loader(csv_path)
    except Exception:
        pass
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"id", "path", "question", *LETTERS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path}: 필수 컬럼이 없습니다: {sorted(missing)}")
    return frame.drop_duplicates("id").reset_index(drop=True)


def resolve_image_path(path: str, data_root: str | Path) -> Path:
    """CSV의 상대 path를 DATA_ROOT 기준 절대 경로로 만든다."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(data_root)
    direct = root / candidate
    if direct.exists():
        return direct
    # 하위 디렉터리 없이 파일 이름만 있는 경우를 보조로 찾는다.
    for sub in ("train", "test", "dev"):
        alt = root / sub / candidate.name
        if alt.exists():
            return alt
    return direct


def load_image(path: str, data_root: str | Path):
    from PIL import Image

    resolved = resolve_image_path(path, data_root)
    with Image.open(resolved) as handle:
        return handle.convert("RGB")


# --------------------------------------------------------------------------
# 선택지 순열
# --------------------------------------------------------------------------


def is_order_dependent(options: Sequence[str]) -> bool:
    """'위의 모두' 같은 순서 의존 보기가 있으면 True."""
    for option in options:
        lowered = str(option).strip().lower()
        if any(marker in lowered for marker in ORDER_DEPENDENT_MARKERS):
            return True
    return False


def option_permutation(seed: int, epoch: int, sample_id: str) -> tuple[int, ...]:
    """(seed, epoch, sample id)만으로 정해지는 순열. 재개해도 같은 값이 나온다."""
    payload = f"{int(seed)}|{int(epoch)}|{sample_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(ALL_PERMUTATIONS)
    return ALL_PERMUTATIONS[index]


def apply_permutation(
    options: Sequence[str],
    answer_idx: int | None,
    perm: Sequence[int],
) -> tuple[list[str], int | None]:
    """perm[j] = 화면 j자리에 놓을 원래 index. 정답 index도 같이 옮긴다."""
    shown = [options[j] for j in perm]
    if answer_idx is None:
        return shown, None
    return shown, list(perm).index(answer_idx)


def permuted_view(
    record: VQARecord,
    *,
    seed: int,
    epoch: int,
    shuffle_options: bool,
) -> tuple[list[str], int | None, tuple[int, ...]]:
    """학습용 선택지 뷰. 순서 의존 보기는 원래 순서를 유지한다."""
    identity = tuple(range(4))
    if not shuffle_options or is_order_dependent(record.options):
        return list(record.options), record.answer_idx, identity
    perm = option_permutation(seed, epoch, record.id)
    shown, gold = apply_permutation(record.options, record.answer_idx, perm)
    return shown, gold, perm


def invert_probs(shown_probs, perm: Sequence[int]):
    """화면 순서 확률을 원래 a~d 순서로 되돌린다.

    shown_probs[:, j]는 화면 j자리 = 원래 perm[j]번 선택지의 확률이다.
    """
    import numpy as np

    shown = np.asarray(shown_probs)
    original = np.zeros_like(shown)
    for shown_j, orig_j in enumerate(perm):
        original[..., orig_j] = shown[..., shown_j]
    return original


# --------------------------------------------------------------------------
# 텐서 도우미
# --------------------------------------------------------------------------


def last_nonpad_index(attention_mask):
    """오른쪽 padding에서 마지막 실제 토큰의 index.

    뒤집어서 첫 1을 찾는 방식이라 중간에 0이 섞여 있어도 마지막 1을 정확히 잡는다.
    """
    import torch

    mask = attention_mask
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    flipped = torch.flip(mask, dims=[1])
    offset = flipped.float().argmax(dim=1)
    index = mask.shape[1] - 1 - offset
    return index.squeeze(0) if squeeze else index


# --------------------------------------------------------------------------
# Dataset / Collator
# --------------------------------------------------------------------------


@dataclass
class SampleView:
    """collator에 넘기는 한 표본."""

    id: str
    image: Any
    question: str
    options: list[str]
    gold_letter: str | None = None
    perm: tuple[int, ...] = (0, 1, 2, 3)
    ocr: str | None = None


def build_sample_view(
    record: VQARecord,
    *,
    data_root: str | Path,
    seed: int,
    epoch: int,
    shuffle_options: bool,
    with_gold: bool,
    perm: Sequence[int] | None = None,
) -> SampleView:
    """레코드에서 이미지까지 읽어 SampleView를 만든다."""
    if perm is not None:
        shown, gold_idx = apply_permutation(record.options, record.answer_idx, perm)
        used_perm = tuple(perm)
    else:
        shown, gold_idx, used_perm = permuted_view(
            record, seed=seed, epoch=epoch, shuffle_options=shuffle_options
        )
    gold_letter = None
    if with_gold:
        if gold_idx is None:
            raise ValueError(f"{record.id}: 정답이 없는 행을 학습에 쓸 수 없습니다.")
        gold_letter = LETTERS[gold_idx]
    return SampleView(
        id=record.id,
        image=load_image(record.path, data_root),
        question=record.question,
        options=shown,
        gold_letter=gold_letter,
        perm=used_perm,
        ocr=record.ocr,
    )


def inference_text(processor: Any, spec: PromptSpec, view: "SampleView") -> str:
    """이 표본에 대해 추론이 쓰는 프롬프트 문자열."""
    return prompt_mod.generation_text(
        processor, spec, view.image, view.question, view.options, view.ocr
    )


def training_text(processor: Any, spec: PromptSpec, view: "SampleView") -> str:
    """이 표본에 대한 학습 텍스트. 추론 프롬프트 뒤에 정답 letter와 꼬리를 붙인다."""
    if view.gold_letter is None:
        raise LabelMaskError(f"{view.id}: gold letter가 없습니다.")
    return prompt_mod.training_text(
        processor, spec, view.image, view.question, view.options, view.gold_letter, view.ocr
    )


def inference_input_ids(processor: Any, spec: PromptSpec, view: "SampleView") -> list[int]:
    """추론 경로와 똑같이 인코딩한 input_ids. prefix 비교의 기준값이다."""
    enc = processor(
        text=[inference_text(processor, spec, view)],
        images=[view.image],
        padding=False,
        return_tensors="pt",
    )
    return [int(i) for i in enc["input_ids"][0]]


def compare_prefix_with_inference(
    processor: Any,
    spec: PromptSpec,
    view: "SampleView",
    train_input_ids,
    position: int,
) -> dict[str, Any]:
    """감독 위치 앞의 토큰 열이 추론 쪽 input_ids와 같은지 본다."""
    train_prefix = [int(i) for i in train_input_ids[: int(position)]]
    infer_ids = inference_input_ids(processor, spec, view)
    match = train_prefix == infer_ids
    first_diff = None
    if not match:
        for index in range(min(len(train_prefix), len(infer_ids))):
            if train_prefix[index] != infer_ids[index]:
                first_diff = index
                break
        if first_diff is None:
            first_diff = min(len(train_prefix), len(infer_ids))
    return {
        "sample": view.id,
        "match": match,
        "train_prefix_len": len(train_prefix),
        "infer_len": len(infer_ids),
        "first_diff_index": first_diff,
        "train_around_diff": train_prefix[max(0, (first_diff or 0) - 4) : (first_diff or 0) + 4],
        "infer_around_diff": infer_ids[max(0, (first_diff or 0) - 4) : (first_diff or 0) + 4],
    }


@dataclass
class AnswerOnlyCollator:
    """정답 letter 토큰 한 개에만 loss를 거는 collator."""

    processor: Any
    spec: PromptSpec
    choice_tokens: ChoiceTokens | None = None
    #: 정답 토큰 뒤에 몇 개의 토큰까지 더 허용할지. None이면 end-of-turn 꼬리 길이에서 정한다.
    max_tail_tokens: int | None = None
    #: 첫 배치의 첫 표본에서 추론 prefix와 일치하는지 실제로 확인할지.
    verify_prefix: bool = True

    def __post_init__(self) -> None:
        self._tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._candidate_ids = self._build_candidate_ids()
        self._prefix_verified = False
        self._suffix = prompt_mod.end_of_turn_suffix(self.processor)
        self._suffix_len = (
            len(self._tokenizer.encode(self._suffix, add_special_tokens=False)) if self._suffix else 0
        )
        if self.max_tail_tokens is None:
            # 학습 텍스트의 정답 토큰 뒤에는 꼬리 토큰만 남는다. 여유를 조금 둔다.
            self.max_tail_tokens = max(8, self._suffix_len + 2)

    def _build_candidate_ids(self) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for letter in LETTERS:
            ids: set[int] = set()
            for variant in (letter, " " + letter):
                encoded = self._tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
            if self.choice_tokens is not None:
                idx = LETTERS.index(letter)
                ids.add(int(self.choice_tokens.ids[idx]))
                for seq in (self.choice_tokens.tail_ids.get(letter) or []):
                    ids.add(int(seq))
            if not ids:
                raise LabelMaskError(f"{letter!r}의 단일 토큰 id를 찾지 못했습니다.")
            out[letter] = ids
        return out

    def __call__(self, batch: Sequence[SampleView]):
        import torch

        texts = []
        images = []
        golds = []
        for view in batch:
            if view.gold_letter is None:
                raise LabelMaskError(f"{view.id}: gold letter가 없습니다.")
            texts.append(training_text(self.processor, self.spec, view))
            images.append(view.image)
            golds.append(view.gold_letter)

        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = torch.full_like(input_ids, -100)

        for b, (gold, view) in enumerate(zip(golds, batch)):
            position, token_id = find_answer_position(
                input_ids[b],
                attention_mask[b],
                self._candidate_ids[gold],
            )
            labels[b, position] = token_id
            self._assert_position(b, view, position, token_id, gold, input_ids, attention_mask)

        supervised = (labels != -100).sum(dim=1)
        if not bool((supervised == 1).all()):
            raise LabelMaskError(f"표본당 감독 토큰이 1개가 아닙니다: {supervised.tolist()}")

        if self.verify_prefix and not self._prefix_verified:
            # 배치마다가 아니라 처음 한 번만 본다. 추론 쪽 인코딩을 한 번 더 돌리기 때문이다.
            self._prefix_verified = True
            first_position = int(torch.where(labels[0] != -100)[0][0])
            report = compare_prefix_with_inference(
                self.processor, self.spec, batch[0], input_ids[0], first_position
            )
            if not report["match"]:
                raise LabelMaskError(
                    "학습 prefix가 추론 prefix와 다릅니다. 같은 프롬프트로 학습/추론하고 있지 "
                    f"않습니다: {report}"
                )

        enc["labels"] = labels
        return enc

    def _assert_position(self, b, view, position, token_id, gold, input_ids, attention_mask) -> None:
        decoded = self._tokenizer.decode([int(token_id)]).strip()
        if decoded.lower() != gold:
            raise LabelMaskError(
                f"{view.id}: 감독 토큰이 정답 letter로 디코딩되지 않습니다. "
                f"pos={int(position)} id={int(token_id)} decoded={decoded!r} gold={gold!r}"
            )
        if self.choice_tokens is not None:
            # scoring에 쓰는 id와 감독하는 id가 다르면 학습과 추론이 다른 토큰을 본다.
            expected = int(self.choice_tokens.ids[LETTERS.index(gold)])
            if int(token_id) != expected:
                raise LabelMaskError(
                    f"{view.id}: 감독 토큰 id가 scoring token id와 다릅니다. "
                    f"supervised={int(token_id)} scoring={expected} gold={gold!r} "
                    f"source={self.choice_tokens.source}"
                )
        last = int(last_nonpad_index(attention_mask[b].unsqueeze(0))[0])
        if last - int(position) > int(self.max_tail_tokens):
            raise LabelMaskError(
                f"{view.id}: 정답 토큰이 assistant turn 끝에서 너무 멉니다. "
                f"pos={int(position)} last_valid={last} max_tail_tokens={self.max_tail_tokens}"
            )


def find_answer_position(input_ids, attention_mask, candidate_ids: Iterable[int]) -> tuple[int, int]:
    """유효 구간에서 뒤에서부터 정답 후보 토큰을 찾는다.

    processor가 이미지 토큰을 확장한 뒤의 실제 sequence에서 찾으므로, 프롬프트 안에
    같은 letter가 여러 번 나와도 마지막(=assistant turn)의 것을 고른다.
    """
    import torch

    candidates = torch.tensor(sorted(int(i) for i in candidate_ids), dtype=input_ids.dtype)
    valid = attention_mask.bool()
    match = torch.isin(input_ids, candidates.to(input_ids.device)) & valid
    positions = torch.where(match)[0]
    if positions.numel() == 0:
        raise LabelMaskError("정답 letter 토큰을 sequence에서 찾지 못했습니다.")
    position = int(positions[-1])
    return position, int(input_ids[position])
