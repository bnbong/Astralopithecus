"""고정 validation ID에서 예측 확률을 정답과 비교하는 평가 모듈.

정확도 외에 오답 분석에 필요한 정보를 함께 낸다: 정답 letter별 정확도, confusion matrix,
예측 letter 분포, 평균 margin, 질문 유형별 분해, 그리고 margin을 붙인 오답 ID 목록.

정식 정확도 비교는 고정 validation ID에서만 한다. test에는 정답이 없으므로 이 모듈로
test를 평가하지 않는다.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from vqa.predictions import (
    LETTERS,
    PROB_COLUMNS,
    PredictionError,
    decide_answers,
    sha256_file,
)

#: 실험 기록 로그의 기본 위치.
DEFAULT_EXPERIMENTS_LOG = Path("artifacts/experiments.jsonl")

#: 질문 유형 heuristic이 쓰는 분류 이름.
QUESTION_TYPES: tuple[str, ...] = (
    "phone_like",
    "has_digits",
    "mostly_english",
    "korean_options",
    "other",
)

GOLD_OPTION_COLUMNS: tuple[str, ...] = ("a", "b", "c", "d")

_PHONE_RE = re.compile(r"\d{2,4}[-\s.]?\d{3,4}[-\s.]?\d{4}")
_DIGIT_RE = re.compile(r"\d")
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_LATIN_RE = re.compile(r"[A-Za-z]")


class EvaluationError(ValueError):
    """평가 입력이 맞지 않을 때 발생한다."""


@dataclass
class EvalResult:
    """한 run의 validation 평가 결과."""

    run_id: str
    n: int
    correct: int
    accuracy: float
    mean_margin: float
    mean_margin_correct: float
    mean_margin_wrong: float
    per_gold_letter: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    predicted_distribution: dict[str, int]
    gold_distribution: dict[str, int]
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)
    wrong: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_summary(self) -> dict[str, Any]:
        """JSON으로 기록하기 좋은 요약 사전을 만든다."""
        return {
            "run_id": self.run_id,
            "n": self.n,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "mean_margin": self.mean_margin,
            "mean_margin_correct": self.mean_margin_correct,
            "mean_margin_wrong": self.mean_margin_wrong,
            "per_gold_letter": self.per_gold_letter,
            "confusion": self.confusion,
            "predicted_distribution": self.predicted_distribution,
            "gold_distribution": self.gold_distribution,
            "by_question_type": self.by_question_type,
        }

    def format_report(self, *, max_wrong: int = 20) -> str:
        """사람이 읽는 평가 보고서를 문자열로 만든다."""
        lines: list[str] = []
        lines.append(f"run_id            : {self.run_id}")
        lines.append(f"평가 문항 수      : {self.n}")
        lines.append(f"정확도            : {self.accuracy:.4f} ({self.correct}/{self.n})")
        lines.append(f"평균 margin       : {self.mean_margin:.4f}")
        lines.append(f"  정답 행 평균    : {self.mean_margin_correct:.4f}")
        lines.append(f"  오답 행 평균    : {self.mean_margin_wrong:.4f}")

        lines.append("")
        lines.append("정답 letter별 정확도")
        lines.append("  letter    n   correct   accuracy")
        for letter in LETTERS:
            stats = self.per_gold_letter.get(letter, {"n": 0, "correct": 0, "accuracy": float("nan")})
            lines.append(
                f"  {letter:>6} {int(stats['n']):>4} {int(stats['correct']):>9}"
                f"   {stats['accuracy']:.4f}"
            )

        lines.append("")
        lines.append("confusion matrix (행=정답, 열=예측)")
        lines.append("        " + "".join(f"{letter:>8}" for letter in LETTERS))
        for gold in LETTERS:
            row = self.confusion.get(gold, {})
            lines.append(
                f"  {gold:>4}  " + "".join(f"{row.get(pred, 0):>8}" for pred in LETTERS)
            )

        lines.append("")
        total_pred = max(sum(self.predicted_distribution.values()), 1)
        lines.append("예측 letter 분포")
        for letter in LETTERS:
            count = self.predicted_distribution.get(letter, 0)
            lines.append(f"  {letter}: {count:>6}  ({count / total_pred:6.2%})")
        lines.append("정답 letter 분포")
        total_gold = max(sum(self.gold_distribution.values()), 1)
        for letter in LETTERS:
            count = self.gold_distribution.get(letter, 0)
            lines.append(f"  {letter}: {count:>6}  ({count / total_gold:6.2%})")

        if self.by_question_type:
            lines.append("")
            lines.append("질문 유형별 분해 (heuristic)")
            lines.append("  type                 n   correct   accuracy   mean_margin")
            for name, stats in self.by_question_type.items():
                lines.append(
                    f"  {name:<16} {int(stats['n']):>5} {int(stats['correct']):>9}"
                    f"   {stats['accuracy']:.4f}      {stats['mean_margin']:.4f}"
                )

        if len(self.wrong) > 0:
            lines.append("")
            shown = min(max_wrong, len(self.wrong))
            lines.append(f"오답 {len(self.wrong)}건 중 margin이 큰 순으로 {shown}건")
            lines.append("  id                         gold  pred   margin   p_pred")
            for _, row in self.wrong.head(shown).iterrows():
                lines.append(
                    f"  {str(row['id']):<26} {row['gold']:>4} {row['answer']:>5}"
                    f"   {row['margin']:.4f}   {row['p_pred']:.4f}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# gold 읽기와 질문 유형 heuristic
# --------------------------------------------------------------------------


def load_gold(path: str | os.PathLike[str]) -> pd.DataFrame:
    """정답이 있는 CSV(train.csv 등)를 읽는다.

    `id`와 `answer`는 필수이고, `question`과 `a`~`d`가 있으면 질문 유형 분해에 쓴다.
    BOM이 있어도 읽는다.
    """
    target = Path(path)
    if not target.exists():
        raise EvaluationError(f"gold 파일이 없습니다: {target}")
    frame = pd.read_csv(target, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]
    for required in ("id", "answer"):
        if required not in frame.columns:
            raise EvaluationError(
                f"{target}: '{required}' 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}"
            )
    frame["id"] = frame["id"].astype(str).str.strip()
    frame["answer"] = frame["answer"].astype(str).str.strip().str.lower()
    bad = sorted(set(frame["answer"]) - set(LETTERS))
    if bad:
        raise EvaluationError(f"{target}: answer가 a~d가 아닌 값이 있습니다: {bad[:5]}")
    duplicated = frame["id"][frame["id"].duplicated()].unique().tolist()
    if duplicated:
        raise EvaluationError(f"{target}: id가 중복됩니다: {sorted(duplicated)[:5]}")
    return frame.reset_index(drop=True)


def classify_question_type(question: str, options: Sequence[str]) -> str:
    """질문과 선택지를 보고 유형을 하나로 분류한다.

    우선순위는 전화번호 형태, 숫자 포함, 선택지가 대부분 영문, 선택지가 한글, 그 외다.
    형태소 분석 없이 문자 구성만 보는 거친 heuristic이며 오답 경향을 훑는 용도다.
    """
    texts = [question or ""] + [opt or "" for opt in options]
    joined = " ".join(texts)
    if any(_PHONE_RE.search(text) for text in texts):
        return "phone_like"
    if _DIGIT_RE.search(joined):
        return "has_digits"

    option_text = " ".join(opt or "" for opt in options)
    latin = len(_LATIN_RE.findall(option_text))
    hangul = len(_HANGUL_RE.findall(option_text))
    if latin > 0 and latin >= 2 * max(hangul, 1) and latin >= 3:
        return "mostly_english"
    if hangul > latin:
        return "korean_options"
    return "other"


def add_question_types(gold: pd.DataFrame) -> pd.Series:
    """gold DataFrame의 각 행에 질문 유형 라벨을 붙인 Series를 만든다."""
    has_options = all(col in gold.columns for col in GOLD_OPTION_COLUMNS)
    question_col = gold["question"] if "question" in gold.columns else pd.Series([""] * len(gold))
    labels: list[str] = []
    for idx in range(len(gold)):
        question = question_col.iloc[idx]
        question = "" if pd.isna(question) else str(question)
        if has_options:
            options = [
                "" if pd.isna(gold[col].iloc[idx]) else str(gold[col].iloc[idx])
                for col in GOLD_OPTION_COLUMNS
            ]
        else:
            options = []
        labels.append(classify_question_type(question, options))
    return pd.Series(labels, index=gold.index, dtype="string")


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------


def evaluate_predictions(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    ids: Iterable[str] | None = None,
    breakdown: bool = True,
    require_full_coverage: bool = True,
) -> EvalResult:
    """예측 확률을 정답과 비교한다.

    Args:
        predictions: `vqa.predictions.load_predictions`가 돌려준 DataFrame.
        gold: `load_gold`가 돌려준 DataFrame.
        ids: 평가할 ID 부분집합. None이면 예측과 gold의 교집합 전체를 쓴다.
        breakdown: True면 질문 유형별 분해를 계산한다.
        require_full_coverage: True면 요청한 ID가 예측이나 gold에 없을 때 오류를 낸다.

    Returns:
        EvalResult.
    """
    pred = decide_answers(predictions)
    pred = pred.set_index("id")
    gold_indexed = gold.set_index("id")

    if ids is None:
        target_ids = [i for i in pred.index if i in gold_indexed.index]
        if not target_ids:
            raise EvaluationError("예측과 gold에 공통 ID가 없습니다.")
    else:
        target_ids = [str(i).strip() for i in ids]
        target_ids = [i for i in target_ids if i]
        missing_pred = [i for i in target_ids if i not in pred.index]
        missing_gold = [i for i in target_ids if i not in gold_indexed.index]
        if require_full_coverage and (missing_pred or missing_gold):
            raise EvaluationError(
                f"평가 ID 중 예측에 없는 것 {len(missing_pred)}개(예: {missing_pred[:5]}), "
                f"gold에 없는 것 {len(missing_gold)}개(예: {missing_gold[:5]})"
            )
        skip = set(missing_pred) | set(missing_gold)
        target_ids = [i for i in target_ids if i not in skip]
        if not target_ids:
            raise EvaluationError("평가할 수 있는 ID가 없습니다.")

    pred_sub = pred.loc[target_ids]
    gold_sub = gold_indexed.loc[target_ids]

    answers = pred_sub["answer"].astype(str).to_numpy()
    golds = gold_sub["answer"].astype(str).to_numpy()
    margins = pred_sub["margin"].to_numpy(dtype=float)
    probs = pred_sub[list(PROB_COLUMNS)].to_numpy(dtype=float)
    letter_index = {letter: i for i, letter in enumerate(LETTERS)}
    p_pred = probs[np.arange(len(answers)), [letter_index[a] for a in answers]]
    p_gold = probs[np.arange(len(golds)), [letter_index[g] for g in golds]]

    hit = answers == golds
    n = int(len(answers))
    correct = int(hit.sum())

    per_gold_letter: dict[str, dict[str, float]] = {}
    for letter in LETTERS:
        mask = golds == letter
        count = int(mask.sum())
        hits = int(hit[mask].sum()) if count else 0
        per_gold_letter[letter] = {
            "n": float(count),
            "correct": float(hits),
            "accuracy": (hits / count) if count else float("nan"),
        }

    confusion = {
        gold_letter: {
            pred_letter: int(((golds == gold_letter) & (answers == pred_letter)).sum())
            for pred_letter in LETTERS
        }
        for gold_letter in LETTERS
    }

    predicted_distribution = {letter: int((answers == letter).sum()) for letter in LETTERS}
    gold_distribution = {letter: int((golds == letter).sum()) for letter in LETTERS}

    by_question_type: dict[str, dict[str, float]] = {}
    if breakdown:
        types = add_question_types(gold_sub.reset_index()).to_numpy()
        for name in QUESTION_TYPES:
            mask = types == name
            count = int(mask.sum())
            if count == 0:
                continue
            hits = int(hit[mask].sum())
            by_question_type[name] = {
                "n": float(count),
                "correct": float(hits),
                "accuracy": hits / count,
                "mean_margin": float(margins[mask].mean()),
            }

    wrong_mask = ~hit
    wrong = pd.DataFrame(
        {
            "id": np.asarray(target_ids)[wrong_mask],
            "gold": golds[wrong_mask],
            "answer": answers[wrong_mask],
            "margin": margins[wrong_mask],
            "p_pred": p_pred[wrong_mask],
            "p_gold": p_gold[wrong_mask],
        }
    ).sort_values("margin", ascending=False, kind="stable").reset_index(drop=True)
    if breakdown and len(wrong) > 0:
        type_map = dict(zip(target_ids, add_question_types(gold_sub.reset_index())))
        wrong["question_type"] = [type_map[i] for i in wrong["id"]]

    run_ids = sorted(set(pred_sub["run_id"].astype(str)))
    run_id = run_ids[0] if len(run_ids) == 1 else "+".join(run_ids)

    return EvalResult(
        run_id=run_id,
        n=n,
        correct=correct,
        accuracy=correct / n,
        mean_margin=float(margins.mean()),
        mean_margin_correct=float(margins[hit].mean()) if correct else float("nan"),
        mean_margin_wrong=float(margins[wrong_mask].mean()) if n - correct else float("nan"),
        per_gold_letter=per_gold_letter,
        confusion=confusion,
        predicted_distribution=predicted_distribution,
        gold_distribution=gold_distribution,
        by_question_type=by_question_type,
        wrong=wrong,
    )


def accuracy_only(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    ids: Iterable[str] | None = None,
) -> float:
    """결합 가중치 비교처럼 정확도만 필요할 때 쓰는 가벼운 경로."""
    return evaluate_predictions(
        predictions, gold, ids=ids, breakdown=False
    ).accuracy


# --------------------------------------------------------------------------
# 실험 기록
# --------------------------------------------------------------------------


def append_experiment_log(
    result: EvalResult,
    *,
    log_path: str | os.PathLike[str] = DEFAULT_EXPERIMENTS_LOG,
    prediction_path: str | os.PathLike[str] | None = None,
    seconds_per_item: float | None = None,
    note: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """평가 결과 한 줄을 experiments.jsonl에 덧붙인다.

    한 줄에 run_id, accuracy, n, seconds_per_item, 평가한 예측 파일의 sha256을 남긴다.
    기존 내용을 다시 쓰지 않고 이어붙이므로 여러 사람이 순서대로 기록해도 안전하다.
    """
    entry: dict[str, Any] = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": result.run_id,
        "accuracy": result.accuracy,
        "n": result.n,
        "correct": result.correct,
        "mean_margin": result.mean_margin,
        "seconds_per_item": seconds_per_item,
        "prediction_file": str(prediction_path) if prediction_path else None,
        "file_sha256": sha256_file(prediction_path) if prediction_path else None,
        "predicted_distribution": result.predicted_distribution,
        "note": note,
    }
    if extra:
        entry.update(extra)
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def read_experiment_log(
    log_path: str | os.PathLike[str] = DEFAULT_EXPERIMENTS_LOG,
) -> list[dict[str, Any]]:
    """experiments.jsonl을 읽는다. 없으면 빈 목록을 반환한다."""
    target = Path(log_path)
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


__all__ = [
    "DEFAULT_EXPERIMENTS_LOG",
    "EvalResult",
    "EvaluationError",
    "QUESTION_TYPES",
    "accuracy_only",
    "add_question_types",
    "append_experiment_log",
    "classify_question_type",
    "evaluate_predictions",
    "load_gold",
    "read_experiment_log",
]
