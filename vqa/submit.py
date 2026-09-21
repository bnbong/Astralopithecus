"""예측 확률에서 제출 CSV를 만들고 제출 파일을 검사하는 모듈.

제출 CSV는 `sample_submission.csv`와 ID 집합·행 수·행 순서를 모두 맞춘다. answer는
a~d 중 하나이며 결측·중복·추가 index 컬럼이 없어야 한다. 파싱 실패를 'a'로 채우지 않고
그 자리에서 실패로 처리한다.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from vqa.predictions import (
    LETTERS,
    PROB_COLUMNS,
    atomic_write_bytes,
    decide_answers,
    sha256_file,
)

SUBMISSION_COLUMNS: tuple[str, ...] = ("id", "answer")

#: 한 letter가 이 비율을 넘으면 분포가 치우쳤다고 경고한다.
EXTREME_SHARE_WARNING = 0.45

_BOM = b"\xef\xbb\xbf"


class SubmissionError(ValueError):
    """제출 파일을 만들거나 검사하는 중 규칙 위반이 발견되면 발생한다."""


@dataclass
class SubmissionCheck:
    """제출 파일 검사 결과."""

    path: Path
    sha256: str
    n_rows: int
    distribution: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_report(self) -> str:
        lines = [
            f"파일        : {self.path}",
            f"sha256      : {self.sha256}",
            f"행 수       : {self.n_rows}",
            "answer 분포 :",
        ]
        total = max(sum(self.distribution.values()), 1)
        for letter in LETTERS:
            count = self.distribution.get(letter, 0)
            lines.append(f"  {letter}: {count:>6}  ({count / total:6.2%})")
        if self.warnings:
            lines.append("")
            lines.append("경고")
            for message in self.warnings:
                lines.append(f"  - {message}")
        lines.append("")
        if self.errors:
            lines.append(f"결과: 실패 ({len(self.errors)}건)")
            for message in self.errors:
                lines.append(f"  - {message}")
        else:
            lines.append("결과: 통과")
        return "\n".join(lines)


def load_sample_submission(path: str | os.PathLike[str]) -> pd.DataFrame:
    """sample_submission.csv를 읽어 기준 ID 순서를 얻는다."""
    target = Path(path)
    if not target.exists():
        raise SubmissionError(f"sample_submission 파일이 없습니다: {target}")
    frame = pd.read_csv(target, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]
    if "id" not in frame.columns:
        raise SubmissionError(
            f"{target}: 'id' 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}"
        )
    frame["id"] = frame["id"].astype(str).str.strip()
    duplicated = frame["id"][frame["id"].duplicated()].unique().tolist()
    if duplicated:
        raise SubmissionError(
            f"{target}: sample_submission의 id가 중복됩니다: {sorted(duplicated)[:5]}"
        )
    return frame


def build_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """sample_submission의 ID 순서 그대로 제출 DataFrame을 만든다.

    예측에 없는 ID가 하나라도 있으면 기본값으로 채우지 않고 예외를 낸다.

    Raises:
        SubmissionError: 예측이 sample_submission의 ID를 모두 덮지 못할 때.
    """
    decided = decide_answers(predictions)
    lookup = decided.set_index("id")["answer"]
    if lookup.index.has_duplicates:
        duplicated = lookup.index[lookup.index.duplicated()].unique().tolist()
        raise SubmissionError(f"예측 id가 중복됩니다: {sorted(duplicated)[:5]}")

    sample_ids = sample_submission["id"].tolist()
    missing = [sid for sid in sample_ids if sid not in lookup.index]
    if missing:
        raise SubmissionError(
            f"예측에 없는 sample_submission id가 {len(missing)}개 있습니다"
            f"(예: {missing[:5]}). 기본값으로 채우지 않고 중단합니다."
        )
    extra = [pid for pid in lookup.index if pid not in set(sample_ids)]

    answers = [lookup.loc[sid] for sid in sample_ids]
    bad = sorted({str(ans) for ans in answers} - set(LETTERS))
    if bad:
        raise SubmissionError(f"a~d가 아닌 answer가 만들어졌습니다: {bad[:5]}")

    frame = pd.DataFrame({"id": sample_ids, "answer": [str(a) for a in answers]})
    if extra:
        # 제출에는 쓰이지 않지만 인계 기록을 위해 남겨둔다.
        frame.attrs["extra_prediction_ids"] = extra
    return frame


def write_submission(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    line_terminator: str = "\n",
) -> Path:
    """제출 CSV를 BOM 없이, index 컬럼 없이 원자적으로 저장한다."""
    missing = [col for col in SUBMISSION_COLUMNS if col not in frame.columns]
    if missing:
        raise SubmissionError(f"제출 DataFrame에 컬럼이 없습니다: {missing}")
    text = frame.loc[:, list(SUBMISSION_COLUMNS)].to_csv(
        index=False, lineterminator=line_terminator
    )
    return atomic_write_bytes(path, text.encode("utf-8"))


def answer_distribution(frame: pd.DataFrame) -> dict[str, int]:
    """answer 컬럼의 letter별 개수를 센다."""
    counts = Counter(str(value) for value in frame["answer"].tolist())
    return {letter: counts.get(letter, 0) for letter in sorted(counts)}


def check_submission(
    path: str | os.PathLike[str],
    sample_submission_path: str | os.PathLike[str],
    *,
    extreme_share: float = EXTREME_SHARE_WARNING,
) -> SubmissionCheck:
    """제출 CSV가 대회 형식을 지키는지 검사한다.

    ID 집합, 행 수, 행 순서, answer 값, 결측·중복·추가 index 컬럼, BOM을 확인하고
    sha256과 answer 분포를 결과에 담는다. 분포가 한쪽으로 치우치면 경고를 남긴다.
    """
    target = Path(path)
    if not target.exists():
        raise SubmissionError(f"제출 파일이 없습니다: {target}")

    errors: list[str] = []
    warnings: list[str] = []

    raw = target.read_bytes()
    if raw.startswith(_BOM):
        errors.append(
            "파일이 UTF-8 BOM으로 시작합니다. BOM 없이 다시 저장하세요."
        )

    frame = pd.read_csv(target, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]

    unnamed = [col for col in frame.columns if col.startswith("Unnamed:")]
    if unnamed:
        errors.append(f"이름 없는 추가 index 컬럼이 있습니다: {unnamed}")
    missing_cols = [col for col in SUBMISSION_COLUMNS if col not in frame.columns]
    if missing_cols:
        errors.append(f"필수 컬럼이 없습니다: {missing_cols}")
    extra_cols = [
        col
        for col in frame.columns
        if col not in SUBMISSION_COLUMNS and not col.startswith("Unnamed:")
    ]
    if extra_cols:
        errors.append(f"예상하지 않은 컬럼이 있습니다: {extra_cols}")

    distribution: dict[str, int] = {}
    n_rows = int(len(frame))

    if not missing_cols:
        ids = frame["id"]
        if ids.isna().any():
            errors.append(f"id가 비어 있는 행이 {int(ids.isna().sum())}개 있습니다.")
        ids = ids.fillna("").astype(str).str.strip()

        answers = frame["answer"]
        na_answers = int(answers.isna().sum())
        if na_answers:
            errors.append(f"answer가 비어 있는 행이 {na_answers}개 있습니다.")
        answers = answers.fillna("").astype(str).str.strip()
        blank = int((answers == "").sum()) - na_answers
        if blank > 0:
            errors.append(f"answer가 공백인 행이 {blank}개 있습니다.")
        bad_values = sorted(set(answers) - set(LETTERS) - {""})
        if bad_values:
            errors.append(f"answer가 a~d가 아닌 값이 있습니다: {bad_values[:5]}")

        duplicated = ids[ids.duplicated()].unique().tolist()
        if duplicated:
            errors.append(f"id가 중복됩니다({len(duplicated)}개): {sorted(duplicated)[:5]}")

        sample = load_sample_submission(sample_submission_path)
        sample_ids = sample["id"].tolist()
        if n_rows != len(sample_ids):
            errors.append(
                f"행 수가 다릅니다: 제출 {n_rows}행, sample_submission {len(sample_ids)}행"
            )
        submitted_set = set(ids)
        sample_set = set(sample_ids)
        missing_ids = sorted(sample_set - submitted_set)
        unexpected_ids = sorted(submitted_set - sample_set)
        if missing_ids:
            errors.append(
                f"sample_submission에 있지만 제출에 없는 id {len(missing_ids)}개"
                f"(예: {missing_ids[:5]})"
            )
        if unexpected_ids:
            errors.append(
                f"sample_submission에 없는 id가 제출에 {len(unexpected_ids)}개"
                f"(예: {unexpected_ids[:5]})"
            )
        if not missing_ids and not unexpected_ids and n_rows == len(sample_ids):
            mismatch = [
                position
                for position, (left, right) in enumerate(zip(ids.tolist(), sample_ids))
                if left != right
            ]
            if mismatch:
                first = mismatch[0]
                errors.append(
                    f"행 순서가 sample_submission과 다릅니다. 어긋난 행 {len(mismatch)}개, "
                    f"첫 위치 {first}(제출 '{ids.iloc[first]}', sample '{sample_ids[first]}')"
                )

        distribution = {letter: int((answers == letter).sum()) for letter in LETTERS}
        total = sum(distribution.values())
        if total:
            for letter, count in distribution.items():
                share = count / total
                if share > extreme_share:
                    warnings.append(
                        f"answer '{letter}' 비중이 {share:.1%}로 {extreme_share:.0%}를 넘습니다. "
                        "예측이 한쪽으로 쏠렸는지 확인하세요."
                    )
            for letter, count in distribution.items():
                if count == 0:
                    warnings.append(f"answer '{letter}'가 한 번도 나오지 않았습니다.")

    return SubmissionCheck(
        path=target,
        sha256=sha256_file(target),
        n_rows=n_rows,
        distribution=distribution,
        errors=errors,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# CLI: python -m vqa.submit
# --------------------------------------------------------------------------


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vqa.submit",
        description=(
            "예측 확률 파일에서 sample_submission의 ID 순서 그대로 제출 CSV를 만든다. "
            "예측에 없는 ID가 있으면 기본값으로 채우지 않고 실패한다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "종료 코드\n"
            "  0  제출 파일 생성 완료\n"
            "  2  입력 오류 (파일 없음, 형식 위반, ID 누락)\n\n"
            "사용 예\n"
            "  python -m vqa.submit --pred artifacts/preds/combined_test.csv \\\n"
            "      --sample sample_submission.csv --out artifacts/submission/final.csv\n"
        ),
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        metavar="PATH",
        help="예측 확률 CSV (id,p_a,p_b,p_c,p_d,run_id)",
    )
    parser.add_argument(
        "--sample",
        required=True,
        type=Path,
        metavar="PATH",
        help="sample_submission.csv. ID 집합과 행 순서의 기준이 된다.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PATH",
        help="만들 제출 CSV 경로.",
    )
    parser.add_argument(
        "--margins-out",
        type=Path,
        metavar="PATH",
        help="제출한 답과 margin을 함께 저장할 CSV 경로. 오답 검토용 기록이다.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="확률 합의 허용 오차 (기본: 1e-3)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="저장 후 형식 검사를 생략한다.",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    import sys

    from vqa.predictions import PredictionError, atomic_write_text, load_predictions

    args = build_parser().parse_args(argv)
    try:
        predictions = load_predictions(args.pred, tolerance=args.tolerance)
        sample = load_sample_submission(args.sample)
        frame = build_submission(predictions, sample)
    except (PredictionError, SubmissionError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2

    out = write_submission(frame, args.out)
    print(f"제출 파일 생성  : {out} ({len(frame)}행)")
    extra = frame.attrs.get("extra_prediction_ids")
    if extra:
        print(
            f"  안내          : 예측에만 있고 제출에는 쓰이지 않은 id {len(extra)}개"
            f" (예: {extra[:5]})"
        )

    if args.margins_out is not None:
        decided = decide_answers(predictions).set_index("id")
        detail = pd.DataFrame(
            {
                "id": frame["id"],
                "answer": frame["answer"],
                "margin": [decided.loc[i, "margin"] for i in frame["id"]],
                **{
                    col: [decided.loc[i, col] for i in frame["id"]]
                    for col in PROB_COLUMNS
                },
            }
        )
        atomic_write_text(
            args.margins_out, detail.to_csv(index=False, lineterminator="\n")
        )
        print(f"margin 기록     : {args.margins_out}")

    if not args.no_check:
        check = check_submission(out, args.sample)
        print("")
        print(check.format_report())
        if not check.ok:
            return 2
    return 0


__all__ = [
    "EXTREME_SHARE_WARNING",
    "SUBMISSION_COLUMNS",
    "SubmissionCheck",
    "SubmissionError",
    "answer_distribution",
    "build_submission",
    "check_submission",
    "load_sample_submission",
    "write_submission",
]


if __name__ == "__main__":
    raise SystemExit(main())
