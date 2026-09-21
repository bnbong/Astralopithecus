#!/usr/bin/env python3
"""제출 CSV가 대회 형식을 지키는지 검사한다.

sample_submission과 ID 집합·행 수·행 순서를 대조하고, answer가 a~d인지, 결측·중복·추가
index 컬럼과 BOM이 없는지 확인한다. 파일 sha256과 answer 분포를 출력하며, 한 letter의
비중이 지나치게 크면 경고한다.

종료 코드
  0  모든 검사 통과
  1  검사 실패
  2  입력 오류 (파일 없음 등)
  4  통과했지만 경고가 있고 --strict-warnings를 주었을 때

사용 예
  python scripts/check_submission.py \
      --submission artifacts/submission/final.csv \
      --sample sample_submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.submit import (  # noqa: E402
    EXTREME_SHARE_WARNING,
    SubmissionError,
    check_submission,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT_ERROR = 2
EXIT_WARNINGS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_submission.py",
        description="제출 CSV의 ID·행수·순서·answer·인코딩을 검사한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submission",
        required=True,
        type=Path,
        metavar="PATH",
        help="검사할 제출 CSV.",
    )
    parser.add_argument(
        "--sample",
        required=True,
        type=Path,
        metavar="PATH",
        help="기준이 되는 sample_submission.csv.",
    )
    parser.add_argument(
        "--extreme-share",
        type=float,
        default=EXTREME_SHARE_WARNING,
        help=(
            "한 letter의 비중이 이 값을 넘으면 경고한다 "
            f"(기본: {EXTREME_SHARE_WARNING})"
        ),
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="경고가 하나라도 있으면 종료 코드 4로 끝낸다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check = check_submission(
            args.submission, args.sample, extreme_share=args.extreme_share
        )
    except SubmissionError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(check.format_report())
    if not check.ok:
        return EXIT_FAILED
    if check.warnings and args.strict_warnings:
        return EXIT_WARNINGS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
