#!/usr/bin/env python3
"""고정 validation ID에서 예측 파일의 정확도와 오답 분포를 본다.

정확도, 정답 letter별 정확도, confusion matrix, 예측 letter 분포, 평균 margin,
질문 유형별 분해를 출력하고 오답 ID를 margin과 함께 저장할 수 있다. 결과 한 줄은
`artifacts/experiments.jsonl`에 덧붙인다.

종료 코드
  0  평가 완료
  2  입력 오류 (파일 없음, 형식 위반, ID 불일치)
  3  --min-accuracy 기준에 못 미침

사용 예
  python scripts/evaluate_run.py \
      --pred artifacts/preds/ft27_val.csv \
      --gold train.csv \
      --val-ids artifacts/data/val_ids.txt \
      --wrong-out artifacts/eval/ft27_val_wrong.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.evaluate import (  # noqa: E402
    DEFAULT_EXPERIMENTS_LOG,
    EvaluationError,
    append_experiment_log,
    evaluate_predictions,
    load_gold,
)
from vqa.predictions import (  # noqa: E402
    DEFAULT_TOLERANCE,
    PredictionError,
    atomic_write_text,
    load_predictions,
    read_id_list,
    read_manifest,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_BELOW_THRESHOLD = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_run.py",
        description="예측 확률 파일을 고정 validation 정답과 비교한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        metavar="PATH",
        help="평가할 예측 CSV (id,p_a,p_b,p_c,p_d,run_id)",
    )
    parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        metavar="PATH",
        help="정답이 있는 CSV. 보통 train.csv.",
    )
    parser.add_argument(
        "--val-ids",
        type=Path,
        metavar="PATH",
        help="평가할 ID 목록. .csv면 --id-column을, 그 밖이면 한 줄에 하나로 읽는다. "
        "생략하면 예측과 gold의 교집합 전체를 쓴다.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="--val-ids가 CSV일 때 읽을 컬럼 이름 (기본: id)",
    )
    parser.add_argument(
        "--wrong-out",
        type=Path,
        metavar="PATH",
        help="오답 목록을 저장할 CSV 경로. margin이 큰 순으로 정렬한다.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        metavar="PATH",
        help="평가 요약을 저장할 JSON 경로.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_EXPERIMENTS_LOG,
        help=f"실험 기록 JSONL 경로 (기본: {DEFAULT_EXPERIMENTS_LOG})",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="실험 기록에 남기지 않는다.",
    )
    parser.add_argument(
        "--seconds-per-item",
        type=float,
        help="문항당 평균 추론 시간(초). 생략하면 manifest 값을 쓴다.",
    )
    parser.add_argument(
        "--note",
        help="실험 기록에 남길 짧은 메모.",
    )
    parser.add_argument(
        "--no-breakdown",
        action="store_true",
        help="질문 유형별 분해를 계산하지 않는다.",
    )
    parser.add_argument(
        "--allow-missing-ids",
        action="store_true",
        help="--val-ids 중 예측이나 gold에 없는 ID를 오류로 보지 않고 건너뛴다.",
    )
    parser.add_argument(
        "--max-wrong",
        type=int,
        default=20,
        help="화면에 보여줄 오답 개수 (기본: 20)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="이 정확도에 못 미치면 종료 코드 3으로 끝낸다.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"확률 합의 허용 오차 (기본: {DEFAULT_TOLERANCE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        predictions = load_predictions(args.pred, tolerance=args.tolerance)
        gold = load_gold(args.gold)
        ids = (
            read_id_list(args.val_ids, column=args.id_column)
            if args.val_ids is not None
            else None
        )
        result = evaluate_predictions(
            predictions,
            gold,
            ids=ids,
            breakdown=not args.no_breakdown,
            require_full_coverage=not args.allow_missing_ids,
        )
    except (PredictionError, EvaluationError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(result.format_report(max_wrong=args.max_wrong))

    if args.wrong_out is not None:
        args.wrong_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            args.wrong_out, result.wrong.to_csv(index=False, lineterminator="\n")
        )
        print(f"\n오답 목록 저장  : {args.wrong_out} ({len(result.wrong)}행)")

    if args.json_out is not None:
        atomic_write_text(
            args.json_out,
            json.dumps(result.to_summary(), ensure_ascii=False, indent=2) + "\n",
        )
        print(f"요약 저장       : {args.json_out}")

    seconds_per_item = args.seconds_per_item
    if seconds_per_item is None:
        manifest = read_manifest(args.pred)
        if manifest:
            seconds_per_item = manifest.get("seconds_per_item")

    if not args.no_log:
        entry = append_experiment_log(
            result,
            log_path=args.log,
            prediction_path=args.pred,
            seconds_per_item=seconds_per_item,
            note=args.note,
            extra={"gold_file": str(args.gold), "val_ids_file": str(args.val_ids) if args.val_ids else None},
        )
        print(f"실험 기록 추가  : {args.log}")
        print(f"  sha256        : {entry['file_sha256']}")

    if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
        print(
            f"\n[실패] 정확도 {result.accuracy:.4f}가 기준 {args.min_accuracy:.4f}에 못 미칩니다.",
            file=sys.stderr,
        )
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
