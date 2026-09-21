#!/usr/bin/env python3
"""사전 등록한 네 후보만 validation에서 비교하고, 고른 가중치를 test에 그대로 적용한다.

계획 8절이 정한 결합 후보는 아래 넷뿐이다. 새 가중치를 탐색하지 않는다.

  1. base27 단독                       (가장 단순)
  2. FT27 단독
  3. FT27 0.8 + base27 0.2
  4. FT27 0.8 + FT9 0.2

선택은 validation 확률 캐시에서만 한다. test에는 정답이 없으므로 결합 가중치를 고르는
데이터로 쓰지 않는다. 정확도가 같거나 --tie-epsilon 안의 차이면 더 단순한 후보를 고른다.
최종 답과 margin은 선택한 가중치로 만든 같은 확률에서 계산한다.

종료 코드
  0  선택과 적용 완료
  2  입력 오류 (파일 없음, 형식 위반, id 집합 불일치)
  3  후보를 하나도 평가할 수 없음

사용 예
  # validation에서 후보 비교만
  python scripts/combine.py \
      --val-base27 artifacts/preds/base27_val.csv \
      --val-ft27  artifacts/preds/ft27_val.csv \
      --val-ft9   artifacts/preds/ft9_val.csv \
      --gold train.csv --val-ids artifacts/data/val_ids.txt

  # 선택한 가중치를 test 확률에 적용해 결합 파일 만들기
  python scripts/combine.py \
      --val-base27 artifacts/preds/base27_val.csv \
      --val-ft27  artifacts/preds/ft27_val.csv \
      --val-ft9   artifacts/preds/ft9_val.csv \
      --gold train.csv --val-ids artifacts/data/val_ids.txt \
      --test-base27 artifacts/preds/base27_test.csv \
      --test-ft27  artifacts/preds/ft27_test.csv \
      --test-ft9   artifacts/preds/ft9_test.csv \
      --out artifacts/preds/combined_test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    average_predictions,
    build_manifest,
    decide_answers,
    load_predictions,
    read_id_list,
    write_predictions,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_NO_CANDIDATE = 3

#: 모델 이름 → CLI 인자 접미사.
SOURCES: tuple[str, ...] = ("base27", "ft27", "ft9")


@dataclass(frozen=True)
class Candidate:
    """사전 등록된 결합 후보 하나."""

    name: str
    #: 단순한 것부터 0. 동률이면 rank가 작은 후보를 고른다.
    rank: int
    #: 모델 이름 → 가중치.
    weights: dict[str, float]

    @property
    def sources(self) -> list[str]:
        return list(self.weights)

    def describe(self) -> str:
        return " + ".join(f"{name} {weight:g}" for name, weight in self.weights.items())


#: 계획 8절이 사전에 고정한 네 후보. 이 목록 밖의 가중치는 쓰지 않는다.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("base27", 0, {"base27": 1.0}),
    Candidate("ft27", 1, {"ft27": 1.0}),
    Candidate("ft27_0.8+base27_0.2", 2, {"ft27": 0.8, "base27": 0.2}),
    Candidate("ft27_0.8+ft9_0.2", 3, {"ft27": 0.8, "ft9": 0.2}),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="combine.py",
        description="사전 등록한 네 결합 후보를 validation에서 비교하고 test에 적용한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    for name in SOURCES:
        parser.add_argument(
            f"--val-{name}",
            type=Path,
            metavar="PATH",
            help=f"{name}의 validation 확률 파일.",
        )
    for name in SOURCES:
        parser.add_argument(
            f"--test-{name}",
            type=Path,
            metavar="PATH",
            help=f"{name}의 test 확률 파일. --out과 함께 쓴다.",
        )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        metavar="PATH",
        help="정답이 있는 CSV. 보통 train.csv. validation 비교에만 쓴다.",
    )
    parser.add_argument(
        "--val-ids",
        type=Path,
        metavar="PATH",
        help="비교에 쓸 validation ID 목록. 생략하면 공통 ID 전체를 쓴다.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="--val-ids가 CSV일 때 읽을 컬럼 이름 (기본: id)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="PATH",
        help="선택한 가중치로 결합한 test 확률을 저장할 경로.",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        metavar="PATH",
        help="선택한 가중치로 결합한 validation 확률을 저장할 경로.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        metavar="PATH",
        help="후보 비교 결과를 저장할 JSON 경로.",
    )
    parser.add_argument(
        "--run-id",
        help="결합 결과에 기록할 run_id. 생략하면 'combine_<후보이름>'을 쓴다.",
    )
    parser.add_argument(
        "--force-candidate",
        choices=[c.name for c in CANDIDATES],
        help="비교 결과와 무관하게 이 후보를 쓴다. 이유를 반드시 기록에 남길 것.",
    )
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=0.0,
        help="정확도 차이가 이 값 이하면 동률로 보고 더 단순한 후보를 고른다 (기본: 0.0)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"확률 합의 허용 오차 (기본: {DEFAULT_TOLERANCE})",
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
    return parser


def _load_group(
    args: argparse.Namespace, prefix: str, tolerance: float
) -> dict[str, pd.DataFrame]:
    """--val-* 또는 --test-* 인자로 받은 확률 파일들을 읽는다."""
    loaded: dict[str, pd.DataFrame] = {}
    for name in SOURCES:
        path = getattr(args, f"{prefix}_{name}".replace("-", "_"))
        if path is None:
            continue
        loaded[name] = load_predictions(path, tolerance=tolerance)
    return loaded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        val_frames = _load_group(args, "val", args.tolerance)
        test_frames = _load_group(args, "test", args.tolerance)
        gold = load_gold(args.gold)
        val_ids = (
            read_id_list(args.val_ids, column=args.id_column)
            if args.val_ids is not None
            else None
        )
    except (PredictionError, EvaluationError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not val_frames:
        print("[오류] validation 확률 파일을 하나도 주지 않았습니다.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # test는 정답이 없으므로 후보 비교에 절대 쓰지 않는다.
    rows: list[dict[str, object]] = []
    evaluated: dict[str, float] = {}
    for candidate in CANDIDATES:
        missing = [name for name in candidate.sources if name not in val_frames]
        if missing:
            rows.append(
                {
                    "candidate": candidate.name,
                    "rank": candidate.rank,
                    "weights": candidate.weights,
                    "status": f"건너뜀 (validation 파일 없음: {missing})",
                    "accuracy": None,
                    "n": None,
                    "mean_margin": None,
                }
            )
            continue
        try:
            combined = average_predictions(
                [val_frames[name] for name in candidate.sources],
                [candidate.weights[name] for name in candidate.sources],
                run_id=f"combine_{candidate.name}",
            )
            result = evaluate_predictions(
                combined, gold, ids=val_ids, breakdown=False
            )
        except (PredictionError, EvaluationError) as exc:
            print(f"[오류] 후보 '{candidate.name}' 평가 실패: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        evaluated[candidate.name] = result.accuracy
        rows.append(
            {
                "candidate": candidate.name,
                "rank": candidate.rank,
                "weights": candidate.weights,
                "status": "평가함",
                "accuracy": result.accuracy,
                "n": result.n,
                "mean_margin": result.mean_margin,
            }
        )

    print("validation 후보 비교 (test는 선택에 쓰지 않음)")
    print(f"  gold      : {args.gold}")
    print(f"  val ids   : {args.val_ids if args.val_ids else '(공통 ID 전체)'}")
    print("")
    print("  후보                     가중치                  n   accuracy   mean_margin")
    for row in rows:
        if row["accuracy"] is None:
            print(f"  {row['candidate']:<22} {'-':<22} {row['status']}")
        else:
            weights = " ".join(f"{k}={v:g}" for k, v in row["weights"].items())
            print(
                f"  {row['candidate']:<22} {weights:<22} {row['n']:>5}"
                f"   {row['accuracy']:.4f}     {row['mean_margin']:.4f}"
            )

    if not evaluated:
        print("\n[오류] 평가할 수 있는 후보가 없습니다.", file=sys.stderr)
        return EXIT_NO_CANDIDATE

    if args.force_candidate is not None:
        if args.force_candidate not in evaluated:
            print(
                f"\n[오류] --force-candidate '{args.force_candidate}'는 평가되지 않았습니다.",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
        chosen = next(c for c in CANDIDATES if c.name == args.force_candidate)
        reason = "--force-candidate로 지정"
    else:
        best_accuracy = max(evaluated.values())
        # 최고 정확도와의 차이가 tie-epsilon 이하인 후보 중 가장 단순한 것을 고른다.
        tied = [
            c
            for c in CANDIDATES
            if c.name in evaluated
            and best_accuracy - evaluated[c.name] <= args.tie_epsilon
        ]
        chosen = min(tied, key=lambda c: c.rank)
        if len(tied) > 1:
            reason = (
                f"최고 정확도 {best_accuracy:.4f}와 차이가 {args.tie_epsilon:g} 이하인 "
                f"후보 {[c.name for c in tied]} 중 가장 단순한 것"
            )
        else:
            reason = f"validation 정확도 최고 ({best_accuracy:.4f})"

    print("")
    print(f"선택한 후보     : {chosen.name}")
    print(f"가중치          : {chosen.describe()}")
    print(f"선택 근거       : {reason}")
    print(f"validation 정확도: {evaluated[chosen.name]:.4f}")

    run_id = args.run_id or f"combine_{chosen.name}"

    report = {
        "chosen": chosen.name,
        "weights": chosen.weights,
        "reason": reason,
        "tie_epsilon": args.tie_epsilon,
        "val_accuracy": evaluated[chosen.name],
        "candidates": rows,
        "val_files": {k: str(getattr(args, f"val_{k}")) for k in val_frames},
        "test_files": {k: str(getattr(args, f"test_{k}")) for k in test_frames},
        "run_id": run_id,
    }
    if args.report_out is not None:
        atomic_write_text(
            args.report_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"후보 비교 저장  : {args.report_out}")

    if args.val_out is not None:
        combined_val = average_predictions(
            [val_frames[name] for name in chosen.sources],
            [chosen.weights[name] for name in chosen.sources],
            run_id=run_id,
        )
        write_predictions(
            combined_val,
            args.val_out,
            tolerance=args.tolerance,
            manifest=build_manifest(
                run_id=run_id,
                extra={"stage": "val", "candidate": chosen.name, "weights": chosen.weights},
            ),
        )
        print(f"결합 val 저장   : {args.val_out}")

    if args.out is not None:
        missing = [name for name in chosen.sources if name not in test_frames]
        if missing:
            print(
                f"\n[오류] 선택한 후보 '{chosen.name}'에 필요한 test 파일이 없습니다: {missing}",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
        try:
            combined_test = average_predictions(
                [test_frames[name] for name in chosen.sources],
                [chosen.weights[name] for name in chosen.sources],
                run_id=run_id,
            )
        except PredictionError as exc:
            print(f"\n[오류] test 결합 실패: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        write_predictions(
            combined_test,
            args.out,
            tolerance=args.tolerance,
            manifest=build_manifest(
                run_id=run_id,
                extra={
                    "stage": "test",
                    "candidate": chosen.name,
                    "weights": chosen.weights,
                    "reason": reason,
                    "val_accuracy": evaluated[chosen.name],
                    "test_sources": {k: str(getattr(args, f"test_{k}")) for k in chosen.sources},
                },
            ),
        )
        decided = decide_answers(combined_test)
        counts = decided["answer"].value_counts().to_dict()
        print(f"결합 test 저장  : {args.out} ({len(combined_test)}행)")
        print(f"  답 분포       : {{{', '.join(f'{k}: {counts.get(k, 0)}' for k in 'abcd')}}}")
        print(f"  평균 margin   : {decided['margin'].mean():.4f}")

    if not args.no_log:
        append_experiment_log(
            evaluate_predictions(
                average_predictions(
                    [val_frames[name] for name in chosen.sources],
                    [chosen.weights[name] for name in chosen.sources],
                    run_id=run_id,
                ),
                gold,
                ids=val_ids,
                breakdown=False,
            ),
            log_path=args.log,
            prediction_path=args.val_out if args.val_out else None,
            note=f"combine: {chosen.name} ({reason})",
            extra={"candidates": {k: v for k, v in evaluated.items()}},
        )
        print(f"실험 기록 추가  : {args.log}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
