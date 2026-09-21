#!/usr/bin/env python3
"""같은 문항 집합에서 두 예측 파일을 문항 단위로 비교한다.

정확도만 비교하면 "0.9159에서 0.9226으로 올랐다"는 사실은 알 수 있어도, 그 차이가 표본
변동으로 설명되는 수준인지는 알 수 없다. 두 모델이 같은 문항을 풀었으므로 짝지은 비교를
해야 하고, 그 자리에 맞는 검정이 McNemar 검정이다. 한쪽만 맞힌 문항 수 두 개를 세서
이항 검정을 수행한다.

종료 코드
  0  비교 완료
  2  입력 오류 (파일 없음, 형식 위반, id 불일치)

사용 예
  python scripts/compare_predictions.py \
      --pred-a artifacts/runs/colab27b_base_bf16/preds/fold0_base.csv \
      --pred-b artifacts/runs/colab27b_ft_bf16/preds/fold0_ft.csv \
      --gold train.csv \
      --val-ids fold_csv_files/val_fold0.csv \
      --label-a base --label-b ft \
      --changed-out artifacts/eval/base_vs_ft_changed.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.evaluate import EvaluationError, load_gold  # noqa: E402
from vqa.predictions import (  # noqa: E402
    DEFAULT_TOLERANCE,
    PredictionError,
    atomic_write_text,
    decide_answers,
    load_predictions,
    read_id_list,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def mcnemar_exact_p(b: int, c: int) -> float:
    """한쪽만 맞힌 문항 수 b와 c로 양측 이항 검정의 정확 p값을 계산한다.

    귀무가설은 "두 모델이 서로 다르게 푼 문항 중에서 어느 쪽이 맞힐 확률이 각각 0.5"라는
    것이다. b와 c가 모두 0이면 비교할 문항이 없으므로 1.0을 돌려준다.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def mcnemar_statistic(b: int, c: int) -> float:
    """연속성 보정을 적용한 McNemar 카이제곱 통계량이다. 표본이 클 때 참고한다."""
    n = b + c
    if n == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / n


def answers_for(path: Path, *, tolerance: float) -> pd.DataFrame:
    """예측 확률 파일에서 id별 최종 답과 margin을 뽑는다."""
    frame = load_predictions(path, tolerance=tolerance)
    decided = decide_answers(frame)
    return decided.loc[:, ["id", "answer", "margin"]].set_index("id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_predictions.py",
        description="두 예측 파일을 문항 단위로 짝지어 비교하고 McNemar 검정을 수행한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pred-a", required=True, type=Path, metavar="PATH",
                        help="기준이 되는 예측 CSV. 보통 학습 전 원본 모델의 결과다.")
    parser.add_argument("--pred-b", required=True, type=Path, metavar="PATH",
                        help="비교할 예측 CSV. 보통 학습된 adapter의 결과다.")
    parser.add_argument("--gold", required=True, type=Path, metavar="PATH",
                        help="정답이 있는 CSV. 보통 train.csv다.")
    parser.add_argument("--val-ids", type=Path, metavar="PATH",
                        help="비교할 id 목록. .csv면 --id-column을, 그 밖이면 한 줄에 하나로 읽는다. "
                             "생략하면 두 예측과 gold의 교집합 전체를 쓴다.")
    parser.add_argument("--id-column", default="id",
                        help="--val-ids가 CSV일 때 읽을 컬럼 이름 (기본: id)")
    parser.add_argument("--label-a", default="A", help="보고서에 쓸 --pred-a의 이름")
    parser.add_argument("--label-b", default="B", help="보고서에 쓸 --pred-b의 이름")
    parser.add_argument("--changed-out", type=Path, metavar="PATH",
                        help="한쪽만 맞힌 문항을 저장할 CSV 경로")
    parser.add_argument("--json-out", type=Path, metavar="PATH",
                        help="비교 요약을 저장할 JSON 경로")
    parser.add_argument("--max-show", type=int, default=15,
                        help="화면에 보여줄 변화 문항 개수 (기본: 15)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"확률 합의 허용 오차 (기본: {DEFAULT_TOLERANCE})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        left = answers_for(args.pred_a, tolerance=args.tolerance)
        right = answers_for(args.pred_b, tolerance=args.tolerance)
        gold = load_gold(args.gold).set_index("id")
        if args.val_ids is not None:
            ids = read_id_list(args.val_ids, column=args.id_column)
        else:
            ids = sorted(set(left.index) & set(right.index) & set(gold.index))
    except (PredictionError, EvaluationError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    missing = {
        args.label_a: [i for i in ids if i not in left.index][:5],
        args.label_b: [i for i in ids if i not in right.index][:5],
        "gold": [i for i in ids if i not in gold.index][:5],
    }
    for source, examples in missing.items():
        if examples:
            print(
                f"[오류] {source} 에 없는 id가 있습니다. 예시={examples}",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
    if not ids:
        print("[오류] 비교할 id가 하나도 없습니다.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    table = pd.DataFrame(
        {
            "id": ids,
            "gold": [str(gold.loc[i, "answer"]) for i in ids],
            "answer_a": [str(left.loc[i, "answer"]) for i in ids],
            "answer_b": [str(right.loc[i, "answer"]) for i in ids],
            "margin_a": [float(left.loc[i, "margin"]) for i in ids],
            "margin_b": [float(right.loc[i, "margin"]) for i in ids],
        }
    )
    table["correct_a"] = table["answer_a"] == table["gold"]
    table["correct_b"] = table["answer_b"] == table["gold"]

    both_correct = int((table["correct_a"] & table["correct_b"]).sum())
    only_a = int((table["correct_a"] & ~table["correct_b"]).sum())
    only_b = int((~table["correct_a"] & table["correct_b"]).sum())
    both_wrong = int((~table["correct_a"] & ~table["correct_b"]).sum())
    n = len(table)

    accuracy_a = int(table["correct_a"].sum()) / n
    accuracy_b = int(table["correct_b"].sum()) / n
    p_value = mcnemar_exact_p(only_a, only_b)
    statistic = mcnemar_statistic(only_a, only_b)

    print("문항 수: %d" % n)
    print("%s 정확도: %.4f (%d/%d)"
          % (args.label_a, accuracy_a, int(table["correct_a"].sum()), n))
    print("%s 정확도: %.4f (%d/%d)"
          % (args.label_b, accuracy_b, int(table["correct_b"].sum()), n))
    print("정확도 차이: %+.4f (%s 기준)" % (accuracy_b - accuracy_a, args.label_b))
    print()
    print("짝지은 분할표")
    print("  둘 다 맞힌 문항: %d" % both_correct)
    print("  %s 만 맞힌 문항: %d" % (args.label_a, only_a))
    print("  %s 만 맞힌 문항: %d" % (args.label_b, only_b))
    print("  둘 다 틀린 문항: %d" % both_wrong)
    print()
    print("McNemar 검정 (양측 정확 이항 검정)")
    print("  불일치 문항 수: %d" % (only_a + only_b))
    print("  p 값: %.6f" % p_value)
    print("  연속성 보정 카이제곱 통계량: %.4f" % statistic)
    if only_a + only_b == 0:
        print("  판정: 두 예측이 모든 문항에서 같은 답을 냈습니다.")
    elif p_value < 0.05:
        print("  판정: 유의수준 0.05에서 두 모델의 정확도가 같다고 보기 어렵습니다.")
    else:
        print("  판정: 유의수준 0.05에서 두 모델의 차이를 표본 변동과 구분할 수 없습니다.")

    changed = table[table["correct_a"] != table["correct_b"]].copy()
    changed["direction"] = [
        "%s -> %s (개선)" % (args.label_a, args.label_b) if row.correct_b
        else "%s -> %s (악화)" % (args.label_a, args.label_b)
        for row in changed.itertuples()
    ]
    changed = changed.sort_values("margin_b", ascending=False)
    if args.max_show > 0 and len(changed):
        print()
        print("변화한 문항 (margin이 큰 순으로 최대 %d개)" % args.max_show)
        print("  %-20s %5s %5s %5s  %s" % ("id", "gold", args.label_a[:5], args.label_b[:5], "방향"))
        for row in changed.head(args.max_show).itertuples():
            print("  %-20s %5s %5s %5s  %s"
                  % (row.id, row.gold, row.answer_a, row.answer_b, row.direction))

    if args.changed_out is not None:
        args.changed_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.changed_out, changed.to_csv(index=False, lineterminator="\n"))
        print()
        print("변화 문항 저장    : %s (%d행)" % (args.changed_out, len(changed)))

    summary = {
        "pred_a": str(args.pred_a),
        "pred_b": str(args.pred_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "gold": str(args.gold),
        "val_ids": str(args.val_ids) if args.val_ids else None,
        "n": n,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "accuracy_delta": accuracy_b - accuracy_a,
        "both_correct": both_correct,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": both_wrong,
        "mcnemar_p_value": p_value,
        "mcnemar_statistic": statistic,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.json_out, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("요약 저장         : %s" % args.json_out)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
