#!/usr/bin/env python3
"""예측 파일의 컬럼 이름을 팀원 도구 형식과 vqa 패키지 형식 사이에서 바꾼다.

두 형식은 같은 값을 다른 이름으로 담고 있다.

    팀원 도구 (infer_val.py, eval_val.py)  id, pred, prob_a, prob_b, prob_c, prob_d
    vqa 패키지 (vqa.predict, scripts/*)    id, p_a, p_b, p_c, p_d, run_id

사용 예::

    python scripts/convert_preds.py --to teammate \\
        --in artifacts/runs/colab27b_base_bf16/preds/fold0_base.csv \\
        --out preds_27b_fold0.csv

    python scripts/convert_preds.py --to vqa \\
        --in preds_qwen3b_p2.csv --out artifacts/preds/qwen3b_p2.csv \\
        --run-id qwen3b_p2

`--to vqa`는 각 행의 확률 합이 허용 오차 안에 있는지 확인한 다음 정확히 1이 되도록 다시
나눈다. 오차를 넘는 행이 있으면 아무것도 쓰지 않고 종료 코드 2로 끝낸다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.predictions import (  # noqa: E402
    DEFAULT_TOLERANCE,
    LETTERS,
    PRED_COLUMNS,
    PROB_COLUMNS,
    TEAMMATE_PROB_COLUMNS,
    PredictionError,
    atomic_write_text,
    load_predictions,
)

#: 팀원 형식의 컬럼 순서.
TEAMMATE_COLUMNS: tuple[str, ...] = ("id", "pred", *TEAMMATE_PROB_COLUMNS)


def read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PredictionError(f"입력 파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": "string"})
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def to_teammate(path: Path, *, tolerance: float) -> pd.DataFrame:
    """vqa 형식을 읽어 팀원 형식으로 바꾼다. `pred`는 확률의 argmax다."""
    frame = load_predictions(path, tolerance=tolerance)
    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    out = pd.DataFrame({"id": frame["id"].astype(str)})
    out["pred"] = [LETTERS[i] for i in probs.argmax(axis=1)]
    for column, letter_index in zip(TEAMMATE_PROB_COLUMNS, range(len(LETTERS))):
        out[column] = probs[:, letter_index]
    return out.loc[:, list(TEAMMATE_COLUMNS)]


def to_vqa(path: Path, *, run_id: str, tolerance: float) -> pd.DataFrame:
    """팀원 형식을 읽어 vqa 형식으로 바꾼다. 확률은 행마다 합이 1이 되도록 다시 나눈다."""
    frame = read_raw(path)
    if "id" not in frame.columns:
        raise PredictionError(f"{path}: id 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}")

    if all(col in frame.columns for col in TEAMMATE_PROB_COLUMNS):
        source_columns = list(TEAMMATE_PROB_COLUMNS)
    elif all(col in frame.columns for col in PROB_COLUMNS):
        source_columns = list(PROB_COLUMNS)
    else:
        raise PredictionError(
            f"{path}: prob_a~prob_d 도 p_a~p_d 도 없습니다. 있는 컬럼: {list(frame.columns)}"
        )

    values = frame.loc[:, source_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.where(~np.isfinite(values).all(axis=1))[0][:5]
        raise PredictionError(
            f"{path}: 확률이 비어 있거나 숫자가 아닌 행이 있습니다: "
            f"id={frame['id'].iloc[bad].tolist()}"
        )
    sums = values.sum(axis=1)
    off = np.abs(sums - 1.0) > tolerance
    if off.any():
        bad = np.where(off)[0][:5]
        detail = [f"{frame['id'].iloc[i]}={sums[i]:.6f}" for i in bad]
        raise PredictionError(
            f"{path}: 확률 합이 1과 다른 행이 {int(off.sum())}개 있습니다"
            f"(허용 오차 {tolerance}): {detail}"
        )

    out = pd.DataFrame({"id": frame["id"].astype(str).str.strip()})
    normalised = values / sums[:, None]
    for column, letter_index in zip(PROB_COLUMNS, range(len(LETTERS))):
        out[column] = normalised[:, letter_index]
    out["run_id"] = str(run_id)
    return out.loc[:, list(PRED_COLUMNS)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/convert_preds.py",
        description="예측 파일의 컬럼 이름을 팀원 형식과 vqa 형식 사이에서 바꾼다.",
    )
    parser.add_argument("--to", dest="target", required=True, choices=("teammate", "vqa"))
    parser.add_argument("--in", dest="source", required=True, help="입력 CSV")
    parser.add_argument("--out", required=True, help="출력 CSV")
    parser.add_argument(
        "--run-id",
        default=None,
        help="--to vqa 일 때 채워 넣을 run_id. 생략하면 입력 파일 이름을 쓴다.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--float-format", default="%.6g")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    try:
        if args.target == "teammate":
            frame = to_teammate(source, tolerance=args.tolerance)
        else:
            run_id = args.run_id or source.stem
            frame = to_vqa(source, run_id=run_id, tolerance=args.tolerance)
    except PredictionError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2

    text = frame.to_csv(index=False, lineterminator="\n", float_format=args.float_format)
    out_path = atomic_write_text(args.out, text)
    print(f"{source} -> {out_path} ({len(frame)}행, 형식 {args.target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
