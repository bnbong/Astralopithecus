#!/usr/bin/env python3
"""GPU 없이 평가·병합·제출 경로를 끝까지 돌려보기 위한 가짜 확률 파일 생성기.

무작위 확률이나 정답을 목표 정확도만큼만 맞히는 oracle-with-noise 확률을 만든다.
shard 분할과 일부러 망가뜨린 변종도 만들 수 있어서 검사기들이 실제로 걸러내는지
확인할 수 있다. 실제 추론 결과가 아니므로 성능 근거로 쓰지 않는다.

종료 코드
  0  생성 완료
  2  입력 오류

사용 예
  # test 전체에 대한 무작위 확률을 5개 shard로
  python scripts/make_fake_predictions.py --ids sample_submission.csv \\
      --mode random --shards 5 --run-id fake_base27_test \\
      --out artifacts/fake/base27_test.csv

  # train 정답을 62%만 맞히는 validation 캐시
  python scripts/make_fake_predictions.py --ids artifacts/data/val_ids.txt \\
      --gold train.csv --mode oracle --accuracy 0.62 \\
      --run-id fake_ft27_val --out artifacts/fake/ft27_val.csv

  # 일부러 망가뜨린 변종
  python scripts/make_fake_predictions.py --ids sample_submission.csv \\
      --mode random --break duplicate-id --run-id fake_broken \\
      --out artifacts/fake/broken_dup.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.evaluate import EvaluationError, load_gold  # noqa: E402
from vqa.predictions import (  # noqa: E402
    LETTERS,
    PRED_COLUMNS,
    PROB_COLUMNS,
    PredictionError,
    atomic_write_bytes,
    build_manifest,
    predictions_to_csv_text,
    read_id_list,
    write_manifest,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

BREAK_MODES: tuple[str, ...] = (
    "none",
    "missing-id",
    "duplicate-id",
    "nan",
    "wrong-order",
    "bad-sum",
    "extra-index",
    "bom",
    "truncated",
    "multi-run-id",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_fake_predictions.py",
        description="가짜 예측 확률 파일(정상/shard/고장난 변종)을 만든다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ids",
        required=True,
        type=Path,
        metavar="PATH",
        help="ID 목록. .csv면 --id-column 컬럼을, 그 밖이면 한 줄에 하나로 읽는다.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="--ids가 CSV일 때 읽을 컬럼 이름 (기본: id)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PATH",
        help="출력 CSV 경로. --shards를 주면 '<stem>.shard{i}of{n}.csv'로 나눠 쓴다.",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "oracle"),
        default="random",
        help="random은 무작위 확률, oracle은 --gold 정답을 --accuracy만큼 맞히는 확률 (기본: random)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        metavar="PATH",
        help="oracle 모드에서 쓸 정답 CSV. 보통 train.csv.",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=0.6,
        help="oracle 모드의 목표 정확도 (기본: 0.6)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=4.0,
        help=(
            "확률 분포의 뾰족함. 클수록 1등 확률이 높아진다 (기본: 4.0). "
            "Dirichlet 농도의 역수 역할을 한다."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="생성할 파일에 기록할 run_id.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="나눠 쓸 shard 개수 (기본: 1). 계획대로 행번호 mod n으로 나눈다.",
    )
    parser.add_argument(
        "--break",
        dest="break_mode",
        choices=BREAK_MODES,
        default="none",
        help="일부러 만들 결함 종류 (기본: none)",
    )
    parser.add_argument(
        "--break-count",
        type=int,
        default=3,
        help="결함을 적용할 행 수 (기본: 3)",
    )
    parser.add_argument(
        "--break-shard",
        type=int,
        default=1,
        help="--shards와 함께 쓸 때 결함을 넣을 shard 번호 (1부터, 기본: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="난수 seed (기본: 0)",
    )
    parser.add_argument(
        "--seconds-per-item",
        type=float,
        default=1.0,
        help="manifest에 적을 문항당 시간(초) (기본: 1.0)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="manifest sidecar를 만들지 않는다.",
    )
    return parser


def _make_probs(rng: np.random.Generator, n: int, confidence: float) -> np.ndarray:
    """행마다 내림차순으로 정렬된 확률 네 개를 만든다."""
    alpha = np.full(4, max(1.0 / max(confidence, 1e-6), 1e-3))
    probs = rng.dirichlet(alpha, size=n)
    return -np.sort(-probs, axis=1)


def _place(
    rng: np.random.Generator, sorted_probs: np.ndarray, winners: np.ndarray
) -> np.ndarray:
    """정렬된 확률을 각 행의 winner letter가 1등이 되도록 네 자리에 배치한다."""
    n = sorted_probs.shape[0]
    out = np.zeros_like(sorted_probs)
    for row in range(n):
        winner = int(winners[row])
        others = [i for i in range(4) if i != winner]
        rng.shuffle(others)
        out[row, winner] = sorted_probs[row, 0]
        for rank, column in enumerate(others, start=1):
            out[row, column] = sorted_probs[row, rank]
    return out


def generate(
    ids: list[str],
    *,
    mode: str,
    run_id: str,
    seed: int,
    confidence: float,
    accuracy: float,
    gold_map: dict[str, str] | None,
) -> pd.DataFrame:
    """가짜 예측 DataFrame을 만든다."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    sorted_probs = _make_probs(rng, n, confidence)

    if mode == "oracle":
        if gold_map is None:
            raise EvaluationError("oracle 모드에는 --gold가 필요합니다.")
        missing = [i for i in ids if i not in gold_map]
        if missing:
            raise EvaluationError(
                f"gold에 없는 id가 {len(missing)}개 있습니다(예: {missing[:5]})."
            )
        letter_index = {letter: i for i, letter in enumerate(LETTERS)}
        gold_idx = np.array([letter_index[gold_map[i]] for i in ids])
        hit = rng.random(n) < accuracy
        winners = gold_idx.copy()
        # 틀리게 할 행은 정답이 아닌 letter 중 하나를 1등으로 만든다.
        wrong_offset = rng.integers(1, 4, size=n)
        winners[~hit] = (gold_idx[~hit] + wrong_offset[~hit]) % 4
    else:
        winners = rng.integers(0, 4, size=n)

    probs = _place(rng, sorted_probs, winners)
    probs = probs / probs.sum(axis=1, keepdims=True)

    frame = pd.DataFrame(probs, columns=list(PROB_COLUMNS))
    frame.insert(0, "id", pd.array(ids, dtype="string"))
    frame["run_id"] = pd.array([run_id] * n, dtype="string")
    return frame.loc[:, list(PRED_COLUMNS)]


def apply_break(
    frame: pd.DataFrame, mode: str, count: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, object]]:
    """DataFrame 수준에서 만들 수 있는 결함을 넣는다.

    파일 수준 결함(extra-index, bom, truncated)은 저장 단계에서 처리한다.
    """
    frame = frame.copy()
    note: dict[str, object] = {"break": mode}
    count = max(1, min(count, len(frame)))

    if mode == "missing-id":
        drop = rng.choice(len(frame), size=count, replace=False)
        note["removed_ids"] = frame["id"].iloc[drop].tolist()
        frame = frame.drop(index=frame.index[drop]).reset_index(drop=True)
    elif mode == "duplicate-id":
        pick = rng.choice(len(frame), size=count, replace=False)
        note["duplicated_ids"] = frame["id"].iloc[pick].tolist()
        frame = pd.concat([frame, frame.iloc[pick]], ignore_index=True)
    elif mode == "nan":
        pick = rng.choice(len(frame), size=count, replace=False)
        note["nan_ids"] = frame["id"].iloc[pick].tolist()
        for row in pick:
            frame.iloc[row, frame.columns.get_loc("p_b")] = float("nan")
    elif mode == "wrong-order":
        frame = frame.iloc[::-1].reset_index(drop=True)
        note["reversed"] = True
    elif mode == "bad-sum":
        # 각 값은 [0,1] 안에 두고 행의 합만 1에서 벗어나게 해서 합 검사만 건드린다.
        pick = rng.choice(len(frame), size=count, replace=False)
        note["bad_sum_ids"] = frame["id"].iloc[pick].tolist()
        for row in pick:
            for col in PROB_COLUMNS:
                frame.iloc[row, frame.columns.get_loc(col)] *= 0.6
    elif mode == "multi-run-id":
        pick = rng.choice(len(frame), size=count, replace=False)
        note["second_run_id"] = "other_run"
        for row in pick:
            frame.iloc[row, frame.columns.get_loc("run_id")] = "other_run"
    return frame, note


def write_with_file_break(frame: pd.DataFrame, path: Path, mode: str) -> Path:
    """파일 수준 결함을 넣어 저장한다. 결함이 없으면 정상 경로로 저장한다."""
    text = predictions_to_csv_text(frame)
    if mode == "extra-index":
        lines = text.splitlines()
        lines[0] = "," + lines[0]
        lines[1:] = [f"{i}," + line for i, line in enumerate(lines[1:])]
        text = "\n".join(lines) + "\n"
        return atomic_write_bytes(path, text.encode("utf-8"))
    if mode == "bom":
        return atomic_write_bytes(path, b"\xef\xbb\xbf" + text.encode("utf-8"))
    if mode == "truncated":
        # 마지막 행을 중간에서 끊어 중단된 추론을 흉내낸다.
        cut = text.rstrip("\n")
        cut = cut[: -max(len(cut.splitlines()[-1]) // 2, 1)]
        return atomic_write_bytes(path, cut.encode("utf-8"))
    return atomic_write_bytes(path, text.encode("utf-8"))


def shard_path(out: Path, index: int, count: int) -> Path:
    """`preds.shard{i}of{n}.csv` 형태의 shard 경로를 만든다."""
    return out.with_name(f"{out.stem}.shard{index}of{count}{out.suffix}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.shards < 1:
        print("[오류] --shards는 1 이상이어야 합니다.", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not 0.0 <= args.accuracy <= 1.0:
        print("[오류] --accuracy는 0과 1 사이여야 합니다.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        ids = read_id_list(args.ids, column=args.id_column)
    except PredictionError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not ids:
        print(f"[오류] ID 목록이 비어 있습니다: {args.ids}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    gold_map: dict[str, str] | None = None
    if args.mode == "oracle":
        if args.gold is None:
            print("[오류] oracle 모드에는 --gold가 필요합니다.", file=sys.stderr)
            return EXIT_INPUT_ERROR
        try:
            gold = load_gold(args.gold)
        except EvaluationError as exc:
            print(f"[오류] {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        gold_map = dict(zip(gold["id"], gold["answer"]))

    try:
        frame = generate(
            ids,
            mode=args.mode,
            run_id=args.run_id,
            seed=args.seed,
            confidence=args.confidence,
            accuracy=args.accuracy,
            gold_map=gold_map,
        )
    except EvaluationError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    rng = np.random.default_rng(args.seed + 9973)
    written: list[Path] = []

    if args.shards == 1:
        target_frame, note = (
            apply_break(frame, args.break_mode, args.break_count, rng)
            if args.break_mode != "none"
            else (frame, {"break": "none"})
        )
        path = write_with_file_break(target_frame, args.out, args.break_mode)
        written.append(path)
        if not args.no_manifest:
            manifest = build_manifest(
                run_id=args.run_id,
                csv_path=path,
                model_id=f"fake-{args.mode}",
                precision="none",
                prompt_id="fake",
                seconds_per_item=args.seconds_per_item,
                shard_index=1,
                shard_count=1,
                extra={"fake": True, "mode": args.mode, **note},
            )
            write_manifest(path, manifest)
    else:
        # 행번호 mod n으로 나눈다. 계획 12절이 정한 안정적인 분할 방식이다.
        positions = np.arange(len(frame))
        for index in range(1, args.shards + 1):
            part = frame.iloc[positions % args.shards == index - 1].reset_index(drop=True)
            mode = args.break_mode if index == args.break_shard else "none"
            part, note = (
                apply_break(part, mode, args.break_count, rng)
                if mode != "none"
                else (part, {"break": "none"})
            )
            path = write_with_file_break(part, shard_path(args.out, index, args.shards), mode)
            written.append(path)
            if not args.no_manifest:
                write_manifest(
                    path,
                    build_manifest(
                        run_id=args.run_id,
                        csv_path=path,
                        model_id=f"fake-{args.mode}",
                        precision="none",
                        prompt_id="fake",
                        seconds_per_item=args.seconds_per_item,
                        shard_index=index,
                        shard_count=args.shards,
                        extra={"fake": True, "mode": args.mode, **note},
                    ),
                )

    print(f"모드            : {args.mode}" + (f" (목표 정확도 {args.accuracy:g})" if args.mode == "oracle" else ""))
    print(f"run_id          : {args.run_id}")
    print(f"ID 수           : {len(ids)}")
    print(f"결함            : {args.break_mode}")
    print(f"생성한 파일     : {len(written)}개")
    for path in written:
        print(f"  - {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
