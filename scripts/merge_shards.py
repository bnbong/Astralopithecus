#!/usr/bin/env python3
"""여러 대에서 만든 shard 예측 파일을 하나로 합치고 ID 일대일 대응을 검사한다.

추론 장비 5대는 각각 `preds.shard{i}of{n}.csv`에 이어쓰며 중단·재개될 수 있다. 이
스크립트는 부분 파일을 그대로 읽어 합친 뒤, 기대 ID 목록과 대조해 모든 ID가 정확히
한 번씩 있는지 확인한다. 빠진 ID나 예상 밖 ID가 있으면 `--allow-partial`을 주지 않는 한
0이 아닌 코드로 끝난다.

종료 코드
  0  모든 검사 통과 (또는 --allow-partial로 부분 결과를 허용)
  1  ID 검사 실패 (누락, 중복, 예상 밖 ID)
  2  입력 오류 (파일 없음, 형식 위반)

사용 예
  python scripts/merge_shards.py \
      --shards artifacts/preds/base27.shard*of5.csv \
      --expect-ids sample_submission.csv \
      --run-id base27_test \
      --out artifacts/preds/base27_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.predictions import (  # noqa: E402
    DEFAULT_TOLERANCE,
    PRED_COLUMNS,
    PredictionError,
    build_manifest,
    load_partial_predictions,
    read_id_list,
    read_manifest,
    write_predictions,
)

EXIT_OK = 0
EXIT_ID_MISMATCH = 1
EXIT_INPUT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_shards.py",
        description="shard 예측 파일들을 하나로 합치고 기대 ID 목록과 대조한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        type=Path,
        metavar="PATH",
        help="합칠 shard CSV 경로들. 셸 glob을 그대로 넘겨도 된다.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PATH",
        help="합친 결과를 쓸 CSV 경로. 같은 이름의 manifest sidecar도 함께 쓴다.",
    )
    parser.add_argument(
        "--expect-ids",
        type=Path,
        metavar="PATH",
        help="기대 ID 목록. .csv면 --id-column 컬럼을, 그 밖이면 한 줄에 하나로 읽는다.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="--expect-ids가 CSV일 때 읽을 컬럼 이름 (기본: id)",
    )
    parser.add_argument(
        "--run-id",
        help="결과 파일에 기록할 run_id. 생략하면 shard들의 run_id가 하나일 때 그것을 쓴다.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="ID가 빠져 있어도 합친 결과를 저장하고 0으로 끝낸다.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"확률 합이 1에서 벗어나도 되는 허용 오차 (기본: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("last", "error"),
        default="last",
        help=(
            "여러 shard에 같은 ID가 있을 때 처리 방식. "
            "last는 마지막 shard 값을 쓰고, error는 실패로 본다 (기본: last)"
        ),
    )
    parser.add_argument(
        "--seconds-per-item",
        type=float,
        help="manifest에 남길 문항당 평균 추론 시간(초).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="요약 출력을 줄인다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    shard_paths: list[Path] = []
    for path in args.shards:
        if not path.exists():
            print(f"[오류] shard 파일이 없습니다: {path}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        shard_paths.append(path)
    shard_paths.sort(key=lambda p: str(p))

    frames: list[pd.DataFrame] = []
    seen_run_ids: list[str] = []
    origin: dict[str, str] = {}
    cross_shard_duplicates: list[str] = []
    within_shard_duplicates: list[str] = []
    total_read = 0
    total_dropped = 0

    for path in shard_paths:
        try:
            frame, report = load_partial_predictions(path, tolerance=args.tolerance)
        except PredictionError as exc:
            print(f"[오류] {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        total_read += report.total_rows
        total_dropped += report.dropped_rows
        if not args.quiet:
            print(report.describe())
        within_shard_duplicates.extend(report.duplicate_ids)
        for run_id in report.run_ids:
            if run_id not in seen_run_ids:
                seen_run_ids.append(run_id)
        for row_id in frame["id"].astype(str):
            if row_id in origin and origin[row_id] != str(path):
                cross_shard_duplicates.append(row_id)
            origin[row_id] = str(path)
        frames.append(frame)

    if not frames or all(len(f) == 0 for f in frames):
        print("[오류] 읽을 수 있는 예측 행이 없습니다.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    merged = pd.concat(frames, ignore_index=True)
    # shard 안에서 재개하며 다시 쓴 행과 shard 사이에 겹친 행을 함께 본다.
    duplicate_ids = sorted(
        set(merged["id"][merged["id"].duplicated(keep=False)].astype(str))
        | set(within_shard_duplicates)
    )
    if duplicate_ids and args.duplicate_policy == "error":
        print(
            f"[오류] shard 사이에 중복 ID가 {len(duplicate_ids)}개 있습니다: "
            f"{duplicate_ids[:10]}",
            file=sys.stderr,
        )
        return EXIT_ID_MISMATCH
    merged = merged.drop_duplicates(subset="id", keep="last").reset_index(drop=True)

    run_id = args.run_id
    if run_id is None:
        if len(seen_run_ids) == 1:
            run_id = seen_run_ids[0]
        else:
            print(
                f"[오류] shard들의 run_id가 하나가 아닙니다({seen_run_ids}). "
                "--run-id로 지정하세요.",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
    merged["run_id"] = pd.array([run_id] * len(merged), dtype="string")

    missing_ids: list[str] = []
    unexpected_ids: list[str] = []
    if args.expect_ids is not None:
        try:
            expected = read_id_list(args.expect_ids, column=args.id_column)
        except PredictionError as exc:
            print(f"[오류] {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        expected_dupes = sorted({i for i in expected if expected.count(i) > 1})
        if expected_dupes:
            print(
                f"[오류] 기대 ID 목록 자체에 중복이 있습니다: {expected_dupes[:5]}",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
        expected_set = set(expected)
        merged_set = set(merged["id"].astype(str))
        missing_ids = [i for i in expected if i not in merged_set]
        unexpected_ids = sorted(merged_set - expected_set)
        # 기대 목록의 순서를 그대로 따른다.
        order = [i for i in expected if i in merged_set]
        rest = [i for i in merged["id"].astype(str) if i not in expected_set]
        merged = merged.set_index("id").loc[order + rest].reset_index()
        merged = merged.loc[:, list(PRED_COLUMNS)]

    print("")
    print(f"shard 수        : {len(shard_paths)}")
    print(f"읽은 행         : {total_read} (버린 행 {total_dropped})")
    print(f"합친 고유 ID    : {len(merged)}")
    print(f"run_id          : {run_id}")
    if duplicate_ids:
        print(f"중복 ID         : {len(duplicate_ids)} (정책 {args.duplicate_policy})")
        print(f"  예시          : {duplicate_ids[:10]}")
    if cross_shard_duplicates:
        print(f"shard 간 중복ID : {len(set(cross_shard_duplicates))}")
    if args.expect_ids is not None:
        print(f"누락 ID         : {len(missing_ids)}")
        if missing_ids:
            print(f"  예시          : {missing_ids[:10]}")
        print(f"예상 밖 ID      : {len(unexpected_ids)}")
        if unexpected_ids:
            print(f"  예시          : {unexpected_ids[:10]}")

    failed = bool(missing_ids or unexpected_ids)

    if failed and not args.allow_partial:
        print(
            "\n[실패] ID가 기대 목록과 일대일로 대응하지 않습니다. "
            "부분 결과를 그대로 저장하려면 --allow-partial을 주세요.",
            file=sys.stderr,
        )
        return EXIT_ID_MISMATCH

    source_manifests = {
        str(path): read_manifest(path) for path in shard_paths if read_manifest(path)
    }
    manifest = build_manifest(
        run_id=run_id,
        shard_count=len(shard_paths),
        seconds_per_item=args.seconds_per_item,
        extra={
            "merged_from": [str(p) for p in shard_paths],
            "merged_rows": int(len(merged)),
            "dropped_rows": int(total_dropped),
            "missing_ids": len(missing_ids),
            "unexpected_ids": len(unexpected_ids),
            "partial": bool(failed),
            "source_manifests": source_manifests,
        },
    )
    try:
        out = write_predictions(
            merged, args.out, tolerance=args.tolerance, manifest=manifest
        )
    except PredictionError as exc:
        print(f"[오류] 저장한 결과가 형식 검사를 통과하지 못했습니다: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(f"\n저장            : {out}")
    print(f"manifest        : {out.name}.manifest.json")
    if failed:
        print("상태            : 부분 결과 (--allow-partial)")
        return EXIT_OK
    print("상태            : 통과")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
