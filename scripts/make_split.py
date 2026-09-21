"""Build the fixed, group aware validation split of the train set.

Usage::

    python scripts/make_split.py --val-size 500 --seed 20250921

The script writes ``artifacts/data/split_v1.json``, ``artifacts/data/train_ids.txt``,
``artifacts/data/val_ids.txt`` and ``artifacts/data/split_leakage.csv``. Two runs
with the same inputs produce identical files apart from the ``created`` stamp,
which is kept out of the printed content hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.data import file_sha256, fingerprints_for, get_fingerprints, load_csv  # noqa: E402
from vqa.split import (  # noqa: E402
    DEFAULT_PHASH_THRESHOLD,
    SPLIT_VERSION,
    build_groups,
    content_hash,
    exact_duplicate_pairs,
    near_duplicate_pairs,
    split_payload,
    stratified_group_split,
)

SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="DATA_ROOT directory that holds the CSV files and image folders",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <repo>/artifacts/data)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="target number of validation questions",
    )
    parser.add_argument("--seed", type=int, default=20250921)
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="maximum perceptual hash Hamming distance for a near duplicate",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parents[1] / "artifacts" / "data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: load_csv(data_root, split) for split in SPLITS}
    train = frames["train"]

    fingerprints = get_fingerprints(
        data_root,
        frames,
        out_dir / "image_hashes.csv",
        workers=args.workers,
        refresh=args.refresh,
    )
    train_fp = fingerprints_for(fingerprints, "train")

    exact = exact_duplicate_pairs(train_fp)
    near = near_duplicate_pairs(train_fp, threshold=int(args.threshold))
    print(
        f"train duplicates: {len(exact)} exact pairs, {len(near)} near pairs "
        f"(phash threshold {args.threshold})"
    )

    edges = [
        (str(a), str(b))
        for a, b in zip(exact["id_a"], exact["id_b"])
    ] + [
        (str(a), str(b))
        for a, b in zip(near["id_a"], near["id_b"])
    ]
    ids = [str(value) for value in train["id"].tolist()]
    group_of = build_groups(ids, edges)
    n_groups = len(set(group_of.values()))
    group_sizes = Counter(group_of.values())
    multi = sum(1 for size in group_sizes.values() if size > 1)
    print(f"groups: {n_groups} total, {multi} with more than one image")

    result = stratified_group_split(
        train,
        group_of,
        target_size=int(args.val_size),
        seed=int(args.seed),
    )
    print(
        f"split: {len(result.train_ids)} train, {len(result.val_ids)} val "
        f"in {len(result.val_groups)} groups"
    )
    print(f"val answer counts: {result.val_answer_counts}")

    # train 이미지가 test 또는 dev 이미지와 근접 중복인 경우를 목록으로만 남긴다.
    leakage_frames = []
    for other in ("dev", "test"):
        other_fp = fingerprints_for(fingerprints, other)
        pairs = near_duplicate_pairs(
            train_fp, other_fp, threshold=int(args.threshold)
        )
        leakage_frames.append(pairs)
    leakage = (
        pd.concat(leakage_frames, ignore_index=True)
        if leakage_frames
        else pd.DataFrame()
    )
    if len(leakage):
        leakage["train_group"] = leakage["id_a"].map(group_of)
        leakage["in_val"] = leakage["id_a"].isin(set(result.val_ids))
        leakage = leakage.sort_values(
            ["hamming", "id_a", "id_b"], kind="mergesort"
        ).reset_index(drop=True)
    leakage_path = out_dir / "split_leakage.csv"
    leakage.to_csv(leakage_path, index=False, encoding="utf-8")
    leaked_groups = sorted(set(leakage["train_group"].tolist())) if len(leakage) else []
    print(
        f"train groups that near-duplicate a dev or test image: "
        f"{len(leaked_groups)} (listed only, nothing removed)"
    )

    payload = split_payload(
        result,
        seed=int(args.seed),
        train_csv_sha256=file_sha256(data_root / "train.csv"),
        version=SPLIT_VERSION,
        extra={
            "phash_threshold": int(args.threshold),
            "n_exact_duplicate_pairs": int(len(exact)),
            "n_near_duplicate_pairs": int(len(near)),
            "n_multi_image_groups": int(multi),
            "leakage_groups_vs_dev_test": leaked_groups,
            "n_leakage_pairs_vs_dev_test": int(len(leakage)),
            "leakage_csv": leakage_path.name,
        },
    )
    digest = content_hash(payload)
    payload_with_stamp = dict(payload)
    payload_with_stamp["created"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    payload_with_stamp["content_sha256"] = digest

    # The file carries `created`, while the hashed content leaves it out, so two
    # runs differ only in that one line.
    split_path = out_dir / f"split_{SPLIT_VERSION}.json"
    ordered = {"created": payload_with_stamp["created"]}
    ordered.update(
        {
            key: value
            for key, value in sorted(payload_with_stamp.items())
            if key != "created"
        }
    )
    split_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out_dir / "train_ids.txt").write_text(
        "\n".join(result.train_ids) + "\n", encoding="utf-8"
    )
    (out_dir / "val_ids.txt").write_text(
        "\n".join(result.val_ids) + "\n", encoding="utf-8"
    )

    print(f"wrote {split_path}")
    print(f"wrote {out_dir / 'train_ids.txt'}")
    print(f"wrote {out_dir / 'val_ids.txt'}")
    print(f"wrote {leakage_path}")
    print(f"split content sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
