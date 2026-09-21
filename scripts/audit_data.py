"""Audit the competition CSV files and image folders.

Usage::

    python scripts/audit_data.py --data-root . --threshold 5

The script writes ``artifacts/data/audit_report.json``,
``artifacts/data/audit_report.md``, ``artifacts/data/near_duplicate_pairs.csv``
and the per-image hash cache ``artifacts/data/image_hashes.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa.data import (  # noqa: E402
    DEV_ANSWER_COLUMNS,
    VALID_ANSWERS,
    file_sha256,
    fingerprints_for,
    get_fingerprints,
    load_csv,
)
from vqa.split import (  # noqa: E402
    DEFAULT_PHASH_THRESHOLD,
    exact_duplicate_pairs,
    near_duplicate_pairs,
)

SPLITS = ("train", "dev", "test")

# 계획 7절 표에 기록된 dev 투표 통계의 실측값이다.
PLAN_DEV_STATS = {
    "n_rows": 2683,
    "n_columns": 12,
    "valid_5": 2287,
    "valid_4": 364,
    "valid_3": 30,
    "valid_0": 2,
    "top_votes_3": 2026,
    "top_votes_2": 649,
    "top_votes_1": 6,
    "top_votes_0": 2,
    "top_votes_ge4": 0,
    "tie_rows": 465,
    "blank_cells": 434,
    "blank_rows": 396,
}
PLAN_DEV_CSV_SHA256 = (
    "ffc92bc4dfccaad566547128a864e63090f50f4f0b208ae4d2474bc760fb269a"
)


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
        "--threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="maximum perceptual hash Hamming distance for a near duplicate",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the image hash cache and recompute every fingerprint",
    )
    return parser.parse_args()


def summarise_missing(df: pd.DataFrame) -> dict[str, int]:
    """Count blank cells per column."""
    return {
        column: int((df[column].astype(str).str.strip() == "").sum())
        for column in df.columns
    }


def summarise_sizes(fp: pd.DataFrame) -> dict[str, object]:
    """Describe the image resolution distribution of one split."""
    readable = fp[fp["readable"]]
    if len(readable) == 0:
        return {"n_readable": 0}
    width = readable["width"].to_numpy(dtype=float)
    height = readable["height"].to_numpy(dtype=float)
    pixels = width * height
    ratio = width / np.maximum(height, 1.0)

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    resolutions = Counter(
        f"{int(w)}x{int(h)}" for w, h in zip(width.tolist(), height.tolist())
    )
    return {
        "n_readable": int(len(readable)),
        "width": describe(width),
        "height": describe(height),
        "megapixels": describe(pixels / 1e6),
        "aspect_ratio": describe(ratio),
        "portrait_rows": int(np.sum(height > width)),
        "landscape_rows": int(np.sum(width > height)),
        "square_rows": int(np.sum(width == height)),
        "top_resolutions": resolutions.most_common(10),
        "formats": dict(Counter(readable["format"].tolist())),
        "modes": dict(Counter(readable["mode"].tolist())),
        "file_size_bytes": describe(readable["size_bytes"].to_numpy(dtype=float)),
    }


def summarise_dev_votes(dev: pd.DataFrame) -> dict[str, object]:
    """Reproduce the dev vote statistics that plan section 7 lists."""
    votes = dev[list(DEV_ANSWER_COLUMNS)].apply(
        lambda column: column.astype(str).str.strip().str.lower()
    )
    valid_counts = Counter()
    top_vote_counts = Counter()
    tie_rows = 0
    blank_cells = 0
    blank_rows = 0
    top_letter_counter = Counter()

    for _, row in votes.iterrows():
        values = [value for value in row.tolist()]
        blanks = sum(1 for value in values if value == "")
        blank_cells += blanks
        if blanks:
            blank_rows += 1
        valid = [value for value in values if value in VALID_ANSWERS]
        valid_counts[len(valid)] += 1
        if not valid:
            top_vote_counts[0] += 1
            continue
        counter = Counter(valid)
        top = max(counter.values())
        top_vote_counts[top] += 1
        winners = [letter for letter, count in counter.items() if count == top]
        if len(winners) > 1:
            tie_rows += 1
        else:
            top_letter_counter[winners[0]] += 1

    measured = {
        "n_rows": int(len(dev)),
        "n_columns": int(dev.shape[1]),
        "valid_5": int(valid_counts[5]),
        "valid_4": int(valid_counts[4]),
        "valid_3": int(valid_counts[3]),
        "valid_2": int(valid_counts[2]),
        "valid_1": int(valid_counts[1]),
        "valid_0": int(valid_counts[0]),
        "top_votes_5": int(top_vote_counts[5]),
        "top_votes_4": int(top_vote_counts[4]),
        "top_votes_3": int(top_vote_counts[3]),
        "top_votes_2": int(top_vote_counts[2]),
        "top_votes_1": int(top_vote_counts[1]),
        "top_votes_0": int(top_vote_counts[0]),
        "top_votes_ge4": int(top_vote_counts[4] + top_vote_counts[5]),
        "tie_rows": int(tie_rows),
        "blank_cells": int(blank_cells),
        "blank_rows": int(blank_rows),
        "unique_top_letter_counts": {
            letter: int(top_letter_counter[letter]) for letter in VALID_ANSWERS
        },
    }
    mismatches = {
        key: {"plan": value, "measured": measured.get(key)}
        for key, value in PLAN_DEV_STATS.items()
        if measured.get(key) != value
    }
    return {"measured": measured, "plan": PLAN_DEV_STATS, "mismatches": mismatches}


def summarise_questions(df: pd.DataFrame) -> dict[str, object]:
    """Report how often the same question text appears more than once."""
    questions = df["question"].astype(str).str.strip()
    counts = Counter(questions.tolist())
    repeated = {text: count for text, count in counts.items() if count > 1}
    top = sorted(repeated.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "n_unique": int(len(counts)),
        "n_repeated_texts": int(len(repeated)),
        "n_rows_in_repeated_texts": int(sum(repeated.values())),
        "top_repeated": [{"question": text, "count": count} for text, count in top],
    }


def summarise_options(df: pd.DataFrame) -> dict[str, object]:
    """Report rows whose four options are not four distinct strings."""
    option_columns = ["a", "b", "c", "d"]
    options = df[option_columns].astype(str)
    duplicate_option_rows = int(
        (options.nunique(axis=1) < len(option_columns)).sum()
    )
    blank_option_rows = int(
        (options.apply(lambda col: col.str.strip() == "").any(axis=1)).sum()
    )
    return {
        "rows_with_duplicate_options": duplicate_option_rows,
        "rows_with_blank_option": blank_option_rows,
    }


def pair_counts_by_split(pairs: pd.DataFrame) -> dict[str, int]:
    """Count pairs per unordered split combination."""
    if len(pairs) == 0:
        return {}
    keys = [
        "-".join(sorted((str(a), str(b))))
        for a, b in zip(pairs["split_a"], pairs["split_b"])
    ]
    return dict(sorted(Counter(keys).items()))


def _json_default(value: object) -> object:
    """Convert numpy scalars so that :mod:`json` can serialise the report."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def write_markdown(report: dict[str, object], path: Path) -> None:
    """Write the short Korean summary of the audit."""
    lines: list[str] = []
    add = lines.append

    created = report["created"]
    threshold = report["phash_threshold"]

    add("# 데이터 점검 보고서")
    add("")
    add(f"이 보고서는 {created}에 `scripts/audit_data.py`를 실행하여 생성하였습니다.")
    add(
        "점검 대상은 train, dev, test의 CSV 파일과 이미지 폴더이며, "
        f"근접 중복 판정에는 perceptual hash의 Hamming 거리 {threshold} 이하라는 "
        "기준을 적용하였습니다."
    )
    add("")

    add("## 1. 행 수와 파일 해시")
    add("")
    add("| 분할 | 행 수 | 컬럼 수 | CSV SHA256 |")
    add("|---|---|---|---|")
    for split in SPLITS:
        info = report["splits"][split]
        add(
            f"| {split} | {info['n_rows']:,} | {info['n_columns']} | "
            f"`{info['csv_sha256'][:16]}…` |"
        )
    add("")
    dev_hash_matches = report["dev_votes"]["csv_sha256_matches_plan"]
    if dev_hash_matches:
        add(
            "dev.csv의 SHA256은 계획 7절에 기록된 값과 일치하므로, "
            "분석 대상 파일이 동일하다는 사실을 확인하였습니다."
        )
    else:
        add(
            "dev.csv의 SHA256이 계획 7절에 기록된 값과 다르기 때문에, "
            "분석 대상 파일이 교체되었는지 먼저 확인해야 합니다."
        )
    add("")

    add("## 2. 결측값과 이미지 파일 상태")
    add("")
    add("| 분할 | 빈 셀 | 누락 이미지 | 열 수 없는 이미지 | 중복 id |")
    add("|---|---|---|---|---|")
    for split in SPLITS:
        info = report["splits"][split]
        blank_total = sum(info["missing_values"].values())
        add(
            f"| {split} | {blank_total:,} | {info['images']['missing']:,} | "
            f"{info['images']['unreadable']:,} | {info['duplicate_ids']:,} |"
        )
    add("")
    total_missing = sum(
        report["splits"][split]["images"]["missing"] for split in SPLITS
    )
    total_unreadable = sum(
        report["splits"][split]["images"]["unreadable"] for split in SPLITS
    )
    if total_missing == 0 and total_unreadable == 0:
        add(
            "모든 행의 이미지 파일이 실제로 존재하였고 Pillow로 정상적으로 "
            "열렸기 때문에, 경로 정합성 문제는 발견되지 않았습니다."
        )
    else:
        add(
            f"이미지 파일 {total_missing:,}건이 누락되었고 {total_unreadable:,}건은 "
            "열리지 않았으므로, 해당 행은 학습과 추론 대상에서 제외하거나 "
            "원본 파일을 다시 받아야 합니다."
        )
    add("")

    add("## 3. 이미지 크기 분포")
    add("")
    add(
        "| 분할 | 가로 중앙값 | 세로 중앙값 | 화소 수 중앙값(MP) | 화소 수 최댓값(MP) | "
        "가로 최솟값 | 가로 최댓값 | 세로 최솟값 | 세로 최댓값 | 세로가 더 긴 이미지 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for split in SPLITS:
        sizes = report["splits"][split]["image_sizes"]
        if not sizes.get("n_readable"):
            continue
        add(
            f"| {split} | {sizes['width']['median']:,.0f} | "
            f"{sizes['height']['median']:,.0f} | "
            f"{sizes['megapixels']['median']:.2f} | "
            f"{sizes['megapixels']['max']:.2f} | "
            f"{sizes['width']['min']:,.0f} | {sizes['width']['max']:,.0f} | "
            f"{sizes['height']['min']:,.0f} | {sizes['height']['max']:,.0f} | "
            f"{sizes['portrait_rows']:,} |"
        )
    add("")
    add(
        "가로와 세로의 최솟값과 최댓값은 서로 다른 이미지에서 관측된 값이기 때문에, "
        "한 장의 해상도를 나타내지는 않습니다."
    )
    top = report["splits"]["train"]["image_sizes"]["top_resolutions"]
    top_text = ", ".join(
        f"{resolution} {count:,}장" for resolution, count in top[:3]
    )
    add(
        f"train에서 가장 흔한 해상도는 {top_text}이며, 세 분할 모두 720픽셀을 "
        "짧은 변으로 삼은 세로 사진이 다수를 차지합니다."
    )
    add(
        "해상도가 넓은 범위에 걸쳐 분포하기 때문에, 입력 해상도와 pixel budget을 "
        "확정할 때 작은 이미지의 글자 가독성과 큰 이미지의 처리 시간을 함께 "
        "고려해야 합니다."
    )
    add("")

    add("## 4. 중복 이미지와 누수 위험")
    add("")
    add("| 구분 | 쌍 수 |")
    add("|---|---|")
    add(f"| 완전 중복(SHA256 일치) 전체 | {report['exact_duplicates']['n_pairs']:,} |")
    for key, count in report["exact_duplicates"]["by_split_pair"].items():
        add(f"| 완전 중복 {key} | {count:,} |")
    add(f"| 근접 중복(phash) 전체 | {report['near_duplicates']['n_pairs']:,} |")
    for key, count in report["near_duplicates"]["by_split_pair"].items():
        add(f"| 근접 중복 {key} | {count:,} |")
    add("")
    train_train = report["near_duplicates"]["by_split_pair"].get("train-train", 0)
    cross_pairs = sum(
        count
        for key, count in report["near_duplicates"]["by_split_pair"].items()
        if key in {"dev-train", "test-train", "dev-test"}
    )
    add(
        f"train 내부에서 근접 중복 쌍 {train_train:,}건이 발견되었으므로, "
        "validation을 만들 때 이 쌍들을 같은 그룹으로 묶어서 한쪽으로만 "
        "보내야 합니다."
    )
    add(
        f"train, dev, test 사이를 넘나드는 근접 중복 쌍은 {cross_pairs:,}건입니다. "
        "이 목록은 자동 판정 결과이므로 사람이 직접 확인한 뒤에 학습 제외 여부를 "
        "결정해야 합니다."
    )
    semantics = report["train_duplicate_semantics"]
    add(
        f"train 내부의 근접 중복 {semantics['n_pairs']:,}쌍 가운데 질문 문구까지 "
        f"같은 경우는 {semantics['pairs_with_same_question']:,}쌍이고, 정답 문자가 "
        f"같은 경우는 {semantics['pairs_with_same_answer']:,}쌍입니다. 같은 사진에 "
        "서로 다른 문항이 출제된 사례가 대부분이므로, 이미지 단위로 그룹을 묶지 "
        "않으면 validation 점수가 실제 성능보다 높게 나올 수 있습니다."
    )
    add(
        "쌍 목록은 `artifacts/data/near_duplicate_pairs.csv`에 저장하였고, "
        "이미지별 해시는 `artifacts/data/image_hashes.csv`에 캐시하였습니다."
    )
    add("")

    add("## 5. 질문 문구와 정답 분포")
    add("")
    add("| 분할 | 고유 질문 수 | 2회 이상 반복된 질문 수 | 반복 질문에 속한 행 수 |")
    add("|---|---|---|---|")
    for split in SPLITS:
        questions = report["splits"][split]["questions"]
        add(
            f"| {split} | {questions['n_unique']:,} | "
            f"{questions['n_repeated_texts']:,} | "
            f"{questions['n_rows_in_repeated_texts']:,} |"
        )
    add("")
    answer_counts = report["splits"]["train"]["answer_distribution"]
    total_answers = max(1, sum(answer_counts.values()))
    distribution = ", ".join(
        f"{letter} {answer_counts.get(letter, 0):,}건"
        f"({answer_counts.get(letter, 0) / total_answers * 100:.1f}%)"
        for letter in VALID_ANSWERS
    )
    add(f"train의 정답 분포는 {distribution}입니다.")
    add("")

    add("## 6. dev 응답 통계와 계획 표의 비교")
    add("")
    measured = report["dev_votes"]["measured"]
    add("| 항목 | 계획 표 | 실측 |")
    add("|---|---|---|")
    labels = [
        ("n_rows", "행 수"),
        ("valid_5", "유효 응답 5개"),
        ("valid_4", "유효 응답 4개"),
        ("valid_3", "유효 응답 3개"),
        ("valid_0", "유효 응답 0개"),
        ("top_votes_3", "최다 응답 3표"),
        ("top_votes_2", "최다 응답 2표"),
        ("top_votes_1", "최다 응답 1표"),
        ("top_votes_ge4", "최다 응답 4표 이상"),
        ("tie_rows", "최다 응답 동률"),
        ("blank_cells", "빈 응답 칸"),
        ("blank_rows", "빈 응답이 있는 행"),
    ]
    for key, label in labels:
        planned = PLAN_DEV_STATS.get(key)
        planned_text = f"{planned:,}" if isinstance(planned, int) else "-"
        add(f"| {label} | {planned_text} | {measured.get(key, 0):,} |")
    add("")
    if report["dev_votes"]["mismatches"]:
        add(
            "계획 표와 다른 항목이 있으므로, 아래 목록을 확인하고 어느 쪽 수치를 "
            "기준으로 삼을지 먼저 정해야 합니다."
        )
        for key, values in report["dev_votes"]["mismatches"].items():
            add(f"- {key}: 계획 {values['plan']:,}건, 실측 {values['measured']:,}건")
    else:
        add(
            "모든 항목이 계획 7절의 표와 일치하였기 때문에, 의사 라벨 후보 규모를 "
            "그대로 사용할 수 있습니다."
        )
    add("")

    add("## 7. 다음 작업")
    add("")
    add(
        "1. `python scripts/make_split.py`를 실행하여 그룹 단위 validation을 "
        "고정합니다."
    )
    add(
        "2. `artifacts/data/near_duplicate_pairs.csv`에서 train과 test 또는 dev가 "
        "짝을 이루는 행을 사람이 직접 확인합니다."
    )
    add(
        "3. 확인 결과를 데이터 manifest에 기록하여 학습 데이터의 출처와 제외 "
        "사유를 남깁니다."
    )
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parents[1] / "artifacts" / "data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data root: {data_root}")
    frames = {split: load_csv(data_root, split) for split in SPLITS}
    for split, df in frames.items():
        print(f"  loaded {split}.csv: {len(df)} rows, {df.shape[1]} columns")

    fingerprints = get_fingerprints(
        data_root,
        frames,
        out_dir / "image_hashes.csv",
        workers=args.workers,
        refresh=args.refresh,
    )

    report: dict[str, object] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "phash_threshold": int(args.threshold),
        "splits": {},
    }

    for split in SPLITS:
        df = frames[split]
        fp = fingerprints_for(fingerprints, split)
        info: dict[str, object] = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "csv_sha256": file_sha256(data_root / f"{split}.csv"),
            "missing_values": summarise_missing(df),
            "duplicate_ids": int(df["id"].duplicated().sum()),
            "duplicate_paths": int(df["path"].duplicated().sum()),
            "questions": summarise_questions(df),
            "options": summarise_options(df),
            "images": {
                "n_rows": int(len(fp)),
                "missing": int((~fp["exists"]).sum()),
                "unreadable": int((fp["exists"] & ~fp["readable"]).sum()),
                "errors": dict(
                    Counter(
                        error for error in fp["error"].tolist() if error
                    ).most_common(10)
                ),
            },
            "image_sizes": summarise_sizes(fp),
        }
        if "answer" in df.columns:
            answers = df["answer"].astype(str).str.strip().str.lower()
            info["answer_distribution"] = {
                letter: int((answers == letter).sum()) for letter in VALID_ANSWERS
            }
        report["splits"][split] = info

    print("finding exact duplicates...", flush=True)
    exact = exact_duplicate_pairs(fingerprints)
    print(f"  {len(exact)} exact duplicate pairs", flush=True)

    print("finding near duplicates...", flush=True)
    near = near_duplicate_pairs(fingerprints, threshold=int(args.threshold))
    print(f"  {len(near)} near duplicate pairs", flush=True)

    pairs_path = out_dir / "near_duplicate_pairs.csv"
    review = near.copy()
    review["needs_review"] = True
    review.to_csv(pairs_path, index=False, encoding="utf-8")

    report["exact_duplicates"] = {
        "n_pairs": int(len(exact)),
        "n_images_involved": int(
            len(set(exact["id_a"].tolist()) | set(exact["id_b"].tolist()))
        ),
        "by_split_pair": pair_counts_by_split(exact),
        "examples": exact.head(10).to_dict("records"),
    }
    report["near_duplicates"] = {
        "n_pairs": int(len(near)),
        "n_images_involved": int(
            len(set(near["id_a"].tolist()) | set(near["id_b"].tolist()))
        ),
        "by_split_pair": pair_counts_by_split(near),
        "hamming_histogram": {
            str(distance): int(count)
            for distance, count in sorted(
                Counter(near["hamming"].tolist()).items()
            )
        },
        "pairs_csv": str(pairs_path),
        "examples": near.head(10).to_dict("records"),
    }

    # 같은 이미지가 서로 다른 질문으로 다시 출제되었는지 확인한다.
    train_rows = frames["train"].set_index("id")
    train_pairs = near[(near["split_a"] == "train") & (near["split_b"] == "train")]
    same_question = 0
    same_answer = 0
    for _, pair in train_pairs.iterrows():
        left, right = str(pair["id_a"]), str(pair["id_b"])
        same_question += int(
            train_rows.loc[left, "question"] == train_rows.loc[right, "question"]
        )
        same_answer += int(
            train_rows.loc[left, "answer"] == train_rows.loc[right, "answer"]
        )
    report["train_duplicate_semantics"] = {
        "n_pairs": int(len(train_pairs)),
        "pairs_with_same_question": int(same_question),
        "pairs_with_same_answer": int(same_answer),
    }

    dev_votes = summarise_dev_votes(frames["dev"])
    dev_sha = report["splits"]["dev"]["csv_sha256"]
    dev_votes["csv_sha256"] = dev_sha
    dev_votes["csv_sha256_matches_plan"] = bool(dev_sha == PLAN_DEV_CSV_SHA256)
    report["dev_votes"] = dev_votes

    json_path = out_dir / "audit_report.json"
    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    write_markdown(report, out_dir / "audit_report.md")

    print(f"wrote {json_path}")
    print(f"wrote {out_dir / 'audit_report.md'}")
    print(f"wrote {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
