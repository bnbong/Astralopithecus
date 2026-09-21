#!/usr/bin/env python3
"""팀 공식 fold CSV 두 개를 vqa가 읽는 split JSON 하나로 바꾼다.

입력은 팀원이 만든 `fold_csv_files/val_fold0.csv`(1,344문항)와
`fold_csv_files/train_fold1234.csv`(5,358문항)다. 분할 방식은
`fold_csv_files/데이터처리_검증분할.md`에 정리되어 있다.

출력은 다음 두 파일이다.

    artifacts/data/split_fold0.json   vqa.train --split 이 읽는 형식
    artifacts/data/val_fold0_ids.txt  한 줄에 id 하나

두 파일 모두 id만 담고 있어서 저장소에 그대로 추적한다. 같은 입력으로 두 번 실행하면
`created` 줄 하나만 달라지며, 화면에 찍히는 content 해시는 그 줄을 빼고 계산한다.

사용 예::

    python scripts/folds_to_split.py
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

from vqa.data import file_sha256  # noqa: E402
from vqa.split import content_hash  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 이 split의 version 문자열. split_v1(검증 500문항)과 구분하기 위한 이름이다.
SPLIT_VERSION = "fold0"


def read_ids(path: Path, *, column: str = "id") -> list[str]:
    """id 컬럼 하나짜리 CSV를 읽는다. 중복이 있으면 실패한다."""
    if not path.exists():
        raise SystemExit(f"[오류] 입력 파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]
    if column not in frame.columns:
        raise SystemExit(f"[오류] {path}: '{column}' 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}")
    values = [str(value).strip() for value in frame[column].dropna().tolist()]
    values = [value for value in values if value]
    duplicated = sorted({value for value in values if values.count(value) > 1})
    if duplicated:
        raise SystemExit(f"[오류] {path}: id가 중복됩니다({len(duplicated)}개): {duplicated[:5]}")
    return values


def answer_counts(train_csv: Path, ids: list[str]) -> dict[str, int]:
    """train.csv를 읽을 수 있으면 선지별 정답 개수를 센다. 없으면 빈 dict."""
    if not train_csv.exists():
        return {}
    frame = pd.read_csv(train_csv, encoding="utf-8-sig", dtype="string")
    answer_of = {
        str(row_id).strip(): str(answer).strip().lower()
        for row_id, answer in zip(frame["id"], frame["answer"])
    }
    counts = Counter(answer_of[i] for i in ids if i in answer_of)
    return {letter: int(counts[letter]) for letter in ("a", "b", "c", "d")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir",
        default=str(REPO_ROOT / "fold_csv_files"),
        help="val_fold0.csv와 train_fold1234.csv가 있는 폴더",
    )
    parser.add_argument("--val-csv", default="val_fold0.csv")
    parser.add_argument("--train-csv", default="train_fold1234.csv")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT),
        help="train.csv가 있는 폴더. 있으면 sha256과 선지 분포를 함께 기록한다.",
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "data"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fold_dir = Path(args.fold_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = fold_dir / args.val_csv
    train_path = fold_dir / args.train_csv

    val_ids = read_ids(val_path)
    train_ids = read_ids(train_path)

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise SystemExit(
            f"[오류] 학습 id와 검증 id가 {len(overlap)}개 겹칩니다: {overlap[:5]}. "
            "fold 파일을 다시 확인하십시오."
        )

    data_root = Path(args.data_root).resolve()
    main_train_csv = data_root / "train.csv"
    train_csv_sha256 = file_sha256(main_train_csv) if main_train_csv.exists() else None
    if train_csv_sha256 is None:
        print(f"주의: {main_train_csv} 가 없어서 train_csv_sha256을 기록하지 못했습니다.")

    payload: dict[str, object] = {
        "version": SPLIT_VERSION,
        "source": "fold_csv_files (팀 공식 StratifiedGroupKFold 5-fold, fold 0 = 검증)",
        "source_files": {
            val_path.name: file_sha256(val_path),
            train_path.name: file_sha256(train_path),
        },
        "train_csv_sha256": train_csv_sha256,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_answer_counts": answer_counts(main_train_csv, train_ids),
        "val_answer_counts": answer_counts(main_train_csv, val_ids),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    digest = content_hash(payload)

    ordered: dict[str, object] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": digest,
    }
    ordered.update({key: value for key, value in sorted(payload.items())})

    split_path = out_dir / "split_fold0.json"
    split_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    val_ids_path = out_dir / "val_fold0_ids.txt"
    val_ids_path.write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")

    print(f"검증 fold 0 : {len(val_ids)}문항 ({val_path})")
    print(f"학습 fold 1~4: {len(train_ids)}문항 ({train_path})")
    print("겹치는 id    : 0개")
    print(f"val 선지 분포 : {payload['val_answer_counts']}")
    print(f"wrote {split_path}")
    print(f"wrote {val_ids_path}")
    print(f"split content sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
