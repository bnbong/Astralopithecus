#!/usr/bin/env python3
"""대회 배포 dev 세트를 train과 같은 형식으로 바꾸어 학습 CSV 하나로 합친다.

대회가 나눠 준 `dev.csv` 에는 정답 열이 하나가 아니라 `answer1` ~ `answer5` 다섯 개가
있다. 사람 다섯 명이 각자 고른 값이고 빈 칸도 있어서 그대로는 학습에 쓸 수 없다. 팀원이
이 다섯 열을 다수결로 모아 `fold_csv_files/dev_aux.csv` 를 만들어 두었으므로, 이 스크립트는
그 파일을 라벨 원천으로 삼아 dev 문항을 train 형식으로 옮긴다.

출력은 `<data-root>/train_plus_dev.csv` 하나이며, 열 구성은 train.csv와 같은
`id, path, question, a, b, c, d, answer` 여덟 개다. train.csv의 전체 행을 순서 그대로 먼저
쓰고, 그 뒤에 dev_aux.csv에 있는 id의 dev 행을 dev_aux.csv의 순서대로 붙인다. 학습 CSV와
split JSON은 서로 독립이므로, 이 파일에 dev 행이 들어 있어도 split의 train_ids에 dev id가
없으면 그 문항은 학습되지 않는다.

`path` 는 dev.csv의 값을 그대로 옮긴다(`dev/dev_0001.jpg`). 이미지 경로를 `train/` 으로
고정한 곳이 없으므로 로더가 이 값을 그대로 쓴다. `id` 도 `dev_0001.jpg` 그대로 두어서 train
문항과 섞이지 않는다.

만들어진 CSV에는 질문 원문과 정답이 들어 있다. 데이터 파생물이므로 저장소에 올리지 않는다.
`.gitignore` 가 `train_plus_dev.csv` 를 막고 있다.

사용 예::

    python scripts/make_dev_train_csv.py --data-root /content/data
    python scripts/make_dev_train_csv.py --data-root /content/data \
        --dev-aux fold_csv_files/dev_aux.csv --out /content/data/train_plus_dev.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 출력 CSV의 열 순서. train.csv와 같아야 vqa.collator 가 그대로 읽는다.
TRAIN_COLUMNS = ("id", "path", "question", "a", "b", "c", "d", "answer")

#: dev.csv에서 가져다 쓰는 열. 정답은 dev_aux.csv의 다수결 값으로 따로 채운다.
DEV_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")

#: 정답으로 인정하는 값.
VALID_ANSWERS = ("a", "b", "c", "d")

#: dev_aux.csv에서 반드시 있어야 하는 열.
DEV_AUX_REQUIRED = ("id", "answer_majority")

DEFAULT_OUT_NAME = "train_plus_dev.csv"
DEFAULT_DEV_AUX = REPO_ROOT / "fold_csv_files" / "dev_aux.csv"


def read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """문자열 그대로 읽는다. 빈 칸은 NaN이 아니라 빈 문자열이 되어야 한다."""
    if not path.is_file():
        raise SystemExit(f"[오류] {label} 파일이 없습니다: {path}")
    frame = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig"
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.replace("\r", "", regex=False)
    return frame


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str, path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(
            f"[오류] {label}({path}): 필요한 열이 없습니다: {missing}. 있는 열: {list(frame.columns)}"
        )


def build_dev_rows(dev: pd.DataFrame, dev_aux: pd.DataFrame, *, dev_aux_path: Path) -> pd.DataFrame:
    """dev_aux의 id 순서대로 dev.csv 행을 train 형식으로 바꾼다."""
    dev_by_id = {str(row["id"]).strip(): row for _, row in dev.iterrows()}

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    bad_answer: list[str] = []
    seen: set[str] = set()
    for _, aux_row in dev_aux.iterrows():
        row_id = str(aux_row["id"]).strip()
        if not row_id:
            continue
        if row_id in seen:
            raise SystemExit(f"[오류] {dev_aux_path}: id가 중복됩니다: {row_id}")
        seen.add(row_id)

        source = dev_by_id.get(row_id)
        if source is None:
            missing.append(row_id)
            continue

        answer = str(aux_row["answer_majority"]).strip().lower()
        if answer not in VALID_ANSWERS:
            bad_answer.append(f"{row_id}={aux_row['answer_majority']!r}")
            continue

        record = {column: str(source[column]).strip() for column in DEV_COLUMNS}
        record["answer"] = answer
        rows.append(record)

    if missing:
        raise SystemExit(
            f"[오류] {dev_aux_path} 의 id {len(missing)}개가 dev.csv에 없습니다: {missing[:5]}"
        )
    if bad_answer:
        raise SystemExit(
            f"[오류] {dev_aux_path}: answer_majority 가 a~d가 아닌 행이 "
            f"{len(bad_answer)}개 있습니다: {bad_answer[:5]}"
        )
    return pd.DataFrame(rows, columns=list(TRAIN_COLUMNS))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="train.csv, dev.csv, dev/ 이미지가 있는 폴더. 출력도 이 폴더에 쓴다.",
    )
    parser.add_argument(
        "--dev-aux",
        default=str(DEFAULT_DEV_AUX),
        help="dev 다수결 정답 CSV. 'id' 와 'answer_majority' 두 열이 필요하다.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"출력 경로. 기본값은 <data-root>/{DEFAULT_OUT_NAME} 이다.",
    )
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=None,
        help="출력 행 수가 이 값과 다르면 실패한다. 예: --expect-rows 8738",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    dev_aux_path = Path(args.dev_aux)
    if not dev_aux_path.is_absolute():
        dev_aux_path = (Path.cwd() / dev_aux_path).resolve()
    out_path = Path(args.out).resolve() if args.out else data_root / DEFAULT_OUT_NAME

    train_path = data_root / "train.csv"
    dev_path = data_root / "dev.csv"

    train = read_csv(train_path, label="train.csv")
    dev = read_csv(dev_path, label="dev.csv")
    dev_aux = read_csv(dev_aux_path, label="dev_aux")

    require_columns(train, TRAIN_COLUMNS, label="train.csv", path=train_path)
    require_columns(dev, DEV_COLUMNS, label="dev.csv", path=dev_path)
    require_columns(dev_aux, DEV_AUX_REQUIRED, label="dev_aux", path=dev_aux_path)

    train_rows = train[list(TRAIN_COLUMNS)].copy()
    dev_rows = build_dev_rows(dev, dev_aux, dev_aux_path=dev_aux_path)

    overlap = sorted(set(train_rows["id"]) & set(dev_rows["id"]))
    if overlap:
        raise SystemExit(
            f"[오류] train.csv와 dev 행의 id가 {len(overlap)}개 겹칩니다: {overlap[:5]}"
        )

    merged = pd.concat([train_rows, dev_rows], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"train.csv     : {len(train_rows)}행 ({train_path})")
    print(f"dev 추가      : {len(dev_rows)}행 ({dev_aux_path.name} 기준)")
    print(f"합계          : {len(merged)}행")
    print(f"wrote {out_path}")

    if args.expect_rows is not None and len(merged) != args.expect_rows:
        raise SystemExit(
            f"[오류] 출력 행 수가 {len(merged)}인데 --expect-rows 는 {args.expect_rows} 입니다."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
