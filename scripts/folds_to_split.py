#!/usr/bin/env python3
"""팀 공식 fold CSV 두 개를 vqa가 읽는 split JSON 하나로 바꾼다.

입력은 팀원이 만든 `fold_csv_files/val_fold0.csv`(1,344문항)와
`fold_csv_files/train_fold1234.csv`(5,358문항)다. 분할 방식은
`docs/submission_report.md` 2.3절에 정리되어 있다.

출력은 다음 세 파일이다.

    artifacts/data/split_fold0.json   vqa.train --split 이 읽는 형식 (학습 5,358문항)
    artifacts/data/split_full.json    fold 0까지 합친 전체 학습용 (학습 6,690문항)
    artifacts/data/val_fold0_ids.txt  한 줄에 id 하나

`split_full.json`은 fold 0에서 설정을 검증한 다음에 쓰는 최종 학습용 분할이다. 검증 목록이
비어 있으므로 이 분할로 학습한 모델은 fold 0 정확도로 평가할 수 없다.

`fold_csv_files/exclude_train_overlap_test.csv`에는 이미지가 test와 완전히 같은 train
24문항이 들어 있다. `train_fold1234.csv`는 이 24문항을 이미 빼 두었지만 `val_fold0.csv`
에는 12문항이 그대로 남아 있으므로, fold 0을 학습에 합칠 때 이 12문항을 다시 빼지 않으면
test와 겹치는 이미지를 학습하게 된다. 그래서 `--scope full`은 두 fold를 합친 다음에 제외
목록을 한 번 더 적용하며, 몇 문항을 뺐는지를 `excluded_overlap_test` 키에 기록한다.

세 파일 모두 id만 담고 있어서 저장소에 그대로 추적한다. 같은 입력으로 두 번 실행하면
`created` 줄 하나만 달라지며, 화면에 찍히는 content 해시는 그 줄을 빼고 계산한다.

`--scope fold --val-fold N` 은 fold N을 검증으로, 나머지 네 fold를 학습으로 하는
`artifacts/data/split_foldN.json` 을 만든다. 이 경로는 문항마다 fold 번호가 적힌 파일
하나를 필요로 한다. 위의 두 CSV에는 fold 0 목록과 fold 1~4 목록만 있고 1~4의 개별 배정이
없기 때문이다. `fold_csv_files/exclude_train_overlap_test.csv` 에 `fold` 열이 있기는 하지만
test와 겹치는 24문항에만 붙어 있어서 전체 배정으로 쓸 수 없다. 그래서 `--folds-csv` 로
`id,fold` 두 열짜리 파일을 받는다. 인자를 적지 않으면 `artifacts/data/train_folds.csv` 와
`fold_csv_files/folds_all.csv` 를 차례로 찾고, 둘 다 없으면 무엇을 받아야 하는지 안내하고
멈춘다.

fold 분할에도 test 중복 제외 정책을 그대로 적용한다. 제외 목록의 문항은 학습에서만 빼고
검증에는 남긴다. 팀이 만든 `train_fold1234.csv` 가 24문항 가운데 fold 1~4의 12문항을 빼
두었고 `val_fold0.csv` 는 fold 0의 12문항을 그대로 두었으므로, 같은 규칙을 모든 fold에
적용해야 fold 0 분할과 이어진다. 뺀 개수는 `excluded_overlap_test` 에 적는다.

사용 예::

    python scripts/folds_to_split.py                # 두 분할을 모두 만든다
    python scripts/folds_to_split.py --scope full   # split_full.json만 만든다
    python scripts/folds_to_split.py --scope fold --val-fold 2   # split_fold2.json을 만든다

`--with-dev fold_csv_files/dev_aux.csv` 를 붙이면 위의 파일을 그대로 만든 다음에, 학습 목록
뒤에 대회 배포 dev 세트의 id를 이어 붙인 판을 하나 더 쓴다. 이름은 원래 이름에 `_dev` 를 붙인
`split_fold0_dev.json`, `split_full_dev.json` 이다. 검증 목록은 건드리지 않으므로 fold 검증
점수는 dev를 넣기 전과 같은 문항으로 잰다. 몇 문항을 더했는지는 `dev_added` 에 적는다.

dev에도 test 중복 제외 정책을 그대로 적용한다. `fold_csv_files/exclude_dev_overlap_test.csv`
에는 이미지가 test와 완전히 같은 dev 4문항이 들어 있고, `--with-dev` 가 이 파일을 기본으로
읽어 학습 목록에서 뺀다. 뺀 개수와 id는 `dev_excluded_overlap_test` 와
`dev_excluded_overlap_test_ids` 에 적는다. 이 목록은 `--dev-exclude-csv` 로 바꿀 수 있다.
`train_plus_dev.csv` 에는 그 4문항이 그대로 남아 있지만, 학습에 쓰이는 문항은 split의
`train_ids` 가 정하므로 목록에 없으면 학습되지 않는다.

dev id는 `dev_0001.jpg` 형태여서 train id와 겹치지 않는다. 이 분할로 학습하려면
`scripts/make_dev_train_csv.py` 가 만드는 `train_plus_dev.csv` 를 `vqa.train --train-csv` 로
함께 넘겨야 한다. train.csv에는 dev 행이 없기 때문이다.
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

#: fold 0을 검증으로 빼 둔 분할의 version 문자열. split_v1(검증 500문항)과 구분하기 위한 이름이다.
SPLIT_VERSION = "fold0"

#: fold 0까지 학습에 넣은 전체 분할의 version 문자열.
FULL_SPLIT_VERSION = "full"

#: 팀 공식 분할의 fold 개수. --val-fold 가 받는 값의 범위를 정한다.
N_FOLDS = 5

#: 전체 fold 배정 파일을 찾을 자리. --folds-csv 를 적지 않았을 때 이 순서로 본다.
#: 첫째는 팀원이 준 artifacts/data/train_folds.csv (id, fold, overlap_test 세 열)이고,
#: 둘째는 fold_csv_files 아래에 직접 둔 파일이다.
DEFAULT_FOLDS_CSV = "folds_all.csv"
DEFAULT_FOLDS_CSV_IN_ARTIFACTS = "train_folds.csv"

#: --with-dev 가 기본으로 가리키는 자리. 팀원이 dev.csv의 answer1~5를 다수결로 모은 파일이다.
DEFAULT_DEV_AUX = "fold_csv_files/dev_aux.csv"

#: --with-dev 로 만든 분할을 쓸 때 필요한 학습 CSV. scripts/make_dev_train_csv.py 가 만든다.
DEV_TRAIN_CSV_NAME = "train_plus_dev.csv"

#: dev_aux.csv에서 다수결 정답이 적힌 열 이름.
DEV_ANSWER_COLUMN = "answer_majority"

#: 이미지가 test와 완전히 같아서 학습에서 빼야 하는 dev id 목록. --fold-dir 아래에서 찾는다.
#: MD5로 대조한 결과 dev_aux.csv의 2,024문항 가운데 4문항이 test 이미지와 같았다.
DEFAULT_DEV_EXCLUDE_CSV = "exclude_dev_overlap_test.csv"


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


def read_exclude_ids(path: Path) -> list[str]:
    """학습에서 빼야 하는 id 목록을 읽는다. 파일이 없으면 경고만 남기고 빈 목록을 돌려준다."""
    if not path.exists():
        print(f"주의: 제외 목록 {path} 가 없어서 test 중복 문항을 빼지 못했습니다.")
        return []
    return read_ids(path)


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
        "--exclude-csv",
        default="exclude_train_overlap_test.csv",
        help=(
            "test와 이미지가 완전히 같아서 학습에서 빼야 하는 train id 목록. "
            "--scope full에서만 적용한다. 파일이 없으면 경고만 남기고 넘어간다."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT),
        help="train.csv가 있는 폴더. 있으면 sha256과 선지 분포를 함께 기록한다.",
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "data"))
    parser.add_argument(
        "--scope",
        choices=("fold0", "full", "both", "fold"),
        default="both",
        help=(
            "만들 분할을 고른다. fold0은 split_fold0.json만, full은 split_full.json만, "
            "both는 둘 다 만든다. fold는 --val-fold 로 고른 fold 하나를 검증으로 하는 "
            "split_foldN.json을 만들며 --folds-csv 가 필요하다. 기본값은 both다."
        ),
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=None,
        help=f"--scope fold 에서 검증으로 쓸 fold 번호 (0~{N_FOLDS - 1})",
    )
    parser.add_argument(
        "--folds-csv",
        default=None,
        help=(
            "문항마다 fold 번호가 적힌 CSV. 'id' 와 'fold' 두 열이 필요하다. "
            f"적지 않으면 --out-dir 아래의 {DEFAULT_FOLDS_CSV_IN_ARTIFACTS} 와 "
            f"--fold-dir 아래의 {DEFAULT_FOLDS_CSV} 를 차례로 찾는다."
        ),
    )
    parser.add_argument(
        "--with-dev",
        default=None,
        metavar="DEV_AUX_CSV",
        help=(
            "대회 배포 dev 세트를 학습에 더한 판을 하나 더 만든다. 'id' 열이 있는 CSV를 받으며 "
            f"보통은 {DEFAULT_DEV_AUX} 다. 원래 파일은 그대로 두고 이름 뒤에 '_dev' 를 붙인 "
            "파일을 따로 쓴다. 검증 목록은 바꾸지 않는다."
        ),
    )
    parser.add_argument(
        "--dev-exclude-csv",
        default=DEFAULT_DEV_EXCLUDE_CSV,
        help=(
            "이미지가 test와 완전히 같아서 학습에서 빼야 하는 dev id 목록. --fold-dir 아래에서 "
            "찾는다. --with-dev 에서만 쓰며, 파일이 없으면 경고만 남기고 넘어간다."
        ),
    )
    return parser.parse_args()


def read_dev_ids(path: Path) -> list[str]:
    """--with-dev 가 가리키는 CSV에서 dev id를 읽는다. 순서는 파일에 적힌 그대로 둔다."""
    return read_ids(path)


def dev_answer_counts(path: Path) -> dict[str, int]:
    """dev_aux.csv에 answer_majority 열이 있으면 선지별 개수를 센다. 없으면 빈 dict."""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]
    if DEV_ANSWER_COLUMN not in frame.columns:
        return {}
    values = [str(value).strip().lower() for value in frame[DEV_ANSWER_COLUMN].dropna().tolist()]
    counts = Counter(values)
    return {letter: int(counts[letter]) for letter in ("a", "b", "c", "d")}


def write_dev_variant(
    out_dir: Path,
    name: str,
    payload: dict[str, object],
    dev_path: Path,
    dev_exclude_path: Path | None = None,
) -> None:
    """payload의 학습 목록 뒤에 dev id를 붙인 판을 '<name>_dev.json' 으로 쓴다.

    검증 목록은 건드리지 않는다. dev 문항을 학습에 더하더라도 fold 검증 점수는 dev를 넣기
    전과 같은 문항으로 재야 두 run을 비교할 수 있기 때문이다.

    train에 쓰던 test 중복 제외 정책을 dev에도 그대로 적용한다. dev 이미지 가운데 test와
    완전히 같은 것이 있으면 그 문항을 학습하는 순간 test 정답을 외운 셈이 되기 때문이다.
    뺀 id는 `dev_excluded_overlap_test_ids` 에 적는다.

    이미 같은 id 목록을 담은 파일이 있으면 다시 쓰지 않는다. 내용이 같아도 created 줄 때문에
    저장소에 diff가 남기 때문이다.
    """
    dev_ids = read_dev_ids(dev_path)
    dev_exclude_ids = (
        set(read_exclude_ids(dev_exclude_path)) if dev_exclude_path is not None else set()
    )
    dev_removed_ids = sorted(set(dev_ids) & dev_exclude_ids)
    dev_ids = [row_id for row_id in dev_ids if row_id not in dev_exclude_ids]
    base_train_ids = [str(i) for i in payload["train_ids"]]
    overlap = sorted(set(base_train_ids) & set(dev_ids))
    if overlap:
        raise SystemExit(
            f"[오류] {dev_path} 의 id {len(overlap)}개가 이미 학습 목록에 있습니다: {overlap[:5]}"
        )
    val_ids = [str(i) for i in payload["val_ids"]]
    val_overlap = sorted(set(val_ids) & set(dev_ids))
    if val_overlap:
        raise SystemExit(
            f"[오류] {dev_path} 의 id {len(val_overlap)}개가 검증 목록에 있습니다: {val_overlap[:5]}"
        )

    dev_payload = dict(payload)
    dev_payload["version"] = str(payload["version"]) + "_dev"
    dev_payload["source"] = str(payload["source"]) + " + 대회 배포 dev 세트"
    dev_source_files = {
        **dict(payload["source_files"]),  # type: ignore[arg-type]
        dev_path.name: file_sha256(dev_path),
    }
    if dev_exclude_path is not None and dev_exclude_path.exists():
        dev_source_files[dev_exclude_path.name] = file_sha256(dev_exclude_path)
    dev_payload["source_files"] = dev_source_files
    # 학습 목록 뒤에 붙인다. dev id는 'dev_' 로 시작해서 정렬하면 train id 앞으로 가지만,
    # 어느 문항이 나중에 더해진 것인지 파일만 보고 알 수 있도록 순서를 지킨다.
    dev_payload["train_ids"] = base_train_ids + dev_ids
    dev_payload["n_train"] = len(base_train_ids) + len(dev_ids)
    dev_payload["dev_added"] = len(dev_ids)
    dev_payload["dev_aux_file"] = dev_path.name
    dev_payload["dev_answer_counts"] = dev_answer_counts(dev_path)
    dev_payload["dev_excluded_overlap_test"] = len(dev_removed_ids)
    dev_payload["dev_excluded_overlap_test_ids"] = dev_removed_ids
    # 학습 CSV가 train.csv 하나가 아님을 분할 파일에도 남긴다. dev 행은 train.csv에 없다.
    dev_payload["train_csv_required"] = DEV_TRAIN_CSV_NAME

    dev_name = name[: -len(".json")] + "_dev.json" if name.endswith(".json") else name + "_dev"
    existing_path = out_dir / dev_name
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        same = (
            [str(i) for i in existing.get("train_ids", [])] == dev_payload["train_ids"]
            and [str(i) for i in existing.get("val_ids", [])] == val_ids
        )
        if not same:
            raise SystemExit(
                f"[오류] {existing_path} 가 이미 있고 id 목록이 다릅니다.\n"
                f"기존: 학습 {len(existing.get('train_ids', []))}문항 / 검증 "
                f"{len(existing.get('val_ids', []))}문항\n"
                f"지금: 학습 {dev_payload['n_train']}문항 / 검증 {len(val_ids)}문항\n"
                "입력 파일이 예전과 달라졌습니다. 정말 덮어써야 한다면 기존 파일을 먼저 지우십시오."
            )
        print(f"{existing_path} 가 이미 같은 id 목록이라 다시 쓰지 않습니다.")
        return

    print(
        f"dev 추가 학습 : {dev_payload['n_train']}문항 "
        f"(원래 {len(base_train_ids)} + dev {len(dev_ids)}, test 중복 dev 제외 "
        f"{len(dev_removed_ids)}), 검증 {len(val_ids)}문항 그대로"
    )
    print(
        f"이 분할로 학습하려면 `python scripts/make_dev_train_csv.py` 로 만든 "
        f"{DEV_TRAIN_CSV_NAME} 를 vqa.train --train-csv 로 넘기십시오."
    )
    write_split(out_dir, dev_name, dev_payload)


def read_fold_assignment(path: Path) -> dict[str, int]:
    """id -> fold 번호. 파일이 없으면 무엇을 받아야 하는지 안내하고 멈춘다."""
    if not path.exists():
        raise SystemExit(
            f"[오류] 전체 fold 배정 파일이 없습니다: {path}\n"
            "fold 1~4를 각각 검증으로 쓰려면 문항마다 fold 번호가 적힌 파일이 있어야 합니다. "
            "지금 저장소에 있는 것은 fold 0 목록(val_fold0.csv)과 fold 1~4를 합친 "
            "목록(train_fold1234.csv)뿐이고, exclude_train_overlap_test.csv 의 fold 열은 "
            "test와 겹치는 24문항에만 붙어 있어서 전체 배정으로 쓸 수 없습니다.\n"
            "분할을 만든 팀원에게 다음 형식의 CSV 하나를 요청하십시오.\n"
            "    id,fold\n"
            "    train_0001.jpg,3\n"
            "    train_0002.jpg,1\n"
            f"id는 train.csv 의 id 전체(6,714문항)이고 fold는 0~{N_FOLDS - 1} 입니다. "
            "StratifiedGroupKFold 를 돌린 코드에서 그대로 뽑을 수 있습니다. 받은 파일은 "
            f"--folds-csv 로 넘기거나 artifacts/data/{DEFAULT_FOLDS_CSV_IN_ARTIFACTS} 또는 "
            f"fold_csv_files/{DEFAULT_FOLDS_CSV} 자리에 두십시오."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    frame.columns = [str(col).strip() for col in frame.columns]
    for column in ("id", "fold"):
        if column not in frame.columns:
            raise SystemExit(
                f"[오류] {path}: '{column}' 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}"
            )
    assignment: dict[str, int] = {}
    for row_id, fold in zip(frame["id"], frame["fold"]):
        key = str(row_id).strip()
        if not key:
            continue
        try:
            number = int(str(fold).strip())
        except ValueError:
            raise SystemExit(f"[오류] {path}: fold 값이 정수가 아닙니다: {row_id}={fold!r}") from None
        if not 0 <= number < N_FOLDS:
            raise SystemExit(f"[오류] {path}: fold 값이 0~{N_FOLDS - 1} 밖입니다: {row_id}={number}")
        if key in assignment:
            raise SystemExit(f"[오류] {path}: id가 중복됩니다: {key}")
        assignment[key] = number
    if not assignment:
        raise SystemExit(f"[오류] {path}: 읽을 수 있는 행이 없습니다.")
    return assignment


def write_split(out_dir: Path, name: str, payload: dict[str, object]) -> str:
    """payload에 created와 content_sha256을 붙여 out_dir/name 으로 쓰고 해시를 돌려준다."""
    digest = content_hash(payload)
    ordered: dict[str, object] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": digest,
    }
    ordered.update({key: value for key, value in sorted(payload.items())})
    path = out_dir / name
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(f"{name} content sha256: {digest}")
    return digest


def build_fold_split(args: argparse.Namespace, fold_dir: Path, out_dir: Path) -> int:
    """fold N을 검증으로 하는 split_foldN.json 하나를 만든다."""
    if args.val_fold is None:
        raise SystemExit("[오류] --scope fold 에는 --val-fold 번호가 필요합니다. 예: --val-fold 2")
    val_fold = int(args.val_fold)
    if not 0 <= val_fold < N_FOLDS:
        raise SystemExit(f"[오류] --val-fold 는 0~{N_FOLDS - 1} 이어야 합니다: {val_fold}")

    if args.folds_csv:
        folds_path = Path(args.folds_csv).resolve()
    else:
        candidates = [out_dir / DEFAULT_FOLDS_CSV_IN_ARTIFACTS, fold_dir / DEFAULT_FOLDS_CSV]
        folds_path = next((path for path in candidates if path.exists()), candidates[0])
    assignment = read_fold_assignment(folds_path)

    present = sorted(set(assignment.values()))
    if present != list(range(N_FOLDS)):
        raise SystemExit(
            f"[오류] {folds_path}: fold 번호가 0~{N_FOLDS - 1} 을 모두 채우지 않습니다: {present}"
        )
    if val_fold not in present:
        raise SystemExit(f"[오류] {folds_path}: fold {val_fold} 에 해당하는 문항이 없습니다.")

    val_ids = sorted(row_id for row_id, fold in assignment.items() if fold == val_fold)
    rest_ids = sorted(row_id for row_id, fold in assignment.items() if fold != val_fold)

    # test 중복 제외는 학습에만 적용한다. 검증 목록은 그대로 둔다. 팀이 만든 fold 0 분할이
    # 같은 규칙을 따르고 있어서, 여기에서 규칙을 바꾸면 fold 사이의 비교가 어긋난다.
    exclude_path = fold_dir / args.exclude_csv
    exclude_ids = set(read_exclude_ids(exclude_path))
    removed_ids = sorted(set(rest_ids) & exclude_ids)
    train_ids = [row_id for row_id in rest_ids if row_id not in set(removed_ids)]

    data_root = Path(args.data_root).resolve()
    main_train_csv = data_root / "train.csv"
    train_csv_sha256 = file_sha256(main_train_csv) if main_train_csv.exists() else None
    if train_csv_sha256 is None:
        print(f"주의: {main_train_csv} 가 없어서 train_csv_sha256을 기록하지 못했습니다.")

    source_files = {folds_path.name: file_sha256(folds_path)}
    if exclude_path.exists():
        source_files[exclude_path.name] = file_sha256(exclude_path)

    print(f"fold 배정 파일: {folds_path} ({len(assignment)}문항)")
    print(f"검증 fold {val_fold}  : {len(val_ids)}문항")
    print(f"학습 나머지 fold: {len(train_ids)}문항 (test 중복 제외 {len(removed_ids)})")

    name = f"split_fold{val_fold}.json"
    payload: dict[str, object] = {
        "version": f"fold{val_fold}",
        "source": (
            f"fold_csv_files (팀 공식 StratifiedGroupKFold {N_FOLDS}-fold, fold {val_fold} = 검증)"
        ),
        "source_files": source_files,
        "train_csv_sha256": train_csv_sha256,
        "val_fold": val_fold,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_answer_counts": answer_counts(main_train_csv, train_ids),
        "val_answer_counts": answer_counts(main_train_csv, val_ids),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "excluded_overlap_test": len(removed_ids),
        "excluded_overlap_test_ids": removed_ids,
    }

    # fold 0은 팀이 이미 확정해서 저장소에 넣어 둔 분할이다. 배정 파일에서 다시 만든 목록이
    # 그 파일과 다르면 배정 파일이 팀 공식 분할과 어긋난 것이므로 멈춘다. 같으면 파일을 다시
    # 쓰지 않는다. 내용이 같아도 created 줄 때문에 저장소에 diff가 남기 때문이다.
    existing_path = out_dir / name
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        same = (
            [str(i) for i in existing.get("train_ids", [])] == sorted(train_ids)
            and [str(i) for i in existing.get("val_ids", [])] == sorted(val_ids)
        )
        if not same:
            raise SystemExit(
                f"[오류] {existing_path} 가 이미 있고 id 목록이 다릅니다.\n"
                f"기존: 학습 {len(existing.get('train_ids', []))}문항 / 검증 "
                f"{len(existing.get('val_ids', []))}문항\n"
                f"지금: 학습 {len(train_ids)}문항 / 검증 {len(val_ids)}문항\n"
                f"{folds_path} 의 fold 배정이 팀 공식 분할과 어긋납니다. 배정 파일을 다시 "
                "확인하시고, 정말 덮어써야 한다면 기존 파일을 먼저 지우십시오."
            )
        print(f"{existing_path} 가 이미 같은 id 목록이라 다시 쓰지 않습니다.")
        print(
            "선지 분포나 train_csv_sha256 을 새로 채워 넣으려면 그 파일을 지우고 train.csv 가 "
            "있는 자리에서 다시 실행하십시오."
        )
    else:
        print(f"val 선지 분포 : {payload['val_answer_counts']}")
        write_split(out_dir, name, payload)

        val_ids_path = out_dir / f"val_fold{val_fold}_ids.txt"
        val_ids_path.write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
        print(f"wrote {val_ids_path}")

    # dev 판은 원래 파일을 이미 만들어 둔 경우에도 따로 쓴다. 기존 파일이 있다는 이유로
    # 건너뛰면 --with-dev 를 나중에 붙였을 때 아무 일도 일어나지 않기 때문이다.
    if args.with_dev:
        write_dev_variant(
            out_dir,
            name,
            payload,
            Path(args.with_dev).resolve(),
            fold_dir / args.dev_exclude_csv,
        )
    return 0


def main() -> int:
    args = parse_args()
    fold_dir = Path(args.fold_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scope == "fold":
        return build_fold_split(args, fold_dir, out_dir)

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

    source_files = {
        val_path.name: file_sha256(val_path),
        train_path.name: file_sha256(train_path),
    }

    print(f"검증 fold 0 : {len(val_ids)}문항 ({val_path})")
    print(f"학습 fold 1~4: {len(train_ids)}문항 ({train_path})")
    print("겹치는 id    : 0개")

    if args.scope in ("fold0", "both"):
        fold0_payload: dict[str, object] = {
            "version": SPLIT_VERSION,
            "source": "fold_csv_files (팀 공식 StratifiedGroupKFold 5-fold, fold 0 = 검증)",
            "source_files": dict(source_files),
            "train_csv_sha256": train_csv_sha256,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "train_answer_counts": answer_counts(main_train_csv, train_ids),
            "val_answer_counts": answer_counts(main_train_csv, val_ids),
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids),
        }
        print(f"val 선지 분포 : {fold0_payload['val_answer_counts']}")
        write_split(out_dir, "split_fold0.json", fold0_payload)

        val_ids_path = out_dir / "val_fold0_ids.txt"
        val_ids_path.write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
        print(f"wrote {val_ids_path}")

        if args.with_dev:
            write_dev_variant(
                out_dir,
                "split_fold0.json",
                fold0_payload,
                Path(args.with_dev).resolve(),
                fold_dir / args.dev_exclude_csv,
            )

    if args.scope in ("full", "both"):
        merged_ids = sorted(set(train_ids) | set(val_ids))
        # fold 0에서 설정을 검증한 다음에만 쓰는 분할이라, 두 목록의 관계를 여기에서 한 번 더
        # 확인한다. 전체 목록이 fold 1~4와 fold 0의 합집합이 아니면 재현이 깨진 것이다.
        if len(merged_ids) != len(train_ids) + len(val_ids):
            raise SystemExit(
                f"[오류] 전체 id 수가 맞지 않습니다: {len(merged_ids)} != "
                f"{len(train_ids)} + {len(val_ids)}"
            )
        holdout = sorted(set(merged_ids) - set(train_ids))
        if holdout != sorted(val_ids):
            raise SystemExit(
                "[오류] 전체 목록에서 fold 1~4를 뺀 나머지가 fold 0과 다릅니다."
            )

        exclude_path = fold_dir / args.exclude_csv
        exclude_ids = read_exclude_ids(exclude_path)
        removed_ids = sorted(set(merged_ids) & set(exclude_ids))
        full_ids = [row_id for row_id in merged_ids if row_id not in set(removed_ids)]
        fold0_added = len([row_id for row_id in val_ids if row_id not in set(removed_ids)])

        full_source_files = dict(source_files)
        if exclude_path.exists():
            full_source_files[exclude_path.name] = file_sha256(exclude_path)

        full_payload: dict[str, object] = {
            "version": FULL_SPLIT_VERSION,
            "source": "fold_csv_files (팀 공식 StratifiedGroupKFold 5-fold, fold 0까지 학습에 포함)",
            "source_files": full_source_files,
            "train_csv_sha256": train_csv_sha256,
            "n_train": len(full_ids),
            "n_val": 0,
            "train_answer_counts": answer_counts(main_train_csv, full_ids),
            "val_answer_counts": {letter: 0 for letter in ("a", "b", "c", "d")},
            "train_ids": full_ids,
            "val_ids": [],
            "includes_fold0_val": True,
            "n_fold0_val_added": fold0_added,
            "excluded_overlap_test": len(removed_ids),
            "excluded_overlap_test_ids": removed_ids,
        }
        print()
        print(
            f"전체 학습    : {len(full_ids)}문항 "
            f"(fold 1~4 {len(train_ids)} + fold 0 {fold0_added}, "
            f"test 중복 제외 {len(removed_ids)})"
        )
        print("검증 목록    : 0문항. 이 분할로 학습한 모델은 fold 0 정확도로 평가할 수 없습니다.")
        write_split(out_dir, "split_full.json", full_payload)

        if args.with_dev:
            write_dev_variant(
                out_dir,
                "split_full.json",
                full_payload,
                Path(args.with_dev).resolve(),
                fold_dir / args.dev_exclude_csv,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
