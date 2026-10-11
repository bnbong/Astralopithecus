"""대회 배포 dev 세트를 학습 데이터로 쓰는 경로 전체의 테스트.

세 갈래를 지킨다. 첫째는 `scripts/make_dev_train_csv.py` 가 train 형식의 CSV를 제대로
만드는지, 둘째는 `scripts/folds_to_split.py --with-dev` 가 검증 목록을 건드리지 않고 학습
목록만 늘리는지, 셋째는 `vqa.train --train-csv` 가 그 CSV를 받아 dev id를 찾아내는지다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lab_qwen25vl_3b.yaml"

TRAIN_HEADER = "id,path,question,a,b,c,d,answer"
DEV_HEADER = "id,path,question,a,b,c,d,answer1,answer2,answer3,answer4,answer5"


# --------------------------------------------------------------------------
# 가짜 입력
# --------------------------------------------------------------------------
def write_fake_train(root: Path, n: int = 5) -> list[str]:
    """train.csv 와 같은 여덟 열짜리 CSV를 만든다. 돌려주는 값은 id 목록이다."""
    ids = ["train_%04d.jpg" % i for i in range(1, n + 1)]
    lines = [TRAIN_HEADER]
    for index, row_id in enumerate(ids):
        answer = "abcd"[index % 4]
        lines.append(f"{row_id},train/{row_id},질문{index},보기A,보기B,보기C,보기D,{answer}")
    (root / "train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


def write_fake_dev(root: Path, n: int = 4) -> list[str]:
    """dev.csv 와 같은 열두 열짜리 CSV를 만든다. answer1~5에는 빈 칸도 섞는다."""
    ids = ["dev_%04d.jpg" % i for i in range(1, n + 1)]
    lines = [DEV_HEADER]
    for index, row_id in enumerate(ids):
        votes = ["a", "a", "", "b", "a"] if index % 2 == 0 else ["c", "c", "c", "", "d"]
        lines.append(
            f"{row_id},dev/{row_id},dev질문{index},선지A,선지B,선지C,선지D," + ",".join(votes)
        )
    (root / "dev.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


def write_fake_dev_aux(path: Path, answers: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,answer_majority,n_votes,top_votes"]
    for row_id, answer in answers.items():
        lines.append(f"{row_id},{answer},4,3")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dev_exclude_csv(fold_dir: Path, ids: list[str]) -> Path:
    """이미지가 test와 같아서 학습에서 빼야 하는 dev id 목록을 만든다."""
    path = fold_dir / "exclude_dev_overlap_test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id\n" + "".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")
    return path


def run_make_dev_train_csv(argv: list[str]) -> int:
    from scripts import make_dev_train_csv as cli

    return cli.main(argv)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# make_dev_train_csv.py
# --------------------------------------------------------------------------
def test_dev_rows_get_the_train_schema_and_the_majority_answer(tmp_path: Path) -> None:
    train_ids = write_fake_train(tmp_path, 5)
    dev_ids = write_fake_dev(tmp_path, 4)
    aux = write_fake_dev_aux(
        tmp_path / "aux" / "dev_aux.csv", {dev_ids[0]: "a", dev_ids[2]: "C"}
    )

    assert run_make_dev_train_csv(["--data-root", str(tmp_path), "--dev-aux", str(aux)]) == 0

    rows = read_csv_rows(tmp_path / "train_plus_dev.csv")
    assert list(rows[0]) == TRAIN_HEADER.split(",")
    assert len(rows) == len(train_ids) + 2
    # train 행이 먼저, dev 행이 뒤에 온다.
    assert [row["id"] for row in rows[: len(train_ids)]] == train_ids
    added = rows[len(train_ids) :]
    assert [row["id"] for row in added] == [dev_ids[0], dev_ids[2]]
    # path는 dev.csv의 값을 그대로 옮긴다. 이미지는 dev/ 아래에 있다.
    assert [row["path"] for row in added] == [f"dev/{dev_ids[0]}", f"dev/{dev_ids[2]}"]
    # 정답은 dev_aux의 다수결 값이며 소문자로 맞춘다.
    assert [row["answer"] for row in added] == ["a", "c"]
    assert added[0]["question"] == "dev질문0"
    assert added[0]["a"] == "선지A"


def test_expect_rows_catches_a_wrong_total(tmp_path: Path) -> None:
    write_fake_train(tmp_path, 5)
    dev_ids = write_fake_dev(tmp_path, 4)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {dev_ids[0]: "a"})

    with pytest.raises(SystemExit, match="--expect-rows"):
        run_make_dev_train_csv(
            ["--data-root", str(tmp_path), "--dev-aux", str(aux), "--expect-rows", "99"]
        )


def test_unknown_dev_id_is_refused(tmp_path: Path) -> None:
    write_fake_train(tmp_path, 3)
    write_fake_dev(tmp_path, 2)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {"dev_9999.jpg": "a"})

    with pytest.raises(SystemExit, match="dev.csv에 없습니다"):
        run_make_dev_train_csv(["--data-root", str(tmp_path), "--dev-aux", str(aux)])


def test_answer_outside_abcd_is_refused(tmp_path: Path) -> None:
    write_fake_train(tmp_path, 3)
    dev_ids = write_fake_dev(tmp_path, 2)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {dev_ids[0]: "e"})

    with pytest.raises(SystemExit, match="answer_majority"):
        run_make_dev_train_csv(["--data-root", str(tmp_path), "--dev-aux", str(aux)])


def test_output_csv_is_ignored_by_git() -> None:
    """만들어진 학습 CSV에는 질문 원문과 정답이 들어 있어 저장소에 올리면 안 된다."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "some/where/train_plus_dev.csv"],
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, ".gitignore가 train_plus_dev.csv를 막지 않습니다."


# --------------------------------------------------------------------------
# folds_to_split.py --with-dev
# --------------------------------------------------------------------------
def test_with_dev_appends_dev_ids_and_keeps_validation(tmp_path: Path) -> None:
    from tests.test_data_split import (
        make_assignment,
        run_folds_to_split,
        write_fold_files,
        write_folds_csv,
    )

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    aux = write_fake_dev_aux(
        tmp_path / "dev_aux.csv", {"dev_0001.jpg": "a", "dev_0002.jpg": "b", "dev_0003.jpg": "b"}
    )
    out_dir = tmp_path / "out"

    extra = ["--val-fold", "2", "--folds-csv", str(folds_csv), "--with-dev", str(aux)]
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=extra) == 0

    base = json.loads((out_dir / "split_fold2.json").read_text(encoding="utf-8"))
    dev = json.loads((out_dir / "split_fold2_dev.json").read_text(encoding="utf-8"))

    # 기존 파일은 그대로다.
    assert base["n_train"] == 24 and base["n_val"] == 6
    assert not any(str(i).startswith("dev_") for i in base["train_ids"])

    assert dev["version"] == "fold2_dev"
    assert dev["dev_added"] == 3
    assert dev["dev_aux_file"] == "dev_aux.csv"
    assert dev["train_csv_required"] == "train_plus_dev.csv"
    assert dev["dev_answer_counts"] == {"a": 1, "b": 2, "c": 0, "d": 0}
    # 학습 목록은 원래 목록 뒤에 dev id를 붙인 것이다.
    assert dev["train_ids"] == base["train_ids"] + [
        "dev_0001.jpg",
        "dev_0002.jpg",
        "dev_0003.jpg",
    ]
    assert dev["n_train"] == base["n_train"] + 3
    # 검증 목록은 그대로여야 두 run의 점수를 비교할 수 있다.
    assert dev["val_ids"] == base["val_ids"]
    assert dev["n_val"] == base["n_val"]
    assert "dev_aux.csv" in dev["source_files"]


def test_with_dev_writes_a_full_variant(tmp_path: Path) -> None:
    from tests.test_data_split import run_folds_to_split, write_fold_files

    fold_dir = write_fold_files(
        tmp_path, ["train_0001.jpg", "train_0002.jpg"], ["train_0003.jpg"], exclude_ids=[]
    )
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {"dev_0001.jpg": "a", "dev_0002.jpg": "d"})
    out_dir = tmp_path / "out"

    assert (
        run_folds_to_split(
            fold_dir, tmp_path, out_dir, scope="full", extra=["--with-dev", str(aux)]
        )
        == 0
    )
    full = json.loads((out_dir / "split_full.json").read_text(encoding="utf-8"))
    dev = json.loads((out_dir / "split_full_dev.json").read_text(encoding="utf-8"))
    assert dev["version"] == "full_dev"
    assert dev["n_train"] == full["n_train"] + 2
    assert dev["val_ids"] == [] and dev["n_val"] == 0
    assert dev["dev_added"] == 2


def test_with_dev_does_not_rewrite_an_unchanged_file(tmp_path: Path) -> None:
    """같은 입력으로 두 번 실행해도 파일이 바뀌지 않는다. created 줄만으로도 diff가 남는다."""
    from tests.test_data_split import (
        make_assignment,
        run_folds_to_split,
        write_fold_files,
        write_folds_csv,
    )

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {"dev_0001.jpg": "a"})
    out_dir = tmp_path / "out"
    extra = ["--val-fold", "3", "--folds-csv", str(folds_csv), "--with-dev", str(aux)]

    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=extra) == 0
    path = out_dir / "split_fold3_dev.json"
    first = path.read_text(encoding="utf-8")

    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=extra) == 0
    assert path.read_text(encoding="utf-8") == first


def test_with_dev_refuses_ids_that_are_already_in_the_split(tmp_path: Path) -> None:
    from tests.test_data_split import (
        make_assignment,
        run_folds_to_split,
        write_fold_files,
        write_folds_csv,
    )

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    in_train = sorted(i for i, fold in assignment.items() if fold != 2)[0]
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {in_train: "a"})

    with pytest.raises(SystemExit, match="이미 학습 목록에 있습니다"):
        run_folds_to_split(
            fold_dir,
            tmp_path,
            tmp_path / "out",
            scope="fold",
            extra=["--val-fold", "2", "--folds-csv", str(folds_csv), "--with-dev", str(aux)],
        )


def test_committed_fold0_dev_split_matches_the_committed_fold0_split() -> None:
    """저장소에 남긴 split_fold0_dev.json이 fold 0 분할과 dev_aux.csv의 합인지 확인한다."""
    base_path = ROOT / "artifacts" / "data" / "split_fold0.json"
    dev_path = ROOT / "artifacts" / "data" / "split_fold0_dev.json"
    if not dev_path.exists():
        pytest.skip("artifacts/data/split_fold0_dev.json이 아직 없습니다.")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))

    assert dev["n_val"] == 1344
    assert dev["val_ids"] == base["val_ids"]
    assert dev["n_train"] == 7378
    assert dev["dev_added"] == 2020
    assert dev["dev_excluded_overlap_test"] == 4
    assert dev["train_ids"][: base["n_train"]] == base["train_ids"]

    added = dev["train_ids"][base["n_train"] :]
    assert len(added) == 2020
    assert all(str(row_id).startswith("dev_") for row_id in added)
    assert not set(added) & set(dev["val_ids"])

    import csv as csv_mod

    def read_id_column(path: Path) -> list[str]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [row["id"].strip() for row in csv_mod.DictReader(handle)]

    aux_path = ROOT / "fold_csv_files" / "dev_aux.csv"
    if not aux_path.exists():
        pytest.skip("fold_csv_files/dev_aux.csv는 공개 저장소에 포함하지 않아 로컬에만 있습니다.")
    aux_ids = read_id_column(aux_path)
    excluded = read_id_column(ROOT / "fold_csv_files" / "exclude_dev_overlap_test.csv")
    assert dev["dev_excluded_overlap_test_ids"] == sorted(excluded)
    # 제외 목록의 문항만 빠지고 나머지는 dev_aux.csv의 순서를 지킨다.
    assert added == [row_id for row_id in aux_ids if row_id not in set(excluded)]
    assert not set(added) & set(excluded)


def test_committed_dev_exclude_list_holds_only_dev_ids() -> None:
    """MD5로 대조한 dev-test 중복 목록이다. train id가 섞여 들어가면 안 된다."""
    import csv as csv_mod

    path = ROOT / "fold_csv_files" / "exclude_dev_overlap_test.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv_mod.DictReader(handle))
    ids = [row["id"].strip() for row in rows]
    assert ids == ["dev_0171.jpg", "dev_1580.jpg", "dev_2046.jpg", "dev_2080.jpg"]
    assert all(row_id.startswith("dev_") for row_id in ids)


# --------------------------------------------------------------------------
# vqa.train --train-csv
# --------------------------------------------------------------------------
def test_train_csv_alias_points_at_the_same_option() -> None:
    from vqa import train as train_mod

    parser = train_mod.build_parser()
    common = ["--config", str(CONFIG), "--run-id", "r"]
    assert parser.parse_args([*common, "--train-csv", "train_plus_dev.csv"]).csv == (
        "train_plus_dev.csv"
    )
    # 기존 이름도 그대로 받는다.
    assert parser.parse_args([*common, "--csv", "other.csv"]).csv == "other.csv"
    # 적지 않으면 기본값은 None이고, main이 DATA_ROOT/train.csv 로 채운다.
    assert parser.parse_args(common).csv is None


def test_train_reads_dev_rows_from_a_relative_train_csv(tmp_path: Path, monkeypatch) -> None:
    """--train-csv 에 파일 이름만 적어도 DATA_ROOT 아래에서 찾고, dev id를 학습에 넣는다."""
    from vqa import train as train_mod

    data_root = tmp_path / "data"
    data_root.mkdir()
    train_ids = write_fake_train(data_root, 4)
    dev_ids = write_fake_dev(data_root, 2)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {dev_ids[0]: "b"})
    assert run_make_dev_train_csv(["--data-root", str(data_root), "--dev-aux", str(aux)]) == 0

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train_ids": train_ids + [dev_ids[0]], "val_ids": []}), encoding="utf-8"
    )

    seen: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, **kwargs):
            seen["ids"] = list(kwargs["ids"])
            seen["records"] = kwargs["records"]
            seen["data_root"] = kwargs["data_root"]

        def build(self):
            raise SystemExit("여기까지만 확인한다")

    monkeypatch.setattr(train_mod, "VLMBackend", FakeBackend)
    monkeypatch.setenv("VQA_DATA_ROOT", str(data_root))

    with pytest.raises(SystemExit, match="여기까지만"):
        train_mod.main(
            [
                "--config",
                str(CONFIG),
                "--run-id",
                "dev_csv_probe",
                "--run-root",
                str(tmp_path / "runs"),
                "--split",
                str(split_path),
                "--train-csv",
                "train_plus_dev.csv",
            ]
        )

    assert seen["ids"] == train_ids + [dev_ids[0]]
    # 이미지 경로는 CSV의 path 열을 그대로 쓴다. dev 이미지는 dev/ 아래에 있다.
    assert seen["records"][dev_ids[0]].path == f"dev/{dev_ids[0]}"


def test_train_explains_a_missing_dev_row(tmp_path: Path, monkeypatch) -> None:
    """_dev 분할을 train.csv와 함께 넘기면 무엇이 잘못됐는지 알려 주고 멈춘다."""
    from vqa import train as train_mod

    data_root = tmp_path / "data"
    data_root.mkdir()
    train_ids = write_fake_train(data_root, 3)

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train_ids": train_ids + ["dev_0001.jpg"], "val_ids": []}), encoding="utf-8"
    )
    monkeypatch.setenv("VQA_DATA_ROOT", str(data_root))

    with pytest.raises(SystemExit, match="make_dev_train_csv"):
        train_mod.main(
            [
                "--config",
                str(CONFIG),
                "--run-id",
                "dev_csv_missing",
                "--run-root",
                str(tmp_path / "runs"),
                "--split",
                str(split_path),
            ]
        )


def test_with_dev_drops_dev_images_that_overlap_test(tmp_path: Path) -> None:
    """dev 이미지가 test와 같으면 학습에서 뺀다. train에 쓰던 정책을 그대로 적용한 것이다."""
    from tests.test_data_split import (
        make_assignment,
        run_folds_to_split,
        write_fold_files,
        write_folds_csv,
    )

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    aux = write_fake_dev_aux(
        tmp_path / "dev_aux.csv",
        {"dev_0001.jpg": "a", "dev_0002.jpg": "b", "dev_0003.jpg": "c", "dev_0004.jpg": "d"},
    )
    write_dev_exclude_csv(fold_dir, ["dev_0002.jpg", "dev_0004.jpg"])
    out_dir = tmp_path / "out"

    extra = ["--val-fold", "1", "--folds-csv", str(folds_csv), "--with-dev", str(aux)]
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=extra) == 0

    dev = json.loads((out_dir / "split_fold1_dev.json").read_text(encoding="utf-8"))
    assert dev["dev_added"] == 2
    assert dev["dev_excluded_overlap_test"] == 2
    assert dev["dev_excluded_overlap_test_ids"] == ["dev_0002.jpg", "dev_0004.jpg"]
    assert dev["train_ids"][-2:] == ["dev_0001.jpg", "dev_0003.jpg"]
    for row_id in ("dev_0002.jpg", "dev_0004.jpg"):
        assert row_id not in dev["train_ids"]
    # 선지 분포는 dev_aux.csv 전체를 센 값이라 제외 전 개수 그대로다.
    assert sum(dev["dev_answer_counts"].values()) == 4


def test_dev_exclude_list_can_be_pointed_elsewhere(tmp_path: Path) -> None:
    from tests.test_data_split import (
        make_assignment,
        run_folds_to_split,
        write_fold_files,
        write_folds_csv,
    )

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    aux = write_fake_dev_aux(tmp_path / "dev_aux.csv", {"dev_0001.jpg": "a", "dev_0002.jpg": "b"})
    # 기본 이름의 파일에는 아무것도 없고, 따로 둔 파일이 한 건을 뺀다.
    write_dev_exclude_csv(fold_dir, [])
    (fold_dir / "other_exclude.csv").write_text("id\ndev_0001.jpg\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    extra = [
        "--val-fold", "4", "--folds-csv", str(folds_csv), "--with-dev", str(aux),
        "--dev-exclude-csv", "other_exclude.csv",
    ]
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=extra) == 0
    dev = json.loads((out_dir / "split_fold4_dev.json").read_text(encoding="utf-8"))
    assert dev["dev_excluded_overlap_test_ids"] == ["dev_0001.jpg"]
    assert dev["train_ids"][-1:] == ["dev_0002.jpg"]
