"""Tests for vqa.data and vqa.split on small synthetic datasets."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa import data as data_mod  # noqa: E402
from vqa.data import (  # noqa: E402
    IMAGEHASH_HINT,
    SchemaError,
    compute_fingerprints,
    file_sha256,
    get_fingerprints,
    image_path,
    load_csv,
    require_imagehash,
)
from vqa.split import (  # noqa: E402
    UnionFind,
    build_groups,
    canonical_json,
    content_hash,
    exact_duplicate_pairs,
    near_duplicate_pairs,
    split_payload,
    stratified_group_split,
)

LETTERS = ("a", "b", "c", "d")


# --------------------------------------------------------------------------- #
# synthetic data helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_image(path: Path, seed: int, size: tuple[int, int] = (64, 64)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            value = (x * 7 + y * 13 + seed * 29) % 256
            pixels[x, y] = (value, (value * 3) % 256, (value * 5 + seed) % 256)
    image.save(path, format="JPEG", quality=95)


@pytest.fixture()
def tiny_root(tmp_path: Path) -> Path:
    """Build a tiny DATA_ROOT with train, dev and test CSV files and images."""
    n_train = 24
    train_rows = []
    for index in range(n_train):
        row_id = f"train_{index:04d}.jpg"
        # 4개 이미지마다 같은 seed를 사용하여 중복 그룹이 생기게 한다.
        seed = index if index >= 8 else index // 4
        _make_image(tmp_path / "train" / row_id, seed)
        train_rows.append(
            [
                row_id,
                f"train/{row_id}",
                f"질문 {index % 5}",
                "opt-a",
                "opt-b",
                "opt-c",
                "opt-d",
                LETTERS[index % 4],
            ]
        )
    _write_csv(
        tmp_path / "train.csv",
        ["id", "path", "question", "a", "b", "c", "d", "answer"],
        train_rows,
    )

    dev_rows = []
    for index in range(6):
        row_id = f"dev_{index:04d}.jpg"
        _make_image(tmp_path / "dev" / row_id, 100 + index)
        votes = ["a", "a", "", "b", "c"] if index % 2 else ["b", "b", "b", "", ""]
        dev_rows.append(
            [row_id, f"dev/{row_id}", f"질문 {index}", "a1", "b1", "c1", "d1", *votes]
        )
    _write_csv(
        tmp_path / "dev.csv",
        [
            "id",
            "path",
            "question",
            "a",
            "b",
            "c",
            "d",
            "answer1",
            "answer2",
            "answer3",
            "answer4",
            "answer5",
        ],
        dev_rows,
    )

    test_rows = []
    for index in range(6):
        row_id = f"test_{index:04d}.jpg"
        # test_0000은 train_0000과 같은 seed를 사용하므로 교차 중복이 된다.
        _make_image(tmp_path / "test" / row_id, 0 if index == 0 else 200 + index)
        test_rows.append(
            [row_id, f"test/{row_id}", f"질문 {index}", "a1", "b1", "c1", "d1"]
        )
    _write_csv(
        tmp_path / "test.csv",
        ["id", "path", "question", "a", "b", "c", "d"],
        test_rows,
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# vqa.data
# --------------------------------------------------------------------------- #
def test_load_csv_preserves_ids_and_paths(tiny_root: Path) -> None:
    train = load_csv(tiny_root, "train")
    assert len(train) == 24
    assert train["id"].iloc[0] == "train_0000.jpg"
    assert train["path"].iloc[0] == "train/train_0000.jpg"
    # The relative path is never rewritten in the dataframe itself.
    assert "/" in train["path"].iloc[0]


def test_image_path_joins_without_hardcoded_separator(tiny_root: Path) -> None:
    resolved = image_path(tiny_root, "train/train_0000.jpg")
    assert resolved == tiny_root / "train" / "train_0000.jpg"
    assert resolved.is_file()
    # Backslash separated values resolve to the same file.
    assert image_path(tiny_root, "train\\train_0000.jpg") == resolved


def test_load_csv_handles_bom_and_blank_dev_answers(tiny_root: Path) -> None:
    dev = load_csv(tiny_root, "dev")
    assert list(dev.columns)[0] == "id"
    assert (dev["answer3"] == "").any()


def test_file_sha256_matches_hashlib(tiny_root: Path) -> None:
    import hashlib

    path = tiny_root / "train" / "train_0000.jpg"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_sha256(path) == expected


def test_missing_column_raises(tiny_root: Path) -> None:
    rows = [["train_0000.jpg", "train/train_0000.jpg", "질문", "a", "b", "c", "d"]]
    _write_csv(
        tiny_root / "train.csv",
        ["id", "path", "question", "a", "b", "c", "d"],
        rows,
    )
    with pytest.raises(SchemaError, match="missing required columns"):
        load_csv(tiny_root, "train")


def test_bad_answer_letter_raises(tiny_root: Path) -> None:
    rows = [
        ["train_0000.jpg", "train/train_0000.jpg", "질문", "a", "b", "c", "d", "e"]
    ]
    _write_csv(
        tiny_root / "train.csv",
        ["id", "path", "question", "a", "b", "c", "d", "answer"],
        rows,
    )
    with pytest.raises(SchemaError, match="answer must be one of"):
        load_csv(tiny_root, "train")


def test_duplicate_id_raises(tiny_root: Path) -> None:
    rows = [
        ["train_0000.jpg", "train/train_0000.jpg", "질문", "a", "b", "c", "d", "a"],
        ["train_0000.jpg", "train/train_0000.jpg", "질문", "a", "b", "c", "d", "b"],
    ]
    _write_csv(
        tiny_root / "train.csv",
        ["id", "path", "question", "a", "b", "c", "d", "answer"],
        rows,
    )
    with pytest.raises(SchemaError, match="duplicate id"):
        load_csv(tiny_root, "train")


def test_dev_answer_column_rejects_free_text(tiny_root: Path) -> None:
    rows = [
        [
            "dev_0000.jpg",
            "dev/dev_0000.jpg",
            "질문",
            "a1",
            "b1",
            "c1",
            "d1",
            "모름",
            "",
            "",
            "",
            "",
        ]
    ]
    _write_csv(
        tiny_root / "dev.csv",
        [
            "id",
            "path",
            "question",
            "a",
            "b",
            "c",
            "d",
            "answer1",
            "answer2",
            "answer3",
            "answer4",
            "answer5",
        ],
        rows,
    )
    with pytest.raises(SchemaError, match="must be blank or one of"):
        load_csv(tiny_root, "dev")


def test_unknown_split_name_raises(tiny_root: Path) -> None:
    with pytest.raises(SchemaError, match="unknown split name"):
        load_csv(tiny_root, "validation")


def test_fingerprint_cache_is_reused(tiny_root: Path, tmp_path: Path) -> None:
    frames = {name: load_csv(tiny_root, name) for name in ("train", "dev", "test")}
    cache = tmp_path / "cache" / "image_hashes.csv"
    first = get_fingerprints(tiny_root, frames, cache, workers=1)
    assert cache.is_file()
    assert len(first) == 36
    assert bool(first["readable"].all())
    second = get_fingerprints(tiny_root, frames, cache, workers=1)
    assert first["sha256"].tolist() == second["sha256"].tolist()
    assert first["phash"].tolist() == second["phash"].tolist()


def test_missing_image_is_reported(tiny_root: Path, tmp_path: Path) -> None:
    (tiny_root / "train" / "train_0000.jpg").unlink()
    frames = {"train": load_csv(tiny_root, "train")}
    fingerprints = get_fingerprints(
        tiny_root, frames, tmp_path / "hashes.csv", workers=1
    )
    row = fingerprints[fingerprints["id"] == "train_0000.jpg"].iloc[0]
    assert not bool(row["exists"])
    assert row["error"] == "missing"


def test_missing_imagehash_raises_instead_of_marking_images_unreadable(
    monkeypatch: pytest.MonkeyPatch, tiny_root: Path, tmp_path: Path
) -> None:
    """A missing package must be a loud setup error, not a silent empty report.

    Before this guard the import sat inside the per-image ``except Exception``,
    so every row came back ``readable=False`` with an empty phash and near
    duplicate detection quietly found nothing.
    """
    # ``import imagehash`` fails with ImportError when the entry is None.
    monkeypatch.setitem(sys.modules, "imagehash", None)

    with pytest.raises(ImportError) as excinfo:
        require_imagehash()
    assert "pip install ImageHash" in str(excinfo.value)
    assert IMAGEHASH_HINT in str(excinfo.value)

    frames = {name: load_csv(tiny_root, name) for name in ("train",)}
    with pytest.raises(ImportError, match="ImageHash"):
        compute_fingerprints(tiny_root, frames, workers=1)
    with pytest.raises(ImportError, match="ImageHash"):
        get_fingerprints(tiny_root, frames, tmp_path / "hashes.csv", workers=1)


def test_unreadable_image_stays_a_per_image_finding(tmp_path: Path) -> None:
    """A broken file is still just one row with an error, not a crash."""
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "train" / "train_0000.jpg").write_bytes(b"not an image")

    record = data_mod._fingerprint_one(
        ("train", "train_0000.jpg", "train/train_0000.jpg", str(root))
    )
    assert record["exists"] is True
    assert record["readable"] is False
    assert record["sha256"]
    assert record["phash"] == ""
    assert record["error"]


# --------------------------------------------------------------------------- #
# vqa.split
# --------------------------------------------------------------------------- #
def test_union_find_components() -> None:
    union = UnionFind(["a", "b", "c", "d"])
    union.union("a", "b")
    union.union("b", "c")
    components = union.components()
    assert sorted(len(members) for members in components.values()) == [1, 3]


def test_build_groups_is_order_independent() -> None:
    ids = ["x1", "x2", "x3", "x4"]
    left = build_groups(ids, [("x1", "x2"), ("x2", "x3")])
    right = build_groups(ids, [("x3", "x2"), ("x2", "x1")])
    assert left == right
    assert left["x1"] == left["x2"] == left["x3"]
    assert left["x4"] != left["x1"]


def test_duplicate_detection_finds_identical_images(
    tiny_root: Path, tmp_path: Path
) -> None:
    frames = {name: load_csv(tiny_root, name) for name in ("train", "dev", "test")}
    fingerprints = get_fingerprints(
        tiny_root, frames, tmp_path / "hashes.csv", workers=1
    )
    train_fp = fingerprints[fingerprints["split"] == "train"].reset_index(drop=True)
    exact = exact_duplicate_pairs(train_fp)
    # train_0000..0003 and train_0004..0007 are two groups of four identical files.
    assert len(exact) == 12
    assert bool(exact["same_sha256"].all())

    near = near_duplicate_pairs(train_fp, threshold=0)
    assert len(near) >= 12

    test_fp = fingerprints[fingerprints["split"] == "test"].reset_index(drop=True)
    cross = near_duplicate_pairs(train_fp, test_fp, threshold=0)
    assert "test_0000.jpg" in set(cross["id_b"].tolist())


def test_split_keeps_groups_together_and_is_complete(
    tiny_root: Path, tmp_path: Path
) -> None:
    train = load_csv(tiny_root, "train")
    frames = {"train": train}
    fingerprints = get_fingerprints(
        tiny_root, frames, tmp_path / "hashes.csv", workers=1
    )
    exact = exact_duplicate_pairs(fingerprints)
    pairs = [(str(a), str(b)) for a, b in zip(exact["id_a"], exact["id_b"])]
    ids = [str(value) for value in train["id"].tolist()]
    group_of = build_groups(ids, pairs)

    result = stratified_group_split(train, group_of, target_size=8, seed=7)

    assert set(result.train_ids) & set(result.val_ids) == set()
    assert set(result.train_ids) | set(result.val_ids) == set(ids)
    assert len(result.train_ids) + len(result.val_ids) == len(ids)

    val_groups = {group_of[row_id] for row_id in result.val_ids}
    train_groups = {group_of[row_id] for row_id in result.train_ids}
    assert val_groups & train_groups == set()


def test_split_is_deterministic(tiny_root: Path, tmp_path: Path) -> None:
    train = load_csv(tiny_root, "train")
    fingerprints = get_fingerprints(
        tiny_root, {"train": train}, tmp_path / "hashes.csv", workers=1
    )
    exact = exact_duplicate_pairs(fingerprints)
    pairs = [(str(a), str(b)) for a, b in zip(exact["id_a"], exact["id_b"])]
    group_of = build_groups([str(v) for v in train["id"].tolist()], pairs)

    first = stratified_group_split(train, group_of, target_size=8, seed=11)
    second = stratified_group_split(train, group_of, target_size=8, seed=11)
    assert first.val_ids == second.val_ids
    assert first.train_ids == second.train_ids

    other = stratified_group_split(train, group_of, target_size=8, seed=12)
    assert set(other.val_ids) | set(other.train_ids) == set(first.val_ids) | set(
        first.train_ids
    )


def test_content_hash_ignores_created_stamp(tiny_root: Path) -> None:
    train = load_csv(tiny_root, "train")
    ids = [str(value) for value in train["id"].tolist()]
    group_of = build_groups(ids, [])
    result = stratified_group_split(train, group_of, target_size=8, seed=3)
    payload = split_payload(result, seed=3, train_csv_sha256="deadbeef")

    first = dict(payload)
    first["created"] = "2026-01-01T00:00:00+00:00"
    second = dict(payload)
    second["created"] = "2026-09-21T12:34:56+00:00"
    assert content_hash(first) == content_hash(second)
    assert "created" not in canonical_json(first)


def test_stratification_tracks_the_answer_distribution(tiny_root: Path) -> None:
    train = load_csv(tiny_root, "train")
    ids = [str(value) for value in train["id"].tolist()]
    group_of = build_groups(ids, [])
    result = stratified_group_split(train, group_of, target_size=8, seed=5)
    assert len(result.val_ids) == 8
    # The synthetic answers are perfectly balanced, so each letter appears twice.
    assert set(result.val_answer_counts.values()) == {2}


# --------------------------------------------------------------------------- #
# fold CSV -> split JSON
# --------------------------------------------------------------------------- #
def write_fold_files(
    root: Path,
    val_ids: list[str],
    train_ids: list[str],
    exclude_ids: list[str] | None = None,
) -> Path:
    fold_dir = root / "fold_csv_files"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "val_fold0.csv").write_text("id\n" + "\n".join(val_ids) + "\n", encoding="utf-8")
    (fold_dir / "train_fold1234.csv").write_text(
        "id\n" + "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    if exclude_ids is not None:
        (fold_dir / "exclude_train_overlap_test.csv").write_text(
            "id,fold\n" + "\n".join(f"{row_id},0" for row_id in exclude_ids) + "\n",
            encoding="utf-8",
        )
    return fold_dir


def run_folds_to_split(
    fold_dir: Path,
    data_root: Path,
    out_dir: Path,
    scope: str | None = None,
    extra: list[str] | None = None,
) -> int:
    import sys as _sys

    from scripts import folds_to_split as folds_cli

    argv = _sys.argv
    _sys.argv = [
        "folds_to_split.py",
        "--fold-dir",
        str(fold_dir),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
    ]
    if scope is not None:
        _sys.argv += ["--scope", scope]
    _sys.argv += list(extra or [])
    try:
        return folds_cli.main()
    finally:
        _sys.argv = argv


def write_folds_csv(path: Path, assignment: dict[str, int]) -> Path:
    """id,fold 두 열짜리 전체 배정 파일을 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,fold"] + [f"{row_id},{fold}" for row_id, fold in sorted(assignment.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_folds_to_split_writes_the_schema_train_py_reads(tmp_path: Path) -> None:
    import json

    val_ids = ["train_0002.jpg", "train_0004.jpg"]
    train_ids = ["train_0001.jpg", "train_0003.jpg", "train_0005.jpg"]
    fold_dir = write_fold_files(tmp_path, val_ids, train_ids)
    out_dir = tmp_path / "out"

    assert run_folds_to_split(fold_dir, tmp_path, out_dir) == 0

    payload = json.loads((out_dir / "split_fold0.json").read_text(encoding="utf-8"))
    assert payload["version"] == "fold0"
    assert payload["train_ids"] == sorted(train_ids)
    assert payload["val_ids"] == sorted(val_ids)
    assert payload["n_train"] == 3 and payload["n_val"] == 2
    assert set(payload["source_files"]) == {"val_fold0.csv", "train_fold1234.csv"}
    # content 해시는 created 줄을 빼고 계산한다.
    assert payload["content_sha256"] == content_hash(
        {k: v for k, v in payload.items() if k not in {"created", "content_sha256"}}
    )

    ids = (out_dir / "val_fold0_ids.txt").read_text(encoding="utf-8").split()
    assert ids == sorted(val_ids)


def test_folds_to_split_refuses_overlapping_ids(tmp_path: Path) -> None:
    fold_dir = write_fold_files(
        tmp_path, ["train_0001.jpg"], ["train_0001.jpg", "train_0002.jpg"]
    )
    with pytest.raises(SystemExit, match="겹칩니다"):
        run_folds_to_split(fold_dir, tmp_path, tmp_path / "out")


def test_folds_to_split_output_is_committed_and_current() -> None:
    """저장소에 들어 있는 split_fold0.json이 fold CSV와 같은 내용인지 확인한다."""
    import json

    repo = Path(__file__).resolve().parents[1]
    split_path = repo / "artifacts" / "data" / "split_fold0.json"
    if not split_path.exists():
        pytest.skip("artifacts/data/split_fold0.json이 아직 없습니다.")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload["n_val"] == 1344
    assert payload["n_train"] == 5358
    assert not set(payload["train_ids"]) & set(payload["val_ids"])


def test_folds_to_split_writes_full_split_with_empty_validation(tmp_path: Path) -> None:
    """--scope full은 fold 0까지 합친 학습 목록과 빈 검증 목록을 쓴다."""
    import json

    val_ids = ["train_0002.jpg", "train_0004.jpg"]
    train_ids = ["train_0001.jpg", "train_0003.jpg", "train_0005.jpg"]
    fold_dir = write_fold_files(tmp_path, val_ids, train_ids)
    out_dir = tmp_path / "out"

    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="full") == 0

    payload = json.loads((out_dir / "split_full.json").read_text(encoding="utf-8"))
    assert payload["version"] == "full"
    assert payload["train_ids"] == sorted(train_ids + val_ids)
    assert payload["val_ids"] == []
    assert payload["n_train"] == 5 and payload["n_val"] == 0
    assert payload["includes_fold0_val"] is True
    assert payload["n_fold0_val_added"] == len(val_ids)
    # 제외 목록 파일이 없으면 아무것도 빼지 않고, 그 사실을 0으로 기록한다.
    assert payload["excluded_overlap_test"] == 0
    assert payload["excluded_overlap_test_ids"] == []
    assert payload["content_sha256"] == content_hash(
        {k: v for k, v in payload.items() if k not in {"created", "content_sha256"}}
    )
    # fold0 전용으로 돌렸으면 전체 분할 파일은 생기지 않는다.
    assert not (out_dir / "split_fold0.json").exists()


def test_full_split_drops_ids_that_overlap_test(tmp_path: Path) -> None:
    """fold 0에 남아 있는 test 중복 문항은 전체 분할에서 빠져야 한다.

    `train_fold1234.csv`는 test와 이미지가 같은 문항을 이미 빼 두었지만 `val_fold0.csv`는
    빼지 않았다. 두 fold를 그냥 합치면 그 문항들이 학습에 다시 들어온다.
    """
    import json

    val_ids = ["train_0002.jpg", "train_0004.jpg", "train_0006.jpg"]
    train_ids = ["train_0001.jpg", "train_0003.jpg", "train_0005.jpg"]
    # train_0004는 fold 0에만 남아 있는 중복이고, train_0099는 어느 fold에도 없는 id다.
    fold_dir = write_fold_files(
        tmp_path, val_ids, train_ids, exclude_ids=["train_0004.jpg", "train_0099.jpg"]
    )
    out_dir = tmp_path / "out"

    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="full") == 0

    payload = json.loads((out_dir / "split_full.json").read_text(encoding="utf-8"))
    assert payload["excluded_overlap_test"] == 1
    assert payload["excluded_overlap_test_ids"] == ["train_0004.jpg"]
    assert "train_0004.jpg" not in payload["train_ids"]
    assert payload["n_train"] == 5
    assert payload["train_ids"] == sorted(set(train_ids + val_ids) - {"train_0004.jpg"})
    # fold 0에서 실제로 더해진 문항 수도 제외한 뒤의 값을 적는다.
    assert payload["n_fold0_val_added"] == 2
    # 제외 목록 파일도 출처로 기록한다.
    assert "exclude_train_overlap_test.csv" in payload["source_files"]

    # fold0 분할은 제외 목록의 영향을 받지 않는다. 검증 1,344문항을 그대로 두기 위해서다.
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold0") == 0
    fold0 = json.loads((out_dir / "split_fold0.json").read_text(encoding="utf-8"))
    assert fold0["val_ids"] == sorted(val_ids)
    assert "exclude_train_overlap_test.csv" not in fold0["source_files"]


def test_train_accepts_a_split_with_no_validation_ids(tmp_path: Path) -> None:
    """vqa.train의 split 로더는 train_ids만 읽으므로 빈 val_ids를 그대로 받아들인다."""
    import json

    from vqa.train import load_split_ids

    split_path = tmp_path / "split_full.json"
    split_path.write_text(
        json.dumps({"train_ids": ["train_0001.jpg", "train_0002.jpg"], "val_ids": []}),
        encoding="utf-8",
    )
    ids, split_hash = load_split_ids(str(split_path), [])
    assert ids == ["train_0001.jpg", "train_0002.jpg"]
    assert split_hash


def test_full_split_output_is_committed_and_current() -> None:
    """저장소의 split_full.json이 fold0 분할과 제외 목록에 정확히 맞물리는지 확인한다."""
    import json

    import pandas as pd

    repo = Path(__file__).resolve().parents[1]
    full_path = repo / "artifacts" / "data" / "split_full.json"
    fold0_path = repo / "artifacts" / "data" / "split_fold0.json"
    exclude_path = repo / "fold_csv_files" / "exclude_train_overlap_test.csv"
    if not (full_path.exists() and fold0_path.exists() and exclude_path.exists()):
        pytest.skip("artifacts/data의 split 파일이 아직 없습니다.")

    full = json.loads(full_path.read_text(encoding="utf-8"))
    fold0 = json.loads(fold0_path.read_text(encoding="utf-8"))
    exclude_ids = {
        str(value).strip()
        for value in pd.read_csv(exclude_path, encoding="utf-8-sig", dtype="string")["id"]
    }

    # 5,358 + 1,344 에서 fold 0에 남아 있던 test 중복 12문항을 뺀 값이다.
    assert full["n_train"] == 6690
    assert full["n_val"] == 0
    assert full["val_ids"] == []
    assert len(full["train_ids"]) == 6690
    assert len(set(full["train_ids"])) == 6690
    assert full["excluded_overlap_test"] == 12
    assert len(full["excluded_overlap_test_ids"]) == 12
    assert full["n_fold0_val_added"] == 1332

    full_ids = set(full["train_ids"])
    fold0_train = set(fold0["train_ids"])
    fold0_val = set(fold0["val_ids"])
    # test와 이미지가 같은 문항은 전체 학습 목록에 하나도 남아 있으면 안 된다.
    assert not (full_ids & exclude_ids)
    assert set(full["excluded_overlap_test_ids"]) == fold0_val & exclude_ids
    # fold0의 학습 목록은 전체의 부분집합이고, 나머지는 fold 0에서 제외 문항을 뺀 나머지다.
    assert fold0_train <= full_ids
    assert full_ids - fold0_train == fold0_val - exclude_ids
    assert len(fold0_val) == 1344
    # 두 파일이 같은 fold CSV에서 나왔는지 확인한다. 전체 분할에는 제외 목록이 하나 더 붙는다.
    assert full["source_files"]["val_fold0.csv"] == fold0["source_files"]["val_fold0.csv"]
    assert (
        full["source_files"]["train_fold1234.csv"]
        == fold0["source_files"]["train_fold1234.csv"]
    )
    assert "exclude_train_overlap_test.csv" in full["source_files"]
    assert full["content_sha256"] == content_hash(
        {k: v for k, v in full.items() if k not in {"created", "content_sha256"}}
    )


def test_previous_full_split_is_preserved_for_the_running_colab_run() -> None:
    """6,702문항으로 이미 시작한 학습이 참조할 수 있도록 옛 분할을 남겨 둔다."""
    import json

    repo = Path(__file__).resolve().parents[1]
    legacy_path = repo / "artifacts" / "data" / "split_full_v1_6702.json"
    if not legacy_path.exists():
        pytest.skip("artifacts/data/split_full_v1_6702.json이 아직 없습니다.")

    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert legacy["n_train"] == 6702
    assert len(legacy["train_ids"]) == 6702
    assert legacy["val_ids"] == []
    # 옛 분할은 다시 만들지 않는 보존본이므로 내용 해시가 바뀌면 안 된다.
    assert (
        legacy["content_sha256"]
        == "d394c8369c8dff020eea12e5d2d6cc00a7b4eaf97d1cb6f411d2e04a8ac4db01"
    )


# --------------------------------------------------------------------------- #
# --scope fold: fold N을 검증으로 하는 분할
# --------------------------------------------------------------------------- #
def make_assignment(n_per_fold: int = 6, n_folds: int = 5) -> dict[str, int]:
    """id -> fold. id는 fold 번호와 무관하게 이름만 보고 섞이지 않도록 붙인다."""
    assignment: dict[str, int] = {}
    index = 0
    for fold in range(n_folds):
        for _ in range(n_per_fold):
            index += 1
            assignment[f"train_{index:04d}.jpg"] = fold
    return assignment


def test_fold_scope_needs_a_fold_assignment_file(tmp_path: Path) -> None:
    """배정 파일이 없으면 무엇을 받아야 하는지 알려 주고 멈춘다."""
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0002.jpg"])
    with pytest.raises(SystemExit) as caught:
        run_folds_to_split(
            fold_dir, tmp_path, tmp_path / "out", scope="fold", extra=["--val-fold", "2"]
        )
    message = str(caught.value)
    assert "id,fold" in message
    # 찾아본 자리와 받아야 할 파일 이름을 둘 다 알려 준다.
    assert "train_folds.csv" in message
    assert "folds_all.csv" in message


def test_fold_scope_needs_a_val_fold_number(tmp_path: Path) -> None:
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0002.jpg"])
    with pytest.raises(SystemExit, match="--val-fold"):
        run_folds_to_split(fold_dir, tmp_path, tmp_path / "out", scope="fold")


def test_fold_scope_rejects_a_fold_number_outside_the_range(tmp_path: Path) -> None:
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0002.jpg"])
    with pytest.raises(SystemExit, match="0~4"):
        run_folds_to_split(
            fold_dir, tmp_path, tmp_path / "out", scope="fold", extra=["--val-fold", "5"]
        )


def test_fold_scope_splits_on_the_chosen_fold(tmp_path: Path) -> None:
    """fold N만 검증으로 가고 나머지 네 fold가 학습이 된다."""
    import json

    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    out_dir = tmp_path / "out"

    assert (
        run_folds_to_split(
            fold_dir,
            tmp_path,
            out_dir,
            scope="fold",
            extra=["--val-fold", "2", "--folds-csv", str(folds_csv)],
        )
        == 0
    )
    payload = json.loads((out_dir / "split_fold2.json").read_text(encoding="utf-8"))
    assert payload["version"] == "fold2"
    assert payload["val_fold"] == 2
    expected_val = sorted(i for i, fold in assignment.items() if fold == 2)
    assert payload["val_ids"] == expected_val
    assert payload["n_val"] == 6
    assert payload["n_train"] == 24
    assert not set(payload["train_ids"]) & set(payload["val_ids"])
    assert payload["excluded_overlap_test"] == 0
    # 검증 id 목록도 함께 남는다.
    ids = (out_dir / "val_fold2_ids.txt").read_text(encoding="utf-8").split()
    assert ids == expected_val


def test_fold_scope_excludes_test_overlap_from_training_only(tmp_path: Path) -> None:
    """test 중복 문항은 학습에서만 빠지고 검증에는 남는다. fold 0 분할과 같은 규칙이다."""
    import json

    assignment = make_assignment()
    # 검증이 될 fold 2에서 하나, 학습이 될 fold 4에서 둘을 제외 목록에 넣는다.
    in_val = sorted(i for i, fold in assignment.items() if fold == 2)[0]
    in_train = sorted(i for i, fold in assignment.items() if fold == 4)[:2]
    fold_dir = write_fold_files(
        tmp_path,
        ["train_0001.jpg"],
        ["train_0007.jpg"],
        exclude_ids=[in_val, *in_train],
    )
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    out_dir = tmp_path / "out"

    assert (
        run_folds_to_split(
            fold_dir,
            tmp_path,
            out_dir,
            scope="fold",
            extra=["--val-fold", "2", "--folds-csv", str(folds_csv)],
        )
        == 0
    )
    payload = json.loads((out_dir / "split_fold2.json").read_text(encoding="utf-8"))
    assert payload["excluded_overlap_test"] == 2
    assert payload["excluded_overlap_test_ids"] == sorted(in_train)
    assert payload["n_train"] == 22
    # 검증 목록은 그대로다.
    assert in_val in payload["val_ids"]
    assert payload["n_val"] == 6
    for row_id in in_train:
        assert row_id not in payload["train_ids"]


def test_fold_scope_refuses_an_incomplete_assignment(tmp_path: Path) -> None:
    assignment = {row_id: fold for row_id, fold in make_assignment().items() if fold != 4}
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    with pytest.raises(SystemExit, match="모두 채우지 않습니다"):
        run_folds_to_split(
            fold_dir,
            tmp_path,
            tmp_path / "out",
            scope="fold",
            extra=["--val-fold", "2", "--folds-csv", str(folds_csv)],
        )


def test_fold_scope_never_rewrites_an_existing_split_with_the_same_ids(tmp_path: Path) -> None:
    """이미 있는 분할과 내용이 같으면 파일을 건드리지 않는다.

    저장소에 들어 있는 split_fold0.json 을 지키기 위한 규칙이다. 내용이 같아도 다시 쓰면
    created 줄 때문에 diff가 남는다.
    """
    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    out_dir = tmp_path / "out"
    args = ["--val-fold", "1", "--folds-csv", str(folds_csv)]

    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=args) == 0
    written = out_dir / "split_fold1.json"
    before = written.read_bytes()
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=args) == 0
    assert written.read_bytes() == before


def test_fold_scope_refuses_to_overwrite_a_split_with_different_ids(tmp_path: Path) -> None:
    assignment = make_assignment()
    fold_dir = write_fold_files(tmp_path, ["train_0001.jpg"], ["train_0007.jpg"], exclude_ids=[])
    out_dir = tmp_path / "out"
    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    args = ["--val-fold", "1", "--folds-csv", str(folds_csv)]
    assert run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=args) == 0

    # 배정을 뒤섞어서 같은 fold 번호에 다른 문항이 들어가게 만든다.
    shuffled = {row_id: (fold + 1) % 5 for row_id, fold in assignment.items()}
    write_folds_csv(folds_csv, shuffled)
    with pytest.raises(SystemExit, match="id 목록이 다릅니다"):
        run_folds_to_split(fold_dir, tmp_path, out_dir, scope="fold", extra=args)


def test_committed_fold0_split_is_reproducible_from_a_fold_assignment(tmp_path: Path) -> None:
    """저장소의 fold CSV로 전체 배정을 만들면 split_fold0.json 의 id 목록이 그대로 나온다.

    --scope fold 가 팀 공식 fold 0 분할과 같은 규칙으로 도는지 확인한다. 학습에서만 test
    중복을 빼고 검증은 그대로 두는 규칙이 맞아야 두 목록이 일치한다.
    """
    import json

    repo = Path(__file__).resolve().parents[1]
    split_path = repo / "artifacts" / "data" / "split_fold0.json"
    val_path = repo / "fold_csv_files" / "val_fold0.csv"
    all_path = repo / "fold_csv_files" / "train_fold1234_all.csv"
    if not (split_path.exists() and val_path.exists() and all_path.exists()):
        pytest.skip("저장소의 fold CSV 가 없습니다.")

    pd = pytest.importorskip("pandas")
    frames = pd.read_csv(val_path, dtype="string")["id"].tolist()
    rest = pd.read_csv(all_path, dtype="string")["id"].tolist()
    assignment = {str(row_id).strip(): 0 for row_id in frames}
    for index, row_id in enumerate(rest):
        assignment[str(row_id).strip()] = 1 + index % 4

    folds_csv = write_folds_csv(tmp_path / "folds_all.csv", assignment)
    out_dir = tmp_path / "out"
    assert (
        run_folds_to_split(
            repo / "fold_csv_files",
            tmp_path,
            out_dir,
            scope="fold",
            extra=["--val-fold", "0", "--folds-csv", str(folds_csv)],
        )
        == 0
    )
    made = json.loads((out_dir / "split_fold0.json").read_text(encoding="utf-8"))
    committed = json.loads(split_path.read_text(encoding="utf-8"))
    assert made["train_ids"] == committed["train_ids"]
    assert made["val_ids"] == committed["val_ids"]
    assert made["excluded_overlap_test"] == 12
