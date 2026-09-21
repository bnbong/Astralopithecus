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
def write_fold_files(root: Path, val_ids: list[str], train_ids: list[str]) -> Path:
    fold_dir = root / "fold_csv_files"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "val_fold0.csv").write_text("id\n" + "\n".join(val_ids) + "\n", encoding="utf-8")
    (fold_dir / "train_fold1234.csv").write_text(
        "id\n" + "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    return fold_dir


def run_folds_to_split(
    fold_dir: Path, data_root: Path, out_dir: Path, scope: str | None = None
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
    try:
        return folds_cli.main()
    finally:
        _sys.argv = argv


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
    assert payload["content_sha256"] == content_hash(
        {k: v for k, v in payload.items() if k not in {"created", "content_sha256"}}
    )
    # fold0 전용으로 돌렸으면 전체 분할 파일은 생기지 않는다.
    assert not (out_dir / "split_fold0.json").exists()


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
    """저장소의 split_full.json이 fold0 분할과 정확히 맞물리는지 확인한다."""
    import json

    repo = Path(__file__).resolve().parents[1]
    full_path = repo / "artifacts" / "data" / "split_full.json"
    fold0_path = repo / "artifacts" / "data" / "split_fold0.json"
    if not (full_path.exists() and fold0_path.exists()):
        pytest.skip("artifacts/data의 split 파일이 아직 없습니다.")

    full = json.loads(full_path.read_text(encoding="utf-8"))
    fold0 = json.loads(fold0_path.read_text(encoding="utf-8"))

    assert full["n_train"] == 6702
    assert full["n_val"] == 0
    assert full["val_ids"] == []
    assert len(full["train_ids"]) == 6702
    assert len(set(full["train_ids"])) == 6702

    full_ids = set(full["train_ids"])
    fold0_train = set(fold0["train_ids"])
    fold0_val = set(fold0["val_ids"])
    # fold0의 학습 목록은 전체의 부분집합이고, 나머지가 정확히 fold 0의 1,344문항이다.
    assert fold0_train <= full_ids
    assert full_ids - fold0_train == fold0_val
    assert len(fold0_val) == 1344
    # 두 파일이 같은 fold CSV에서 나왔는지 확인한다.
    assert full["source_files"] == fold0["source_files"]
    assert full["content_sha256"] == content_hash(
        {k: v for k, v in full.items() if k not in {"created", "content_sha256"}}
    )
