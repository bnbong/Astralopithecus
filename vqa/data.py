"""CSV loading, schema validation and image fingerprinting helpers.

Rules that this module keeps:

* ``id`` values are preserved verbatim (they contain the ``.jpg`` suffix).
* the relative ``path`` column is never rewritten; it is joined to DATA_ROOT
  only at access time via :func:`image_path`.
* every filesystem path is built with :mod:`pathlib`, so the same code runs on
  Windows, Linux (Colab) and macOS.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence

import pandas as pd

# CSV files may be stored with a BOM, so always decode with utf-8-sig.
CSV_ENCODING = "utf-8-sig"

VALID_ANSWERS = ("a", "b", "c", "d")

BASE_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")
DEV_ANSWER_COLUMNS = tuple(f"answer{i}" for i in range(1, 6))

REQUIRED_COLUMNS = {
    "train": BASE_COLUMNS + ("answer",),
    "test": BASE_COLUMNS,
    "dev": BASE_COLUMNS + DEV_ANSWER_COLUMNS,
}

SPLIT_NAMES = tuple(REQUIRED_COLUMNS)


class SchemaError(ValueError):
    """Raised when a CSV file does not satisfy the agreed input schema."""


@dataclass(frozen=True)
class ImageFingerprint:
    """One row of the per-image hash cache."""

    split: str
    id: str
    path: str
    exists: bool
    readable: bool
    width: int
    height: int
    mode: str
    format: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    phash: str
    error: str


FINGERPRINT_COLUMNS = (
    "split",
    "id",
    "path",
    "exists",
    "readable",
    "width",
    "height",
    "mode",
    "format",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "phash",
    "error",
)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def image_path(data_root: os.PathLike[str] | str, rel_path: str) -> Path:
    """Join a CSV ``path`` value to ``data_root`` without touching the CSV value.

    The CSV stores POSIX style relative paths such as ``train/train_0001.jpg``.
    They are split on ``/`` (and on ``\\`` for safety) and re-joined with the
    platform separator, so the function behaves identically on every OS.
    """
    normalized = str(rel_path).replace("\\", "/").strip()
    parts = PurePosixPath(normalized).parts
    return Path(data_root).joinpath(*parts)


def iter_image_paths(
    df: pd.DataFrame, data_root: os.PathLike[str] | str
) -> Iterator[tuple[str, Path]]:
    """Yield ``(id, absolute path)`` pairs for every row of ``df``."""
    for row_id, rel_path in zip(df["id"], df["path"]):
        yield row_id, image_path(data_root, rel_path)


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA256 hex digest of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def _read_raw_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SchemaError(f"CSV file not found: {csv_path}")
    # dtype=str + na_filter=False keeps ids verbatim and turns blank answers
    # into empty strings instead of NaN.
    return pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding=CSV_ENCODING,
    )


def validate_schema(df: pd.DataFrame, name: str) -> None:
    """Validate the dataframe of split ``name`` and raise :class:`SchemaError`."""
    if name not in REQUIRED_COLUMNS:
        raise SchemaError(
            f"unknown split name {name!r}; expected one of {SPLIT_NAMES}"
        )

    required = REQUIRED_COLUMNS[name]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SchemaError(f"[{name}] missing required columns: {missing}")

    ids = df["id"].astype(str)
    if (ids.str.strip() == "").any():
        empty_rows = df.index[ids.str.strip() == ""].tolist()[:10]
        raise SchemaError(f"[{name}] empty id values at rows {empty_rows}")

    duplicated = ids[ids.duplicated()].unique().tolist()
    if duplicated:
        raise SchemaError(
            f"[{name}] duplicate id values ({len(duplicated)}): {duplicated[:10]}"
        )

    paths = df["path"].astype(str)
    if (paths.str.strip() == "").any():
        empty_rows = df.index[paths.str.strip() == ""].tolist()[:10]
        raise SchemaError(f"[{name}] empty path values at rows {empty_rows}")

    if name == "train":
        answers = df["answer"].astype(str).str.strip().str.lower()
        bad = sorted(set(answers) - set(VALID_ANSWERS))
        if bad:
            raise SchemaError(
                f"[train] answer must be one of {VALID_ANSWERS}; found {bad[:10]}"
            )

    if name == "dev":
        allowed = set(VALID_ANSWERS) | {""}
        for column in DEV_ANSWER_COLUMNS:
            values = df[column].astype(str).str.strip().str.lower()
            bad = sorted(set(values) - allowed)
            if bad:
                raise SchemaError(
                    f"[dev] column {column} must be blank or one of "
                    f"{VALID_ANSWERS}; found {bad[:10]}"
                )


def load_csv(
    data_root: os.PathLike[str] | str,
    name: str,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load ``{data_root}/{name}.csv`` as a dataframe of strings.

    Parameters
    ----------
    data_root:
        DATA_ROOT directory that holds the CSV files and the image folders.
    name:
        One of ``train``, ``test`` or ``dev``.
    validate:
        When true the schema is checked and :class:`SchemaError` is raised on
        any violation.
    """
    if name not in REQUIRED_COLUMNS:
        raise SchemaError(
            f"unknown split name {name!r}; expected one of {SPLIT_NAMES}"
        )

    root = Path(data_root)
    df = _read_raw_csv(root / f"{name}.csv")
    df.columns = [str(column).strip() for column in df.columns]

    for column in df.columns:
        # Strip only the carriage returns and surrounding blanks that CRLF files
        # leave behind; the visible text of ids and paths stays untouched.
        df[column] = df[column].astype(str).str.replace("\r", "", regex=False)

    if validate:
        validate_schema(df, name)

    df.attrs["split_name"] = name
    df.attrs["data_root"] = str(root)
    return df


def load_all(
    data_root: os.PathLike[str] | str, names: Sequence[str] = SPLIT_NAMES
) -> dict[str, pd.DataFrame]:
    """Load several splits at once and return them keyed by split name."""
    return {name: load_csv(data_root, name) for name in names}


# --------------------------------------------------------------------------- #
# image fingerprints
# --------------------------------------------------------------------------- #
IMAGEHASH_HINT = (
    "the 'imagehash' package is required for perceptual hashing and near-duplicate "
    "detection; install it with `pip install ImageHash`"
)


def require_imagehash():
    """Import :mod:`imagehash` or raise a clear :class:`ImportError`.

    This is deliberately called outside the per-image ``try``. A missing package
    is a setup error for the whole run: swallowing it would mark every image as
    unreadable and silently return an empty near-duplicate report.
    """
    try:
        import imagehash
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(IMAGEHASH_HINT) from exc
    return imagehash


def _fingerprint_one(args: tuple[str, str, str, str]) -> dict[str, object]:
    split, row_id, rel_path, data_root = args
    # Outside the try below: a missing package must not look like a broken image.
    imagehash = require_imagehash()
    absolute = image_path(data_root, rel_path)
    record: dict[str, object] = {
        "split": split,
        "id": row_id,
        "path": rel_path,
        "exists": False,
        "readable": False,
        "width": 0,
        "height": 0,
        "mode": "",
        "format": "",
        "size_bytes": 0,
        "mtime_ns": 0,
        "sha256": "",
        "phash": "",
        "error": "",
    }

    if not absolute.is_file():
        record["error"] = "missing"
        return record

    stat = absolute.stat()
    record["exists"] = True
    record["size_bytes"] = int(stat.st_size)
    record["mtime_ns"] = int(stat.st_mtime_ns)

    try:
        record["sha256"] = file_sha256(absolute)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        record["error"] = f"read_error: {exc}"
        return record

    try:
        from PIL import Image

        with Image.open(absolute) as image:
            image.load()
            record["width"] = int(image.width)
            record["height"] = int(image.height)
            record["mode"] = str(image.mode)
            record["format"] = str(image.format or "")
            record["phash"] = str(imagehash.phash(image.convert("RGB")))
        record["readable"] = True
    except Exception as exc:  # noqa: BLE001 - any decoding failure is a finding
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def compute_fingerprints(
    data_root: os.PathLike[str] | str,
    frames: dict[str, pd.DataFrame],
    *,
    workers: int | None = None,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Hash every image of every split and return one dataframe of fingerprints."""
    # Fail fast in the parent process rather than inside every worker.
    require_imagehash()

    jobs: list[tuple[str, str, str, str]] = []
    for split, df in frames.items():
        for row_id, rel_path in zip(df["id"], df["path"]):
            jobs.append((split, str(row_id), str(rel_path), str(data_root)))

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    records: list[dict[str, object]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            records.append(_fingerprint_one(job))
            if progress_every and index % progress_every == 0:
                print(f"  fingerprinted {index}/{len(jobs)} images", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(
                pool.map(_fingerprint_one, jobs, chunksize=32), start=1
            ):
                records.append(record)
                if progress_every and index % progress_every == 0:
                    print(f"  fingerprinted {index}/{len(jobs)} images", flush=True)

    return pd.DataFrame.from_records(records, columns=list(FINGERPRINT_COLUMNS))


def load_fingerprint_cache(cache_path: os.PathLike[str] | str) -> pd.DataFrame | None:
    """Read a previously written fingerprint cache, or return ``None``."""
    path = Path(cache_path)
    if not path.is_file():
        return None
    cached = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing = [c for c in FINGERPRINT_COLUMNS if c not in cached.columns]
    if missing:
        return None
    for column in ("width", "height", "size_bytes", "mtime_ns"):
        cached[column] = pd.to_numeric(cached[column], errors="coerce").fillna(0)
        cached[column] = cached[column].astype("int64")
    for column in ("exists", "readable"):
        cached[column] = cached[column].isin({"True", "true", "1"})
    return cached


def get_fingerprints(
    data_root: os.PathLike[str] | str,
    frames: dict[str, pd.DataFrame],
    cache_path: os.PathLike[str] | str,
    *,
    workers: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return fingerprints for every row, reusing the cache when it is valid.

    A cached row is reused when the ``(split, id)`` key is present and the file
    size and modification time on disk still match the cached values, so reruns
    stay fast while a changed image is hashed again.
    """
    cache_file = Path(cache_path)
    cached = None if refresh else load_fingerprint_cache(cache_file)

    wanted: list[tuple[str, str, str]] = []
    for split, df in frames.items():
        for row_id, rel_path in zip(df["id"], df["path"]):
            wanted.append((split, str(row_id), str(rel_path)))

    reusable: dict[tuple[str, str], dict[str, object]] = {}
    if cached is not None:
        by_key = {
            (str(row["split"]), str(row["id"])): row
            for _, row in cached.iterrows()
        }
        for split, row_id, rel_path in wanted:
            row = by_key.get((split, row_id))
            if row is None or str(row["path"]) != rel_path:
                continue
            absolute = image_path(data_root, rel_path)
            if not absolute.is_file():
                continue
            stat = absolute.stat()
            if (
                int(row["size_bytes"]) == int(stat.st_size)
                and int(row["mtime_ns"]) == int(stat.st_mtime_ns)
                and str(row["sha256"])
            ):
                reusable[(split, row_id)] = row.to_dict()

    todo = {
        split: df[~df["id"].astype(str).map(lambda i, s=split: (s, i) in reusable)]
        for split, df in frames.items()
    }
    todo = {split: df for split, df in todo.items() if len(df) > 0}

    fresh_count = sum(len(df) for df in todo.values())
    print(
        f"fingerprints: {len(reusable)} reused from cache, {fresh_count} to compute",
        flush=True,
    )

    frames_out: list[pd.DataFrame] = []
    if reusable:
        frames_out.append(
            pd.DataFrame.from_records(
                list(reusable.values()), columns=list(FINGERPRINT_COLUMNS)
            )
        )
    if todo:
        frames_out.append(
            compute_fingerprints(data_root, todo, workers=workers)
        )

    result = pd.concat(frames_out, ignore_index=True)
    # Keep a stable row order so the cache file itself is reproducible.
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    result["_order"] = result["split"].map(lambda s: split_order.get(s, 99))
    result = result.sort_values(["_order", "id"], kind="mergesort").drop(
        columns="_order"
    )
    result = result.reset_index(drop=True)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_file, index=False, encoding="utf-8")
    return result


def fingerprints_for(fingerprints: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the fingerprint rows that belong to one split."""
    return fingerprints[fingerprints["split"] == split].reset_index(drop=True)


__all__ = [
    "BASE_COLUMNS",
    "CSV_ENCODING",
    "DEV_ANSWER_COLUMNS",
    "FINGERPRINT_COLUMNS",
    "IMAGEHASH_HINT",
    "ImageFingerprint",
    "REQUIRED_COLUMNS",
    "SPLIT_NAMES",
    "SchemaError",
    "VALID_ANSWERS",
    "compute_fingerprints",
    "file_sha256",
    "fingerprints_for",
    "get_fingerprints",
    "image_path",
    "iter_image_paths",
    "load_all",
    "load_csv",
    "load_fingerprint_cache",
    "require_imagehash",
    "text_sha256",
    "validate_schema",
]
