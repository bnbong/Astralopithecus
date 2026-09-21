"""Group aware, deterministic validation split for the train set.

A group is a connected component of the graph whose edges are exact duplicate
images (identical SHA256) and near duplicate images (perceptual hash within a
small Hamming distance). Whole groups move together, so no scene is seen on
both sides of the split.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import VALID_ANSWERS, text_sha256

DEFAULT_PHASH_THRESHOLD = 5
SPLIT_VERSION = "v1"


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #
class UnionFind:
    """Union-find over arbitrary hashable keys, with path compression."""

    def __init__(self, keys: Iterable[str]) -> None:
        self._parent: dict[str, str] = {key: key for key in keys}
        self._rank: dict[str, int] = {key: 0 for key in self._parent}

    def add(self, key: str) -> None:
        if key not in self._parent:
            self._parent[key] = key
            self._rank[key] = 0

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1

    def components(self) -> dict[str, list[str]]:
        """Return ``{root: sorted members}`` for every component."""
        groups: dict[str, list[str]] = {}
        for key in self._parent:
            groups.setdefault(self.find(key), []).append(key)
        for members in groups.values():
            members.sort()
        return groups


# --------------------------------------------------------------------------- #
# duplicate detection
# --------------------------------------------------------------------------- #
def phash_to_uint64(values: Sequence[str]) -> np.ndarray:
    """Convert 16 hex character perceptual hashes into an array of uint64."""
    return np.array([int(value, 16) for value in values], dtype=np.uint64)


def _hamming_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(left[:, None], right[None, :])
    return np.bitwise_count(xor).astype(np.int16)


def near_duplicate_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame | None = None,
    *,
    threshold: int = DEFAULT_PHASH_THRESHOLD,
    block: int = 512,
) -> pd.DataFrame:
    """Return every image pair whose perceptual hashes are within ``threshold``.

    When ``right`` is omitted the search runs inside ``left`` and each pair is
    reported once. Rows without a usable perceptual hash are skipped, so missing
    or unreadable images never produce a pair.
    """
    columns = ["split_a", "id_a", "split_b", "id_b", "hamming", "same_sha256"]

    left = left[left["phash"].astype(str).str.len() == 16].reset_index(drop=True)
    within = right is None
    right = left if within else right[
        right["phash"].astype(str).str.len() == 16
    ].reset_index(drop=True)

    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame(columns=columns)

    left_hashes = phash_to_uint64(left["phash"].tolist())
    right_hashes = phash_to_uint64(right["phash"].tolist())
    left_ids = left["id"].tolist()
    right_ids = right["id"].tolist()
    left_splits = left["split"].tolist()
    right_splits = right["split"].tolist()
    left_sha = left["sha256"].tolist()
    right_sha = right["sha256"].tolist()

    records: list[tuple[str, str, str, str, int, bool]] = []
    for start in range(0, len(left_hashes), block):
        stop = min(start + block, len(left_hashes))
        distances = _hamming_block(left_hashes[start:stop], right_hashes)
        rows, cols = np.nonzero(distances <= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            i = start + row
            j = col
            if within and j <= i:
                continue
            records.append(
                (
                    left_splits[i],
                    left_ids[i],
                    right_splits[j],
                    right_ids[j],
                    int(distances[row, col]),
                    bool(left_sha[i]) and left_sha[i] == right_sha[j],
                )
            )

    pairs = pd.DataFrame.from_records(records, columns=columns)
    if len(pairs) == 0:
        return pairs
    return pairs.sort_values(
        ["hamming", "id_a", "id_b"], kind="mergesort"
    ).reset_index(drop=True)


def exact_duplicate_pairs(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Return the pairs of rows that share an identical SHA256 digest."""
    columns = ["split_a", "id_a", "split_b", "id_b", "hamming", "same_sha256"]
    usable = fingerprints[fingerprints["sha256"].astype(str) != ""]
    records: list[tuple[str, str, str, str, int, bool]] = []
    for _, block in usable.groupby("sha256", sort=True):
        if len(block) < 2:
            continue
        rows = block.sort_values("id", kind="mergesort").to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                records.append(
                    (
                        str(rows[i]["split"]),
                        str(rows[i]["id"]),
                        str(rows[j]["split"]),
                        str(rows[j]["id"]),
                        0,
                        True,
                    )
                )
    return pd.DataFrame.from_records(records, columns=columns)


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def build_groups(
    ids: Sequence[str], pairs: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """Return ``{id: group label}`` for connected components over ``pairs``.

    Group labels are derived from the sorted order of each component's smallest
    member, so the labels do not depend on the order in which pairs arrive.
    """
    union = UnionFind(ids)
    known = set(ids)
    for left, right in pairs:
        if left in known and right in known:
            union.union(left, right)

    components = sorted(union.components().values(), key=lambda m: m[0])
    width = max(4, len(str(max(1, len(components) - 1))))
    group_of: dict[str, str] = {}
    for index, members in enumerate(components):
        label = f"g{index:0{width}d}"
        for member in members:
            group_of[member] = label
    return group_of


# --------------------------------------------------------------------------- #
# stratified group split
# --------------------------------------------------------------------------- #
def _largest_remainder(target: np.ndarray, total: int) -> np.ndarray:
    """Round a fractional target histogram to integers that sum to ``total``."""
    floor = np.floor(target).astype(int)
    shortfall = int(total - floor.sum())
    if shortfall > 0:
        remainder = target - floor
        # Ties go to the earlier letter, which keeps the result deterministic.
        order = np.lexsort((np.arange(len(target)), -remainder))
        floor[order[:shortfall]] += 1
    return floor.astype(float)


@dataclass
class SplitResult:
    """Outcome of one validation split."""

    train_ids: list[str]
    val_ids: list[str]
    group_of: dict[str, str]
    val_groups: list[str]
    target_size: int
    val_answer_counts: dict[str, int] = field(default_factory=dict)
    train_answer_counts: dict[str, int] = field(default_factory=dict)


def stratified_group_split(
    df: pd.DataFrame,
    group_of: Mapping[str, str],
    *,
    target_size: int = 500,
    seed: int = 20250921,
    answer_column: str = "answer",
) -> SplitResult:
    """Pick whole groups for validation while matching the answer distribution.

    The greedy loop repeatedly takes the candidate group that brings the running
    validation answer histogram closest to the target histogram. Ties are broken
    by a permutation drawn from ``seed``, so two runs with the same inputs give
    the same split.
    """
    ids = [str(value) for value in df["id"].tolist()]
    answers = [str(value).strip().lower() for value in df[answer_column].tolist()]
    answer_of = dict(zip(ids, answers))

    members: dict[str, list[str]] = {}
    for row_id in ids:
        members.setdefault(group_of[row_id], []).append(row_id)
    for group_members in members.values():
        group_members.sort()

    letters = list(VALID_ANSWERS)
    overall = Counter(answers)
    total = max(1, len(ids))
    target_size = min(int(target_size), len(ids))
    target_vector = np.array(
        [overall[letter] / total * target_size for letter in letters], dtype=float
    )

    group_labels = sorted(members)
    group_vectors = {
        label: np.array(
            [
                sum(1 for member in group_members if answer_of[member] == letter)
                for letter in letters
            ],
            dtype=float,
        )
        for label, group_members in members.items()
    }
    group_sizes = {label: len(group_members) for label, group_members in members.items()}

    rng = np.random.default_rng(seed)
    priority = {
        label: int(order)
        for label, order in zip(
            group_labels, rng.permutation(len(group_labels)).tolist()
        )
    }

    # Groups are ordered by the seeded priority, so both passes below visit the
    # groups in the same order on every run.
    ordered = sorted(group_labels, key=lambda label: priority[label])
    matrix = (
        np.stack([group_vectors[label] for label in ordered])
        if ordered
        else np.zeros((0, len(letters)), dtype=float)
    )
    sizes = np.array([group_sizes[label] for label in ordered], dtype=int)
    available = np.ones(len(ordered), dtype=bool)
    quota = _largest_remainder(target_vector, target_size)

    chosen: list[str] = []
    current = np.zeros(len(letters), dtype=float)
    current_size = 0

    # First pass: accept groups in the shuffled order as long as no answer letter
    # exceeds its quota. Scanning in a seeded random order keeps the sample
    # representative instead of favouring the larger duplicate groups.
    for index in range(len(ordered)):
        if current_size >= target_size:
            break
        size = int(sizes[index])
        if current_size + size > target_size:
            continue
        candidate = current + matrix[index]
        if np.any(candidate > quota):
            continue
        chosen.append(ordered[index])
        current = candidate
        current_size += size
        available[index] = False

    # Second pass: the quota rule can leave a few seats open, so fill them with
    # the group that brings the histogram closest to the target.
    while current_size < target_size:
        eligible = available & (sizes <= target_size - current_size)
        if not eligible.any():
            break
        costs = np.abs(matrix + (current - target_vector)).sum(axis=1)
        costs = np.where(eligible, costs, np.inf)
        best_index = int(np.argmin(costs))
        chosen.append(ordered[best_index])
        current = current + matrix[best_index]
        current_size += int(sizes[best_index])
        available[best_index] = False

    val_groups = sorted(chosen)
    val_ids = sorted(
        member for label in val_groups for member in members[label]
    )
    val_id_set = set(val_ids)
    train_ids = sorted(row_id for row_id in ids if row_id not in val_id_set)

    val_counts = Counter(answer_of[row_id] for row_id in val_ids)
    train_counts = Counter(answer_of[row_id] for row_id in train_ids)

    return SplitResult(
        train_ids=train_ids,
        val_ids=val_ids,
        group_of={row_id: group_of[row_id] for row_id in ids},
        val_groups=val_groups,
        target_size=target_size,
        val_answer_counts={letter: int(val_counts[letter]) for letter in letters},
        train_answer_counts={letter: int(train_counts[letter]) for letter in letters},
    )


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def split_payload(
    result: SplitResult,
    *,
    seed: int,
    train_csv_sha256: str,
    version: str = SPLIT_VERSION,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the dictionary that is written to ``split_v1.json`` minus ``created``."""
    payload: dict[str, object] = {
        "version": version,
        "seed": int(seed),
        "train_csv_sha256": train_csv_sha256,
        "target_val_size": int(result.target_size),
        "n_train": len(result.train_ids),
        "n_val": len(result.val_ids),
        "n_groups": len(set(result.group_of.values())),
        "n_val_groups": len(result.val_groups),
        "val_answer_counts": dict(result.val_answer_counts),
        "train_answer_counts": dict(result.train_answer_counts),
        "train_ids": list(result.train_ids),
        "val_ids": list(result.val_ids),
        "group_of": dict(result.group_of),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def canonical_json(payload: Mapping[str, object]) -> str:
    """Serialise a payload deterministically, leaving out the ``created`` stamp."""
    filtered = {key: value for key, value in payload.items() if key != "created"}
    return json.dumps(filtered, sort_keys=True, ensure_ascii=False, indent=2)


def content_hash(payload: Mapping[str, object]) -> str:
    """Return the SHA256 of the canonical payload, ignoring ``created``."""
    return text_sha256(canonical_json(payload))


__all__ = [
    "DEFAULT_PHASH_THRESHOLD",
    "SPLIT_VERSION",
    "SplitResult",
    "UnionFind",
    "build_groups",
    "canonical_json",
    "content_hash",
    "exact_duplicate_pairs",
    "near_duplicate_pairs",
    "phash_to_uint64",
    "split_payload",
    "stratified_group_split",
]
