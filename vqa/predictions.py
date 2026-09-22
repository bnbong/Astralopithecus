"""예측 확률 파일의 형식, 검증, 원자적 저장을 담당하는 공통 모듈.

합의된 예측 파일 형식 (CSV, UTF-8):

    id,p_a,p_b,p_c,p_d,run_id

- 한 문항당 한 행이며 `p_a`~`p_d`는 네 선택지 logit에 softmax를 적용한 확률이다.
- `run_id`는 파일 전체에서 하나여야 한다.
- 같은 이름의 `<name>.manifest.json` sidecar에 실행 메타데이터를 남긴다.

추론 장비는 shard 파일(`preds.shard{i}of{n}.csv`)에 이어쓰며 중단·재개될 수 있으므로
부분 파일은 정상 상태로 취급한다. 부분 파일 읽기는 `load_partial_predictions`를 쓴다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LETTERS: tuple[str, ...] = ("a", "b", "c", "d")
PROB_COLUMNS: tuple[str, ...] = ("p_a", "p_b", "p_c", "p_d")
PRED_COLUMNS: tuple[str, ...] = ("id", *PROB_COLUMNS, "run_id")

#: 팀원 도구(experiments/eval_tools/infer_val.py, experiments/eval_tools/eval_val.py)가 쓰는 확률 컬럼 이름.
TEAMMATE_PROB_COLUMNS: tuple[str, ...] = ("prob_a", "prob_b", "prob_c", "prob_d")

#: 팀원 파일에만 있고 vqa 형식에는 없는 컬럼. alias 변환 때만 버린다.
TEAMMATE_ONLY_COLUMNS: tuple[str, ...] = ("pred", "answer")

#: run_id가 없는 팀원 파일을 읽을 때 채워 넣는 기본값.
DEFAULT_RUN_ID = "unknown"

#: 확률 합이 1에서 벗어나는 것을 허용하는 기본 오차.
DEFAULT_TOLERANCE = 1e-3

#: 파일을 sha256으로 읽을 때 쓰는 블록 크기.
_HASH_BLOCK = 1 << 20


class PredictionError(ValueError):
    """예측 파일이 합의된 형식을 지키지 않을 때 발생한다."""


@dataclass
class PartialLoadReport:
    """부분 예측 파일을 읽으면서 버린 행들의 요약."""

    path: Path
    total_rows: int = 0
    kept_rows: int = 0
    truncated_tail: bool = False
    malformed_rows: list[int] = field(default_factory=list)
    bad_prob_rows: list[int] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)

    @property
    def dropped_rows(self) -> int:
        return self.total_rows - self.kept_rows

    def describe(self) -> str:
        parts = [
            f"{self.path}: {self.kept_rows}/{self.total_rows} 행 사용",
        ]
        if self.truncated_tail:
            parts.append("마지막 행이 잘려 있어 버림")
        if self.malformed_rows:
            parts.append(f"형식 오류 행 {len(self.malformed_rows)}개")
        if self.bad_prob_rows:
            parts.append(f"확률 오류 행 {len(self.bad_prob_rows)}개")
        if self.duplicate_ids:
            parts.append(f"중복 id {len(self.duplicate_ids)}개(마지막 값 사용)")
        if len(self.run_ids) > 1:
            parts.append(f"run_id 여러 개: {sorted(self.run_ids)}")
        return ", ".join(parts)


# --------------------------------------------------------------------------
# 파일 입출력 기본 도구
# --------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    """파일 전체의 sha256 16진 문자열을 반환한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_BLOCK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """바이트열의 sha256 16진 문자열을 반환한다."""
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """같은 디렉터리에 임시 파일을 쓴 뒤 `os.replace`로 바꿔 끼운다.

    `os.replace`는 Windows에서도 대상 파일을 덮어쓰므로 세 운영체제에서 모두 동작한다.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(target.parent),
        prefix=target.name + ".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | os.PathLike[str], text: str, encoding: str = "utf-8"
) -> Path:
    """텍스트를 원자적으로 저장한다."""
    return atomic_write_bytes(path, text.encode(encoding))


def manifest_path_for(path: str | os.PathLike[str]) -> Path:
    """예측 파일 경로에 대응하는 manifest sidecar 경로를 만든다."""
    target = Path(path)
    return target.with_name(target.name + ".manifest.json")


def build_manifest(
    *,
    run_id: str,
    csv_path: str | os.PathLike[str] | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    adapter: str | None = None,
    precision: str | None = None,
    prompt_id: str | None = None,
    prompt_hash: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    seconds_per_item: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """run manifest 사전을 만든다. `csv_path`를 주면 sha256을 함께 기록한다."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter": adapter,
        "precision": precision,
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        # 해상도 하한과 상한을 둘 다 남긴다. min_pixels만 올린 확대본 설정과 기본 설정은
        # max_pixels가 같아서, 하한을 적지 않으면 두 실행을 구분할 수 없다.
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "csv_sha256": sha256_file(csv_path) if csv_path is not None else None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds_per_item": seconds_per_item,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | os.PathLike[str], manifest: dict[str, Any]) -> Path:
    """manifest를 sidecar 파일로 원자적으로 저장한다."""
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(manifest_path_for(path), text)


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """sidecar manifest를 읽는다. 없으면 None을 반환한다."""
    sidecar = manifest_path_for(path)
    if not sidecar.exists():
        return None
    with open(sidecar, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# 읽기와 검증
# --------------------------------------------------------------------------


def _read_raw_csv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """BOM을 허용하면서 예측 CSV를 읽는다."""
    target = Path(path)
    if not target.exists():
        raise PredictionError(f"예측 파일이 없습니다: {target}")
    try:
        frame = pd.read_csv(
            target,
            encoding="utf-8-sig",
            dtype={"id": "string", "run_id": "string"},
        )
    except pd.errors.EmptyDataError as exc:
        raise PredictionError(f"예측 파일이 비어 있습니다: {target}") from exc
    except Exception as exc:  # pragma: no cover - pandas 파싱 오류 전달용
        raise PredictionError(f"예측 파일을 읽지 못했습니다: {target}: {exc}") from exc
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _apply_teammate_aliases(frame: pd.DataFrame, *, default_run_id: str) -> tuple[pd.DataFrame, bool]:
    """팀원 형식(`id,pred,prob_a..prob_d`)을 vqa 형식으로 맞춘다.

    `p_a`~`p_d`가 이미 있으면 아무것도 하지 않는다. `prob_*`만 있을 때 이름을 바꾸고,
    `pred`처럼 팀원 도구에만 있는 컬럼을 버리고, `run_id`가 없으면 기본값으로 채운다.

    Returns:
        (변환한 DataFrame, 변환이 일어났는지 여부)
    """
    has_vqa = all(col in frame.columns for col in PROB_COLUMNS)
    has_teammate = all(col in frame.columns for col in TEAMMATE_PROB_COLUMNS)
    if has_vqa or not has_teammate:
        return frame, False

    frame = frame.rename(columns=dict(zip(TEAMMATE_PROB_COLUMNS, PROB_COLUMNS)))
    drop = [col for col in TEAMMATE_ONLY_COLUMNS if col in frame.columns]
    if drop:
        frame = frame.drop(columns=drop)
    if "run_id" not in frame.columns:
        frame["run_id"] = pd.array([str(default_run_id)] * len(frame), dtype="string")
    return frame, True


def _check_columns(frame: pd.DataFrame, source: str) -> None:
    missing = [col for col in PRED_COLUMNS if col not in frame.columns]
    if missing:
        raise PredictionError(f"{source}: 필수 컬럼이 없습니다: {missing}")
    unnamed = [col for col in frame.columns if col.startswith("Unnamed:")]
    if unnamed:
        raise PredictionError(
            f"{source}: 이름 없는 추가 index 컬럼이 있습니다: {unnamed}. "
            "저장할 때 index=False를 사용하세요."
        )
    extra = [col for col in frame.columns if col not in PRED_COLUMNS]
    if extra:
        raise PredictionError(f"{source}: 예상하지 않은 컬럼이 있습니다: {extra}")


def load_predictions(
    path: str | os.PathLike[str],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    renormalize: bool = False,
    default_run_id: str = DEFAULT_RUN_ID,
) -> pd.DataFrame:
    """완결된 예측 파일을 읽고 형식을 검증한다.

    검사 항목: 컬럼 구성, 추가 index 컬럼 없음, 확률의 유한성과 범위, 행 합이 1에 근접,
    id 중복 없음, run_id 단일성.

    Args:
        path: 예측 CSV 경로.
        tolerance: 확률 합이 1에서 벗어나도 되는 허용 오차.
        renormalize: True면 검증 후 각 행의 확률을 합이 정확히 1이 되도록 다시 나눈다.
        default_run_id: 팀원 형식 파일에 `run_id`가 없을 때 채워 넣을 값.

    Returns:
        `id, p_a, p_b, p_c, p_d, run_id` 컬럼을 이 순서로 가진 DataFrame.

    Raises:
        PredictionError: 어느 검사든 통과하지 못했을 때.
    """
    source = str(path)
    frame = _read_raw_csv(path)
    frame, _aliased = _apply_teammate_aliases(frame, default_run_id=default_run_id)
    _check_columns(frame, source)
    frame = frame.loc[:, list(PRED_COLUMNS)].copy()

    if len(frame) == 0:
        raise PredictionError(f"{source}: 예측 행이 하나도 없습니다.")

    ids = frame["id"]
    if ids.isna().any():
        bad = frame.index[ids.isna()].tolist()[:5]
        raise PredictionError(f"{source}: id가 비어 있는 행이 있습니다(행 {bad}).")
    frame["id"] = ids.astype(str).str.strip()
    if (frame["id"] == "").any():
        raise PredictionError(f"{source}: id가 공백인 행이 있습니다.")

    duplicated = frame["id"][frame["id"].duplicated(keep=False)].unique().tolist()
    if duplicated:
        raise PredictionError(
            f"{source}: id가 중복됩니다({len(duplicated)}개): {sorted(duplicated)[:5]}"
        )

    run_ids = frame["run_id"]
    if run_ids.isna().any():
        raise PredictionError(f"{source}: run_id가 비어 있는 행이 있습니다.")
    frame["run_id"] = run_ids.astype(str).str.strip()
    unique_runs = sorted(frame["run_id"].unique().tolist())
    if len(unique_runs) != 1:
        raise PredictionError(
            f"{source}: run_id는 하나여야 하는데 {len(unique_runs)}개입니다: {unique_runs[:5]}"
        )

    for col in PROB_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        bad_rows = np.where(~np.isfinite(probs).all(axis=1))[0][:5]
        bad_ids = frame["id"].iloc[bad_rows].tolist()
        raise PredictionError(
            f"{source}: 확률이 NaN이거나 무한대인 행이 있습니다: id={bad_ids}"
        )
    if (probs < -tolerance).any() or (probs > 1.0 + tolerance).any():
        bad_rows = np.where((probs < -tolerance) | (probs > 1.0 + tolerance))[0][:5]
        bad_ids = frame["id"].iloc[np.unique(bad_rows)].tolist()
        raise PredictionError(
            f"{source}: 확률이 [0,1] 범위를 벗어난 행이 있습니다: id={bad_ids}"
        )

    sums = probs.sum(axis=1)
    off = np.abs(sums - 1.0) > tolerance
    if off.any():
        bad_rows = np.where(off)[0][:5]
        detail = [
            f"{frame['id'].iloc[i]}={sums[i]:.6f}" for i in bad_rows
        ]
        raise PredictionError(
            f"{source}: 확률 합이 1과 다른 행이 {int(off.sum())}개 있습니다"
            f"(허용 오차 {tolerance}): {detail}"
        )

    if renormalize:
        frame.loc[:, list(PROB_COLUMNS)] = probs / sums[:, None]

    return frame.reset_index(drop=True)


def load_partial_predictions(
    path: str | os.PathLike[str],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[pd.DataFrame, PartialLoadReport]:
    """중단된 추론이 남긴 부분 파일을 안전하게 읽는다.

    추론 장비는 행 단위로 이어쓰므로 마지막 행이 잘려 있을 수 있고, 재개 과정에서
    같은 id가 두 번 기록될 수 있다. 이 함수는 예외를 던지는 대신 사용할 수 없는 행을
    버리고 무엇을 버렸는지 보고한다. 같은 id가 여러 번 나오면 마지막 값을 쓴다.

    Returns:
        (검증을 통과한 행만 담은 DataFrame, PartialLoadReport)
    """
    target = Path(path)
    report = PartialLoadReport(path=target)
    if not target.exists():
        return _empty_predictions(), report

    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        raw_text = handle.read()
    if not raw_text.strip():
        return _empty_predictions(), report

    # 마지막 줄이 개행으로 끝나지 않으면 쓰다 만 행으로 본다.
    ends_clean = raw_text.endswith("\n")
    lines = raw_text.splitlines()
    if not ends_clean and len(lines) > 1:
        report.truncated_tail = True
        lines = lines[:-1]

    reader = csv.reader(lines)
    try:
        header = next(reader)
    except StopIteration:
        return _empty_predictions(), report
    header = [col.strip() for col in header]
    missing = [col for col in PRED_COLUMNS if col not in header]
    if missing:
        raise PredictionError(f"{target}: 필수 컬럼이 없습니다: {missing}")
    index_of = {col: header.index(col) for col in PRED_COLUMNS}

    records: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(reader, start=2):
        report.total_rows += 1
        if len(row) != len(header):
            report.malformed_rows.append(line_no)
            continue
        row_id = row[index_of["id"]].strip()
        run_id = row[index_of["run_id"]].strip()
        if not row_id or not run_id:
            report.malformed_rows.append(line_no)
            continue
        try:
            values = [float(row[index_of[col]]) for col in PROB_COLUMNS]
        except ValueError:
            report.bad_prob_rows.append(line_no)
            continue
        if not all(math.isfinite(v) for v in values):
            report.bad_prob_rows.append(line_no)
            continue
        if any(v < -tolerance or v > 1.0 + tolerance for v in values):
            report.bad_prob_rows.append(line_no)
            continue
        if abs(sum(values) - 1.0) > tolerance:
            report.bad_prob_rows.append(line_no)
            continue
        if row_id in records:
            report.duplicate_ids.append(row_id)
        if run_id not in report.run_ids:
            report.run_ids.append(run_id)
        records[row_id] = {
            "id": row_id,
            **{col: val for col, val in zip(PROB_COLUMNS, values)},
            "run_id": run_id,
        }

    report.kept_rows = len(records)
    if not records:
        return _empty_predictions(), report
    frame = pd.DataFrame(list(records.values()), columns=list(PRED_COLUMNS))
    frame["id"] = frame["id"].astype("string")
    frame["run_id"] = frame["run_id"].astype("string")
    return frame.reset_index(drop=True), report


def _empty_predictions() -> pd.DataFrame:
    frame = pd.DataFrame({col: [] for col in PRED_COLUMNS})
    frame["id"] = frame["id"].astype("string")
    frame["run_id"] = frame["run_id"].astype("string")
    for col in PROB_COLUMNS:
        frame[col] = frame[col].astype(float)
    return frame


# --------------------------------------------------------------------------
# 쓰기
# --------------------------------------------------------------------------


def predictions_to_csv_text(frame: pd.DataFrame, *, float_format: str = "%.6g") -> str:
    """예측 DataFrame을 합의된 컬럼 순서의 CSV 문자열로 만든다."""
    missing = [col for col in PRED_COLUMNS if col not in frame.columns]
    if missing:
        raise PredictionError(f"저장할 DataFrame에 컬럼이 없습니다: {missing}")
    ordered = frame.loc[:, list(PRED_COLUMNS)]
    return ordered.to_csv(index=False, lineterminator="\n", float_format=float_format)


def write_predictions(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    validate: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    float_format: str = "%.6g",
    manifest: dict[str, Any] | None = None,
) -> Path:
    """예측 DataFrame을 원자적으로 저장하고, 필요하면 manifest도 함께 남긴다.

    `validate=True`면 저장한 파일을 다시 읽어 형식 검사를 통과하는지 확인한다.
    manifest의 `csv_sha256`은 저장이 끝난 실제 파일에서 계산한다.
    """
    text = predictions_to_csv_text(frame, float_format=float_format)
    target = atomic_write_bytes(path, text.encode("utf-8"))
    if validate:
        load_predictions(target, tolerance=tolerance)
    if manifest is not None:
        manifest = dict(manifest)
        manifest["csv_sha256"] = sha256_file(target)
        write_manifest(target, manifest)
    return target


def append_predictions(
    rows: Iterable[dict[str, Any]],
    path: str | os.PathLike[str],
    *,
    float_format: str = "%.6g",
) -> int:
    """추론 중 행 단위로 이어쓴다. 파일이 없으면 헤더를 먼저 쓴다.

    각 행을 쓴 뒤 flush와 fsync를 해서 중단되어도 마지막 완결 행까지는 남게 한다.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    need_header = not target.exists() or target.stat().st_size == 0
    written = 0
    with open(target, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if need_header:
            writer.writerow(PRED_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["id"],
                    *(float_format % float(row[col]) for col in PROB_COLUMNS),
                    row["run_id"],
                ]
            )
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written


# --------------------------------------------------------------------------
# 확률에서 답과 margin 만들기
# --------------------------------------------------------------------------


def decide_answers(frame: pd.DataFrame) -> pd.DataFrame:
    """확률에서 최종 답 letter와 margin을 계산해 컬럼으로 붙인다.

    margin은 1등 확률과 2등 확률의 차이다. 동률이면 a, b, c, d 순서로 앞선 letter를 쓴다.
    최종 답과 margin은 항상 같은 확률 행렬에서 계산한다.
    """
    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    if probs.size == 0:
        out = frame.copy()
        out["answer"] = pd.Series(dtype="string")
        out["margin"] = pd.Series(dtype=float)
        return out
    best = probs.argmax(axis=1)  # argmax는 동률에서 가장 앞 index를 고른다.
    ordered = np.sort(probs, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    out = frame.copy()
    out["answer"] = pd.array([LETTERS[i] for i in best], dtype="string")
    out["margin"] = margin
    return out


def average_predictions(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float],
    *,
    run_id: str,
) -> pd.DataFrame:
    """동일한 id 집합을 가진 예측들의 가중 평균을 낸다.

    가중치는 합이 1이 되도록 정규화하고, 평균 뒤에도 각 행의 확률 합을 1로 맞춘다.

    Raises:
        PredictionError: id 집합이 서로 다르거나 가중치 개수가 맞지 않을 때.
    """
    if len(frames) != len(weights):
        raise PredictionError("예측 파일 수와 가중치 수가 다릅니다.")
    if not frames:
        raise PredictionError("평균낼 예측이 없습니다.")
    total = float(sum(weights))
    if total <= 0:
        raise PredictionError("가중치의 합이 0 이하입니다.")

    base = frames[0].set_index("id").sort_index()
    base_ids = set(base.index)
    accum = np.zeros((len(base), len(PROB_COLUMNS)), dtype=float)
    for frame, weight in zip(frames, weights):
        indexed = frame.set_index("id").sort_index()
        if set(indexed.index) != base_ids:
            only_a = sorted(base_ids - set(indexed.index))[:5]
            only_b = sorted(set(indexed.index) - base_ids)[:5]
            raise PredictionError(
                "결합할 예측들의 id 집합이 다릅니다. "
                f"첫 파일에만 있는 id 예시={only_a}, 다른 파일에만 있는 id 예시={only_b}"
            )
        accum += (weight / total) * indexed[list(PROB_COLUMNS)].to_numpy(dtype=float)

    accum = accum / accum.sum(axis=1, keepdims=True)
    out = pd.DataFrame(accum, columns=list(PROB_COLUMNS))
    out.insert(0, "id", pd.array(base.index.astype(str), dtype="string"))
    out["run_id"] = pd.array([run_id] * len(out), dtype="string")
    return out.loc[:, list(PRED_COLUMNS)].reset_index(drop=True)


def read_id_list(path: str | os.PathLike[str], *, column: str = "id") -> list[str]:
    """기대 id 목록을 읽는다. `.csv`면 `column` 컬럼을, 아니면 한 줄에 하나로 읽는다."""
    target = Path(path)
    if not target.exists():
        raise PredictionError(f"id 목록 파일이 없습니다: {target}")
    if target.suffix.lower() == ".csv":
        frame = pd.read_csv(target, encoding="utf-8-sig", dtype="string")
        frame.columns = [str(col).strip() for col in frame.columns]
        if column not in frame.columns:
            raise PredictionError(
                f"{target}: '{column}' 컬럼이 없습니다. 있는 컬럼: {list(frame.columns)}"
            )
        values = frame[column].dropna().astype(str).str.strip().tolist()
    else:
        with open(target, "r", encoding="utf-8-sig") as handle:
            values = [line.strip() for line in handle]
    return [value for value in values if value]
