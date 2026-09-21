"""checkpoint 저장·검증·재개.

Colab 세션은 언제든 끊기고 Drive FUSE는 쓰다 만 파일을 남길 수 있다. 그래서
저장은 항상 다음 순서를 지킨다.

1. LOCAL_SCRATCH의 임시 디렉터리에 전부 쓰고 fsync한다.
2. 각 파일을 다시 읽어 파싱되는지 보고 해시가 그대로인지 확인한다.
3. RUN_ROOT 아래 임시 디렉터리로 복사한 뒤 사본의 해시를 다시 검증한다.
4. 파일 해시를 담은 COMPLETE를 마지막에 쓴다.
5. 임시 디렉터리를 step_XXXXXX로 rename한다.
6. latest.json을 원자적으로 갱신한다.
7. keep_last 개만 남기고 정리한다.

재개는 COMPLETE 해시가 실제로 맞는 가장 최근 checkpoint를 고른다. 최신 것이 깨졌으면
그 다음 것으로 물러난다.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import runtime

COMPLETE_NAME = "COMPLETE"
LATEST_NAME = "latest.json"
CHECKPOINT_PREFIX = "step_"
STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
RNG_NAME = "rng.pt"
ADAPTER_DIR = "adapter"
PROCESSOR_DIR = "processor"


class CheckpointError(RuntimeError):
    """checkpoint 저장/검증/재개 실패."""


class ResumeRefused(CheckpointError):
    """설정이나 split이 달라 재개를 거부했을 때."""


# --------------------------------------------------------------------------
# trainer state
# --------------------------------------------------------------------------


@dataclass
class TrainerState:
    """재개에 필요한 학습 진행 상태 전부."""

    epoch: int = 0
    global_step: int = 0
    samples_consumed_in_epoch: int = 0
    data_order_seed: int = 42
    config_hash: str = ""
    split_hash: str = ""
    loss_ema: float | None = None
    total_steps: int | None = None
    total_samples_seen: int = 0
    run_id: str = ""
    created: str = field(default_factory=runtime.now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainerState":
        known = {f: payload.get(f) for f in cls.__dataclass_fields__ if f in payload}
        state = cls(**known)  # type: ignore[arg-type]
        if state.extra is None:
            state.extra = {}
        return state


# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------


def checkpoints_dir(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / "checkpoints"


def step_dir_name(step: int) -> str:
    return f"{CHECKPOINT_PREFIX}{int(step):06d}"


def parse_step(path: str | os.PathLike[str]) -> int | None:
    name = Path(path).name
    if not name.startswith(CHECKPOINT_PREFIX):
        return None
    try:
        return int(name[len(CHECKPOINT_PREFIX) :])
    except ValueError:
        return None


def list_checkpoints(run_dir: str | os.PathLike[str]) -> list[Path]:
    """step 번호 오름차순 checkpoint 디렉터리 목록 (검증 전)."""
    root = checkpoints_dir(run_dir)
    if not root.exists():
        return []
    entries = [p for p in root.iterdir() if p.is_dir() and parse_step(p) is not None]
    return sorted(entries, key=lambda p: parse_step(p) or -1)


# --------------------------------------------------------------------------
# 해시와 검증
# --------------------------------------------------------------------------


def hash_tree(directory: str | os.PathLike[str], *, skip: tuple[str, ...] = (COMPLETE_NAME,)) -> dict[str, str]:
    """디렉터리 안 모든 파일의 상대경로 -> sha256."""
    root = Path(directory)
    out: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(root).as_posix()
        if rel in skip:
            continue
        out[rel] = runtime.sha256_file(item)
    return out


def verify_checkpoint(directory: str | os.PathLike[str]) -> tuple[bool, str]:
    """COMPLETE에 적힌 해시가 실제 파일과 맞는지 확인한다."""
    root = Path(directory)
    complete = root / COMPLETE_NAME
    if not complete.exists():
        return False, "COMPLETE 없음"
    payload = runtime.read_json_or_none(complete)
    if not isinstance(payload, dict) or "files" not in payload:
        return False, "COMPLETE를 읽을 수 없음"
    files = payload.get("files") or {}
    if not files:
        return False, "COMPLETE에 파일 목록이 없음"
    for rel, expected in files.items():
        target = root / rel
        if not target.exists():
            return False, f"파일 없음: {rel}"
        try:
            actual = runtime.sha256_file(target)
        except OSError as exc:
            return False, f"파일을 읽을 수 없음: {rel} ({exc})"
        if actual != expected:
            return False, f"해시 불일치: {rel}"
    present = set(hash_tree(root))
    extra = present - set(files)
    if extra:
        return False, f"COMPLETE에 없는 파일이 있음: {sorted(extra)[:3]}"
    return True, "ok"


def _torch_load(path: str | os.PathLike[str]) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 아주 오래된 torch
        return torch.load(path, map_location="cpu")


def _verify_reload(directory: Path) -> None:
    """상태 파일을 다시 읽어 파싱되는지와 해시가 변하지 않았는지 본다."""
    before = hash_tree(directory)
    for rel in sorted(before):
        target = directory / rel
        if rel.endswith(".pt"):
            try:
                _torch_load(target)
            except Exception as exc:
                raise CheckpointError(f"저장한 상태 파일을 다시 읽지 못했습니다: {rel}: {exc}") from exc
        elif rel.endswith(".json"):
            if runtime.read_json_or_none(target) is None:
                raise CheckpointError(f"저장한 JSON을 다시 읽지 못했습니다: {rel}")
    after = hash_tree(directory)
    if before != after:
        raise CheckpointError("재검증 중 파일 해시가 바뀌었습니다.")


def _save_torch(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        torch.save(obj, handle)
        runtime.fsync_file(handle)


# --------------------------------------------------------------------------
# 저장
# --------------------------------------------------------------------------


def save_checkpoint(
    *,
    run_dir: str | os.PathLike[str],
    scratch_dir: str | os.PathLike[str],
    state: TrainerState,
    adapter_saver: Callable[[Path], None],
    optimizer: Any = None,
    scheduler: Any = None,
    keep_last: int = 2,
    rng: dict[str, Any] | None = None,
) -> Path:
    """checkpoint 하나를 저장하고 최종 디렉터리 경로를 돌려준다."""
    step = int(state.global_step)
    run_path = Path(run_dir)
    ckpt_root = runtime.ensure_dir(checkpoints_dir(run_path))
    scratch = runtime.ensure_dir(Path(scratch_dir))

    build = scratch / f".build_{step_dir_name(step)}_{os.getpid()}"
    if build.exists():
        shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)

    try:
        # 1. LOCAL_SCRATCH에 전부 쓴다.
        adapter_saver(runtime.ensure_dir(build / ADAPTER_DIR))
        if optimizer is not None:
            _save_torch(optimizer.state_dict(), build / OPTIMIZER_NAME)
        if scheduler is not None:
            _save_torch(scheduler.state_dict(), build / SCHEDULER_NAME)
        _save_torch(rng if rng is not None else runtime.rng_state(), build / RNG_NAME)
        state_payload = state.to_dict()
        state_payload["created"] = runtime.now_iso()
        runtime.atomic_write_json(build / STATE_NAME, state_payload)
        runtime.fsync_dir(build)

        # 2. 다시 읽어 확인한다.
        _verify_reload(build)
        local_hashes = hash_tree(build)

        # 3. RUN_ROOT의 임시 디렉터리로 복사하고 사본을 검증한다.
        staged = ckpt_root / f".tmp_{step_dir_name(step)}_{os.getpid()}"
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        runtime.copy_tree_fsync(build, staged)
        copied_hashes = hash_tree(staged)
        if copied_hashes != local_hashes:
            raise CheckpointError(
                "RUN_ROOT 사본의 해시가 로컬과 다릅니다. Drive 쓰기가 실패했을 수 있습니다."
            )

        # 4. COMPLETE를 마지막에 쓴다.
        complete_payload = {
            "step": step,
            "created": runtime.now_iso(),
            "files": copied_hashes,
            "state": state_payload,
        }
        runtime.atomic_write_json(staged / COMPLETE_NAME, complete_payload)
        runtime.fsync_dir(staged)

        # 5. rename.
        final = ckpt_root / step_dir_name(step)
        runtime.replace_dir(staged, final)

        ok, reason = verify_checkpoint(final)
        if not ok:
            raise CheckpointError(f"저장 직후 검증 실패: {reason}")

        # 6. latest.json.
        write_latest(run_path, final, state_payload)

        # 7. 정리.
        prune_checkpoints(run_path, keep_last=keep_last)
        return final
    finally:
        shutil.rmtree(build, ignore_errors=True)


def write_latest(run_dir: str | os.PathLike[str], checkpoint: Path, state_payload: dict[str, Any]) -> Path:
    payload = {
        "step": parse_step(checkpoint),
        "dir": checkpoint.name,
        "path": str(checkpoint),
        "created": runtime.now_iso(),
        "state": state_payload,
    }
    return runtime.atomic_write_json(checkpoints_dir(run_dir) / LATEST_NAME, payload)


def read_latest(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    return runtime.read_json_or_none(checkpoints_dir(run_dir) / LATEST_NAME)


def prune_checkpoints(run_dir: str | os.PathLike[str], *, keep_last: int = 2) -> list[Path]:
    """완결된 checkpoint를 최근 keep_last개만 남긴다. 임시 디렉터리도 치운다."""
    root = checkpoints_dir(run_dir)
    if not root.exists():
        return []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(".tmp_"):
            shutil.rmtree(item, ignore_errors=True)

    complete: list[Path] = []
    incomplete: list[Path] = []
    for path in list_checkpoints(run_dir):
        ok, _ = verify_checkpoint(path)
        (complete if ok else incomplete).append(path)

    removed: list[Path] = []
    if keep_last > 0 and len(complete) > keep_last:
        for path in complete[: len(complete) - keep_last]:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    # 완결된 것이 충분히 있으면 깨진 잔해도 치운다.
    if len(complete) >= 1:
        for path in incomplete:
            step = parse_step(path)
            newest = parse_step(complete[-1])
            if step is not None and newest is not None and step <= newest:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
    return removed


# --------------------------------------------------------------------------
# 재개
# --------------------------------------------------------------------------


def find_resume_checkpoint(run_dir: str | os.PathLike[str]) -> tuple[Path | None, list[str]]:
    """검증을 통과하는 가장 최근 checkpoint. 건너뛴 이유도 함께 돌려준다."""
    notes: list[str] = []
    for path in reversed(list_checkpoints(run_dir)):
        ok, reason = verify_checkpoint(path)
        if ok:
            return path, notes
        notes.append(f"{path.name}: {reason}")
    return None, notes


def read_state(checkpoint: str | os.PathLike[str]) -> TrainerState:
    payload = runtime.read_json_or_none(Path(checkpoint) / STATE_NAME)
    if payload is None:
        raise CheckpointError(f"{checkpoint}: trainer_state.json을 읽을 수 없습니다.")
    return TrainerState.from_dict(payload)


def load_checkpoint(
    checkpoint: str | os.PathLike[str],
    *,
    adapter_loader: Callable[[Path], None] | None = None,
    optimizer: Any = None,
    scheduler: Any = None,
    restore_rng: bool = True,
    expected_config_hash: str | None = None,
    expected_split_hash: str | None = None,
    force: bool = False,
) -> TrainerState:
    """checkpoint에서 adapter/optimizer/scheduler/RNG/state를 복원한다."""
    path = Path(checkpoint)
    ok, reason = verify_checkpoint(path)
    if not ok:
        raise CheckpointError(f"{path}: 검증 실패 ({reason})")

    state = read_state(path)
    mismatches = []
    if expected_config_hash is not None and state.config_hash and state.config_hash != expected_config_hash:
        mismatches.append(f"config_hash {state.config_hash[:12]} != {expected_config_hash[:12]}")
    if expected_split_hash is not None and state.split_hash and state.split_hash != expected_split_hash:
        mismatches.append(f"split_hash {state.split_hash[:12]} != {expected_split_hash[:12]}")
    if mismatches and not force:
        raise ResumeRefused(
            "checkpoint의 설정이 지금 설정과 다릅니다: "
            + "; ".join(mismatches)
            + ". 정말 이어서 하려면 --force-resume을 쓰세요."
        )

    if adapter_loader is not None:
        adapter_loader(path / ADAPTER_DIR)
    if optimizer is not None and (path / OPTIMIZER_NAME).exists():
        optimizer.load_state_dict(_torch_load(path / OPTIMIZER_NAME))
    if scheduler is not None and (path / SCHEDULER_NAME).exists():
        scheduler.load_state_dict(_torch_load(path / SCHEDULER_NAME))
    if restore_rng and (path / RNG_NAME).exists():
        runtime.set_rng_state(_torch_load(path / RNG_NAME))
    return state


# --------------------------------------------------------------------------
# 최종 adapter
# --------------------------------------------------------------------------


def final_adapter_dir(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / "final_adapter"


def final_processor_dir(run_dir: str | os.PathLike[str]) -> Path:
    """학습이 끝났을 때 한 번만 남기는 processor 사본 위치.

    추론은 이 디렉터리를 읽지 않는다. base 모델과 config가 그대로 진실이고,
    이것은 나중에 어떤 processor 설정으로 학습했는지 확인하기 위한 기록이다.
    step checkpoint에는 넣지 않는다. 같은 파일을 수십 번 복사할 이유가 없다.
    """
    return final_adapter_dir(run_dir) / PROCESSOR_DIR


def training_is_finished(run_dir: str | os.PathLike[str]) -> bool:
    """final_adapter/COMPLETE가 검증을 통과하면 학습이 끝난 run이다."""
    final = final_adapter_dir(run_dir)
    if not (final / COMPLETE_NAME).exists():
        return False
    ok, _ = verify_checkpoint(final)
    return ok


def save_final_adapter(
    *,
    run_dir: str | os.PathLike[str],
    scratch_dir: str | os.PathLike[str],
    adapter_saver: Callable[[Path], None],
    state: TrainerState,
    processor_saver: Callable[[Path], None] | None = None,
) -> Path:
    """최종 adapter를 checkpoint와 같은 안전 절차로 저장한다.

    `processor_saver`를 주면 `final_adapter/processor`에 processor도 한 번만 남긴다.
    COMPLETE를 쓰기 전에 넣어야 해시 목록에 포함되므로 여기서 함께 처리한다.
    """
    run_path = Path(run_dir)
    scratch = runtime.ensure_dir(Path(scratch_dir))
    build = scratch / f".build_final_{os.getpid()}"
    if build.exists():
        shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    try:
        adapter_saver(build)
        if processor_saver is not None:
            processor_saver(runtime.ensure_dir(build / PROCESSOR_DIR))
        runtime.atomic_write_json(build / STATE_NAME, state.to_dict())
        runtime.fsync_dir(build)
        local_hashes = hash_tree(build)

        staged = run_path / f".tmp_final_adapter_{os.getpid()}"
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        runtime.copy_tree_fsync(build, staged)
        if hash_tree(staged) != local_hashes:
            raise CheckpointError("final_adapter 사본의 해시가 다릅니다.")
        runtime.atomic_write_json(
            staged / COMPLETE_NAME,
            {"created": runtime.now_iso(), "files": hash_tree(staged), "state": state.to_dict()},
        )
        runtime.fsync_dir(staged)
        final = runtime.replace_dir(staged, final_adapter_dir(run_path))
        ok, reason = verify_checkpoint(final)
        if not ok:
            raise CheckpointError(f"final_adapter 검증 실패: {reason}")
        return final
    finally:
        shutil.rmtree(build, ignore_errors=True)
