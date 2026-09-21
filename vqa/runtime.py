"""실행 환경 공통 유틸리티.

경로 환경변수 해석, 원자적 파일 쓰기, 해시, RNG 상태 저장/복원, 장치 정보처럼
train/predict/preflight/jobs가 모두 쓰는 저수준 기능만 모아 둔다.

환경변수:
    VQA_DATA_ROOT      train.csv / train/ 등이 있는 데이터 루트
    VQA_MODEL_ROOT     로컬 모델 디렉터리 또는 HF 캐시 루트 (선택)
    VQA_RUN_ROOT       run 산출물을 영구 보관하는 루트 (Colab에서는 Drive)
    VQA_LOCAL_SCRATCH  빠른 로컬 디스크 (Colab에서는 /content/scratch)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HASH_BLOCK = 1 << 20


# --------------------------------------------------------------------------
# 시간 / 문자열
# --------------------------------------------------------------------------


def now_iso() -> str:
    """초 단위 UTC ISO 문자열."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_tag() -> str:
    """파일 이름에 쓰는 UTC 타임스탬프 태그."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def format_seconds(seconds: float | None) -> str:
    """초를 사람이 읽는 h/m/s 문자열로 바꾼다."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------


def _env_path(name: str, default: str | os.PathLike[str] | None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


def data_root(override: str | os.PathLike[str] | None = None) -> Path:
    """이미지와 CSV가 있는 데이터 루트."""
    if override:
        return Path(override).expanduser()
    return _env_path("VQA_DATA_ROOT", Path.cwd())  # type: ignore[return-value]


def model_root(override: str | os.PathLike[str] | None = None) -> Path | None:
    """로컬 모델 디렉터리 루트. 설정하지 않으면 None."""
    if override:
        return Path(override).expanduser()
    return _env_path("VQA_MODEL_ROOT", None)


def run_root(override: str | os.PathLike[str] | None = None) -> Path:
    """run 산출물 루트. 기본값은 CWD/artifacts/runs."""
    if override:
        return Path(override).expanduser()
    return _env_path("VQA_RUN_ROOT", Path.cwd() / "artifacts" / "runs")  # type: ignore[return-value]


def local_scratch(override: str | os.PathLike[str] | None = None) -> Path:
    """checkpoint를 먼저 쓰는 빠른 로컬 디스크."""
    if override:
        return Path(override).expanduser()
    env = _env_path("VQA_LOCAL_SCRATCH", None)
    if env is not None:
        return env
    return Path(tempfile.gettempdir()) / "vqa_scratch"


def run_dir(run_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    """RUN_ROOT/<run_id> 경로."""
    return run_root(root) / run_id


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------
# 해시와 원자적 쓰기
# --------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    """JSON 직렬화 가능한 값의 정규화 해시."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(text.encode("utf-8"))


def fsync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def fsync_dir(path: str | os.PathLike[str]) -> None:
    """디렉터리 엔트리를 디스크에 내린다. Windows에서는 조용히 넘어간다."""
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """같은 디렉터리에 임시 파일을 쓰고 fsync한 뒤 rename한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            fsync_file(handle)
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    fsync_dir(target.parent)
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    return atomic_write_text(path, text)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_json_or_none(path: str | os.PathLike[str]) -> Any | None:
    """읽을 수 없거나 깨진 JSON이면 None을 돌려준다."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except (OSError, json.JSONDecodeError):
        return None


def append_jsonl(path: str | os.PathLike[str], payload: Any) -> None:
    """jsonl 한 줄을 이어쓰고 fsync한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line)
        fsync_file(handle)


def replace_dir(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path:
    """dst를 src로 교체한다. dst가 있으면 먼저 지운다 (Windows 호환)."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        shutil.rmtree(dst_path, ignore_errors=True)
    os.replace(src_path, dst_path)
    fsync_dir(dst_path.parent)
    return dst_path


def copy_tree_fsync(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """디렉터리를 복사하면서 각 파일을 fsync한다. Drive FUSE 대비."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_path.rglob("*")):
        rel = item.relative_to(src_path)
        target = dst_path / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(item, "rb") as reader, open(target, "wb") as writer:
            shutil.copyfileobj(reader, writer, _HASH_BLOCK)
            fsync_file(writer)
    for directory in [dst_path, *[p for p in dst_path.rglob("*") if p.is_dir()]]:
        fsync_dir(directory)


# --------------------------------------------------------------------------
# 난수
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """python / numpy / torch의 전역 시드를 맞춘다."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # numpy가 없어도 동작해야 한다
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rng_state() -> dict[str, Any]:
    """python / numpy / torch(cpu, cuda) RNG 상태를 모은다."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        else:
            state["torch_cuda"] = None
    except Exception:
        state["torch_cpu"] = None
        state["torch_cuda"] = None
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """rng_state()가 만든 상태를 복원한다. 장치가 달라 실패하면 건너뛴다."""
    if state.get("python") is not None:
        random.setstate(tuple_state(state["python"]))
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    try:
        import torch

        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"].cpu().to(dtype=torch_uint8()))
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
    except Exception:
        pass


def torch_uint8():
    import torch

    return torch.uint8


def tuple_state(value: Any) -> tuple:
    """torch.load가 list로 되살린 random.getstate() 값을 tuple로 되돌린다."""
    if isinstance(value, tuple):
        return tuple(tuple_state(v) if isinstance(v, (list, tuple)) else v for v in value)
    if isinstance(value, list):
        return tuple(tuple_state(v) if isinstance(v, (list, tuple)) else v for v in value)
    return value


# --------------------------------------------------------------------------
# 장치 / 환경 리포트
# --------------------------------------------------------------------------


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cuda_sync() -> None:
    """CUDA 커널이 끝날 때까지 기다린다. CUDA가 없으면 아무 일도 하지 않는다."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# --------------------------------------------------------------------------
# TF32
# --------------------------------------------------------------------------


def _set_quietly(fn: Any) -> None:
    """torch 버전에 따라 없을 수 있는 속성을 조용히 건드린다."""
    try:
        fn()
    except Exception:
        pass


def _get_quietly(fn: Any) -> Any:
    try:
        return fn()
    except Exception:
        return None


def tf32_state() -> dict[str, Any]:
    """현재 TF32 관련 torch 플래그 값을 읽는다. torch가 없으면 None만 담긴다."""
    state: dict[str, Any] = {"matmul": None, "cudnn": None, "float32_matmul_precision": None}
    try:
        import torch
    except Exception:
        return state
    state["matmul"] = _get_quietly(lambda: bool(torch.backends.cuda.matmul.allow_tf32))
    state["cudnn"] = _get_quietly(lambda: bool(torch.backends.cudnn.allow_tf32))
    state["float32_matmul_precision"] = _get_quietly(torch.get_float32_matmul_precision)
    return state


def apply_tf32(enabled: bool) -> dict[str, Any]:
    """TF32 matmul 경로를 켜거나 끄고, 실제로 반영된 값을 돌려준다.

    Ampere 이상에서 fp32 matmul을 TF32로 돌리면 같은 코드가 크게 빨라진다. 특히
    Gated DeltaNet의 torch fallback처럼 float32로 캐스팅하는 경로가 섞인 모델에서는
    전체 레이어의 상당수가 fp32 matmul이라 차이가 더 벌어진다.

    CPU 전용 torch에서 플래그를 세팅하는 것 자체는 무해하지만, 버전에 따라 속성이
    없을 수 있으므로 어떤 경우에도 예외를 올리지 않는다.
    """
    enabled = bool(enabled)
    try:
        import torch
    except Exception as exc:
        return {"requested": enabled, "matmul": None, "cudnn": None, "float32_matmul_precision": None, "error": repr(exc)}

    _set_quietly(lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", enabled))
    _set_quietly(lambda: setattr(torch.backends.cudnn, "allow_tf32", enabled))
    _set_quietly(lambda: torch.set_float32_matmul_precision("high" if enabled else "highest"))

    state = tf32_state()
    state["requested"] = enabled
    return state


def tf32_line(state: dict[str, Any] | None) -> str:
    """프로세스 시작 로그에 한 줄로 찍는 TF32 상태."""
    state = state or {}
    return (
        f"[runtime] TF32 matmul={state.get('matmul')} cudnn={state.get('cudnn')} "
        f"float32_matmul_precision={state.get('float32_matmul_precision')}"
    )


def peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / 1024**3
    except Exception:
        return None


def reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def environment_report(paths: dict[str, Any] | None = None) -> dict[str, Any]:
    """preflight와 run_manifest에 남기는 환경 요약."""
    report: dict[str, Any] = {
        "created": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy", "pandas", "PIL")
        },
        "env": {
            key: os.environ.get(key)
            for key in (
                "VQA_DATA_ROOT",
                "VQA_MODEL_ROOT",
                "VQA_RUN_ROOT",
                "VQA_LOCAL_SCRATCH",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    report["tf32"] = tf32_state()
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "name": props.name,
                "total_gb": round(props.total_memory / 1024**3, 2),
                "capability": f"{props.major}.{props.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
    except Exception as exc:
        report["torch"] = {"error": repr(exc)}

    if paths:
        disk = {}
        for key, value in paths.items():
            if value is None:
                disk[key] = None
                continue
            try:
                usage = shutil.disk_usage(str(value))
                disk[key] = {
                    "path": str(value),
                    "exists": Path(value).exists(),
                    "free_gb": round(usage.free / 1024**3, 2),
                    "total_gb": round(usage.total / 1024**3, 2),
                }
            except OSError:
                disk[key] = {"path": str(value), "exists": False}
        report["disk"] = disk
    return report


def is_offline() -> bool:
    """HF 오프라인 모드가 켜져 있는지."""
    return any(
        os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
