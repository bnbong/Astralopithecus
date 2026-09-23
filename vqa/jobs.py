"""분리 실행 프로세스 관리자.

노트북 셀을 멈추거나 브라우저 연결이 끊겨도 학습이 계속되도록, 명령을 부모와 분리된
프로세스로 띄운다. Linux는 새 세션(start_new_session)으로, Windows는
CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS로 띄운다.

`--auto-restart`로 띄우면 job 기록의 pid는 감시 프로세스이고, 실제 작업은 그 자식이다.
그래서 감시 프로세스는 자식 pid(`child_pid`)와 최종 종료 코드(`exit_code`)를 기록에 남기고,
SIGTERM/SIGINT를 받으면 자식에게 그대로 전달한 뒤 재시작하지 않고 끝난다.

GPU는 한 장뿐이라 job을 겹쳐 띄우면 그대로 OOM이 난다. 그래서 `start`는 기본적으로 같은
run-root 아래에 살아 있는 다른 job이 있으면 시작을 거부한다.

job 기록은 run-root 아래에 남고, Colab에서 run-root는 Drive다. 즉 가상 머신이 바뀌어도
기록은 그대로 남아 있고, 새 머신에서 옛 pid를 검사하면 무관한 프로세스가 걸려 살아 있다고
오판한다. 그래서 기록에는 이 기록을 쓴 머신/부팅의 식별자(`machine_id`)를 함께 남기고,
생존 판정은 식별자가 현재 머신과 같은 기록에만 적용한다.

사용 예:
    python -m vqa.jobs start --name train3b -- python -m vqa.train --config configs/... --run-id R
    python -m vqa.jobs status --name train3b
    python -m vqa.jobs tail --name train3b --lines 50
    python -m vqa.jobs stop --name train3b
    python -m vqa.jobs start --name train3b --auto-restart 5 -- python -m vqa.train ...
    python -m vqa.jobs wait --name train3b --poll 30
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import runtime

#: stop의 기본 유예 시간(초). 학습은 SIGTERM을 받아도 진행 중인 optimizer step을 끝내고
#: checkpoint를 저장한 뒤에야 종료한다. step당 60초에 Drive로 adapter/optimizer를 내리는
#: 시간까지 더하면 몇 분이 걸리므로 넉넉히 15분을 준다.
DEFAULT_STOP_TIMEOUT = 900.0


def jobs_dir(run_root: str | os.PathLike[str] | None = None) -> Path:
    return runtime.ensure_dir(runtime.run_root(run_root) / "jobs")


def log_path(name: str, run_root=None) -> Path:
    return jobs_dir(run_root) / f"{name}.log"


def pid_path(name: str, run_root=None) -> Path:
    return jobs_dir(run_root) / f"{name}.pid.json"


# --------------------------------------------------------------------------
# 프로세스 상태
# --------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """pid가 살아 있는지. Windows에서는 tasklist 대신 OpenProcess를 쓴다."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    # 우리가 직접 띄운 자식이 zombie로 남아 있으면 거둬들이고 죽은 것으로 본다.
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass  # 우리 자식이 아니다. 아래 신호 검사로 넘어간다.
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _as_pid(value: Any) -> int:
    """기록에서 읽은 pid 값을 int로. 없거나 깨졌으면 -1."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def machine_id() -> str | None:
    """이 기록을 쓴 머신/부팅을 가리키는 식별자. 아무것도 얻지 못하면 None.

    Linux에서는 부팅마다 바뀌는 boot_id를 쓴다. Colab은 가상 머신이 바뀌어도 Drive에 있는
    job 기록이 그대로 남으므로, 머신 단위가 아니라 부팅 단위로 구분해야 옛 pid를 살아 있는
    것으로 오판하지 않는다. boot_id를 읽을 수 없는 macOS/Windows에서는 호스트 이름으로
    대신하고, 그마저 얻지 못하면 None을 돌려주어 기존 동작을 그대로 둔다.
    """
    try:
        text = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        if text:
            return f"boot:{text}"
    except OSError:
        pass
    try:
        import socket

        host = socket.gethostname().strip()
        if host:
            return f"host:{host}"
    except OSError:
        pass
    return None


def _record_is_foreign(record: dict[str, Any]) -> bool:
    """기록을 쓴 머신이 지금 이 머신과 다른지. 판단할 수 없으면 False.

    식별자 필드가 없는 기록은 이 필드를 남기지 않던 옛 코드가 쓴 것이다. boot_id를 읽을 수
    있는 환경(Linux)이라면 그 기록의 pid는 이전 부팅의 것일 수 있으므로 다른 머신의 기록으로
    본다. 무관한 프로세스 때문에 시작이 거부되는 쪽이 지금은 더 해롭고, GPU 여유는
    --min-free-gb가 따로 막아 준다. boot_id가 없는 환경에서는 판단을 포기하고 기존 동작을
    유지한다.
    """
    if not isinstance(record, dict):
        return False
    current = machine_id()
    if current is None:
        return False
    stored = record.get("machine_id")
    if not stored:
        return current.startswith("boot:")
    return stored != current


def _alive_in_record(record: dict[str, Any], pid: int) -> bool:
    """기록에 적힌 pid가 지금 이 머신에서 살아 있는지."""
    if pid <= 0 or _record_is_foreign(record):
        return False
    return pid_alive(pid)


def _record_finished(record: dict[str, Any]) -> bool:
    """감시 프로세스가 종료 코드를 남겼는지. pid 재사용으로 인한 오탐을 줄인다."""
    return record.get("finished") is not None or record.get("exit_code") is not None


def _update_job(name: str, run_root, **fields: Any) -> dict[str, Any]:
    """pid 기록을 읽어 필드만 갈아끼우고 다시 쓴다."""
    record = read_job(name, run_root) or {"name": name}
    record.update(fields)
    # 기록을 갱신하는 것은 언제나 지금 이 머신의 프로세스다.
    record["machine_id"] = machine_id()
    runtime.atomic_write_json(pid_path(name, run_root), record)
    return record


def _popen_kwargs() -> dict[str, Any]:
    """부모와 분리해서 띄우기 위한 플랫폼별 인자."""
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        return {"creationflags": flags}
    return {"start_new_session": True}


# --------------------------------------------------------------------------
# GPU 점유 가드
# --------------------------------------------------------------------------


def gpu_free_gb() -> float | None:
    """nvidia-smi로 본 GPU 여유 메모리(GB) 중 가장 큰 값. 확인할 수 없으면 None.

    torch를 import하지 않고 subprocess만 쓴다. CPU 전용 환경이나 nvidia-smi가 없는
    곳에서는 None을 돌려주고, 호출하는 쪽이 가드를 건너뛴다.
    """
    try:
        completed = subprocess.run(  # noqa: S603,S607
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            values.append(float(token) / 1024.0)  # nvidia-smi는 MiB 단위로 준다
        except ValueError:
            continue
    return max(values) if values else None


def running_jobs(run_root=None, *, exclude: str | None = None) -> list[dict[str, Any]]:
    """같은 run-root 아래에서 아직 살아 있는 job 기록을 모은다."""
    found: list[dict[str, Any]] = []
    for path in sorted(jobs_dir(run_root).glob("*.pid.json")):
        record = runtime.read_json_or_none(path) or {}
        if not isinstance(record, dict) or _record_finished(record):
            continue
        job_name = record.get("name") or path.name[: -len(".pid.json")]
        if exclude is not None and job_name == exclude:
            continue
        if _record_is_foreign(record):
            continue  # 다른 머신(또는 이전 부팅)의 기록이므로 pid를 검사하지 않는다.
        pid = _as_pid(record.get("pid"))
        child = _as_pid(record.get("child_pid"))
        alive = _alive_in_record(record, pid)
        child_alive = _alive_in_record(record, child)
        if alive or child_alive:
            found.append(
                {
                    "name": job_name,
                    "pid": pid,
                    "child_pid": child if child > 0 else None,
                    "alive": alive,
                    "child_alive": child_alive,
                }
            )
    return found


def _describe_running(jobs_found: Sequence[dict[str, Any]]) -> str:
    parts = []
    for item in jobs_found:
        text = f"{item['name']}(pid={item['pid']}"
        if item.get("child_pid"):
            text += f", child_pid={item['child_pid']}"
        parts.append(text + ")")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# start / stop / status
# --------------------------------------------------------------------------


def start_job(
    name: str,
    command: Sequence[str],
    *,
    run_root=None,
    run_id: str | None = None,
    auto_restart: int = 0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    allow_concurrent: bool = False,
    min_free_gb: float | None = None,
) -> dict[str, Any]:
    """명령을 분리 프로세스로 띄우고 pid 파일을 쓴다.

    GPU가 한 장뿐이므로 기본적으로 같은 run-root 아래에 살아 있는 다른 job이 있으면
    시작을 거부한다. 일부러 겹쳐 띄우려면 allow_concurrent=True를 준다.
    """
    if not command:
        raise ValueError("실행할 명령이 비어 있습니다.")
    existing = read_job(name, run_root)
    if existing and not _record_is_foreign(existing):
        existing_pid = _as_pid(existing.get("pid"))
        existing_child = _as_pid(existing.get("child_pid"))
        if existing_pid > 0 and pid_alive(existing_pid):
            raise RuntimeError(f"job {name!r}이 이미 pid={existing_pid}로 실행 중입니다.")
        if existing_child > 0 and pid_alive(existing_child):
            raise RuntimeError(f"job {name!r}의 자식 프로세스 pid={existing_child}가 아직 살아 있습니다.")

    if not allow_concurrent:
        others = running_jobs(run_root, exclude=name)
        if others:
            raise RuntimeError(
                f"같은 run-root에 아직 살아 있는 job이 있어 {name!r}을 시작하지 않습니다: "
                f"{_describe_running(others)}\n"
                "GPU를 나눠 쓰면 모델 로딩 단계에서 OOM이 납니다. "
                "`python -m vqa.jobs wait --name <이름>`으로 끝나기를 기다리거나 "
                "`python -m vqa.jobs stop --name <이름>`으로 먼저 정리하세요. "
                "일부러 겹쳐 띄우려면 --allow-concurrent를 주세요."
            )

    if min_free_gb is not None and float(min_free_gb) > 0:
        free = gpu_free_gb()
        if free is None:
            print(
                f"[jobs] nvidia-smi로 GPU 여유 메모리를 읽지 못해 --min-free-gb {min_free_gb} 검사를 건너뜁니다.",
                flush=True,
            )
        elif free < float(min_free_gb):
            raise RuntimeError(
                f"GPU 여유 메모리가 {free:.1f} GB뿐이라 요구치 {float(min_free_gb):.1f} GB에 못 미칩니다. "
                f"{name!r}을 시작하지 않습니다. `nvidia-smi`와 `python -m vqa.jobs list`로 "
                "GPU를 점유 중인 프로세스를 확인하세요."
            )

    logfile = log_path(name, run_root)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    launch = list(command)
    if auto_restart and auto_restart > 0:
        launch = [
            sys.executable,
            "-m",
            "vqa.jobs",
            "_supervise",
            "--name",
            name,
            "--max-restarts",
            str(int(auto_restart)),
        ]
        if run_root:
            launch += ["--run-root", str(runtime.run_root(run_root))]
        launch += ["--", *command]

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    if run_root:
        merged_env["VQA_RUN_ROOT"] = str(runtime.run_root(run_root))

    handle = open(logfile, "ab", buffering=0)
    try:
        handle.write(
            f"\n=== [{runtime.now_iso()}] start: {' '.join(launch)} ===\n".encode("utf-8")
        )
        process = subprocess.Popen(  # noqa: S603
            launch,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=cwd or os.getcwd(),
            env=merged_env,
            close_fds=True,
            **_popen_kwargs(),
        )
    finally:
        handle.close()

    record = {
        "name": name,
        "pid": process.pid,
        # auto_restart가 있으면 pid는 감시 프로세스다. 실제 작업 pid는 감시 프로세스가 채운다.
        "child_pid": None,
        "exit_code": None,
        "finished": None,
        "restarts": 0,
        "command": list(command),
        "launched": launch,
        "auto_restart": int(auto_restart),
        # 이 기록의 pid는 이 머신/부팅에서만 뜻이 있다.
        "machine_id": machine_id(),
        "run_id": run_id or _guess_run_id(command),
        "run_root": str(runtime.run_root(run_root)),
        "cwd": cwd or os.getcwd(),
        "started": runtime.now_iso(),
        "log": str(logfile),
    }
    runtime.atomic_write_json(pid_path(name, run_root), record)
    return record


def _guess_run_id(command: Sequence[str]) -> str | None:
    """명령줄에서 --run-id 값을 찾아 heartbeat 위치를 추정한다."""
    items = list(command)
    for i, token in enumerate(items):
        if token == "--run-id" and i + 1 < len(items):
            return items[i + 1]
        if token.startswith("--run-id="):
            return token.split("=", 1)[1]
    return None


def read_job(name: str, run_root=None) -> dict[str, Any] | None:
    return runtime.read_json_or_none(pid_path(name, run_root))


def job_status(name: str, run_root=None) -> dict[str, Any]:
    """pid 생존, heartbeat, latest checkpoint를 묶어 돌려준다."""
    record = read_job(name, run_root)
    if record is None:
        return {"name": name, "exists": False}

    pid = _as_pid(record.get("pid"))
    child_pid = _as_pid(record.get("child_pid"))
    foreign = _record_is_foreign(record)
    alive = _alive_in_record(record, pid)
    child_alive = _alive_in_record(record, child_pid)
    status: dict[str, Any] = {
        "name": name,
        "exists": True,
        "pid": pid,
        "alive": alive,
        # 다른 머신(또는 이전 부팅)이 쓴 기록이면 pid를 믿을 수 없으므로 죽은 것으로 본다.
        "foreign": foreign,
        "machine_id": record.get("machine_id"),
        "child_pid": child_pid if child_pid > 0 else None,
        "child_alive": child_alive,
        # 감시 프로세스가 죽고 자식만 고아로 남은 경우까지 잡아내는 판정.
        "any_alive": alive or child_alive,
        "exit_code": record.get("exit_code"),
        "finished": record.get("finished"),
        "restarts": record.get("restarts", 0),
        "started": record.get("started"),
        "command": record.get("command"),
        "auto_restart": record.get("auto_restart", 0),
        "log": record.get("log"),
        "run_id": record.get("run_id"),
    }

    run_id = record.get("run_id")
    if run_id:
        run_dir = runtime.run_dir(run_id, record.get("run_root") or run_root)
        heartbeat = runtime.read_json_or_none(run_dir / "heartbeat.json")
        status["heartbeat"] = heartbeat
        if heartbeat and heartbeat.get("updated"):
            status["heartbeat_age_seconds"] = _age_seconds(heartbeat["updated"])
        latest = runtime.read_json_or_none(run_dir / "checkpoints" / "latest.json")
        if latest:
            status["last_checkpoint_step"] = latest.get("step")
            status["last_checkpoint_dir"] = latest.get("dir")
            status["last_checkpoint_created"] = latest.get("created")
    return status


def _age_seconds(iso: str) -> float | None:
    from datetime import datetime, timezone

    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def stop_job(
    name: str,
    *,
    run_root=None,
    timeout: float = DEFAULT_STOP_TIMEOUT,
    force: bool = False,
) -> dict[str, Any]:
    """먼저 정중한 종료 신호를 보내고, 기다린 뒤에도 살아 있으면 강제 종료한다.

    학습 프로세스는 이 신호를 받으면 optimizer step 경계에서 checkpoint를 저장한다.
    감시 프로세스가 살아 있으면 그쪽에만 보낸다. 감시 프로세스가 신호를 자식에게 전달하고
    재시작하지 않으므로, 학습이 저장을 마치고 끝날 때까지 기다리면 된다. 감시 프로세스가
    이미 죽고 자식만 고아로 남았으면 자식에게 직접 보낸다.
    """
    record = read_job(name, run_root)
    if record is None:
        return {"name": name, "exists": False}
    pid = _as_pid(record.get("pid"))
    child_pid = _as_pid(record.get("child_pid"))
    if _record_is_foreign(record):
        # 다른 머신(또는 이전 부팅)의 pid다. 같은 번호의 무관한 프로세스에 신호를 보내지 않는다.
        return {
            "name": name,
            "exists": True,
            "alive": False,
            "child_alive": False,
            "child_pid": child_pid if child_pid > 0 else None,
            "foreign": True,
            "machine_id": record.get("machine_id"),
            "action": "foreign_record",
            "note": (
                "다른 머신(또는 이전 세션)에서 남긴 기록이라 신호를 보내지 않았습니다. "
                "그 프로세스는 이 머신에 없습니다."
            ),
        }
    alive = _alive_in_record(record, pid)
    child_alive = _alive_in_record(record, child_pid)
    if not alive and not child_alive:
        return {
            "name": name,
            "exists": True,
            "alive": False,
            "child_alive": False,
            "child_pid": child_pid if child_pid > 0 else None,
            "action": "already_dead",
        }

    sent = _terminate(pid) if alive else _terminate(child_pid)
    if alive and child_alive and force:
        # force면 감시 프로세스의 전달을 기다리지 않고 자식에게도 바로 보낸다.
        _terminate(child_pid)

    seen_children = {child_pid} if child_pid > 0 else set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = read_job(name, run_root) or record
        now_child = _as_pid(current.get("child_pid"))
        if now_child > 0:
            seen_children.add(now_child)
        alive = pid > 0 and pid_alive(pid)
        child_alive = any(pid_alive(p) for p in seen_children if p > 0)
        if not alive and not child_alive:
            return {
                "name": name,
                "exists": True,
                "alive": False,
                "child_alive": False,
                "child_pid": now_child if now_child > 0 else None,
                "action": sent,
            }
        time.sleep(0.5)

    # 유예 시간이 지나도 살아 있으면 자식부터 강제로 끝낸다.
    for target in [*sorted(p for p in seen_children if p > 0), pid]:
        if target > 0 and pid_alive(target):
            _kill(target)
    time.sleep(0.5)
    return {
        "name": name,
        "exists": True,
        "alive": pid > 0 and pid_alive(pid),
        "child_alive": any(pid_alive(p) for p in seen_children if p > 0),
        "child_pid": child_pid if child_pid > 0 else None,
        "action": f"{sent}+kill",
    }


def _terminate(pid: int) -> str:
    if os.name == "nt":
        sig = getattr(signal, "CTRL_BREAK_EVENT", None)
        if sig is not None:
            try:
                os.kill(pid, sig)
                return "CTRL_BREAK_EVENT"
            except OSError:
                pass
        try:
            os.kill(pid, signal.SIGTERM)
            return "SIGTERM"
        except OSError:
            return "none"
    try:
        os.kill(pid, signal.SIGTERM)
        return "SIGTERM"
    except OSError:
        return "none"


def _kill(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)  # noqa: S603,S607
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def tail_log(name: str, *, lines: int = 40, run_root=None) -> str:
    path = log_path(name, run_root)
    if not path.exists():
        return f"(로그 없음: {path})"
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:])


# --------------------------------------------------------------------------
# 자동 재시작 supervisor
# --------------------------------------------------------------------------


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return str(signum)


def _await_own_record(name: str, run_root=None, timeout: float = 15.0) -> dict[str, Any] | None:
    """start_job이 우리 pid로 기록을 쓸 때까지 기다린다.

    감시 프로세스가 먼저 child_pid를 써 버리면 뒤이은 start_job의 쓰기에 덮여 사라진다.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = read_job(name, run_root)
        if record and _as_pid(record.get("pid")) == os.getpid():
            return record
        time.sleep(0.1)
    return None


def _interruptible_sleep(seconds: float, should_stop) -> None:
    """종료 신호를 받으면 곧바로 빠져나오는 대기.

    time.sleep은 시그널 핸들러가 돌아온 뒤 남은 시간을 마저 자므로(PEP 475), backoff를
    통째로 자면 stop이 최대 300초를 기다리게 된다. 1초씩 끊어 잔다.
    """
    waited = 0.0
    while waited < seconds and not should_stop():
        time.sleep(min(1.0, seconds - waited))
        waited += 1.0


def supervise(name: str, command: Sequence[str], max_restarts: int, run_root=None) -> int:
    """명령이 0이 아닌 코드로 끝나면 backoff를 두고 다시 띄운다.

    `--resume auto`와 함께 쓰면 세션이 끊겨도 스스로 이어서 학습한다.
    자식 pid와 최종 종료 코드는 job 기록에 남긴다. SIGTERM/SIGINT를 받으면 자식에게
    그대로 전달하고, 자식이 checkpoint를 저장하고 끝나기를 기다린 뒤 재시작하지 않는다.
    """
    attempt = 0
    backoff = 10.0
    state: dict[str, Any] = {"stopping": False, "signal": None, "process": None}

    def _handle(signum, _frame):
        state["stopping"] = True
        state["signal"] = _signal_name(signum)
        process = state.get("process")
        child = getattr(process, "pid", None)
        print(
            f"[jobs] {state['signal']} 수신. 자식 pid={child}에 전달하고 재시작하지 않습니다.",
            flush=True,
        )
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
            except (OSError, ValueError):
                pass

    previous: list[tuple[Any, Any]] = []
    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous.append((sig, signal.signal(sig, _handle)))
        except (ValueError, OSError):
            pass  # 메인 스레드가 아니거나 지원하지 않는 신호

    code = 0
    try:
        while True:
            if state["stopping"]:
                break
            print(f"[jobs] supervise 시작 attempt={attempt}: {' '.join(command)}", flush=True)
            process = subprocess.Popen(list(command), cwd=os.getcwd())  # noqa: S603
            state["process"] = process
            _update_job(
                name,
                run_root,
                child_pid=process.pid,
                child_started=runtime.now_iso(),
                restarts=attempt,
                exit_code=None,
                finished=None,
            )
            code = process.wait()
            state["process"] = None
            print(f"[jobs] 종료 코드 {code} (attempt={attempt})", flush=True)
            if state["stopping"]:
                print(f"[jobs] {state['signal']}로 멈췄으므로 재시작하지 않습니다.", flush=True)
                break
            if code == 0:
                break
            if attempt >= max_restarts:
                print(f"[jobs] 재시작 한도 {max_restarts}회를 넘겨 중단합니다.", flush=True)
                break
            attempt += 1
            wait = min(backoff * attempt, 300.0)
            print(f"[jobs] {wait:.0f}초 뒤 재시작합니다.", flush=True)
            _interruptible_sleep(wait, lambda: bool(state["stopping"]))
    finally:
        for sig, handler in previous:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        _update_job(
            name,
            run_root,
            child_pid=None,
            exit_code=code,
            finished=runtime.now_iso(),
            restarts=attempt,
            stopped_by=state["signal"],
        )
    return code


# --------------------------------------------------------------------------
# wait
# --------------------------------------------------------------------------


def last_log_line(name: str, run_root=None) -> str:
    text = tail_log(name, lines=1, run_root=run_root).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "(로그 없음)"


def _poll_sleep(poll: float, deadline: float | None) -> None:
    """poll초만큼 자되 timeout 시각을 넘기지 않고, 0.5초씩 끊어 잔다."""
    end = time.time() + max(0.05, float(poll))
    if deadline is not None:
        end = min(end, deadline)
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.5, remaining))


def wait_job(
    name: str,
    *,
    run_root=None,
    poll: float = 30.0,
    timeout: float | None = None,
    record_grace: float = 30.0,
    printer=print,
) -> dict[str, Any]:
    """job(자식 포함)이 끝날 때까지 기다린다.

    poll마다 로그의 마지막 한 줄을 찍는다. 결과의 result는
    ok / failed / timeout / unknown / missing 중 하나다. auto-restart 없이 띄운 job은
    종료 코드를 기록해 줄 감시 프로세스가 없으므로 unknown이 된다.

    run-root가 Google Drive(FUSE) 위에 있는 Colab 환경에서는 start가 방금 쓴 pid 기록이
    곧바로 뒤이은 wait의 읽기에 보이지 않을 수 있다. 그래서 기록이 아직 없으면 곧바로
    missing으로 끝내지 않고, record_grace초 동안 1초 간격으로 기록이 나타나기를 기다린다.
    """
    record = read_job(name, run_root)
    if record is None and record_grace > 0:
        grace_deadline = time.time() + float(record_grace)
        while record is None and time.time() < grace_deadline:
            time.sleep(1.0)
            record = read_job(name, run_root)
    if record is None:
        return {"name": name, "exists": False, "result": "missing", "exit_code": None}
    if _record_is_foreign(record):
        # 이 머신에서 띄운 job이 아니다. 종료 코드를 알 수 없으므로 성공으로 넘기지 않는다.
        return {
            "name": name,
            "exists": True,
            "result": "foreign",
            "exit_code": None,
            "machine_id": record.get("machine_id"),
            "last_line": last_log_line(name, run_root),
        }

    deadline = None if timeout is None else time.time() + float(timeout)
    while True:
        status = job_status(name, run_root)
        if not status.get("any_alive"):
            break
        if deadline is not None and time.time() >= deadline:
            return {
                "name": name,
                "exists": True,
                "result": "timeout",
                "exit_code": None,
                "pid": status.get("pid"),
                "child_pid": status.get("child_pid"),
                "last_line": last_log_line(name, run_root),
            }
        printer(
            f"[jobs] {name} 대기 중 pid={status.get('pid')} child_pid={status.get('child_pid')} "
            f"| {last_log_line(name, run_root)}"
        )
        _poll_sleep(poll, deadline)

    final = read_job(name, run_root) or {}
    exit_code = final.get("exit_code")
    if exit_code is None:
        result = "unknown"
    else:
        result = "ok" if int(exit_code) == 0 else "failed"
    return {
        "name": name,
        "exists": True,
        "result": result,
        "exit_code": exit_code,
        "restarts": final.get("restarts", 0),
        "finished": final.get("finished"),
        "log": final.get("log"),
        "last_line": last_log_line(name, run_root),
    }


#: wait 결과를 CLI 종료 코드로 옮기는 표. 이 값은 노트북이 그대로 쓰므로 바꾸지 않는다.
WAIT_EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "unknown": 0,
    "failed": 1,
    "timeout": 2,
    "missing": 1,
    "foreign": 1,
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vqa.jobs", description="분리 실행 프로세스 관리")
    parser.add_argument("action", choices=["start", "status", "tail", "stop", "wait", "list", "_supervise"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--auto-restart", type=int, default=0, help="0이 아니면 비정상 종료 시 N회까지 재시작")
    parser.add_argument("--max-restarts", type=int, default=0, help="_supervise 내부용")
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"stop은 기본 {DEFAULT_STOP_TIMEOUT:.0f}초, wait은 기본 무제한",
    )
    parser.add_argument("--poll", type=float, default=30.0, help="wait에서 로그를 찍는 간격(초)")
    parser.add_argument(
        "--record-grace",
        type=float,
        default=30.0,
        help="wait에서 job 기록이 아직 안 보이면 missing 처리 전에 기다리는 시간(초). "
        "Drive처럼 쓰기 직후 읽기가 늦게 반영되는 run-root를 위한 값",
    )
    parser.add_argument(
        "--allow-concurrent",
        action="store_true",
        help="같은 run-root에 살아 있는 다른 job이 있어도 시작한다 (GPU OOM 주의)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="nvidia-smi로 본 GPU 여유 메모리가 이 값보다 적으면 시작을 거부한다",
    )
    parser.add_argument("--force", action="store_true", help="stop에서 자식에게도 곧바로 신호를 보낸다")
    parser.add_argument("--json", action="store_true", help="status를 JSON으로 출력")
    return parser


def split_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """첫 `--`를 기준으로 (이 CLI의 인자, 실행할 명령)으로 나눈다.

    argparse의 REMAINDER는 첫 위치 인자 뒤의 옵션까지 삼켜 버리므로 직접 나눈다.
    """
    items = list(argv)
    if "--" not in items:
        return items, []
    index = items.index("--")
    return items[:index], items[index + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    head, command = split_command(raw)
    args = build_parser().parse_args(head)

    if args.action == "list":
        for path in sorted(jobs_dir(args.run_root).glob("*.pid.json")):
            record = runtime.read_json_or_none(path) or {}
            pid = _as_pid(record.get("pid"))
            child = _as_pid(record.get("child_pid"))
            foreign = _record_is_foreign(record)
            alive = _alive_in_record(record, pid)
            child_alive = _alive_in_record(record, child)
            mark = "  <- 다른 머신(또는 이전 세션)의 기록" if foreign else ""
            print(
                f"{record.get('name', path.stem):20s} pid={pid} alive={alive} "
                f"child_pid={child if child > 0 else None} child_alive={child_alive} "
                f"exit={record.get('exit_code')}{mark}"
            )
        return 0

    if not args.name:
        raise SystemExit("--name이 필요합니다.")

    if args.action == "start":
        if not command:
            raise SystemExit("-- 뒤에 실행할 명령을 적어야 합니다.")
        try:
            record = start_job(
                args.name,
                command,
                run_root=args.run_root,
                run_id=args.run_id,
                auto_restart=args.auto_restart,
                allow_concurrent=args.allow_concurrent,
                min_free_gb=args.min_free_gb,
            )
        except RuntimeError as exc:
            print(f"[jobs] 시작 거부: {exc}")
            return 1
        print(f"[jobs] 시작: {args.name} pid={record['pid']} log={record['log']}")
        return 0

    if args.action == "_supervise":
        if not command:
            raise SystemExit("-- 뒤에 실행할 명령을 적어야 합니다.")
        # start_job이 pid 기록을 다 쓴 뒤에 child_pid를 덧쓰기 위해 먼저 기다린다.
        _await_own_record(args.name, args.run_root)
        return supervise(args.name, command, args.max_restarts, run_root=args.run_root)

    if args.action == "status":
        status = job_status(args.name, args.run_root)
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(_format_status(status))
        return 0 if status.get("alive") else 1

    if args.action == "tail":
        print(tail_log(args.name, lines=args.lines, run_root=args.run_root))
        return 0

    if args.action == "stop":
        timeout = DEFAULT_STOP_TIMEOUT if args.timeout is None else args.timeout
        result = stop_job(args.name, run_root=args.run_root, timeout=timeout, force=args.force)
        print(f"[jobs] stop: {result}")
        return 0

    if args.action == "wait":
        result = wait_job(
            args.name,
            run_root=args.run_root,
            poll=args.poll,
            timeout=args.timeout,
            record_grace=args.record_grace,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(_format_wait(result))
        return WAIT_EXIT_CODES.get(str(result.get("result")), 1)

    return 2


def _format_wait(result: dict[str, Any]) -> str:
    name = result.get("name")
    kind = result.get("result")
    if kind == "missing":
        return f"[jobs] {name}: 기록 없음"
    if kind == "foreign":
        return (
            f"[jobs] {name}: 다른 머신(또는 이전 세션)에서 남긴 기록이라 기다릴 대상이 없습니다. "
            "이 머신에서 다시 시작하십시오."
        )
    if kind == "timeout":
        return f"[jobs] {name}: timeout. 아직 실행 중입니다 (pid={result.get('pid')} child_pid={result.get('child_pid')})"
    if kind == "unknown":
        return (
            f"[jobs] {name}: 종료했지만 종료 코드를 알 수 없습니다 "
            f"(--auto-restart 없이 띄운 job). 마지막 로그: {result.get('last_line')}"
        )
    verdict = "정상 종료" if kind == "ok" else "실패"
    return (
        f"[jobs] {name}: {verdict} exit_code={result.get('exit_code')} "
        f"restarts={result.get('restarts')}\n  마지막 로그: {result.get('last_line')}"
    )


def _format_status(status: dict[str, Any]) -> str:
    if not status.get("exists"):
        return f"job {status['name']}: 기록 없음"
    lines = [
        f"job {status['name']}: {'ALIVE' if status['alive'] else 'DEAD'} pid={status['pid']}",
        f"  started : {status.get('started')}",
        f"  command : {' '.join(status.get('command') or [])}",
        f"  log     : {status.get('log')}",
    ]
    if status.get("foreign"):
        lines.insert(
            1,
            "  주의    : 다른 머신(또는 이전 세션)에서 남긴 기록입니다. "
            "pid는 이 머신에 없으므로 DEAD로 봅니다.",
        )
    if status.get("auto_restart"):
        lines.insert(
            1,
            f"  child   : pid={status.get('child_pid')} "
            f"{'ALIVE' if status.get('child_alive') else 'DEAD'} "
            f"restarts={status.get('restarts')}",
        )
    if status.get("exit_code") is not None:
        lines.append(f"  exit    : {status.get('exit_code')} ({status.get('finished')})")
    heartbeat = status.get("heartbeat")
    if heartbeat:
        lines.append(
            f"  progress: step {heartbeat.get('step')}/{heartbeat.get('total_steps')} "
            f"loss_ema={heartbeat.get('loss_ema')} "
            f"s/step={heartbeat.get('seconds_per_step')} ETA={heartbeat.get('eta_human')}"
        )
        lines.append(
            f"  vram    : peak {heartbeat.get('peak_vram_gb')} GB, "
            f"heartbeat {status.get('heartbeat_age_seconds')}초 전"
        )
    if status.get("last_checkpoint_step") is not None:
        lines.append(
            f"  ckpt    : step {status['last_checkpoint_step']} "
            f"({status.get('last_checkpoint_dir')}, {status.get('last_checkpoint_created')})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
