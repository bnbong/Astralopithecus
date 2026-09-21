"""분리 실행 프로세스 관리자.

노트북 셀을 멈추거나 브라우저 연결이 끊겨도 학습이 계속되도록, 명령을 부모와 분리된
프로세스로 띄운다. Linux는 새 세션(start_new_session)으로, Windows는
CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS로 띄운다.

사용 예:
    python -m vqa.jobs start --name train3b -- python -m vqa.train --config configs/... --run-id R
    python -m vqa.jobs status --name train3b
    python -m vqa.jobs tail --name train3b --lines 50
    python -m vqa.jobs stop --name train3b
    python -m vqa.jobs start --name train3b --auto-restart 5 -- python -m vqa.train ...
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


def _popen_kwargs() -> dict[str, Any]:
    """부모와 분리해서 띄우기 위한 플랫폼별 인자."""
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        return {"creationflags": flags}
    return {"start_new_session": True}


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
) -> dict[str, Any]:
    """명령을 분리 프로세스로 띄우고 pid 파일을 쓴다."""
    if not command:
        raise ValueError("실행할 명령이 비어 있습니다.")
    existing = read_job(name, run_root)
    if existing and pid_alive(int(existing.get("pid", -1))):
        raise RuntimeError(f"job {name!r}이 이미 pid={existing['pid']}로 실행 중입니다.")

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
            "--",
            *command,
        ]

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
        "command": list(command),
        "launched": launch,
        "auto_restart": int(auto_restart),
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

    pid = int(record.get("pid", -1))
    status: dict[str, Any] = {
        "name": name,
        "exists": True,
        "pid": pid,
        "alive": pid_alive(pid),
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


def stop_job(name: str, *, run_root=None, timeout: float = 120.0, force: bool = False) -> dict[str, Any]:
    """먼저 정중한 종료 신호를 보내고, 기다린 뒤에도 살아 있으면 강제 종료한다.

    학습 프로세스는 이 신호를 받으면 optimizer step 경계에서 checkpoint를 저장한다.
    """
    record = read_job(name, run_root)
    if record is None:
        return {"name": name, "exists": False}
    pid = int(record.get("pid", -1))
    if not pid_alive(pid):
        return {"name": name, "exists": True, "alive": False, "action": "already_dead"}

    sent = _terminate(pid)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return {"name": name, "exists": True, "alive": False, "action": sent}
        time.sleep(0.5)

    # 유예 시간이 지나도 살아 있으면 강제로 끝낸다.
    _kill(pid)
    time.sleep(0.5)
    return {"name": name, "exists": True, "alive": pid_alive(pid), "action": f"{sent}+kill"}


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


def supervise(name: str, command: Sequence[str], max_restarts: int, run_root=None) -> int:
    """명령이 0이 아닌 코드로 끝나면 backoff를 두고 다시 띄운다.

    `--resume auto`와 함께 쓰면 세션이 끊겨도 스스로 이어서 학습한다.
    """
    attempt = 0
    backoff = 10.0
    while True:
        print(f"[jobs] supervise 시작 attempt={attempt}: {' '.join(command)}", flush=True)
        completed = subprocess.run(list(command), cwd=os.getcwd())  # noqa: S603
        code = completed.returncode
        print(f"[jobs] 종료 코드 {code} (attempt={attempt})", flush=True)
        if code == 0:
            return 0
        if attempt >= max_restarts:
            print(f"[jobs] 재시작 한도 {max_restarts}회를 넘겨 중단합니다.", flush=True)
            return code
        attempt += 1
        wait = min(backoff * attempt, 300.0)
        print(f"[jobs] {wait:.0f}초 뒤 재시작합니다.", flush=True)
        time.sleep(wait)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vqa.jobs", description="분리 실행 프로세스 관리")
    parser.add_argument("action", choices=["start", "status", "tail", "stop", "list", "_supervise"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--auto-restart", type=int, default=0, help="0이 아니면 비정상 종료 시 N회까지 재시작")
    parser.add_argument("--max-restarts", type=int, default=0, help="_supervise 내부용")
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=120.0)
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
            alive = pid_alive(int(record.get("pid", -1)))
            print(f"{record.get('name', path.stem):20s} pid={record.get('pid')} alive={alive}")
        return 0

    if not args.name:
        raise SystemExit("--name이 필요합니다.")

    if args.action == "start":
        if not command:
            raise SystemExit("-- 뒤에 실행할 명령을 적어야 합니다.")
        record = start_job(
            args.name,
            command,
            run_root=args.run_root,
            run_id=args.run_id,
            auto_restart=args.auto_restart,
        )
        print(f"[jobs] 시작: {args.name} pid={record['pid']} log={record['log']}")
        return 0

    if args.action == "_supervise":
        if not command:
            raise SystemExit("-- 뒤에 실행할 명령을 적어야 합니다.")
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
        result = stop_job(args.name, run_root=args.run_root, timeout=args.timeout)
        print(f"[jobs] stop: {result}")
        return 0

    return 2


def _format_status(status: dict[str, Any]) -> str:
    if not status.get("exists"):
        return f"job {status['name']}: 기록 없음"
    lines = [
        f"job {status['name']}: {'ALIVE' if status['alive'] else 'DEAD'} pid={status['pid']}",
        f"  started : {status.get('started')}",
        f"  command : {' '.join(status.get('command') or [])}",
        f"  log     : {status.get('log')}",
    ]
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
