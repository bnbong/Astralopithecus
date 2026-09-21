"""분리 실행 프로세스 관리 테스트. 더미 sleep 명령으로 start/status/stop을 돌린다."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa import jobs  # noqa: E402
from vqa import runtime  # noqa: E402


SLEEP_COMMAND = [sys.executable, "-c", "import time; time.sleep(120)"]


def wait_until(predicate, timeout=15.0, interval=0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_start_status_stop_roundtrip(tmp_path):
    record = jobs.start_job("dummy", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert record["pid"] > 0
        assert wait_until(lambda: jobs.job_status("dummy", tmp_path)["alive"])

        status = jobs.job_status("dummy", tmp_path)
        assert status["exists"] and status["alive"]
        assert status["command"] == SLEEP_COMMAND
        assert Path(status["log"]).exists()

        jobs.stop_job("dummy", run_root=tmp_path, timeout=5.0)
        assert wait_until(lambda: not jobs.job_status("dummy", tmp_path)["alive"])
    finally:
        jobs.stop_job("dummy", run_root=tmp_path, timeout=2.0)

    dead = jobs.job_status("dummy", tmp_path)
    assert dead["exists"] and not dead["alive"]


def test_status_for_unknown_job(tmp_path):
    status = jobs.job_status("nope", tmp_path)
    assert status == {"name": "nope", "exists": False}


def test_start_refuses_duplicate_live_job(tmp_path):
    jobs.start_job("dup", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("dup", tmp_path)["alive"])
        with pytest.raises(RuntimeError):
            jobs.start_job("dup", SLEEP_COMMAND, run_root=tmp_path)
    finally:
        jobs.stop_job("dup", run_root=tmp_path, timeout=5.0)


def test_log_captures_child_output(tmp_path):
    command = [sys.executable, "-c", "print('hello-from-child', flush=True)"]
    jobs.start_job("logger", command, run_root=tmp_path)
    assert wait_until(lambda: "hello-from-child" in jobs.tail_log("logger", run_root=tmp_path))


def test_status_reads_heartbeat_and_latest_checkpoint(tmp_path):
    run_id = "run42"
    run_dir = runtime.ensure_dir(Path(tmp_path) / run_id)
    runtime.atomic_write_json(
        run_dir / "heartbeat.json",
        {"step": 12, "total_steps": 100, "seconds_per_step": 3.4, "eta_human": "4m58s", "loss_ema": 0.42},
    )
    runtime.atomic_write_json(
        run_dir / "checkpoints" / "latest.json",
        {"step": 10, "dir": "step_000010", "created": runtime.now_iso()},
    )
    command = [sys.executable, "-c", "import time; time.sleep(120)", "--run-id", run_id]
    jobs.start_job("hb", command, run_root=tmp_path, run_id=run_id)
    try:
        status = jobs.job_status("hb", tmp_path)
        assert status["heartbeat"]["step"] == 12
        assert status["last_checkpoint_step"] == 10
        assert "step 12/100" in jobs._format_status(status)
    finally:
        jobs.stop_job("hb", run_root=tmp_path, timeout=5.0)


def test_run_id_is_guessed_from_command():
    assert jobs._guess_run_id(["python", "-m", "vqa.train", "--run-id", "abc"]) == "abc"
    assert jobs._guess_run_id(["python", "-m", "vqa.train", "--run-id=abc"]) == "abc"
    assert jobs._guess_run_id(["python", "-c", "pass"]) is None


def test_supervisor_retries_failing_command(tmp_path, monkeypatch):
    """비정상 종료를 max_restarts 만큼 다시 띄운다. 대기 시간은 건너뛴다."""
    monkeypatch.setattr(jobs.time, "sleep", lambda _seconds: None)
    marker = tmp_path / "attempts.txt"
    command = [
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'a').write('x'); raise SystemExit(3)",
    ]
    code = jobs.supervise("s", command, max_restarts=2, run_root=tmp_path)
    assert code == 3
    assert marker.read_text() == "xxx"  # 최초 1회 + 재시작 2회


def test_supervisor_stops_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.time, "sleep", lambda _seconds: None)
    command = [sys.executable, "-c", "raise SystemExit(0)"]
    assert jobs.supervise("s", command, max_restarts=5, run_root=tmp_path) == 0


def test_cli_start_status_stop(tmp_path, capsys):
    argv_start = ["start", "--name", "cli", "--run-root", str(tmp_path), "--", *SLEEP_COMMAND]
    assert jobs.main(argv_start) == 0
    try:
        assert wait_until(lambda: jobs.job_status("cli", tmp_path)["alive"])
        assert jobs.main(["status", "--name", "cli", "--run-root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "ALIVE" in out
    finally:
        assert jobs.main(["stop", "--name", "cli", "--run-root", str(tmp_path)]) == 0
    assert wait_until(lambda: not jobs.job_status("cli", tmp_path)["alive"])
    assert jobs.main(["status", "--name", "cli", "--run-root", str(tmp_path)]) == 1


def test_detached_child_survives_parent_group_signal(tmp_path):
    """새 세션으로 띄우므로 부모 프로세스 그룹에 보낸 신호를 받지 않는다."""
    import os
    import signal

    if os.name == "nt":
        pytest.skip("POSIX 프로세스 그룹 동작 확인용")
    jobs.start_job("detached", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("detached", tmp_path)["alive"])
        pid = jobs.job_status("detached", tmp_path)["pid"]
        assert os.getpgid(pid) != os.getpgid(os.getpid())
        time.sleep(0.3)
        assert jobs.pid_alive(pid)
    finally:
        jobs.stop_job("detached", run_root=tmp_path, timeout=5.0)
