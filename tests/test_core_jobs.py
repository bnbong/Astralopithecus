"""분리 실행 프로세스 관리 테스트. 더미 sleep 명령으로 start/status/stop을 돌린다."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqa import jobs  # noqa: E402
from vqa import runtime  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

SLEEP_COMMAND = [sys.executable, "-c", "import time; time.sleep(120)"]


def wait_until(predicate, timeout=15.0, interval=0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def graceful_child_command(marker: Path, ready: Path) -> list[str]:
    """SIGTERM을 받으면 marker를 남기고 0으로 끝나는 자식. 학습의 저장-후-종료를 흉내 낸다."""
    script = (
        "import signal, time\n"
        f"marker = {str(marker)!r}\n"
        f"ready = {str(ready)!r}\n"
        "def handle(signum, frame):\n"
        "    open(marker, 'w').write('saved')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, handle)\n"
        "open(ready, 'w').write('1')\n"
        "time.sleep(300)\n"
    )
    return [sys.executable, "-c", script]


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


# --------------------------------------------------------------------------
# auto-restart: 자식 pid 추적과 신호 전달
# --------------------------------------------------------------------------


def test_supervisor_records_child_pid(tmp_path):
    """감시 프로세스가 자식 pid를 job 기록에 남긴다."""
    jobs.start_job("sup", SLEEP_COMMAND, run_root=tmp_path, auto_restart=2, cwd=str(REPO_ROOT))
    try:
        assert wait_until(lambda: jobs.job_status("sup", tmp_path).get("child_pid"), timeout=30.0)
        status = jobs.job_status("sup", tmp_path)
        assert status["child_pid"] != status["pid"]
        assert status["child_alive"]
        assert status["any_alive"]
        assert "child" in jobs._format_status(status)
    finally:
        jobs.stop_job("sup", run_root=tmp_path, timeout=20.0)


def test_stop_forwards_signal_to_child_and_waits(tmp_path):
    """stop은 감시 프로세스를 거쳐 자식까지 끝낸다. 자식은 저장할 틈을 얻는다."""
    if os.name == "nt":
        pytest.skip("POSIX 시그널 전달 확인용")
    marker = tmp_path / "saved.txt"
    ready = tmp_path / "ready.txt"
    command = graceful_child_command(marker, ready)
    jobs.start_job("graceful", command, run_root=tmp_path, auto_restart=3, cwd=str(REPO_ROOT))

    assert wait_until(ready.exists, timeout=30.0), "자식이 뜨지 않았습니다."
    status = jobs.job_status("graceful", tmp_path)
    child_pid = status["child_pid"]
    assert child_pid and jobs.pid_alive(child_pid)

    result = jobs.stop_job("graceful", run_root=tmp_path, timeout=30.0)
    assert result["alive"] is False
    assert result["child_alive"] is False
    assert result["action"] == "SIGTERM"  # 강제 kill까지 가지 않았다
    assert marker.read_text() == "saved"
    assert not jobs.pid_alive(child_pid)

    # 신호를 받았으므로 재시작하지 않고, 종료 코드를 남긴다.
    final = jobs.job_status("graceful", tmp_path)
    assert final["exit_code"] == 0
    assert final["restarts"] == 0
    assert final["child_pid"] is None


def test_stop_kills_orphaned_child_when_supervisor_is_gone(tmp_path):
    """감시 프로세스가 먼저 죽어도 고아가 된 자식을 stop이 정리한다."""
    if os.name == "nt":
        pytest.skip("POSIX 시그널 전달 확인용")
    import signal as signal_mod

    jobs.start_job("orphan", SLEEP_COMMAND, run_root=tmp_path, auto_restart=2, cwd=str(REPO_ROOT))
    assert wait_until(lambda: jobs.job_status("orphan", tmp_path).get("child_pid"), timeout=30.0)
    status = jobs.job_status("orphan", tmp_path)
    child_pid = status["child_pid"]

    os.kill(status["pid"], signal_mod.SIGKILL)  # 감시 프로세스만 급사시킨다
    assert wait_until(lambda: not jobs.pid_alive(status["pid"]), timeout=15.0)
    assert jobs.pid_alive(child_pid)
    assert jobs.job_status("orphan", tmp_path)["any_alive"]

    jobs.stop_job("orphan", run_root=tmp_path, timeout=20.0)
    assert wait_until(lambda: not jobs.pid_alive(child_pid), timeout=15.0)


def test_supervisor_leaves_exit_code_in_record(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.time, "sleep", lambda _seconds: None)
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    assert jobs.supervise("code", command, max_restarts=0, run_root=tmp_path) == 7
    record = jobs.read_job("code", tmp_path)
    assert record["exit_code"] == 7
    assert record["finished"] is not None
    assert record["child_pid"] is None


# --------------------------------------------------------------------------
# GPU 점유 가드
# --------------------------------------------------------------------------


def test_start_refuses_when_another_job_is_alive(tmp_path):
    jobs.start_job("first", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("first", tmp_path)["alive"])
        with pytest.raises(RuntimeError) as excinfo:
            jobs.start_job("second", SLEEP_COMMAND, run_root=tmp_path)
        message = str(excinfo.value)
        assert "first" in message
        assert str(jobs.job_status("first", tmp_path)["pid"]) in message
        assert "wait" in message and "stop" in message
    finally:
        jobs.stop_job("first", run_root=tmp_path, timeout=5.0)


def test_allow_concurrent_bypasses_the_guard(tmp_path):
    jobs.start_job("one", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("one", tmp_path)["alive"])
        jobs.start_job("two", SLEEP_COMMAND, run_root=tmp_path, allow_concurrent=True)
        assert wait_until(lambda: jobs.job_status("two", tmp_path)["alive"])
    finally:
        jobs.stop_job("one", run_root=tmp_path, timeout=5.0)
        jobs.stop_job("two", run_root=tmp_path, timeout=5.0)


def test_guard_ignores_finished_jobs(tmp_path):
    """종료 코드가 남은 기록은 pid가 재사용돼도 살아 있는 job으로 보지 않는다."""
    jobs._update_job("done", tmp_path, pid=os.getpid(), exit_code=0, finished=runtime.now_iso())
    assert jobs.running_jobs(tmp_path) == []
    jobs.start_job("next", SLEEP_COMMAND, run_root=tmp_path)
    jobs.stop_job("next", run_root=tmp_path, timeout=5.0)


def test_min_free_gb_refuses_when_gpu_is_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "gpu_free_gb", lambda: 5.0)
    with pytest.raises(RuntimeError) as excinfo:
        jobs.start_job("greedy", SLEEP_COMMAND, run_root=tmp_path, min_free_gb=40.0)
    assert "5.0" in str(excinfo.value)
    assert jobs.read_job("greedy", tmp_path) is None


def test_min_free_gb_is_skipped_without_nvidia_smi(tmp_path, monkeypatch, capsys):
    """CPU 전용 환경에서는 가드를 건너뛰고 경고만 남긴다."""
    monkeypatch.setattr(jobs, "gpu_free_gb", lambda: None)
    jobs.start_job("cpu-only", SLEEP_COMMAND, run_root=tmp_path, min_free_gb=40.0)
    try:
        assert "건너뜁니다" in capsys.readouterr().out
        assert wait_until(lambda: jobs.job_status("cpu-only", tmp_path)["alive"])
    finally:
        jobs.stop_job("cpu-only", run_root=tmp_path, timeout=5.0)


def test_gpu_free_gb_without_nvidia_smi(monkeypatch):
    def boom(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(jobs.subprocess, "run", boom)
    assert jobs.gpu_free_gb() is None


def test_gpu_free_gb_parses_mib_output(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "11264\n40960\n"

    monkeypatch.setattr(jobs.subprocess, "run", lambda *a, **k: Completed())
    free = jobs.gpu_free_gb()
    assert free == pytest.approx(40.0)


def test_cli_start_reports_refusal(tmp_path, capsys):
    jobs.start_job("busy", SLEEP_COMMAND, run_root=tmp_path)
    try:
        head = ["start", "--name", "other", "--run-root", str(tmp_path)]
        assert jobs.main([*head, "--", *SLEEP_COMMAND]) == 1
        assert "시작 거부" in capsys.readouterr().out
        assert jobs.main([*head, "--allow-concurrent", "--", *SLEEP_COMMAND]) == 0
    finally:
        jobs.stop_job("busy", run_root=tmp_path, timeout=5.0)
        jobs.stop_job("other", run_root=tmp_path, timeout=5.0)


# --------------------------------------------------------------------------
# 머신/부팅 식별자: 다른 VM이 남긴 기록의 pid를 믿지 않는다
# --------------------------------------------------------------------------


def test_machine_id_is_stamped_on_start_and_update(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    record = jobs.start_job("stamp", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert record["machine_id"] == "boot:aaa"
        updated = jobs._update_job("stamp", tmp_path, restarts=1)
        assert updated["machine_id"] == "boot:aaa"
    finally:
        jobs.stop_job("stamp", run_root=tmp_path, timeout=5.0)


def test_same_machine_record_is_still_alive(tmp_path, monkeypatch):
    """식별자가 같으면 지금까지와 똑같이 살아 있는 job으로 본다."""
    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    jobs.start_job("same", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("same", tmp_path)["alive"])
        status = jobs.job_status("same", tmp_path)
        assert status["foreign"] is False
        assert status["any_alive"]
        assert [j["name"] for j in jobs.running_jobs(tmp_path)] == ["same"]
        with pytest.raises(RuntimeError):
            jobs.start_job("same", SLEEP_COMMAND, run_root=tmp_path)
    finally:
        jobs.stop_job("same", run_root=tmp_path, timeout=5.0)


def test_other_machine_record_is_treated_as_dead(tmp_path, monkeypatch):
    """다른 머신이 남긴 기록은 pid가 살아 있어도 죽은 것으로 본다."""
    import signal as signal_mod

    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    old_pid = jobs.start_job("old", SLEEP_COMMAND, run_root=tmp_path)["pid"]
    try:
        assert wait_until(lambda: jobs.job_status("old", tmp_path)["alive"])

        monkeypatch.setattr(jobs, "machine_id", lambda: "boot:bbb")  # 새 VM으로 바뀌었다
        status = jobs.job_status("old", tmp_path)
        assert status["foreign"] is True
        assert status["alive"] is False and status["any_alive"] is False
        assert "다른 머신" in jobs._format_status(status)
        assert jobs.running_jobs(tmp_path) == []

        # 가드가 막지 않으므로 다른 이름의 job을 시작할 수 있다.
        jobs.start_job("new", SLEEP_COMMAND, run_root=tmp_path)
        jobs.stop_job("new", run_root=tmp_path, timeout=5.0)

        # 같은 이름으로 다시 시작하는 것도 막지 않는다.
        again = jobs.start_job("old", SLEEP_COMMAND, run_root=tmp_path)
        assert again["pid"] != old_pid
        assert jobs.pid_alive(old_pid), "무관한 기록 때문에 옛 프로세스를 건드리면 안 됩니다."
    finally:
        jobs.stop_job("old", run_root=tmp_path, timeout=5.0)  # 새로 띄운 쪽을 정리한다
        if jobs.pid_alive(old_pid):
            os.kill(old_pid, signal_mod.SIGKILL)


def test_stop_does_not_signal_other_machine_record(tmp_path, monkeypatch):
    """다른 머신의 기록이면 신호를 보내지 않고 그 사실을 돌려준다."""
    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    jobs.start_job("stale", SLEEP_COMMAND, run_root=tmp_path)
    pid = jobs.job_status("stale", tmp_path)["pid"]
    try:
        assert wait_until(lambda: jobs.pid_alive(pid))
        monkeypatch.setattr(jobs, "machine_id", lambda: "boot:bbb")
        result = jobs.stop_job("stale", run_root=tmp_path, timeout=5.0)
        assert result["action"] == "foreign_record"
        assert result["foreign"] is True
        assert jobs.pid_alive(pid), "무관한 프로세스에 신호를 보내면 안 됩니다."
    finally:
        monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
        jobs.stop_job("stale", run_root=tmp_path, timeout=5.0)


def test_record_without_machine_id_is_foreign_on_linux(tmp_path, monkeypatch):
    """식별자 필드가 없는 옛 기록은 boot_id를 읽을 수 있는 환경에서 죽은 것으로 본다."""
    runtime.atomic_write_json(
        jobs.pid_path("legacy", tmp_path),
        {"name": "legacy", "pid": os.getpid(), "child_pid": None},
    )
    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    assert jobs.running_jobs(tmp_path) == []
    assert jobs.job_status("legacy", tmp_path)["any_alive"] is False
    assert jobs.stop_job("legacy", run_root=tmp_path)["action"] == "foreign_record"

    # boot_id가 없는 환경(hostname 대체)에서는 판단을 포기하고 기존 동작을 유지한다.
    monkeypatch.setattr(jobs, "machine_id", lambda: "host:mac")
    assert [j["name"] for j in jobs.running_jobs(tmp_path)] == ["legacy"]
    assert jobs.job_status("legacy", tmp_path)["any_alive"] is True

    # 식별자를 전혀 얻지 못하는 환경도 마찬가지다.
    monkeypatch.setattr(jobs, "machine_id", lambda: None)
    assert [j["name"] for j in jobs.running_jobs(tmp_path)] == ["legacy"]


def test_machine_id_falls_back_to_hostname(monkeypatch):
    monkeypatch.setattr(jobs.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    value = jobs.machine_id()
    assert value is not None and value.startswith("host:")


def test_wait_refuses_other_machine_record(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:aaa")
    command = [sys.executable, "-c", "print('done', flush=True)"]
    jobs.start_job("far", command, run_root=tmp_path, auto_restart=1, cwd=str(REPO_ROOT))
    jobs.wait_job("far", run_root=tmp_path, poll=0.2, timeout=60.0)

    monkeypatch.setattr(jobs, "machine_id", lambda: "boot:bbb")
    result = jobs.wait_job("far", run_root=tmp_path, poll=0.2, timeout=5.0)
    assert result["result"] == "foreign"
    assert jobs.WAIT_EXIT_CODES["foreign"] == 1
    assert "다른 머신" in jobs._format_wait(result)


# --------------------------------------------------------------------------
# wait
# --------------------------------------------------------------------------


def test_wait_reports_success(tmp_path):
    command = [sys.executable, "-c", "print('done', flush=True)"]
    jobs.start_job("ok", command, run_root=tmp_path, auto_restart=1, cwd=str(REPO_ROOT))
    result = jobs.wait_job("ok", run_root=tmp_path, poll=0.2, timeout=60.0)
    assert result["result"] == "ok"
    assert result["exit_code"] == 0
    assert jobs.WAIT_EXIT_CODES[result["result"]] == 0


def test_wait_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.time, "sleep", lambda _seconds: None)
    command = [sys.executable, "-c", "print('boom', flush=True); raise SystemExit(4)"]
    jobs._update_job("bad", tmp_path, pid=-1, log=str(jobs.log_path("bad", tmp_path)))
    jobs.supervise("bad", command, max_restarts=0, run_root=tmp_path)
    result = jobs.wait_job("bad", run_root=tmp_path, poll=0.1)
    assert result["result"] == "failed"
    assert result["exit_code"] == 4
    assert jobs.main(["wait", "--name", "bad", "--run-root", str(tmp_path), "--poll", "0.1"]) == 1


def test_wait_times_out_and_returns_two(tmp_path, capsys):
    jobs.start_job("slow", SLEEP_COMMAND, run_root=tmp_path)
    try:
        assert wait_until(lambda: jobs.job_status("slow", tmp_path)["alive"])
        argv = ["wait", "--name", "slow", "--run-root", str(tmp_path), "--poll", "0.2", "--timeout", "1"]
        assert jobs.main(argv) == 2
        out = capsys.readouterr().out
        assert "대기 중" in out and "timeout" in out
    finally:
        jobs.stop_job("slow", run_root=tmp_path, timeout=5.0)


def test_wait_without_supervisor_is_unknown(tmp_path):
    """auto-restart 없이 띄우면 종료 코드를 알려 줄 감시 프로세스가 없다."""
    command = [sys.executable, "-c", "raise SystemExit(5)"]
    jobs.start_job("plain", command, run_root=tmp_path)
    result = jobs.wait_job("plain", run_root=tmp_path, poll=0.1, timeout=30.0)
    assert result["result"] == "unknown"
    assert result["exit_code"] is None
    assert jobs.WAIT_EXIT_CODES[result["result"]] == 0


def test_wait_for_unknown_job(tmp_path):
    result = jobs.wait_job("nope", run_root=tmp_path, poll=0.1, record_grace=0)
    assert result["result"] == "missing"
    assert (
        jobs.main(["wait", "--name", "nope", "--run-root", str(tmp_path), "--record-grace", "0"])
        == 1
    )


def test_wait_waits_out_record_grace_before_reporting_missing(tmp_path):
    """기록이 끝내 나타나지 않으면 record_grace를 다 기다린 뒤에야 missing이 된다."""
    started = time.time()
    result = jobs.wait_job("still-nope", run_root=tmp_path, poll=0.1, record_grace=0.5)
    elapsed = time.time() - started
    assert result["result"] == "missing"
    assert elapsed >= 0.5


def test_wait_finds_record_that_appears_during_grace(tmp_path, monkeypatch):
    """Drive처럼 쓰기 직후 읽기가 늦게 반영되는 상황을 흉내 낸다: 기록이 grace 도중 나타난다."""
    real_read_job = jobs.read_job
    calls = {"n": 0}

    def delayed_read_job(name, run_root=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return None
        return real_read_job(name, run_root)

    monkeypatch.setattr(jobs, "read_job", delayed_read_job)
    command = [sys.executable, "-c", "print('done', flush=True)"]
    jobs.start_job("late", command, run_root=tmp_path, auto_restart=1, cwd=str(REPO_ROOT))
    result = jobs.wait_job("late", run_root=tmp_path, poll=0.2, timeout=60.0, record_grace=10.0)
    assert result["result"] == "ok"
    assert result["exit_code"] == 0
    assert calls["n"] >= 3


def test_wait_prints_last_log_line(tmp_path):
    lines = []
    command = [sys.executable, "-c", "import time; print('진행 중', flush=True); time.sleep(120)"]
    jobs.start_job("noisy", command, run_root=tmp_path)
    try:
        assert wait_until(lambda: "진행 중" in jobs.tail_log("noisy", run_root=tmp_path))
        jobs.wait_job("noisy", run_root=tmp_path, poll=0.1, timeout=0.5, printer=lines.append)
        assert any("진행 중" in line for line in lines)
    finally:
        jobs.stop_job("noisy", run_root=tmp_path, timeout=5.0)


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
