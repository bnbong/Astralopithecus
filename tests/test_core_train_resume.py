"""checkpoint / 재개 동등성 테스트.

GPU도 큰 모델도 쓰지 않는다. 진짜 학습 루프(vqa.train.run_training)에 아주 작은
torch 모델과 결정적인 가짜 데이터를 끼워 넣고, 끊김 없이 N step을 돈 결과와
k step에서 죽였다가 N까지 재개한 결과가 완전히 같은지 본다.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from vqa import checkpoint as ckpt_mod  # noqa: E402
from vqa import train as train_mod  # noqa: E402


# --------------------------------------------------------------------------
# 작은 backend
# --------------------------------------------------------------------------


def _features(sample_id: str) -> torch.Tensor:
    """id만으로 정해지는 입력. 데이터 로딩을 흉내 낸다."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    values = [((digest[i] / 255.0) * 2.0 - 1.0) for i in range(8)]
    return torch.tensor(values, dtype=torch.float32)


def _target(sample_id: str) -> torch.Tensor:
    digest = hashlib.sha256(("y" + sample_id).encode("utf-8")).digest()
    return torch.tensor([digest[0] / 255.0], dtype=torch.float32)


class TinyBackend:
    """무작위 초기화된 작은 모델 + 결정적 가짜 데이터."""

    def __init__(self, ids: list[str], init_seed: int = 1234) -> None:
        torch.manual_seed(init_seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )
        self.ids = list(ids)
        self.seen: list[str] = []

    def sample_ids(self) -> list[str]:
        return list(self.ids)

    def make_micro_batch(self, ids, epoch):
        self.seen.extend(list(ids))
        x = torch.stack([_features(i) for i in ids])
        y = torch.stack([_target(i) for i in ids])
        return (x, y)

    def compute_loss(self, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_adapter(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "weights.pt", "wb") as handle:
            torch.save(self.model.state_dict(), handle)
            handle.flush()

    def load_adapter(self, src: Path) -> None:
        state = torch.load(Path(src) / "weights.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)


def make_cfg(**train_overrides):
    cfg = {
        "train": {
            "lr": 0.05,
            "epochs": 1,
            "micro_batch": 1,
            "grad_accum": 2,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "seed": 7,
            "save_every_steps": 2,
            "save_every_minutes": 0,
            "keep_last": 2,
            "heartbeat_every_steps": 1,
            "log_every_steps": 1,
        }
    }
    cfg["train"].update(train_overrides)
    return cfg


IDS = [f"train_{i:04d}.jpg" for i in range(20)]


def flat_state(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def optimizer_tensors(optimizer) -> list[torch.Tensor]:
    out = []
    state = optimizer.state_dict()["state"]
    for key in sorted(state):
        for name in sorted(state[key]):
            value = state[key][name]
            if torch.is_tensor(value):
                out.append(value.detach().clone().reshape(-1))
            else:
                out.append(torch.tensor([float(value)]))
    return out


# --------------------------------------------------------------------------
# (a) 재개 동등성
# --------------------------------------------------------------------------


def test_resume_matches_uninterrupted_run(tmp_path):
    cfg = make_cfg()

    straight = TinyBackend(IDS)
    out_a = train_mod.run_training(
        straight,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "runA",
        scratch_dir=tmp_path / "scratchA",
        run_id="A",
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="none",
        save_final=False,
    )
    assert out_a.state.global_step == 10  # 20 표본 / (micro 1 * accum 2)

    part1 = TinyBackend(IDS)
    out_b1 = train_mod.run_training(
        part1,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "runB",
        scratch_dir=tmp_path / "scratchB",
        run_id="B",
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="none",
        max_steps=4,
        save_final=False,
    )
    assert out_b1.reason == "max_steps"
    assert out_b1.state.global_step == 4

    # 세션이 죽었다고 보고 완전히 새 backend를 만든다.
    part2 = TinyBackend(IDS, init_seed=999)  # 초기값이 달라도 adapter 복원으로 같아져야 한다
    out_b2 = train_mod.run_training(
        part2,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "runB",
        scratch_dir=tmp_path / "scratchB",
        run_id="B",
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="auto",
        save_final=False,
    )
    assert out_b2.state.global_step == 10

    # 1. 최종 가중치가 정확히 같아야 한다.
    assert torch.equal(flat_state(straight.model), flat_state(part2.model))

    # 2. optimizer 상태도 같아야 한다.
    a_state = optimizer_tensors(out_a.optimizer)
    b_state = optimizer_tensors(out_b2.optimizer)
    assert len(a_state) == len(b_state)
    for lhs, rhs in zip(a_state, b_state):
        assert torch.equal(lhs, rhs)

    # 3. 표본 소비 순서가 같아야 한다.
    assert part1.seen + part2.seen == straight.seen
    assert len(straight.seen) == len(IDS)


def test_resume_skips_consumed_samples_without_touching_them(tmp_path):
    """재개한 프로세스는 이미 쓴 표본을 다시 만들지 않는다 (이미지 로딩 회피)."""
    cfg = make_cfg()
    part1 = TinyBackend(IDS)
    train_mod.run_training(
        part1,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        max_steps=3,
        save_final=False,
    )
    part2 = TinyBackend(IDS)
    train_mod.run_training(
        part2,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="auto",
        save_final=False,
    )
    assert len(part1.seen) == 6
    assert len(part2.seen) == 14
    assert set(part1.seen) & set(part2.seen) == set()


def test_epoch_order_is_pure_function_of_seed_and_epoch():
    first = train_mod.epoch_order(IDS, 42, 0)
    again = train_mod.epoch_order(IDS, 42, 0)
    other_epoch = train_mod.epoch_order(IDS, 42, 1)
    other_seed = train_mod.epoch_order(IDS, 43, 0)
    assert first == again
    assert sorted(first) == sorted(IDS)
    assert first != other_epoch
    assert first != other_seed


# --------------------------------------------------------------------------
# (b) 최신 checkpoint 손상 -> 이전 것으로 되돌아가기
# --------------------------------------------------------------------------


def test_corrupt_newest_checkpoint_falls_back(tmp_path):
    cfg = make_cfg(save_every_steps=2, keep_last=5)
    backend = TinyBackend(IDS)
    train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        max_steps=6,
        save_final=False,
    )
    checkpoints = ckpt_mod.list_checkpoints(tmp_path / "run")
    assert [ckpt_mod.parse_step(p) for p in checkpoints] == [2, 4, 6]

    newest = checkpoints[-1]
    with open(newest / "optimizer.pt", "ab") as handle:
        handle.write(b"\x00corrupted")

    ok, reason = ckpt_mod.verify_checkpoint(newest)
    assert not ok and "해시" in reason

    chosen, notes = ckpt_mod.find_resume_checkpoint(tmp_path / "run")
    assert chosen is not None
    assert ckpt_mod.parse_step(chosen) == 4
    assert notes and "step_000006" in notes[0]

    # 재개 자체도 step 4에서 이어진다.
    resumed = TinyBackend(IDS, init_seed=555)
    outcome = train_mod.run_training(
        resumed,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="auto",
        save_final=False,
    )
    assert outcome.state.global_step == 10
    assert len(resumed.seen) == 12  # step 4 이후 남은 12 표본


def test_missing_complete_marker_is_not_resumable(tmp_path):
    cfg = make_cfg(save_every_steps=2)
    backend = TinyBackend(IDS)
    train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        max_steps=2,
        save_final=False,
    )
    newest = ckpt_mod.list_checkpoints(tmp_path / "run")[-1]
    (newest / ckpt_mod.COMPLETE_NAME).unlink()
    ok, reason = ckpt_mod.verify_checkpoint(newest)
    assert not ok and "COMPLETE" in reason
    chosen, _ = ckpt_mod.find_resume_checkpoint(tmp_path / "run")
    assert chosen is None


# --------------------------------------------------------------------------
# (c) config / split 해시 불일치
# --------------------------------------------------------------------------


def test_config_hash_mismatch_refuses_resume(tmp_path):
    cfg = make_cfg(save_every_steps=2)
    backend = TinyBackend(IDS)
    train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="hash-one",
        split_hash="split-one",
        resume="none",
        max_steps=2,
        save_final=False,
    )

    with pytest.raises(ckpt_mod.ResumeRefused):
        train_mod.run_training(
            TinyBackend(IDS),
            copy.deepcopy(cfg),
            run_dir=tmp_path / "run",
            scratch_dir=tmp_path / "scratch",
            run_id="R",
            config_hash="hash-TWO",
            split_hash="split-one",
            resume="auto",
            save_final=False,
        )

    with pytest.raises(ckpt_mod.ResumeRefused):
        train_mod.run_training(
            TinyBackend(IDS),
            copy.deepcopy(cfg),
            run_dir=tmp_path / "run",
            scratch_dir=tmp_path / "scratch",
            run_id="R",
            config_hash="hash-one",
            split_hash="split-TWO",
            resume="auto",
            save_final=False,
        )

    # --force-resume에 해당하는 경로는 통과한다.
    forced = train_mod.run_training(
        TinyBackend(IDS),
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="hash-TWO",
        split_hash="split-one",
        resume="auto",
        force_resume=True,
        save_final=False,
    )
    assert forced.state.global_step == 10


# --------------------------------------------------------------------------
# 보관 개수 / 완료 표시 / heartbeat
# --------------------------------------------------------------------------


def test_keep_last_prunes_old_checkpoints(tmp_path):
    cfg = make_cfg(save_every_steps=1, keep_last=2)
    backend = TinyBackend(IDS)
    train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        max_steps=6,
        save_final=False,
    )
    steps = [ckpt_mod.parse_step(p) for p in ckpt_mod.list_checkpoints(tmp_path / "run")]
    assert steps == [5, 6]
    latest = ckpt_mod.read_latest(tmp_path / "run")
    assert latest["step"] == 6


def test_final_adapter_marks_run_finished(tmp_path):
    cfg = make_cfg()
    backend = TinyBackend(IDS)
    run_dir = tmp_path / "run"
    assert not ckpt_mod.training_is_finished(run_dir)
    train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=run_dir,
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        save_final=True,
    )
    assert ckpt_mod.training_is_finished(run_dir)
    assert (run_dir / "train_log.jsonl").exists()
    heartbeat = run_dir / "heartbeat.json"
    assert heartbeat.exists()


def test_signal_stop_saves_checkpoint(tmp_path):
    """SIGTERM 상당의 플래그가 켜지면 step 경계에서 저장하고 빠져나온다."""
    cfg = make_cfg(save_every_steps=0, save_every_minutes=0)
    backend = TinyBackend(IDS)
    flag = train_mod.StopFlag()
    flag.requested = True
    flag.signal_name = "SIGTERM"
    outcome = train_mod.run_training(
        backend,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
        run_id="R",
        config_hash="h",
        split_hash="s",
        resume="none",
        stop_flag=flag,
        save_final=False,
    )
    assert outcome.reason == "stopped:SIGTERM"
    assert outcome.state.global_step == 1
    chosen, _ = ckpt_mod.find_resume_checkpoint(tmp_path / "run")
    assert chosen is not None and ckpt_mod.parse_step(chosen) == 1
