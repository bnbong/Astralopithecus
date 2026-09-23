"""epoch 경계 adapter 보존과 4지 제한 손실 테스트.

두 기능 모두 설정 파일에 키를 적었을 때에만 켜진다. 그래서 여기에서 확인하는 것은 두
가지다. 켰을 때 약속한 대로 동작하는가, 그리고 끄면 지금까지와 한 글자도 달라지지 않는가.
두 번째가 더 중요하다. 이미 돌고 있는 27B run의 재개와 손실 값이 이 변경으로 흔들리면 안
되기 때문이다.

GPU도 큰 모델도 쓰지 않는다. 학습 루프는 test_core_train_resume.py와 같은 방식으로 아주
작은 torch 모델을 끼워 넣고, 손실 쪽은 logits을 직접 만들어 넣는다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from vqa import checkpoint as ckpt_mod  # noqa: E402
from vqa import config as config_mod  # noqa: E402
from vqa import train as train_mod  # noqa: E402


# --------------------------------------------------------------------------
# 작은 backend
# --------------------------------------------------------------------------


def _features(sample_id: str) -> "torch.Tensor":
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return torch.tensor([((digest[i] / 255.0) * 2.0 - 1.0) for i in range(8)], dtype=torch.float32)


def _target(sample_id: str) -> "torch.Tensor":
    digest = hashlib.sha256(("y" + sample_id).encode("utf-8")).digest()
    return torch.tensor([digest[0] / 255.0], dtype=torch.float32)


class TinyBackend:
    """무작위 초기화된 작은 모델 + 결정적인 가짜 데이터."""

    def __init__(self, ids: list[str], init_seed: int = 1234) -> None:
        torch.manual_seed(init_seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )
        self.ids = list(ids)

    def sample_ids(self) -> list[str]:
        return list(self.ids)

    def make_micro_batch(self, ids, epoch):
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


IDS = [f"train_{i:04d}.jpg" for i in range(8)]


def make_cfg(**train_overrides):
    cfg = {
        "train": {
            "lr": 0.05,
            "epochs": 3,
            "micro_batch": 1,
            "grad_accum": 2,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "seed": 7,
            "save_every_steps": 1,
            "save_every_minutes": 0,
            "keep_last": 2,
            "heartbeat_every_steps": 1,
            "log_every_steps": 1,
        }
    }
    cfg["train"].update(train_overrides)
    return cfg


def train_once(tmp_path: Path, cfg: dict, name: str = "run", **kwargs):
    return train_mod.run_training(
        TinyBackend(IDS),
        copy.deepcopy(cfg),
        run_dir=tmp_path / name,
        scratch_dir=tmp_path / ("scratch_" + name),
        run_id=name,
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="none",
        **kwargs,
    )


# --------------------------------------------------------------------------
# (a) epoch 경계 adapter
# --------------------------------------------------------------------------


def test_epoch_adapters_are_written_for_every_epoch(tmp_path):
    """epoch마다 하나씩, final_adapter와 같은 형식으로 남는다."""
    outcome = train_once(tmp_path, make_cfg(keep_epoch_adapters=True))
    run_dir = tmp_path / "run"

    assert outcome.state.global_step == 12  # 8표본 / accum 2 = 4 step * 3 epoch
    assert [p.name for p in outcome.epoch_adapters] == ["epoch_1", "epoch_2", "epoch_3"]
    for path in outcome.epoch_adapters:
        assert path.parent == ckpt_mod.epoch_adapters_dir(run_dir)
        ok, reason = ckpt_mod.verify_checkpoint(path)
        assert ok, reason
        # 추론에 그대로 넘길 수 있어야 한다. adapter 가중치가 있고 optimizer 상태는 없다.
        assert (path / "weights.pt").exists()
        assert not (path / ckpt_mod.OPTIMIZER_NAME).exists()
        assert not (path / ckpt_mod.SCHEDULER_NAME).exists()

    # final_adapter와 파일 구성이 같아야 vqa.predict --adapter가 둘을 구분하지 않는다.
    final = ckpt_mod.final_adapter_dir(run_dir)
    assert sorted(p.name for p in final.iterdir()) == sorted(
        p.name for p in outcome.epoch_adapters[-1].iterdir()
    )

    # 마지막 epoch의 adapter는 final_adapter와 같은 가중치다.
    last = torch.load(outcome.epoch_adapters[-1] / "weights.pt", map_location="cpu", weights_only=False)
    done = torch.load(final / "weights.pt", map_location="cpu", weights_only=False)
    for key in done:
        assert torch.equal(last[key], done[key])


def test_epoch_boundary_checkpoints_survive_keep_last(tmp_path):
    """keep_last 정리가 epoch 경계 checkpoint를 지우지 않는다."""
    outcome = train_once(tmp_path, make_cfg(keep_epoch_adapters=True, keep_last=1))
    run_dir = tmp_path / "run"

    kept = ckpt_mod.list_checkpoints(run_dir)
    boundaries = {ckpt_mod.parse_step(p): ckpt_mod.epoch_boundary_of(p) for p in kept}
    # 4, 8, 12 step이 각 epoch의 마지막 step이다.
    assert {step for step, epoch in boundaries.items() if epoch is not None} == {4, 8, 12}
    assert boundaries[4] == 1 and boundaries[8] == 2 and boundaries[12] == 3
    # 경계가 아닌 checkpoint는 keep_last 개만 남는다.
    assert sum(1 for epoch in boundaries.values() if epoch is None) <= 1
    assert outcome.finished


def test_epoch_adapters_stay_off_by_default(tmp_path):
    """설정에 키가 없으면 디렉터리도 표시도 생기지 않는다."""
    outcome = train_once(tmp_path, make_cfg())
    run_dir = tmp_path / "run"
    assert outcome.epoch_adapters == []
    assert not ckpt_mod.epoch_adapters_dir(run_dir).exists()
    for path in ckpt_mod.list_checkpoints(run_dir):
        assert ckpt_mod.epoch_boundary_of(path) is None
    # keep_last 가 2이므로 정리도 예전 그대로다.
    assert len(ckpt_mod.list_checkpoints(run_dir)) == 2


def test_epoch_adapters_need_two_epochs(tmp_path):
    """epoch이 하나뿐이면 final_adapter와 같은 값이라 저장하지 않는다."""
    outcome = train_once(tmp_path, make_cfg(epochs=1, keep_epoch_adapters=True))
    assert outcome.epoch_adapters == []
    assert not ckpt_mod.epoch_adapters_dir(tmp_path / "run").exists()


def test_epoch_boundary_mark_does_not_leak_into_later_checkpoints(tmp_path):
    """재개해도 표시가 다음 checkpoint로 옮겨붙지 않는다."""
    cfg = make_cfg(keep_epoch_adapters=True)
    run_dir = tmp_path / "run"
    scratch = tmp_path / "scratch"
    first = TinyBackend(IDS)
    train_mod.run_training(
        first,
        copy.deepcopy(cfg),
        run_dir=run_dir,
        scratch_dir=scratch,
        run_id="run",
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="none",
        max_steps=4,  # epoch 1의 마지막 step에서 멈춘다
        save_final=False,
    )
    assert ckpt_mod.epoch_boundary_of(ckpt_mod.checkpoints_dir(run_dir) / "step_000004") == 1

    second = TinyBackend(IDS, init_seed=999)
    outcome = train_mod.run_training(
        second,
        copy.deepcopy(cfg),
        run_dir=run_dir,
        scratch_dir=scratch,
        run_id="run",
        config_hash="cfg-hash",
        split_hash="split-hash",
        resume="auto",
    )
    assert outcome.state.global_step == 12
    marked = {
        ckpt_mod.parse_step(p)
        for p in ckpt_mod.list_checkpoints(run_dir)
        if ckpt_mod.epoch_boundary_of(p) is not None
    }
    assert marked == {4, 8, 12}


def test_resume_equivalence_is_unchanged_with_epoch_adapters(tmp_path):
    """epoch adapter를 켜도 재개 결과가 끊김 없이 돈 결과와 같아야 한다."""
    cfg = make_cfg(keep_epoch_adapters=True)

    straight = TinyBackend(IDS)
    train_mod.run_training(
        straight,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "A",
        scratch_dir=tmp_path / "sA",
        run_id="A",
        config_hash="c",
        split_hash="s",
        resume="none",
        save_final=False,
    )

    part1 = TinyBackend(IDS)
    train_mod.run_training(
        part1,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "B",
        scratch_dir=tmp_path / "sB",
        run_id="B",
        config_hash="c",
        split_hash="s",
        resume="none",
        max_steps=5,
        save_final=False,
    )
    part2 = TinyBackend(IDS, init_seed=999)
    train_mod.run_training(
        part2,
        copy.deepcopy(cfg),
        run_dir=tmp_path / "B",
        scratch_dir=tmp_path / "sB",
        run_id="B",
        config_hash="c",
        split_hash="s",
        resume="auto",
        save_final=False,
    )

    lhs = torch.cat([p.detach().reshape(-1) for p in straight.model.parameters()])
    rhs = torch.cat([p.detach().reshape(-1) for p in part2.model.parameters()])
    assert torch.equal(lhs, rhs)


# --------------------------------------------------------------------------
# (b) 4지 제한 손실
# --------------------------------------------------------------------------

CHOICE_IDS = [64, 65, 66, 67]


def make_logits_and_labels(seq_len: int = 6, vocab: int = 100, batch: int = 3, position: int = 4):
    """감독 위치가 하나뿐인 labels와 그 자리를 예측하는 logits을 만든다."""
    torch.manual_seed(0)
    logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)
    for row in range(batch):
        labels[row, position] = CHOICE_IDS[row % 4]
    return logits, labels


def test_choice4_loss_matches_a_hand_written_four_way_ce():
    logits, labels = make_logits_and_labels()
    loss = train_mod.choice4_loss(logits, labels, CHOICE_IDS)

    expected = []
    for row in range(labels.shape[0]):
        position = int(torch.where(labels[row] != -100)[0][0])
        picked = logits[row, position - 1, CHOICE_IDS]
        gold = CHOICE_IDS.index(int(labels[row, position]))
        expected.append(
            torch.nn.functional.cross_entropy(picked.unsqueeze(0), torch.tensor([gold]))
        )
    assert torch.allclose(loss, torch.stack(expected).mean(), atol=1e-6)


def test_choice4_loss_ignores_everything_outside_the_four_letters():
    """a/b/c/d 밖의 logits을 아무리 키워도 손실이 달라지지 않는다."""
    logits, labels = make_logits_and_labels()
    before = train_mod.choice4_loss(logits, labels, CHOICE_IDS)
    other = [i for i in range(logits.shape[2]) if i not in CHOICE_IDS]
    logits[:, :, other] += 50.0
    after = train_mod.choice4_loss(logits, labels, CHOICE_IDS)
    assert torch.allclose(before, after, atol=1e-6)


def test_choice4_loss_backpropagates_to_the_supervised_position_only():
    logits, labels = make_logits_and_labels()
    logits.requires_grad_(True)
    train_mod.choice4_loss(logits, labels, CHOICE_IDS).backward()
    grad = logits.grad
    position = int(torch.where(labels[0] != -100)[0][0])
    assert bool((grad[:, position - 1, CHOICE_IDS] != 0).all())
    # 그 한 자리, 그 네 개의 열을 뺀 나머지는 손실에 들어가지 않았다.
    mask = torch.ones_like(grad, dtype=torch.bool)
    mask[:, position - 1, CHOICE_IDS] = False
    assert bool((grad[mask] == 0).all())


def test_choice4_loss_refuses_labels_with_more_than_one_supervised_token():
    logits, labels = make_logits_and_labels()
    labels[0, 2] = CHOICE_IDS[0]
    with pytest.raises(ValueError, match="감독 토큰이 1개가 아닙니다"):
        train_mod.choice4_loss(logits, labels, CHOICE_IDS)


def test_choice4_loss_refuses_a_gold_token_outside_the_four():
    logits, labels = make_logits_and_labels()
    labels[labels != -100] = 999
    with pytest.raises(ValueError, match="a/b/c/d 토큰 id 안에 없습니다"):
        train_mod.choice4_loss(logits, labels, CHOICE_IDS)


# --------------------------------------------------------------------------
# (c) backend 경로
# --------------------------------------------------------------------------


class FakeChoiceTokens:
    ids = CHOICE_IDS


class FakeVLMOutput:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss


class FakeVLM:
    """logits과 자체 loss를 함께 돌려주는 가짜 모델."""

    def __init__(self, logits, loss_value: float = 12.5) -> None:
        self.logits = logits
        self.loss_value = loss_value
        self.calls: list[list[str]] = []

    def __call__(self, **kwargs):
        self.calls.append(sorted(kwargs))
        return FakeVLMOutput(self.logits, torch.tensor(self.loss_value))


def make_backend(loss_mode: str, logits):
    backend = train_mod.VLMBackend(cfg={}, data_root=Path("."), records={}, ids=[])
    backend.model = FakeVLM(logits)
    backend.device = "cpu"
    backend.loss_mode = loss_mode
    backend.choice_tokens = FakeChoiceTokens()
    return backend


def test_full_vocab_backend_path_is_untouched():
    """기본값에서는 labels를 그대로 넘기고 모델이 낸 loss를 그대로 쓴다."""
    logits, labels = make_logits_and_labels()
    backend = make_backend("full_vocab", logits)
    loss = backend.compute_loss({"input_ids": torch.zeros(3, 6, dtype=torch.long), "labels": labels})
    assert float(loss) == 12.5
    assert backend.model.calls == [["input_ids", "labels", "use_cache"]]


def test_choice4_backend_path_drops_labels_and_scores_four():
    logits, labels = make_logits_and_labels()
    backend = make_backend("choice4", logits)
    loss = backend.compute_loss({"input_ids": torch.zeros(3, 6, dtype=torch.long), "labels": labels})
    # 모델에게 전체 vocab CE를 시키지 않는다.
    assert backend.model.calls == [["input_ids", "use_cache"]]
    assert torch.allclose(loss, train_mod.choice4_loss(logits, labels, CHOICE_IDS))


# --------------------------------------------------------------------------
# (d) 설정 키
# --------------------------------------------------------------------------


def test_new_keys_are_absent_from_defaults():
    """DEFAULTS에 들어가면 이미 돌고 있는 run의 config_hash가 바뀐다."""
    assert "loss" not in config_mod.DEFAULTS["train"]
    assert "keep_epoch_adapters" not in config_mod.DEFAULTS["train"]


def test_config_hash_only_moves_when_the_file_says_so():
    base = config_mod.load_config(None)
    assert config_mod.config_hash(base) == config_mod.config_hash(config_mod.load_config(None))
    changed = config_mod.load_config(None, ["train.loss=choice4"])
    assert config_mod.config_hash(changed) != config_mod.config_hash(base)
    epochs = config_mod.load_config(None, ["train.keep_epoch_adapters=true"])
    assert config_mod.config_hash(epochs) != config_mod.config_hash(base)


def test_config_rejects_unknown_loss_mode():
    with pytest.raises(config_mod.ConfigError, match="train.loss"):
        config_mod.load_config(None, ["train.loss=margin"])
    with pytest.raises(config_mod.ConfigError, match="keep_epoch_adapters"):
        config_mod.load_config(None, ["train.keep_epoch_adapters=2"])


#: 새 키(`train.loss`, `train.keep_epoch_adapters`)를 의도적으로 켜는 실험용 설정.
#: 이 테스트의 목적은 기본 설정 A/B의 config_hash가 실수로 바뀌지 않도록 보호하는
#: 것이지, 실험 설정까지 새 키를 못 켜게 막는 것이 아니므로 이 목록만 예외로 둔다.
EXPERIMENT_CONFIGS = {"colab_qwen35_27b_bf16_upscale_3ep.yaml"}


def test_shipped_configs_do_not_turn_the_new_keys_on():
    """저장소의 설정 파일 가운데 어느 것도 기본 동작을 바꾸지 않는다."""
    repo = Path(__file__).resolve().parents[1]
    pytest.importorskip("yaml")
    configs_dir = repo / "configs"
    for name in EXPERIMENT_CONFIGS:
        assert (configs_dir / name).is_file(), name
    for path in sorted(configs_dir.glob("*.yaml")):
        if path.name in EXPERIMENT_CONFIGS:
            continue
        raw = json.loads(json.dumps(config_mod.load_yaml(path)))
        train = raw.get("train") or {}
        assert "loss" not in train, path.name
        assert "keep_epoch_adapters" not in train, path.name
