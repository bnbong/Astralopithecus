"""answer-only LoRA 학습 루프와 CLI.

루프는 모델에 의존하지 않는다. `TrainingBackend` 프로토콜만 만족하면 무엇이든 넣을
수 있어서, 테스트에서는 아주 작은 torch 모델과 가짜 데이터로 같은 루프를 돌린다.

데이터 순서는 (seed, epoch)만의 순수 함수다. 그래서 재개할 때 이미 소비한 표본을
이미지를 열지 않고 건너뛸 수 있다.

사용 예:
    python -m vqa.train --config configs/lab_qwen25vl_3b.yaml --run-id lab3b_v1 \\
        --split artifacts/data/split_v1.json
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Protocol, Sequence

from . import checkpoint as ckpt_mod
from . import config as config_mod
from . import runtime
from .checkpoint import TrainerState


# --------------------------------------------------------------------------
# 데이터 순서
# --------------------------------------------------------------------------


def epoch_order(sample_ids: Sequence[str], seed: int, epoch: int) -> list[str]:
    """(seed, epoch)만으로 정해지는 표본 순서.

    재개할 때 같은 순서를 다시 만들 수 있어야 하므로 전역 RNG를 쓰지 않는다.
    """
    import numpy as np

    rng = np.random.RandomState((int(seed) * 1_000_003 + int(epoch) * 7919) % (2**32))
    order = rng.permutation(len(sample_ids))
    return [sample_ids[int(i)] for i in order]


# --------------------------------------------------------------------------
# backend 프로토콜
# --------------------------------------------------------------------------


class TrainingBackend(Protocol):
    """학습 루프가 모델에 대해 알아야 하는 전부."""

    def sample_ids(self) -> list[str]:
        ...

    def make_micro_batch(self, ids: Sequence[str], epoch: int) -> Any:
        ...

    def compute_loss(self, batch: Any) -> Any:
        ...

    def trainable_parameters(self) -> list[Any]:
        ...

    def save_adapter(self, dest: Path) -> None:
        ...

    def load_adapter(self, src: Path) -> None:
        ...


# --------------------------------------------------------------------------
# optimizer / scheduler
# --------------------------------------------------------------------------


def build_optimizer(params: Sequence[Any], cfg: dict[str, Any]):
    import torch

    betas = config_mod.get_path(cfg, "train.adam_betas", [0.9, 0.95]) or [0.9, 0.95]
    return torch.optim.AdamW(
        list(params),
        lr=float(config_mod.get_path(cfg, "train.lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(config_mod.get_path(cfg, "train.weight_decay", 0.01)),
    )


def build_scheduler(optimizer: Any, name: str, warmup_steps: int, total_steps: int):
    """warmup + linear/cosine/constant. transformers에 의존하지 않는다."""
    import torch

    name = (name or "linear").lower()
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# 4지 제한 손실
# --------------------------------------------------------------------------


def choice4_loss(logits: Any, labels: Any, choice_ids: Sequence[int]) -> Any:
    """감독 위치의 logits에서 a/b/c/d 네 개만 뽑아 4-way cross entropy를 만든다.

    collator가 표본마다 정답 letter 토큰 한 자리에만 label을 넣으므로, 감독 위치는
    `labels != -100` 인 자리 하나다. 그 토큰을 예측하는 logits은 한 칸 앞에 있다.
    전체 vocab에 대한 CE와 달리 오답 letter 세 개와만 경쟁하므로, 모델이 letter가 아닌
    토큰을 눌러야 할 까닭이 사라진다.

    vocab 전체를 softmax하지 않고 4개만 골라 쓰므로 추론의 scoring과 같은 비교를 학습에서
    그대로 한다. 수치가 흔들리지 않도록 float32로 올려서 계산한다.
    """
    import torch

    if labels.dim() != 2:
        raise ValueError(f"labels는 (batch, seq) 모양이어야 합니다: {tuple(labels.shape)}")
    if logits.dim() != 3:
        raise ValueError(f"logits는 (batch, seq, vocab) 모양이어야 합니다: {tuple(logits.shape)}")
    ids = [int(i) for i in choice_ids]
    if len(ids) != 4:
        raise ValueError(f"choice_ids는 a/b/c/d 네 개여야 합니다: {ids}")
    if len(set(ids)) != 4:
        raise ValueError(f"choice_ids가 서로 다르지 않습니다: {ids}")

    supervised = labels != -100
    counts = supervised.sum(dim=1)
    if not bool((counts == 1).all()):
        raise ValueError(f"표본당 감독 토큰이 1개가 아닙니다: {counts.tolist()}")
    positions = torch.argmax(supervised.to(torch.int32), dim=1)
    if not bool((positions > 0).all()):
        raise ValueError("감독 위치가 sequence 맨 앞입니다. 그 토큰을 예측하는 logits이 없습니다.")

    rows = torch.arange(labels.shape[0], device=logits.device)
    choice = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
    # 한 칸 앞의 logits이 감독 위치의 토큰을 예측한다. 모델 내부 CE와 같은 정렬이다.
    picked = logits[rows, positions.to(logits.device) - 1, :].index_select(1, choice)

    gold = labels[rows.to(labels.device), positions]
    match = gold.to(choice.device).unsqueeze(1) == choice.unsqueeze(0)
    if not bool(match.any(dim=1).all()):
        raise ValueError(
            "감독 토큰이 a/b/c/d 토큰 id 안에 없습니다. collator와 scoring이 다른 토큰을 "
            f"보고 있습니다: labels={gold.tolist()} choice_ids={ids}"
        )
    target = torch.argmax(match.to(torch.int32), dim=1)
    return torch.nn.functional.cross_entropy(picked.float(), target)


# --------------------------------------------------------------------------
# 중단 신호
# --------------------------------------------------------------------------


class StopFlag:
    """SIGTERM/SIGINT를 받으면 켜지는 플래그. Windows에서도 안전하게 설치한다."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._previous: list[tuple[int, Any]] = []

    def install(self) -> "StopFlag":
        names = ["SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"]
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                previous = signal.signal(sig, self._handle)
            except (ValueError, OSError, RuntimeError):
                # 메인 스레드가 아니거나 해당 플랫폼이 지원하지 않는 신호
                continue
            self._previous.append((sig, previous))
        return self

    def restore(self) -> None:
        for sig, previous in self._previous:
            try:
                signal.signal(sig, previous)
            except Exception:
                pass
        self._previous.clear()

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        self.requested = True
        self.signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n[train] {self.signal_name} 수신. 다음 optimizer step 경계에서 저장하고 종료합니다.", flush=True)


# --------------------------------------------------------------------------
# 학습 루프
# --------------------------------------------------------------------------


@dataclass
class TrainOutcome:
    """루프가 끝난 이유와 최종 상태."""

    state: TrainerState
    finished: bool
    reason: str
    last_checkpoint: Path | None = None
    steps_done: int = 0
    #: 테스트와 후처리에서 상태를 확인할 수 있도록 그대로 노출한다.
    optimizer: Any = None
    scheduler: Any = None
    #: save_final=True로 끝났을 때 저장된 final_adapter 경로.
    final_adapter: Path | None = None
    #: train.keep_epoch_adapters가 켜져 있을 때 이번 실행이 저장한 epoch adapter 경로들.
    epoch_adapters: list[Path] = dataclass_field(default_factory=list)


def run_training(
    backend: TrainingBackend,
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    scratch_dir: Path,
    run_id: str,
    config_hash: str,
    split_hash: str,
    resume: str = "auto",
    force_resume: bool = False,
    max_steps: int | None = None,
    stop_flag: StopFlag | None = None,
    save_final: bool = True,
) -> TrainOutcome:
    """checkpoint/재개를 포함한 전체 학습 루프."""
    import torch

    run_dir = runtime.ensure_dir(run_dir)
    scratch_dir = runtime.ensure_dir(scratch_dir)

    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 1))
    micro_batch = int(train_cfg.get("micro_batch", 1))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_every_steps = int(train_cfg.get("save_every_steps", 50) or 0)
    save_every_minutes = float(train_cfg.get("save_every_minutes", 15) or 0)
    keep_last = int(train_cfg.get("keep_last", 2))
    # 설정 파일에 적어 두었을 때에만 켜지는 키다. DEFAULTS에 없으므로 기존 run의 config_hash는
    # 그대로다. epoch이 하나뿐이면 final_adapter와 같은 값이라 저장할 까닭이 없다.
    keep_epoch_adapters = bool(train_cfg.get("keep_epoch_adapters", False)) and epochs >= 2
    heartbeat_every = max(int(train_cfg.get("heartbeat_every_steps", 5) or 1), 1)
    log_every = max(int(train_cfg.get("log_every_steps", 1) or 1), 1)

    ids = list(backend.sample_ids())
    num_samples = len(ids)
    if num_samples == 0:
        raise ValueError("학습 표본이 없습니다.")

    micro_per_epoch = math.ceil(num_samples / micro_batch)
    steps_per_epoch = math.ceil(micro_per_epoch / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(round(total_steps * float(train_cfg.get("warmup_ratio", 0.03))))

    optimizer = build_optimizer(backend.trainable_parameters(), cfg)
    scheduler = build_scheduler(optimizer, str(train_cfg.get("scheduler", "linear")), warmup_steps, total_steps)

    state = TrainerState(
        epoch=0,
        global_step=0,
        samples_consumed_in_epoch=0,
        data_order_seed=seed,
        config_hash=config_hash,
        split_hash=split_hash,
        loss_ema=None,
        total_steps=total_steps,
        run_id=run_id,
    )

    # ---- 재개 ----
    resumed_from: Path | None = None
    if resume and resume != "none":
        target: Path | None = None
        if resume == "auto":
            target, notes = ckpt_mod.find_resume_checkpoint(run_dir)
            for note in notes:
                print(f"[train] checkpoint 건너뜀 — {note}", flush=True)
        else:
            target = Path(resume)
            if not target.exists():
                raise FileNotFoundError(f"--resume 경로가 없습니다: {target}")
        if target is not None and target.exists():
            state = ckpt_mod.load_checkpoint(
                target,
                adapter_loader=backend.load_adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                restore_rng=True,
                expected_config_hash=config_hash,
                expected_split_hash=split_hash,
                force=force_resume,
            )
            state.config_hash = config_hash
            state.split_hash = split_hash
            state.total_steps = total_steps
            # epoch 경계 표시는 그 checkpoint를 설명하는 값이지 지금 상태를 설명하는 값이
            # 아니다. 지우지 않으면 재개 뒤에 저장하는 checkpoint마다 표시가 따라붙어서
            # 정리 단계가 아무것도 지우지 못한다.
            if isinstance(state.extra, dict):
                state.extra.pop(ckpt_mod.EPOCH_BOUNDARY_KEY, None)
            resumed_from = target
            print(
                f"[train] 재개: {target.name} epoch={state.epoch} step={state.global_step} "
                f"consumed={state.samples_consumed_in_epoch}/{num_samples}",
                flush=True,
            )
        else:
            runtime.seed_everything(seed)
    else:
        runtime.seed_everything(seed)

    stop = stop_flag if stop_flag is not None else StopFlag().install()
    log_path = run_dir / "train_log.jsonl"
    heartbeat_path = run_dir / "heartbeat.json"
    last_save_time = time.time()
    last_checkpoint: Path | None = resumed_from
    epoch_adapters: list[Path] = []
    steps_done = 0
    step_times: list[float] = []
    runtime.reset_peak_vram()

    def current_rng() -> dict[str, Any]:
        return runtime.rng_state()

    def do_save(reason: str, *, extra_state: dict[str, Any] | None = None) -> Path:
        nonlocal last_save_time, last_checkpoint
        # 같은 step의 완결 checkpoint가 이미 있으면 다시 쓰지 않는다.
        # (덮어쓰는 동안 잠깐 그 checkpoint가 사라지는 창을 만들지 않기 위해서다.)
        # extra_state를 남겨야 하는 저장은 예외다. 같은 step이라도 표시가 들어간 state를
        # 새로 써야 정리 단계가 그 checkpoint를 epoch 경계로 알아볼 수 있다.
        if (
            extra_state is None
            and last_checkpoint is not None
            and ckpt_mod.parse_step(last_checkpoint) == state.global_step
        ):
            already, _ = ckpt_mod.verify_checkpoint(last_checkpoint)
            if already:
                print(f"[train] checkpoint 생략 ({reason}): step {state.global_step}은 이미 저장됨", flush=True)
                return last_checkpoint
        path = ckpt_mod.save_checkpoint(
            run_dir=run_dir,
            scratch_dir=scratch_dir,
            state=state,
            adapter_saver=backend.save_adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            keep_last=keep_last,
            rng=current_rng(),
            extra_state=extra_state,
        )
        last_save_time = time.time()
        last_checkpoint = path
        print(f"[train] checkpoint 저장 ({reason}): {path}", flush=True)
        return path

    def write_heartbeat(seconds_per_step: float | None) -> None:
        remaining = max(total_steps - state.global_step, 0)
        eta = remaining * seconds_per_step if seconds_per_step else None
        runtime.atomic_write_json(
            heartbeat_path,
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "updated": runtime.now_iso(),
                "epoch": state.epoch,
                "step": state.global_step,
                "total_steps": total_steps,
                "samples_consumed_in_epoch": state.samples_consumed_in_epoch,
                "samples_per_epoch": num_samples,
                "seconds_per_step": seconds_per_step,
                "eta_seconds": eta,
                "eta_human": runtime.format_seconds(eta),
                "loss_ema": state.loss_ema,
                "last_checkpoint_step": ckpt_mod.parse_step(last_checkpoint) if last_checkpoint else None,
                "peak_vram_gb": runtime.peak_vram_gb(),
            },
        )

    reason = "completed"
    finished = True
    write_heartbeat(None)

    try:
        for epoch in range(state.epoch, epochs):
            if state.epoch != epoch:
                state.epoch = epoch
                state.samples_consumed_in_epoch = 0
            order = epoch_order(ids, seed, epoch)
            position = int(state.samples_consumed_in_epoch)
            if position >= num_samples:
                # 이 epoch은 이미 다 소비했다.
                state.epoch = epoch + 1
                state.samples_consumed_in_epoch = 0
                continue

            micro_in_accum = 0
            samples_in_step = 0
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_start = time.time()

            while position < num_samples:
                chunk = order[position : position + micro_batch]
                batch = backend.make_micro_batch(chunk, epoch)
                loss = backend.compute_loss(batch)
                scaled = loss / grad_accum
                if not bool(torch.isfinite(scaled)):
                    raise RuntimeError(f"non-finite loss at step {state.global_step}: {float(loss)}")
                scaled.backward()
                accum_loss += float(loss.detach())
                position += len(chunk)
                micro_in_accum += 1
                samples_in_step += len(chunk)

                at_epoch_end = position >= num_samples
                if micro_in_accum < grad_accum and not at_epoch_end:
                    continue

                # ---- optimizer step 경계 ----
                params = backend.trainable_parameters()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                mean_loss = accum_loss / max(micro_in_accum, 1)
                state.global_step += 1
                state.samples_consumed_in_epoch = position
                state.total_samples_seen += samples_in_step
                state.loss_ema = mean_loss if state.loss_ema is None else 0.98 * state.loss_ema + 0.02 * mean_loss
                steps_done += 1
                elapsed = time.time() - step_start
                step_times.append(elapsed)
                if len(step_times) > 50:
                    step_times.pop(0)
                seconds_per_step = sum(step_times) / len(step_times)
                step_start = time.time()
                accum_loss = 0.0
                micro_in_accum = 0
                samples_in_step = 0

                if state.global_step % log_every == 0:
                    runtime.append_jsonl(
                        log_path,
                        {
                            "ts": runtime.now_iso(),
                            "run_id": run_id,
                            "epoch": epoch,
                            "step": state.global_step,
                            "total_steps": total_steps,
                            "loss": mean_loss,
                            "loss_ema": state.loss_ema,
                            "lr": float(scheduler.get_last_lr()[0]),
                            "samples_consumed_in_epoch": state.samples_consumed_in_epoch,
                            "seconds_per_step": seconds_per_step,
                            "peak_vram_gb": runtime.peak_vram_gb(),
                        },
                    )
                if state.global_step % heartbeat_every == 0:
                    write_heartbeat(seconds_per_step)

                due_steps = save_every_steps > 0 and state.global_step % save_every_steps == 0
                due_time = save_every_minutes > 0 and (time.time() - last_save_time) >= save_every_minutes * 60
                if due_steps or due_time:
                    do_save("steps" if due_steps else "minutes")

                if keep_epoch_adapters and at_epoch_end:
                    # 순서가 중요하다. adapter를 먼저 쓰고 표시가 붙은 checkpoint를 나중에 쓴다.
                    # 그래야 둘 사이에서 죽었을 때 재개가 이 epoch의 마지막 step을 다시 밟아서
                    # adapter를 다시 쓴다.
                    finished_epoch = epoch + 1
                    saved = ckpt_mod.save_epoch_adapter(
                        run_dir=run_dir,
                        scratch_dir=scratch_dir,
                        adapter_saver=backend.save_adapter,
                        state=state,
                        epoch=finished_epoch,
                    )
                    epoch_adapters.append(saved)
                    print(f"[train] epoch {finished_epoch} adapter: {saved}", flush=True)
                    do_save(
                        f"epoch_end:{finished_epoch}",
                        extra_state={ckpt_mod.EPOCH_BOUNDARY_KEY: finished_epoch},
                    )

                if stop.requested:
                    do_save(f"signal:{stop.signal_name}")
                    write_heartbeat(seconds_per_step)
                    return TrainOutcome(
                        state,
                        False,
                        f"stopped:{stop.signal_name}",
                        last_checkpoint,
                        steps_done,
                        optimizer,
                        scheduler,
                        epoch_adapters=epoch_adapters,
                    )

                if max_steps is not None and state.global_step >= max_steps:
                    do_save("max_steps")
                    write_heartbeat(seconds_per_step)
                    return TrainOutcome(
                        state,
                        False,
                        "max_steps",
                        last_checkpoint,
                        steps_done,
                        optimizer,
                        scheduler,
                        epoch_adapters=epoch_adapters,
                    )

            state.epoch = epoch + 1
            state.samples_consumed_in_epoch = 0

    except KeyboardInterrupt:
        do_save("KeyboardInterrupt")
        write_heartbeat(None)
        return TrainOutcome(
            state,
            False,
            "KeyboardInterrupt",
            last_checkpoint,
            steps_done,
            optimizer,
            scheduler,
            epoch_adapters=epoch_adapters,
        )
    finally:
        if stop_flag is None:
            stop.restore()

    last_checkpoint = do_save("final")
    final_adapter: Path | None = None
    if save_final:
        # processor는 step checkpoint가 아니라 여기서 한 번만 남긴다.
        final_adapter = ckpt_mod.save_final_adapter(
            run_dir=run_dir,
            scratch_dir=scratch_dir,
            adapter_saver=backend.save_adapter,
            state=state,
            processor_saver=getattr(backend, "save_processor", None),
        )
    write_heartbeat(sum(step_times) / len(step_times) if step_times else None)
    return TrainOutcome(
        state,
        finished,
        reason,
        last_checkpoint,
        steps_done,
        optimizer,
        scheduler,
        final_adapter,
        epoch_adapters,
    )


# --------------------------------------------------------------------------
# 실제 VLM backend
# --------------------------------------------------------------------------


@dataclass
class VLMBackend:
    """Qwen 계열 VLM + LoRA backend."""

    cfg: dict[str, Any]
    data_root: Path
    records: dict[str, Any]
    ids: list[str]
    processor: Any = None
    model: Any = None
    collator: Any = None
    spec: Any = None
    targets: Any = None
    device: str = "cpu"
    #: "full_vocab"(기본) 또는 "choice4". build()에서 설정을 보고 정한다.
    loss_mode: str = config_mod.DEFAULT_LOSS_MODE

    def build(self) -> "VLMBackend":
        import torch

        from . import collator as collator_mod
        from . import modeling
        from . import prompt as prompt_mod

        self.spec = prompt_mod.get_prompt(str(config_mod.get_path(self.cfg, "prompt_id")))
        self.loss_mode = str(
            config_mod.get_path(self.cfg, "train.loss", config_mod.DEFAULT_LOSS_MODE)
            or config_mod.DEFAULT_LOSS_MODE
        )
        if self.loss_mode not in config_mod.LOSS_MODES:
            raise ValueError(f"train.loss를 알 수 없습니다: {self.loss_mode!r}")
        self.processor = modeling.load_processor(self.cfg)
        base, loader_name = modeling.load_base_model(self.cfg, for_training=True)
        self.model, self.targets = modeling.build_lora_model(base, self.cfg)
        self.loader_name = loader_name

        bad = modeling.verify_no_vision_adapters(self.model)
        if bad:
            raise RuntimeError(f"vision 경로에 LoRA가 붙었습니다: {bad[:5]}")

        first = self.records[self.ids[0]]
        image = collator_mod.load_image(first.path, self.data_root)
        tail = prompt_mod.full_generation_prompt(self.processor, self.spec, image)
        self.choice_tokens = prompt_mod.resolve_choice_tokens(self.processor, tail)
        self.collator = collator_mod.AnswerOnlyCollator(
            processor=self.processor, spec=self.spec, choice_tokens=self.choice_tokens
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.train()
        return self

    # --- TrainingBackend ---

    def sample_ids(self) -> list[str]:
        return list(self.ids)

    def make_micro_batch(self, ids: Sequence[str], epoch: int) -> Any:
        from . import collator as collator_mod

        seed = int(config_mod.get_path(self.cfg, "train.seed", 42))
        shuffle = bool(config_mod.get_path(self.cfg, "train.shuffle_options", True))
        views = [
            collator_mod.build_sample_view(
                self.records[sid],
                data_root=self.data_root,
                seed=seed,
                epoch=epoch,
                shuffle_options=shuffle,
                with_gold=True,
            )
            for sid in ids
        ]
        return self.collator(views)

    def compute_loss(self, batch: Any) -> Any:
        import torch

        moved = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        choice4_labels = None
        if self.loss_mode == "choice4":
            # labels를 빼고 forward한다. 손실을 감독 위치의 4개 logits에서 직접 만들 것이므로
            # 모델에게 vocab 15만 개짜리 CE를 한 번 더 시킬 까닭이 없다.
            choice4_labels = moved.pop("labels")
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(**moved, use_cache=False)
        else:
            out = self.model(**moved, use_cache=False)
        if choice4_labels is not None:
            return choice4_loss(out.logits, choice4_labels, self.choice_tokens.ids)
        return out.loss

    def trainable_parameters(self) -> list[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_adapter(self, dest: Path) -> None:
        self.model.save_pretrained(str(dest))

    def load_adapter(self, src: Path) -> None:
        from peft import load_peft_weights, set_peft_model_state_dict

        set_peft_model_state_dict(self.model, load_peft_weights(str(src)))

    def save_processor(self, dest: Path) -> None:
        """재현용 processor 사본. 추론은 이 파일을 읽지 않는다."""
        self.processor.save_pretrained(str(dest))


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def load_split_ids(split_path: str | None, records: Sequence[Any]) -> tuple[list[str], str]:
    """split 파일에서 train_ids를 읽고 split_hash를 만든다."""
    if not split_path:
        ids = [r.id for r in records]
        return ids, runtime.sha256_json({"source": "all", "ids": sorted(ids)})
    payload = runtime.read_json(split_path)
    if isinstance(payload, dict) and "train_ids" in payload:
        ids = [str(i) for i in payload["train_ids"]]
    elif isinstance(payload, list):
        ids = [str(i) for i in payload]
    else:
        raise ValueError(f"{split_path}: train_ids를 찾을 수 없습니다.")
    return ids, runtime.sha256_json({"source": str(split_path), "ids": sorted(ids)})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vqa.train", description="answer-only LoRA 학습")
    config_mod.add_config_arguments(parser)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default=None, help="artifacts/data/split_v1.json")
    parser.add_argument("--csv", default=None, help="학습 CSV. 기본값은 DATA_ROOT/train.csv")
    parser.add_argument("--resume", default="auto", help="auto | none | <checkpoint 경로>")
    parser.add_argument("--force-resume", action="store_true", help="config/split 해시가 달라도 재개")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.config_from_args(args)

    # 모델을 올리기 전에 한 번만 적용한다. TF32는 fp32 matmul 경로의 속도를 좌우한다.
    tf32_info = runtime.apply_tf32(bool(config_mod.get_path(cfg, "runtime.tf32", True)))
    print(runtime.tf32_line(tf32_info))

    data_root = runtime.data_root(args.data_root)
    run_dir = runtime.run_dir(args.run_id, args.run_root)
    scratch = runtime.ensure_dir(runtime.local_scratch(args.local_scratch) / args.run_id)
    runtime.ensure_dir(run_dir)

    if ckpt_mod.training_is_finished(run_dir):
        print(f"[train] 이미 끝난 run입니다: {ckpt_mod.final_adapter_dir(run_dir)}")
        return 0

    from . import collator as collator_mod

    csv_path = Path(args.csv) if args.csv else data_root / "train.csv"
    records = collator_mod.load_records(csv_path, data_root=data_root)
    train_ids, split_hash = load_split_ids(args.split, records)

    by_id = {r.id: r for r in records}
    missing = [i for i in train_ids if i not in by_id]
    if missing:
        raise SystemExit(f"split에 있는 id {len(missing)}개가 CSV에 없습니다: {missing[:5]}")
    train_ids = [i for i in train_ids if by_id[i].answer_idx is not None]
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]

    cfg_hash = config_mod.config_hash(cfg)
    backend = VLMBackend(cfg=cfg, data_root=data_root, records=by_id, ids=train_ids).build()

    manifest = {
        "run_id": args.run_id,
        "created": runtime.now_iso(),
        "command": " ".join([sys.executable, "-m", "vqa.train", *(argv or sys.argv[1:])]),
        "config": cfg,
        "config_hash": cfg_hash,
        "split": str(args.split) if args.split else None,
        "split_hash": split_hash,
        "num_train": len(train_ids),
        "model_source": config_mod.model_source(cfg),
        "loader_class": getattr(backend, "loader_name", None),
        "prompt_id": config_mod.get_path(cfg, "prompt_id"),
        "prompt_hash": backend.spec.hash(),
        # 손실 방식과 그때 쓴 a/b/c/d 토큰 id를 함께 남긴다. choice4 로 학습한 run 인지
        # 나중에 manifest 만 보고 알 수 있어야 하기 때문이다.
        "loss": backend.loss_mode,
        "choice_tokens": backend.choice_tokens.as_dict(),
        "lora_targets": backend.targets.as_dict(),
        "parameters": _count_params(backend.model),
        "environment": runtime.environment_report(
            {"data_root": data_root, "run_root": run_dir, "scratch": scratch}
        ),
        "tf32": tf32_info,
    }
    runtime.atomic_write_json(run_dir / "run_manifest.json", manifest)
    print(
        f"[train] LoRA target {manifest['lora_targets']['num_modules']}개 "
        f"(leaf {manifest['lora_targets']['leaf_names']}), "
        f"trainable={manifest['parameters']['trainable']:,}"
    )

    try:
        outcome = run_training(
            backend,
            cfg,
            run_dir=run_dir,
            scratch_dir=scratch,
            run_id=args.run_id,
            config_hash=cfg_hash,
            split_hash=split_hash,
            resume=args.resume,
            force_resume=args.force_resume,
            max_steps=args.max_steps,
        )
    except ckpt_mod.ResumeRefused as exc:
        print(f"[train] 재개 거부: {exc}")
        return 2

    saved_epoch_adapters = ckpt_mod.list_epoch_adapters(run_dir)
    if saved_epoch_adapters:
        manifest["epoch_adapters"] = [str(path) for path in saved_epoch_adapters]
        runtime.atomic_write_json(run_dir / "run_manifest.json", manifest)
        for path in saved_epoch_adapters:
            print(f"[train] epoch adapter: {path}")

    if outcome.final_adapter is not None:
        processor_dir = ckpt_mod.final_processor_dir(run_dir)
        manifest["final_adapter"] = str(outcome.final_adapter)
        manifest["final_processor"] = str(processor_dir) if processor_dir.exists() else None
        runtime.atomic_write_json(run_dir / "run_manifest.json", manifest)
        print(f"[train] final_adapter: {outcome.final_adapter}")
        if manifest["final_processor"]:
            print(f"[train] processor 사본(재현용): {processor_dir}")

    print(f"[train] 종료: {outcome.reason} step={outcome.state.global_step} loss_ema={outcome.state.loss_ema}")
    return 0


def _count_params(model: Any) -> dict[str, int]:
    from . import modeling

    return modeling.count_parameters(model)


if __name__ == "__main__":
    raise SystemExit(main())
