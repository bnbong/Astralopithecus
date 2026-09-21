"""아주 작은 진짜 VLM으로 train -> resume -> predict 전체 경로를 CPU에서 돌린다.

`hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration`은 수 MB짜리 무작위
가중치 모델이지만 모듈 구조(model.visual / model.language_model)는 실제 Qwen과 같아서
로더 fallback, LoRA target 탐지, label 마스킹, checkpoint 저장/재개, logits scoring을
한 번에 확인할 수 있다. 성능이나 수치 품질은 확인하지 않는다.

네트워크가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("peft")

from vqa import checkpoint as ckpt_mod  # noqa: E402
from vqa import config as config_mod  # noqa: E402
from vqa import predict as predict_mod  # noqa: E402
from vqa import predictions as pred_mod  # noqa: E402
from vqa import train as train_mod  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lab_qwen25vl_3b.yaml"

N_ROWS = 8


@pytest.fixture(scope="module")
def tiny_available():
    from transformers import AutoProcessor

    try:
        AutoProcessor.from_pretrained(TINY_MODEL)
    except Exception as exc:
        pytest.skip(f"tiny 모델을 받을 수 없습니다: {exc}")
    return True


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    from PIL import Image

    root = tmp_path_factory.mktemp("data")
    (root / "train").mkdir()
    rows = ["id,path,question,a,b,c,d,answer"]
    for i in range(1, N_ROWS + 1):
        name = f"train_{i:04d}.jpg"
        Image.new("RGB", (64, 64), (i * 20 % 255, 80, 160)).save(root / "train" / name)
        answer = "abcd"[i % 4]
        rows.append(f"{name},train/{name},질문 {i} 입니다,보기A{i},보기B{i},보기C{i},보기D{i},{answer}")
    (root / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def tiny_cfg():
    cfg = config_mod.load_config(
        CONFIG,
        [
            f"model_id={TINY_MODEL}",
            "precision=bf16",
            "min_pixels=3136",  # 4 * 28 * 28
            "max_pixels=12544",  # 16 * 28 * 28
            "train.lr=0.01",
            "train.epochs=1",
            "train.micro_batch=1",
            "train.grad_accum=2",
            "train.save_every_steps=2",
            "train.save_every_minutes=0",
            "train.keep_last=2",
            "infer.batch_size=2",
            "infer.save_every_items=2",
        ],
    )
    return cfg


def modeling_leafs():
    from vqa import modeling

    return set(modeling.LORA_LEAF_CANDIDATES)


def build_backend(cfg, dataset):
    from vqa import collator as collator_mod

    records = collator_mod.load_records(dataset / "train.csv", data_root=dataset)
    by_id = {r.id: r for r in records}
    ids = [r.id for r in records]
    return train_mod.VLMBackend(cfg=cfg, data_root=dataset, records=by_id, ids=ids).build(), records


def test_end_to_end_train_resume_predict(tiny_available, dataset, tmp_path):
    cfg = tiny_cfg()
    run_dir = tmp_path / "runs" / "tiny"
    scratch = tmp_path / "scratch"

    backend, records = build_backend(cfg, dataset)

    # 1. 로더 fallback과 LoRA target이 language 쪽만 잡혔는지.
    assert backend.loader_name in {
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    }
    targets = backend.targets
    assert targets.names, "LoRA target을 찾지 못했습니다."
    for name in targets.names:
        assert "visual" not in name and "merger" not in name
    # language 루트 밖의 후보 모듈은 하나도 포함되지 않아야 한다.
    assert targets.language_prefix == "model.language_model"
    for name in targets.names:
        assert name.startswith(targets.language_prefix + ".")
    outside = [
        name
        for name, module in backend.model.named_modules()
        if name.rsplit(".", 1)[-1] in modeling_leafs()
        and isinstance(module, torch.nn.Linear)
        and not name.replace("base_model.model.", "").startswith("model.language_model.")
    ]
    for name in outside:
        assert name not in targets.names

    from vqa import modeling

    assert modeling.verify_no_vision_adapters(backend.model) == []
    counts = modeling.count_parameters(backend.model)
    assert 0 < counts["trainable"] < counts["total"]

    # 2. 두 step만 돌고 죽은 상황.
    first = train_mod.run_training(
        backend,
        cfg,
        run_dir=run_dir,
        scratch_dir=scratch,
        run_id="tiny",
        config_hash=config_mod.config_hash(cfg),
        split_hash="split",
        resume="none",
        max_steps=2,
        save_final=False,
    )
    assert first.state.global_step == 2
    chosen, _ = ckpt_mod.find_resume_checkpoint(run_dir)
    assert chosen is not None
    assert (chosen / "adapter" / "adapter_model.safetensors").exists() or (
        chosen / "adapter" / "adapter_model.bin"
    ).exists()

    # 3. 완전히 새 backend로 재개해 끝까지.
    backend2, _ = build_backend(cfg, dataset)
    second = train_mod.run_training(
        backend2,
        cfg,
        run_dir=run_dir,
        scratch_dir=scratch,
        run_id="tiny",
        config_hash=config_mod.config_hash(cfg),
        split_hash="split",
        resume="auto",
        save_final=True,
    )
    assert second.state.global_step == 4  # 8 표본 / (micro 1 * accum 2)
    assert ckpt_mod.training_is_finished(run_dir)

    # 3-1. processor 사본은 final_adapter 안에만 한 번 생긴다. step checkpoint에는 없다.
    processor_dir = ckpt_mod.final_processor_dir(run_dir)
    assert processor_dir.is_dir()
    assert any(processor_dir.iterdir())
    for step_dir in ckpt_mod.list_checkpoints(run_dir):
        assert not (step_dir / "processor").exists()
    # COMPLETE 해시 목록에 들어 있어야 검증이 통과한다.
    ok, reason = ckpt_mod.verify_checkpoint(ckpt_mod.final_adapter_dir(run_dir))
    assert ok, reason

    log_lines = (run_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 4
    assert json.loads(log_lines[-1])["step"] == 4

    # 4. 학습한 adapter로 추론.
    out_dir = run_dir / "preds"
    rc = predict_mod.main(
        [
            "--config",
            str(CONFIG),
            "--set",
            f"model_id={TINY_MODEL}",
            "--set",
            "precision=bf16",
            "--set",
            "min_pixels=3136",
            "--set",
            "max_pixels=12544",
            "--set",
            "infer.batch_size=2",
            "--set",
            "infer.save_every_items=2",
            "--run-id",
            "tiny",
            "--csv",
            str(dataset / "train.csv"),
            "--adapter",
            str(ckpt_mod.final_adapter_dir(run_dir)),
            "--also-base",
            "--out-name",
            "train_ft",
            "--data-root",
            str(dataset),
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 0

    frame = pred_mod.load_predictions(out_dir / "train_ft.csv")
    assert len(frame) == N_ROWS
    assert frame["id"].tolist() == [r.id for r in records]
    probs = frame[list(pred_mod.PROB_COLUMNS)].to_numpy(dtype=float)
    assert abs(probs.sum(axis=1) - 1.0).max() < 1e-3

    base = pred_mod.load_predictions(out_dir / "train_ft.base.csv")
    assert len(base) == N_ROWS
    # 학습한 adapter 확률과 base 확률은 달라야 한다 (adapter가 실제로 적용되었다는 뜻).
    base_probs = base[list(pred_mod.PROB_COLUMNS)].to_numpy(dtype=float)
    assert abs(probs - base_probs).max() > 0

    manifest = pred_mod.read_manifest(out_dir / "train_ft.csv")
    assert manifest["prompt_id"] == "v1_image_only"
    assert manifest["tta"] == 1
    assert manifest["choice_token_ids"]


def test_train_prefix_matches_inference_on_the_real_model_path(tiny_available, dataset):
    """실제 processor + 실제 이미지 토큰 확장에서도 학습/추론 prefix가 같아야 한다."""
    from vqa import collator as collator_mod
    from vqa import preflight as preflight_mod
    from vqa import prompt as prompt_mod

    cfg = tiny_cfg()
    records = collator_mod.load_records(dataset / "train.csv", data_root=dataset)
    from vqa import modeling

    processor = modeling.load_processor(cfg)
    spec = prompt_mod.get_prompt(str(config_mod.get_path(cfg, "prompt_id")))
    image = collator_mod.load_image(records[0].path, dataset)
    tokens = prompt_mod.resolve_choice_tokens(
        processor, prompt_mod.full_generation_prompt(processor, spec, image)
    )

    result = preflight_mod.check_prefix_match(processor, spec, records, dataset, tokens, n=4)
    assert result.status == preflight_mod.PASS, result.as_dict()
    assert result.detail["checked"] == 4
    assert all(r["match"] for r in result.detail["reports"])

    # scoring token id와 감독 token id가 같은지도 같은 경로에서 확인한다.
    collate = collator_mod.AnswerOnlyCollator(processor=processor, spec=spec, choice_tokens=tokens)
    views = [
        collator_mod.build_sample_view(
            r, data_root=dataset, seed=0, epoch=0, shuffle_options=False, with_gold=True
        )
        for r in records[:4]
    ]
    enc = collate(views)
    labels = enc["labels"]
    for b, view in enumerate(views):
        position = int(torch.where(labels[b] != -100)[0][0])
        assert int(labels[b, position]) == tokens.ids["abcd".index(view.gold_letter)]


def test_end_to_end_tta_and_shard(tiny_available, dataset, tmp_path):
    """TTA 4와 shard 분할이 실제 모델 경로에서 파일을 남기는지."""
    run_root = tmp_path / "runs"
    common = [
        "--config",
        str(CONFIG),
        "--set",
        f"model_id={TINY_MODEL}",
        "--set",
        "precision=bf16",
        "--set",
        "min_pixels=3136",
        "--set",
        "max_pixels=12544",
        "--set",
        "infer.batch_size=2",
        "--run-id",
        "tta",
        "--csv",
        str(dataset / "train.csv"),
        "--adapter",
        "none",
        "--data-root",
        str(dataset),
        "--run-root",
        str(run_root),
    ]
    assert predict_mod.main([*common, "--tta", "4", "--out-name", "tta_base"]) == 0
    preds = run_root / "tta" / "preds"
    for k in range(4):
        assert (preds / f"tta_base.perm{k}.csv").exists()
    frame = pred_mod.load_predictions(preds / "tta_base.csv")
    assert len(frame) == N_ROWS

    seen = []
    for i in range(2):
        assert predict_mod.main([*common, "--shard", f"{i}/2", "--out-name", "sharded"]) == 0
        part = pred_mod.load_predictions(preds / f"sharded.shard{i}of2.csv")
        seen.extend(part["id"].tolist())
    assert sorted(seen) == sorted(frame["id"].tolist())


def test_train_cli_writes_manifest_and_exits_zero_when_finished(tiny_available, dataset, tmp_path, capsys):
    """같은 명령을 다시 실행하면 이어서 하고, 이미 끝났으면 0으로 조용히 끝난다."""
    run_root = tmp_path / "runs"
    argv = [
        "--config",
        str(CONFIG),
        "--set",
        f"model_id={TINY_MODEL}",
        "--set",
        "precision=bf16",
        "--set",
        "min_pixels=3136",
        "--set",
        "max_pixels=12544",
        "--set",
        "train.grad_accum=4",
        "--set",
        "train.save_every_minutes=0",
        "--run-id",
        "cli",
        "--csv",
        str(dataset / "train.csv"),
        "--data-root",
        str(dataset),
        "--run-root",
        str(run_root),
        "--local-scratch",
        str(tmp_path / "scratch"),
    ]
    assert train_mod.main(argv) == 0

    run_dir = run_root / "cli"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "cli"
    assert manifest["lora_targets"]["num_modules"] > 0
    assert manifest["lora_targets"]["counts_by_leaf"]
    assert manifest["parameters"]["trainable"] > 0
    assert manifest["choice_tokens"]["ids"]
    assert manifest["prompt_id"] == "v1_image_only"
    assert manifest["config_hash"] and manifest["split_hash"]
    assert ckpt_mod.training_is_finished(run_dir)
    # 학습이 끝나면 manifest가 final_adapter와 processor 사본 경로를 기록한다.
    assert Path(manifest["final_adapter"]) == ckpt_mod.final_adapter_dir(run_dir)
    assert Path(manifest["final_processor"]) == ckpt_mod.final_processor_dir(run_dir)
    assert Path(manifest["final_processor"]).is_dir()

    # 두 번째 실행은 모델을 로드하지도 않고 바로 끝난다.
    assert train_mod.main(argv) == 0
    assert "이미 끝난 run" in capsys.readouterr().out
