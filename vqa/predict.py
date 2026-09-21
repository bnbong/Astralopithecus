"""logits 기반 4지선다 추론 CLI.

생성을 하지 않는다. 한 번의 forward에서 마지막 non-pad 위치의 logits를 읽고 a/b/c/d
네 토큰만 골라 softmax한다. 학습의 전체 vocabulary CE와는 다른 계산이라는 점을 분명히
구분한다.

사용 예:
    python -m vqa.predict --config configs/lab_qwen25vl_3b.yaml --run-id lab3b_v1 \\
        --csv test.csv --tta 4 --out-name test_ft
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import collator as collator_mod
from . import config as config_mod
from . import predictions as pred_mod
from . import prompt as prompt_mod
from . import runtime
from .collator import PERM_BANK, VQARecord
from .prompt import LETTERS


# --------------------------------------------------------------------------
# 대상 선택
# --------------------------------------------------------------------------


def select_records(
    records: Sequence[VQARecord],
    *,
    ids_file: str | None = None,
    shard: str | None = None,
    limit: int | None = None,
) -> list[VQARecord]:
    """ids-file / shard / limit을 적용한다. shard는 CSV 행 번호 mod n으로 나눈다."""
    selected = list(records)
    if ids_file:
        wanted = set(read_id_list(ids_file))
        selected = [r for r in selected if r.id in wanted]
    if shard:
        index, count = parse_shard(shard)
        # 행 번호는 원본 CSV 순서를 쓴다. 필터 뒤 순서를 쓰면 분할이 불안정해진다.
        row_of = {r.id: i for i, r in enumerate(records)}
        selected = [r for r in selected if row_of[r.id] % count == index]
    if limit:
        selected = selected[:limit]
    return selected


def parse_shard(text: str) -> tuple[int, int]:
    """`i/n` 문자열을 (i, n)으로 바꾼다. i는 0부터 센다."""
    if "/" not in text:
        raise ValueError(f"--shard 형식은 i/n 입니다: {text!r}")
    left, right = text.split("/", 1)
    index, count = int(left), int(right)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard 범위가 잘못되었습니다: {text!r}")
    return index, count


def read_id_list(path: str | Path) -> list[str]:
    try:
        return pred_mod.read_id_list(path)
    except Exception:
        text = Path(path).read_text(encoding="utf-8-sig")
        return [line.strip() for line in text.splitlines() if line.strip()]


def output_path(
    preds_dir: Path,
    out_name: str,
    *,
    shard: str | None = None,
    perm_index: int | None = None,
    base: bool = False,
) -> Path:
    stem = out_name[:-4] if out_name.endswith(".csv") else out_name
    if base:
        stem += ".base"
    if perm_index is not None:
        stem += f".perm{perm_index}"
    if shard:
        index, count = parse_shard(shard)
        stem += f".shard{index}of{count}"
    return preds_dir / f"{stem}.csv"


# --------------------------------------------------------------------------
# 모델 래퍼
# --------------------------------------------------------------------------


@dataclass
class Scorer:
    """processor + model + choice token id를 묶은 추론기."""

    cfg: dict[str, Any]
    data_root: Path
    processor: Any = None
    model: Any = None
    spec: Any = None
    choice_tokens: Any = None
    device: str = "cpu"
    adapter_path: str | None = None
    loader_name: str | None = None

    def build(self, sample_record: VQARecord) -> "Scorer":
        import torch

        from . import modeling

        # processor는 언제나 base 모델 + config에서 만든다. run의
        # final_adapter/processor는 재현용 기록일 뿐이라 추론 경로에서 읽지 않는다.
        self.spec = prompt_mod.get_prompt(str(config_mod.get_path(self.cfg, "prompt_id")))
        self.processor = modeling.load_processor(self.cfg)
        base, loader_name = modeling.load_base_model(self.cfg, for_training=False)
        self.loader_name = loader_name
        if self.adapter_path:
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, self.adapter_path, is_trainable=False)
        self.model = base
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        image = collator_mod.load_image(sample_record.path, self.data_root)
        tail = prompt_mod.full_generation_prompt(self.processor, self.spec, image)
        self.choice_tokens = prompt_mod.resolve_choice_tokens(self.processor, tail)
        return self

    @property
    def has_adapter(self) -> bool:
        return bool(self.adapter_path)

    def encode(self, records: Sequence[VQARecord], perm: Sequence[int]):
        texts = []
        images = []
        for record in records:
            shown, _ = collator_mod.apply_permutation(record.options, record.answer_idx, perm)
            image = collator_mod.load_image(record.path, self.data_root)
            texts.append(
                prompt_mod.generation_text(
                    self.processor, self.spec, image, record.question, shown, record.ocr
                )
            )
            images.append(image)
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")

    def score(self, enc, *, also_base: bool = False):
        """(shown 순서 확률, base 확률 또는 None)."""
        import torch

        moved = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        with torch.inference_mode():
            main = self._forward_probs(moved)
            base = None
            if also_base and self.has_adapter:
                with self.model.disable_adapter():
                    base = self._forward_probs(moved)
        return main, base

    def _forward_probs(self, moved):
        import torch

        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(**moved, use_cache=False)
        else:
            out = self.model(**moved, use_cache=False)
        logits = out.logits
        index = collator_mod.last_nonpad_index(moved["attention_mask"]).to(logits.device)
        batch_index = torch.arange(logits.shape[0], device=logits.device)
        choice_ids = torch.tensor(self.choice_tokens.ids, dtype=torch.long, device=logits.device)
        picked = logits[batch_index, index][:, choice_ids]
        return torch.softmax(picked.float(), dim=-1).cpu().numpy()


# --------------------------------------------------------------------------
# 한 permutation 실행
# --------------------------------------------------------------------------


def run_permutation(
    scorer: Scorer,
    records: Sequence[VQARecord],
    perm: Sequence[int],
    *,
    out_file: Path,
    base_file: Path | None,
    run_id: str,
    batch_size: int,
    save_every_items: int,
    also_base: bool,
) -> dict[str, Any]:
    """한 순열에 대해 남은 id만 추론하고 부분 저장한다."""
    done, report = pred_mod.load_partial_predictions(out_file)
    done_ids = set(done["id"].tolist()) if len(done) else set()
    if done_ids:
        print(f"[predict] {out_file.name}: 이미 끝난 {len(done_ids)}건을 건너뜁니다. {report.describe()}")

    todo = [r for r in records if r.id not in done_ids]
    # 길이가 비슷한 것끼리 묶여 padding 낭비를 줄인다. 출력은 id 기준이므로 순서는 자유롭다.
    todo.sort(key=lambda r: r.approx_length())

    buffer: list[dict[str, Any]] = []
    base_buffer: list[dict[str, Any]] = []
    started = time.time()
    processed = 0

    def flush() -> None:
        if buffer:
            pred_mod.append_predictions(buffer, out_file)
            buffer.clear()
        if base_buffer and base_file is not None:
            pred_mod.append_predictions(base_buffer, base_file)
            base_buffer.clear()

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        enc = scorer.encode(chunk, perm)
        shown, base_shown = scorer.score(enc, also_base=also_base)
        original = collator_mod.invert_probs(shown, perm)
        for i, record in enumerate(chunk):
            buffer.append(_row(record.id, original[i], run_id))
        if base_shown is not None:
            base_original = collator_mod.invert_probs(base_shown, perm)
            for i, record in enumerate(chunk):
                base_buffer.append(_row(record.id, base_original[i], run_id + "+base"))
        processed += len(chunk)
        if len(buffer) >= save_every_items:
            flush()
    flush()

    elapsed = time.time() - started
    return {
        "processed": processed,
        "skipped": len(done_ids),
        "seconds": elapsed,
        "seconds_per_item": (elapsed / processed) if processed else None,
    }


def _row(row_id: str, probs, run_id: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "p_a": float(probs[0]),
        "p_b": float(probs[1]),
        "p_c": float(probs[2]),
        "p_d": float(probs[3]),
        "run_id": run_id,
    }


def average_perm_files(files: Sequence[Path], order_ids: Sequence[str], run_id: str):
    """permutation별 확률 파일을 id 기준으로 평균낸다."""
    import numpy as np
    import pandas as pd

    frames = []
    for path in files:
        frame, _ = pred_mod.load_partial_predictions(path)
        if len(frame) == 0:
            raise ValueError(f"{path}: 비어 있어 평균을 낼 수 없습니다.")
        frames.append(frame.set_index("id"))

    common = set(frames[0].index)
    for frame in frames[1:]:
        common &= set(frame.index)
    missing = [i for i in order_ids if i not in common]
    if missing:
        raise ValueError(f"모든 permutation에 없는 id가 {len(missing)}개 있습니다: {missing[:5]}")

    ids = [i for i in order_ids if i in common]
    stacked = np.stack(
        [frame.loc[ids, list(pred_mod.PROB_COLUMNS)].to_numpy(dtype=float) for frame in frames], axis=0
    )
    mean = stacked.mean(axis=0)
    out = pd.DataFrame({"id": ids})
    for col_index, col in enumerate(pred_mod.PROB_COLUMNS):
        out[col] = mean[:, col_index]
    out["run_id"] = run_id
    return out


def frame_from_file(path: Path, order_ids: Sequence[str], run_id: str):
    import pandas as pd

    frame, _ = pred_mod.load_partial_predictions(path)
    frame = frame.set_index("id")
    ids = [i for i in order_ids if i in frame.index]
    out = pd.DataFrame({"id": ids})
    for col in pred_mod.PROB_COLUMNS:
        out[col] = frame.loc[ids, col].to_numpy(dtype=float)
    out["run_id"] = run_id
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vqa.predict", description="logits 기반 4지선다 추론")
    config_mod.add_config_arguments(parser)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--csv", required=True, help="test.csv | train.csv | dev.csv (DATA_ROOT 기준 상대 경로 가능)")
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--adapter", default=None, help="adapter 경로 또는 none. 기본값은 run의 final_adapter")
    parser.add_argument("--also-base", action="store_true", help="같은 forward에서 base 확률도 함께 계산")
    parser.add_argument("--shard", default=None, metavar="i/n")
    parser.add_argument("--tta", type=int, default=1, choices=[1, 4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-name", required=True)
    return parser


def resolve_adapter(args: argparse.Namespace, run_dir: Path) -> str | None:
    if args.adapter is None:
        from . import checkpoint as ckpt_mod

        final = ckpt_mod.final_adapter_dir(run_dir)
        if final.exists():
            return str(final)
        print("[predict] adapter를 찾지 못해 base 모델로 추론합니다.")
        return None
    if str(args.adapter).lower() == "none":
        return None
    return str(args.adapter)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.config_from_args(args)

    data_root = runtime.data_root(args.data_root)
    run_dir = runtime.run_dir(args.run_id, args.run_root)
    preds_dir = runtime.ensure_dir(run_dir / "preds")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = data_root / args.csv
    all_records = collator_mod.load_records(csv_path, data_root=data_root)
    records = select_records(all_records, ids_file=args.ids_file, shard=args.shard, limit=args.limit)
    if not records:
        raise SystemExit("추론 대상이 없습니다.")
    print(f"[predict] {csv_path.name}: 전체 {len(all_records)} 중 {len(records)}건을 처리합니다.")

    adapter = resolve_adapter(args, run_dir)
    scorer = Scorer(cfg=cfg, data_root=data_root, adapter_path=adapter).build(records[0])

    perms = list(PERM_BANK[: args.tta]) if args.tta > 1 else [tuple(range(4))]
    batch_size = int(config_mod.get_path(cfg, "infer.batch_size", 1))
    save_every = int(config_mod.get_path(cfg, "infer.save_every_items", 50))
    order_ids = [r.id for r in records]

    stats = []
    perm_files: list[Path] = []
    base_files: list[Path] = []
    for k, perm in enumerate(perms):
        perm_index = k if args.tta > 1 else None
        out_file = output_path(preds_dir, args.out_name, shard=args.shard, perm_index=perm_index)
        base_file = (
            output_path(preds_dir, args.out_name, shard=args.shard, perm_index=perm_index, base=True)
            if args.also_base and scorer.has_adapter
            else None
        )
        perm_files.append(out_file)
        if base_file is not None:
            base_files.append(base_file)
        info = run_permutation(
            scorer,
            records,
            perm,
            out_file=out_file,
            base_file=base_file,
            run_id=args.run_id,
            batch_size=batch_size,
            save_every_items=save_every,
            also_base=args.also_base,
        )
        info["perm"] = list(perm)
        stats.append(info)
        print(f"[predict] perm{k} {tuple(perm)}: {info}")

    final_file = output_path(preds_dir, args.out_name, shard=args.shard)
    if args.tta > 1:
        frame = average_perm_files(perm_files, order_ids, args.run_id)
    else:
        frame = frame_from_file(perm_files[0], order_ids, args.run_id)

    manifest = pred_mod.build_manifest(
        run_id=args.run_id,
        model_id=config_mod.get_path(cfg, "model_id"),
        model_revision=config_mod.get_path(cfg, "revision"),
        adapter=adapter or "none",
        precision=config_mod.get_path(cfg, "precision"),
        prompt_id=config_mod.get_path(cfg, "prompt_id"),
        prompt_hash=scorer.spec.hash(),
        max_pixels=config_mod.get_path(cfg, "max_pixels"),
        shard_index=parse_shard(args.shard)[0] if args.shard else None,
        shard_count=parse_shard(args.shard)[1] if args.shard else None,
        seconds_per_item=stats[0].get("seconds_per_item") if stats else None,
        extra={
            "tta": args.tta,
            "permutations": [list(p) for p in perms],
            "perm_files": [p.name for p in perm_files],
            "base_files": [p.name for p in base_files],
            "csv": str(csv_path),
            "ids_file": args.ids_file,
            "num_items": len(records),
            "choice_token_ids": scorer.choice_tokens.ids,
            "choice_token_source": scorer.choice_tokens.source,
            "loader_class": scorer.loader_name,
            "per_perm_stats": stats,
            "command": " ".join([sys.executable, "-m", "vqa.predict", *(argv or sys.argv[1:])]),
        },
    )
    pred_mod.write_predictions(frame, final_file, manifest=manifest)
    print(f"[predict] 완료: {final_file} ({len(frame)}행)")

    if base_files:
        base_final = output_path(preds_dir, args.out_name, shard=args.shard, base=True)
        base_frame = (
            average_perm_files(base_files, order_ids, args.run_id + "+base")
            if len(base_files) > 1
            else frame_from_file(base_files[0], order_ids, args.run_id + "+base")
        )
        base_manifest = dict(manifest)
        base_manifest["adapter"] = "none"
        base_manifest["run_id"] = args.run_id + "+base"
        pred_mod.write_predictions(base_frame, base_final, manifest=base_manifest)
        print(f"[predict] base 확률: {base_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
