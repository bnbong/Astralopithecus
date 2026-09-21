#!/usr/bin/env python3
"""VQA 4지선다 추론 스크립트 (SSAFY AI 챌린지 2회차)

정답을 글로 생성시키지 않고, 프롬프트 직후 첫 토큰에서 a/b/c/d 네 글자의 로짓만 비교한다.
  - 생성보다 빠르고, 확률(prob_a~prob_d)이 함께 나와 나중에 모델 결합에 그대로 쓸 수 있다.
  - 결과 csv 는 eval_val.py 가 바로 읽을 수 있는 형식(id, pred, prob_a~d)이다.

사용 예 (기준 정확도 측정):
    python infer_val.py --csv data/train.csv --split tmp_split_300.csv --data-dir data \
        --model downloads/models/Qwen2.5-VL-3B-Instruct --prompt p1 \
        --out preds_qwen3b_p1_tmp300.csv

먼저 5문항으로 동작 확인:  ... --limit 5 --out preds_try.csv

test 를 5대로 나눠 추론 (2번째 조각):
    python infer_val.py --csv data/test.csv --data-dir data --model <모델 경로> \
        --shard 1/5 --out preds_test_s2of5.csv

중간에 끊겨도 같은 명령을 다시 실행하면 이미 끝난 id 는 건너뛰고 이어서 진행한다.
주의: test.csv 에는 정답이 없다. test 로 학습하거나 튜닝하지 않는다(규칙: Data Leakage 금지).
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

LETTERS = list("abcd")
EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)

# 프롬프트 버전. 새 버전을 추가하면 --prompt 로 고를 수 있다. (바꾼 내용은 노션 실험 로그에 기록)
PROMPTS = {
    "p1": (
        "{question}\n\n"
        "a. {a}\nb. {b}\nc. {c}\nd. {d}\n\n"
        "위 질문의 정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요."
    ),
    "p2": (
        "이미지를 자세히 보고, 이미지에 적힌 글자를 정확히 읽어서 질문에 답하세요.\n\n"
        "질문: {question}\n\n"
        "a. {a}\nb. {b}\nc. {c}\nd. {d}\n\n"
        "정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요."
    ),
}


def norm_id(x):
    return EXT_RE.sub("", str(x).strip())


def load_rows(csv_path, split_path=None, shard=None, limit=None):
    """추론할 행을 고른다: (분할 필터) -> (샤드) -> (limit) 순서."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = ["id", "path", "question"] + LETTERS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[오류] {csv_path} 에 필요한 컬럼이 없습니다: {missing}")

    if split_path:
        sp = pd.read_csv(split_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        col = "id" if "id" in sp.columns else sp.columns[0]
        keep = {norm_id(x) for x in sp[col]}
        df = df[df["id"].map(norm_id).isin(keep)]
        if len(df) != len(keep):
            print(f"[경고] 분할의 id {len(keep)}개 중 {len(df)}개만 {csv_path} 에서 찾았습니다.")

    if shard:
        k, n = shard
        lo, hi = len(df) * k // n, len(df) * (k + 1) // n   # 연속 구간으로 나눔
        df = df.iloc[lo:hi]
    if limit:
        df = df.iloc[:limit]
    return df.reset_index(drop=True)


def parse_shard(text):
    """'1/5' -> (0, 5)  (사람 기준 1번째 조각 = 인덱스 0)"""
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if not m or not (1 <= int(m.group(1)) <= int(m.group(2))):
        raise argparse.ArgumentTypeError("--shard 는 '2/5' 처럼 (몇 번째)/(전체 개수) 형식이어야 합니다.")
    return int(m.group(1)) - 1, int(m.group(2))


class Scorer:
    """모델을 올리고, 문항 묶음에 대해 a~d 확률을 계산한다."""

    def __init__(self, model_path, max_pixels, min_pixels, load_4bit, prompt):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.template = PROMPTS[prompt]

        kwargs = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

        try:
            from transformers import AutoModelForImageTextToText as ModelCls
        except ImportError:  # 구버전 transformers
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        self.model = ModelCls.from_pretrained(model_path, device_map="auto", **kwargs).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"   # 배치 시 마지막 위치를 맞추기 위해

        # a~d 각각이 토큰 1개인지 확인
        self.letter_ids = []
        for c in LETTERS:
            ids = self.processor.tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"[오류] '{c}' 가 토큰 1개가 아닙니다: {ids}. 토크나이저를 확인하세요.")
            self.letter_ids.append(ids[0])

    def score(self, rows, images):
        """rows: 딕셔너리 리스트, images: PIL 이미지 리스트 -> (n, 4) 확률 (numpy)"""
        torch = self.torch
        texts = []
        for r in rows:
            prompt = self.template.format(
                question=r["question"], a=r["a"], b=r["b"], c=r["c"], d=r["d"])
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]      # 다음 토큰 분포
        four = logits[:, self.letter_ids].float()               # a,b,c,d 만 비교
        return torch.softmax(four, dim=-1).cpu().numpy()


def run(df, scorer, out_path, data_dir, batch_size, log_every=20):
    """행을 순회하며 결과를 out_path 에 이어 쓴다. 이미 있는 id 는 건너뛴다."""
    from PIL import Image

    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype=str, encoding="utf-8-sig")
        done = set(prev["id"])
        print(f"[이어하기] 이미 완료된 {len(done)}개는 건너뜁니다.")
    todo = df[~df["id"].isin(done)]
    if todo.empty:
        print("추론할 문항이 없습니다 (모두 완료).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_header = not os.path.exists(out_path)
    start, n_done = time.time(), 0

    for i in range(0, len(todo), batch_size):
        chunk = todo.iloc[i:i + batch_size]
        rows = chunk.to_dict("records")
        images = []
        for r in rows:
            p = os.path.join(data_dir, r["path"])
            if not os.path.exists(p):
                raise SystemExit(f"[오류] 이미지가 없습니다: {p} (id={r['id']}). --data-dir 를 확인하세요.")
            images.append(Image.open(p).convert("RGB"))

        probs = scorer.score(rows, images)
        out = pd.DataFrame({
            "id": [r["id"] for r in rows],
            "pred": [LETTERS[int(p.argmax())] for p in probs],
            **{f"prob_{c}": probs[:, j].round(6) for j, c in enumerate(LETTERS)},
        })
        out.to_csv(out_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        write_header = False

        n_done += len(rows)
        if (i // batch_size) % max(1, log_every // batch_size) == 0 or n_done == len(todo):
            el = time.time() - start
            print(f"{n_done}/{len(todo)}  문항당 {el / n_done:.2f}초  경과 {el / 60:.1f}분", flush=True)

    el = time.time() - start
    print(f"\n완료: {n_done}문항, 문항당 평균 {el / n_done:.2f}초 → 결과 {out_path}")


def main():
    ap = argparse.ArgumentParser(description="a/b/c/d 로짓 기반 VQA 추론")
    ap.add_argument("--csv", required=True, help="문항 csv (train.csv 또는 test.csv)")
    ap.add_argument("--data-dir", default="data", help="csv 의 path 컬럼 기준 폴더")
    ap.add_argument("--model", required=True, help="로컬 모델 폴더")
    ap.add_argument("--out", required=True, help="결과 csv (이어쓰기 지원)")
    ap.add_argument("--split", default=None, help="이 id 목록의 문항만 추론 (csv, id 컬럼)")
    ap.add_argument("--shard", type=parse_shard, default=None, help="예: 2/5 = 전체를 5조각으로 나눈 2번째")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (동작 확인용)")
    ap.add_argument("--prompt", default="p1", choices=sorted(PROMPTS), help="프롬프트 버전")
    ap.add_argument("--batch-size", type=int, default=2, help="OOM 이 나면 1로")
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28, help="이미지 해상도 상한")
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--load-4bit", action="store_true", help="4bit 양자화로 로드 (VRAM 절약)")
    args = ap.parse_args()

    if os.path.basename(args.csv).lower().startswith("test") and args.split:
        print("[주의] test 에 --split 을 쓰고 있습니다. 검증 분할은 train 기준입니다.")

    df = load_rows(args.csv, args.split, args.shard, args.limit)
    print(f"추론 대상 {len(df)}문항 | 프롬프트 {args.prompt} | max_pixels {args.max_pixels} | 배치 {args.batch_size}")

    scorer = Scorer(args.model, args.max_pixels, args.min_pixels, args.load_4bit, args.prompt)
    run(df, scorer, args.out, args.data_dir, args.batch_size)


if __name__ == "__main__":
    main()