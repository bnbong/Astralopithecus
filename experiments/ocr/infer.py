#!/usr/bin/env python3
"""VQA 4지선다 추론 스크립트 - OCR 프롬프트 실험용 (experiments/ocr 전용, SSAFY AI 챌린지 2회차)

infer_val.py 의 방식(정답을 생성하지 않고 프롬프트 직후 첫 토큰에서 a/b/c/d 네 글자의
로짓만 비교, --split 기반 필터링, 이어쓰기 지원)을 그대로 참고해 새로 작성했다.
기존 infer_val.py / eval_val.py 는 읽기만 하고 수정하지 않는다.

p3 는 이미지에서 OCR(EasyOCR)로 뽑은 텍스트를 프롬프트에 함께 넣는 버전이다.
OCR 은 이미지 경로 기준으로 experiments/ocr/ocr_cache.json 에 캐시해서, 같은 이미지를
여러 문항/여러 실행에서 다시 만나도 한 번만 돌린다.

사용 예 (val_fold0 기준):
    python experiments/ocr/infer.py --csv data/train.csv --split fold_csv_files/val_fold0.csv \
        --data-dir data --model experiments/ocr/models/Qwen2.5-VL-3B-Instruct \
        --prompt p3 --out experiments/ocr/preds/preds_qwen3b_p3_fold0.csv

먼저 5문항으로 동작 확인: ... --limit 5 --out experiments/ocr/preds/preds_try.csv

중간에 끊겨도 같은 명령을 다시 실행하면 이미 끝난 id 는 건너뛰고 이어서 진행한다.
주의: test.csv 에는 정답이 없다. test 로 학습하거나 튜닝하지 않는다(규칙: Data Leakage 금지).
"""
import argparse
import json
import os
import re
import time

import pandas as pd

LETTERS = list("abcd")
EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)
OCR_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache.json")

# 프롬프트 버전.
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
    "p3": (
        "이미지를 자세히 보고 질문에 답하세요.\n\n"
        "- 이미지 안에 텍스트(간판, 문서, 표지판, 숫자, 날짜 등)가 보이면, "
        "그 글자를 정확히 읽어서 답에 반영하세요.\n"
        "- 이미지에 텍스트가 없거나 텍스트만으로 답할 수 없는 경우, "
        "이미지 속 사물, 현상, 상황을 관찰하여 알고 있는 지식을 바탕으로 판단하세요.\n\n"
        "질문: {question}\n\n"
        "a. {a}\nb. {b}\nc. {c}\nd. {d}\n\n"
        "정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요."
    ),
    "p4": (
        "당신은 이미지를 보고 질문에 정확하게 답하는 어시스턴트입니다.\n\n"
        "아래는 이미지에서 OCR로 추출한 텍스트입니다. 이미지와 함께 참고하세요.\n"
        "(OCR은 인식 오류가 있을 수 있으니, 이미지와 대조하여 신뢰도를 판단하세요.\n"
        " 이미지에 텍스트가 없는 경우 OCR 결과는 비어 있을 수 있습니다.)\n\n"
        "[OCR 추출 텍스트]\n{ocr_text}\n\n"
        "지침:\n"
        "1. OCR 텍스트와 실제 이미지를 함께 확인하여 텍스트 관련 질문에 정확히 답하세요.\n"
        "2. OCR에 오탈자나 누락이 의심되면 이미지를 우선하여 판단하세요.\n"
        "3. OCR 텍스트가 비어 있거나 질문이 텍스트와 무관한 경우, 이미지 속 사물·\n"
        "   현상·상황을 직접 관찰하고 알고 있는 지식을 활용하여 답하세요.\n"
        "   (예: 자연 현상, 사물의 종류, 장면의 상황 등)\n\n"
        "질문: {question}\n"
        "a. {a}\nb. {b}\nc. {c}\nd. {d}\n\n"
        "정답을 a, b, c, d 중 하나의 알파벳 소문자로만 답하세요."
    ),
}

# {ocr_text} 를 쓰는 프롬프트만 OCR 단계를 거친다.
NEEDS_OCR = {name for name, tpl in PROMPTS.items() if "{ocr_text}" in tpl}


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
        lo, hi = len(df) * k // n, len(df) * (k + 1) // n  # 연속 구간으로 나눔
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


class OcrCache:
    """이미지 경로 기준 OCR 결과 캐시. 같은 이미지를 다시 만나면 캐시에서 바로 읽는다."""

    def __init__(self, path=OCR_CACHE_PATH, langs=("ko", "en")):
        self.path = path
        self.langs = list(langs)
        self.data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.reader = None
        self._dirty = False

    def _ensure_reader(self):
        if self.reader is None:
            import easyocr
            import torch
            gpu = torch.cuda.is_available()
            print(f"[OCR] EasyOCR 리더 로드 중 (langs={self.langs}, gpu={gpu}) ...")
            self.reader = easyocr.Reader(self.langs, gpu=gpu)

    def get(self, image_path):
        key = os.path.abspath(image_path)
        if key in self.data:
            return self.data[key]
        self._ensure_reader()
        result = self.reader.readtext(key, detail=0, paragraph=True)
        text = "\n".join(result).strip()
        self.data[key] = text
        self._dirty = True
        return text

    def save(self):
        if self._dirty:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
            self._dirty = False


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
        self.processor.tokenizer.padding_side = "left"  # 배치 시 마지막 위치를 맞추기 위해

        # a~d 각각이 토큰 1개인지 확인
        self.letter_ids = []
        for c in LETTERS:
            ids = self.processor.tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"[오류] '{c}' 가 토큰 1개가 아닙니다: {ids}. 토크나이저를 확인하세요.")
            self.letter_ids.append(ids[0])

    def score(self, rows, images, ocr_texts=None):
        """rows: 딕셔너리 리스트, images: PIL 이미지 리스트 -> (n, 4) 확률 (numpy)"""
        torch = self.torch
        texts = []
        for i, r in enumerate(rows):
            fmt = dict(question=r["question"], a=r["a"], b=r["b"], c=r["c"], d=r["d"])
            if ocr_texts is not None:
                fmt["ocr_text"] = ocr_texts[i] or "(텍스트 없음)"
            prompt = self.template.format(**fmt)
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]  # 다음 토큰 분포
        four = logits[:, self.letter_ids].float()  # a,b,c,d 만 비교
        return torch.softmax(four, dim=-1).cpu().numpy()


def run(df, scorer, out_path, data_dir, batch_size, needs_ocr, ocr_cache, log_every=20):
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
        images, paths = [], []
        for r in rows:
            p = os.path.join(data_dir, r["path"])
            if not os.path.exists(p):
                raise SystemExit(f"[오류] 이미지가 없습니다: {p} (id={r['id']}). --data-dir 를 확인하세요.")
            images.append(Image.open(p).convert("RGB"))
            paths.append(p)

        ocr_texts = None
        if needs_ocr:
            ocr_texts = [ocr_cache.get(p) for p in paths]
            ocr_cache.save()

        probs = scorer.score(rows, images, ocr_texts)
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
    ap = argparse.ArgumentParser(description="a/b/c/d 로짓 기반 VQA 추론 (OCR 프롬프트 실험용)")
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
    ap.add_argument("--ocr-langs", default="ko,en", help="EasyOCR 언어 목록 (쉼표 구분)")
    args = ap.parse_args()

    if os.path.basename(args.csv).lower().startswith("test") and args.split:
        print("[주의] test 에 --split 을 쓰고 있습니다. 검증 분할은 train 기준입니다.")

    df = load_rows(args.csv, args.split, args.shard, args.limit)
    needs_ocr = args.prompt in NEEDS_OCR
    print(f"추론 대상 {len(df)}문항 | 프롬프트 {args.prompt} (OCR {'사용' if needs_ocr else '미사용'}) "
          f"| max_pixels {args.max_pixels} | 배치 {args.batch_size}")

    ocr_cache = OcrCache(langs=tuple(args.ocr_langs.split(","))) if needs_ocr else None
    scorer = Scorer(args.model, args.max_pixels, args.min_pixels, args.load_4bit, args.prompt)
    run(df, scorer, args.out, args.data_dir, args.batch_size, needs_ocr, ocr_cache)


if __name__ == "__main__":
    main()
