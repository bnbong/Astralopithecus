#!/usr/bin/env python3
"""오답 문항 검토용 페이지 생성 (SSAFY AI 챌린지 2회차)

이미지를 한 장씩 열지 않고, 문항 여러 개를 한 페이지에서 훑어볼 수 있도록
"이미지 + 질문 + 선택지(정답 표시) + 모델 예측"을 하나의 HTML 파일로 묶는다.

사용법:
    python review_sheet.py --csv both_wrong_3b7b_fold0_300.csv --data-dir data --out review.html

    # 일부 문항만 (예: 라벨이 의심스러운 4개)
    python review_sheet.py --csv both_wrong_3b7b_fold0_300.csv --data-dir data --out review_4.html \
        --ids train_0047.jpg train_0968.jpg train_0381.jpg train_1000.jpg

입력 csv 에 필요한 컬럼: id, question, a, b, c, d, answer
  선택: pred / pred_3b / pred_7b 처럼 pred 로 시작하는 컬럼(모델 예측), 유형(수동)
  이미지 경로는 csv 의 path 컬럼, 없으면 --labels(train.csv)에서 id 로 찾고, 그것도 없으면 train/<id>

결과 html 은 이미지가 안에 포함되어 있어(자체 완결) 다운로드해서 브라우저로 열면 된다.
"""
import argparse
import base64
import html
import io
import os
import sys

import pandas as pd
from PIL import Image

LETTERS = list("abcd")


def read_csv(path):
    if not os.path.exists(path):
        sys.exit(f"[오류] 파일이 없습니다: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def image_data_uri(path, max_width):
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        h = round(img.height * max_width / img.width)
        img = img.resize((max_width, h))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def resolve_path(row, id_to_path, data_dir):
    if row.get("path"):
        rel = row["path"]
    else:
        rel = id_to_path.get(row["id"], f"train/{row['id']}")
    return os.path.join(data_dir, rel)


def render_item(idx, row, img_uri, pred_cols):
    e = html.escape
    picks = {}   # 선지 -> ["3B", "7B"] 처럼 그 선지를 고른 모델 이름
    for c in pred_cols:
        name = c[len("pred"):].strip("_").upper() or "모델"
        picks.setdefault(row[c].strip().lower(), []).append(name)

    opts = []
    for L in LETTERS:
        marks = []
        if row["answer"].strip().lower() == L:
            marks.append('<span class="gold">정답</span>')
        for name in picks.get(L, []):
            marks.append(f'<span class="pick">{e(name)} 선택</span>')
        cls = "opt gold-row" if row["answer"].strip().lower() == L else "opt"
        opts.append(f'<li class="{cls}"><b>{L}.</b> {e(row[L])} {" ".join(marks)}</li>')

    tag = row.get("유형(수동)", "")
    tag_html = f'<span class="tag">{e(tag)}</span>' if tag else ""
    img_html = (f'<img src="{img_uri}">' if img_uri
                else '<div class="missing">이미지를 찾지 못했습니다</div>')
    return f"""
<section>
  <h2>{idx}. {e(row['id'])} {tag_html}</h2>
  <div class="grid">
    <div class="img">{img_html}</div>
    <div class="txt">
      <p class="q">{e(row['question'])}</p>
      <ul>{''.join(opts)}</ul>
    </div>
  </div>
</section>"""


STYLE = """
body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:20px} section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}
h2{font-size:16px;margin:0 0 12px} .grid{display:flex;gap:20px;flex-wrap:wrap} .img img{max-width:640px;border:1px solid #ccc}
.txt{flex:1;min-width:300px} .q{font-weight:bold;font-size:16px} ul{list-style:none;padding:0}
.opt{padding:6px 8px;margin:4px 0;border-radius:4px;background:#f3f3f3}
.gold-row{background:#e3f4e3} .gold{background:#2e8b57;color:#fff;padding:1px 6px;border-radius:4px;font-size:12px}
.pick{background:#c0392b;color:#fff;padding:1px 6px;border-radius:4px;font-size:12px;margin-left:4px}
.tag{background:#eee;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:normal}
.missing{padding:40px;background:#fdd}
"""


def main():
    ap = argparse.ArgumentParser(description="오답 문항 검토 페이지(HTML) 생성")
    ap.add_argument("--csv", required=True, help="오답 목록 csv")
    ap.add_argument("--data-dir", default="data", help="이미지가 있는 데이터 폴더")
    ap.add_argument("--labels", default="data/train.csv", help="id -> path 를 찾을 csv (없으면 무시)")
    ap.add_argument("--out", default="review.html")
    ap.add_argument("--ids", nargs="*", default=None, help="이 id 만 포함")
    ap.add_argument("--max-width", type=int, default=900, help="이미지 가로 최대 크기(파일 용량 조절)")
    args = ap.parse_args()

    df = read_csv(args.csv)
    need = ["id", "question", "answer"] + LETTERS
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[오류] csv 에 필요한 컬럼이 없습니다: {miss}")
    if args.ids:
        df = df[df["id"].isin(args.ids)]
        if df.empty:
            sys.exit("[오류] --ids 에 해당하는 문항이 csv 에 없습니다.")

    id_to_path = {}
    if os.path.exists(args.labels):
        lab = read_csv(args.labels)
        if {"id", "path"} <= set(lab.columns):
            id_to_path = dict(zip(lab["id"], lab["path"]))

    pred_cols = [c for c in df.columns if c.startswith("pred")]

    parts, missing = [], 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        row = row.to_dict()
        p = resolve_path(row, id_to_path, args.data_dir)
        if os.path.exists(p):
            uri = image_data_uri(p, args.max_width)
        else:
            uri, missing = None, missing + 1
            print(f"[경고] 이미지를 찾지 못했습니다: {p}")
        parts.append(render_item(i, row, uri, pred_cols))

    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>오답 문항 검토</title>"
            f"<style>{STYLE}</style></head><body>"
            f"<h1>오답 문항 검토 ({len(parts)}문항)</h1>"
            f"<p>초록색이 정답 라벨, 빨간색이 모델이 고른 선택지입니다. 이미지로 정답이 명확한지, "
            f"질문·선택지가 애매하지 않은지 확인하세요.</p>{''.join(parts)}</body></html>")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    size_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f"{len(parts)}문항 → {args.out} ({size_mb:.1f}MB), 이미지 누락 {missing}개")


if __name__ == "__main__":
    main()