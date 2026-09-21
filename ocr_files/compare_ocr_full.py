#!/usr/bin/env python3
"""val_fold0 전체(1,344건) 기준 승자 프롬프트(p2) vs p4(OCR) 교차비교.

extract_npn.py 의 키워드 분류를 재사용해 전체 문항에 이름/가격/숫자/기타
유형을 매기고, p2 대비 p4 가 유형별로 어디서 이기고 지는지 살펴본다.

사용법:
    python ocr_files/compare_ocr_full.py
"""
import re

import pandas as pd

EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)

NAME_KWS = ["이름", "상호", "상표", "제품명", "브랜드", "누구", "회사"]
PRICE_KWS = ["가격", "얼마", "금액", "요금", "원가"]
NUMBER_KWS = [
    r"번호", r"몇\s*개", r"몇\s*명", r"몇\s*시", r"날짜", r"며칠",
    r"언제", r"연도", r"기간", r"시간", r"층", r"개수",
]
NAME_RE = re.compile("|".join(re.escape(k) for k in NAME_KWS))
PRICE_RE = re.compile("|".join(re.escape(k) for k in PRICE_KWS))
NUMBER_RE = re.compile("|".join(NUMBER_KWS))
PRIORITY = {"name": 0, "price": 1, "number": 2}


def norm_id(x):
    return EXT_RE.sub("", str(x).strip())


def primary_type(question):
    types = []
    if NAME_RE.search(question):
        types.append("name")
    if PRICE_RE.search(question):
        types.append("price")
    if NUMBER_RE.search(question):
        types.append("number")
    if not types:
        return "other"
    return sorted(types, key=PRIORITY.get)[0]


def main():
    train = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    train["id"] = train["id"].map(norm_id)
    train["answer"] = train["answer"].str.strip().str.lower()

    val_ids = set(
        pd.read_csv("fold_csv_files/val_fold0.csv", dtype=str, keep_default_na=False,
                    encoding="utf-8-sig")["id"].map(norm_id)
    )
    sub = train[train["id"].isin(val_ids)].copy()
    sub["primary_type"] = sub["question"].map(primary_type)

    p2 = pd.read_csv("ocr_files/preds/preds_qwen3b_p2_fold0.csv", dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    p2["id"] = p2["id"].map(norm_id)
    p2 = p2[["id", "pred"]].rename(columns={"pred": "p2_pred"})

    p4 = pd.read_csv("ocr_files/preds/preds_qwen3b_p4_fold0_full.csv", dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    p4["id"] = p4["id"].map(norm_id)
    p4 = p4[["id", "pred"]].rename(columns={"pred": "p4_pred"})

    df = sub.merge(p2, on="id", how="left").merge(p4, on="id", how="left")
    df["p2_correct"] = df["p2_pred"].str.lower() == df["answer"]
    df["p4_correct"] = df["p4_pred"].str.lower() == df["answer"]

    helped = df[(~df["p2_correct"]) & (df["p4_correct"])]
    hurt = df[(df["p2_correct"]) & (~df["p4_correct"])]
    both_wrong = df[(~df["p2_correct"]) & (~df["p4_correct"])]
    both_correct = df[(df["p2_correct"]) & (df["p4_correct"])]

    print(f"평가 문항 수: {len(df)}")
    print(f"둘 다 정답: {len(both_correct)}")
    print(f"둘 다 오답: {len(both_wrong)}")
    print(f"OCR 덕분에 맞음 (p2 오답 -> p4 정답): {len(helped)}")
    print(f"OCR 때문에 틀림 (p2 정답 -> p4 오답): {len(hurt)}")
    print(f"순변화(도움-손해): {len(helped) - len(hurt)}")

    print("\n[전체 유형 분포] (fold0 1344건)")
    print(df["primary_type"].value_counts())

    print("\n[유형별] OCR 덕분에 맞음")
    print(helped["primary_type"].value_counts())
    print("\n[유형별] OCR 때문에 틀림")
    print(hurt["primary_type"].value_counts())

    print("\n[유형별] 순변화 (도움-손해) 및 유형 내 비율")
    for t in ["name", "price", "number", "other"]:
        n_type = (df["primary_type"] == t).sum()
        h = (helped["primary_type"] == t).sum()
        u = (hurt["primary_type"] == t).sum()
        print(f"  {t:7s} n={n_type:4d}  도움={h:3d}  손해={u:3d}  순변화={h - u:+3d}")

    def dump(frame, path, n=10):
        with open(path, "w", encoding="utf-8") as f:
            for _, r in frame.head(n).iterrows():
                f.write(f"id={r['id']} type={r['primary_type']} answer={r['answer']} "
                        f"p2={r['p2_pred']} p4={r['p4_pred']}\n")
                f.write(f"  Q: {r['question']}\n")

    dump(helped, "ocr_files/_tmp_helped_full.txt")
    dump(hurt, "ocr_files/_tmp_hurt_full.txt")
    print("\n예시 저장: ocr_files/_tmp_helped_full.txt, ocr_files/_tmp_hurt_full.txt")


if __name__ == "__main__":
    main()
