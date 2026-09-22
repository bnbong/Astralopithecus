#!/usr/bin/env python3
"""fold0_npn(300건) 기준 승자 프롬프트(p2) vs p4(OCR) 를 문항 단위로 교차비교한다.

- OCR 덕분에 맞은 문항 (p2 오답 -> p4 정답)
- OCR 때문에 오히려 틀린 문항 (p2 정답 -> p4 오답)
- 유형(name/price/number)별 위 두 그룹 건수

사용법:
    python experiments/ocr/compare_ocr.py
"""
import re

import pandas as pd

EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)


def norm_id(x):
    return EXT_RE.sub("", str(x).strip())


def main():
    sample = pd.read_csv("experiments/ocr/fold0_name_price_number.csv", dtype=str,
                          keep_default_na=False, encoding="utf-8-sig")
    sample["id"] = sample["id"].map(norm_id)

    labels = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False,
                          encoding="utf-8-sig")
    labels["id"] = labels["id"].map(norm_id)
    labels["answer"] = labels["answer"].str.strip().str.lower()

    p2 = pd.read_csv("experiments/ocr/preds/preds_qwen3b_p2_fold0.csv", dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    p2["id"] = p2["id"].map(norm_id)
    p2 = p2[p2["id"].isin(set(sample["id"]))][["id", "pred"]].rename(columns={"pred": "p2_pred"})

    p4 = pd.read_csv("experiments/ocr/preds/preds_qwen3b_p4_fold0_npn.csv", dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    p4["id"] = p4["id"].map(norm_id)
    p4 = p4[["id", "pred"]].rename(columns={"pred": "p4_pred"})

    df = sample.merge(labels[["id", "answer"]], on="id", how="left")
    df = df.merge(p2, on="id", how="left").merge(p4, on="id", how="left")

    df["p2_correct"] = df["p2_pred"].str.lower() == df["answer"]
    df["p4_correct"] = df["p4_pred"].str.lower() == df["answer"]

    helped = df[(~df["p2_correct"]) & (df["p4_correct"])]     # OCR 덕분에 맞음
    hurt = df[(df["p2_correct"]) & (~df["p4_correct"])]        # OCR 때문에 틀림
    both_wrong = df[(~df["p2_correct"]) & (~df["p4_correct"])]
    both_correct = df[(df["p2_correct"]) & (df["p4_correct"])]

    print(f"평가 문항 수: {len(df)}")
    print(f"둘 다 정답: {len(both_correct)}")
    print(f"둘 다 오답: {len(both_wrong)}")
    print(f"OCR 덕분에 맞음 (p2 오답 -> p4 정답): {len(helped)}")
    print(f"OCR 때문에 틀림 (p2 정답 -> p4 오답): {len(hurt)}")

    print("\n[유형별] OCR 덕분에 맞음")
    print(helped["primary_type"].value_counts())
    print("\n[유형별] OCR 때문에 틀림")
    print(hurt["primary_type"].value_counts())

    def dump(sub, path, n=8):
        with open(path, "w", encoding="utf-8") as f:
            for _, r in sub.head(n).iterrows():
                f.write(f"id={r['id']} type={r['primary_type']} answer={r['answer']} "
                        f"p2={r['p2_pred']} p4={r['p4_pred']}\n")
                f.write(f"  Q: {r['question']}\n")

    dump(helped, "experiments/ocr/_tmp_helped_examples.txt")
    dump(hurt, "experiments/ocr/_tmp_hurt_examples.txt")
    print("\n예시 저장: experiments/ocr/_tmp_helped_examples.txt, experiments/ocr/_tmp_hurt_examples.txt")


if __name__ == "__main__":
    main()
