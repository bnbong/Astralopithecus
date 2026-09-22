#!/usr/bin/env python3
"""val_fold0 문항 중 '이름/가격/숫자' 유형 텍스트 질문을 골라 샘플 csv로 저장한다.

이름형/가격형/숫자형 여부는 question 텍스트의 키워드로 판정한다 (완벽하지 않은
휴리스틱). 세 유형 중 하나라도 걸리면 후보에 포함하고, 후보가 300건을 넘으면
random_state=42 로 무작위 300건만 뽑는다.

사용법:
    python experiments/ocr/extract_npn.py
"""
import re

import pandas as pd

NAME_KWS = ["이름", "상호", "상표", "제품명", "브랜드", "누구", "회사"]
PRICE_KWS = ["가격", "얼마", "금액", "요금", "원가"]
NUMBER_KWS = [
    r"번호", r"몇\s*개", r"몇\s*명", r"몇\s*시", r"날짜", r"며칠",
    r"언제", r"연도", r"기간", r"시간", r"층", r"개수",
]

NAME_RE = re.compile("|".join(re.escape(k) for k in NAME_KWS))
PRICE_RE = re.compile("|".join(re.escape(k) for k in PRICE_KWS))
NUMBER_RE = re.compile("|".join(NUMBER_KWS))

N_SAMPLE = 300
SEED = 42


def classify(question):
    types = []
    if NAME_RE.search(question):
        types.append("name")
    if PRICE_RE.search(question):
        types.append("price")
    if NUMBER_RE.search(question):
        types.append("number")
    return types


def main():
    train = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    val_ids = set(
        pd.read_csv("fold_csv_files/val_fold0.csv", dtype=str, keep_default_na=False,
                    encoding="utf-8-sig")["id"]
    )
    sub = train[train["id"].isin(val_ids)].reset_index(drop=True)
    print(f"fold0 전체 문항: {len(sub)}")

    types = sub["question"].map(classify)
    matched = types.map(len) > 0
    cand = sub[matched].copy()
    cand["matched_types"] = types[matched].map(lambda t: ",".join(t))
    # 우선순위: name > price > number (표시용 대표 유형 하나만 필요할 때 사용)
    priority = {"name": 0, "price": 1, "number": 2}
    cand["primary_type"] = types[matched].map(lambda t: sorted(t, key=priority.get)[0])

    print(f"이름/가격/숫자 후보 문항: {len(cand)}")

    if len(cand) > N_SAMPLE:
        out = cand.sample(n=N_SAMPLE, random_state=SEED).sort_index()
    else:
        out = cand

    out_path = "experiments/ocr/fold0_name_price_number.csv"
    out[["id", "question", "matched_types", "primary_type"]].to_csv(
        out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(out)}건)")

    print("\n[후보 전체] 유형별 매칭 건수 (중복 포함, 한 문항이 여러 유형에 걸릴 수 있음)")
    for t in ("name", "price", "number"):
        print(f"  {t}: {sum(t in x for x in types[matched])}")

    print("\n[최종 샘플] 대표 유형(primary_type) 분포")
    print(out["primary_type"].value_counts())


if __name__ == "__main__":
    main()
