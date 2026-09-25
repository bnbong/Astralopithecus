#!/usr/bin/env python3
"""두 모델의 예측 확률을 가중 평균해 결합 예측·제출 파일을 만든다 (SSAFY AI 챌린지 2회차)

사용 예 (27B 와 9B, 가중치 0.5 고정 = 27B 0.5 + 9B 0.5):
    # 1) 먼저 fold 0 에서 재현 확인 (정확도가 0.9576 근처로 나와야 함)
    python blend_preds.py --a fold0_base.csv --b preds_9b_zeroshot_fold0.csv --w-a 0.5 \
        --labels data/train.csv --out-probs blend_fold0_probs.csv

    # 2) test 결합 + 제출 파일 만들기
    python blend_preds.py --a <27B test 예측> --b preds_test_9b_p2.csv --w-a 0.5 \
        --sample data/sample_submission.csv --out-probs blend27b9b_w05_test_probs.csv \
        --submission blend27b9b_w05_submission.csv

    python check_submission.py --submission blend27b9b_w05_submission.csv \
        --sample data/sample_submission.csv --test data/test.csv

    # 3) 세 모델 결합 (A + B + Gemma). --w-c 로 C 의 가중치만 정하면 나머지를 A 와 B 가 반씩 나눈다.
    python blend_preds.py --a <A test 예측> --b <B test 예측> --c <Gemma test 예측> --w-c 0.4 \
        --sample data/sample_submission.csv --submission blend_abg_submission.csv

입력 예측 파일: id 와 확률 4개 컬럼(p_a~p_d 또는 prob_a~prob_d) 이 있으면 된다. 다른 컬럼은 무시한다.
id 는 .jpg 유무와 상관없이 같은 id 로 취급한다. 제출 파일의 id 는 sample_submission 의 표기를 그대로 쓴다.
결합식: 확률 = w_a * A + (1 - w_a) * B  (가중치는 검증에서 세밀하게 고르지 말고 미리 정한 값을 쓸 것)
--c 로 세 번째 파일을 주면 기본값은 균등 가중치(각 1/3)이고, --w-c 로 C 의 가중치만 정하면
나머지 (1 - w_c) 를 A 와 B 가 반씩 나눈다. 세 파일을 넣을 때 --w-a 는 쓸 수 없다.
"""
import argparse
import os
import re
import sys

import pandas as pd

LETTERS = list("abcd")
EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)


def norm_id(x):
    return EXT_RE.sub("", str(x).strip())


def load_probs(path, name):
    if not os.path.exists(path):
        sys.exit(f"[오류] {name}: 파일이 없습니다: {path}")
    df = pd.read_csv(path, dtype={"id": str}, encoding="utf-8-sig")
    if "id" not in df.columns:
        sys.exit(f"[오류] {name}: id 컬럼이 없습니다 ({list(df.columns)})")
    for prefix in ("prob_", "p_"):
        cols = [prefix + c for c in LETTERS]
        if all(c in df.columns for c in cols):
            out = df[["id"] + cols].copy()
            out.columns = ["id"] + [f"prob_{c}" for c in LETTERS]
            break
    else:
        sys.exit(f"[오류] {name}: p_a~p_d 또는 prob_a~prob_d 컬럼이 없습니다 ({list(df.columns)})")
    out["nid"] = out["id"].map(norm_id)
    if out["nid"].duplicated().any():
        sys.exit(f"[오류] {name}: 중복된 id 가 {out['nid'].duplicated().sum()}개 있습니다.")
    probs = out[[f"prob_{c}" for c in LETTERS]]
    if probs.isna().any().any():
        sys.exit(f"[오류] {name}: 확률에 빈 값이 있습니다.")
    sums = probs.sum(axis=1)
    if ((sums - 1).abs() > 0.05).any():
        print(f"[경고] {name}: 확률 합이 1 에서 0.05 이상 벗어난 행이 {int(((sums - 1).abs() > 0.05).sum())}개 있습니다.")
    return out.set_index("nid")


def main():
    ap = argparse.ArgumentParser(description="두 모델 확률 결합")
    ap.add_argument("--a", required=True, help="예측 파일 A (예: 27B)")
    ap.add_argument("--b", required=True, help="예측 파일 B (예: 9B)")
    ap.add_argument("--c", default=None, help="예측 파일 C (선택). 주면 A 와 B 와 C 를 균등 가중치로 평균한다")
    ap.add_argument("--w-a", type=float, default=None, help="A 의 가중치 (B 는 1-w, 기본값 0.5). --c 와 함께 쓸 수 없다")
    ap.add_argument("--w-c", type=float, default=None, help="C 의 가중치 (기본값 1/3). 나머지는 A 와 B 가 반씩 나눈다")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--name-c", default="C")
    ap.add_argument("--sample", default=None, help="sample_submission.csv (제출 파일을 만들 때 필요, id 기준)")
    ap.add_argument("--labels", default=None, help="정답이 있는 train.csv (fold 0 재현 확인용)")
    ap.add_argument("--out-probs", default=None, help="결합 확률 csv (id, pred, prob_a~d)")
    ap.add_argument("--submission", default=None, help="제출 csv (id, answer)")
    args = ap.parse_args()
    if args.c is not None and args.w_a is not None:
        sys.exit("[오류] --c 로 세 파일을 결합할 때는 --w-a 대신 --w-c 로 C 의 가중치만 정합니다.")
    if args.c is None and args.w_c is not None:
        sys.exit("[오류] --w-c 는 --c 로 세 번째 파일을 줄 때만 쓸 수 있습니다.")
    w_a = 0.5 if args.w_a is None else args.w_a
    if not 0.0 <= w_a <= 1.0:
        sys.exit("[오류] --w-a 는 0~1 사이여야 합니다.")
    w_c = 1.0 / 3 if args.w_c is None else args.w_c
    if not 0.0 <= w_c <= 1.0:
        sys.exit("[오류] --w-c 는 0~1 사이여야 합니다.")
    if args.submission and not args.sample:
        sys.exit("[오류] 제출 파일을 만들려면 --sample 이 필요합니다.")

    frames = [(args.name_a, load_probs(args.a, args.name_a)), (args.name_b, load_probs(args.b, args.name_b))]
    if args.c is not None:
        frames.append((args.name_c, load_probs(args.c, args.name_c)))
    names = [name for name, _ in frames]
    if args.c is None:
        weights = [w_a, 1.0 - w_a]
    else:
        weights = [(1.0 - w_c) / 2, (1.0 - w_c) / 2, w_c]
    pc = [f"prob_{c}" for c in LETTERS]

    if args.sample:
        s = pd.read_csv(args.sample, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        order = list(s["id"])
        nids = [norm_id(x) for x in order]
        missing = [(name, set(nids) - set(df.index)) for name, df in frames]
        if any(miss for _, miss in missing):
            every = sorted(set().union(*(miss for _, miss in missing)))
            sys.exit("[오류] sample 의 id 중 " + ", ".join(f"{name}에 없는 것 {len(miss)}개" for name, miss in missing)
                     + f" (예: {every[:3]}). 추론이 끝까지 안 됐거나 다른 파일일 수 있어요.")
        extra = set().union(*(set(df.index) for _, df in frames)) - set(nids)
        if extra:
            print(f"[경고] sample 에 없는 id 가 {len(extra)}개 있어 제외합니다 (예: {sorted(extra)[:3]}).")
        ids_out = dict(zip(nids, order))
    else:
        first = frames[0][1]
        nids = [i for i in first.index if all(i in df.index for _, df in frames[1:])]
        if any(len(nids) != len(df) for _, df in frames):
            detail = ", ".join(f"{name} {len(df)}개" for name, df in frames)
            print(f"[경고] 공통 id {len(nids)}개만 결합합니다 ({detail}).")
        ids_out = {n: first.loc[n, "id"] for n in nids}
        order = [ids_out[n] for n in nids]

    mats = [df.loc[nids, pc].values for _, df in frames]
    blend = sum(w * m for w, m in zip(weights, mats))
    preds = [[LETTERS[i] for i in m.argmax(1)] for m in mats]
    pred = [LETTERS[i] for i in blend.argmax(1)]

    n = len(nids)
    print(f"결합 문항 수: {n} | 가중치 " + " + ".join(f"{name} {w:.2f}" for name, w in zip(names, weights)))
    print(f"{'와 '.join(names)}의 예측이 같은 문항: {sum(len(set(row)) == 1 for row in zip(*preds))}/{n}")
    print("결합 결과가 " + " | ".join(f"{name}와 다른 문항: {sum(x != y for x, y in zip(pred, p))}"
                                     for name, p in zip(names, preds)))
    dist = pd.Series(pred).value_counts(normalize=True).reindex(LETTERS, fill_value=0)
    print("결합 예측 분포:", ", ".join(f"{c}={dist[c]:.1%}" for c in LETTERS))

    if args.labels:
        lab = pd.read_csv(args.labels, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        gold = dict(zip(lab["id"].map(norm_id), lab["answer"].str.strip().str.lower()))
        have = [i for i, n_ in enumerate(nids) if n_ in gold]
        if have:
            def acc(p):
                return sum(p[i] == gold[nids[i]] for i in have) / len(have)
            parts = [f"{name} {acc(p):.4f}" for name, p in zip(names, preds)]
            print(f"[정확도 확인 {len(have)}문항] " + " | ".join(parts) + f" | 결합 {acc(pred):.4f}")
        else:
            print("[정확도 확인] 정답이 있는 id 가 없습니다 (test 는 정답이 없어요).")

    out = pd.DataFrame({"id": order, "pred": pred})
    for j, c in enumerate(LETTERS):
        out[f"prob_{c}"] = blend[:, j].round(6)
    if args.out_probs:
        out.to_csv(args.out_probs, index=False, encoding="utf-8-sig")
        print(f"결합 확률 저장: {args.out_probs}")
    if args.submission:
        pd.DataFrame({"id": order, "answer": pred}).to_csv(args.submission, index=False)
        print(f"제출 파일 저장: {args.submission} ({n}행) → check_submission.py 로 검사한 뒤 제출하세요.")


if __name__ == "__main__":
    main()
