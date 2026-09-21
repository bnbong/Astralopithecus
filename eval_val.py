#!/usr/bin/env python3
"""검증 세트 평가 스크립트 (SSAFY AI 챌린지 2회차)

사용법:
    python eval_val.py --pred preds.csv --split val_ids.csv --labels data/train.csv \
        --model qwen9b --prompt p1 --split-name val --log-csv results_log.csv

입력:
    --pred    예측 파일. id 와 pred(또는 answer) 컬럼 필수.
              pred 컬럼이 없고 prob_a~prob_d 가 있으면 argmax 로 계산.
    --split   검증 분할 id 목록. id 컬럼이 있는 csv, 또는 한 줄에 id 하나인 txt.
              (id 는 train_0001.jpg 처럼 확장자가 있든 없든 같은 id 로 취급)
    --labels  정답(answer)이 있는 train.csv.
              test.csv 는 정답이 없으므로 평가에 사용할 수 없다.

출력:
    - 정확도, 정답/예측 분포, 정답 선지별 정확도
    - 오답 목록 csv (out-dir/wrong_<model>_<prompt>_<split>.csv)
    - 로그 한 줄 (노션 실험 로그에 그대로 붙여넣기)
    - --log-csv 지정 시 결과를 누적 기록

종료 코드: 정상 0, 입력 오류 1
"""
import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd

LETTERS = list("abcd")
PROB_COLS = [f"prob_{c}" for c in LETTERS]


EXT_RE = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)


def norm_id(x):
    """id 비교용 정규화: 공백과 이미지 확장자(.jpg 등)를 제거한다.
    (실제 데이터의 id 는 test_0001.jpg 형태라, 파일마다 확장자가 붙고 빠지는 실수를 막는다)"""
    return EXT_RE.sub("", x.strip())


def norm_ids(series):
    return series.str.strip().str.replace(EXT_RE, "", regex=True)


class EvalError(Exception):
    """입력 데이터가 잘못되어 평가를 진행할 수 없을 때 발생."""


def read_csv(path):
    """모든 값을 문자열로 읽는다 (id 앞자리 0, 빈 값 보존)."""
    if not os.path.exists(path):
        raise EvalError(f"파일이 없습니다: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_split_ids(path):
    """검증 분할 id 목록을 읽어 set 으로 반환."""
    if path.lower().endswith(".txt"):
        with open(path, encoding="utf-8-sig") as f:
            ids = [norm_id(line) for line in f if line.strip()]
    else:
        df = read_csv(path)
        if "id" in df.columns:
            ids = norm_ids(df["id"]).tolist()
        elif df.shape[1] == 1:
            ids = norm_ids(df.iloc[:, 0]).tolist()
        else:
            raise EvalError(f"분할 파일에서 id 컬럼을 찾지 못했습니다: {list(df.columns)}")
    if len(ids) != len(set(ids)):
        raise EvalError("분할 파일에 중복된 id 가 있습니다.")
    if not ids:
        raise EvalError("분할 파일이 비어 있습니다.")
    return set(ids)


def load_labels(path):
    df = read_csv(path)
    for col in ("id", "answer"):
        if col not in df.columns:
            raise EvalError(
                f"정답 파일에 '{col}' 컬럼이 없습니다. train.csv 를 지정했는지 확인하세요 "
                "(test.csv, dev.csv 에는 정답이 없습니다)."
            )
    df["id_raw"] = df["id"].str.strip()   # 오답 파일에는 원래 id 를 남긴다
    df["id"] = norm_ids(df["id"])
    df["answer"] = df["answer"].str.strip().str.lower()
    return df


def load_pred(path):
    """예측 파일을 읽어 id, pred (+ prob_a~d) 컬럼으로 정리."""
    df = read_csv(path)
    if "id" not in df.columns:
        raise EvalError(f"예측 파일에 id 컬럼이 없습니다: {list(df.columns)}")
    df["id"] = norm_ids(df["id"])

    has_probs = all(c in df.columns for c in PROB_COLS)
    if has_probs:
        for c in PROB_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "pred" in df.columns:
        label_col = "pred"
    elif "answer" in df.columns:
        label_col = "answer"
    elif has_probs:
        idx = df[PROB_COLS].values.argmax(axis=1)
        df["pred"] = [LETTERS[i] for i in idx]
        label_col = "pred"
    else:
        raise EvalError("예측 파일에 pred(또는 answer) 컬럼도, prob_a~prob_d 컬럼도 없습니다.")

    df["pred"] = df[label_col].str.strip().str.lower()
    keep = ["id", "pred"] + (PROB_COLS if has_probs else [])
    return df[keep], has_probs


def evaluate(pred, labels, split_ids, allow_subset=False):
    """예측과 정답을 id 기준으로 합쳐 채점한다.

    반환: (merged DataFrame, warnings 리스트)
    """
    warnings = []

    dup = pred["id"][pred["id"].duplicated()].unique()
    if len(dup):
        raise EvalError(f"예측 파일에 중복된 id 가 {len(dup)}개 있습니다. 예: {list(dup[:3])}")

    label_ids = set(labels["id"])
    pred_ids = set(pred["id"])

    # 정답이 없는 id (test 데이터가 섞였을 가능성) - 항상 오류
    no_label = pred_ids - label_ids
    if no_label:
        raise EvalError(
            f"정답 파일(train.csv)에 없는 id 가 예측에 {len(no_label)}개 있습니다. "
            f"예: {sorted(no_label)[:3]}. test 예측을 넣은 것은 아닌지 확인하세요."
        )

    unknown_split = split_ids - label_ids
    if unknown_split:
        raise EvalError(
            f"분할 파일의 id 중 {len(unknown_split)}개가 정답 파일에 없습니다. "
            f"예: {sorted(unknown_split)[:3]}"
        )

    extra = pred_ids - split_ids
    missing = split_ids - pred_ids
    if extra:
        if not allow_subset:
            raise EvalError(
                f"예측에 검증 분할에 없는 id 가 {len(extra)}개 있습니다. "
                f"예: {sorted(extra)[:3]}. (train 로 학습한 문항이 섞이면 점수가 부풀려집니다)"
            )
        warnings.append(f"검증 분할 밖의 id {len(extra)}개는 제외하고 평가합니다.")
    if missing:
        if not allow_subset:
            raise EvalError(
                f"검증 분할의 id 중 {len(missing)}개가 예측에 없습니다. "
                f"예: {sorted(missing)[:3]}. 일부만 빠르게 볼 때는 --allow-subset 을 사용하세요."
            )
        warnings.append(f"예측이 없는 검증 id {len(missing)}개는 제외하고 평가합니다.")

    eval_ids = split_ids & pred_ids
    if not eval_ids:
        raise EvalError("평가할 id 가 하나도 없습니다.")

    gold = labels[labels["id"].isin(eval_ids)]
    merged = gold.merge(pred, on="id", how="inner")
    merged["correct"] = merged["pred"] == merged["answer"]

    invalid = ~merged["pred"].isin(LETTERS)
    if invalid.any():
        warnings.append(f"a~d 가 아닌 예측 {int(invalid.sum())}개는 오답으로 처리했습니다.")
    return merged, warnings


def summarize(merged):
    n = len(merged)
    acc = merged["correct"].mean()
    gold_dist = merged["answer"].value_counts().reindex(LETTERS, fill_value=0)
    pred_dist = merged["pred"].value_counts().reindex(LETTERS, fill_value=0)
    per_gold = merged.groupby("answer")["correct"].mean().reindex(LETTERS)
    return n, acc, gold_dist, pred_dist, per_gold


def save_wrong(merged, path, has_probs):
    wrong = merged[~merged["correct"]].copy()
    if has_probs:
        wrong["conf"] = wrong[PROB_COLS].max(axis=1)
        wrong["p_gold"] = [
            row[f"prob_{row['answer']}"] if row["answer"] in LETTERS else float("nan")
            for _, row in wrong.iterrows()
        ]
    if "id_raw" in wrong.columns:
        wrong = wrong.drop(columns=["id"]).rename(columns={"id_raw": "id"})
    cols = [c for c in ["id", "question", "a", "b", "c", "d", "answer", "pred", "conf", "p_gold"]
            if c in wrong.columns]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wrong[cols].to_csv(path, index=False, encoding="utf-8-sig")
    return len(wrong)


def append_log(path, row):
    df = pd.DataFrame([row])
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="검증 세트 평가")
    ap.add_argument("--pred", required=True, help="예측 csv")
    ap.add_argument("--split", required=True, help="검증 분할 id 목록 (csv 또는 txt)")
    ap.add_argument("--labels", required=True, help="정답이 있는 train.csv")
    ap.add_argument("--model", default="model", help="모델 이름 (로그용)")
    ap.add_argument("--prompt", default="p1", help="프롬프트 버전 (로그용)")
    ap.add_argument("--split-name", default="val", help="분할 이름 (로그용)")
    ap.add_argument("--out-dir", default="eval_out", help="오답 파일 저장 폴더")
    ap.add_argument("--log-csv", default=None, help="결과를 누적 기록할 csv")
    ap.add_argument("--allow-subset", action="store_true",
                    help="예측이 분할의 일부일 때 교집합으로만 평가 (빠른 확인용)")
    args = ap.parse_args()

    try:
        split_ids = load_split_ids(args.split)
        labels = load_labels(args.labels)
        pred, has_probs = load_pred(args.pred)
        merged, warnings = evaluate(pred, labels, split_ids, args.allow_subset)
    except EvalError as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)

    for w in warnings:
        print(f"[경고] {w}")

    n, acc, gold_dist, pred_dist, per_gold = summarize(merged)

    print(f"\n평가 문항 수 : {n}")
    print(f"정확도       : {acc:.4f}  ({int(merged['correct'].sum())}/{n})")
    print("\n선지별 분포와 정확도")
    print(f"{'':6s}{'정답':>8s}{'예측':>8s}{'정답 선지별 정확도':>20s}")
    for c in LETTERS:
        print(f"{c:6s}{gold_dist[c]:8d}{pred_dist[c]:8d}{per_gold[c]:20.4f}")
    skew = pred_dist.max() / max(n, 1)
    if skew > 0.40:
        print(f"\n[주의] 예측이 한 선지({pred_dist.idxmax()})에 {skew:.0%} 쏠려 있습니다. 위치 편향을 확인하세요.")

    wrong_path = os.path.join(args.out_dir, f"wrong_{args.model}_{args.prompt}_{args.split_name}.csv")
    n_wrong = save_wrong(merged, wrong_path, has_probs)
    print(f"\n오답 {n_wrong}개 저장: {wrong_path}")

    line = f"{args.model} | {args.prompt} | {args.split_name} | n={n} | acc={acc:.4f}"
    print(f"\n[노션 로그용] {line}")

    if args.log_csv:
        append_log(args.log_csv, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": args.model,
            "prompt": args.prompt,
            "split": args.split_name,
            "n": n,
            "acc": round(float(acc), 4),
            "pred_file": os.path.basename(args.pred),
        })
        print(f"결과 누적 기록: {args.log_csv}")


if __name__ == "__main__":
    main()