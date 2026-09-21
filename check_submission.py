#!/usr/bin/env python3
"""제출 파일 검사기 (SSAFY AI 챌린지 2회차)

사용법:
    python check_submission.py --submission submission.csv --sample data/sample_submission.csv
    python check_submission.py --submission submission.csv --sample data/sample_submission.csv \
        --test data/test.csv

검사 항목:
    1. 컬럼이 정확히 id, answer 인가 (인덱스 컬럼이 딸려 나오지 않았는가)
    2. 행 수가 sample_submission 과 같은가
    3. id 중복 / 누락 / 초과가 없는가 (--test 지정 시 test.csv 의 id 와도 대조)
    4. 결측이 없는가
    5. 값이 a, b, c, d 중 하나인가 (대문자, 공백, 따옴표 포함 여부)
    6. 답 분포가 한쪽으로 쏠리지 않았는가 (경고만)

종료 코드: 통과 0, 오류 있음 1 (경고만 있으면 0)
"""
import argparse
import os
import re
import sys

import pandas as pd

VALID = ["a", "b", "c", "d"]


def read_csv(path):
    if not os.path.exists(path):
        print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
        sys.exit(1)
    # 빈 값이 NaN 으로 바뀌지 않도록 전부 문자열로 읽는다.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def strip_ext(series):
    return series.str.strip().str.replace(r"(?i)\.(jpe?g|png)$", "", regex=True)


def sample_ids(items, k=5):
    items = sorted(items)
    more = f" 외 {len(items) - k}개" if len(items) > k else ""
    return f"{items[:k]}{more}"


def check(sub, sample, test=None):
    """(errors, warnings) 리스트를 반환한다."""
    errors, warnings = [], []

    # 1. 컬럼
    cols = list(sub.columns)
    if cols != ["id", "answer"]:
        msg = f"컬럼이 ['id', 'answer'] 가 아닙니다: {cols}"
        if cols and (cols[0].startswith("Unnamed") or cols[0] == ""):
            msg += "  → to_csv(index=False) 로 저장했는지 확인하세요."
        errors.append(msg)
        return errors, warnings  # 이후 검사는 컬럼이 맞아야 의미가 있다.

    # 2. 행 수
    if len(sub) != len(sample):
        errors.append(f"행 수가 다릅니다: 제출 {len(sub)}행 vs sample {len(sample)}행")

    # 3. id 검사
    ids = sub["id"]
    stripped = ids.str.strip()
    if (ids != stripped).any():
        errors.append(f"id 앞뒤에 공백이 있는 행이 {(ids != stripped).sum()}개 있습니다.")
    dup = stripped[stripped.duplicated()].unique()
    if len(dup):
        errors.append(f"중복된 id 가 {len(dup)}개 있습니다: {sample_ids(dup)}")

    sub_set, sample_set = set(stripped), set(sample["id"].str.strip())
    missing = sample_set - sub_set
    extra = sub_set - sample_set
    if (missing or extra) and set(strip_ext(sub["id"])) == set(strip_ext(sample["id"])):
        errors.append(
            "id 는 같지만 확장자(.jpg) 표기만 다릅니다. sample_submission 처럼 "
            f"'{sample['id'].iloc[0]}' 형태(확장자 포함)로 제출해야 합니다."
        )
    elif missing:
        errors.append(f"sample 에는 있는데 제출에 없는 id {len(missing)}개: {sample_ids(missing)}")
    if extra and not (set(strip_ext(sub["id"])) == set(strip_ext(sample["id"]))):
        errors.append(f"제출에만 있고 sample 에 없는 id {len(extra)}개: {sample_ids(extra)}")

    if test is not None and "id" in test.columns:
        test_set = set(test["id"].str.strip())
        if test_set != sample_set:
            warnings.append(
                f"test.csv 의 id({len(test_set)}개)와 sample_submission 의 id({len(sample_set)}개)가 "
                "다릅니다. 어느 쪽을 기준으로 제출해야 하는지 확인이 필요합니다."
            )
        t_missing = test_set - sub_set
        if t_missing:
            errors.append(f"test.csv 에는 있는데 제출에 없는 id {len(t_missing)}개: {sample_ids(t_missing)}")

    if not errors and list(stripped) != list(sample["id"].str.strip()):
        warnings.append("행 순서가 sample_submission 과 다릅니다. (내용이 같으면 보통 문제 없음)")

    # 4. 결측
    empty = sub["answer"].str.strip() == ""
    if empty.any():
        bad = sub.loc[empty, "id"].tolist()
        errors.append(f"answer 가 비어 있는 행이 {int(empty.sum())}개 있습니다: {sample_ids(bad)}")

    # 5. 값 형식
    ans = sub["answer"]
    invalid = ~ans.isin(VALID) & ~empty
    if invalid.any():
        shown = ans[invalid].map(repr).value_counts().head(5).to_dict()
        errors.append(
            f"a, b, c, d 이외의 값이 {int(invalid.sum())}개 있습니다. 값별 개수: {shown} "
            "(대문자, 공백, 따옴표, 개행이 섞이지 않았는지 확인)"
        )

    # 6. 분포
    valid_ans = ans[ans.isin(VALID)]
    if len(valid_ans) > 0:
        dist = valid_ans.value_counts(normalize=True).reindex(VALID, fill_value=0)
        if dist.max() > 0.50 or dist.min() < 0.10:
            warnings.append(
                "답 분포가 한쪽으로 쏠려 있습니다: "
                + ", ".join(f"{c}={dist[c]:.1%}" for c in VALID)
                + "  (모델 오류나 위치 편향의 신호일 수 있음)"
            )
    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description="제출 파일 검사")
    ap.add_argument("--submission", required=True, help="검사할 제출 csv")
    ap.add_argument("--sample", required=True, help="sample_submission.csv")
    ap.add_argument("--test", default=None, help="test.csv (선택, id 대조용)")
    args = ap.parse_args()

    sub = read_csv(args.submission)
    sample = read_csv(args.sample)
    test = read_csv(args.test) if args.test else None

    if "id" not in sample.columns:
        print(f"[오류] sample 파일에 id 컬럼이 없습니다: {list(sample.columns)}", file=sys.stderr)
        sys.exit(1)

    errors, warnings = check(sub, sample, test)

    print(f"제출 파일 : {args.submission}  ({len(sub)}행)")
    print(f"기준 파일 : {args.sample}  ({len(sample)}행)")
    print()
    for w in warnings:
        print(f"[경고] {w}")
    for e in errors:
        print(f"[오류] {e}")

    if errors:
        print(f"\n검사 실패: 오류 {len(errors)}건, 경고 {len(warnings)}건")
        sys.exit(1)
    print(f"\n검사 통과 (경고 {len(warnings)}건)")


if __name__ == "__main__":
    main()