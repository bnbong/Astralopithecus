import pandas as pd

p = pd.read_csv("preds_test_qwen3b_p2.csv", dtype={"id": str, "pred": str})
s = pd.read_csv("data/sample_submission.csv", dtype=str)

print("예측", len(p), "| id 중복", p.id.duplicated().sum(), "| sample", len(s))

m = s[["id"]].merge(p[["id", "pred"]], on="id", how="left").rename(columns={"pred": "answer"})
print("빈 답", m.answer.isna().sum())
print(m.answer.value_counts(normalize=True).round(3).to_dict())

m.to_csv("qwen3bp2_submission.csv", index=False)