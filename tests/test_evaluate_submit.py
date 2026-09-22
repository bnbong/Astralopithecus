"""예측 형식, 병합, 평가, 결합, 제출 검사기에 대한 pytest 묶음.

GPU나 실제 추론 없이 가짜 확률 파일만으로 돈다. 정상 입력뿐 아니라 일부러 망가뜨린
입력(누락 ID, 중복 ID, NaN, 순서 뒤바뀜, 합이 1이 아닌 행, BOM, 추가 index 컬럼)에서
각 검사기가 실제로 걸러내는지 확인한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_tools import combine as combine_cli  # noqa: E402
from scripts import check_submission as check_submission_cli  # noqa: E402
from scripts import evaluate_run as evaluate_run_cli  # noqa: E402
from scripts import make_fake_predictions as fake_cli  # noqa: E402
from scripts import merge_shards as merge_cli  # noqa: E402
from vqa import submit as submit_mod  # noqa: E402
from vqa.evaluate import (  # noqa: E402
    EvaluationError,
    append_experiment_log,
    classify_question_type,
    evaluate_predictions,
    load_gold,
    read_experiment_log,
)
from vqa.predictions import (  # noqa: E402
    LETTERS,
    PRED_COLUMNS,
    PROB_COLUMNS,
    PredictionError,
    append_predictions,
    atomic_write_bytes,
    average_predictions,
    build_manifest,
    decide_answers,
    load_partial_predictions,
    load_predictions,
    manifest_path_for,
    read_id_list,
    read_manifest,
    write_predictions,
)
from vqa.submit import (  # noqa: E402
    SubmissionError,
    build_submission,
    check_submission,
    load_sample_submission,
    write_submission,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------
# 공용 fixture
# --------------------------------------------------------------------------


def make_ids(n: int, prefix: str = "test") -> list[str]:
    return [f"{prefix}_{i:04d}.jpg" for i in range(1, n + 1)]


def make_frame(ids: list[str], run_id: str = "run1", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(4, 0.4), size=len(ids))
    frame = pd.DataFrame(probs, columns=list(PROB_COLUMNS))
    frame.insert(0, "id", pd.array(ids, dtype="string"))
    frame["run_id"] = pd.array([run_id] * len(ids), dtype="string")
    return frame.loc[:, list(PRED_COLUMNS)]


@pytest.fixture
def ids() -> list[str]:
    return make_ids(40)


@pytest.fixture
def gold_csv(tmp_path: Path, ids: list[str]) -> Path:
    rng = np.random.default_rng(7)
    rows = []
    for index, row_id in enumerate(ids):
        answer = LETTERS[index % 4]
        if index % 5 == 0:
            question = "전화번호는 무엇인가요?"
            options = ["02-123-4567", "031-222-3333", "010-1111-2222", "070-9999-8888"]
        elif index % 5 == 1:
            question = "간판에 적힌 영문 문구는 무엇인가요?"
            options = ["SUNDAY BURGER", "MONDAY SALAD", "FRIDAY PIZZA", "SOUP DAY"]
        elif index % 5 == 2:
            question = "몇 층에 있나요?"
            options = ["3층", "4층", "5층", "6층"]
        else:
            question = "이 가게의 대표 메뉴는 무엇인가요?"
            options = ["직화낙지", "된장찌개", "김치찌개", "비빔밥"]
        rows.append(
            {
                "id": row_id,
                "path": f"train/{row_id}",
                "question": question,
                "a": options[0],
                "b": options[1],
                "c": options[2],
                "d": options[3],
                "answer": answer,
            }
        )
    path = tmp_path / "gold.csv"
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    assert rng is not None
    return path


@pytest.fixture
def sample_submission(tmp_path: Path, ids: list[str]) -> Path:
    path = tmp_path / "sample_submission.csv"
    # 실제 대회 파일처럼 CRLF와 빈 answer를 쓴다.
    text = "id,answer\r\n" + "".join(f"{i},\r\n" for i in ids)
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def pred_csv(tmp_path: Path, ids: list[str]) -> Path:
    path = tmp_path / "pred.csv"
    write_predictions(make_frame(ids), path)
    return path


# --------------------------------------------------------------------------
# vqa.predictions: 형식 검증
# --------------------------------------------------------------------------


def test_round_trip_write_and_load(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids)
    path = write_predictions(frame, tmp_path / "p.csv")
    loaded = load_predictions(path)
    assert list(loaded.columns) == list(PRED_COLUMNS)
    assert loaded["id"].tolist() == ids
    assert loaded["run_id"].unique().tolist() == ["run1"]
    np.testing.assert_allclose(
        loaded[list(PROB_COLUMNS)].to_numpy().sum(axis=1), 1.0, atol=1e-6
    )


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path, ids: list[str]) -> None:
    target = tmp_path / "p.csv"
    write_predictions(make_frame(ids), target)
    write_predictions(make_frame(ids, seed=1), target)
    assert [p.name for p in tmp_path.iterdir()] == ["p.csv"]


def test_atomic_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"


def test_load_rejects_missing_column(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids).drop(columns=["p_d"])
    path = tmp_path / "p.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredictionError, match="필수 컬럼"):
        load_predictions(path)


def test_load_rejects_extra_index_column(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "p.csv"
    make_frame(ids).to_csv(path, index=True)
    with pytest.raises(PredictionError, match="index 컬럼"):
        load_predictions(path)


def test_load_rejects_nan_probability(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids)
    frame.loc[3, "p_b"] = float("nan")
    path = tmp_path / "p.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredictionError, match="NaN"):
        load_predictions(path)


def test_load_rejects_rows_not_summing_to_one(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids)
    frame.loc[5, "p_a"] = frame.loc[5, "p_a"] + 0.5
    path = tmp_path / "p.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredictionError, match="확률 합"):
        load_predictions(path)


def test_load_rejects_duplicate_ids(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "p.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredictionError, match="중복"):
        load_predictions(path)


def test_load_rejects_multiple_run_ids(tmp_path: Path, ids: list[str]) -> None:
    frame = make_frame(ids)
    frame.loc[2, "run_id"] = "other"
    path = tmp_path / "p.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredictionError, match="run_id"):
        load_predictions(path)


def test_load_accepts_bom(tmp_path: Path, ids: list[str]) -> None:
    from vqa.predictions import predictions_to_csv_text

    path = tmp_path / "p.csv"
    atomic_write_bytes(path, b"\xef\xbb\xbf" + predictions_to_csv_text(make_frame(ids)).encode())
    loaded = load_predictions(path)
    assert len(loaded) == len(ids)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="없습니다"):
        load_predictions(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# vqa.predictions: 부분 파일과 이어쓰기
# --------------------------------------------------------------------------


def test_append_then_partial_load(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "shard.csv"
    frame = make_frame(ids[:10])
    written = append_predictions(frame.to_dict("records"), path)
    assert written == 10
    loaded, report = load_partial_predictions(path)
    assert report.kept_rows == 10
    assert report.dropped_rows == 0
    assert loaded["id"].tolist() == ids[:10]


def test_partial_load_drops_truncated_tail(tmp_path: Path, ids: list[str]) -> None:
    from vqa.predictions import predictions_to_csv_text

    path = tmp_path / "shard.csv"
    text = predictions_to_csv_text(make_frame(ids[:10]))
    truncated = text.rstrip("\n")
    truncated = truncated[: -8]  # 마지막 행을 중간에서 끊는다.
    path.write_bytes(truncated.encode("utf-8"))
    loaded, report = load_partial_predictions(path)
    assert report.truncated_tail is True
    assert report.kept_rows == 9


def test_partial_load_keeps_last_value_for_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "shard.csv"
    path.write_text(
        "id,p_a,p_b,p_c,p_d,run_id\n"
        "x1,0.7,0.1,0.1,0.1,r\n"
        "x1,0.1,0.7,0.1,0.1,r\n",
        encoding="utf-8",
    )
    loaded, report = load_partial_predictions(path)
    assert report.duplicate_ids == ["x1"]
    assert len(loaded) == 1
    assert loaded.loc[0, "p_b"] == pytest.approx(0.7)


def test_partial_load_drops_nan_and_bad_sum_rows(tmp_path: Path) -> None:
    path = tmp_path / "shard.csv"
    path.write_text(
        "id,p_a,p_b,p_c,p_d,run_id\n"
        "ok,0.7,0.1,0.1,0.1,r\n"
        "nanrow,nan,0.1,0.1,0.1,r\n"
        "badsum,0.9,0.9,0.1,0.1,r\n",
        encoding="utf-8",
    )
    loaded, report = load_partial_predictions(path)
    assert loaded["id"].tolist() == ["ok"]
    assert len(report.bad_prob_rows) == 2


def test_partial_load_of_missing_file_is_empty(tmp_path: Path) -> None:
    loaded, report = load_partial_predictions(tmp_path / "nope.csv")
    assert len(loaded) == 0
    assert report.total_rows == 0


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def test_manifest_records_csv_hash(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "p.csv"
    write_predictions(
        make_frame(ids),
        path,
        manifest=build_manifest(
            run_id="run1", model_id="m", precision="bf16", seconds_per_item=1.5
        ),
    )
    manifest = read_manifest(path)
    assert manifest is not None
    assert manifest["run_id"] == "run1"
    assert manifest["seconds_per_item"] == 1.5
    assert len(manifest["csv_sha256"]) == 64
    assert manifest_path_for(path).name == "p.csv.manifest.json"


# --------------------------------------------------------------------------
# 답과 margin, 가중 평균
# --------------------------------------------------------------------------


def test_decide_answers_picks_argmax_and_margin() -> None:
    frame = pd.DataFrame(
        {
            "id": pd.array(["x"], dtype="string"),
            "p_a": [0.1],
            "p_b": [0.6],
            "p_c": [0.2],
            "p_d": [0.1],
            "run_id": pd.array(["r"], dtype="string"),
        }
    )
    decided = decide_answers(frame)
    assert decided.loc[0, "answer"] == "b"
    assert decided.loc[0, "margin"] == pytest.approx(0.4)


def test_average_predictions_weights_and_renormalizes() -> None:
    left = pd.DataFrame(
        {
            "id": pd.array(["x"], dtype="string"),
            "p_a": [1.0], "p_b": [0.0], "p_c": [0.0], "p_d": [0.0],
            "run_id": pd.array(["l"], dtype="string"),
        }
    )
    right = left.copy()
    right[["p_a", "p_b"]] = [[0.0, 1.0]]
    combined = average_predictions([left, right], [0.8, 0.2], run_id="c")
    assert combined.loc[0, "p_a"] == pytest.approx(0.8)
    assert combined.loc[0, "p_b"] == pytest.approx(0.2)
    assert combined["run_id"].tolist() == ["c"]


def test_average_predictions_rejects_different_id_sets(ids: list[str]) -> None:
    with pytest.raises(PredictionError, match="id 집합"):
        average_predictions(
            [make_frame(ids), make_frame(ids[:-1])], [0.5, 0.5], run_id="c"
        )


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------


def test_perfect_oracle_scores_one(gold_csv: Path, ids: list[str]) -> None:
    gold = load_gold(gold_csv)
    letter_index = {letter: i for i, letter in enumerate(LETTERS)}
    probs = np.full((len(ids), 4), 0.1)
    for row, answer in enumerate(gold["answer"]):
        probs[row, letter_index[answer]] = 0.7
    frame = pd.DataFrame(probs, columns=list(PROB_COLUMNS))
    frame.insert(0, "id", pd.array(ids, dtype="string"))
    frame["run_id"] = pd.array(["oracle"] * len(ids), dtype="string")

    result = evaluate_predictions(frame.loc[:, list(PRED_COLUMNS)], gold)
    assert result.accuracy == 1.0
    assert result.n == len(ids)
    assert result.mean_margin == pytest.approx(0.6)
    assert sum(result.confusion[g][p] for g in LETTERS for p in LETTERS) == len(ids)
    assert len(result.wrong) == 0
    assert set(result.by_question_type) <= {
        "phone_like", "has_digits", "mostly_english", "korean_options", "other"
    }


def test_evaluate_on_id_subset(gold_csv: Path, ids: list[str]) -> None:
    gold = load_gold(gold_csv)
    result = evaluate_predictions(make_frame(ids), gold, ids=ids[:10])
    assert result.n == 10


def test_evaluate_rejects_unknown_id(gold_csv: Path, ids: list[str]) -> None:
    gold = load_gold(gold_csv)
    with pytest.raises(EvaluationError, match="예측에 없는"):
        evaluate_predictions(make_frame(ids), gold, ids=ids + ["ghost.jpg"])


def test_evaluate_allows_missing_id_when_requested(gold_csv: Path, ids: list[str]) -> None:
    gold = load_gold(gold_csv)
    result = evaluate_predictions(
        make_frame(ids), gold, ids=ids + ["ghost.jpg"], require_full_coverage=False
    )
    assert result.n == len(ids)


def test_wrong_list_is_sorted_by_margin_desc(gold_csv: Path, ids: list[str]) -> None:
    gold = load_gold(gold_csv)
    result = evaluate_predictions(make_frame(ids, seed=3), gold)
    margins = result.wrong["margin"].tolist()
    assert margins == sorted(margins, reverse=True)
    assert set(result.wrong["id"]) <= set(ids)


def test_load_gold_rejects_bad_answer(tmp_path: Path) -> None:
    path = tmp_path / "g.csv"
    path.write_text("id,answer\nx,z\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="a~d"):
        load_gold(path)


def test_load_gold_reads_bom(tmp_path: Path) -> None:
    path = tmp_path / "g.csv"
    path.write_bytes("﻿id,answer\nx,a\n".encode("utf-8"))
    assert load_gold(path)["id"].tolist() == ["x"]


@pytest.mark.parametrize(
    "question,options,expected",
    [
        ("전화번호는?", ["02-123-4567", "가", "나", "다"], "phone_like"),
        ("몇 층인가요?", ["3층", "4층", "5층", "6층"], "has_digits"),
        ("문구는?", ["SUNDAY BURGER", "MONDAY SALAD", "PIZZA", "SOUP"], "mostly_english"),
        ("대표 메뉴는?", ["직화낙지", "된장찌개", "김치찌개", "비빔밥"], "korean_options"),
    ],
)
def test_question_type_heuristic(question, options, expected) -> None:
    assert classify_question_type(question, options) == expected


def test_experiment_log_appends_one_line(tmp_path: Path, gold_csv: Path, ids: list[str], pred_csv: Path) -> None:
    gold = load_gold(gold_csv)
    result = evaluate_predictions(load_predictions(pred_csv), gold)
    log = tmp_path / "experiments.jsonl"
    append_experiment_log(result, log_path=log, prediction_path=pred_csv, seconds_per_item=2.0)
    append_experiment_log(result, log_path=log, prediction_path=pred_csv, note="두 번째")
    entries = read_experiment_log(log)
    assert len(entries) == 2
    assert entries[0]["accuracy"] == pytest.approx(result.accuracy)
    assert entries[0]["seconds_per_item"] == 2.0
    assert len(entries[0]["file_sha256"]) == 64
    assert entries[1]["note"] == "두 번째"


# --------------------------------------------------------------------------
# 제출 생성과 검사
# --------------------------------------------------------------------------


def test_build_submission_follows_sample_order(sample_submission: Path, ids: list[str]) -> None:
    sample = load_sample_submission(sample_submission)
    shuffled = make_frame(ids).sample(frac=1.0, random_state=1).reset_index(drop=True)
    frame = build_submission(shuffled, sample)
    assert frame["id"].tolist() == ids
    assert set(frame["answer"]) <= set(LETTERS)


def test_build_submission_fails_on_missing_id_instead_of_filling_a(
    sample_submission: Path, ids: list[str]
) -> None:
    sample = load_sample_submission(sample_submission)
    partial = make_frame(ids[:-3])
    with pytest.raises(SubmissionError, match="예측에 없는"):
        build_submission(partial, sample)


def test_check_submission_passes_on_good_file(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    sample = load_sample_submission(sample_submission)
    out = tmp_path / "sub.csv"
    write_submission(build_submission(make_frame(ids), sample), out)
    check = check_submission(out, sample_submission)
    assert check.ok, check.errors
    assert check.n_rows == len(ids)
    assert len(check.sha256) == 64
    assert sum(check.distribution.values()) == len(ids)


def test_check_submission_detects_wrong_order(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    sample = load_sample_submission(sample_submission)
    frame = build_submission(make_frame(ids), sample).iloc[::-1].reset_index(drop=True)
    out = tmp_path / "sub.csv"
    write_submission(frame, out)
    check = check_submission(out, sample_submission)
    assert not check.ok
    assert any("행 순서" in e for e in check.errors)


def test_check_submission_detects_missing_and_extra_ids(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    rows = [{"id": i, "answer": "a"} for i in ids[:-1]] + [{"id": "ghost.jpg", "answer": "b"}]
    write_submission(pd.DataFrame(rows), out)
    check = check_submission(out, sample_submission)
    assert not check.ok
    assert any("제출에 없는 id" in e for e in check.errors)
    assert any("sample_submission에 없는 id" in e for e in check.errors)


def test_check_submission_detects_bad_answer_value(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    rows = [{"id": i, "answer": "a"} for i in ids]
    rows[0]["answer"] = "e"
    write_submission(pd.DataFrame(rows), out)
    check = check_submission(out, sample_submission)
    assert any("a~d가 아닌" in e for e in check.errors)


def test_check_submission_detects_nan_answer(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    text = "id,answer\n" + "".join(
        f"{i},{'' if n == 0 else 'a'}\n" for n, i in enumerate(ids)
    )
    out.write_text(text, encoding="utf-8")
    check = check_submission(out, sample_submission)
    assert any("비어 있는" in e for e in check.errors)


def test_check_submission_detects_duplicate_id(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    rows = [{"id": i, "answer": "a"} for i in ids]
    rows[1]["id"] = rows[0]["id"]
    write_submission(pd.DataFrame(rows), out)
    check = check_submission(out, sample_submission)
    assert any("중복" in e for e in check.errors)


def test_check_submission_detects_extra_index_column(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    pd.DataFrame({"id": ids, "answer": ["a"] * len(ids)}).to_csv(out, index=True)
    check = check_submission(out, sample_submission)
    assert any("index 컬럼" in e for e in check.errors)


def test_check_submission_detects_bom(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    text = "id,answer\n" + "".join(f"{i},a\n" for i in ids)
    out.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    check = check_submission(out, sample_submission)
    assert any("BOM" in e for e in check.errors)


def test_check_submission_warns_on_extreme_distribution(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "sub.csv"
    write_submission(pd.DataFrame({"id": ids, "answer": ["a"] * len(ids)}), out)
    check = check_submission(out, sample_submission)
    assert check.ok, check.errors
    assert any("비중이" in w for w in check.warnings)


# --------------------------------------------------------------------------
# CLI: make_fake_predictions
# --------------------------------------------------------------------------


def test_fake_cli_random_single_file(tmp_path: Path, sample_submission: Path) -> None:
    out = tmp_path / "fake.csv"
    code = fake_cli.main(
        ["--ids", str(sample_submission), "--mode", "random", "--run-id", "r1", "--out", str(out)]
    )
    assert code == 0
    loaded = load_predictions(out)
    assert len(loaded) == 40
    assert read_manifest(out)["run_id"] == "r1"


def test_fake_cli_oracle_hits_target_accuracy(
    tmp_path: Path, gold_csv: Path, ids: list[str]
) -> None:
    out = tmp_path / "fake.csv"
    code = fake_cli.main(
        [
            "--ids", str(gold_csv), "--gold", str(gold_csv), "--mode", "oracle",
            "--accuracy", "0.75", "--run-id", "o1", "--out", str(out), "--seed", "5",
        ]
    )
    assert code == 0
    result = evaluate_predictions(load_predictions(out), load_gold(gold_csv))
    assert 0.5 <= result.accuracy <= 0.95


def test_fake_cli_shards_cover_all_ids_exactly_once(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    out = tmp_path / "fake.csv"
    assert fake_cli.main(
        ["--ids", str(sample_submission), "--shards", "5", "--run-id", "s", "--out", str(out)]
    ) == 0
    shards = sorted(tmp_path.glob("fake.shard*of5.csv"))
    assert len(shards) == 5
    seen: list[str] = []
    for shard in shards:
        seen.extend(load_predictions(shard)["id"].tolist())
    assert sorted(seen) == sorted(ids)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("nan", "NaN"),
        ("bad-sum", "확률 합"),
        ("duplicate-id", "중복"),
        ("extra-index", "index 컬럼"),
        ("multi-run-id", "run_id"),
    ],
)
def test_fake_cli_broken_variants_are_rejected_by_loader(
    tmp_path: Path, sample_submission: Path, mode: str, expected: str
) -> None:
    out = tmp_path / f"broken_{mode}.csv"
    assert fake_cli.main(
        ["--ids", str(sample_submission), "--run-id", "b", "--break", mode, "--out", str(out)]
    ) == 0
    with pytest.raises(PredictionError, match=expected):
        load_predictions(out)


def test_fake_cli_missing_id_variant_is_short(tmp_path: Path, sample_submission: Path) -> None:
    out = tmp_path / "broken.csv"
    assert fake_cli.main(
        [
            "--ids", str(sample_submission), "--run-id", "b", "--break", "missing-id",
            "--break-count", "4", "--out", str(out),
        ]
    ) == 0
    assert len(load_predictions(out)) == 36


def test_fake_cli_truncated_variant_loads_as_partial(tmp_path: Path, sample_submission: Path) -> None:
    out = tmp_path / "broken.csv"
    assert fake_cli.main(
        ["--ids", str(sample_submission), "--run-id", "b", "--break", "truncated", "--out", str(out)]
    ) == 0
    _, report = load_partial_predictions(out)
    assert report.truncated_tail is True


# --------------------------------------------------------------------------
# CLI: merge_shards
# --------------------------------------------------------------------------


def _make_shards(tmp_path: Path, ids_path: Path, *, break_mode: str = "none", count: int = 3) -> list[Path]:
    base = tmp_path / "shards" / "preds.csv"
    args = [
        "--ids", str(ids_path), "--run-id", "shardrun", "--shards", str(count),
        "--out", str(base),
    ]
    if break_mode != "none":
        args += ["--break", break_mode, "--break-shard", "2"]
    assert fake_cli.main(args) == 0
    return sorted((tmp_path / "shards").glob("preds.shard*.csv"))


def test_merge_shards_success(tmp_path: Path, sample_submission: Path, ids: list[str]) -> None:
    shards = _make_shards(tmp_path, sample_submission)
    out = tmp_path / "merged.csv"
    code = merge_cli.main(
        [
            "--shards", *[str(s) for s in shards],
            "--expect-ids", str(sample_submission),
            "--out", str(out),
        ]
    )
    assert code == 0
    merged = load_predictions(out)
    assert merged["id"].tolist() == ids  # 기대 ID 목록의 순서를 따른다.
    manifest = read_manifest(out)
    assert manifest["merged_rows"] == len(ids)
    assert manifest["partial"] is False


def test_merge_shards_reports_missing_id_and_exits_nonzero(
    tmp_path: Path, sample_submission: Path
) -> None:
    shards = _make_shards(tmp_path, sample_submission, break_mode="missing-id")
    out = tmp_path / "merged.csv"
    code = merge_cli.main(
        [
            "--shards", *[str(s) for s in shards],
            "--expect-ids", str(sample_submission),
            "--out", str(out),
        ]
    )
    assert code == 1
    assert not out.exists()


def test_merge_shards_allow_partial(tmp_path: Path, sample_submission: Path) -> None:
    shards = _make_shards(tmp_path, sample_submission, break_mode="missing-id")
    out = tmp_path / "merged.csv"
    code = merge_cli.main(
        [
            "--shards", *[str(s) for s in shards],
            "--expect-ids", str(sample_submission),
            "--out", str(out), "--allow-partial",
        ]
    )
    assert code == 0
    assert read_manifest(out)["partial"] is True
    assert len(load_predictions(out)) < 40


def test_merge_shards_duplicate_policy_error(tmp_path: Path, sample_submission: Path) -> None:
    shards = _make_shards(tmp_path, sample_submission, break_mode="duplicate-id")
    out = tmp_path / "merged.csv"
    code = merge_cli.main(
        [
            "--shards", *[str(s) for s in shards],
            "--expect-ids", str(sample_submission),
            "--out", str(out), "--duplicate-policy", "error",
        ]
    )
    assert code == 1


def test_merge_shards_duplicate_policy_last_keeps_one_row(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    shards = _make_shards(tmp_path, sample_submission, break_mode="duplicate-id")
    out = tmp_path / "merged.csv"
    assert merge_cli.main(
        [
            "--shards", *[str(s) for s in shards],
            "--expect-ids", str(sample_submission),
            "--out", str(out),
        ]
    ) == 0
    assert load_predictions(out)["id"].tolist() == ids


def test_merge_shards_detects_unexpected_id(tmp_path: Path, sample_submission: Path) -> None:
    shards = _make_shards(tmp_path, sample_submission)
    extra = tmp_path / "shards" / "preds.shard9of9.csv"
    write_predictions(make_frame(["ghost.jpg"], run_id="shardrun"), extra)
    out = tmp_path / "merged.csv"
    code = merge_cli.main(
        [
            "--shards", *[str(s) for s in shards], str(extra),
            "--expect-ids", str(sample_submission),
            "--out", str(out),
        ]
    )
    assert code == 1


def test_merge_shards_missing_file_exits_two(tmp_path: Path) -> None:
    assert merge_cli.main(
        ["--shards", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "m.csv")]
    ) == 2


# --------------------------------------------------------------------------
# CLI: evaluate_run
# --------------------------------------------------------------------------


def test_evaluate_run_cli_writes_log_and_wrong_list(
    tmp_path: Path, gold_csv: Path, pred_csv: Path, ids: list[str]
) -> None:
    val_ids = tmp_path / "val_ids.txt"
    val_ids.write_text("\n".join(ids[:20]) + "\n", encoding="utf-8")
    log = tmp_path / "experiments.jsonl"
    wrong = tmp_path / "wrong.csv"
    summary = tmp_path / "summary.json"
    code = evaluate_run_cli.main(
        [
            "--pred", str(pred_csv), "--gold", str(gold_csv),
            "--val-ids", str(val_ids), "--log", str(log),
            "--wrong-out", str(wrong), "--json-out", str(summary),
        ]
    )
    assert code == 0
    entries = read_experiment_log(log)
    assert len(entries) == 1
    assert entries[0]["n"] == 20
    assert wrong.exists()
    assert json.loads(summary.read_text(encoding="utf-8"))["n"] == 20


def test_evaluate_run_cli_min_accuracy_gate(
    tmp_path: Path, gold_csv: Path, pred_csv: Path
) -> None:
    code = evaluate_run_cli.main(
        [
            "--pred", str(pred_csv), "--gold", str(gold_csv),
            "--no-log", "--min-accuracy", "0.99",
        ]
    )
    assert code == 3


def test_evaluate_run_cli_rejects_broken_pred(
    tmp_path: Path, gold_csv: Path, sample_submission: Path
) -> None:
    broken = tmp_path / "broken.csv"
    assert fake_cli.main(
        ["--ids", str(sample_submission), "--run-id", "b", "--break", "nan", "--out", str(broken)]
    ) == 0
    code = evaluate_run_cli.main(
        ["--pred", str(broken), "--gold", str(gold_csv), "--no-log"]
    )
    assert code == 2


# --------------------------------------------------------------------------
# CLI: combine
# --------------------------------------------------------------------------


def _write_candidate_files(tmp_path: Path, gold_csv: Path, accuracies: dict[str, float]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for seed, (name, accuracy) in enumerate(accuracies.items()):
        path = tmp_path / f"{name}_val.csv"
        assert fake_cli.main(
            [
                "--ids", str(gold_csv), "--gold", str(gold_csv), "--mode", "oracle",
                "--accuracy", str(accuracy), "--run-id", name, "--out", str(path),
                "--seed", str(seed + 1),
            ]
        ) == 0
        paths[name] = path
    return paths


def test_combine_evaluates_four_registered_candidates(
    tmp_path: Path, gold_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_candidate_files(
        tmp_path, gold_csv, {"base27": 0.4, "ft27": 0.8, "ft9": 0.5}
    )
    report = tmp_path / "report.json"
    code = combine_cli.main(
        [
            "--val-base27", str(paths["base27"]),
            "--val-ft27", str(paths["ft27"]),
            "--val-ft9", str(paths["ft9"]),
            "--gold", str(gold_csv), "--no-log",
            "--report-out", str(report),
        ]
    )
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    evaluated = [c for c in data["candidates"] if c["accuracy"] is not None]
    assert len(evaluated) == 4
    assert {c["candidate"] for c in evaluated} == {
        "base27", "ft27", "ft27_0.8+base27_0.2", "ft27_0.8+ft9_0.2"
    }


def test_combine_skips_candidates_without_files(tmp_path: Path, gold_csv: Path) -> None:
    paths = _write_candidate_files(tmp_path, gold_csv, {"ft27": 0.8})
    report = tmp_path / "report.json"
    assert combine_cli.main(
        [
            "--val-ft27", str(paths["ft27"]), "--gold", str(gold_csv),
            "--no-log", "--report-out", str(report),
        ]
    ) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["chosen"] == "ft27"
    skipped = [c for c in data["candidates"] if c["accuracy"] is None]
    assert len(skipped) == 3


def test_combine_tie_prefers_simpler_candidate(tmp_path: Path, gold_csv: Path) -> None:
    # 같은 파일을 세 이름으로 쓰면 네 후보의 정확도가 모두 같아진다.
    path = tmp_path / "same.csv"
    assert fake_cli.main(
        [
            "--ids", str(gold_csv), "--gold", str(gold_csv), "--mode", "oracle",
            "--accuracy", "0.7", "--run-id", "same", "--out", str(path),
        ]
    ) == 0
    report = tmp_path / "report.json"
    assert combine_cli.main(
        [
            "--val-base27", str(path), "--val-ft27", str(path), "--val-ft9", str(path),
            "--gold", str(gold_csv), "--no-log", "--report-out", str(report),
        ]
    ) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["chosen"] == "base27"  # 동률이면 가장 단순한 후보


def test_combine_applies_chosen_weights_to_test(
    tmp_path: Path, gold_csv: Path, sample_submission: Path
) -> None:
    val = _write_candidate_files(tmp_path, gold_csv, {"base27": 0.3, "ft27": 0.85})
    test_paths: dict[str, Path] = {}
    for name in ("base27", "ft27"):
        path = tmp_path / f"{name}_test.csv"
        assert fake_cli.main(
            ["--ids", str(sample_submission), "--run-id", f"{name}_test", "--out", str(path)]
        ) == 0
        test_paths[name] = path
    out = tmp_path / "combined_test.csv"
    code = combine_cli.main(
        [
            "--val-base27", str(val["base27"]), "--val-ft27", str(val["ft27"]),
            "--test-base27", str(test_paths["base27"]), "--test-ft27", str(test_paths["ft27"]),
            "--gold", str(gold_csv), "--no-log", "--out", str(out),
        ]
    )
    assert code == 0
    combined = load_predictions(out)
    assert len(combined) == 40
    manifest = read_manifest(out)
    assert manifest["candidate"] in {"ft27", "ft27_0.8+base27_0.2"}


def test_combine_fails_when_test_ids_differ(
    tmp_path: Path, gold_csv: Path, sample_submission: Path
) -> None:
    val = _write_candidate_files(tmp_path, gold_csv, {"base27": 0.3, "ft27": 0.85})
    full = tmp_path / "ft27_test.csv"
    short = tmp_path / "base27_test.csv"
    assert fake_cli.main(["--ids", str(sample_submission), "--run-id", "t1", "--out", str(full)]) == 0
    assert fake_cli.main(
        ["--ids", str(sample_submission), "--run-id", "t2", "--break", "missing-id", "--out", str(short)]
    ) == 0
    code = combine_cli.main(
        [
            "--val-base27", str(val["base27"]), "--val-ft27", str(val["ft27"]),
            "--test-base27", str(short), "--test-ft27", str(full),
            "--gold", str(gold_csv), "--no-log", "--out", str(tmp_path / "c.csv"),
            "--force-candidate", "ft27_0.8+base27_0.2",
        ]
    )
    assert code == 2


# --------------------------------------------------------------------------
# CLI: vqa.submit / check_submission
# --------------------------------------------------------------------------


def test_submit_cli_end_to_end(
    tmp_path: Path, sample_submission: Path, pred_csv: Path, ids: list[str]
) -> None:
    out = tmp_path / "submission.csv"
    margins = tmp_path / "margins.csv"
    code = submit_mod.main(
        [
            "--pred", str(pred_csv), "--sample", str(sample_submission),
            "--out", str(out), "--margins-out", str(margins),
        ]
    )
    assert code == 0
    assert out.read_bytes()[:3] != b"\xef\xbb\xbf"
    frame = pd.read_csv(out, dtype="string")
    assert frame["id"].tolist() == ids
    assert margins.exists()
    assert check_submission_cli.main(
        ["--submission", str(out), "--sample", str(sample_submission)]
    ) == 0


def test_submit_cli_fails_on_incomplete_predictions(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    partial = tmp_path / "partial.csv"
    write_predictions(make_frame(ids[:30]), partial)
    code = submit_mod.main(
        ["--pred", str(partial), "--sample", str(sample_submission), "--out", str(tmp_path / "s.csv")]
    )
    assert code == 2


def test_check_submission_cli_exit_codes(
    tmp_path: Path, sample_submission: Path, ids: list[str]
) -> None:
    good = tmp_path / "good.csv"
    sample = load_sample_submission(sample_submission)
    write_submission(build_submission(make_frame(ids), sample), good)
    assert check_submission_cli.main(
        ["--submission", str(good), "--sample", str(sample_submission)]
    ) == 0

    bad = tmp_path / "bad.csv"
    write_submission(pd.DataFrame({"id": ids[::-1], "answer": ["a"] * len(ids)}), bad)
    assert check_submission_cli.main(
        ["--submission", str(bad), "--sample", str(sample_submission)]
    ) == 1

    assert check_submission_cli.main(
        ["--submission", str(tmp_path / "nope.csv"), "--sample", str(sample_submission)]
    ) == 2

    skewed = tmp_path / "skewed.csv"
    write_submission(pd.DataFrame({"id": ids, "answer": ["a"] * len(ids)}), skewed)
    assert check_submission_cli.main(
        ["--submission", str(skewed), "--sample", str(sample_submission), "--strict-warnings"]
    ) == 4


# --------------------------------------------------------------------------
# CLI --help 가 모든 스크립트에서 동작하는지
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        ["scripts/make_fake_predictions.py", "--help"],
        ["scripts/merge_shards.py", "--help"],
        ["scripts/evaluate_run.py", "--help"],
        ["experiments/eval_tools/combine.py", "--help"],
        ["scripts/check_submission.py", "--help"],
        ["-m", "vqa.submit", "--help"],
    ],
)
def test_cli_help_works(command: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_read_id_list_from_csv_and_txt(tmp_path: Path, sample_submission: Path, ids: list[str]) -> None:
    assert read_id_list(sample_submission) == ids
    txt = tmp_path / "ids.txt"
    txt.write_text("\n".join(ids) + "\n", encoding="utf-8")
    assert read_id_list(txt) == ids


# --------------------------------------------------------------------------
# 팀원 형식(prob_a~prob_d)과의 상호 변환
# --------------------------------------------------------------------------


def write_teammate_csv(path: Path, ids: list[str], *, with_run_id: bool = False) -> pd.DataFrame:
    """infer_val.py가 남기는 형식(id, pred, prob_a~prob_d)으로 파일을 쓴다."""
    frame = make_frame(ids, run_id="teamrun", seed=3)
    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    out = pd.DataFrame({"id": ids})
    out["pred"] = [LETTERS[i] for i in probs.argmax(axis=1)]
    for index, letter in enumerate(LETTERS):
        out[f"prob_{letter}"] = probs[:, index].round(6)
    if with_run_id:
        out["run_id"] = "teamrun"
    out.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return out


def test_load_predictions_reads_a_teammate_file(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "preds_teammate.csv"
    written = write_teammate_csv(path, ids)

    loaded = load_predictions(path)
    assert list(loaded.columns) == list(PRED_COLUMNS)
    assert loaded["id"].tolist() == ids
    # run_id가 없는 파일이므로 기본값으로 채워진다.
    assert loaded["run_id"].unique().tolist() == ["unknown"]
    np.testing.assert_allclose(
        loaded[list(PROB_COLUMNS)].to_numpy(dtype=float),
        written[[f"prob_{c}" for c in LETTERS]].to_numpy(dtype=float),
        atol=1e-9,
    )


def test_load_predictions_keeps_the_run_id_a_teammate_file_already_has(
    tmp_path: Path, ids: list[str]
) -> None:
    path = tmp_path / "preds_teammate_runid.csv"
    write_teammate_csv(path, ids, with_run_id=True)
    loaded = load_predictions(path, default_run_id="ignored")
    assert loaded["run_id"].unique().tolist() == ["teamrun"]


def test_load_predictions_default_run_id_is_configurable(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "preds_teammate2.csv"
    write_teammate_csv(path, ids)
    assert load_predictions(path, default_run_id="qwen3b_p2")["run_id"].unique().tolist() == [
        "qwen3b_p2"
    ]


def test_evaluate_run_reads_a_teammate_file(tmp_path: Path, gold_csv: Path, ids: list[str]) -> None:
    """scripts/evaluate_run.py가 팀원 파일을 그대로 읽을 수 있어야 한다."""
    path = tmp_path / "preds_teammate_eval.csv"
    write_teammate_csv(path, ids)
    result = evaluate_predictions(load_predictions(path), load_gold(gold_csv))
    assert result.n == len(ids)
    assert result.run_id == "unknown"


def test_convert_preds_round_trip(tmp_path: Path, ids: list[str]) -> None:
    from scripts import convert_preds as convert_cli

    original = tmp_path / "vqa.csv"
    frame = make_frame(ids, run_id="run1", seed=11)
    write_predictions(frame, original)

    teammate = tmp_path / "teammate.csv"
    assert convert_cli.main(["--to", "teammate", "--in", str(original), "--out", str(teammate)]) == 0
    team_frame = pd.read_csv(teammate, encoding="utf-8-sig")
    assert list(team_frame.columns) == ["id", "pred", "prob_a", "prob_b", "prob_c", "prob_d"]
    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    assert team_frame["pred"].tolist() == [LETTERS[i] for i in probs.argmax(axis=1)]

    back = tmp_path / "back.csv"
    assert (
        convert_cli.main(
            ["--to", "vqa", "--in", str(teammate), "--out", str(back), "--run-id", "run1"]
        )
        == 0
    )
    restored = load_predictions(back)
    assert restored["id"].tolist() == ids
    assert restored["run_id"].unique().tolist() == ["run1"]
    np.testing.assert_allclose(
        restored[list(PROB_COLUMNS)].to_numpy(dtype=float), probs, atol=1e-5
    )


def test_convert_preds_to_vqa_renormalises_within_tolerance(tmp_path: Path) -> None:
    from scripts import convert_preds as convert_cli

    source = tmp_path / "rounded.csv"
    source.write_text(
        "id,pred,prob_a,prob_b,prob_c,prob_d\n"
        "test_0001.jpg,a,0.5001,0.2,0.2,0.1\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    assert convert_cli.main(["--to", "vqa", "--in", str(source), "--out", str(out), "--run-id", "r"]) == 0
    loaded = load_predictions(out)
    assert float(loaded[list(PROB_COLUMNS)].to_numpy(dtype=float).sum()) == pytest.approx(1.0)


def test_convert_preds_to_vqa_refuses_rows_that_do_not_sum_to_one(tmp_path: Path) -> None:
    from scripts import convert_preds as convert_cli

    source = tmp_path / "broken.csv"
    source.write_text(
        "id,pred,prob_a,prob_b,prob_c,prob_d\n"
        "test_0001.jpg,a,0.9,0.9,0.9,0.9\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    assert convert_cli.main(["--to", "vqa", "--in", str(source), "--out", str(out), "--run-id", "r"]) == 2
    assert not out.exists()
