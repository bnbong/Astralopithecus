"""두 예측 파일의 문항 단위 비교와 McNemar 검정을 확인한다.

여기서 지키려는 것은 두 가지다. 첫째로 짝지은 분할표의 네 칸이 정확히 세어져야 한다.
둘째로 p값이 양측 정확 이항 검정의 정의와 맞아야 한다. 정확도 차이만 보고 학습이
효과가 있었다고 판단하지 않기 위해서다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

from scripts import compare_predictions as cmp_mod  # noqa: E402

LETTERS = "abcd"


def write_gold(path: Path, answers: dict[str, str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "question", "a", "b", "c", "d", "answer"])
        for item_id, letter in answers.items():
            writer.writerow([item_id, "질문입니다", "가", "나", "다", "라", letter])
    return path


def write_pred(path: Path, answers: dict[str, str], *, run_id: str = "test") -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "p_a", "p_b", "p_c", "p_d", "run_id"])
        for item_id, letter in answers.items():
            probs = [0.05, 0.05, 0.05, 0.05]
            probs[LETTERS.index(letter)] = 0.85
            writer.writerow([item_id] + ["%.6g" % value for value in probs] + [run_id])
    return path


# --------------------------------------------------------------------------
# McNemar 검정 자체
# --------------------------------------------------------------------------


def test_exact_p_value_matches_the_binomial_definition():
    # b=1, c=9 이면 양측 p는 2 * (C(10,0) + C(10,1)) / 2^10 이다.
    assert cmp_mod.mcnemar_exact_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)
    # 완전히 대칭이면 p는 1로 잘린다.
    assert cmp_mod.mcnemar_exact_p(5, 5) == 1.0
    # 불일치가 없으면 비교할 것이 없으므로 1이다.
    assert cmp_mod.mcnemar_exact_p(0, 0) == 1.0


def test_exact_p_value_is_symmetric():
    assert cmp_mod.mcnemar_exact_p(3, 12) == cmp_mod.mcnemar_exact_p(12, 3)


def test_continuity_corrected_statistic():
    assert cmp_mod.mcnemar_statistic(1, 9) == pytest.approx((8 - 1) ** 2 / 10)
    assert cmp_mod.mcnemar_statistic(0, 0) == 0.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def fixture_files(tmp_path):
    """20문항 중에서 B만 맞힌 문항이 6개, A만 맞힌 문항이 1개인 상황을 만든다."""
    ids = ["train_%04d.jpg" % i for i in range(1, 21)]
    gold = {item_id: "a" for item_id in ids}
    pred_a = dict(gold)
    pred_b = dict(gold)
    for item_id in ids[:6]:
        pred_a[item_id] = "b"  # B만 맞힌 문항
    pred_b[ids[6]] = "c"  # A만 맞힌 문항
    for item_id in ids[7:9]:
        pred_a[item_id] = "b"  # 둘 다 틀린 문항
        pred_b[item_id] = "d"
    write_gold(tmp_path / "gold.csv", gold)
    write_pred(tmp_path / "a.csv", pred_a, run_id="base")
    write_pred(tmp_path / "b.csv", pred_b, run_id="ft")
    (tmp_path / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return tmp_path


def run_cli(tmp_path, *extra):
    argv = [
        "--pred-a", str(tmp_path / "a.csv"),
        "--pred-b", str(tmp_path / "b.csv"),
        "--gold", str(tmp_path / "gold.csv"),
        "--val-ids", str(tmp_path / "ids.txt"),
        "--json-out", str(tmp_path / "out.json"),
    ]
    code = cmp_mod.main(argv + list(extra))
    summary = None
    if (tmp_path / "out.json").exists():
        summary = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    return code, summary


def test_contingency_table_counts_each_cell(fixture_files):
    code, summary = run_cli(fixture_files)
    assert code == cmp_mod.EXIT_OK
    assert summary["n"] == 20
    assert summary["only_b_correct"] == 6
    assert summary["only_a_correct"] == 1
    assert summary["both_wrong"] == 2
    assert summary["both_correct"] == 11
    assert summary["accuracy_a"] == pytest.approx(12 / 20)
    assert summary["accuracy_b"] == pytest.approx(17 / 20)
    assert summary["accuracy_delta"] == pytest.approx(5 / 20)
    assert summary["mcnemar_p_value"] == pytest.approx(cmp_mod.mcnemar_exact_p(1, 6))


def test_changed_items_are_written_with_a_direction(fixture_files):
    out = fixture_files / "changed.csv"
    code, _ = run_cli(fixture_files, "--changed-out", str(out), "--label-a", "base",
                      "--label-b", "ft")
    assert code == cmp_mod.EXIT_OK
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 7
    improved = [r for r in rows if "개선" in r["direction"]]
    worsened = [r for r in rows if "악화" in r["direction"]]
    assert len(improved) == 6
    assert len(worsened) == 1


def test_missing_id_is_an_input_error(fixture_files, capsys):
    ids_path = fixture_files / "ids.txt"
    ids_path.write_text(ids_path.read_text(encoding="utf-8") + "train_9999.jpg\n", encoding="utf-8")
    code, _ = run_cli(fixture_files)
    assert code == cmp_mod.EXIT_INPUT_ERROR
    assert "train_9999.jpg" in capsys.readouterr().err


def test_missing_file_is_an_input_error(tmp_path, capsys):
    code = cmp_mod.main([
        "--pred-a", str(tmp_path / "없는파일.csv"),
        "--pred-b", str(tmp_path / "없는파일.csv"),
        "--gold", str(tmp_path / "없는파일.csv"),
    ])
    assert code == cmp_mod.EXIT_INPUT_ERROR
    assert "[오류]" in capsys.readouterr().err


def test_identical_predictions_report_no_difference(tmp_path):
    ids = ["train_%04d.jpg" % i for i in range(1, 11)]
    gold = {item_id: "b" for item_id in ids}
    write_gold(tmp_path / "gold.csv", gold)
    write_pred(tmp_path / "a.csv", gold)
    write_pred(tmp_path / "b.csv", gold)
    (tmp_path / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    code, summary = run_cli(tmp_path)
    assert code == cmp_mod.EXIT_OK
    assert summary["only_a_correct"] == 0 and summary["only_b_correct"] == 0
    assert summary["mcnemar_p_value"] == 1.0
