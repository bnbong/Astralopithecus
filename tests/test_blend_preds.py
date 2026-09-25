"""확률 결합 스크립트가 두 파일과 세 파일을 모두 올바르게 평균하는지 확인한다.

여기서 지키려는 것은 두 가지다. 첫째로 --c 를 붙여 세 번째 파일을 넣더라도 기존 두 파일
결합의 결과와 출력이 달라지지 않아야 한다. 둘째로 세 파일 결합은 가중치를 검증에서 고르지
않는다는 원칙에 따라 균등 가중치(각 1/3)로만 동작해야 한다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

from scripts import blend_preds  # noqa: E402

LETTERS = "abcd"


def write_probs(path: Path, rows: dict[str, list[float]], *, prefix: str = "p_") -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id"] + [prefix + letter for letter in LETTERS])
        for item_id, probs in rows.items():
            writer.writerow([item_id] + ["%.6g" % value for value in probs])
    return path


def write_sample(path: Path, ids: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "answer"])
        for item_id in ids:
            writer.writerow([item_id, "a"])
    return path


def read_answers(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {row["id"]: row["answer"] for row in csv.DictReader(fh)}


def run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["blend_preds.py"] + argv)
    blend_preds.main()


# --------------------------------------------------------------------------
# 두 파일 결합 (기존 동작)
# --------------------------------------------------------------------------


def test_two_files_average_the_probabilities_evenly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A 는 a 를, B 는 b 를 고르지만 B 가 더 확신하므로 균등 평균은 b 가 된다.
    a = write_probs(tmp_path / "a.csv", {"t1": [0.55, 0.45, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.05, 0.95, 0.0, 0.0]})
    sample = write_sample(tmp_path / "sample.csv", ["t1"])
    out = tmp_path / "sub.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--sample", str(sample), "--submission", str(out)])
    assert read_answers(out) == {"t1": "b"}


def test_two_files_respect_an_explicit_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A 에 0.9 를 주면 A 의 선택인 a 가 살아남는다.
    a = write_probs(tmp_path / "a.csv", {"t1": [0.55, 0.45, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.05, 0.95, 0.0, 0.0]})
    sample = write_sample(tmp_path / "sample.csv", ["t1"])
    out = tmp_path / "sub.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--w-a", "0.9",
                      "--sample", str(sample), "--submission", str(out)])
    assert read_answers(out) == {"t1": "a"}


# --------------------------------------------------------------------------
# 세 파일 균등 결합
# --------------------------------------------------------------------------


def test_three_files_are_averaged_with_equal_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 세 확률의 평균은 a 가 (0.6+0.6+0.0)/3=0.40, b 가 (0.2+0.2+0.9)/3≈0.433 이므로 b 가 이긴다.
    # 두 파일만 결합하면 a 가 이기므로, 세 번째 파일이 실제로 평균에 들어갔는지 구분할 수 있다.
    a = write_probs(tmp_path / "a.csv", {"t1": [0.6, 0.2, 0.1, 0.1]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.6, 0.2, 0.1, 0.1]})
    c = write_probs(tmp_path / "c.csv", {"t1": [0.0, 0.9, 0.05, 0.05]}, prefix="prob_")
    sample = write_sample(tmp_path / "sample.csv", ["t1"])
    two, three = tmp_path / "two.csv", tmp_path / "three.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--sample", str(sample), "--submission", str(two)])
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c),
                      "--sample", str(sample), "--submission", str(three)])
    assert read_answers(two) == {"t1": "a"}
    assert read_answers(three) == {"t1": "b"}


def test_three_file_blend_writes_the_averaged_probabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    c = write_probs(tmp_path / "c.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    probs = tmp_path / "probs.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c), "--out-probs", str(probs)])
    with probs.open(encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    assert row["pred"] == "b"
    assert float(row["prob_a"]) == pytest.approx(1 / 3, abs=1e-5)
    assert float(row["prob_b"]) == pytest.approx(2 / 3, abs=1e-5)


def test_third_file_weight_moves_the_rest_to_a_and_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # --w-c 0.4 이면 A 와 B 가 각각 0.3 이다. a 는 0.3+0.3=0.60, b 는 0.4 이므로 a 가 이긴다.
    # 균등 가중치(각 1/3)였다면 a 가 2/3*0.5=0.333, b 가 1/3 이므로 이번에도 a 이지만,
    # 확률 값 자체로 두 경우를 구분할 수 있다.
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    c = write_probs(tmp_path / "c.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    probs = tmp_path / "probs.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c), "--w-c", "0.4", "--out-probs", str(probs)])
    with probs.open(encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    assert row["pred"] == "a"
    assert float(row["prob_a"]) == pytest.approx(0.6, abs=1e-5)
    assert float(row["prob_b"]) == pytest.approx(0.4, abs=1e-5)


def test_third_file_weight_can_flip_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # C 의 가중치를 0.6 으로 올리면 A 와 B 가 각각 0.2 가 되어 C 의 선택인 b 가 이긴다.
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    c = write_probs(tmp_path / "c.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    probs = tmp_path / "probs.csv"
    run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c), "--w-c", "0.6", "--out-probs", str(probs)])
    with probs.open(encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    assert row["pred"] == "b"


def test_third_file_weight_needs_a_third_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--a", str(a), "--b", str(b), "--w-c", "0.4"])


def test_three_files_reject_an_explicit_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [0.0, 1.0, 0.0, 0.0]})
    c = write_probs(tmp_path / "c.csv", {"t1": [0.0, 0.0, 1.0, 0.0]})
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c), "--w-a", "0.5"])


def test_missing_id_in_the_third_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = write_probs(tmp_path / "a.csv", {"t1": [1.0, 0.0, 0.0, 0.0], "t2": [1.0, 0.0, 0.0, 0.0]})
    b = write_probs(tmp_path / "b.csv", {"t1": [1.0, 0.0, 0.0, 0.0], "t2": [1.0, 0.0, 0.0, 0.0]})
    c = write_probs(tmp_path / "c.csv", {"t1": [1.0, 0.0, 0.0, 0.0]})
    sample = write_sample(tmp_path / "sample.csv", ["t1", "t2"])
    out = tmp_path / "sub.csv"
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--a", str(a), "--b", str(b), "--c", str(c),
                          "--sample", str(sample), "--submission", str(out)])
