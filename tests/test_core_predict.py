"""추론 경로 테스트: shard 분할, 부분 저장 재개, TTA 평균.

모델을 쓰지 않는다. 확률을 결정적으로 돌려주는 가짜 scorer를 끼워 넣어 파일 입출력과
순열 역매핑만 검증한다.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from vqa import collator as collator_mod  # noqa: E402
from vqa import predict as predict_mod  # noqa: E402
from vqa import predictions as pred_mod  # noqa: E402
from vqa.collator import PERM_BANK, VQARecord  # noqa: E402


def make_records(n: int = 37) -> list[VQARecord]:
    return [
        VQARecord(
            id=f"test_{i:04d}.jpg",
            path=f"test/test_{i:04d}.jpg",
            question=f"질문 {i}" + "?" * (i % 7),
            options=[f"보기{i}_{j}" * (1 + j % 3) for j in range(4)],
            answer_idx=i % 4,
        )
        for i in range(n)
    ]


def original_probs(sample_id: str) -> np.ndarray:
    """id만으로 정해지는 원래 a~d 확률."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    raw = np.array([digest[i] + 1 for i in range(4)], dtype=float)
    return raw / raw.sum()


class FakeScorer:
    """encode/score만 흉내 내는 가짜 추론기."""

    def __init__(self, *, has_adapter: bool = False, fail_after: int | None = None) -> None:
        self.has_adapter = has_adapter
        self.fail_after = fail_after
        self.scored = 0
        self.batches = 0

    def encode(self, records, perm):
        return {"records": list(records), "perm": tuple(perm)}

    def score(self, enc, *, also_base: bool = False):
        records = enc["records"]
        perm = enc["perm"]
        if self.fail_after is not None and self.scored + len(records) > self.fail_after:
            raise RuntimeError("세션이 끊긴 상황을 흉내 냅니다.")
        self.scored += len(records)
        self.batches += 1
        # 화면 j자리는 원래 perm[j]번 보기이므로 확률도 그렇게 늘어놓는다.
        shown = np.stack([original_probs(r.id)[list(perm)] for r in records])
        base = None
        if also_base and self.has_adapter:
            base = np.stack([np.full(4, 0.25) for _ in records])
        return shown, base


# --------------------------------------------------------------------------
# (d) shard 분할
# --------------------------------------------------------------------------


def test_shard_covers_every_id_exactly_once():
    records = make_records(37)
    for count in (2, 3, 5, 8):
        seen: list[str] = []
        for index in range(count):
            part = predict_mod.select_records(records, shard=f"{index}/{count}")
            seen.extend(r.id for r in part)
        assert sorted(seen) == sorted(r.id for r in records)
        assert len(seen) == len(set(seen))


def test_shard_is_stable_under_ids_file_filter(tmp_path):
    """id 목록으로 걸러도 같은 행은 같은 shard에 남는다."""
    records = make_records(37)
    subset = [r.id for r in records[::3]]
    ids_file = tmp_path / "val_ids.txt"
    ids_file.write_text("\n".join(subset), encoding="utf-8")

    full_assignment = {}
    for index in range(5):
        for record in predict_mod.select_records(records, shard=f"{index}/5"):
            full_assignment[record.id] = index

    for index in range(5):
        part = predict_mod.select_records(records, ids_file=str(ids_file), shard=f"{index}/5")
        for record in part:
            assert full_assignment[record.id] == index


def test_parse_shard_rejects_bad_input():
    assert predict_mod.parse_shard("0/5") == (0, 5)
    for bad in ("5/5", "-1/3", "3", "7/2"):
        with pytest.raises(ValueError):
            predict_mod.parse_shard(bad)


def test_output_path_naming(tmp_path):
    assert predict_mod.output_path(tmp_path, "test_ft").name == "test_ft.csv"
    assert predict_mod.output_path(tmp_path, "test_ft.csv", shard="2/5").name == "test_ft.shard2of5.csv"
    assert predict_mod.output_path(tmp_path, "test_ft", perm_index=3).name == "test_ft.perm3.csv"
    assert predict_mod.output_path(tmp_path, "test_ft", base=True).name == "test_ft.base.csv"


# --------------------------------------------------------------------------
# (d) 부분 저장과 재개
# --------------------------------------------------------------------------


def run_once(scorer, records, out_file, *, perm=(0, 1, 2, 3), batch_size=4, save_every=5, base_file=None, also_base=False):
    return predict_mod.run_permutation(
        scorer,
        records,
        perm,
        out_file=out_file,
        base_file=base_file,
        run_id="R1",
        batch_size=batch_size,
        save_every_items=save_every,
        also_base=also_base,
    )


def test_partial_prediction_resume_skips_done_ids(tmp_path):
    records = make_records(20)
    out_file = tmp_path / "preds.csv"

    crashed = FakeScorer(fail_after=8)
    with pytest.raises(RuntimeError):
        run_once(crashed, records, out_file)

    partial, _ = pred_mod.load_partial_predictions(out_file)
    done_before = set(partial["id"].tolist())
    assert 0 < len(done_before) < len(records)

    resumed = FakeScorer()
    info = run_once(resumed, records, out_file)
    assert info["skipped"] == len(done_before)
    assert info["processed"] == len(records) - len(done_before)

    final, _ = pred_mod.load_partial_predictions(out_file)
    assert sorted(final["id"].tolist()) == sorted(r.id for r in records)
    assert len(final) == len(records)

    # 값이 원래 a~d 순서로 복원되어 있어야 한다.
    for row in final.to_dict("records"):
        expected = original_probs(row["id"])
        got = np.array([row["p_a"], row["p_b"], row["p_c"], row["p_d"]])
        assert np.allclose(got, expected, atol=1e-5)


def test_rerun_after_completion_does_nothing(tmp_path):
    records = make_records(10)
    out_file = tmp_path / "preds.csv"
    run_once(FakeScorer(), records, out_file)
    again = FakeScorer()
    info = run_once(again, records, out_file)
    assert info["processed"] == 0
    assert again.scored == 0


def test_probs_are_written_in_original_order_for_each_permutation(tmp_path):
    records = make_records(12)
    for k, perm in enumerate(PERM_BANK):
        out_file = tmp_path / f"preds.perm{k}.csv"
        run_once(FakeScorer(), records, out_file, perm=perm)
        frame, _ = pred_mod.load_partial_predictions(out_file)
        frame = frame.set_index("id")
        for record in records:
            got = frame.loc[record.id, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
            assert np.allclose(got, original_probs(record.id), atol=1e-5)


# --------------------------------------------------------------------------
# (e) TTA 평균
# --------------------------------------------------------------------------


def test_tta_average_matches_single_view_when_model_is_consistent(tmp_path):
    records = make_records(15)
    files = []
    for k, perm in enumerate(PERM_BANK):
        out_file = tmp_path / f"preds.perm{k}.csv"
        run_once(FakeScorer(), records, out_file, perm=perm)
        files.append(out_file)

    order_ids = [r.id for r in records]
    averaged = predict_mod.average_perm_files(files, order_ids, "R1")
    assert averaged["id"].tolist() == order_ids
    for row in averaged.to_dict("records"):
        expected = original_probs(row["id"])
        got = np.array([row["p_a"], row["p_b"], row["p_c"], row["p_d"]])
        assert np.allclose(got, expected, atol=1e-5)


def test_tta_average_rejects_missing_ids(tmp_path):
    records = make_records(10)
    full = tmp_path / "full.csv"
    partial = tmp_path / "partial.csv"
    run_once(FakeScorer(), records, full)
    run_once(FakeScorer(), records[:5], partial)
    with pytest.raises(ValueError):
        predict_mod.average_perm_files([full, partial], [r.id for r in records], "R1")


def test_output_is_keyed_by_id_not_by_processing_order(tmp_path):
    """길이 정렬로 처리 순서가 바뀌어도 최종 파일은 입력 순서를 따른다."""
    records = make_records(20)
    out_file = tmp_path / "preds.csv"
    scorer = FakeScorer()
    run_once(scorer, records, out_file, batch_size=3)

    order_ids = [r.id for r in records]
    frame = predict_mod.frame_from_file(out_file, order_ids, "R1")
    assert frame["id"].tolist() == order_ids

    written, _ = pred_mod.load_partial_predictions(out_file)
    # 저장 순서는 길이 정렬을 따르므로 입력 순서와 달라야 정렬이 실제로 작동한 것이다.
    assert written["id"].tolist() != order_ids


def test_also_base_writes_second_file(tmp_path):
    records = make_records(8)
    out_file = tmp_path / "preds.csv"
    base_file = tmp_path / "preds.base.csv"
    scorer = FakeScorer(has_adapter=True)
    run_once(scorer, records, out_file, base_file=base_file, also_base=True)
    assert out_file.exists() and base_file.exists()
    base, _ = pred_mod.load_partial_predictions(base_file)
    assert len(base) == len(records)
    assert np.allclose(base[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float), 0.25)
