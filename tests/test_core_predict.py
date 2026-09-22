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
        # 실제 Scorer와 같은 기준으로 문항마다 쓸 순열을 정한다.
        return {
            "records": list(records),
            "perms": predict_mod.resolve_perms(records, perm),
        }

    def score(self, enc, *, also_base: bool = False):
        records = enc["records"]
        perms = enc["perms"]
        if self.fail_after is not None and self.scored + len(records) > self.fail_after:
            raise RuntimeError("세션이 끊긴 상황을 흉내 냅니다.")
        self.scored += len(records)
        self.batches += 1
        # 화면 j자리는 원래 perm[j]번 보기이므로 확률도 그렇게 늘어놓는다.
        shown = np.stack(
            [original_probs(r.id)[list(p)] for r, p in zip(records, perms)]
        )
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


# --------------------------------------------------------------------------
# 순서 의존 보기 보호
# --------------------------------------------------------------------------


def make_order_dependent_record(index: int = 100) -> VQARecord:
    """'위의 모두'가 들어 있어 선택지를 섞으면 안 되는 문항."""
    return VQARecord(
        id=f"test_{index:04d}.jpg",
        path=f"test/test_{index:04d}.jpg",
        question="다음 중 옳은 것은 무엇입니까?",
        options=["첫째 보기", "둘째 보기", "셋째 보기", "위의 모두"],
        answer_idx=3,
    )


def test_resolve_perms_keeps_order_dependent_options_in_place():
    records = make_records(3) + [make_order_dependent_record()]
    for perm in PERM_BANK:
        resolved = predict_mod.resolve_perms(records, perm)
        assert resolved[:3] == [tuple(perm)] * 3
        assert resolved[3] == predict_mod.IDENTITY_PERM


def test_tta_default_is_unchanged_by_the_order_dependent_guard():
    """tta=1은 원래부터 항등 순열이므로 보호 장치가 동작을 바꾸지 않는다."""
    records = make_records(3) + [make_order_dependent_record()]
    resolved = predict_mod.resolve_perms(records, (0, 1, 2, 3))
    assert resolved == [predict_mod.IDENTITY_PERM] * 4


def test_order_dependent_item_is_scored_in_the_original_order(tmp_path):
    """순서 의존 문항은 어느 순열에서도 원래 순서로 채점하고 그대로 되돌린다."""
    plain = make_records(4)
    special = make_order_dependent_record()
    records = plain + [special]

    for index, perm in enumerate(PERM_BANK):
        out_file = tmp_path / f"preds.perm{index}.csv"
        run_once(FakeScorer(), records, out_file, perm=perm)
        frame, _ = pred_mod.load_partial_predictions(out_file)
        frame = frame.set_index("id")
        for record in records:
            got = frame.loc[record.id, list(pred_mod.PROB_COLUMNS)].to_numpy(dtype=float)
            # 순열을 무엇으로 주든 원래 a~d 순서의 확률이 나와야 한다.
            assert np.allclose(got, original_probs(record.id), atol=1e-6)


def test_order_dependent_item_gets_the_same_prompt_in_every_permutation():
    """순열이 달라도 순서 의존 문항에 보여 주는 선택지는 같아야 한다."""
    special = make_order_dependent_record()
    shown_sets = []
    for perm in PERM_BANK:
        record_perm = predict_mod.resolve_perms([special], perm)[0]
        shown, _ = collator_mod.apply_permutation(
            special.options, special.answer_idx, record_perm
        )
        shown_sets.append(shown)
    assert all(shown == special.options for shown in shown_sets)


def test_invert_probs_per_record_uses_each_records_own_permutation():
    shown = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    perms = [(1, 3, 0, 2), predict_mod.IDENTITY_PERM]
    original = predict_mod.invert_probs_per_record(shown, perms)
    assert np.allclose(original[0], collator_mod.invert_probs(shown[0], (1, 3, 0, 2)))
    assert np.allclose(original[1], shown[1])


# --------------------------------------------------------------------------
# 예측 캐시 manifest 검사
# --------------------------------------------------------------------------


def current_settings() -> dict:
    return {
        "model_id": "Qwen/Qwen3.5-VL-27B",
        "model_revision": "fc05dae",
        "adapter": "runs/r1/final_adapter",
        "prompt_id": "v2_read_text",
        "prompt_hash": "abc123",
        "precision": "bf16",
        "min_pixels": 262144,
        "max_pixels": 1048576,
    }


def test_build_manifest_records_both_pixel_limits():
    manifest = pred_mod.build_manifest(run_id="R1", min_pixels=262144, max_pixels=1048576)
    assert manifest["min_pixels"] == 262144
    assert manifest["max_pixels"] == 1048576


def test_resume_manifest_check_accepts_matching_settings():
    expected = current_settings()
    manifest = pred_mod.build_manifest(run_id="R1", **expected)
    assert predict_mod.check_resume_manifest(manifest, expected, source="p.csv") == []


@pytest.mark.parametrize(
    "key, changed",
    [
        ("model_id", "Qwen/Qwen3.5-VL-9B"),
        ("model_revision", "0000000"),
        ("adapter", "none"),
        ("prompt_id", "v1_image_only"),
        ("prompt_hash", "zzz999"),
        ("precision", "nf4"),
        ("min_pixels", 1048576),
        ("max_pixels", 2097152),
    ],
)
def test_resume_manifest_check_refuses_changed_settings(key, changed):
    expected = current_settings()
    manifest = pred_mod.build_manifest(run_id="R1", **expected)
    manifest[key] = changed
    with pytest.raises(predict_mod.ResumeMismatch, match=key):
        predict_mod.check_resume_manifest(manifest, expected, source="p.csv")


def test_old_manifest_without_min_pixels_only_warns():
    """오늘 낮에 만든 예측 파일에는 min_pixels 키가 없다. 그 항목만 넘어간다."""
    expected = current_settings()
    manifest = pred_mod.build_manifest(run_id="R1", **expected)
    del manifest["min_pixels"]
    warnings = predict_mod.check_resume_manifest(manifest, expected, source="p.csv")
    assert len(warnings) == 1 and "min_pixels" in warnings[0]

    # 키가 없더라도 나머지 항목이 다르면 여전히 멈춘다.
    manifest["precision"] = "nf4"
    with pytest.raises(predict_mod.ResumeMismatch, match="precision"):
        predict_mod.check_resume_manifest(manifest, expected, source="p.csv")


def test_missing_manifest_is_not_an_error():
    assert predict_mod.check_resume_manifest(None, current_settings(), source="p.csv") == []


def test_run_permutation_refuses_to_append_to_a_foreign_prediction_file(tmp_path):
    """이어 쓰려는 파일이 다른 설정으로 만들어졌으면 추론을 시작하지 않는다."""
    records = make_records(6)
    out_file = tmp_path / "preds.csv"
    expected = current_settings()

    run_once(FakeScorer(), records[:3], out_file)
    pred_mod.write_manifest(out_file, pred_mod.build_manifest(run_id="R1", **expected))

    changed = dict(expected, min_pixels=1048576)
    scorer = FakeScorer()
    with pytest.raises(predict_mod.ResumeMismatch, match="min_pixels"):
        predict_mod.run_permutation(
            scorer,
            records,
            (0, 1, 2, 3),
            out_file=out_file,
            base_file=None,
            run_id="R1",
            batch_size=4,
            save_every_items=5,
            also_base=False,
            expected_manifest=changed,
            manifest_paths=[out_file],
        )
    # 멈춘 자리에서 한 건도 추론하지 않았어야 한다.
    assert scorer.scored == 0


def test_run_permutation_continues_when_the_manifest_matches(tmp_path):
    records = make_records(6)
    out_file = tmp_path / "preds.csv"
    expected = current_settings()

    run_once(FakeScorer(), records[:3], out_file)
    pred_mod.write_manifest(out_file, pred_mod.build_manifest(run_id="R1", **expected))

    info = predict_mod.run_permutation(
        FakeScorer(),
        records,
        (0, 1, 2, 3),
        out_file=out_file,
        base_file=None,
        run_id="R1",
        batch_size=4,
        save_every_items=5,
        also_base=False,
        expected_manifest=expected,
        manifest_paths=[out_file],
    )
    assert info["skipped"] == 3 and info["processed"] == 3
    frame, _ = pred_mod.load_partial_predictions(out_file)
    assert len(frame) == 6
