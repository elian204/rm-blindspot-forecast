"""Audit statistics, checked on small hand-computed examples (no models involved)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from blindspots.audit import (
    AnswerRecord,
    add_advantages,
    auc,
    blind_spot_candidates,
    format_summary_table,
    group_advantages,
    run_audit,
    within_problem_auc,
)
from blindspots.io import read_jsonl, write_jsonl


def rec(pid: str, idx: int, grade: float, correct: bool, **kwargs) -> AnswerRecord:
    status = kwargs.pop("status", "correct" if correct else "wrong")
    return AnswerRecord(
        problem_id=pid,
        sample_idx=idx,
        response=kwargs.pop("response", f"response {pid}/{idx}"),
        correct=correct,
        rm_grade=grade,
        status=status,
        **kwargs,
    )


def example_records() -> list[AnswerRecord]:
    """Four problems: mixed, all correct, all wrong with equal grades, RM prefers wrong."""
    a = [(0.9, True), (0.8, False), (0.3, True), (0.1, False)]
    b = [(0.5, True), (0.6, True), (0.7, True)]
    c = [(0.2, False), (0.2, False), (0.2, False)]
    d = [(0.7, False), (0.6, False), (0.4, True), (0.5, False)]
    out = []
    for pid, rows in (("A", a), ("B", b), ("C", c), ("D", d)):
        out += [rec(pid, i, g, ok) for i, (g, ok) in enumerate(rows)]
    return out


# --- group advantages ---


def test_group_advantages_use_sample_std() -> None:
    assert group_advantages([1.0, 2.0, 3.0]) == pytest.approx([-1.0, 0.0, 1.0])


def test_group_advantages_population_std_option() -> None:
    expected = [-1 / math.sqrt(2 / 3), 0.0, 1 / math.sqrt(2 / 3)]
    assert group_advantages([1.0, 2.0, 3.0], ddof=0) == pytest.approx(expected)


@pytest.mark.parametrize("values", [[0.5, 0.5, 0.5], [1.0], [0.3, 0.3 + 1e-9, 0.3 - 1e-9]])
def test_zero_spread_group_gets_zero_advantage_not_nan(values: list[float]) -> None:
    adv = group_advantages(values)
    assert adv == [0.0] * len(values)


def test_advantages_are_zero_mean_and_unit_std_when_there_is_spread() -> None:
    adv = np.asarray(group_advantages([0.1, 0.4, 0.4, 0.9, 0.2]))
    assert adv.mean() == pytest.approx(0.0, abs=1e-12)
    assert adv.std(ddof=1) == pytest.approx(1.0)


def test_both_advantages_are_stored_per_record() -> None:
    records = [
        rec("A", i, g, ok)
        for i, (g, ok) in enumerate([(0.9, True), (0.8, False), (0.3, True), (0.1, False)])
    ]
    out = add_advantages(records)
    # Correctness 1/0/1/0: mean 0.5, sample std sqrt(1/3).
    expected_ok = [0.5 / math.sqrt(1 / 3), -0.5 / math.sqrt(1 / 3)] * 2
    assert [r.correctness_advantage for r in out] == pytest.approx(expected_ok)
    assert [r.rm_advantage for r in out] == pytest.approx(group_advantages([0.9, 0.8, 0.3, 0.1]))
    # The input records are not modified.
    assert all(r.rm_advantage is None for r in records)


def test_advantages_are_computed_within_each_problem_and_keep_input_order() -> None:
    interleaved = [
        rec("A", 0, 0.9, True),
        rec("B", 0, 0.1, True),
        rec("A", 1, 0.1, False),
        rec("B", 1, 0.9, False),
    ]
    out = add_advantages(interleaved)
    assert [(r.problem_id, r.sample_idx) for r in out] == [("A", 0), ("B", 0), ("A", 1), ("B", 1)]
    # Same grades in both problems, opposite roles: advantages follow the problem.
    assert out[0].rm_advantage > 0 > out[2].rm_advantage
    assert out[3].rm_advantage > 0 > out[1].rm_advantage


def test_all_correct_group_has_zero_correctness_advantage_but_rm_advantage() -> None:
    out = add_advantages([rec("B", i, g, True) for i, g in enumerate([0.5, 0.6, 0.7])])
    assert [r.correctness_advantage for r in out] == [0.0, 0.0, 0.0]
    assert out[2].rm_advantage > 0


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        add_advantages([rec("A", 0, 0.1, True), rec("A", 0, 0.2, False)])


# --- AUC ---


def test_auc_perfect_reversed_and_tied() -> None:
    assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_auc_treats_float_noise_as_a_tie() -> None:
    # Identical answers can be graded 1e-9 apart when scored in different batches.
    assert auc([0.5, 0.5 + 1e-9, 0.5 - 1e-9, 0.5], [True, False, True, False]) == 0.5


def test_noise_level_grade_differences_do_not_split_the_top_set() -> None:
    records = [rec("A", i, 0.9 + i * 1e-10, i % 2 == 0) for i in range(8)]
    summary, _, _ = run_audit(records, top_fraction=0.25)
    assert summary.n_top_graded == 8  # all are the same grade at 1e-6 resolution


def test_auc_is_undefined_with_one_class() -> None:
    assert auc([0.1, 0.2], [True, True]) is None
    assert auc([0.1, 0.2], [False, False]) is None


def test_auc_matches_sklearn_including_ties() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.integers(0, 6, size=40) / 5  # many ties on purpose
        labels = rng.random(40) < 0.4
        if labels.all() or not labels.any():
            continue
        assert auc(scores.tolist(), labels.tolist()) == pytest.approx(roc_auc_score(labels, scores))


def test_within_problem_auc_averages_only_problems_with_both_classes() -> None:
    mean_auc, n_used = within_problem_auc(example_records())
    # A: 3 of 4 pairs ordered correctly. D: the one correct answer is lowest -> 0.
    assert n_used == 2
    assert mean_auc == pytest.approx((0.75 + 0.0) / 2)


def test_pooled_and_within_problem_auc_can_differ() -> None:
    summary, _, _ = run_audit(example_records())
    records = example_records()
    pos = [r.rm_grade for r in records if r.correct]
    neg = [r.rm_grade for r in records if not r.correct]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert summary.auc_pooled == pytest.approx(brute)
    assert summary.auc_within_problem == pytest.approx(0.375)
    assert summary.auc_pooled != pytest.approx(summary.auc_within_problem)


# --- zero-spread group counts ---


def test_zero_spread_groups_are_counted_separately_for_rm_and_correctness() -> None:
    summary, _, _ = run_audit(example_records())
    assert summary.n_groups_zero_spread_rm == 1  # C: all grades equal
    assert summary.n_groups_zero_spread_correctness == 2  # B all correct, C all wrong
    assert summary.n_groups_all_correct == 1
    assert summary.n_groups_all_wrong == 1


def test_rm_zero_spread_with_mixed_correctness_is_counted_for_rm_only() -> None:
    records = [rec("A", 0, 0.4, True), rec("A", 1, 0.4, False), rec("A", 2, 0.4, False)]
    summary, out, candidates = run_audit(records)
    assert summary.n_groups_zero_spread_rm == 1
    assert summary.n_groups_zero_spread_correctness == 0
    assert all(r.rm_advantage == 0.0 for r in out)
    assert candidates == []  # zero advantage means nothing is rewarded


# --- false-positive rate on top-graded answers ---


def test_top_graded_false_positive_rate() -> None:
    grades = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    correct = [True, True, True, True, True, True, False, True]
    records = [rec("A", i, g, ok) for i, (g, ok) in enumerate(zip(grades, correct, strict=True))]
    summary, _, _ = run_audit(records, top_fraction=0.25)
    # 75th percentile is 0.625, so the top set is {0.7, 0.8}; one of two is wrong.
    assert summary.grade_threshold == pytest.approx(0.625)
    assert summary.n_top_graded == 2
    assert summary.false_positive_rate_top == pytest.approx(0.5)


def test_top_fraction_is_configurable() -> None:
    grades = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    correct = [True, True, True, True, False, False, False, False]
    records = [rec("A", i, g, ok) for i, (g, ok) in enumerate(zip(grades, correct, strict=True))]
    summary, _, _ = run_audit(records, top_fraction=0.5)
    assert summary.n_top_graded == 4
    assert summary.false_positive_rate_top == 1.0


def test_tied_grades_can_enlarge_the_top_set() -> None:
    records = [rec("A", i, 0.9, i % 2 == 0) for i in range(8)]
    summary, _, _ = run_audit(records, top_fraction=0.25)
    assert summary.n_top_graded == 8  # all tied at the threshold; none is dropped silently


# --- share of positive advantage going to wrong answers ---


def test_positive_advantage_share_to_wrong_answers() -> None:
    grades = [3.0, 2.0, 1.0, 0.0]
    correct = [False, True, False, True]
    records = [rec("A", i, g, ok) for i, (g, ok) in enumerate(zip(grades, correct, strict=True))]
    summary, _, _ = run_audit(records)
    std = math.sqrt(5 / 3)
    top_adv, second_adv = 1.5 / std, 0.5 / std
    assert summary.n_positive_rm_advantage == 2
    assert summary.positive_advantage_mass_to_wrong == pytest.approx(
        top_adv / (top_adv + second_adv)
    )
    assert summary.positive_advantage_answers_wrong_share == pytest.approx(0.5)


def test_positive_advantage_share_is_none_without_any_spread() -> None:
    records = [rec("A", i, 0.5, i == 0) for i in range(4)]
    summary, _, _ = run_audit(records)
    assert summary.n_positive_rm_advantage == 0
    assert summary.positive_advantage_mass_to_wrong is None


# --- candidates ---


def test_candidates_are_wrong_answers_with_positive_rm_advantage() -> None:
    records = [
        rec("A", 0, 0.9, False, truncated=True, finish_reason="length", n_tokens=256),
        rec("A", 1, 0.8, True),
        rec("A", 2, 0.1, False),  # wrong but graded low: not a candidate
        rec("A", 3, 0.7, False, multiple_distinct_answers=True, finish_reason="stop"),
    ]
    _, with_adv, candidates = run_audit(records)
    assert [c["sample_idx"] for c in candidates] == [0, 3]
    first = candidates[0]
    assert first["truncated"] is True
    assert first["finish_reason"] == "length"
    assert first["n_tokens"] == 256
    assert first["n_chars"] == len("response A/0")
    assert first["rm_grade"] == 0.9
    assert first["rm_advantage"] > 0
    assert candidates[1]["multiple_distinct_answers"] is True
    assert blind_spot_candidates(with_adv) == candidates


def test_candidates_require_advantages() -> None:
    with pytest.raises(ValueError, match="advantages"):
        blind_spot_candidates([rec("A", 0, 0.9, False)])


# --- counts reported in the summary ---


def test_flag_and_status_counts() -> None:
    records = [
        rec("A", 0, 0.9, True),
        rec("A", 1, 0.5, False, truncated=True, status="no_answer"),
        rec("A", 2, 0.4, False, multiple_distinct_answers=True),
        rec("A", 3, 0.1, False, truncated=True, status="unparseable"),
    ]
    summary, _, _ = run_audit(records)
    assert summary.accuracy == 0.25
    assert (summary.n_correct, summary.n_wrong) == (1, 1)
    assert (summary.n_no_answer, summary.n_unparseable) == (1, 1)
    assert summary.n_check_error == 0
    assert summary.n_truncated == 2
    assert summary.n_multiple_distinct_answers == 1
    assert summary.n_unknown_status == 0


def test_check_errors_are_counted_separately_and_are_wrong() -> None:
    records = [
        rec("A", 0, 0.9, True),
        rec("A", 1, 0.5, False, status="check_error"),
        rec("A", 2, 0.1, False),
    ]
    summary, _, _ = run_audit(records)
    assert summary.n_check_error == 1
    assert summary.n_unknown_status == 0
    assert summary.accuracy == pytest.approx(1 / 3)
    assert "1 check errors" in format_summary_table(summary)


def test_records_without_status_are_counted_as_unknown() -> None:
    records = [
        AnswerRecord("A", 0, "x", True, 0.9),
        AnswerRecord("A", 1, "y", False, 0.1),
    ]
    summary, _, _ = run_audit(records)
    assert summary.n_unknown_status == 2


# --- input validation, files, printing ---


def test_run_audit_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="No records"):
        run_audit([])
    with pytest.raises(ValueError, match="finite"):
        run_audit([rec("A", 0, float("nan"), True), rec("A", 1, 0.1, False)])
    with pytest.raises(ValueError, match="top_fraction"):
        run_audit(example_records(), top_fraction=1.0)


def test_records_round_trip_through_jsonl(tmp_path) -> None:
    _, with_adv, _ = run_audit(example_records())
    path = tmp_path / "records.jsonl"
    assert write_jsonl(path, with_adv) == len(with_adv)
    loaded = [AnswerRecord(**row) for row in read_jsonl(path)]
    assert loaded == with_adv


def test_summary_table_mentions_the_key_statistics() -> None:
    summary, _, _ = run_audit(example_records())
    table = format_summary_table(summary)
    for text in (
        "accuracy",
        "AUC within problem",
        "AUC pooled",
        "false-positive rate",
        "no spread: RM grade",
        "no spread: correctness",
        "truncated",
    ):
        assert text in table
