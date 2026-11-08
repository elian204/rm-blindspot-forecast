"""Audit statistics, computed from records of graded answers.

Input is a list of :class:`AnswerRecord` (one per sampled answer): whether the
answer is correct and the grade the reward model (RM) gave it. No model is
loaded here, so every statistic is unit-testable.

Group advantages are group-relative: within the answers to one problem,
``(x - group mean) / group std``. The std uses Bessel's correction (ddof=1), as in
TRL's GRPOTrainer. A group with no spread has advantage 0 and is counted.

Grades are compared at a resolution of 1e-6 everywhere (ties in AUC, the top
threshold, zero-spread groups), because identical answers can get grades that
differ by float noise when they are scored in different batches.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Grades closer together than this count as "no spread". Identical answers can
# get grades that differ by float noise when scored in different batches.
SPREAD_TOLERANCE = 1e-6
GRADE_DECIMALS = 6


@dataclass
class AnswerRecord:
    """One sampled answer with its correctness and RM grade."""

    problem_id: str
    sample_idx: int
    response: str
    correct: bool
    rm_grade: float
    # Context carried along so candidates can be inspected without re-joining files.
    problem: str | None = None
    ground_truth: str | None = None
    extracted_answer: str | None = None
    status: str | None = None  # "correct", "wrong", "no_answer" or "unparseable"
    multiple_distinct_answers: bool = False
    truncated: bool = False
    finish_reason: str | None = None
    n_tokens: int | None = None
    step_scores: list[float] | None = None
    # Filled in by add_advantages: advantage within the problem's group.
    rm_advantage: float | None = None
    correctness_advantage: float | None = None  # correctness is 1.0 / 0.0


@dataclass
class AuditSummary:
    """All statistics reported by the audit."""

    n_problems: int
    n_answers: int
    accuracy: float
    # Status breakdown. Missing and unparseable answers are wrong, counted here.
    n_correct: int
    n_wrong: int
    n_no_answer: int
    n_unparseable: int
    n_check_error: int  # math-verify timed out or failed (counted as wrong)
    n_unknown_status: int  # records without a status field
    # Flags from the checker.
    n_multiple_distinct_answers: int
    n_truncated: int
    # Ranking quality of the RM. None when undefined (e.g. all answers correct).
    auc_within_problem: float | None  # primary: advantages are formed within a problem
    n_problems_with_both_classes: int
    auc_pooled: float | None
    # Share of highly graded answers (top `top_fraction` of grades) that are wrong.
    top_fraction: float
    grade_threshold: float
    n_top_graded: int
    false_positive_rate_top: float
    # Group-relative advantages under the RM grade.
    n_positive_rm_advantage: int
    positive_advantage_mass_to_wrong: float | None  # None if no positive advantage exists
    positive_advantage_answers_wrong_share: float | None
    # Groups without spread (advantage set to 0), counted separately.
    n_groups_zero_spread_rm: int
    n_groups_zero_spread_correctness: int
    n_groups_all_correct: int
    n_groups_all_wrong: int
    n_blind_spot_candidates: int


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def group_by_problem(records: Sequence[AnswerRecord]) -> dict[str, list[AnswerRecord]]:
    """Group records by problem id, keeping the input order."""
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[record.problem_id].append(record)
    return dict(groups)


def has_spread(values: Sequence[float], tol: float = SPREAD_TOLERANCE) -> bool:
    """True if the values are not (numerically) all the same."""
    return len(values) > 1 and (max(values) - min(values)) > tol


def group_advantages(
    values: Sequence[float], ddof: int = 1, tol: float = SPREAD_TOLERANCE
) -> list[float]:
    """``(x - mean) / std`` within one group; all zeros if the group has no spread."""
    if not has_spread(values, tol):
        return [0.0] * len(values)
    arr = np.asarray(values, dtype=float)
    return ((arr - arr.mean()) / arr.std(ddof=ddof)).tolist()


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability that a random positive outranks a random negative (ties count 1/2).

    ``None`` if there is no positive or no negative. O(n log n) via average ranks.
    """
    scores = np.round(np.asarray(scores, dtype=float), GRADE_DECIMALS)
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average 1-based rank of the tie block
        i = j + 1
    pos_rank_sum = ranks[np.asarray(labels, dtype=bool)].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_problem_auc(records: Sequence[AnswerRecord]) -> tuple[float | None, int]:
    """Mean AUC over problems that have both correct and wrong answers.

    Returns ``(mean_auc, number_of_problems_used)``; the mean is unweighted.
    """
    aucs = []
    for group in group_by_problem(records).values():
        value = auc([r.rm_grade for r in group], [r.correct for r in group])
        if value is not None:
            aucs.append(value)
    return (float(np.mean(aucs)) if aucs else None), len(aucs)


def add_advantages(
    records: Sequence[AnswerRecord], ddof: int = 1, tol: float = SPREAD_TOLERANCE
) -> list[AnswerRecord]:
    """Copy of the records with both advantages set, each within its problem group."""
    out: dict[tuple[str, int], AnswerRecord] = {}
    for group in group_by_problem(records).values():
        rm_adv = group_advantages([r.rm_grade for r in group], ddof, tol)
        ok_adv = group_advantages([1.0 if r.correct else 0.0 for r in group], ddof, tol)
        for r, a, b in zip(group, rm_adv, ok_adv, strict=True):
            out[(r.problem_id, r.sample_idx)] = dataclasses.replace(
                r, rm_advantage=a, correctness_advantage=b
            )
    if len(out) != len(records):
        raise ValueError("Duplicate (problem_id, sample_idx) pairs in records")
    return [out[(r.problem_id, r.sample_idx)] for r in records]


def blind_spot_candidates(records: Sequence[AnswerRecord]) -> list[dict[str, Any]]:
    """Wrong answers that the RM rewards (positive advantage within their problem).

    Records must already carry advantages (see :func:`add_advantages`). Each
    entry keeps the text, grade, length, finish reason and flags for later inspection.
    """
    candidates = []
    for r in records:
        if r.rm_advantage is None:
            raise ValueError("Records need advantages; call add_advantages first")
        if not r.correct and r.rm_advantage > 0:
            candidates.append(
                {
                    "problem_id": r.problem_id,
                    "sample_idx": r.sample_idx,
                    "problem": r.problem,
                    "ground_truth": r.ground_truth,
                    "extracted_answer": r.extracted_answer,
                    "status": r.status,
                    "response": r.response,
                    "rm_grade": r.rm_grade,
                    "rm_advantage": r.rm_advantage,
                    "n_chars": len(r.response),
                    "n_tokens": r.n_tokens,
                    "finish_reason": r.finish_reason,
                    "truncated": r.truncated,
                    "multiple_distinct_answers": r.multiple_distinct_answers,
                    "step_scores": r.step_scores,
                }
            )
    return candidates


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------


def run_audit(
    records: Sequence[AnswerRecord],
    top_fraction: float = 0.25,
    ddof: int = 1,
    tol: float = SPREAD_TOLERANCE,
) -> tuple[AuditSummary, list[AnswerRecord], list[dict[str, Any]]]:
    """Compute all audit statistics.

    Returns ``(summary, records_with_advantages, blind_spot_candidates)``.
    """
    if not records:
        raise ValueError("No records to audit")
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if not all(math.isfinite(r.rm_grade) for r in records):
        raise ValueError("RM grades must be finite numbers")

    with_adv = add_advantages(records, ddof, tol)
    groups = group_by_problem(with_adv)

    grades = [r.rm_grade for r in with_adv]
    correct = [r.correct for r in with_adv]

    # "Highly graded" = grade at or above the (1 - top_fraction) quantile of all
    # grades. With tied grades the top set can be larger than top_fraction.
    resolved = [round(g, GRADE_DECIMALS) for g in grades]
    threshold = float(np.quantile(resolved, 1.0 - top_fraction))
    top = [r for r, g in zip(with_adv, resolved, strict=True) if g >= threshold]
    false_positive_rate = sum(not r.correct for r in top) / len(top)

    positive = [r for r in with_adv if (r.rm_advantage or 0.0) > 0]
    positive_mass = sum(r.rm_advantage for r in positive)  # type: ignore[misc]
    wrong_mass = sum(r.rm_advantage for r in positive if not r.correct)  # type: ignore[misc]

    auc_within, n_both = within_problem_auc(with_adv)
    status_counts = defaultdict(int)
    for r in with_adv:
        status_counts[r.status] += 1

    candidates = blind_spot_candidates(with_adv)
    summary = AuditSummary(
        n_problems=len(groups),
        n_answers=len(with_adv),
        accuracy=sum(correct) / len(correct),
        n_correct=status_counts["correct"],
        n_wrong=status_counts["wrong"],
        n_no_answer=status_counts["no_answer"],
        n_unparseable=status_counts["unparseable"],
        n_check_error=status_counts["check_error"],
        n_unknown_status=len(with_adv)
        - sum(
            status_counts[k]
            for k in ("correct", "wrong", "no_answer", "unparseable", "check_error")
        ),
        n_multiple_distinct_answers=sum(r.multiple_distinct_answers for r in with_adv),
        n_truncated=sum(r.truncated for r in with_adv),
        auc_within_problem=auc_within,
        n_problems_with_both_classes=n_both,
        auc_pooled=auc(grades, correct),
        top_fraction=top_fraction,
        grade_threshold=threshold,
        n_top_graded=len(top),
        false_positive_rate_top=false_positive_rate,
        n_positive_rm_advantage=len(positive),
        positive_advantage_mass_to_wrong=(wrong_mass / positive_mass) if positive else None,
        positive_advantage_answers_wrong_share=(
            sum(not r.correct for r in positive) / len(positive) if positive else None
        ),
        n_groups_zero_spread_rm=sum(
            not has_spread([r.rm_grade for r in g], tol) for g in groups.values()
        ),
        n_groups_zero_spread_correctness=sum(
            not has_spread([float(r.correct) for r in g], tol) for g in groups.values()
        ),
        n_groups_all_correct=sum(all(r.correct for r in g) for g in groups.values()),
        n_groups_all_wrong=sum(not any(r.correct for r in g) for g in groups.values()),
        n_blind_spot_candidates=len(candidates),
    )
    return summary, with_adv, candidates


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_summary_table(s: AuditSummary) -> str:
    """Short plain-text table for the console."""
    rows = [
        ("problems / answers", f"{s.n_problems} / {s.n_answers}"),
        ("accuracy", _fmt(s.accuracy)),
        (
            "correct / wrong / no answer / unparseable",
            f"{s.n_correct} / {s.n_wrong} / {s.n_no_answer} / {s.n_unparseable}"
            + (f" (+{s.n_check_error} check errors)" if s.n_check_error else ""),
        ),
        ("multiple distinct boxed answers", str(s.n_multiple_distinct_answers)),
        ("truncated answers", str(s.n_truncated)),
        (
            f"RM AUC within problem (primary, {s.n_problems_with_both_classes} problems)",
            _fmt(s.auc_within_problem),
        ),
        ("RM AUC pooled", _fmt(s.auc_pooled)),
        (
            f"false-positive rate, top {s.top_fraction:.0%} grades "
            f"({s.n_top_graded} answers, grade >= {s.grade_threshold:.3f})",
            _fmt(s.false_positive_rate_top),
        ),
        (
            "positive RM advantage mass going to wrong answers",
            _fmt(s.positive_advantage_mass_to_wrong),
        ),
        (
            "answers with positive RM advantage that are wrong",
            _fmt(s.positive_advantage_answers_wrong_share),
        ),
        ("groups with no spread: RM grade", str(s.n_groups_zero_spread_rm)),
        (
            "groups with no spread: correctness (all correct / all wrong)",
            f"{s.n_groups_zero_spread_correctness} "
            f"({s.n_groups_all_correct} / {s.n_groups_all_wrong})",
        ),
        ("blind-spot candidates", str(s.n_blind_spot_candidates)),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)
