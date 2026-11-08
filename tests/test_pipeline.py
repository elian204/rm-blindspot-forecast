"""The whole audit pipeline with fake samples and a fake reward model (no model weights)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from blindspots.audit import AnswerRecord
from blindspots.config import RunConfig
from blindspots.data import Problem, load_problems
from blindspots.io import read_jsonl
from blindspots.pipeline import (
    build_records,
    check_sample_coverage,
    load_samples,
    reward_model_inputs,
    run_pipeline,
)
from blindspots.rewards import RewardScores
from blindspots.sampling import Sample


def make_sample(problem: Problem, idx: int, response: str, reason: str = "stop") -> Sample:
    return Sample(
        problem_id=problem.problem_id,
        sample_idx=idx,
        prompt=f"<|im_start|>user\n{problem.question}\n\nPut the answer in \\boxed{{}}.<|im_end|>",
        response=response,
        finish_reason=reason,
        n_prompt_tokens=50,
        n_response_tokens=len(response.split()),
    )


class LengthBiasedRM:
    """Fake reward model: longer responses get higher grades (a planted flaw)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> RewardScores:
        self.seen += list(zip(prompts, responses, strict=True))
        steps = [[min(1.0, len(r) / 100)] for r in responses]
        return RewardScores(grades=[s[0] for s in steps], step_scores=steps)


def test_reward_model_sees_the_raw_problem_not_the_policy_prompt() -> None:
    problems = load_problems("gsm8k", n_problems=2)
    samples = [make_sample(problems[0], 0, "so \\boxed{1}"), make_sample(problems[1], 0, "x")]
    questions, responses = reward_model_inputs(problems, samples)
    assert questions == [problems[0].question, problems[1].question]
    assert responses == ["so \\boxed{1}", "x"]
    assert not any("im_start" in q or "boxed" in q for q in questions)


def test_records_carry_checker_results_flags_and_scores() -> None:
    problems = load_problems("gsm8k", n_problems=1)
    truth = problems[0].ground_truth  # "18"
    samples = [
        make_sample(problems[0], 0, f"Done. \\boxed{{{truth}}}"),
        make_sample(problems[0], 1, "First \\boxed{1} then \\boxed{2}"),
        make_sample(problems[0], 2, "I am still thinking about", reason="length"),
    ]
    scores = RewardScores(grades=[0.9, 0.8, 0.1], step_scores=[[0.9], [0.8], [0.1]])
    records = build_records(problems, samples, scores)
    assert [r.status for r in records] == ["correct", "wrong", "no_answer"]
    assert [r.correct for r in records] == [True, False, False]
    assert [r.multiple_distinct_answers for r in records] == [False, True, False]
    assert [r.truncated for r in records] == [False, False, True]
    assert [r.rm_grade for r in records] == [0.9, 0.8, 0.1]
    assert records[2].extracted_answer is None  # missing answers are kept, not dropped
    assert records[0].ground_truth == truth


def test_pipeline_end_to_end_writes_results_with_config_and_commit(tmp_path: Path) -> None:
    config = RunConfig()
    config.data.n_problems = 3
    problems = load_problems("gsm8k", n_problems=3)
    samples = []
    for p in problems:
        samples += [
            make_sample(p, 0, f"Short. \\boxed{{{p.ground_truth}}}"),
            make_sample(p, 1, "A very long and wrong explanation " * 5 + "\\boxed{-1}"),
            make_sample(p, 2, "wrong \\boxed{-2}"),
        ]
    rm = LengthBiasedRM()
    out_dir = tmp_path / "run"
    summary, records, candidates = run_pipeline(
        config, output_dir=out_dir, samples=samples, reward_model=rm
    )

    assert summary.n_problems == 3 and summary.n_answers == 9
    assert summary.accuracy == 3 / 9
    # The planted length bias: the long wrong answer gets the top grade in every problem.
    assert [c["sample_idx"] for c in candidates] == [1, 1, 1]
    assert summary.auc_within_problem is not None and summary.auc_within_problem < 0.5
    assert all(q == p.question for (q, _), p in zip(rm.seen[::3], problems, strict=True))

    for name in ("records.jsonl", "blind_spot_candidates.jsonl", "summary.json", "meta_audit.json"):
        assert (out_dir / name).exists(), name
    meta = json.loads((out_dir / "meta_audit.json").read_text())
    assert meta["config"]["data"]["n_problems"] == 3
    assert len(meta["git_commit"]) == 40
    saved = [AnswerRecord(**row) for row in read_jsonl(out_dir / "records.jsonl")]
    assert saved == records
    assert all(r.rm_advantage is not None and r.correctness_advantage is not None for r in saved)
    assert len(list(read_jsonl(out_dir / "blind_spot_candidates.jsonl"))) == 3


def test_samples_round_trip_through_jsonl(tmp_path: Path) -> None:
    problems = load_problems("gsm8k", n_problems=1)
    samples = [make_sample(problems[0], i, f"r{i}") for i in range(2)]
    from blindspots.io import write_jsonl

    write_jsonl(tmp_path / "s.jsonl", samples)
    assert load_samples(tmp_path / "s.jsonl") == samples


def test_samples_are_saved_before_grading_so_a_grading_failure_does_not_lose_them(
    tmp_path: Path, monkeypatch
) -> None:
    import blindspots.pipeline as pipeline

    problems = load_problems("gsm8k", n_problems=2)
    fake = [make_sample(p, i, f"r{i} \\boxed{{1}}") for p in problems for i in range(2)]
    monkeypatch.setattr(pipeline, "sample_responses", lambda *args, **kwargs: fake)

    class BrokenRM:
        def score(self, prompts, responses):
            raise RuntimeError("out of memory")

    config = RunConfig()
    config.data.n_problems = 2
    with pytest.raises(RuntimeError, match="out of memory"):
        run_pipeline(config, output_dir=tmp_path, reward_model=BrokenRM())

    assert load_samples(tmp_path / "samples.jsonl") == fake
    assert json.loads((tmp_path / "meta_sample.json").read_text())["stage"] == "sample"


# --- samples reused from an earlier stage ---


def test_sample_coverage_must_match_the_configured_problems_exactly() -> None:
    problems = load_problems("gsm8k", n_problems=3)

    def full(ps):
        return [make_sample(p, i, "x") for p in ps for i in range(2)]

    check_sample_coverage(problems, full(problems))  # fine
    with pytest.raises(ValueError, match="1 problems have no samples"):
        check_sample_coverage(problems, full(problems[:2]))  # silently auditing 2 of 3
    bigger = load_problems("gsm8k", n_problems=4)
    with pytest.raises(ValueError, match="1 sampled problems are not in the config"):
        check_sample_coverage(problems, full(bigger))
    uneven = full(problems) + [make_sample(problems[0], 2, "x")]
    with pytest.raises(ValueError, match="different numbers"):
        check_sample_coverage(problems, uneven)


def test_audit_of_a_samples_file_records_where_the_samples_came_from(tmp_path: Path) -> None:
    from blindspots.config import config_to_dict
    from blindspots.io import file_sha256, write_jsonl, write_meta

    config = RunConfig()
    config.data.n_problems = 2
    problems = load_problems("gsm8k", n_problems=2)
    samples = [make_sample(p, i, f"r{i} \\boxed{{1}}") for p in problems for i in range(2)]
    sample_dir = tmp_path / "gpu_box"
    write_jsonl(sample_dir / "samples.jsonl", samples)
    write_meta(sample_dir, {**config_to_dict(config), "marker": "sampling-config"}, "sample")

    audit_dir = tmp_path / "audit"
    run_pipeline(
        config,
        output_dir=audit_dir,
        samples_path=sample_dir / "samples.jsonl",
        reward_model=LengthBiasedRM(),
    )
    meta = json.loads((audit_dir / "meta_audit.json").read_text())
    assert meta["samples_from"] == "file"
    assert meta["samples_sha256"] == file_sha256(sample_dir / "samples.jsonl")
    assert meta["samples_meta"]["config"]["marker"] == "sampling-config"
    assert not (audit_dir / "meta_sample.json").exists()  # not overwritten, not invented
