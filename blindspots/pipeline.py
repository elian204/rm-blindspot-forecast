"""Glue for one audit run: sample, check, grade, compute statistics, save.

Kept separate from ``audit.py`` so the statistics stay free of any model code.
Each stage can be given ready-made inputs (samples, a reward model), which is
how the tests run the whole pipeline without loading models, and how a GPU run
can sample first (vLLM) and grade later in a separate process.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from blindspots.audit import (
    AnswerRecord,
    AuditSummary,
    format_summary_table,
    run_audit,
)
from blindspots.checker import check_response
from blindspots.config import RunConfig, config_to_dict
from blindspots.data import Problem, load_problems
from blindspots.io import file_sha256, read_jsonl, write_jsonl, write_meta
from blindspots.rewards import RewardModel, RewardScores
from blindspots.sampling import Sample, sample_responses


def reward_model_inputs(
    problems: Sequence[Problem], samples: Sequence[Sample]
) -> tuple[list[str], list[str]]:
    """Texts handed to the reward model: the raw problem and the response.

    Deliberately not the policy's prompt (no chat template, no answer-format
    instruction), so what the reward model sees does not depend on the policy.
    """
    question = {p.problem_id: p.question for p in problems}
    return [question[s.problem_id] for s in samples], [s.response for s in samples]


def build_records(
    problems: Sequence[Problem], samples: Sequence[Sample], scores: RewardScores
) -> list[AnswerRecord]:
    """Check every sample against its ground truth and attach its RM grade."""
    if len(scores.grades) != len(samples):
        raise ValueError("Need exactly one grade per sample")
    by_id = {p.problem_id: p for p in problems}
    records = []
    for i, sample in enumerate(samples):
        problem = by_id[sample.problem_id]
        result = check_response(sample.response, problem.ground_truth, sample.finish_reason)
        records.append(
            AnswerRecord(
                problem_id=sample.problem_id,
                sample_idx=sample.sample_idx,
                response=sample.response,
                correct=result.correct,
                rm_grade=scores.grades[i],
                problem=problem.question,
                ground_truth=problem.ground_truth,
                extracted_answer=result.extracted,
                status=result.status.value,
                multiple_distinct_answers=result.multiple_distinct_answers,
                truncated=result.truncated,
                finish_reason=sample.finish_reason,
                n_tokens=sample.n_response_tokens,
                step_scores=None if scores.step_scores is None else scores.step_scores[i],
            )
        )
    return records


def check_sample_coverage(problems: Sequence[Problem], samples: Sequence[Sample]) -> None:
    """Samples must cover exactly the configured problems, with equal counts.

    Anything else would silently audit a different set than the config says.
    """
    wanted = [p.problem_id for p in problems]
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.problem_id] = counts.get(s.problem_id, 0) + 1
    missing = [i for i in wanted if i not in counts]
    extra = [i for i in counts if i not in set(wanted)]
    if missing or extra:
        raise ValueError(
            f"Samples do not match the configured problems: {len(missing)} problems have no "
            f"samples, {len(extra)} sampled problems are not in the config "
            f"({len(wanted)} problems configured, {len(counts)} sampled)"
        )
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"Problems have different numbers of samples: {sorted(set(counts.values()))}"
        )


def save_audit(
    out_dir: str | Path,
    config: RunConfig,
    summary: AuditSummary,
    records: Sequence[AnswerRecord],
    candidates: list[dict],
    provenance: dict | None = None,
) -> None:
    """Write all result files, with config and commit hash (meta_audit.json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "records.jsonl", records)
    write_jsonl(out_dir / "blind_spot_candidates.jsonl", candidates)
    (out_dir / "summary.json").write_text(
        json.dumps(dataclasses.asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    write_meta(out_dir, config_to_dict(config), "audit", **(provenance or {}))


def run_pipeline(
    config: RunConfig,
    output_dir: str | Path | None = None,
    samples: Sequence[Sample] | None = None,
    samples_path: str | Path | None = None,
    reward_model: RewardModel | None = None,
) -> tuple[AuditSummary, list[AnswerRecord], list[dict]]:
    """Run one audit. Returns ``(summary, records_with_advantages, candidates)``.

    ``samples_path`` (or ``samples``) skips the sampling stage, e.g. for samples
    made earlier on a GPU machine; their file hash and the meta file next to them
    are recorded in ``meta_audit.json``. ``reward_model`` replaces the one named
    in the config.
    """
    config.validate()
    problems = load_problems(config.data.dataset, config.data.split, config.data.n_problems)

    out_dir = output_dir if output_dir is not None else config.audit.output_dir
    provenance: dict = {"samples_from": "this run"}
    if samples is not None and samples_path is not None:
        raise ValueError("Give either samples or samples_path, not both")
    if samples_path is not None:
        samples = load_samples(samples_path)
        sample_meta = Path(samples_path).with_name("meta_sample.json")
        provenance = {
            "samples_from": "file",
            "samples_sha256": file_sha256(samples_path),
            "samples_meta": (
                json.loads(sample_meta.read_text(encoding="utf-8"))
                if sample_meta.exists()
                else None
            ),
        }
    if samples is None:
        samples = sample_responses(problems, config.prompt.template, config.sampling)
        # Saved right away: sampling is the expensive stage, so it survives a failed grading stage.
        write_jsonl(Path(out_dir) / "samples.jsonl", samples)
        write_meta(out_dir, config_to_dict(config), "sample")
    else:
        if samples_path is None:
            provenance = {"samples_from": "argument"}
        check_sample_coverage(problems, samples)

    if reward_model is None:
        from blindspots.rewards import load_reward_model

        reward_model = load_reward_model(config.reward)
    questions, responses = reward_model_inputs(problems, samples)
    scores = reward_model.score(questions, responses)

    records = build_records(problems, samples, scores)
    summary, with_advantages, candidates = run_audit(records, config.audit.top_fraction)

    save_audit(out_dir, config, summary, with_advantages, candidates, provenance)
    return summary, with_advantages, candidates


def load_samples(path: str | Path) -> list[Sample]:
    """Read samples written by an earlier sampling stage."""
    return [Sample(**row) for row in read_jsonl(path)]


def print_summary(summary: AuditSummary) -> None:
    """Print the short summary table."""
    print(format_summary_table(summary))
