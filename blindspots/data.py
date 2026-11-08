"""Load math problems and render the policy prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Hugging Face dataset IDs and the columns we read from each.
GSM8K_ID = "openai/gsm8k"  # config "main"; columns: question, answer ("... #### 18")
MATH500_ID = "HuggingFaceH4/MATH-500"  # columns: problem, solution, answer, unique_id, ...


@dataclass(frozen=True)
class Problem:
    """One problem with the final answer to check responses against."""

    problem_id: str
    question: str
    ground_truth: str  # final answer only (a number or a LaTeX expression)
    source: str  # "gsm8k" or "math500"


def gsm8k_final_answer(solution: str) -> str:
    """Final answer of a GSM8K solution: the text after the last '####'.

    GSM8K writes large numbers with thousands separators ("1,000"); drop them
    so the answer is a plain number.
    """
    if "####" not in solution:
        raise ValueError("GSM8K solution without '####' marker")
    return solution.rsplit("####", 1)[1].strip().replace(",", "")


def problems_from_gsm8k_rows(rows: list[dict[str, Any]]) -> list[Problem]:
    """Convert raw GSM8K rows (in file order) into Problems."""
    return [
        Problem(
            problem_id=f"gsm8k-{i}",
            question=row["question"],
            ground_truth=gsm8k_final_answer(row["answer"]),
            source="gsm8k",
        )
        for i, row in enumerate(rows)
    ]


def problems_from_math500_rows(rows: list[dict[str, Any]]) -> list[Problem]:
    """Convert raw MATH-500 rows into Problems (ids come from the dataset)."""
    return [
        Problem(
            problem_id=f"math500-{row['unique_id']}",
            question=row["problem"],
            ground_truth=row["answer"].strip(),
            source="math500",
        )
        for row in rows
    ]


def load_problems(
    dataset: str, split: str = "test", n_problems: int | None = None
) -> list[Problem]:
    """Load GSM8K or MATH-500 from the Hugging Face hub (cached after first use).

    ``n_problems`` takes the first n problems in dataset order, so a subset is
    deterministic and the same on every machine.
    """
    from datasets import load_dataset  # imported lazily: slow and not needed by tests

    if dataset == "gsm8k":
        rows = list(load_dataset(GSM8K_ID, "main", split=split))
        problems = problems_from_gsm8k_rows(rows)
    elif dataset == "math500":
        rows = list(load_dataset(MATH500_ID, split=split))
        problems = problems_from_math500_rows(rows)
    else:
        raise ValueError(f"Unknown dataset '{dataset}' (expected 'gsm8k' or 'math500')")
    return problems if n_problems is None else problems[:n_problems]


def render_prompt(template: str, question: str) -> str:
    """Fill the prompt template. A plain replace keeps braces like \\boxed{} literal."""
    if "{problem}" not in template:
        raise ValueError("template must contain '{problem}'")
    return template.replace("{problem}", question)
