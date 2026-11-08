"""Dataset loading and prompt rendering."""

from __future__ import annotations

import pytest

from blindspots.checker import parse_ground_truth
from blindspots.data import (
    gsm8k_final_answer,
    load_problems,
    problems_from_gsm8k_rows,
    problems_from_math500_rows,
    render_prompt,
)


def test_gsm8k_final_answer_is_the_text_after_the_marker() -> None:
    assert gsm8k_final_answer("She sells 9 eggs.\n#### 18") == "18"
    assert gsm8k_final_answer("work\n#### 1,000") == "1000"  # thousands separator removed
    assert gsm8k_final_answer("text with #### inside\nmore\n#### -3.5") == "-3.5"


def test_gsm8k_solution_without_marker_is_an_error() -> None:
    with pytest.raises(ValueError, match="####"):
        gsm8k_final_answer("no marker here")


def test_gsm8k_rows_become_problems_with_stable_ids() -> None:
    rows = [{"question": "Q1", "answer": "a\n#### 5"}, {"question": "Q2", "answer": "b\n#### 7"}]
    problems = problems_from_gsm8k_rows(rows)
    assert [(p.problem_id, p.question, p.ground_truth) for p in problems] == [
        ("gsm8k-0", "Q1", "5"),
        ("gsm8k-1", "Q2", "7"),
    ]
    assert all(p.source == "gsm8k" for p in problems)


def test_math500_rows_use_the_dataset_ids_and_the_answer_column() -> None:
    rows = [
        {
            "problem": "P",
            "solution": "long solution",
            "answer": r" \frac{1}{2} ",
            "unique_id": "test/x/1.json",
        }
    ]
    (problem,) = problems_from_math500_rows(rows)
    assert problem.problem_id == "math500-test/x/1.json"
    assert problem.ground_truth == r"\frac{1}{2}"  # answer only, not the solution text
    assert problem.source == "math500"


def test_render_prompt_keeps_braces_literal() -> None:
    template = "{problem}\n\nPut the answer in \\boxed{}."
    out = render_prompt(template, "What is {x}?")
    assert out == "What is {x}?\n\nPut the answer in \\boxed{}."


def test_render_prompt_requires_the_placeholder() -> None:
    with pytest.raises(ValueError):
        render_prompt("no placeholder", "Q")


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_problems("not-a-dataset")


# --- the real datasets (small downloads, cached after the first run) ---


@pytest.mark.parametrize(("name", "size"), [("gsm8k", 1319), ("math500", 500)])
def test_real_datasets_load_and_every_ground_truth_parses(name: str, size: int) -> None:
    problems = load_problems(name)
    assert len(problems) == size
    assert len({p.problem_id for p in problems}) == size
    for problem in problems:
        assert problem.question.strip()
        parse_ground_truth(problem.ground_truth)  # raises if the checker cannot read it


def test_subset_is_the_first_n_problems() -> None:
    assert load_problems("gsm8k", n_problems=3) == load_problems("gsm8k")[:3]
