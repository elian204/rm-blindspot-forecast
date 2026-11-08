"""Step splitting, input rendering and score combination (no model weights needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blindspots.config import RewardConfig
from blindspots.rewards import combine_steps
from blindspots.rewards.skywork import SkyworkPRM, build_input_ids, split_steps

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_ID = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL_REVISION = "98d69606595eedbdbbbf0a7d28efdcd462ba6a67"


# --- step splitting: the one deterministic rule ---


def test_each_line_is_one_step() -> None:
    assert split_steps("a\nb\nc") == ["a", "b", "c"]


def test_response_without_line_breaks_is_a_single_step() -> None:
    text = "First add 3 and 4 to get 7, then subtract from 16 to get 9, so the answer is 18."
    assert split_steps(text) == [text]


def test_response_with_very_many_lines() -> None:
    lines = [f"step {i}" for i in range(2000)]
    steps = split_steps("\n".join(lines))
    assert steps == lines  # nothing lost, nothing merged, order kept


def test_blank_lines_and_paragraph_breaks_do_not_create_steps() -> None:
    assert split_steps("a\n\nb\n\n\nc\n") == ["a", "b", "c"]
    assert split_steps("\n\na\n") == ["a"]


def test_whitespace_only_lines_are_dropped_but_content_lines_are_kept_verbatim() -> None:
    assert split_steps("  indented step \n   \n\t\nnext") == ["  indented step ", "next"]


def test_windows_and_old_mac_line_endings_are_normalised() -> None:
    assert split_steps("a\r\nb\rc") == ["a", "b", "c"]


@pytest.mark.parametrize("text", ["", "\n", "  \n \n"])
def test_empty_response_is_one_empty_step(text: str) -> None:
    assert split_steps(text) == [""]


def test_splitting_is_deterministic_and_does_not_modify_the_input() -> None:
    text = "x\n\ny\nz"
    assert split_steps(text) == split_steps(text)
    assert text == "x\n\ny\nz"


def test_model_card_examples_split_into_the_steps_the_official_code_sees() -> None:
    examples = json.loads((FIXTURES / "skywork_card_examples.json").read_text())["examples"]
    reference = json.loads((FIXTURES / "skywork_card_reference.json").read_text())
    for example, official_steps in zip(examples, reference["steps"], strict=True):
        # The official code appends the step token to every step.
        assert [s + "\n" for s in split_steps(example["response"])] == official_steps


# --- input rendering (needs only the tokenizer files) ---


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.txt"])
    return AutoTokenizer.from_pretrained(path)


def test_input_starts_with_bos_and_problem_and_marks_every_step_end(tokenizer) -> None:
    ids, ends = build_input_ids(tokenizer, "What is 2+2?", "Add.\n\nSo 4.")
    newline_id = tokenizer.encode("\n")[-1]
    prompt_ids = tokenizer.encode("<|im_start|>What is 2+2?\n")
    assert tokenizer.bos_token == "<|im_start|>"
    assert ids[: len(prompt_ids)] == prompt_ids
    assert len(ends) == 2  # "Add." and "So 4." (the blank line is not a step)
    assert all(ids[i] == newline_id for i in ends)
    assert ends[-1] == len(ids) - 1
    assert ends == sorted(ends)


def test_each_step_is_tokenised_on_its_own(tokenizer) -> None:
    ids, ends = build_input_ids(tokenizer, "Q", "alpha\nbeta")
    prompt_len = len(tokenizer.encode("<|im_start|>Q\n"))
    newline_id = tokenizer.encode("\n")[-1]
    expected = tokenizer.encode("alpha") + [newline_id] + tokenizer.encode("beta") + [newline_id]
    assert ids[prompt_len:] == expected
    assert ends[0] == prompt_len + len(tokenizer.encode("alpha"))


def test_empty_response_still_gets_one_scored_position(tokenizer) -> None:
    ids, ends = build_input_ids(tokenizer, "Q", "")
    assert len(ends) == 1 and ends[0] == len(ids) - 1


def test_overlong_input_is_rejected_not_silently_truncated(tokenizer) -> None:
    with pytest.raises(ValueError, match="limit"):
        build_input_ids(tokenizer, "Q", "word " * 200, max_length=50)


def test_all_overlong_inputs_are_reported_at_once_before_any_model_compute(tokenizer) -> None:
    rm = object.__new__(SkyworkPRM)  # no weights: the length check runs before the model
    rm.tokenizer, rm.model, rm.config = tokenizer, None, RewardConfig(max_length=60)
    long_text = "word " * 100
    with pytest.raises(ValueError, match=r"2 of 3 inputs exceed.*\(0,.*\(2,"):
        rm.step_scores(["Q", "Q", "Q"], [long_text, "short", long_text])


# --- combining step scores ---


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("min", 0.5), ("last", 0.8), ("prod", 0.9 * 0.5 * 0.8), ("mean", (0.9 + 0.5 + 0.8) / 3)],
)
def test_combine_steps(rule: str, expected: float) -> None:
    assert combine_steps([0.9, 0.5, 0.8], rule) == pytest.approx(expected)


def test_combine_rules_can_disagree_about_which_answer_is_better() -> None:
    steady = [0.7, 0.7, 0.7]
    one_bad_step = [0.95, 0.2, 0.95]
    assert combine_steps(one_bad_step, "last") > combine_steps(steady, "last")
    assert combine_steps(one_bad_step, "min") < combine_steps(steady, "min")


def test_combine_rejects_unknown_rule_and_empty_input() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        combine_steps([0.5], "median")
    with pytest.raises(ValueError, match="empty"):
        combine_steps([], "min")
