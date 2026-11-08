"""Sampling helpers. The real CPU generation test is marked slow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from blindspots.config import PromptConfig, SamplingConfig
from blindspots.data import Problem
from blindspots.sampling import (
    build_policy_prompts,
    hf_generation_config,
    load_tokenizer,
    sample_responses,
    samples_from_vllm_outputs,
    split_generated,
    vllm_sampling_kwargs,
)

QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def problem(i: int) -> Problem:
    return Problem(f"p{i}", f"What is {i}+{i}?", str(2 * i), "test")


# --- cutting generated rows at the end-of-sequence token ---


def test_split_generated_stops_at_first_eos_and_keeps_it() -> None:
    tokens, reason = split_generated([5, 6, 99, 0, 0], eos_ids={99, 0})
    assert (tokens, reason) == ([5, 6, 99], "stop")


def test_split_generated_without_eos_is_truncated() -> None:
    assert split_generated([5, 6, 7], eos_ids={99}) == ([5, 6, 7], "length")


def test_split_generated_eos_first_and_multiple_eos_ids() -> None:
    assert split_generated([99, 5], eos_ids={99, 100}) == ([99], "stop")
    assert split_generated([5, 100, 99], eos_ids={99, 100}) == ([5, 100], "stop")


# --- vLLM output conversion (the vLLM fields are checked against its source, v0.33) ---


def fake_request(n_prompt: int, completions: list[tuple[str, list[int], str | None]]):
    return SimpleNamespace(
        prompt_token_ids=list(range(n_prompt)),
        outputs=[
            SimpleNamespace(text=t, token_ids=ids, finish_reason=reason)
            for t, ids, reason in completions
        ],
    )


def test_vllm_outputs_become_samples_in_problem_then_sample_order() -> None:
    problems = [problem(1), problem(2)]
    outputs = [
        fake_request(7, [("a", [1, 2], "stop"), ("b", [1, 2, 3], "length")]),
        fake_request(9, [("c", [4], "stop"), ("d", [], "stop")]),
    ]
    samples = samples_from_vllm_outputs(problems, ["P1", "P2"], outputs)
    assert [(s.problem_id, s.sample_idx, s.response) for s in samples] == [
        ("p1", 0, "a"),
        ("p1", 1, "b"),
        ("p2", 0, "c"),
        ("p2", 1, "d"),
    ]
    assert [s.finish_reason for s in samples] == ["stop", "length", "stop", "stop"]
    assert [s.n_response_tokens for s in samples] == [2, 3, 1, 0]
    assert [s.n_prompt_tokens for s in samples] == [7, 7, 9, 9]
    assert [s.prompt for s in samples] == ["P1", "P1", "P2", "P2"]


@pytest.mark.parametrize("reason", [None, "abort", "repetition"])
def test_unexpected_finish_reasons_are_errors_not_guesses(reason) -> None:
    outputs = [fake_request(3, [("a", [1], reason)])]
    with pytest.raises(ValueError, match="finish reason"):
        samples_from_vllm_outputs([problem(1)], ["P"], outputs)


def test_vllm_conversion_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        samples_from_vllm_outputs([problem(1)], ["P1"], [])


def test_plain_sampling_parameters_are_set_explicitly_for_both_backends() -> None:
    config = SamplingConfig(n_samples=5, temperature=0.8, top_p=0.9, max_new_tokens=77)
    assert vllm_sampling_kwargs(config) == {
        "n": 5,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": 77,
    }
    hf = hf_generation_config(config, {7, 9}, pad_token_id=3)
    assert (hf.temperature, hf.top_p, hf.top_k, hf.repetition_penalty) == (0.8, 0.9, 0, 1.0)
    assert (hf.do_sample, hf.max_new_tokens, hf.num_return_sequences) == (True, 77, 5)
    assert hf.eos_token_id == [7, 9] and hf.pad_token_id == 3


def test_greedy_temperature_is_rejected_before_any_model_is_loaded() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_responses([problem(1)], PromptConfig().template, SamplingConfig(temperature=0.0))


# --- chat template ---


def test_policy_prompt_uses_the_chat_template_and_asks_for_a_boxed_answer() -> None:
    config = SamplingConfig(revision=QWEN_REVISION)
    prompts = build_policy_prompts([problem(3)], PromptConfig().template, load_tokenizer(config))
    (prompt,) = prompts
    assert "<|im_start|>user\nWhat is 3+3?" in prompt
    assert "\\boxed{}" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")  # generation prompt added


# --- real generation on CPU ---


@pytest.mark.slow
def test_hf_backend_generates_n_samples_per_problem_reproducibly() -> None:
    # Prompts of different lengths, so left padding is actually exercised.
    long_question = "A farmer has 12 cows and buys 7 more. Then he sells 5. How many are left?"
    problems = [problem(1), Problem("p2", long_question, "14", "test")]
    config = SamplingConfig(
        revision=QWEN_REVISION, n_samples=2, max_new_tokens=16, batch_size=2, seed=3
    )
    first = sample_responses(problems, PromptConfig().template, config)
    second = sample_responses(problems, PromptConfig().template, config)

    assert [(s.problem_id, s.sample_idx) for s in first] == [
        ("p1", 0),
        ("p1", 1),
        ("p2", 0),
        ("p2", 1),
    ]
    assert [s.response for s in first] == [s.response for s in second]  # same seed, same CPU
    for s in first:
        assert s.finish_reason in {"stop", "length"}
        assert 0 < s.n_response_tokens <= 16
        if s.finish_reason == "length":
            assert s.n_response_tokens == 16
    assert first[0].prompt == first[1].prompt != first[2].prompt
    # Prompt token counts are the unpadded lengths, even though the batch was padded.
    tokenizer = load_tokenizer(config)
    expected = [len(tokenizer(s.prompt, add_special_tokens=False)["input_ids"]) for s in first]
    assert [s.n_prompt_tokens for s in first] == expected
    assert first[0].n_prompt_tokens != first[2].n_prompt_tokens


@pytest.mark.slow
def test_explicit_sampling_values_override_the_models_own_generation_defaults() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", revision=QWEN_REVISION, dtype=torch.float32
    )
    # Qwen ships non-default sampling settings; if this stops being true the check
    # below would prove nothing.
    assert (model.generation_config.top_k, model.generation_config.repetition_penalty) != (0, 1.0)

    config = SamplingConfig(revision=QWEN_REVISION)
    wanted = hf_generation_config(config, {151645}, pad_token_id=151643)
    effective, _ = model._prepare_generation_config(wanted)  # what generate() will really use
    assert effective.temperature == config.temperature == 1.0
    assert effective.top_p == config.top_p == 1.0
    assert effective.top_k == 0
    assert effective.repetition_penalty == 1.0
