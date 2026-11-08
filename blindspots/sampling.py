"""Sample several answers per problem from a policy.

Two backends behind one function:

* ``vllm``: fast, for GPU runs.
* ``hf``: plain transformers ``generate``, for CPU demos and tests only.

Both use the policy's own chat template, sample from the unmodified model
distribution (temperature and top-p from the config; no top-k and no repetition
penalty, which some models enable by default in their generation config), and
return the same :class:`Sample` records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from blindspots.config import SamplingConfig
from blindspots.data import Problem, render_prompt


@dataclass
class Sample:
    """One sampled response."""

    problem_id: str
    sample_idx: int
    prompt: str  # exactly the text the policy saw (chat template applied)
    response: str
    finish_reason: str  # "stop" (ended by itself) or "length" (hit max_new_tokens)
    n_prompt_tokens: int
    n_response_tokens: int  # includes the end-of-sequence token when there is one


_TORCH_DTYPES = ("float32", "float16", "bfloat16")


def load_tokenizer(config: SamplingConfig) -> Any:
    """Tokenizer of the policy, which also provides its chat template."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)


def build_policy_prompts(problems: Sequence[Problem], template: str, tokenizer: Any) -> list[str]:
    """Render each problem into the policy's chat format (user turn + generation prompt)."""
    prompts = []
    for problem in problems:
        message = [{"role": "user", "content": render_prompt(template, problem.question)}]
        prompts.append(
            tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def sample_responses(
    problems: Sequence[Problem], prompt_template: str, config: SamplingConfig
) -> list[Sample]:
    """Sample ``config.n_samples`` responses for every problem.

    Output order: problems in input order, then ``sample_idx`` 0..n-1.
    """
    if config.temperature <= 0:
        raise ValueError(
            "temperature must be > 0 (this module samples, it does not decode greedily)"
        )
    tokenizer = load_tokenizer(config)
    prompts = build_policy_prompts(problems, prompt_template, tokenizer)
    if config.backend == "vllm":
        return _sample_vllm(problems, prompts, tokenizer, config)
    if config.backend == "hf":
        return _sample_hf(problems, prompts, tokenizer, config)
    raise ValueError(f"Unknown sampling backend '{config.backend}'")


# --------------------------------------------------------------------------
# vLLM backend (GPU runs)
# --------------------------------------------------------------------------


def _check_finish_reason(reason: str | None) -> str:
    """Accept only the two reasons we can interpret; anything else is an error, not a guess."""
    if reason not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason from the sampler: {reason!r}")
    return reason


def vllm_sampling_kwargs(config: SamplingConfig) -> dict[str, Any]:
    """Arguments for ``vllm.SamplingParams``: plain sampling, set explicitly.

    ``top_k=0`` disables top-k and ``repetition_penalty=1.0`` disables the
    penalty (both are documented ``SamplingParams`` fields in vLLM 0.33).
    """
    return {
        "n": config.n_samples,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": config.max_new_tokens,
    }


def samples_from_vllm_outputs(
    problems: Sequence[Problem], prompts: Sequence[str], outputs: Sequence[Any]
) -> list[Sample]:
    """Convert vLLM ``RequestOutput`` objects into :class:`Sample` records.

    Uses only documented fields: ``prompt_token_ids`` and ``outputs`` on the
    request, and ``text``, ``token_ids`` and ``finish_reason`` on each completion.
    ``LLM.generate`` returns one output per prompt, in prompt order.
    """
    if not len(problems) == len(prompts) == len(outputs):
        raise ValueError("problems, prompts and outputs must have the same length")
    samples = []
    for problem, prompt, request in zip(problems, prompts, outputs, strict=True):
        for idx, completion in enumerate(request.outputs):
            samples.append(
                Sample(
                    problem_id=problem.problem_id,
                    sample_idx=idx,
                    prompt=prompt,
                    response=completion.text,
                    finish_reason=_check_finish_reason(completion.finish_reason),
                    n_prompt_tokens=len(request.prompt_token_ids),
                    n_response_tokens=len(completion.token_ids),
                )
            )
    return samples


def _sample_vllm(
    problems: Sequence[Problem], prompts: Sequence[str], tokenizer: Any, config: SamplingConfig
) -> list[Sample]:
    from vllm import LLM, SamplingParams  # needs the "vllm" extra and a GPU

    llm = LLM(
        model=config.model_id,
        revision=config.revision,
        dtype=config.dtype,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        seed=config.seed,
    )
    # Token ids come from the same tokenizer call as in the hf backend (no special
    # tokens added: the chat template already contains them), so both backends feed
    # the model identical inputs.
    token_ids = tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids],
        SamplingParams(**vllm_sampling_kwargs(config)),
    )
    return samples_from_vllm_outputs(problems, prompts, outputs)


# --------------------------------------------------------------------------
# transformers backend (CPU demos and tests)
# --------------------------------------------------------------------------


def split_generated(generated: Sequence[int], eos_ids: set[int]) -> tuple[list[int], str]:
    """Cut a generated row at its first end-of-sequence token.

    Returns ``(tokens, finish_reason)``. The tokens include the EOS token if one
    was produced (matching how vLLM counts); everything after it is padding.
    """
    for i, token in enumerate(generated):
        if token in eos_ids:
            return list(generated[: i + 1]), "stop"
    return list(generated), "length"


def hf_generation_config(config: SamplingConfig, eos_ids: set[int], pad_token_id: int) -> Any:
    """Plain-sampling ``GenerationConfig``: every sampling field is set explicitly.

    transformers fills fields that are left unset from the model's own
    ``generation_config.json`` (Qwen ships top_k=20, top_p=0.8 and
    repetition_penalty=1.1), which would silently change the distribution.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=0,  # 0 disables top-k
        repetition_penalty=1.0,
        max_new_tokens=config.max_new_tokens,
        num_return_sequences=config.n_samples,
        pad_token_id=pad_token_id,
        eos_token_id=sorted(eos_ids),
    )


def _sample_hf(
    problems: Sequence[Problem], prompts: Sequence[str], tokenizer: Any, config: SamplingConfig
) -> list[Sample]:
    import torch
    from transformers import AutoModelForCausalLM

    if config.dtype not in _TORCH_DTYPES:
        raise ValueError(f"dtype must be one of {_TORCH_DTYPES}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.revision, dtype=getattr(torch, config.dtype)
    )
    model.to(config.device).eval()

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    eos_ids.discard(None)
    generation_config = hf_generation_config(config, eos_ids, tokenizer.pad_token_id)
    tokenizer.padding_side = "left"  # generation continues from the right edge

    samples: list[Sample] = []
    for start in range(0, len(prompts), config.batch_size):
        batch_prompts = list(prompts[start : start + config.batch_size])
        batch_problems = problems[start : start + config.batch_size]
        # The seed depends on the batch start, so reruns of the same config match
        # on the same hardware regardless of what ran before.
        torch.manual_seed(config.seed + start)
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model.generate(
                **{k: v.to(config.device) for k, v in enc.items()},
                generation_config=generation_config,
            )
        width = enc["input_ids"].shape[1]
        prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()
        # Rows are grouped by prompt: n_samples consecutive rows per prompt.
        for row, sequence in enumerate(out):
            prompt_idx, sample_idx = divmod(row, config.n_samples)
            tokens, finish_reason = split_generated(sequence[width:].tolist(), eos_ids)
            text_tokens = tokens[:-1] if finish_reason == "stop" else tokens  # drop the EOS
            samples.append(
                Sample(
                    problem_id=batch_problems[prompt_idx].problem_id,
                    sample_idx=sample_idx,
                    prompt=batch_prompts[prompt_idx],
                    response=tokenizer.decode(text_tokens, skip_special_tokens=True),
                    finish_reason=finish_reason,
                    n_prompt_tokens=int(prompt_lengths[prompt_idx]),
                    n_response_tokens=len(tokens),
                )
            )
    return samples
