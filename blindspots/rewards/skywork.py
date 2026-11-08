"""Adapter for Skywork-o1-Open-PRM-Qwen-2.5-1.5B (a process reward model).

Follows the model card and the official inference code
(https://github.com/SkyworkAI/skywork-o1-prm-inference, Apache-2.0,
``model_utils/io_utils.py`` and ``model_utils/prm_model.py``):

* Input is ``<|im_start|>`` + problem + ``"\\n"``, then the response one step at
  a time; every step is tokenised on its own and followed by one ``"\\n"`` token.
* The model is a Qwen2 backbone with a linear value head. The score of a step is
  ``sigmoid(value_head(hidden state at that step's "\\n" token))``.

Steps are the response's lines (see :func:`split_steps`). The only intended
differences from the official code: line endings are normalised, and blank
(whitespace-only) lines do not become steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

from blindspots.config import RewardConfig
from blindspots.rewards import RewardScores, combine_steps

STEP_DELIMITER = "\n"
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def split_steps(response: str) -> list[str]:
    """Split a response into the steps the PRM expects. Deterministic rule:

    1. Normalise line endings (``\\r\\n`` and ``\\r`` become ``\\n``).
    2. Every line is one step, as in the model card (``step_token="\\n"``).
    3. Lines that are empty or only whitespace are dropped, so paragraph breaks
       (``"\\n\\n"``) and a trailing newline do not create phantom steps that the
       model never saw in this form.
    4. A line is kept exactly as written (indentation included).
    5. A response with no non-blank line is a single empty step, which is what
       the official code does for the empty string; there is always >= 1 step.

    A response without any line break is therefore one step.
    """
    text = response.replace("\r\n", "\n").replace("\r", "\n")
    steps = [line for line in text.split(STEP_DELIMITER) if line.strip()]
    return steps or [""]


def build_input_ids(
    tokenizer: Any, problem: str, response: str, max_length: int | None = None
) -> tuple[list[int], list[int]]:
    """Token ids for one (problem, response) plus the index of each step's last token."""
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + STEP_DELIMITER)
    step_token_id = tokenizer.encode(STEP_DELIMITER)[-1]
    ids = list(prompt_ids)
    step_ends: list[int] = []
    for step in split_steps(response):
        ids.extend(tokenizer.encode(step) if step else [])
        ids.append(step_token_id)
        step_ends.append(len(ids) - 1)
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f"Input has {len(ids)} tokens, more than the reward model's limit of {max_length}"
        )
    return ids, step_ends


class PRMWithValueHead(nn.Module):
    """Qwen2 backbone + linear value head; returns one probability per token."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.v_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Probability (0 to 1) from the value head at every token position."""
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        # The head runs in float32 whatever the backbone dtype, for stable scores.
        return torch.sigmoid(self.v_head(hidden.float()).squeeze(-1))


def load_skywork_prm(
    model_id: str, device: str, dtype: str, revision: str | None = None
) -> tuple[PRMWithValueHead, Any]:
    """Load backbone, value head and tokenizer from the checkpoint (``pytorch_model.bin``)."""
    path = snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=["*.json", "*.txt", "*.bin"],
    )
    config = Qwen2Config.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    state = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
    model = PRMWithValueHead(config)
    backbone = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.model.load_state_dict(backbone, strict=True)
    model.v_head.load_state_dict(
        {
            k.removeprefix("v_head.summary."): v
            for k, v in state.items()
            if k.startswith("v_head.summary.")
        },
        strict=True,
    )
    model.model.to(device=device, dtype=_DTYPES[dtype])
    model.v_head.to(device=device, dtype=torch.float32)
    return model.eval(), tokenizer


class SkyworkPRM:
    """Skywork PRM behind the common ``score(prompts, responses)`` interface."""

    def __init__(self, config: RewardConfig) -> None:
        if config.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        self.config = config
        self.model, self.tokenizer = load_skywork_prm(
            config.model_id, config.device, config.dtype, config.revision
        )

    def step_scores(self, prompts: Sequence[str], responses: Sequence[str]) -> list[list[float]]:
        """Per-step probabilities for each response."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        encoded = [
            build_input_ids(self.tokenizer, p, r) for p, r in zip(prompts, responses, strict=True)
        ]
        too_long = [
            (i, len(ids)) for i, (ids, _) in enumerate(encoded) if len(ids) > self.config.max_length
        ]
        if too_long:
            # Reported all at once, before any model compute is spent.
            raise ValueError(
                f"{len(too_long)} of {len(encoded)} inputs exceed the reward model's limit of "
                f"{self.config.max_length} tokens (index, tokens): {too_long[:10]}"
            )
        results: list[list[float]] = []
        device = torch.device(self.config.device)
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(encoded), self.config.batch_size):
            batch = encoded[start : start + self.config.batch_size]
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (ids, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention_mask[row, : len(ids)] = 1
            with torch.inference_mode():
                probs = self.model(input_ids.to(device), attention_mask.to(device)).cpu()
            for row, (_, ends) in enumerate(batch):
                results.append(probs[row, ends].tolist())
        return results

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> RewardScores:
        """Per-step probabilities and one combined grade per response."""
        steps = self.step_scores(prompts, responses)
        grades = [combine_steps(s, self.config.combine) for s in steps]
        return RewardScores(grades=grades, step_scores=steps)
