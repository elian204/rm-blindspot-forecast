"""Reward-model adapters with one common interface.

Every adapter exposes ``score(prompts, responses) -> RewardScores``. The
``prompts`` are the raw problem texts (not the policy's chat-formatted prompt),
and each adapter re-renders problem and response in its own reward model's
format, so a grade never depends on which policy wrote the response.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from blindspots.config import COMBINE_RULES, RewardConfig


@dataclass
class RewardScores:
    """Grades for a batch of responses, plus per-step scores for process RMs."""

    grades: list[float]
    step_scores: list[list[float]] | None = None  # None for outcome reward models


class RewardModel(Protocol):
    """What the audit needs from a reward model."""

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> RewardScores:
        """Grade each ``responses[i]`` to problem ``prompts[i]``."""
        ...


def combine_steps(step_scores: Sequence[float], rule: str) -> float:
    """Turn per-step scores into one grade.

    * ``min``: the weakest step decides.
    * ``last``: only the final step counts.
    * ``prod``: product of all steps (long answers are penalised).
    * ``mean``: average over steps (how the Skywork model card weights PRM best-of-N).

    The choice changes which answers get rewarded, so it is always explicit.
    """
    if not step_scores:
        raise ValueError("Cannot combine an empty list of step scores")
    if rule == "min":
        return float(min(step_scores))
    if rule == "last":
        return float(step_scores[-1])
    if rule == "prod":
        return float(math.prod(step_scores))
    if rule == "mean":
        return float(sum(step_scores) / len(step_scores))
    raise ValueError(f"Unknown combine rule '{rule}' (expected one of {COMBINE_RULES})")


def load_reward_model(config: RewardConfig) -> RewardModel:
    """Build the adapter named in the config (plain dispatch, no registry)."""
    if config.name == "skywork_prm":
        from blindspots.rewards.skywork import SkyworkPRM  # lazy: pulls in torch

        return SkyworkPRM(config)
    raise ValueError(f"Unknown reward model '{config.name}' (expected 'skywork_prm')")
