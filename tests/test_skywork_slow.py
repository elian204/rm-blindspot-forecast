"""Slow test: load the real Skywork PRM on CPU (fp32) and reproduce the card's example.

The model card prints no expected numbers, so the reference values in
``fixtures/skywork_card_reference.json`` come from running the official
inference code (see scripts/reference/skywork_official_scores.py).

Skip with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blindspots.config import RewardConfig
from blindspots.rewards.skywork import SkyworkPRM

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_REVISION = "98d69606595eedbdbbbf0a7d28efdcd462ba6a67"


@pytest.mark.slow
def test_adapter_reproduces_official_step_scores_on_the_model_card_example() -> None:
    examples = json.loads((FIXTURES / "skywork_card_examples.json").read_text())["examples"]
    reference = json.loads((FIXTURES / "skywork_card_reference.json").read_text())
    assert reference["model_revision"] == MODEL_REVISION

    rm = SkyworkPRM(
        RewardConfig(
            revision=MODEL_REVISION, device="cpu", dtype="float32", combine="min", batch_size=2
        )
    )
    scores = rm.score([e["problem"] for e in examples], [e["response"] for e in examples])

    assert scores.step_scores is not None
    for got, want in zip(scores.step_scores, reference["step_rewards"], strict=True):
        assert got == pytest.approx(want, abs=1e-4)
    # The combined grade follows the configured rule.
    for grade, want in zip(scores.grades, reference["step_rewards"], strict=True):
        assert grade == pytest.approx(min(want), abs=1e-4)


@pytest.mark.slow
def test_blank_lines_and_crlf_do_not_change_the_scores() -> None:
    """The one intentional difference from the official code: it must not change scores."""
    example = json.loads((FIXTURES / "skywork_card_examples.json").read_text())["examples"][1]
    response = example["response"]
    noisy = "\n\n".join(response.split("\n")).replace("\n\n", "\r\n\r\n") + "\n\n"
    rm = SkyworkPRM(
        RewardConfig(revision=MODEL_REVISION, device="cpu", dtype="float32", batch_size=2)
    )
    clean, messy = rm.step_scores([example["problem"]] * 2, [response, noisy])
    assert len(clean) == len(messy) == 16
    assert messy == pytest.approx(clean, abs=1e-4)
