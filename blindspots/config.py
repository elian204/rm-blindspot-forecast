"""Run configuration: small YAML files loaded into plain dataclasses.

Everything that changes between a CPU demo and a GPU run (device, dtype,
sampling backend, batch sizes) lives here, so moving to GPUs is a config change.
"""

from __future__ import annotations

import dataclasses
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

COMBINE_RULES = ("min", "last", "prod", "mean")
SAMPLING_BACKENDS = ("hf", "vllm")
DATASETS = ("gsm8k", "math500")
REWARD_NAMES = ("skywork_prm",)
DTYPES = ("float32", "float16", "bfloat16")
_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")


@dataclass
class DataConfig:
    """Which problems to load."""

    dataset: str = "gsm8k"  # "gsm8k" or "math500"
    split: str = "test"
    n_problems: int | None = None  # take the first n problems; None means all


@dataclass
class PromptConfig:
    """Prompt template shown to the policy. Versioned together with the run."""

    # Must contain the "{problem}" placeholder. It is filled with a plain string
    # replace (not str.format), so other braces such as \boxed{} stay literal.
    template: str = "{problem}\n\nReason step by step, and put your final answer within \\boxed{}."


@dataclass
class SamplingConfig:
    """How to sample answers from the policy."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None  # pin a Hub commit for reproducibility
    backend: str = "hf"  # "hf" (CPU demos and tests) or "vllm" (GPU runs)
    n_samples: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    seed: int = 0
    device: str = "cpu"  # hf backend only; vLLM picks its devices itself
    dtype: str = "float32"  # "float32", "float16" or "bfloat16"
    batch_size: int = 8  # hf backend only: prompts per generate() call
    # vLLM-only settings, ignored by the hf backend.
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None


@dataclass
class RewardConfig:
    """Which reward model to use and how to turn step scores into one grade."""

    name: str = "skywork_prm"
    model_id: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    revision: str | None = None  # pin a Hub commit for reproducibility
    # How per-step scores are combined into one grade. This changes which
    # answers get rewarded, so it is always an explicit choice.
    combine: str = "min"  # one of COMBINE_RULES
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    max_length: int = 4096  # the model's context window


@dataclass
class AuditConfig:
    """Audit settings."""

    top_fraction: float = 0.25  # "highly graded" means the top 25% of RM grades
    output_dir: str = "results/audit"


@dataclass
class RunConfig:
    """Everything one audit run needs."""

    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def validate(self) -> None:
        """Fail early on values that would only break deep inside a run."""
        d, p, sa, r, a = self.data, self.prompt, self.sampling, self.reward, self.audit
        _require(d.dataset in DATASETS, f"data.dataset must be one of {DATASETS}")
        _require(d.n_problems is None or d.n_problems >= 1, "data.n_problems must be >= 1")
        _require("{problem}" in p.template, "prompt.template must contain '{problem}'")
        _require(
            sa.backend in SAMPLING_BACKENDS, f"sampling.backend must be in {SAMPLING_BACKENDS}"
        )
        _require(sa.n_samples >= 1, "sampling.n_samples must be at least 1")
        _require(sa.temperature > 0, "sampling.temperature must be > 0")
        _require(0 < sa.top_p <= 1, "sampling.top_p must be in (0, 1]")
        _require(sa.max_new_tokens >= 1, "sampling.max_new_tokens must be >= 1")
        _require(sa.batch_size >= 1, "sampling.batch_size must be >= 1")
        _require(sa.dtype in DTYPES, f"sampling.dtype must be one of {DTYPES}")
        _require(
            bool(_DEVICE.match(sa.device)), "sampling.device must be 'cpu', 'cuda' or 'cuda:N'"
        )
        _require(r.name in REWARD_NAMES, f"reward.name must be one of {REWARD_NAMES}")
        _require(r.combine in COMBINE_RULES, f"reward.combine must be one of {COMBINE_RULES}")
        _require(r.dtype in DTYPES, f"reward.dtype must be one of {DTYPES}")
        _require(bool(_DEVICE.match(r.device)), "reward.device must be 'cpu', 'cuda' or 'cuda:N'")
        _require(r.batch_size >= 1, "reward.batch_size must be >= 1")
        _require(r.max_length >= 1, "reward.max_length must be >= 1")
        _require(0.0 < a.top_fraction < 1.0, "audit.top_fraction must be in (0, 1)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build(cls: type[T], values: dict[str, Any] | None, where: str) -> T:
    """Build a dataclass from a dict, rejecting unknown keys (typos in YAML)."""
    values = values or {}
    hints = typing.get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"Unknown config keys in '{where}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        nested = hints[name]
        if dataclasses.is_dataclass(nested):
            kwargs[name] = _build(nested, value, f"{where}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML config file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = _build(RunConfig, raw, "root")
    config.validate()
    return config


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Plain-dict view of a config, for saving next to results."""
    return dataclasses.asdict(config)


def wants_gpu(config: RunConfig) -> bool:
    """Does any stage of this config use a GPU?"""
    return (
        config.sampling.backend == "vllm"
        or config.sampling.device.startswith("cuda")
        or config.reward.device.startswith("cuda")
    )


def hide_gpus_unless_requested(config: RunConfig) -> None:
    """Make GPUs invisible unless the config asks for them.

    Scripts call this right after loading the config and BEFORE importing torch
    or vLLM (they read CUDA_VISIBLE_DEVICES once, at import). A CPU config can
    therefore never touch a GPU by accident; a GPU config just works.
    """
    if not wants_gpu(config):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
