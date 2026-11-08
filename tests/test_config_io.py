"""Config loading and result-file helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blindspots.config import (
    RunConfig,
    config_to_dict,
    hide_gpus_unless_requested,
    load_config,
    wants_gpu,
)
from blindspots.io import git_info, read_jsonl, write_jsonl, write_meta

CONFIGS = Path(__file__).parent.parent / "configs"


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_demo_config_loads_and_asks_for_boxed_answers() -> None:
    config = load_config(CONFIGS / "demo_cpu.yaml")
    assert config.data.dataset == "gsm8k" and config.data.n_problems == 10
    assert (config.sampling.n_samples, config.sampling.max_new_tokens) == (4, 256)
    assert config.sampling.device == "cpu" and config.reward.device == "cpu"
    assert "\\boxed{}" in config.prompt.template  # braces survive YAML unchanged
    assert "{problem}" in config.prompt.template
    assert config.sampling.revision and config.reward.revision  # model versions are pinned


def test_gpu_example_differs_in_backend_device_dtype_and_run_size_not_in_prompt_or_rm() -> None:
    cpu, gpu = load_config(CONFIGS / "demo_cpu.yaml"), load_config(CONFIGS / "gpu_example.yaml")
    assert gpu.sampling.backend == "vllm" and gpu.reward.device == "cuda"
    assert cpu.reward.model_id == gpu.reward.model_id
    assert cpu.prompt == gpu.prompt


def test_missing_sections_fall_back_to_defaults(tmp_path) -> None:
    config = load_config(write_yaml(tmp_path, "data:\n  dataset: math500\n"))
    assert config.data.dataset == "math500"
    assert config.sampling.n_samples == 8 and config.sampling.temperature == 1.0
    assert config.reward.combine == "min"


def test_unknown_keys_are_rejected_to_catch_typos(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_sample"):
        load_config(write_yaml(tmp_path, "sampling:\n  n_sample: 4\n"))
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(write_yaml(tmp_path, "samplng:\n  n_samples: 4\n"))


@pytest.mark.parametrize(
    "text",
    [
        "reward:\n  combine: median\n",
        "sampling:\n  backend: tgi\n",
        "audit:\n  top_fraction: 1.5\n",
        "sampling:\n  n_samples: 0\n",
        "sampling:\n  temperature: 0\n",
        "sampling:\n  top_p: 0\n",
        "sampling:\n  batch_size: 0\n",
        "sampling:\n  max_new_tokens: 0\n",
        "sampling:\n  dtype: bf16\n",
        "sampling:\n  device: gpu\n",
        "reward:\n  dtype: bf16\n",
        "reward:\n  device: gpu\n",
        "reward:\n  batch_size: 0\n",
        "reward:\n  name: some_other_rm\n",
        "data:\n  dataset: aime\n",
        "data:\n  n_problems: 0\n",
        "prompt:\n  template: no placeholder\n",
    ],
)
def test_invalid_values_fail_at_load_time(tmp_path, text: str) -> None:
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, text))


def test_gpu_visibility_follows_the_config(monkeypatch) -> None:
    cpu, gpu = load_config(CONFIGS / "demo_cpu.yaml"), load_config(CONFIGS / "gpu_example.yaml")
    assert not wants_gpu(cpu) and wants_gpu(gpu)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    hide_gpus_unless_requested(gpu)
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0,1"  # untouched
    hide_gpus_unless_requested(cpu)
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == ""


def test_tests_run_without_gpus() -> None:
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == ""


def test_default_config_dict_round_trips_through_json() -> None:
    assert json.loads(json.dumps(config_to_dict(RunConfig())))["sampling"]["temperature"] == 1.0


# --- result files ---


def test_jsonl_round_trip_keeps_unicode(tmp_path) -> None:
    rows = [{"a": 1, "text": "élan \\boxed{5}"}, {"a": 2, "text": ""}]
    assert write_jsonl(tmp_path / "x" / "r.jsonl", rows) == 2
    assert list(read_jsonl(tmp_path / "x" / "r.jsonl")) == rows


def test_meta_file_holds_config_and_commit(tmp_path) -> None:
    path = write_meta(tmp_path, config_to_dict(RunConfig()), "test")
    assert path.name == "meta_test.json"
    meta = json.loads(path.read_text())
    assert meta["config"]["sampling"]["n_samples"] == 8
    assert meta["stage"] == "test"
    assert set(git_info()) == {"git_commit", "git_dirty"}
    assert meta["git_commit"] == git_info()["git_commit"]
    assert len(meta["git_commit"]) == 40  # a real commit hash (tests run inside the repo)
