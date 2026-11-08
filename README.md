# rm-blindspot-forecast

Tools for auditing a reward model against a given LLM before RL training. The audit samples the LLM's answers to math problems, grades them with the reward model, checks them against the ground truth, and surfaces answers that the reward model grades highly although they are wrong.

**Status:** early development. The foundations are in place and tested on CPU: data loading, answer checker, one reward-model adapter, sampling backends, the audit statistics and a tiny end-to-end demo. The vLLM (GPU) sampling path is written but has not been run on a GPU yet.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Linux. Python 3.11 is installed by uv if missing.

```bash
git clone https://github.com/elian204/rm-blindspot-forecast.git
cd rm-blindspot-forecast
uv sync                  # CPU demo, tests
uv sync --extra vllm     # additionally: fast sampling on GPUs
```

The default PyTorch wheels from PyPI include the CUDA libraries (several GB). Models and datasets are downloaded from the Hugging Face Hub on first use (set `HF_HOME` to choose where they are cached).

## Run the tests

```bash
uv run pytest -m "not slow"   # fast tests, about 30 s, no model weights (small tokenizer files and datasets are downloaded on first run)
uv run pytest -m slow         # loads real models on CPU (a few minutes)
```

The tests always run on CPU (`CUDA_VISIBLE_DEVICES=""` is set in `tests/conftest.py`).

## Run the demo

Qwen2.5-0.5B-Instruct answers 10 GSM8K problems (4 samples each, at most 256 new tokens), the Skywork 1.5B process reward model grades them, and the audit prints a summary table. It runs on CPU only and takes several minutes. It only shows that the pipeline works; the numbers are not meaningful at this size.

```bash
uv run python scripts/audit.py --config configs/demo_cpu.yaml
```

Results go to `results/demo_cpu/` (git-ignored):

| File | Content |
| --- | --- |
| `samples.jsonl` | prompt, response, finish reason (`stop` or `length`), token counts |
| `records.jsonl` | one line per answer: correctness, RM grade, step scores, checker flags, both advantages |
| `blind_spot_candidates.jsonl` | wrong answers with positive RM advantage, with text, grade, length, finish reason, flags |
| `summary.json` | all audit statistics |
| `meta_sample.json`, `meta_audit.json` | the full config and the git commit hash of each stage (the audit meta also records the hash of the samples file it used) |

Grades are compared at a resolution of 1e-6 (ties in AUC, the top-quartile threshold, groups without spread).

Statistics can be recomputed from a records file alone, without any model:
`uv run python scripts/audit.py --records results/demo_cpu/records.jsonl`.

## How it works

- **Config.** Small YAML files in `configs/` are loaded into dataclasses (`blindspots/config.py`). The prompt template is part of the config, so it is versioned with every run. Device, dtype and sampling backend are config fields; scripts hide the GPUs unless the config asks for them.
- **Data** (`blindspots/data.py`): GSM8K (`openai/gsm8k`) and MATH-500 (`HuggingFaceH4/MATH-500`).
- **Checker** (`blindspots/checker.py`): the final answer is the last non-empty `\boxed{...}` (otherwise the last "The answer is ..."), compared to the ground truth with [math-verify](https://github.com/huggingface/Math-Verify). A response with no answer, an answer that cannot be parsed, or a math-verify failure counts as wrong and keeps its own status (`no_answer`, `unparseable`, `check_error`). Two flags are recorded: `multiple_distinct_answers` (several boxed answers that are not equivalent; the last one still decides) and `truncated` (the generation hit the token limit).
- **Reward models** (`blindspots/rewards/`): every adapter implements `score(prompts, responses)`, where `prompts` are the raw problems. Each adapter re-renders problem and response in its own model's format, so a grade does not depend on which policy wrote the response. The Skywork adapter follows the model card: one step per non-blank line, and one score per step. A rule combines step scores into one grade (`min`, `last`, `prod` or `mean`, set in the config). Its step scores match the official inference code on the model card's example to within 1e-4 (the card prints no numbers, so the reference was produced by running the official code; see `scripts/reference/`).
- **Sampling** (`blindspots/sampling.py`): vLLM for GPU runs, plain transformers on CPU for demos and tests. Both use the policy's chat template and sample from the unmodified model distribution (temperature from the config, no top-k, no repetition penalty).
- **Audit** (`blindspots/audit.py`), per (policy, reward model) pair:
  - accuracy of the samples, with counts of missing, unparseable, truncated and multiple-answer responses;
  - the reward model's AUC at separating correct from wrong answers, within each problem (primary; averaged over problems that have both kinds of answer) and pooled over all answers;
  - the share of wrong answers among the top-graded ones (top quartile of grades by default);
  - advantages within each problem's group of answers, `(x - group mean) / group std` (sample std), once for the RM grade and once for correctness; groups with no spread get advantage 0 and are counted separately for the grade and for correctness;
  - the share of positive RM advantage that goes to wrong answers (by advantage mass and by number of answers);
  - the blind-spot candidates file.

## Layout

```
blindspots/
  config.py  data.py  checker.py  sampling.py  audit.py
  pipeline.py            # glue: sample -> check -> grade -> audit -> save
  io.py                  # JSONL, config + commit hash next to every result
  rewards/               # common interface + Skywork adapter
configs/                 # demo_cpu.yaml, gpu_example.yaml
scripts/                 # audit.py, sample.py, reference/ (official-code reference run)
tests/
docs/                    # local notes
results/                 # git-ignored
```

## Next steps once GPUs are available

1. Run the vLLM sampling path (`configs/gpu_example.yaml`: `scripts/sample.py`, then `scripts/audit.py --samples ...`) and compare it with the transformers backend on a small subset.
2. Run the audit on the full GSM8K and MATH-500 sets with Qwen2.5-0.5B/1.5B-Instruct.
3. Add an adapter for a second process reward model (`Qwen/Qwen2.5-Math-PRM-7B`).
4. Add a small training-loop integration test.

## Licenses

The code is MIT-licensed (see `LICENSE`). Models are not redistributed. The Skywork PRM is used under the Skywork Community License, and its inference approach follows the Apache-2.0 reference code at <https://github.com/SkyworkAI/skywork-o1-prm-inference>.
