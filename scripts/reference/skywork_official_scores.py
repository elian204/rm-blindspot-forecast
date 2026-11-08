"""Run the OFFICIAL Skywork inference code on the model card's examples.

This produces the reference numbers that tests/test_skywork_slow.py compares
our adapter against, because the model card prints no expected values.

It must run in a separate environment that matches the official code, since the
official wrapper targets older transformers versions than this project uses:

    uv venv --python 3.11 skywork-ref
    VIRTUAL_ENV=skywork-ref uv pip install torch==2.5.1 \\
        --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=skywork-ref uv pip install transformers==4.46.3 accelerate==1.1.1 \\
        "numpy<2" "huggingface_hub<1.0" safetensors
    git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git   # commit 719b56b
    CUDA_VISIBLE_DEVICES="" skywork-ref/bin/python scripts/reference/skywork_official_scores.py \
        --official-repo skywork-o1-prm-inference

Usage follows the model card, except that ``device_map="auto"`` is dropped: on a
machine without GPUs the official wrapper refuses CPU placement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "skywork_card_examples.json"
MODEL_ID = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, help="clone of skywork-o1-prm-inference")
    parser.add_argument("--out", default=str(FIXTURE.with_name("skywork_card_reference.json")))
    args = parser.parse_args()

    sys.path.insert(0, args.official_repo)
    from model_utils.io_utils import (  # official code
        derive_step_rewards,
        prepare_batch_input_for_model,
        prepare_input,
    )
    from model_utils.prm_model import PRM_MODEL

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    datas = fixture["examples"]
    revision = fixture["model_revision"]
    prm_model_path = snapshot_download(MODEL_ID, revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
    processed = [
        prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n")
        for d in datas
    ]
    input_ids, steps, reward_flags = zip(*processed, strict=True)

    model = PRM_MODEL.from_pretrained(prm_model_path).eval()
    input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(
        input_ids, reward_flags, tokenizer.pad_token_id
    )
    with torch.no_grad():
        _, _, rewards = model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
    step_rewards = derive_step_rewards(rewards, reward_flags)

    commit = subprocess.run(
        ["git", "-C", args.official_repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    result = {
        "note": "Official Skywork inference code on the model card examples (fp32, CPU).",
        "official_repo_commit": commit,
        "model_revision": revision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "steps": [[s for s in example_steps] for example_steps in steps],
        "step_rewards": step_rewards,
    }
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for r in step_rewards:
        print("step_rewards:", r)


if __name__ == "__main__":
    main()
