"""Sample responses only (stage 1 of a GPU run). Writes samples.jsonl."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")  # shared machines: avoid CPU oversubscription

from blindspots.config import config_to_dict, hide_gpus_unless_requested, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None, help="overrides audit.output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    hide_gpus_unless_requested(config)  # must happen before torch / vLLM are imported

    from blindspots.data import load_problems
    from blindspots.io import write_jsonl, write_meta
    from blindspots.sampling import sample_responses

    out_dir = Path(args.output_dir or config.audit.output_dir)
    problems = load_problems(config.data.dataset, config.data.split, config.data.n_problems)
    samples = sample_responses(problems, config.prompt.template, config.sampling)
    n = write_jsonl(out_dir / "samples.jsonl", samples)
    write_meta(out_dir, config_to_dict(config), "sample")
    print(f"Wrote {n} samples to {out_dir / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
