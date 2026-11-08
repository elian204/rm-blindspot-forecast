"""Audit for one (policy, reward model) pair.

Examples:
    uv run python scripts/audit.py --config configs/demo_cpu.yaml
    uv run python scripts/audit.py --config configs/gpu_example.yaml \\
        --samples results/x/samples.jsonl
    uv run python scripts/audit.py --records results/x/records.jsonl     # statistics only

"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")  # shared machines: avoid CPU oversubscription

from blindspots.config import hide_gpus_unless_requested, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", help="YAML config (sampling + reward model + audit)")
    parser.add_argument("--samples", help="reuse samples.jsonl instead of sampling")
    parser.add_argument("--records", help="only recompute statistics from a records.jsonl")
    parser.add_argument("--top-fraction", type=float, default=0.25, help="used with --records")
    parser.add_argument("--output-dir", default=None, help="overrides audit.output_dir")
    args = parser.parse_args()

    if args.records:  # statistics only: no model, no GPU
        from blindspots.audit import AnswerRecord, run_audit
        from blindspots.io import read_jsonl
        from blindspots.pipeline import print_summary

        records = [AnswerRecord(**row) for row in read_jsonl(args.records)]
        summary, _, _ = run_audit(records, top_fraction=args.top_fraction)
        print_summary(summary)
        return
    if not args.config:
        parser.error("--config is required unless --records is given")

    config = load_config(args.config)
    hide_gpus_unless_requested(config)  # must happen before torch / vLLM are imported

    from blindspots.pipeline import print_summary, run_pipeline

    summary, _, _ = run_pipeline(config, output_dir=args.output_dir, samples_path=args.samples)
    print()
    print_summary(summary)
    print(f"\nResults in {args.output_dir or config.audit.output_dir}")


if __name__ == "__main__":
    main()
