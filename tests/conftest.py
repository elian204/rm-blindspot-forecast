"""Test setup: force CPU and cap CPU threads. Must run before torch is imported."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Shared machines can be oversubscribed; many spinning threads make CPU runs crawl.
os.environ.setdefault("OMP_NUM_THREADS", "8")
