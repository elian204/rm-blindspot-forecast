"""Reading and writing result files, always together with config and commit hash."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """Write dicts or dataclasses as JSON lines. Returns the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if dataclasses.is_dataclass(row) and not isinstance(row, type):
                row = dataclasses.asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _git(*args: str) -> str | None:
    """Run a git command in the repo; None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """Commit hash and whether the working tree had uncommitted changes."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
    }


def write_meta(out_dir: str | Path, config: dict[str, Any], stage: str, **extra: Any) -> Path:
    """Save config + commit hash for one pipeline stage as ``meta_<stage>.json``.

    Each stage writes its own file, so a results directory whose samples came
    from an earlier run (possibly at another commit) still records where each
    file came from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **git_info(),
        "config": config,
        **extra,
    }
    path = out_dir / f"meta_{stage}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, to tie results to the exact input file they were made from."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
