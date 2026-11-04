"""Provenance metadata attached to every measurement, benchmark and training run."""
from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import __version__

ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def host_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__}


def provenance(**extra) -> dict:
    d = {"titan_version": __version__, "commit": git_sha(),
         "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "host": host_info()}
    d.update(extra)
    return d
