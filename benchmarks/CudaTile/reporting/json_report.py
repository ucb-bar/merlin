"""Structured JSON report generation and loading."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _get_metadata(sm_arch: str, build_dir: Path) -> dict:
    gpu_name = "unknown"
    cuda_version = "unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            cuda_version = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    iree_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(build_dir.parent) if build_dir.exists() else ".",
        )
        if result.returncode == 0:
            iree_commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    branch = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(build_dir.parent) if build_dir.exists() else ".",
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "sm_arch": sm_arch,
        "driver_version": cuda_version,
        "hostname": platform.node(),
        "merlin_commit": iree_commit,
        "merlin_branch": branch,
    }


def save_json_report(
    results: list[dict],
    output_path: Path,
    sm_arch: str,
    build_dir: Path,
) -> None:
    report = {
        "metadata": _get_metadata(sm_arch, build_dir),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def load_json_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
