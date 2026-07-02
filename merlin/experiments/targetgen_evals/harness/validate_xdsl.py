"""Check xDSL artifact presence and collect stats."""

from __future__ import annotations

import re
from pathlib import Path


_OP_PATTERNS = [
    re.compile(r"@irdl_op_definition"),
    re.compile(r"class \w+\(IRDLOperation\)"),
    re.compile(r"class \w+\(Operation\)"),
]


def _estimate_ops(py_file: Path) -> int:
    text = py_file.read_text(errors="replace")
    return sum(1 for pat in _OP_PATTERNS if pat.search(text))


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    xdsl_dir = run_dir / "generated" / f"{target}-mlir" / "xdsl"

    metrics: dict = {
        "validator": "xdsl",
        "xdsl_dir_exists": False,
        "xdsl_files": 0,
        "xdsl_op_estimate": 0,
        "errors": [],
    }

    if not xdsl_dir.exists():
        metrics["errors"].append(
            f"xdsl/ directory does not exist under generated/{target}-mlir/; "
            "no xDSL artifacts produced yet"
        )
        return metrics

    py_files = list(xdsl_dir.rglob("*.py"))
    metrics["xdsl_dir_exists"] = True
    metrics["xdsl_files"] = len(py_files)
    metrics["xdsl_op_estimate"] = sum(_estimate_ops(f) for f in py_files)

    if not py_files:
        metrics["errors"].append("xdsl/ directory exists but contains no .py files")

    return metrics
