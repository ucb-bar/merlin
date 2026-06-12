"""Validate the runtime mock JSON against expected golden output."""

from __future__ import annotations

import json
from pathlib import Path


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    golden_mock = (
        Path(__file__).parent.parent / "datasets" / target
        / "tests" / "runtime_mock" / "matmul_exact_i8_i32.json"
    )
    generated_mock = run_dir / "generated" / f"{target}-mlir" / "runtime_mock" / "matmul_exact_i8_i32.json"

    metrics: dict = {
        "validator": "runtime_mock",
        "golden_exists": golden_mock.exists(),
        "generated_exists": generated_mock.exists(),
        "match": None,
        "errors": [],
    }

    if not golden_mock.exists():
        metrics["errors"].append(f"Golden runtime mock not found: {golden_mock}")
        return metrics

    if not generated_mock.exists():
        metrics["errors"].append(
            "Generated runtime mock not found; "
            "method has not produced runtime_mock/matmul_exact_i8_i32.json"
        )
        return metrics

    with open(golden_mock) as f:
        golden = json.load(f)
    with open(generated_mock) as f:
        generated = json.load(f)

    metrics["match"] = golden == generated
    if not metrics["match"]:
        metrics["errors"].append("Generated runtime mock does not match golden")

    return metrics
