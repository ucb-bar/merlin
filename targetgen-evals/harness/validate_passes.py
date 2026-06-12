"""Check whether generated MLIR tests pass/fail as expected.

For now: structural check only — counts positive/negative test files present.
Future: invoke mlir-opt or xdsl-opt and parse output.
"""

from __future__ import annotations

from pathlib import Path


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    tests_dir = (
        Path(__file__).parent.parent / "datasets" / target / "tests"
    )

    metrics: dict = {
        "validator": "passes",
        "pass_tests_pass": 0,
        "pass_tests_total": 0,
        "negative_tests_correctly_rejected": 0,
        "negative_tests_total": 0,
        "errors": [],
    }

    if not tests_dir.exists():
        metrics["errors"].append(f"No test directory found at {tests_dir}")
        return metrics

    positive = list((tests_dir / "positive").glob("*.mlir")) if (tests_dir / "positive").exists() else []
    negative = list((tests_dir / "negative").glob("*.mlir")) if (tests_dir / "negative").exists() else []

    metrics["pass_tests_total"] = len(positive)
    metrics["negative_tests_total"] = len(negative)

    # Placeholder: tests cannot run without mlir-opt/xdsl-opt installed.
    # When the executor is wired up, replace these with real results.
    metrics["errors"].append(
        f"MLIR executor not wired up; {len(positive)} positive and "
        f"{len(negative)} negative test files found but not executed"
    )

    return metrics
