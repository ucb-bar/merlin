"""Check dialect design quality against golden expected_dialect_features.yaml."""

from __future__ import annotations

from pathlib import Path


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    golden_path = (
        Path(__file__).parent.parent / "datasets" / target / "golden"
        / "expected_dialect_features.yaml"
    )
    dialect_plan_path = run_dir / "contracts" / "dialect_plan.yaml"

    metrics: dict = {
        "validator": "design",
        "ops_expected": 0,
        "ops_generated": 0,
        "ops_coverage": None,
        "missing_ops": [],
        "extra_ops": [],
        "errors": [],
    }

    if not golden_path.exists():
        metrics["errors"].append(f"Golden dialect features not found: {golden_path}")
        return metrics

    import yaml
    with open(golden_path) as f:
        golden = yaml.safe_load(f) or {}

    high_level_ops = set(golden.get("high_level_ops", []))
    optional_ops = set(golden.get("optional_low_level_ops", []))
    expected_ops = high_level_ops | optional_ops
    metrics["ops_expected"] = len(expected_ops)

    if not dialect_plan_path.exists():
        metrics["errors"].append(
            "contracts/dialect_plan.yaml not found; "
            "cannot compare generated ops to golden expectations for an empty run"
        )
        return metrics

    with open(dialect_plan_path) as f:
        plan = yaml.safe_load(f) or {}

    generated_ops = {op.get("name", "?") for op in plan.get("ops", [])}
    metrics["ops_generated"] = len(generated_ops)

    if expected_ops:
        metrics["ops_coverage"] = round(
            len(generated_ops & expected_ops) / len(expected_ops), 3
        )

    metrics["missing_ops"] = sorted(expected_ops - generated_ops)
    metrics["extra_ops"] = sorted(generated_ops - expected_ops)

    return metrics
