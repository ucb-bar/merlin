"""Orchestrate validation of a single run directory."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness import architecture_rules, collect_metrics
from harness import (
    validate_schema,
    validate_evidence,
    validate_xdsl,
    validate_passes,
    validate_dialect_design,
    validate_runtime_mock,
    validate_merlin_integration,
)


_VALIDATOR_MAP = {
    "schema": validate_schema,
    "naming": None,  # covered by R1 arch rule
    "evidence": validate_evidence,
    "xdsl": validate_xdsl,
    "passes": validate_passes,
    "design": validate_dialect_design,
    "runtime_mock": validate_runtime_mock,
    "merlin_integration": validate_merlin_integration,
}


def _load_manifest(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.exists():
        print(f"ERROR: run_manifest.yaml not found in {run_dir}", file=sys.stderr)
        return None
    with open(manifest_path) as f:
        return yaml.safe_load(f) or {}


def _load_budget(root: Path, budget_name: str) -> dict:
    budget_path = root / "configs" / "budgets" / f"{budget_name}.yaml"
    if not budget_path.exists():
        return {"validators": list(_VALIDATOR_MAP.keys())}
    with open(budget_path) as f:
        return yaml.safe_load(f) or {}


def validate_run(
    run_path: Path,
    root: Path,
    tracking_mode: str = "local",
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
    otel_endpoint: str | None = None,
) -> int:
    run_dir = run_path.resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    manifest = _load_manifest(run_dir)
    if manifest is None:
        return 1

    # Respect tracking_mode from manifest if not overridden by caller
    obs = manifest.get("observability", {})
    effective_tracking = tracking_mode
    if tracking_mode == "local" and obs.get("tracking_mode", "local") != "local":
        effective_tracking = obs["tracking_mode"]

    budget_name = manifest.get("budget", "cheap_smoke")
    budget = _load_budget(root, budget_name)
    enabled_validators = budget.get("validators", list(_VALIDATOR_MAP.keys()))
    if enabled_validators == ["all"]:
        enabled_validators = list(_VALIDATOR_MAP.keys())

    from harness.materialize_run import git_root
    repo_root = git_root(root)  # the merlin subject checkout (discovered, not root.parent)

    # Start logger
    from harness.tracking import TargetGenRunLogger
    logger = TargetGenRunLogger.start(
        target=manifest.get("target", "unknown"),
        method=manifest.get("method", "unknown"),
        seed=manifest.get("seed", 0),
        run_id=manifest["run_id"],
        run_path=run_dir,
        tracking_mode=effective_tracking,
        mlflow_tracking_uri=mlflow_tracking_uri or obs.get("mlflow", {}).get("tracking_uri"),
        experiment_name=experiment_name or obs.get("mlflow", {}).get("experiment_name"),
        otel_endpoint=otel_endpoint or obs.get("opentelemetry", {}).get("endpoint"),
    )
    logger.log_event("validation.started", {"budget": budget_name})
    logger.log_params({
        "budget": budget_name,
        "git_hash": manifest.get("git_hash_at_init"),
        "is_smoke_test": manifest.get("is_smoke_test", True),
    })

    start_time = time.monotonic()

    # Architecture rules (always run regardless of budget)
    with logger.start_span("validate.architecture_rules"):
        arch_rules = architecture_rules.check_all(run_dir, manifest, repo_root)
    arch_passed = sum(1 for r in arch_rules if r["passed"])
    arch_failed = sum(1 for r in arch_rules if not r["passed"])
    logger.log_metric("arch_rules_passed", arch_passed, source="harness")
    logger.log_metric("arch_rules_failed", arch_failed, source="harness")

    # Per-validator checks
    schemas_dir = root / "harness" / "schemas"
    validator_results: dict = {}

    for name in enabled_validators:
        mod = _VALIDATOR_MAP.get(name)
        if mod is None:
            continue
        with logger.start_span(f"validate.{name}"):
            try:
                if name == "schema":
                    result = mod.run(run_dir, manifest, schemas_dir)
                else:
                    result = mod.run(run_dir, manifest)
                validator_results[name] = result
            except Exception as e:
                result = {"validator": name, "errors": [f"EXCEPTION: {e}"]}
                validator_results[name] = result
        logger.log_event(f"validation.{name}.completed", {
            "errors": result.get("errors", []),
        })
        numeric = {k: v for k, v in result.items() if isinstance(v, (int, float, bool))}
        if numeric:
            logger.log_metrics(numeric, prefix=name)

    wall_clock = round(time.monotonic() - start_time, 3)
    logger.log_metric("wall_clock_seconds", wall_clock, source="harness")

    # Build summary and write metrics
    summary = collect_metrics.build_summary(manifest, validator_results, arch_rules)
    summary["wall_clock_seconds"] = wall_clock
    collect_metrics.write_metrics(run_dir, summary, validator_results, arch_rules)

    logger.log_metrics(summary, prefix=None)

    # Write validation_report.json
    all_errors = []
    for vr in validator_results.values():
        all_errors.extend(vr.get("errors", []))

    report = {
        "status": "validated",
        "validated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "is_smoke_test": manifest.get("is_smoke_test", True),
        "architecture_rules": {
            "passed": arch_passed,
            "failed": arch_failed,
            "results": arch_rules,
        },
        "validator_results": validator_results,
        "errors": all_errors,
        "overall": "pass" if (arch_failed == 0 and not all_errors) else "fail",
        "tracking": {
            "tracking_mode": effective_tracking,
            "mlflow_run_id": logger.mlflow_run_id,
            "otel_trace_id": logger.otel_trace_id,
            "wall_clock_seconds": wall_clock,
        },
    }
    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Update summary.md
    smoke_note = " *(smoke test)*" if manifest.get("is_smoke_test") else ""
    (run_dir / "summary.md").write_text(
        f"# Run: {manifest['run_id']}{smoke_note}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| target | `{manifest['target']}` |\n"
        f"| method | `{manifest['method']}` |\n"
        f"| seed | {manifest['seed']} |\n"
        f"| budget | `{manifest.get('budget')}` |\n"
        f"| git | `{manifest.get('git_hash_at_init', 'unknown')}` |\n\n"
        f"## Architecture Rules\n\n"
        f"Passed: {arch_passed} / {arch_passed + arch_failed}\n\n"
        + "\n".join(
            f"- {'✓' if r['passed'] else '✗'} **{r['rule_id']}** {r['name']}: {r['message']}"
            for r in arch_rules
        )
        + f"\n\n## Overall: `{report['overall']}`\n\n"
        f"{''.join(chr(10) + '- ' + e for e in all_errors) if all_errors else '*No errors.*'}\n"
    )

    # Log artifacts and finish
    logger.log_artifact(run_dir / "run_manifest.yaml")
    logger.log_artifact(run_dir / "validation_report.json")
    logger.log_artifacts(run_dir / "metrics")
    logger.log_generated_dir(run_dir / "generated" / f"{manifest['target']}-mlir", manifest["target"])

    overall = report["overall"]
    logger.log_event("validation.completed", {"overall": overall, "errors": len(all_errors)})
    logger.finish(overall)
    logger.patch_manifest(run_dir / "run_manifest.yaml")
    logger.close()

    # Print summary
    status_sym = "PASS" if overall == "pass" else "FAIL"
    print(f"[{status_sym}] {manifest['run_id']}")
    print(f"  arch rules:     {arch_passed} passed, {arch_failed} failed")
    print(f"  validators:     {len(validator_results)} ran, {len(all_errors)} errors")
    print(f"  wall clock:     {wall_clock}s")
    print(f"  tracking mode:  {effective_tracking}")
    print(f"  local logs:     {run_dir / 'logs'}")
    print(f"  report:         {run_dir / 'validation_report.json'}")

    return 0
