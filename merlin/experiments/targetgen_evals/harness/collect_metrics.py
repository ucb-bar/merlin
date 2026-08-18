"""Aggregate per-validator metrics into summary_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_NA = None  # sentinel for metrics.csv NA column

_ALL_COLUMNS = [
    # core identity
    "run_id", "method", "seed", "is_smoke_test", "budget",
    # validation quality
    "schema_valid", "xdsl_files", "xdsl_op_estimate",
    "pass_tests_pass", "pass_tests_total",
    "evidence_coverage", "unsupported_claim_rate",
    "arch_rules_passed", "arch_rules_failed",
    # paper ablation columns (legacy names kept for compatibility)
    "human_interventions", "cost_usd", "time_to_first_pass_s",
    # generalization matrix (G0-G5): held-out success per axis, not shape alone
    "heldout_shape_success", "heldout_layout_success", "heldout_dtype_success",
    "heldout_surface_success", "heldout_composition_success", "heldout_fusion_success",
    "heldout_model_success", "heldout_kernel_success",
    "merlin_core_files_modified",
    # tracking metadata
    "tracking_mode", "mlflow_run_id", "otel_trace_id",
    # effort columns
    "observed_cost_usd", "estimated_cost_usd", "cost_source",
    "tokens_input", "tokens_output", "token_source",
    "wall_clock_seconds", "time_to_first_validation_s",
    "agent_turns", "tool_calls",
]

# Columns that are not numeric — excluded from mean±std in compare_runs
_STRING_COLUMNS = frozenset({
    "run_id", "method", "budget", "tracking_mode",
    "mlflow_run_id", "otel_trace_id", "cost_source", "token_source",
})


def build_summary(manifest: dict, validator_results: dict, arch_rules: list[dict]) -> dict:
    arch_passed = sum(1 for r in arch_rules if r["passed"])
    arch_failed = sum(1 for r in arch_rules if not r["passed"])

    schema = validator_results.get("schema", {})
    xdsl = validator_results.get("xdsl", {})
    evidence = validator_results.get("evidence", {})
    passes = validator_results.get("passes", {})
    merlin = validator_results.get("merlin_integration", {})

    obs = manifest.get("observability", {})
    mlf = obs.get("mlflow", {})
    otel = obs.get("opentelemetry", {})

    return {
        "run_id": manifest["run_id"],
        "method": manifest["method"],
        "seed": manifest["seed"],
        "is_smoke_test": manifest.get("is_smoke_test", True),
        "budget": manifest.get("budget", "unknown"),
        "schema_valid": schema.get("schema_valid", _NA),
        "xdsl_files": xdsl.get("xdsl_files", _NA),
        "xdsl_op_estimate": xdsl.get("xdsl_op_estimate", _NA),
        "pass_tests_pass": passes.get("pass_tests_pass", _NA),
        "pass_tests_total": passes.get("pass_tests_total", _NA),
        "evidence_coverage": evidence.get("evidence_coverage", _NA),
        "unsupported_claim_rate": evidence.get("unsupported_claim_rate", _NA),
        "arch_rules_passed": arch_passed,
        "arch_rules_failed": arch_failed,
        "human_interventions": _NA,
        "cost_usd": _NA,
        "time_to_first_pass_s": _NA,
        "heldout_shape_success": _NA,
        "heldout_layout_success": _NA,
        "heldout_dtype_success": _NA,
        "heldout_surface_success": _NA,
        "heldout_composition_success": _NA,
        "heldout_fusion_success": _NA,
        "heldout_model_success": _NA,
        "heldout_kernel_success": _NA,
        "merlin_core_files_modified": merlin.get("merlin_core_files_modified", _NA),
        # tracking
        "tracking_mode": obs.get("tracking_mode", "local"),
        "mlflow_run_id": mlf.get("run_id"),
        "otel_trace_id": otel.get("trace_id"),
        # effort (filled by validate_run via metrics injection)
        "observed_cost_usd": _NA,
        "estimated_cost_usd": _NA,
        "cost_source": _NA,
        "tokens_input": _NA,
        "tokens_output": _NA,
        "token_source": _NA,
        "wall_clock_seconds": _NA,
        "time_to_first_validation_s": _NA,
        "agent_turns": _NA,
        "tool_calls": _NA,
    }


def write_metrics(run_dir: Path, summary: dict, validator_results: dict, arch_rules: list[dict]) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    def _write(name: str, data: Any) -> None:
        (metrics_dir / name).write_text(json.dumps(data, indent=2) + "\n")

    _write("schema_metrics.json", validator_results.get("schema", {}))
    _write("evidence_metrics.json", validator_results.get("evidence", {}))
    _write("xdsl_metrics.json", validator_results.get("xdsl", {}))
    _write("pass_metrics.json", validator_results.get("passes", {}))
    _write("design_metrics.json", validator_results.get("design", {}))
    _write("effort_metrics.json", {
        "human_interventions": None,
        "cost_usd": None,
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "tokens_input": summary.get("tokens_input"),
        "tokens_output": summary.get("tokens_output"),
        "agent_turns": summary.get("agent_turns"),
        "tool_calls": summary.get("tool_calls"),
    })
    _write("summary_metrics.json", summary)
