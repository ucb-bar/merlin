"""The ``model_check_report`` document: construction, agent view, validation and exit codes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.common.schemas import validate_or_raise
from merlin.model_check import CONTRACT_VERSION, REPORT_SCHEMA

SCHEMA_NAME = "model_check_report"

EXIT_PASS = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2
EXIT_INFRASTRUCTURE = 3

_EXIT_OF_STATUS = {"pass": EXIT_PASS, "fail": EXIT_GATE_FAILED}


def _tool_commit() -> str | None:
    """The commit this tool's code was taken from: an export's own source record, never the HEAD of
    a checkout that happens to enclose it (a frozen run export inside a live worktree reported the
    worktree's moving HEAD)."""
    from merlin.common.provenance import UNKNOWN
    from merlin.targetgen.provenance import source_revision

    try:
        revision = source_revision(repo_root())
    except OSError:
        return None
    return None if revision == UNKNOWN else revision


def new_report(command: str, *, model: str | None, split: str | None, submission: Mapping[str, Any]) -> dict:
    """An empty report for ``command``; every required field is present from the start."""
    return {
        "schema": REPORT_SCHEMA,
        "command": command,
        "status": "error",
        "exit_code": EXIT_INFRASTRUCTURE,
        "tool": {"name": "merlin.model_check", "contract": CONTRACT_VERSION, "commit": _tool_commit()},
        "submission": dict(submission),
        "model": model,
        "split": split,
        "agent_view": False,
        "bundle": None,
        "binding": None,
        "gates": {},
        "programs": [],
        "model_metrics": None,
        "failures": [],
        "timing": {},
    }


def settle(report: dict) -> dict:
    """Derive ``status`` and ``exit_code`` from the gates and failures already recorded.

    Any infrastructure failure makes the report ``error``; otherwise every gate must be ``pass`` (and
    no gate partial) for ``pass``. A ``pending`` gate -- one the models file defers (full mode's differential
    under ``full.differential: cached``) -- is recorded and neither passes nor fails. A report with no
    gates, or with only pending ones, is ``error``: nothing was judged.
    """
    infra = [f for f in report["failures"] if f.get("kind") in {"infrastructure", "usage"}]
    if infra:
        usage = any(f.get("kind") == "usage" for f in infra)
        report["status"] = "error"
        report["exit_code"] = EXIT_USAGE if usage else EXIT_INFRASTRUCTURE
        return report
    gates = report["gates"].values()
    judged = [g for g in gates if g.get("status") != "pending"]
    if not judged:
        report["status"] = "error"
        report["exit_code"] = EXIT_INFRASTRUCTURE
        return report
    passed = all(g.get("status") == "pass" for g in judged) and not any(g.get("partial", False) for g in gates)
    report["status"] = "pass" if passed else "fail"
    report["exit_code"] = _EXIT_OF_STATUS[report["status"]]
    return report


def _redact_bundle_paths(value: Any, root: str) -> Any:
    """Remove this report's known host-only bundle location from visible metadata."""
    if isinstance(value, str):
        return "<numerics_bundle>" if value == root else value.replace(root + "/", "<numerics_bundle>/")
    if isinstance(value, dict):
        return {key: _redact_bundle_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_bundle_paths(item, root) for item in value]
    return value


def agent_view(report: Mapping[str, Any]) -> dict:
    """The report with every per-input value of the sealed split removed (aggregate gates kept)."""
    view = deepcopy(dict(report))
    view["agent_view"] = True
    bundle = view.get("bundle")
    if isinstance(bundle, dict):
        root = bundle.pop("path", None)
        if isinstance(root, str) and root not in {"", "/"} and Path(root).is_absolute():
            view = _redact_bundle_paths(view, root.rstrip("/"))
    for program in view.get("programs", []):
        program["runs"] = [r for r in program.get("runs", []) if r.get("split") != "sealed"]
        for kernel in program.get("kernels", []):
            kernel.pop("sealed", None)
        if program.get("split") == "sealed":
            # Sealed shape variants: per-status counts only (no symbol, label or refusal text).
            statuses: dict[str, int] = {}
            for kernel in program.pop("kernels", []):
                statuses[kernel.get("status", "unknown")] = statuses.get(kernel.get("status", "unknown"), 0) + 1
            program.pop("refused", None)
            program.pop("images", None)
            program["kernels"] = []
            program["summary"] = statuses
    metrics = view.get("model_metrics")
    if isinstance(metrics, dict):
        metrics["per_input"] = [m for m in metrics.get("per_input", []) if m.get("split") != "sealed"]
        metrics.pop("sealed", None)
    view["failures"] = [
        {k: v for k, v in f.items() if k not in {"input", "detail"}} if f.get("split") == "sealed" else f
        for f in view.get("failures", [])
    ]
    return view


def write(report: Mapping[str, Any], path: Path) -> None:
    """Validate the top-level fields and write the report as JSON."""
    validate_or_raise(dict(report), SCHEMA_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str) + "\n")
