"""The ``model_check_report`` document: construction, agent view, validation and exit codes."""

from __future__ import annotations

import json
import subprocess
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
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() or None


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
    not partial) for ``pass``. A report with no gates is ``error``: nothing was judged.
    """
    infra = [f for f in report["failures"] if f.get("kind") in {"infrastructure", "usage"}]
    if infra:
        usage = any(f.get("kind") == "usage" for f in infra)
        report["status"] = "error"
        report["exit_code"] = EXIT_USAGE if usage else EXIT_INFRASTRUCTURE
        return report
    gates = report["gates"]
    if not gates:
        report["status"] = "error"
        report["exit_code"] = EXIT_INFRASTRUCTURE
        return report
    passed = all(g.get("status") == "pass" and not g.get("partial", False) for g in gates.values())
    report["status"] = "pass" if passed else "fail"
    report["exit_code"] = _EXIT_OF_STATUS[report["status"]]
    return report


def agent_view(report: Mapping[str, Any]) -> dict:
    """The report with every per-input value of the sealed split removed (aggregate gates kept)."""
    view = deepcopy(dict(report))
    view["agent_view"] = True
    for program in view.get("programs", []):
        program["runs"] = [r for r in program.get("runs", []) if r.get("split") != "sealed"]
        for kernel in program.get("kernels", []):
            kernel.pop("sealed", None)
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
