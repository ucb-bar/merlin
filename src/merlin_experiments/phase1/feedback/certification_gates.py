"""Say which certification gate holds a test case back, and what evidence would open it.

Two kinds of test case passed Spike L2 for a whole run without reaching gSIM L3, and the agent's
verdict never said why:

* A whole-program test case is scheduled only after a declared fraction of operator test cases
  passes. The scheduler recorded "op pass fraction X < gate Y" in the grade, but the verdict row the
  agent reads is projected from per-test-case result files, which a deferred test case never writes,
  so the row read ``gated`` with every reason field null.
* A calibration witness reaches gSIM only through an explicit ``--sim gsim`` request: a few initial
  diagnostics, then operator-authorized batches for a stated question. The verdict listed each witness
  as ``pending_matching_spike_pass`` from the pre-grade selection, even when the same grade passed it.

This module only explains. It changes no status, count, gate threshold or admission rule, and it uses
only what the agent may already see: its own pass/fail results, the public gate fraction declared in
the test case, and its own request budget. Reference outputs never appear here.
"""

from __future__ import annotations

import json
from typing import Any

_DEFERRAL_PREFIX = "whole-model capsule deferred:"
_ROUTE = (
    "Background grades count only operator test cases passing in the same grade. A self-check that "
    "names this test case (`python agent_selfcheck.py --capsules NAME`) instead counts operator test "
    "cases holding L2 and L3 certificates for your current compiler bytes, so let operator "
    "certificates land on an unchanged compiler before requesting it."
)


def _fraction_pair(detail: str) -> tuple[float | None, float | None]:
    """``(observed, required)`` from the scheduler's own deferral sentence, or ``None`` parts."""
    if not detail.startswith(_DEFERRAL_PREFIX):
        return None, None
    _, _, tail = detail.partition("fraction ")
    observed_text, _, required_text = tail.partition(" < gate ")

    def number(text: str) -> float | None:
        try:
            value = float(text.strip())
        except ValueError:
            return None
        return value if 0.0 <= value <= 1.0 else None

    return number(observed_text), number(required_text)


def gated_row_fields(summary_row: dict, redact) -> dict[str, Any]:
    """Reason fields for a deferred row whose per-test-case result file does not exist.

    ``summary_row`` is the grade's own summary entry (``gate_reason`` from the operator gate, or
    ``background_deferred`` when the grade stopped admitting); ``redact`` is the verdict's ordinary
    detail scrubber. Returns ``{}`` for a row that is not deferred or carries no reason.
    """
    if summary_row.get("status") != "gated":
        return {}
    gate_reason = summary_row.get("gate_reason")
    deferral = summary_row.get("background_deferred")
    reason = gate_reason if isinstance(gate_reason, str) and gate_reason else deferral
    if not isinstance(reason, str) or not reason:
        return {}
    fields: dict[str, Any] = {
        "failure_plane": "gate",
        "failure_category": "GATED" if reason.startswith(_DEFERRAL_PREFIX) else "ADMISSION_DEFERRED",
        "failure_detail": redact(reason),
    }
    if summary_row.get("kind") != "model":
        return fields
    observed, required = _fraction_pair(reason)
    gate: dict[str, Any] = {
        "gate": "after_op_pass_fraction",
        "scope": "operator test cases of the selected correctness roster",
        "how_to_run_it": _ROUTE,
        "advisory_only": True,
    }
    if observed is not None and required is not None:
        gate.update(observed_operator_pass_fraction=round(observed, 2), required_operator_pass_fraction=required)
    else:
        gate["deferred_because"] = (
            "this grade stopped admitting test cases before the operator gate opened; a whole-program "
            "test case starts only after enough operator test cases pass within the same grade")
    fields["certification_gate"] = gate
    return fields


def _l2_status(row: dict) -> str:
    tier = (row.get("tiers") or {}).get("L2")
    status = tier.get("status") if isinstance(tier, dict) else tier
    if status in {"pass", "fail"}:
        return status
    return "not_run_in_this_grade"


def _initial_diagnostics() -> dict[str, Any] | None:
    """How many question-free witness gSIM diagnostics this campaign still allows, when known."""
    try:
        from .. import joint_control
        from ..joint_measurements import limits

        selected = joint_control.selected()
        if selected is None:
            return None
        allowed = limits(selected["policy"])["initial_diagnostics_max"]
        path = joint_control.gsim_ledger()
        used = len(json.loads(path.read_text() or "{}").get("admissions", [])) if path.is_file() else 0
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {"allowed": allowed, "used": min(used, allowed), "remaining": max(0, allowed - used)}


def calibration_access(per_capsule, witnesses: list[dict]) -> dict[str, Any]:
    """This grade's Spike result per selected witness, and how a witness reaches gSIM."""
    rows = {row.get("capsule"): row for row in per_capsule or () if isinstance(row, dict)}
    this_grade = {w["capsule"]: _l2_status(rows.get(w["capsule"], {})) for w in witnesses if "capsule" in w}
    access: dict[str, Any] = {
        "spike_l2_this_grade": this_grade,
        "spike_l2_passes_this_grade": sum(status == "pass" for status in this_grade.values()),
        "request": "python agent_selfcheck.py --sim gsim --capsules NAME  (one witness per request)",
        "rule": ("A witness reaches gSIM only by request: the first few diagnostics need no question; after "
                 "them, each needs an operator-authorized batch for a stated unresolved question. Run one "
                 "when its cycles would change a schedule decision; matching earlier evidence is reused."),
        "advisory_only": True,
    }
    initial = _initial_diagnostics()
    if initial is not None:
        access["initial_diagnostics"] = initial
    return access
