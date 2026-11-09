"""Strict final comparison arithmetic, independent of convergence accounting.

This is the new campaign policy. The historical v1 helper keeps its original
five-percent/20x semantics. Neither helper authenticates execution evidence:
only the protected final lifecycle may construct its comparison inputs.
"""

from __future__ import annotations

import math

from merlin.common.digest import is_sha256

from .component_experiment import FinalMemberComparison
from .contracts import StageGateError

SCHEMA = "merlin.component_final_gate.v2"
POLICY = "per_member_match_or_beat_with_separate_convergence_accounting"


def _wall(value: float | None, label: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
    ):
        raise StageGateError(label + " must be positive finite seconds or UNKNOWN")


def strict_final_component_campaign_gate(
    comparisons: tuple[FinalMemberComparison, ...],
    *,
    expected_members: tuple[str, ...],
    phase12_wall_s: float | None = None,
    handwritten_wall_s: float | None = None,
) -> dict:
    """Require parity for every member; unavailable historical time is separate.

    The caller is the private evaluator and must independently authenticate
    builds, hardware, input/output scope and original accuracy. Caller-supplied
    booleans or hashes alone do not establish those claims. Integer comparison
    avoids floating-point errors at a one-cycle boundary.
    """
    if (
        type(expected_members) is not tuple
        or not expected_members
        or any(not isinstance(name, str) or not name for name in expected_members)
        or len(set(expected_members)) != len(expected_members)
    ):
        raise StageGateError("strict final gate requires unique nonempty member identities")
    if type(comparisons) is not tuple or any(type(row) is not FinalMemberComparison for row in comparisons):
        raise StageGateError("strict final gate requires typed immutable comparison inputs")
    if len(comparisons) != len(expected_members) or {row.member for row in comparisons} != set(expected_members):
        raise StageGateError("strict final comparison membership is incomplete or duplicated")
    failures, unknowns, results = [], [], []
    by_member = {row.member: row for row in comparisons}
    for member in expected_members:
        row = by_member[member]
        if not all(
            is_sha256(value)
            for value in (
                row.reference_execution_sha256,
                row.candidate_execution_sha256,
                row.comparison_identity_sha256,
            )
        ):
            raise StageGateError("strict final comparison lacks exact evidence identities")
        if any(
            type(value) is not bool
            for value in (row.accuracy_passed, row.final_executable_passed, row.hardware_verified)
        ):
            raise StageGateError("strict final evidence verdicts must be explicit booleans")
        for value in (row.reference_cycles, row.candidate_cycles):
            if value is not None and (type(value) is not int or value <= 0):
                raise StageGateError("strict final hardware cycles must be positive integers or UNKNOWN")
        correctness = row.accuracy_passed and row.final_executable_passed
        hardware = row.hardware_verified and row.reference_cycles is not None and row.candidate_cycles is not None
        if not correctness:
            failures.append(member + ": correctness or final executable gate failed")
        if not hardware:
            unknowns.append(member + ": matched hardware cycles unavailable")
        worse = hardware and row.candidate_cycles > row.reference_cycles
        if worse:
            failures.append(member + ": exceeds frozen handwritten reference cycles")
        results.append(
            {
                "member": member,
                "status": "fail" if not correctness or worse else "pass" if hardware else "unknown",
                "reference_cycles": row.reference_cycles,
                "candidate_cycles": row.candidate_cycles,
                "delta_cycles": row.candidate_cycles - row.reference_cycles if hardware else None,
                "reference_execution_sha256": row.reference_execution_sha256,
                "candidate_execution_sha256": row.candidate_execution_sha256,
                "comparison_identity_sha256": row.comparison_identity_sha256,
            }
        )
    _wall(phase12_wall_s, "Phase1/2 elapsed time")
    _wall(handwritten_wall_s, "handwritten elapsed time")
    speedup = None
    if phase12_wall_s is not None and handwritten_wall_s is not None:
        value = handwritten_wall_s / phase12_wall_s
        if math.isfinite(value):
            speedup = value
    return {
        "schema": SCHEMA,
        "policy": POLICY,
        "status": "fail" if failures else "unknown" if unknowns else "pass",
        "failures": failures,
        "unknowns": unknowns,
        "parity_allowance": 0,
        "members": results,
        "convergence": {
            "status": "reported" if speedup is not None else "unknown",
            "phase12_wall_s": phase12_wall_s,
            "handwritten_wall_s": handwritten_wall_s,
            "speedup": speedup,
            "gates_performance": False,
            "scope": "arithmetic on separately authenticated comparable elapsed times",
        },
        "scope": "final frozen held-out comparison arithmetic; execution authority is separate",
    }
