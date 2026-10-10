"""Must-refuse capsules: an operation the target cannot perform under the capsule's own placement.

Most capsules ask a compiler to compute something and grade the answer. Some ask the opposite: the
declared operation needs a stage, dtype or form the target's evidence says its hardware does not
perform (an integer-shift requantization on a readout that only applies a float scale, say), and the
capsule's placement policy forbids moving that work elsewhere. The right answer is then a STATED
refusal -- the backend's declared decline -- and any emitted program is wrong: it either computes
something the hardware cannot do or silently relocates work the capsule pinned to the accelerator.

A capsule says so with ``expected.outcome: refuse`` and an ``expected.refusal`` record naming what
is refused and why (derived evidence, never prose about a particular implementation). The grade then
inverts: a stated decline passes, a produced program fails with ``MUST_REFUSE_VIOLATED``, and a run
that measured nothing (an infrastructure fault, a harness crash) keeps its own status. Nothing here
names a target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The ``expected.outcome`` value of a must-refuse capsule. Absent means the ordinary "compute".
REFUSE = "refuse"
COMPUTE = "compute"
OUTCOMES = (COMPUTE, REFUSE)
#: Failure category of a must-refuse capsule whose backend produced a program anyway.
VIOLATED = "MUST_REFUSE_VIOLATED"
#: Statuses that measured nothing about the submission; a must-refuse verdict cannot be read from them.
_UNMEASURED = frozenset({"error", "infrastructure_fault", "not_gradeable_no_oracle", "budget_exhausted"})


def expects_refusal(capsule: Mapping[str, Any]) -> bool:
    """``capsule`` declares that the only correct answer is a stated refusal."""
    expected = capsule.get("expected") if isinstance(capsule, Mapping) else None
    return isinstance(expected, Mapping) and expected.get("outcome") == REFUSE


def declaration(*, stage: str | None = None, reason: str, evidence: str) -> dict[str, Any]:
    """The ``expected`` additions a must-refuse capsule carries."""
    if not reason or not evidence:
        raise ValueError("a must-refuse capsule must state what is refused and the evidence for it")
    refusal: dict[str, Any] = {"reason": reason, "evidence": evidence}
    if stage:
        refusal["stage"] = stage
    return {"outcome": REFUSE, "refusal": refusal}


def verdict(
    capsule: Mapping[str, Any], status: str, failure: Mapping[str, Any] | None, declined: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """``(status, failure, record)`` for a graded capsule; unchanged unless it expects a refusal."""
    if not expects_refusal(capsule):
        return status, dict(failure) if failure is not None else None, None
    expected = (capsule.get("expected") or {}).get("refusal") or {}
    record: dict[str, Any] = {"expected": dict(expected)}
    if status in _UNMEASURED:
        record["status"] = "unmeasured"
        return status, dict(failure) if failure is not None else None, record
    if status == "declined":
        record.update({"status": "met", "declined": dict(declined or {})})
        return "pass", None, record
    record["status"] = "violated"
    detail = (
        f"this capsule's operation must be refused ({expected.get('reason') or 'declared must-refuse'}); "
        f"the backend produced a program instead (graded {status!r})"
    )
    previous = dict(failure) if failure is not None else None
    return "fail", {"plane": "expected_refusal", "category": VIOLATED, "detail": detail, "graded": previous}, record
