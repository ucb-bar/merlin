"""Interpret recorded numerical verdicts and already-executed certificate tiers."""

from __future__ import annotations

from merlin.targetgen.oracle_schedule import FAIL as _FAIL
from merlin.targetgen.oracle_schedule import PASS as _PASS


def row_status(row) -> str | None:
    """Return pass/fail for an explicit verdict, otherwise None.

    A boolean pass field takes precedence over a pass/fail status string.
    Refusals, missing verdicts and unrecognized statuses remain unknown; they
    must not become fabricated numerical failures in the promotion ledger."""
    if isinstance(row, dict):
        flag = row.get("pass")
        if isinstance(flag, bool):
            return _PASS if flag else _FAIL
        status = row.get("status")
        if isinstance(status, str) and status.strip() in (_PASS, _FAIL):
            return status.strip()
    return None


def cert_already_executed(verdict, cert_tier) -> bool:
    """Whether the verdict's own oracle selection already ran ``cert_tier`` for its capsules.

    A self-check on the grade's own selection runs the cert tier for every capsule its loop tier
    passed, in the same request. Enqueueing a promotion for those bytes buys the same simulation a
    second time. Read only from the verdict's declared ``oracle_tiers``: a verdict that declares none
    keeps promoting as before.
    """
    tiers = verdict.get("oracle_tiers") if isinstance(verdict, dict) else None
    return isinstance(cert_tier, str) and isinstance(tiers, list) and cert_tier in tiers
