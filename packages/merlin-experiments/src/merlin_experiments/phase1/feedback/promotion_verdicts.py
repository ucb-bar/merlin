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


def hydrated(ws, verdict):
    """``verdict`` with each compact index row reopened to its full public row.

    A broker's verdict lists compact rows whose complete record (``tier_boundary`` among it) lives in
    the workspace case store under a content digest (:mod:`..tools.result_details`). Reading the compact
    row dropped every L3 boundary observation, so no physical pass could join its L2 record: measured on
    fresh7ref, 227 passed gSIM jobs were retired as unattributable and re-run up to four times. A row
    whose detail cannot be reopened (or whose digest differs) stays as it was: nothing is invented.
    """
    from ..tools import result_details

    return result_details.hydrate(verdict, workspace=ws) if isinstance(verdict, dict) and ws is not None else verdict
