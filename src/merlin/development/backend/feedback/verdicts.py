"""Interpret recorded verdicts and count their execution/evidence obligations.

Unknown or refused work stays distinct from numerical failure. Public detail
hydration supplies recorded rows only; it does not issue qualification.
"""

from __future__ import annotations

from collections import Counter

from merlin.development.targetgen.oracles.schedule import FAIL as _FAIL
from merlin.development.targetgen.oracles.schedule import PASS as _PASS


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
    from merlin.development.backend.tools import result_details

    return result_details.hydrate(verdict, workspace=ws) if isinstance(verdict, dict) and ws is not None else verdict


def candidate_evidence_complete(summary: dict) -> bool:
    """Require the closed candidate's source, placement, dispatch and tier evidence."""
    tiers = summary.get("tiers")
    required = summary.get("required_tiers")
    source = summary.get("source_coverage")
    components = summary.get("components")
    return (
        summary.get("status") == "pass"
        and isinstance(tiers, dict)
        and isinstance(required, list)
        and bool(required)
        and all(isinstance(tier, str) and tiers.get(tier) == "pass" for tier in required)
        and isinstance(source, dict)
        and source.get("status") == "verified"
        and isinstance(components, dict)
        and all(
            components.get(key) == expected
            for key, expected in (
                ("emitted_host_compute", "clean"),
                ("source_placement", "clean"),
                ("completed_dispatch", "verified"),
                ("candidate_required_tiers", "pass"),
            )
        )
    )


def summarize(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        passed = row.get("status") == "pass"
        refusal = (row.get("expected_refusal") or {}).get("status") == "met" and passed
        counts["result_passed"] += int(passed)
        counts["execution_passed"] += int(passed and not refusal)
        native = row.get("candidate_native_verification")
        if isinstance(native, dict):
            complete = passed and candidate_evidence_complete(native)
            counts["model_programs_verified"] += int(complete)
            counts["model_programs_unverified"] += int(not complete)
        elif refusal:
            counts["expected_refusals_met"] += 1
        else:
            requirement = row.get("trace_requirement_status")
            complete = row.get("trace_status") == "pass" and requirement in (None, "pass")
            counts["execution_passes_with_trace_evidence"] += int(passed and complete)
            counts["execution_passes_without_trace_evidence"] += int(passed and not complete)
            if isinstance(requirement, str) and requirement.startswith("unavailable"):
                counts["trace_evidence_unavailable"] += 1
            elif row.get("trace_status") == "fail":
                counts["trace_evidence_failed"] += 1
    fields = (
        "result_passed",
        "execution_passed",
        "execution_passes_with_trace_evidence",
        "execution_passes_without_trace_evidence",
        "trace_evidence_unavailable",
        "trace_evidence_failed",
        "model_programs_verified",
        "model_programs_unverified",
        "expected_refusals_met",
    )
    return {
        "schema": "merlin.phase1.qualification_counts.v1",
        "n_completed": len(rows),
        **{name: counts[name] for name in fields},
        "scope": "execution and structural evidence counts; final source/cohort/tier gates remain separate",
    }
