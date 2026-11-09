"""When selected Phase 0 evidence may call itself ``verified`` (operator policy, 2026-10-01).

Evidence is ``verified`` only when every one of these holds; otherwise it is ``diagnostic``:

* the selection produced no evidence diagnostic at all;
* the RTL facts are fresh (their recorded source production re-validates, which the selection
  already checks) AND audited: an ``rtl-source-audit`` report beside the facts verifies them, bound to
  the exact facts and hardware-spec bytes selected here;
* every selected application capture was issued by the admitted sealed runner, with its attestation
  re-checked against the capture bytes on disk now.

A missing condition is recorded as a diagnostic with its reason, so the status is a function of the
diagnostics alone and nothing can be ``verified`` while a reason against it is on record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

AUDIT_SCHEMA = "merlin.rtl_source_audit.v1"
AUDIT_MEMBER = "validation.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rtl_audit_diagnostic(
    audit_raw: bytes | None, *, facts_raw: bytes | None, hardware_raw: bytes | None
) -> dict | None:
    """The reason the selected facts are not audited, as a diagnostic, or ``None`` when they are."""

    def refused(reason: str) -> dict:
        return {"component": "rtl-audit", "status": "unknown", "reason": reason}

    if audit_raw is None:
        return refused(f"no rtl-source-audit report ({AUDIT_MEMBER}) beside the selected facts")
    if facts_raw is None or hardware_raw is None:
        return refused("the audit cannot be bound without selected facts and hardware-spec bytes")
    try:
        report = json.loads(audit_raw)
    except (UnicodeDecodeError, ValueError):
        return refused("the rtl-source-audit report is unreadable")
    if not isinstance(report, Mapping) or report.get("schema") != AUDIT_SCHEMA:
        return refused("unsupported rtl-source-audit report schema")
    if report.get("facts_sha256") != _sha(facts_raw) or report.get("hardware_spec_sha256") != _sha(hardware_raw):
        return refused("the rtl-source-audit report audits different facts or hardware-spec bytes")
    checks = report.get("checks")
    if (
        report.get("status") != "verified"
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(row, Mapping)
            or row.get("source_status") != "verified"
            or (row.get("extraction") or {}).get("status") != "agrees"
            or row.get("gap") is not None
            for row in checks
        )
    ):
        return refused("the rtl-source-audit report does not verify every audited fact")
    return None


def capture_attestation_diagnostics(applications: Mapping[str, Any], attestations: Mapping[str, Any]) -> list[dict]:
    """One diagnostic per selected application capture without an admitted, re-verified execution."""
    from .coverage_commitment import attestation_failure

    out = []
    for label, application in sorted((applications or {}).items()):
        reason = attestation_failure(
            {
                "capture_sha256": (application or {}).get("capture_sha256"),
                "capture_receipt": (application or {}).get("capture_receipt") or {},
                "capture_execution_attestation": (attestations or {}).get(label),
            }
        )
        if reason is not None:
            out.append({"component": "capture-execution", "status": "unknown", "application": label, "reason": reason})
    return out


def status(diagnostics: list) -> str:
    return "diagnostic" if diagnostics else "verified"
