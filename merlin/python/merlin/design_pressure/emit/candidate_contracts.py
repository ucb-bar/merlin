"""Emit a candidate_contracts payload (the I0–I3 ladder for a workload)."""
from __future__ import annotations


def emit_candidate_contracts(workload: str, contracts: list[dict]) -> dict:
    """Build the candidate_contracts artifact from ``legal_contracts`` output."""
    return {
        "workload": workload,
        "contracts": [dict(c) for c in contracts],
        "legal_contracts": [c["name"] for c in contracts if c.get("legal")],
    }
