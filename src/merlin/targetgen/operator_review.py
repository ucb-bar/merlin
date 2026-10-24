"""Separate authored declarations from review and explicit compatibility acceptance."""

from __future__ import annotations

import os

PENDING = "authored_pending_operator_review"
STATUSES = frozenset({"reviewed", "unreviewed", PENDING})
ACCEPT_ENV = "MERLIN_ACCEPT_AUTHORED_PENDING_OPERATOR_REVIEW"


def review_status(document, *, default="reviewed"):
    status = document.get("status", default)
    if status != "reviewed":
        return status
    evidence = document.get("evidence", {})
    marker = evidence.get("review_status", "") if isinstance(evidence, dict) else ""
    contract = document.get("numerical_contract")
    # A stale reviewed label cannot override the author's own pending evidence.
    if (isinstance(marker, str) and "pending operator review" in marker.lower()) or (
        isinstance(contract, dict) and review_status(contract) == PENDING
    ):
        return PENDING
    return status


def acceptance():
    """Default 1 retains the historical admission behavior; 0 refuses pending claims."""
    selected = os.environ.get(ACCEPT_ENV, "1")
    if selected not in {"0", "1"}:
        raise ValueError(f"{ACCEPT_ENV} must be 0 or 1")
    return dict(
        switch=ACCEPT_ENV,
        enabled=selected == "1",
        selection="environment" if ACCEPT_ENV in os.environ else "default_compatibility",
    )


def review_record(*documents, default="reviewed"):
    states = [review_status(document, default=default) for document in documents]
    state = (
        "unreviewed"
        if any(s not in {"reviewed", PENDING} for s in states)
        else (PENDING if PENDING in states else "reviewed")
    )
    policy = acceptance() if state == PENDING else None
    return dict(
        review_status=state,
        operator_accepted=state == "reviewed" or (state == PENDING and policy["enabled"]),
        **({"operator_acceptance": policy} if policy is not None else {}),
    )


def review_accepted(*documents, default="reviewed"):
    return review_record(*documents, default=default)["operator_accepted"]
