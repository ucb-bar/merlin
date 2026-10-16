"""Conformance summaries used by the Phase 1 authoring continuation."""

from __future__ import annotations

from pathlib import Path


def conformance_ever(rounds) -> dict:
    """Fold the run-level gate: true if ever satisfied, None if always inapplicable.

    Resume rebuilds this same view from persisted round records.
    """
    ever: dict = {}
    for entry in rounds or []:
        checks = ((entry or {}).get("conformance") or {}).get("checks") or {}
        for key, value in checks.items():
            if value is None:
                ever.setdefault(key, None)
            else:
                ever[key] = bool(ever.get(key)) or bool(value)
    return ever


def workflow_conformance(
    tpath: Path, submission_dir: Path, arm: str, endpoint_kind: str, resolved_tools
) -> tuple[dict, bool]:
    """Recompute current workflow evidence from resolved tools; checker errors fail closed."""
    try:
        from merlin_experiments.phase1 import conformance as _CONF

        conf = _CONF.compute(tpath, submission_dir, arm, endpoint_kind, resolved_tools=resolved_tools)
    except Exception as exc:  # noqa: BLE001 — conformance is a formal gate; unavailable means false
        conf = {"conformant": False, "error": f"{type(exc).__name__}: {exc}"}
    return conf, conf.get("conformant") is True
