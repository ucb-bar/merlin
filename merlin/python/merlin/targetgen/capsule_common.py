"""Shared, target-agnostic capsule I/O for the capsule runners.

`capsule_runner` (gemmini: spike/verilator oracle) and `muon_capsule_runner` (cyclotron oracle) had
byte-identical copies of these helpers. They are the single source now; both runners import them (the
oracle-specific `run_capsule`/`run_suite` stay per-runner). Kept in `targetgen` (library), not in the
experiment harness, since the library runners are the consumers.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .contract import schemas


def _flat(nested) -> list:
    out: list = []
    if nested and isinstance(nested[0], list):
        for r in nested:
            out.extend(r)
    else:
        out.extend(nested)
    return out


def _cat(name: str):
    """Resolve a FailureCategory by name, tolerant to the enum's membership."""
    from aet.core.failures import FailureCategory
    try:
        return getattr(FailureCategory, name)
    except AttributeError:
        return FailureCategory.RUNNER_CRASH


def load_capsule(capsule_dir: str | Path, *, contract: str | Path | None = None) -> dict:
    """Load + validate a capsule.yaml; stamp its directory for interface-MLIR resolution."""
    d = Path(capsule_dir)
    cap = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
    schemas.validate(cap, "capsule", contract=contract)
    cap["__dir__"] = str(d)
    return cap


def discover_capsules(root: str | Path, *, labels: set[str] | None = None,
                      contract: str | Path | None = None) -> list[dict]:
    """Load every capsule under ``root`` (recursively), optionally filtered by label."""
    caps = []
    for cy in sorted(Path(root).rglob("capsule.yaml")):
        cap = load_capsule(cy.parent, contract=contract)
        if labels is None or cap.get("label") in labels:
            caps.append(cap)
    return caps
