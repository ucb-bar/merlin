"""Resolve curated DSE analysis inputs independently of writable experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import repo_root, tracked_out_dir


def case_study_dir(explicit: str | Path | None = None) -> Path:
    """The committed analysis snapshot, with explicit input and historical-checkout support.

    New analyses belong below ``artifacts_dir()``; redirecting generated output must never silently
    replace this reference. The reference is checkout-owned, not bundled into the core wheel.
    Installed consumers can pass a directory or set ``MERLIN_DSE_REFERENCE_DIR``. A missing explicit
    directory is returned unchanged so the consumer reports that mistake rather than changing data.
    """
    if explicit is not None:
        return Path(explicit)
    override = os.environ.get("MERLIN_DSE_REFERENCE_DIR")
    if override:
        return Path(override)
    canonical = repo_root() / "experiments/reference-data/dse/case_study"
    if canonical.exists():
        return canonical
    historical = tracked_out_dir() / "artifacts" / "dse-guidance" / "case_study"
    return historical if historical.exists() else canonical
