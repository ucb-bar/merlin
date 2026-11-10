"""Resolve curated DSE analysis inputs independently of writable experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path


def case_study_dir(explicit: str | Path | None = None) -> Path:
    """Select analysis input explicitly, independently of writable output.

    A missing selected directory is returned unchanged so the consumer reports
    that mistake rather than choosing another run's data.
    """
    if explicit is not None:
        return Path(explicit)
    override = os.environ.get("MERLIN_DSE_REFERENCE_DIR")
    if override:
        return Path(override)
    raise ValueError("select a case-study input with --case-study-dir or MERLIN_DSE_REFERENCE_DIR")
