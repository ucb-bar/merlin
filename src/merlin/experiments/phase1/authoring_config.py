"""Invocation selections passed from admission into compiler authoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuthoringRuntime:
    """Native edge selections not already recorded by PreparedRun.

    public_root captures the legacy override at invocation time. Its descriptor
    fallback is still materialized lazily, only when grading needs public capsules.
    """

    bundle_id: str
    oracle_timing: Path
    public_root: Path | None = None


_PASS_LINE = "Baseline pilot passes all required public/dev pilot capsules and is ready for hidden pilot grading."
_STATUS_LINES = (
    _PASS_LINE,
    "Baseline pilot does not yet pass all required public/dev pilot capsules; remaining failures "
    "are listed by capsule and failure plane.",
    "Baseline pilot is not comparable because it violates the compiler/runtime/integrity boundary.",
)
