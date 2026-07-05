"""External-baseline K1-RVV comparison harness.

Runs the SAME models we support through independent external compilers/runtimes (TVM, ExecuTorch,
Buddy, EXO, ggml — pinned under ``third_party/baselines/``) end-to-end on the SAME SpacemiT K1 board
with RVV, and profiles them at two levels (whole-model E2E + per-region "kernel-style"), while being
mechanically honest about any scalar fallback.

Shared building blocks (per-framework runners live in ``baselines/<framework>.py``):
  * :mod:`.bundle`     — resolve ``(model, variant)`` -> capture bundle + correctness tolerance
  * :mod:`.k1_exec`    — board lock + fail-closed push/run on the K1
  * :mod:`.rvv_audit`  — objdump-based RVV-vs-scalar coverage + march enforcement
  * :mod:`.profile`    — parse MERLIN_E2E / MERLIN_REGION markers -> whole-model + region profiles
  * :mod:`.contract`   — the ``BaselineResult`` schema (not_run_is_not_pass; scalar fallback labeled)
  * :mod:`.aggregate`  — collect results -> framework × model matrix (markdown/CSV)
"""
from __future__ import annotations

from merlin.baselines.contract import (
    FRAMEWORKS,
    REGIONS,
    BaselineResult,
    RegionProfile,
    ScalarFallback,
)

__all__ = [
    "FRAMEWORKS",
    "REGIONS",
    "BaselineResult",
    "RegionProfile",
    "ScalarFallback",
]
