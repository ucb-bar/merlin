"""The single place that resolves capsule-corpus locations the LIBRARY reads (reference-by-location).

No library module should hardcode a corpus path; they call :func:`capsule_corpus_roots` /
:func:`find_capsule` instead. This is the ONE sanctioned indirection to a corpus that still lives
under ``experiments/`` (the boundary lint allowlists this module) — see the note below.

Corpora:
- ``merlin/contract/capsules`` — the frozen graded ABI suite (canonical).
- ``merlin/experiments/gemmini_perf_bench/kernels`` — the perf-bench corpus. It is *library-consumed*
  (the RTL checks screen it), so by the consumption-direction rule it is a benchmark input and its
  proper home is ``merlin/benchmarks/``. Relocating it is DEFERRED: three untracked, concurrently-edited
  perf-bench scripts still read it in place, and moving it would break that in-flight work. When they
  land, change the one line below (and repoint the perf harness) — every reader already goes through here.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir


def capsule_corpus_roots() -> list[Path]:
    """Existing capsule-corpus roots the library may screen (canonical first)."""
    roots = [
        merlin_dir() / "contract" / "capsules",
        merlin_dir() / "experiments" / "gemmini_perf_bench" / "kernels",  # perf benchmark (to relocate)
    ]
    return [r for r in roots if r.is_dir()]


def find_capsule(name: str) -> Path | None:
    """Locate a capsule directory by name across the corpus roots (first match wins)."""
    for root in capsule_corpus_roots():
        for cy in root.rglob("capsule.yaml"):
            if cy.parent.name == name:
                return cy.parent
    return None
