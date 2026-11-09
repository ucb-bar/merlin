"""Shared helpers for the gemmini_perf_bench cross-approach performance benchmark.

This benchmark drives the SAME kernels (int8 matmul/conv shapes) through multiple Gemmini
code-generation approaches (golden bareMetalC, generated MLIR OOT backends, the hand-written C++
Gemmini dialect via IREE) and compares cycles / wall-time / utilization / correctness. It reuses the
capsule_bench_v0 libraries (capsule emit, deterministic golden, ELF->sim->cycles path).

Repo-root discovery + run/report routing are shared via ``merlin.benchharness``; the perf-specific
math (align/matmul_macs/utilization_pct) stays here; the array geometry it needs is
derived from the target's RTL facts, not declared.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

# An explicit snapshot root takes precedence over the enclosing live git checkout.
_HERE = Path(__file__).resolve()
_root = os.environ.get("MERLIN_REPO_ROOT", "").strip()
if not _root:
    _root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=str(_HERE.parent), capture_output=True, text=True
    ).stdout.strip()
REPO = Path(_root).expanduser().resolve() if _root else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness import reports_root, runs_root  # noqa: E402
from merlin.common.paths import env as _env  # noqa: E402
from merlin.perf.workload_gen import tile_geometry  # noqa: E402

EXP = REPO / "merlin" / "experiments" / "gemmini_perf_bench"
KERNELS = EXP / "kernels"  # one capsule dir per kernel + corpus.yaml
RUNS = runs_root("gemmini", "perf-bench")  # runs/gemmini/perf-bench
REPORTS = reports_root("plots", "gemmini", "perf-bench")  # artifacts/plots/gemmini/perf-bench
# External model corpus — resolve via .env (MERLIN_M2M_DIR), NOT a "/path/to/..." placeholder.
MODEL2MLIR = Path(_env("MERLIN_M2M_DIR", str(REPO.parent / "model2MLIR"))) / "workloads"

# The systolic array's edge, DERIVED from this target's own RTL discovery rather than written down.
# The hardcoded 16 was correct for the config this bench happened to run and silently wrong for the
# 8x8 and 32x32 Gemmini configs that are also built on disk: every utilization number would have been
# off by the square of the ratio, with nothing in the output saying which array it was about.
# `tile_geometry` fails closed when RTL discovery reports no array, so an underivable edge is an error
# here rather than a plausible default that produces a wrong percentage.
#
# Derived when first READ, not when this module is imported. Every bench script imports this module for
# its repo bootstrap and run/report roots, and most of them never touch the array edge; deriving it at
# import made each of them -- and every test that imports one to check its argument handling -- require
# the target's RTL checkout. The refusal is unchanged; it now reaches the code that needs the number.
TARGET = "gemmini"  # this bench is ABOUT one target; the geometry still is not


@functools.cache
def _mesh():
    return tile_geometry(TARGET)


def __getattr__(name: str):
    # PEP 562: ``PB.DIM`` / ``PB.PEAK_MACS_PER_CYCLE`` keep their spelling for every caller.
    if name == "DIM":
        return _mesh().rows
    if name == "PEAK_MACS_PER_CYCLE":
        return _mesh().rows * _mesh().cols
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def align(n: int, m: int | None = None) -> int:
    """Round n up to a multiple of m, by default the array edge (Gemmini tiles are DIM-padded)."""
    m = _mesh().rows if m is None else m
    return ((int(n) + m - 1) // m) * m


def matmul_macs(M: int, K: int, N: int) -> int:
    return int(M) * int(K) * int(N)


def utilization_pct(macs: int, cycles: int | None) -> float | None:
    """Hardware utilization = useful MACs / (cycles x peak MACs/cycle). Diagnostic only."""
    if not cycles or cycles <= 0:
        return None
    return round(100.0 * macs / (cycles * _mesh().rows * _mesh().cols), 2)
