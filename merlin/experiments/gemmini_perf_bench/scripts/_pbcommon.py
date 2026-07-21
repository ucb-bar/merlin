"""Shared helpers for the gemmini_perf_bench cross-approach performance benchmark.

This benchmark drives the SAME kernels (int8 matmul/conv shapes) through multiple Gemmini
code-generation approaches (golden bareMetalC, generated MLIR OOT backends, the hand-written C++
Gemmini dialect via IREE) and compares cycles / wall-time / utilization / correctness. It reuses the
capsule_bench_v0 libraries (capsule emit, deterministic golden, ELF->sim->cycles path).

Repo-root discovery + run/report routing are shared via ``merlin.benchharness``; the perf-specific
constants + math (DIM, PEAK, align/matmul_macs/utilization_pct) stay here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Self-contained bootstrap (git first, parents[] fallback), then put merlin/python on the path.
_HERE = Path(__file__).resolve()
_git = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(_HERE.parent),
                      capture_output=True, text=True).stdout.strip()
REPO = Path(_git) if _git else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness import runs_root, reports_root  # noqa: E402
from merlin.common.paths import env as _env  # noqa: E402

EXP = REPO / "merlin" / "experiments" / "gemmini_perf_bench"
KERNELS = EXP / "kernels"                                  # one capsule dir per kernel + corpus.yaml
RUNS = runs_root("gemmini", "perf-bench")                  # runs/gemmini/perf-bench
REPORTS = reports_root("plots", "gemmini", "perf-bench")   # artifacts/plots/gemmini/perf-bench
# External model corpus — resolve via .env (MERLIN_M2M_DIR), NOT a "/path/to/..." placeholder.
MODEL2MLIR = Path(_env("MERLIN_M2M_DIR", "/scratch/agustin/projects/model2MLIR")) / "workloads"

# Gemmini systolic array dimension (16x16 PE) -> peak 256 MACs/cycle. Used for utilization.
DIM = 16
PEAK_MACS_PER_CYCLE = DIM * DIM


def align(n: int, m: int = DIM) -> int:
    """Round n up to a multiple of m (Gemmini tiles are DIM-padded)."""
    return ((int(n) + m - 1) // m) * m


def matmul_macs(M: int, K: int, N: int) -> int:
    return int(M) * int(K) * int(N)


def utilization_pct(macs: int, cycles: int | None) -> float | None:
    """Hardware utilization = useful MACs / (cycles x peak MACs/cycle). Diagnostic only."""
    if not cycles or cycles <= 0:
        return None
    return round(100.0 * macs / (cycles * PEAK_MACS_PER_CYCLE), 2)
