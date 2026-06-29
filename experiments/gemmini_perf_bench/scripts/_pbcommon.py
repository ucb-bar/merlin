"""Shared helpers for the gemmini_perf_bench cross-approach performance benchmark.

This benchmark drives the SAME kernels (int8 matmul/conv shapes) through multiple Gemmini
code-generation approaches (golden bareMetalC, generated MLIR OOT backends, the hand-written C++
Gemmini dialect via IREE) and compares cycles / wall-time / utilization / correctness. It reuses the
capsule_bench_v0 libraries (capsule emit, deterministic golden, ELF->sim->cycles path)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "gemmini_perf_bench"
KERNELS = EXP / "kernels"            # one capsule dir per kernel + kernel_corpus.yaml
# Generated outputs live under the canonical three-root layout (see CLAUDE.md
# "Generated-output convention"): runs under runs/, figures under artifacts/plots/.
RUNS = REPO / "runs" / "gemmini" / "perf-bench"
REPORTS = REPO / "artifacts" / "plots" / "gemmini" / "perf-bench"
MODEL2MLIR = Path("/scratch/agustin/projects/model2MLIR/workloads")

# Gemmini systolic array dimension (16x16 PE) -> peak 256 MACs/cycle. Used for utilization.
DIM = 16
PEAK_MACS_PER_CYCLE = DIM * DIM

sys.path.insert(0, str(REPO / "merlin" / "python"))


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
