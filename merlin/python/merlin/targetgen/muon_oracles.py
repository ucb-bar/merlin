"""Muon oracle adapters for the parallel Muon capsule runner.

An adapter has the same shape the Gemmini ``capsule_runner`` uses --
``(cb, kernel_src, workdir, timeout) -> {outputs, cycles, oracle, console, timing, ...}`` -- or it
raises :class:`merlin.runtime.backends.muon.MuonUnavailable` to signal honest unavailability (never a
silent pass). Unlike the Gemmini adapter, ``kernel_src`` is the **SIMT C++ kernel** emitted by the
Muon backend's ``lower_target_to_llvm`` entrypoint (clang-muon compiles it), not LLVM-dialect MLIR.

This module imports nothing from the frozen Gemmini ``capsule_runner``; it is a parallel path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ..runtime.backends import muon


def flops_from_cb(cb: dict[str, Any]) -> int | None:
    """Best-effort FLOP count for a command buffer (2*M*K*N per matmul). None if not derivable."""
    tensors = cb.get("tensors", {})
    # resolve resident handles (RES_PACK dst -> src) so a matmul rhs like "W_res" finds W's shape
    resident_source = {c["operands"]["dst"]: c["operands"]["src"]
                       for c in cb.get("commands", []) if c.get("opcode") == "RES_PACK"}

    def shape(name: str) -> list[int] | None:
        name = resident_source.get(name, name)
        t = tensors.get(name)
        return t.get("shape") if isinstance(t, dict) else None

    total = 0
    found = False
    for cmd in cb.get("commands", []):
        op = (cmd.get("opcode") or "").upper()
        if "MATMUL" not in op and "GEMM" not in op:
            continue
        ops = cmd.get("operands", {})
        lhs = shape(ops.get("lhs", ""))
        rhs = shape(ops.get("rhs", "")) or shape(ops.get("weight", ""))
        if lhs and rhs and len(lhs) == 2 and len(rhs) == 2:
            m, k = lhs
            k2, n = rhs
            total += 2 * m * k * n
            found = True
    return total if found else None


def _timing(build_s: float, sim_s: float) -> dict:
    return {"build_s": round(build_s, 3), "sim_active_s": round(sim_s, 3), "oracle_wait_s": 0.0}


def _adapter(simulator: str) -> Callable:
    def run(cb: dict, kernel_src: str, workdir: str | Path, timeout: int) -> dict:
        if not muon.available(simulator):
            raise muon.MuonUnavailable(f"muon {simulator} oracle not available")
        flops = flops_from_cb(cb)
        t0 = time.perf_counter()
        elf = muon.compile_kernel(kernel_src, workdir)
        t1 = time.perf_counter()
        console, cycles, summary = muon.run_elf(elf, simulator=simulator, timeout=timeout)
        t2 = time.perf_counter()
        outputs, raw = muon.parse_output(console, cycles)
        return {
            "outputs": outputs,
            "cycles": cycles,
            "oracle": dict(muon.ORACLE[simulator]),
            "console": console,
            "timing": _timing(t1 - t0, t2 - t1),
            "gflops": muon.gflops(flops, cycles),
            "pct_fp_peak": muon.pct_fp_peak(flops, cycles),
            "summary": summary,
        }
    return run


def cyclotron_adapter() -> Callable:
    """Primary Muon oracle: cyclotron --timing (perf model). Correctness + cycles + %FP-peak."""
    return _adapter("cyclotron")


def vcs_muon_adapter() -> Callable:
    """Certification oracle: RadianceMuonConfig VCS RTL + cyclotron DPI difftest (cycle-exact).

    Fails closed (``MuonUnavailable``) when the kernel-level difftest stalls (WIP upstream).
    """
    return _adapter("vcs")


def default_adapters() -> dict[str, Callable]:
    """Tier -> adapter for the Muon runner. L2 = cyclotron (perf), L3 = VCS-RTL cert."""
    return {"L2": cyclotron_adapter(), "L3": vcs_muon_adapter()}
