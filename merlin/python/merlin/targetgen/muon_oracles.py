"""Muon oracle adapters for the parallel Muon capsule runner.

An adapter has the same shape the Gemmini ``capsule_runner`` uses --
``(cb, kernel_src, workdir, timeout) -> {outputs, cycles, oracle, console, toolchain, timing, ...}`` --
or it raises :class:`merlin.runtime.backends.muon.MuonUnavailable` to signal honest unavailability (never
a silent pass). ``kernel_src`` is the artifact the Muon backend's ``lower_target_to_llvm`` entrypoint emits:
per the generic ``kernel_abi`` it is a whole-computation kernel FUNCTION (``{target}_kernel(...)``), which
the runner-owned self-contained-C harness (:mod:`..runtime.backends.muon_harness`) wraps so the FORK-FREE
driver builds it with a stock toolchain; a full-program artifact (one with ``main``) is compiled directly.
The result records ``toolchain`` (``fork-free`` vs the eval-only ``clang-muon-fork``) so the experiment
measures fork-free coverage.

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
        # If the artifact is a whole-computation kernel FUNCTION (the generic kernel_abi), wrap it in the
        # runner-owned self-contained-C harness (operands from the cb) so the fork-free driver can build it;
        # a full-program artifact (has main) is passed through. Then prefer the FORK-FREE thesis path (stock
        # LLVM + RTL-derived transcode) and record which toolchain produced the graded ELF, so the experiment
        # measures fork-free coverage and never hides a fork fallback (MERLIN_MUON_FORKFREE_ONLY fails closed).
        from ..runtime.backends import muon_harness as _mh
        target = cb.get("target", "radiance")
        program = _mh.program_from_cb(cb, kernel_src, muon._model_for(target)) or kernel_src
        elf, toolchain = muon.compile_for_oracle(program, workdir, target=target)
        t1 = time.perf_counter()
        console, cycles, summary = muon.run_elf(elf, simulator=simulator, timeout=timeout)
        t2 = time.perf_counter()
        outputs, raw = muon.parse_output(console, cycles)
        return {
            "outputs": outputs,
            "cycles": cycles,
            "oracle": dict(muon.ORACLE[simulator]),
            "console": console,
            "toolchain": toolchain,
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


def _shape2d(shape) -> tuple[int, int]:
    dims = [int(d) for d in (shape or []) if int(d) > 0] or [1]
    if len(dims) == 1:
        return 1, dims[0]
    r = 1
    for d in dims[:-1]:
        r *= d
    return r, dims[-1]


def _output_symbols(elf: str | Path) -> dict[str, int]:
    """Symbol -> address for the ELF's globals, read structurally from the object's symbol table (no regex)."""
    import subprocess
    from .contract.toolchain import mlir_bin
    st = subprocess.run([str(mlir_bin("llvm-objdump")), "-t", str(elf)], capture_output=True, text=True).stdout
    out: dict[str, int] = {}
    for ln in st.splitlines():
        parts = ln.split()
        if len(parts) >= 2 and all(c in "0123456789abcdef" for c in parts[0]) and parts[0]:
            out[parts[-1]] = int(parts[0], 16)
    return out


def arc_readback_adapter() -> Callable:
    """RTL-derived MULTI-WARP oracle: build the emitted kernel fork-free, run it on the target's arc model
    (the model mlc COMPILES from the RTL via CIRCT-arc — the oracle a real new target actually has, unlike a
    vendor sim), and read each output tensor's buffer back from memory. No console print, so it grades a
    multi-warp SIMT run faithfully (console output races across lanes). Each cb output tensor is expected to be
    a file-scope global of the SAME name in the emitted program (the kernel_abi output-buffer contract); its
    ``[addr, addr+rows*cols*4)`` region is read back. Fails closed (``MuonUnavailable``) when the arc model is
    absent — never fabricates a verdict."""
    import struct

    def run(cb: dict, kernel_src: str, workdir: str | Path, timeout: int) -> dict:
        target = cb.get("target", "radiance")
        if not muon.arc_oracle_available(target):
            raise muon.MuonUnavailable(f"RTL-arc model for {target!r} not available")
        t0 = time.perf_counter()
        elf, toolchain = muon.compile_for_oracle(kernel_src, workdir, target=target)
        syms = _output_symbols(elf)
        tensors = cb.get("tensors", {})
        outputs: dict[str, list] = {}
        for name, t in tensors.items():
            if not (isinstance(t, dict) and str(t.get("role", "")).lower() in ("output", "out", "dst")):
                continue
            if name not in syms:
                raise muon.MuonError(f"output tensor {name!r} is not a global in the emitted kernel "
                                     "(the arc-readback oracle needs the output buffer as a named global)")
            r, c = _shape2d(t.get("shape"))
            data = muon.run_elf_arc(elf, target=target, base=syms[name], length=r * c * 4, timeout=timeout)
            outputs[name] = [list(struct.unpack("<%dI" % c, data[i * c * 4:(i + 1) * c * 4])) for i in range(r)]
        t1 = time.perf_counter()
        return {"outputs": outputs, "oracle": {"kind": "rtl-arc", "source": "mlc-cosim-from-rtl"},
                "toolchain": toolchain, "timing": _timing(t1 - t0, 0.0)}
    return run


def default_adapters() -> dict[str, Callable]:
    """Tier -> adapter for the Muon runner. L2 = cyclotron (perf), L3 = VCS-RTL cert. The RTL-arc readback
    oracle (``arc_readback_adapter``) is the sim-independent, multi-warp-capable grade a real new target has
    (its RTL-compiled model); selected by the harness when a run needs memory-readback rather than console."""
    return {"L2": cyclotron_adapter(), "L3": vcs_muon_adapter()}
