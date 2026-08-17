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

from . import muon


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
        from . import muon_harness as _mh
        target = cb.get("target", "radiance")
        if muon.is_mlir_artifact(kernel_src):
            # THESIS PATH: the agent emitted an LLVM-dialect MLIR kernel (a compiler lowering). Build it
            # fork-free (stock LLVM rv32 + RTL-derived Muon re-encode + runner-owned external-kernel harness);
            # this path is fork-free by construction (never clang-muon), so the toolchain stamp is "fork-free".
            elf, toolchain = muon.compile_mlir_forkfree(kernel_src, cb, workdir, target=target), "fork-free"
        else:
            program = _mh.program_from_cb(cb, kernel_src, muon._model_for(target)) or kernel_src
            elf, toolchain = muon.compile_for_oracle(program, workdir, target=target)
        t1 = time.perf_counter()
        console, cycles, summary = muon.run_elf(elf, simulator=simulator, timeout=timeout)
        t2 = time.perf_counter()
        completion_only = False
        try:
            outputs, raw = muon.parse_output(console, cycles)
        except muon.MuonError:
            # The Verilator RTL harness runs the kernel to completion (``run_elf`` only returns here
            # once it reached the RTL "finished execution" marker) but does not surface the kernel's
            # UART console — its ``$finish`` races the print flush, so there are no OUT/DONE lines to
            # numeric-grade. It still certifies RTL COMPLETION + cycle-accurate cycles; CORRECTNESS is
            # the REQUIRED functional tier's (cyclotron L2) job. Any other engine (cyclotron) MUST
            # produce console output, so its absence there is a real error — reraise.
            if simulator != "verilator":
                raise
            outputs, completion_only = {}, True
        result = {
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
        if completion_only:
            result["completion_only"] = True
        return result
    return run


def cyclotron_adapter() -> Callable:
    """Primary Muon oracle: cyclotron --timing (perf model). Correctness + cycles + %FP-peak."""
    return _adapter("cyclotron")


def vcs_muon_adapter() -> Callable:
    """Certification oracle: RadianceMuonConfig VCS RTL + cyclotron DPI difftest (cycle-exact).

    Fails closed (``MuonUnavailable``) when the kernel-level difftest stalls (WIP upstream).
    """
    return _adapter("vcs")


def verilator_muon_adapter() -> Callable:
    """Certification oracle: the RadianceTapeoutSimConfig Verilator RTL sim (open-source cycle-accurate,
    the harness the radiance kernels are evaluated on), replacing the WIP VCS difftest. It fuses the
    emitted ELF into the rv64 SoC carrier, loads it via the ``+loadmem`` backdoor, and runs to the RTL
    ``finished execution`` marker (see :func:`..runtime.backends.muon._run_verilator`). Fails closed
    (``MuonUnavailable``) when the sim / dramsim ini / rv64 SoC-fuse toolchain is absent, so an
    unavailable RTL cert is honest (the tier is optional-when-available), never a fabricated pass."""
    return _adapter("verilator")


def gsim_muon_adapter() -> Callable:
    """Certification oracle: the GSIM-emitted C++ cycle-accurate model of the RadianceGsimConfig SoC.

    GSIM compiles the design's FIRRTL to standalone C++ (an RTL-DERIVED simulator, like Verilator but
    an order of magnitude faster), which is built once into a self-contained ``emu`` binary. This adapter
    mirrors :func:`verilator_muon_adapter` exactly -- it builds the SAME fork-free rv32 ELF, fuses it into
    the rv64 SoC carrier (:func:`..muon.fuse_soc_elf`), and drives it via the ``+loadmem`` backdoor -- but
    runs the prebuilt GSIM emulator instead of the Verilator sim. The radiance kernels self-verify against
    an embedded golden and then go idle on PASS (the emitted model turns the GPUResetAggregator ``stopSim``
    into an early ``exit(0)`` before the cycle cap) or spin on FAIL (the RTL rdtime watchdog trips the
    ``Timeout exceeded`` assertion). Completion is graded on those observables, so a pass means the same
    thing it does on Verilator.

    Gated on the GSIM emu binary (``MERLIN_MUON_GSIM_EMU``, a compiled snapshot of the emitted model) AND
    the rv64 SoC-fuse toolchain being present; fails closed (``MuonUnavailable``) when either is absent, so
    an unavailable GSIM cert is honest (never a fabricated pass), exactly like the Verilator tier. The
    wall-cycle budget is ``MERLIN_MUON_GSIM_MAXCYCLES`` (default 2_000_000)."""
    import os
    import resource
    import subprocess

    def run(cb: dict, kernel_src: str, workdir: str | Path, timeout: int) -> dict:
        emu = os.environ.get("MERLIN_MUON_GSIM_EMU", "").strip()
        if not emu or not Path(emu).is_file() or not os.access(emu, os.X_OK):
            raise muon.MuonUnavailable(
                "GSIM emulator not available -- set MERLIN_MUON_GSIM_EMU to the emitted GSIM emu binary")
        # The GSIM emu loads a fused rv64 SoC image (rv32 Muon kernel + rv64 Rocket carrier) via +loadmem,
        # exactly like the Verilator path, so it needs the same SoC-fuse toolchain.
        if not (muon.soc_fuse_dir() / "fuse_rv32_into_rv64.sh").is_file() or muon.rv64_cross_prefix() is None:
            raise muon.MuonUnavailable("rv64 SoC-fuse toolchain not available for the GSIM oracle")
        flops = flops_from_cb(cb)
        target = cb.get("target", "radiance")
        from . import muon_harness as _mh
        t0 = time.perf_counter()
        # Build the graded ELF identically to the Verilator/cyclotron adapters (fork-free thesis path when
        # the artifact is LLVM-dialect MLIR; otherwise the runner-owned harness + oracle compile).
        if muon.is_mlir_artifact(kernel_src):
            elf, toolchain = muon.compile_mlir_forkfree(kernel_src, cb, workdir, target=target), "fork-free"
        else:
            program = _mh.program_from_cb(cb, kernel_src, muon._model_for(target)) or kernel_src
            elf, toolchain = muon.compile_for_oracle(program, workdir, target=target)
        soc = muon.fuse_soc_elf(Path(elf), Path(workdir))
        t1 = time.perf_counter()
        maxcyc = os.environ.get("MERLIN_MUON_GSIM_MAXCYCLES", "2000000")

        def _unlimited_stack() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_STACK,
                                   (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            except (ValueError, OSError):
                pass

        # +max_core_cycles=0 disables the Rocket rdtime PlusArgTimeout watchdog (default 32). GSIM's
        # slow instruction-cache flush epilogue (invalidating 512 lines x N cores at kernel exit) is a
        # razor's edge past that default, so the watchdog would kill the run just as the GPU is going
        # idle; disabling it (the RTL's own "Off if 0") lets the flush drain and the GPU-idle stopSim
        # fire, which is when a passing self-verifying kernel cleanly exits. Not a correctness relaxation
        # -- the kernel's own on-chip self-verify still gates the clean exit.
        cmd = [emu, str(soc), f"+loadmem={soc}", "+max_core_cycles=0", f"+max-cycles={maxcyc}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  preexec_fn=_unlimited_stack)
        except subprocess.TimeoutExpired as e:
            raise muon.MuonUnavailable(f"GSIM emu wall-timed out after {timeout}s") from e
        t2 = time.perf_counter()
        console = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        # GSIM completion contract (the radiance kernels self-verify against their embedded golden, then
        # go idle on PASS or spin on FAIL):
        #   PASS  => the GPUResetAggregator's stopSim (GPU idle for 1k cycles) fires, which the emitted
        #            model turns into an early ``exit(0)`` BEFORE the cycle cap -- the harness never prints
        #            its "FINISHED: cycles=<cap>" line.
        #   FAIL  => the verify kernel spins, the GPU never idles, and the RTL rdtime watchdog trips the
        #            "Timeout exceeded" PlusArgTimeout assertion (or the run hits the +max-cycles cap).
        # Grade on those observables (structural string containment; no regex, no fabricated pass).
        completed = ("Timeout exceeded" not in console
                     and f"FINISHED: cycles={maxcyc}" not in console)
        if not completed:
            raise muon.MuonUnavailable(
                "GSIM RTL model ran but the kernel did not reach GPU-idle completion within "
                f"{maxcyc} cycles (self-verify failed / hung: rdtime watchdog or cycle cap). "
                f"tail:\n{console[-600:]}")
        cycles = muon._cycles_from_rtl_report(console)
        return {
            "outputs": {},
            "cycles": cycles,
            "oracle": {"kind": "rtl_gsim_muon", "derived_from_rtl": True},
            "console": console,
            "toolchain": toolchain,
            "timing": _timing(t1 - t0, t2 - t1),
            "gflops": muon.gflops(flops, cycles),
            "pct_fp_peak": muon.pct_fp_peak(flops, cycles),
            "completion_only": True,
        }
    return run


def _mx_ctrl_base(target: str) -> int | None:
    """The accelerator MMIO command-window base for ``target``, DERIVED from its ``mx_mmio`` fact (the
    same header-derived, provenance-tagged block the MX kernel emitter reads) — never a baked address.
    None for a target with no MX PE, which leaves the smoke's MX-engagement check honestly unknown."""
    try:
        from merlin.targetgen.rtl import mlc_bridge
        mm = mlc_bridge.mx_mmio_for(target)
        if mm and mm.get("ctrl_base") is not None:
            return int(mm["ctrl_base"])
    except Exception:  # noqa: BLE001 — no contract / no MX fact -> MX-engagement simply unknown
        pass
    return None


def verilator_smoke_adapter(max_cycles: int | None = None, seed: int | None = None) -> Callable:
    """ADVISORY oracle: a bounded-cycle Verilator EXECUTABILITY smoke — the RTL-grounding backstop for the
    (non-RTL-certified) cyclotron perf oracle. It compiles the SAME submission artifact the graded
    adapters do (fork-free), then runs it on the RadianceTapeoutSim Verilator build with a SMALL
    ``+max-cycles`` cap (:func:`..muon.run_elf_smoke`) so it CANNOT hang, and records whether the ELF at
    least RUNS on real RTL (boots, makes forward progress, no illegal-instruction/trap, and — for an MX
    capsule — the MX PE accepts a command). It does NOT numeric-grade (that is the cyclotron L2 oracle's
    job) and the result is carried as an ``executability`` field the runner records as a NON-mandatory,
    never-blocking tier. Fails closed (``MuonUnavailable``) when the RTL build / fuse toolchain is absent,
    so an unavailable smoke degrades honestly — it can never fail a capsule whose L2 grade passed."""
    import os

    def run(cb: dict, kernel_src: str, workdir: str | Path, timeout: int) -> dict:
        if not muon.available("verilator_smoke"):
            raise muon.MuonUnavailable("verilator RTL sim / rv64 SoC-fuse toolchain not available "
                                       "for the executability smoke")
        mc = int(max_cycles if max_cycles is not None
                 else os.environ.get("MERLIN_EXEC_SMOKE_MAXCYCLES", "40000"))
        target = cb.get("target", "radiance")
        from . import muon_harness as _mh
        t0 = time.perf_counter()
        # Build the graded artifact EXACTLY as the real oracle does, so the smoke certifies the same ELF.
        if muon.is_mlir_artifact(kernel_src):
            elf = muon.compile_mlir_forkfree(kernel_src, cb, workdir, target=target)
        else:
            program = _mh.program_from_cb(cb, kernel_src, muon._model_for(target)) or kernel_src
            elf, _tc = muon.compile_for_oracle(program, workdir, target=target)
        t1 = time.perf_counter()
        sig = muon.run_elf_smoke(elf, max_cycles=mc, timeout=timeout, seed=seed,
                                 mx_ctrl_base=_mx_ctrl_base(target))
        t2 = time.perf_counter()
        return {"executability": sig, "advisory": True,
                "console": sig.get("console_tail", ""),
                "cycles": sig.get("cycles"),
                "oracle": dict(muon.ORACLE["verilator_smoke"]),
                "timing": _timing(t1 - t0, t2 - t1)}
    return run


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
    from merlin.targetgen.contract.toolchain import mlir_bin
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
    (its RTL-compiled model); selected by the harness when a run needs memory-readback rather than console.

    ``L3-smoke`` — the ADVISORY bounded-cycle Verilator executability backstop — is added ONLY when
    ``MERLIN_EXEC_SMOKE`` is set, so a normal grade is byte-identical when it is off. It is opt-in
    (off by default) because it spends minutes of Verilator per capsule; when on, the runner records it as
    a non-mandatory, never-blocking ``executability`` tier (RTL-legality grounding for the L2 oracle)."""
    import os
    adapters: dict[str, Callable] = {"L2": cyclotron_adapter(), "L3": verilator_muon_adapter()}
    # When the GSIM emu is present, use the (much faster) RTL-derived GSIM model as the L3 cert and keep the
    # Verilator cert available alongside as ``L3-verilator``. Both grade the SAME fork-free ELF on the SAME
    # RTL completion contract; GSIM (FIRRTL->C++) is ~an order of magnitude faster than Verilator. When the
    # env is unset, L3 stays Verilator and a normal grade is byte-identical.
    if os.environ.get("MERLIN_MUON_GSIM_EMU", "").strip():
        adapters["L3"] = gsim_muon_adapter()
        adapters["L3-verilator"] = verilator_muon_adapter()
    if os.environ.get("MERLIN_EXEC_SMOKE", "").strip().lower() in ("1", "true", "yes", "on"):
        adapters["L3-smoke"] = verilator_smoke_adapter()
    # Offline L2-only certification: L3 is an RTL-cert tier that NEVER gates a capsule (required tiers are
    # L0/L1/L2), and a full-corpus Verilator L3 sweep costs many hours. When MERLIN_MUON_SKIP_RTL_L3 is set,
    # drop every RTL-cert tier (L3*/L4/L5) so the runner certifies the required functional ladder quickly;
    # the required L2 (cyclotron) oracle is untouched, so a pass means exactly what it did before.
    if os.environ.get("MERLIN_MUON_SKIP_RTL_L3", "").strip().lower() in ("1", "true", "yes", "on"):
        adapters = {t: a for t, a in adapters.items()
                    if not (t.startswith("L3") or t in ("L4", "L5"))}
    return adapters
