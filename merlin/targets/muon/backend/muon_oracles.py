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
        from . import muon_mx_codegen as _mx
        target = cb.get("target", "radiance")
        # A block-scaled MX capsule is graded on the HARNESS's reference MX kernel, whatever the artifact.
        #
        # This branch used to exist only inside program_from_cb, i.e. only on the inline-SOURCE path. An
        # agent that emits LLVM-dialect MLIR -- the thesis path -- went to compile_mlir_forkfree instead,
        # whose operand derivation (args_from_cb) has no MX branch: it embeds the decoded element codes as
        # plain f32 and DROPS the E8M0 block scales. Those scales are corpus-seeded from the capsule-name
        # salt and exist only in the golden, so the declared operation is not computable from what the
        # kernel receives. Measured on R5: a plain f32 matmul of the embedded operands lands 252/256
        # elements outside tolerance (max_rel 68), and applying the row/column scales still leaves 159
        # outside -- the golden comes from mlc's mx_ref hardware semantics, not a rescale. Every MX capsule
        # therefore failed for every MLIR submission, identically, in every round, on both arms.
        #
        # ⚠️ These passes measure the REFERENCE KERNEL, not the submission, so they must never be read as
        # compiler quality. The toolchain stamp says so explicitly and rides into the capsule result, so a
        # score that includes them stays decomposable rather than silently overstating the backend by the
        # size of the MX set (which is what an earlier 40/40 on this corpus did).
        # Gate on `mx_operands` ALONE. It is attached by the grading runner from the capsule's own golden
        # (capsule_golden.mx_operands returns None for a non-MX golden), so it is authoritative about
        # whether THIS CAPSULE is block-scaled.
        #
        # The previous condition also required `is_mx_cb(cb)`, which inspects the dtype string the AGENT
        # wrote into its command buffer -- so the agent's spelling chose the grading path. Measured live:
        # one arm spelled the operand `f8E4M3FN` and another `mxfp8` for the SAME capsule
        # (R5_mx_tile_mxfp8); only the first matched, so the second silently took the fork-free path and
        # failed a capsule it could not win. Worse, the registry says `f8E4M3FN` is per-tensor OCP fp8 and
        # NOT block-scaled at all, while `mxfp8` is exactly `mx_block`/`block_e8m0` -- so the string test
        # was matching the wrong formats in both directions.
        _mxprog = None
        if cb.get("mx_operands"):
            try:
                _mxprog = _mx.emit_mx_kernel(cb["mx_operands"], _mx.mx_output_name(cb))
            except Exception as _mxe:  # noqa: BLE001 — emitter fails closed (e.g. fp6/fp4 flash)
                # Do NOT swallow this silently: a fall-through here grades the submission on operands it
                # cannot use, which is the failure this whole branch exists to prevent. Record why.
                _mxprog = None
                _mx_refusal = f"{type(_mxe).__name__}: {_mxe}"
        if _mxprog is not None:
            _elf, _tc = muon.compile_for_oracle(_mxprog, workdir, target=target)
            elf, toolchain = _elf, f"mx-reference-kernel(not-the-submission;{_tc})"
        elif muon.is_mlir_artifact(kernel_src):
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


#: The env spelling this oracle has always honored, kept as an OVERRIDE. It used to be the ONLY way to
#: reach a GSIM emulator, and `default_adapters` gated the whole L3 engine choice on it being set — so on
#: every machine where nobody exported it, L3 was Verilator and nothing said why. The derived home under
#: the build root (``out/build/rtl_engines/<target>/gsim/emulator``) is what normally answers now.
GSIM_EMU_ENV = "MERLIN_MUON_GSIM_EMU"


def gsim_status(target: str) -> tuple[bool, str]:
    """``(available, reason)`` for this target's GSIM cert engine — BOTH halves of what it needs.

    The emulator is resolved (and its build receipt checked) by the shared
    :mod:`merlin.targetgen.gsim_emulator`; the rv64 SoC-fuse toolchain is this oracle's own extra
    requirement, since the emitted model is driven with a fused SoC image. Reported as a sentence rather
    than a bool because the two failures need different work, and the selection record is where anyone
    finds out which one happened.
    """
    from merlin.targetgen import gsim_emulator as _gsim
    ok, why = _gsim.probe(target, env_var=GSIM_EMU_ENV)
    if not ok:
        return False, why
    if not (muon.soc_fuse_dir() / "fuse_rv32_into_rv64.sh").is_file() or muon.rv64_cross_prefix() is None:
        return False, (f"{why}, but the rv64 SoC-fuse toolchain is absent — the emitted model is driven "
                       f"with a fused SoC image, so GSIM cannot run without it")
    return True, why


def verilator_status(target: str) -> tuple[bool, str]:
    """``(available, reason)`` for the Verilator cert engine, in the shape the engine policy consumes.

    It exists so the SLOW engine also has to say why it was chosen: with only GSIM reporting, a run that
    fell back read as though nothing had been decided.
    """
    try:
        if muon.available("verilator"):
            return True, f"Verilator RTL sim available for {target!r} (the slow elaborated-RTL engine)"
    except Exception as exc:                    # noqa: BLE001 — a broken probe is not availability
        return False, f"Verilator probe raised {type(exc).__name__}: {exc}"
    return False, ("Verilator RTL sim unavailable (needs the sim binary, the dramsim ini, and the rv64 "
                   "SoC-fuse toolchain)")


def l3_selection(target: str) -> dict:
    """Which elaborated-RTL engine certifies this SIMT target, routed through the SHARED cost policy.

    This path used to make the choice inline — ``if MERLIN_MUON_GSIM_EMU: L3 = gsim`` — which is the same
    decision the policy makes but with no record of it. Nothing downstream could tell a Verilator cert
    chosen because GSIM was missing from one chosen because GSIM was never considered, and that is
    precisely the question anyone asks of a cert that took 45 minutes a capsule.

    Raises :class:`~merlin.targetgen.rtl_engine_policy.NoEngineAvailable` when neither can run — the
    caller reports the tier absent rather than substituting a lesser one.
    """
    from merlin.targetgen import rtl_engine_policy as _pol
    return _pol.select(target, {"gsim": lambda: gsim_status(target),
                                "verilator": lambda: verilator_status(target)})


def gsim_muon_adapter(target_name: str | None = None) -> Callable:
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
        flops = flops_from_cb(cb)
        target = target_name or cb.get("target", "radiance")
        # The GSIM emu loads a fused rv64 SoC image (rv32 Muon kernel + rv64 Rocket carrier) via +loadmem,
        # exactly like the Verilator path, so it needs the same SoC-fuse toolchain. Both requirements —
        # and the emulator's own provenance — are checked by `gsim_status`, which resolves through the
        # shared home (env override, then out/build/rtl_engines/<target>/gsim/emulator) and REFUSES an emulator whose
        # build receipt describes different bytes rather than certifying against an unknown revision.
        from merlin.targetgen import gsim_emulator as _gsim
        _ok, _why = gsim_status(target)
        if not _ok:
            raise muon.MuonUnavailable(f"GSIM oracle unavailable: {_why}")
        emu = str(_gsim.emulator_path(target, env_var=GSIM_EMU_ENV))
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
        # SPOOL THE CONSOLE TO DISK, NEVER TO RAM. This console carries a per-instruction DASM commit
        # trace, so its size scales with RETIRED INSTRUCTIONS -- not with the cycle cap, and not with
        # anything the caller can see. `capture_output=True` buffers all of it in the parent: measured, a
        # 12,000,000-cycle run on one 16x128x128 tile retired 3.4M instructions and the buffer reached
        # 72.67 GB, which took a 125 GB node to 96% and made Ray's OOM killer terminate an unrelated
        # 10-hour experiment on the same host. Raising MERLIN_MUON_GSIM_MAXCYCLES was unsafe for exactly
        # this reason. Spooling to a file in the run's own workdir keeps parent memory bounded by the
        # read-back window below, and keeps the FULL console on disk for debugging (strictly more than
        # the old path retained, which was capped at the 600-char failure tail anyway).
        log = Path(workdir) / "gsim_console.log"
        try:
            with log.open("wb") as fh:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                               preexec_fn=_unlimited_stack)
        except subprocess.TimeoutExpired as e:
            raise muon.MuonUnavailable(f"GSIM emu wall-timed out after {timeout}s") from e
        t2 = time.perf_counter()
        console, console_bytes, console_truncated = muon._read_console(log)
        # GSIM completion contract (the radiance kernels self-verify against their embedded golden, then
        # go idle on PASS or spin on FAIL):
        #   PASS  => the GPUResetAggregator's stopSim (GPU idle for 1k cycles) fires, which the emitted
        #            model turns into an early ``exit(0)`` BEFORE the cycle cap -- the harness never prints
        #            its "FINISHED: cycles=<cap>" line.
        #   FAIL  => the verify kernel spins, the GPU never idles, and the RTL rdtime watchdog trips the
        #            "Timeout exceeded" PlusArgTimeout assertion (or the run hits the +max-cycles cap).
        # Grade on those observables (structural string containment; no regex, no fabricated pass) --
        # and grade on POSITIVE evidence, not merely on the absence of the two failure markers.
        #
        # MEASURED 2026-09-04, which is why this is not written as a double negative any more: a
        # radiance capsule compiled fork-free, fused, and ran 386,090 cycles on the GSIM model, and the
        # console carried NONE of the four markers -- no `Cycles:`, no `finished execution`, and equally
        # no `Timeout exceeded` and no `FINISHED: cycles=`. The emulator's own stats line read
        # `dram_aw=0 dram_w=0 writes_resultpage=0 uart_chars=0`: the kernel wrote nothing and printed
        # nothing. The old test passed it, because a test that only checks that failure markers are
        # ABSENT cannot tell "the GPU went idle having finished" from "this harness never printed a
        # word". A check that could not run must not report success.
        #
        # The Verilator sibling has always demanded its positive marker (`_run_verilator` grades on
        # "Muon [...] finished execution."), so this only brings the two engines to one standard.
        failed = ("Timeout exceeded" in console or f"FINISHED: cycles={maxcyc}" in console)
        # `_read_console` hoists marker lines out of a truncated console precisely so this search is
        # over the whole run and not over the tail window.
        witness = next((m for m in ("Cycles:", "finished execution") if m in console), None)
        if failed:
            raise muon.MuonUnavailable(
                "GSIM RTL model ran but the kernel did not reach GPU-idle completion within "
                f"{maxcyc} cycles (self-verify failed / hung: rdtime watchdog or cycle cap). "
                f"tail:\n{console[-600:]}")
        if witness is None:
            raise muon.MuonUnavailable(
                "GSIM RTL model ran and neither failed nor completed OBSERVABLY: the console carries "
                f"no completion witness ({('Cycles:', 'finished execution')}) and no failure marker "
                f"either, over {console_bytes} bytes on disk. That is an unread instrument, not a "
                "pass -- a kernel whose output never reaches the console is indistinguishable here "
                "from one that finished, so the tier reports unavailable rather than certifying it. "
                f"tail:\n{console[-600:]}")
        cycles = muon._cycles_from_rtl_report(console)
        return {
            "outputs": {},
            "cycles": cycles,
            "oracle": {"kind": "rtl_gsim_muon", "derived_from_rtl": True},
            "console": console,
            # Say plainly that the console in this record is a window, and where the whole thing is. A
            # truncated console read as complete is how "the marker is absent" gets confused with "the
            # marker never printed", which on this contract is the difference between pass and fail.
            "console_spool": {"path": str(log), "bytes_on_disk": console_bytes,
                              "truncated": console_truncated,
                              "markers_preserved": list(muon._GSIM_MARKERS)},
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


def default_adapters(target: str | None = None) -> dict[str, Callable]:
    """Tier -> adapter for the Muon runner. L2 = cyclotron (perf), L3 = VCS-RTL cert. The RTL-arc readback
    oracle (``arc_readback_adapter``) is the sim-independent, multi-warp-capable grade a real new target has
    (its RTL-compiled model); selected by the harness when a run needs memory-readback rather than console.

    ``L3-smoke`` — the ADVISORY bounded-cycle Verilator executability backstop — is added ONLY when
    ``MERLIN_EXEC_SMOKE`` is set, so a normal grade is byte-identical when it is off. It is opt-in
    (off by default) because it spends minutes of Verilator per capsule; when on, the runner records it as
    a non-mandatory, never-blocking ``executability`` tier (RTL-legality grounding for the L2 oracle)."""
    import os
    tgt = target or "radiance"
    adapters: dict[str, Callable] = {"L2": cyclotron_adapter()}
    # WHICH elaborated-RTL engine certifies L3 is a COST decision at equal fidelity, and the SHARED policy
    # makes it (vcs > gsim > verilator) instead of an inline `if the env var is set`. GSIM and Verilator
    # grade the SAME fork-free ELF on the SAME RTL completion contract; GSIM (FIRRTL->C++) is ~an order of
    # magnitude faster. What changes here is not the outcome but the RECORD: the selection, and every
    # engine passed over WITH the reason, is printed — so a Verilator cert can never again be mistaken for
    # a considered choice when it was really "nobody exported the env var".
    _engine_adapter = {"gsim": lambda: gsim_muon_adapter(tgt), "verilator": verilator_muon_adapter}
    _selected_engine = ""
    try:
        from merlin.targetgen import rtl_engine_policy as _pol
        _sel = l3_selection(tgt)
        _selected_engine = _sel["engine"]
        adapters["L3"] = _engine_adapter[_selected_engine]()
        print(f"[oracle] {tgt} L3 engine: {_pol.describe(_sel)} — {_sel['reason']}", flush=True)
        for _c in _sel["considered"]:
            if not _c["available"]:
                print(f"[oracle] {tgt} L3 passed over {_c['engine']}: {_c['reason']}", flush=True)
    except Exception as exc:  # noqa: BLE001 — no elaborated-RTL engine: NO L3, never a silent demotion
        print(f"[oracle] {tgt}: no elaborated-RTL engine for L3 — {exc}", flush=True)
    # The Verilator cert is NOT kept alongside a GSIM one by default: the runner executes every adapter in
    # the tier map, so a co-resident ``L3-verilator`` would run the slow Verilator sim on every capsule and
    # erase GSIM's whole speed advantage. Opt in to the (advisory) cross-check with
    # MERLIN_MUON_L3_VERILATOR_ALSO.
    if _selected_engine == "gsim":
        if os.environ.get("MERLIN_MUON_L3_VERILATOR_ALSO", "").strip().lower() in ("1", "true", "yes", "on"):
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
