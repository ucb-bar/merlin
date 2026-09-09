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

import hashlib
import json
import os
import struct
import subprocess
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


def _gsim_cycle_budget(cb: dict, *, target: str, flops: int | None) -> tuple[int, dict[str, Any]]:
    """Resolve a positive GSIM cap, preferring a per-capsule L2 measurement.

    The L2-derived cap is an observation bound, never a correctness verdict: exhausting it remains
    unavailable.  An explicit GSIM override keeps its historical precedence.  Until the staged-cohort
    materializer carries the sealed L2 value into ``_oracle_l2_cycles``, a one-capsule launcher may provide
    ``MERLIN_MUON_GSIM_L2_CYCLES``; absence retains the legacy cap rather than inventing a measurement.
    """
    explicit = os.environ.get("MERLIN_MUON_GSIM_MAXCYCLES")

    def positive(value: Any, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise muon.MuonUnavailable(f"{label} must be a positive integer, got {value!r}") from exc
        if parsed <= 0:
            raise muon.MuonUnavailable(f"{label} must be a positive integer, got {value!r}")
        return parsed

    if explicit is not None:
        value = positive(explicit, "MERLIN_MUON_GSIM_MAXCYCLES")
        return value, {"source": "explicit_override", "max_cycles": value}

    l2_raw = cb.get("_oracle_l2_cycles")
    policy = cb.get("_oracle_gsim_cycle_policy")
    sealed = l2_raw is not None
    if l2_raw is None:
        # Diagnostic-only seam for a focused one-capsule replay. Staged evaluation never relies on it:
        # its materialized capsule supplies both the sealed count and descriptor policy above.
        l2_raw = os.environ.get("MERLIN_MUON_GSIM_L2_CYCLES")
    if l2_raw is None:
        return 2_000_000, {"source": "legacy_default", "max_cycles": 2_000_000}
    l2_cycles = positive(l2_raw, "sealed per-capsule L2 cycles")
    if sealed:
        if not isinstance(policy, dict) or policy.get("source") != "sealed_predecessor_tier":
            raise muon.MuonUnavailable(
                "sealed per-capsule L2 cycles require a descriptor-derived GSIM cycle policy")
        factor = positive(policy.get("multiplier"), "GSIM cycle policy multiplier")
        minimum = positive(policy.get("minimum_cycles"), "GSIM cycle policy minimum_cycles")
        floor_factor = positive(policy.get("compute_floor_multiplier"),
                                "GSIM cycle policy compute_floor_multiplier")
    else:
        factor = positive(os.environ.get("MERLIN_MUON_GSIM_L2_CYCLE_FACTOR", "8"),
                          "MERLIN_MUON_GSIM_L2_CYCLE_FACTOR")
        minimum = 120_000
        floor_factor = 2
    scaled = l2_cycles * factor
    compute_floor = None
    if isinstance(flops, int) and flops > 0:
        try:
            peak = int(muon._rtl_machine_capacity(target)["peak_flops_per_cycle"])
            if peak > 0:
                compute_floor = (flops + peak - 1) // peak
                scaled = max(scaled, compute_floor * floor_factor)
        except Exception:  # noqa: BLE001 -- the sealed L2 observation remains a valid bounded basis
            compute_floor = None
    value = max(minimum, scaled)
    return value, {
        "source": ("sealed_l2_cycles" if sealed else "manual_diagnostic_l2_cycles"),
        "max_cycles": value,
        "l2_cycles": l2_cycles,
        "l2_multiplier": factor,
        "minimum_cycles": minimum,
        "compute_floor_multiplier": floor_factor,
        "compute_floor_cycles": compute_floor,
        "performance_measurement": False,
    }


def _compact_numeric_from_console(console: str, *, expected_elements: int) -> dict[str, Any]:
    """Parse the trusted Cyclotron harness's explicit compact verdict.

    A clean process exit is not evidence: require exactly one internally
    consistent verdict plus the ordinary DONE marker.  This keeps a timeout,
    truncated UART stream, or a kernel that never reached validation from
    becoming a pass merely because no full output tensor was printed.
    """
    lines = [line.strip() for line in console.splitlines()]
    markers = [line for line in lines if line.startswith("MERLIN_NUMERIC ")]
    if len(markers) != 1 or lines.count("DONE") != 1:
        raise muon.MuonError(
            "Cyclotron compact numeric validation requires exactly one MERLIN_NUMERIC verdict "
            "and a DONE marker")
    fields = markers[0].split()
    if len(fields) != 4 or fields[1] not in ("PASS", "FAIL"):
        raise muon.MuonError(f"malformed Cyclotron compact numeric verdict: {markers[0]!r}")
    try:
        checked, mismatches = int(fields[2]), int(fields[3])
    except ValueError as exc:
        raise muon.MuonError(f"non-integer Cyclotron compact numeric counts: {markers[0]!r}") from exc
    if (expected_elements <= 0 or checked != expected_elements
            or mismatches < 0 or mismatches > checked):
        raise muon.MuonError(f"invalid Cyclotron compact numeric counts: {markers[0]!r}")
    status = fields[1].lower()
    if (status == "pass") != (mismatches == 0):
        raise muon.MuonError(f"inconsistent Cyclotron compact numeric verdict: {markers[0]!r}")
    return {
        "status": status,
        "elements_checked": checked,
        "mismatch_count": mismatches,
        "witness": "trusted_muon_post_kernel_comparator",
    }


def _cyclotron_host_dump_plan(
    elf: str | Path, outputs: list[Any], workdir: str | Path,
) -> tuple[tuple[int, int, Path], dict[str, Any]]:
    """Seal one evaluator-owned output range without embedding an answer in the ELF.

    Cyclotron currently exposes one contiguous GMEM dump per invocation, so this
    transport deliberately supports exactly one output. The ordinary capsule
    grader still performs the declared comparison over every element of it.
    """
    from . import muon_result_page as _rp

    if len(outputs) != 1:
        raise muon.MuonUnavailable(
            "Cyclotron evaluator-owned GMEM readback currently requires exactly one output")
    out = outputs[0]
    if out.dtype not in ("f32", "i32"):
        raise muon.MuonUnavailable(
            f"Cyclotron evaluator-owned GMEM readback does not support dtype {out.dtype!r}")
    elements = int(out.rows) * int(out.cols)
    byte_length = elements * 4
    symbol = f"_out_{out.name}"
    try:
        layout = _rp.symbol_layouts(elf, (symbol,))[symbol]
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise muon.MuonUnavailable(
            f"could not bind Cyclotron output dump to ELF symbol {symbol!r}: {exc}") from exc
    address = int(layout["address"])
    symbol_size = int(layout["size"])
    if symbol_size != byte_length:
        raise muon.MuonUnavailable(
            f"ELF output symbol {symbol!r} is {symbol_size} bytes, expected {byte_length}")
    if address < 0 or address + byte_length > (1 << 32):
        raise muon.MuonUnavailable(
            f"ELF output symbol {symbol!r} is outside Cyclotron's 32-bit GMEM range")

    elf_path = Path(elf).resolve()
    dump_path = Path(workdir).resolve() / "cyclotron.output.bin"
    digest = hashlib.sha256(elf_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "merlin.cyclotron-host-gmem-output.v1",
        "elf_sha256": digest,
        "transport": "evaluator_owned_gmem_dump",
        "output": {
            "name": out.name,
            "dtype": out.dtype,
            "rows": int(out.rows),
            "cols": int(out.cols),
            "elements": elements,
            "symbol": symbol,
            "address": address,
            "byte_length": byte_length,
            "symbol_size": symbol_size,
            "dump_file": dump_path.name,
        },
    }
    (Path(workdir) / "cyclotron.host_dump.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (address, byte_length, dump_path), manifest


def _cyclotron_host_dump_outputs(
    console: str, dump_path: Path, manifest: dict[str, Any],
) -> dict[str, list]:
    """Decode a complete post-exit dump; malformed transport evidence fails closed."""
    if [line.strip() for line in console.splitlines()].count("DONE") != 1:
        raise muon.MuonError(
            "Cyclotron host readback requires exactly one device DONE completion marker")
    spec = manifest["output"]
    expected_bytes = int(spec["byte_length"])
    try:
        raw = dump_path.read_bytes()
    except OSError as exc:
        raise muon.MuonUnavailable(
            "Cyclotron completed but its evaluator-owned output dump is unavailable") from exc
    if len(raw) != expected_bytes:
        raise muon.MuonUnavailable(
            "Cyclotron evaluator-owned output dump has the wrong size: "
            f"expected {expected_bytes} bytes, got {len(raw)}")
    elements = int(spec["elements"])
    fmt = "f" if spec["dtype"] == "f32" else "i"
    values = list(struct.unpack(f"<{elements}{fmt}", raw))
    cols = int(spec["cols"])
    return {str(spec["name"]): [values[i:i + cols] for i in range(0, elements, cols)]}


def _gsim_host_dump_plan(
    elf: str | Path, outputs: list[Any], workdir: str | Path, *, soc_offset: int,
) -> tuple[tuple[int, int, Path], dict[str, Any]]:
    """Bind one answer-free output buffer to GSIM's physical DRAM aperture.

    Symbol discovery happens on the submitted rv32 ELF.  GSIM loads the fused
    rv64 image, so the only address conversion is the explicit SoC fuse offset.
    The dump path is evaluator-owned and fixed directly inside ``workdir``.
    """
    from . import muon_result_page as _rp

    if len(outputs) != 1:
        raise muon.MuonUnavailable(
            "GSIM evaluator-owned GMEM readback currently requires exactly one output")
    out = outputs[0]
    if out.dtype not in ("f32", "i32"):
        raise muon.MuonUnavailable(
            f"GSIM evaluator-owned GMEM readback does not support dtype {out.dtype!r}")
    if not isinstance(soc_offset, int) or isinstance(soc_offset, bool) or soc_offset < 0:
        raise muon.MuonUnavailable(f"invalid GSIM SoC fuse offset {soc_offset!r}")
    elements = int(out.rows) * int(out.cols)
    byte_length = elements * 4
    symbol = f"_out_{out.name}"
    try:
        layout = _rp.symbol_layouts(elf, (symbol,))[symbol]
    except (KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise muon.MuonUnavailable(
            f"could not bind GSIM output dump to ELF symbol {symbol!r}: {exc}") from exc
    muon_address = int(layout["address"])
    symbol_size = int(layout["size"])
    if symbol_size != byte_length:
        raise muon.MuonUnavailable(
            f"ELF output symbol {symbol!r} is {symbol_size} bytes, expected {byte_length}")
    soc_address = soc_offset + muon_address
    if muon_address < 0 or soc_address < 0 or soc_address + byte_length > (1 << 64):
        raise muon.MuonUnavailable(
            f"ELF output symbol {symbol!r} is outside GSIM's physical address range")

    elf_path = Path(elf).resolve()
    dump_path = Path(workdir).resolve() / "gsim.output.bin"
    manifest = {
        "schema": "merlin.gsim-host-gmem-output.v1",
        "elf_sha256": hashlib.sha256(elf_path.read_bytes()).hexdigest(),
        "soc_fuse_offset": soc_offset,
        "transport": "evaluator_owned_gmem_dump",
        "output": {
            "name": out.name,
            "dtype": out.dtype,
            "rows": int(out.rows),
            "cols": int(out.cols),
            "elements": elements,
            "symbol": symbol,
            "muon_address": muon_address,
            "soc_address": soc_address,
            "byte_length": byte_length,
            "symbol_size": symbol_size,
            "dump_file": dump_path.name,
        },
    }
    (Path(workdir) / "gsim.host_dump.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (soc_address, byte_length, dump_path), manifest


def _gsim_host_dump_outputs(
    console: str, dump_path: Path, manifest: dict[str, Any], *, workdir: str | Path,
) -> dict[str, list]:
    """Decode GSIM's exact binary dump after an explicit RTL completion witness."""
    resolved_workdir = Path(workdir).resolve()
    resolved_dump = Path(dump_path).resolve()
    if resolved_dump.parent != resolved_workdir or resolved_dump.name != "gsim.output.bin":
        raise muon.MuonUnavailable(
            "GSIM output dump must be directly inside the GSIM work directory")
    lines = [line.strip() for line in console.splitlines()]
    finish = [line for line in lines if line.startswith("[gsim-emu] FINISHED:")]
    if len(set(finish)) != 1 or "model_finished=1" not in finish[0]:
        raise muon.MuonError(
            "GSIM host readback requires one RTL model-completion witness")
    spec = manifest["output"]
    expected_bytes = int(spec["byte_length"])
    marker = f"[gsim-emu] BINARY_DUMP complete bytes={expected_bytes}"
    if lines.count(marker) != 1:
        raise muon.MuonUnavailable(
            "GSIM host readback requires one size-bound binary-dump completion marker")
    try:
        raw = resolved_dump.read_bytes()
    except OSError as exc:
        raise muon.MuonUnavailable(
            "GSIM completed but its evaluator-owned output dump is unavailable") from exc
    if len(raw) != expected_bytes:
        raise muon.MuonUnavailable(
            "GSIM evaluator-owned output dump has the wrong size: "
            f"expected {expected_bytes} bytes, got {len(raw)}")
    elements = int(spec["elements"])
    fmt = "f" if spec["dtype"] == "f32" else "i"
    values = list(struct.unpack(f"<{elements}{fmt}", raw))
    cols = int(spec["cols"])
    return {str(spec["name"]): [values[i:i + cols] for i in range(0, elements, cols)]}


def _gsim_cycles_from_console(console: str) -> int | None:
    marker = "[gsim-emu] FINISHED: cycles="
    occurrences = [line for line in console.splitlines() if line.startswith(marker)]
    # ``_read_console`` intentionally prepends preserved marker lines before
    # the tail window; a short log can therefore contain the same physical
    # line twice. Accept one unique witness, never two differing witnesses.
    unique = set(occurrences)
    if len(unique) != 1:
        return None
    token = unique.pop()[len(marker):].split(maxsplit=1)[0]
    return int(token) if token.isdigit() else None


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
        from . import muon_mx_abi as _mxabi
        from . import muon_mx_codegen as _mx
        target = cb.get("target", "radiance")
        expected = cb.get("_oracle_expected_outputs")
        numeric_policy = cb.get("_oracle_numeric_policy")
        # Large Cyclotron outputs use evaluator-owned post-exit GMEM readback:
        # no expected values or tolerance bounds enter the device ELF. This is
        # still explicit opt-in for the frozen/non-adversarial derived cohort;
        # it is a faster result transport, not a broader anti-cheat claim.
        compact_opt_in = os.environ.get(
            "MERLIN_MUON_TRUSTED_COMPACT_NUMERIC", "").strip().lower() in (
                "1", "true", "yes", "on")
        host_dump_numeric = (compact_opt_in and simulator == "cyclotron"
                             and isinstance(expected, dict) and bool(expected)
                             and muon.is_mlir_artifact(kernel_src))
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
        native_mx = _mxabi.is_native_mx_cb(cb)
        if native_mx or cb.get("mx_operands"):
            # These paths own full programs rather than the external-pointer
            # harness whose named output buffer the dump ABI binds.
            host_dump_numeric = False
        if native_mx:
            # The native compiler program is the subject under test.  Validate its ABI binding before
            # even considering the legacy golden-backed substitution below; mixed provenance is an error.
            _mxabi.bind_native_program(cb, kernel_src)
        elif cb.get("mx_operands"):
            try:
                _mxprog = _mx.emit_mx_kernel(cb["mx_operands"], _mx.mx_output_name(cb))
            except Exception as _mxe:  # noqa: BLE001 — emitter fails closed (e.g. fp6/fp4 flash)
                # Do NOT swallow this silently: a fall-through here grades the submission on operands it
                # cannot use, which is the failure this whole branch exists to prevent. Record why.
                _mxprog = None
                _mx_refusal = f"{type(_mxe).__name__}: {_mxe}"
        if native_mx:
            elf, toolchain = muon.compile_for_oracle(kernel_src, workdir, target=target)
        elif _mxprog is not None:
            _elf, _tc = muon.compile_for_oracle(_mxprog, workdir, target=target)
            elf, toolchain = _elf, f"mx-reference-kernel(not-the-submission;{_tc})"
        elif muon.is_mlir_artifact(kernel_src):
            # THESIS PATH: the agent emitted an LLVM-dialect MLIR kernel (a compiler lowering). Build it
            # fork-free (stock LLVM rv32 + RTL-derived Muon re-encode + runner-owned external-kernel harness);
            # this path is fork-free by construction (never clang-muon), so the toolchain stamp is "fork-free".
            elf, toolchain = muon.compile_mlir_forkfree(
                kernel_src, cb, workdir, target=target,
                host_dump_outputs=host_dump_numeric), "fork-free"
        else:
            program = _mh.program_from_cb(cb, kernel_src, muon._model_for(target)) or kernel_src
            elf, toolchain = muon.compile_for_oracle(program, workdir, target=target)
        dump_spec = None
        dump_manifest = None
        if host_dump_numeric:
            operands = _mh.args_from_cb(cb)
            if operands is None:
                raise muon.MuonUnavailable(
                    "could not derive declared outputs for Cyclotron evaluator-owned readback")
            _in_args, out_args = operands
            dump_spec, dump_manifest = _cyclotron_host_dump_plan(elf, out_args, workdir)
        t1 = time.perf_counter()
        engine_binding = None
        engine_binding_path = Path(workdir) / "cyclotron_engine_binding.json"
        if simulator == "cyclotron":
            # Capture immediately around the measured invocation.  A prior failed/replayed run must not
            # leave reusable provenance, and changing either executable or timing config while the
            # process runs makes this measurement unsealable rather than ambiguously attributed.
            engine_binding_path.unlink(missing_ok=True)
            from merlin.targetgen.evaluation_cohort import cyclotron_l2_engine_binding
            engine_binding = cyclotron_l2_engine_binding(str(target))
        console, cycles, summary = muon.run_elf(
            elf, simulator=simulator, timeout=timeout, target=target,
            gmem_dump=dump_spec)
        if simulator == "cyclotron":
            after_binding = cyclotron_l2_engine_binding(str(target))
            if after_binding != engine_binding:
                raise muon.MuonUnavailable(
                    "Cyclotron executable/config identity changed during the measured L2 invocation")
            engine_binding_path.write_text(
                json.dumps(engine_binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        t2 = time.perf_counter()
        completion_only = False
        try:
            if dump_spec is not None and dump_manifest is not None:
                outputs = _cyclotron_host_dump_outputs(console, dump_spec[2], dump_manifest)
                raw = {"cycles": cycles} if cycles is not None else {}
            else:
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
            "gflops": muon.gflops(flops, cycles, target=target),
            "pct_fp_peak": muon.pct_fp_peak(flops, cycles, target=target),
            "summary": summary,
        }
        if completion_only:
            result["completion_only"] = True
        if dump_manifest is not None:
            result["host_gmem_dump"] = {
                **dump_manifest,
                "trust_scope": "frozen_non_adversarial_derived_evaluation_only",
            }
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
    runs the prebuilt GSIM emulator instead of the Verilator sim. When the grading runner supplies its
    private post-submission oracle, a runner-owned Muon harness streams output through a fixed 32-word
    mailbox and a trusted Rocket carrier compares it behind a READY(sequence,count)/ACK handshake. The
    private answer is present only in that carrier, never in the submitted rv32 ELF or its cache key.

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
        from . import muon_result_page as _rp
        t0 = time.perf_counter()
        expected = cb.get("_oracle_expected_outputs")
        numeric_policy = cb.get("_oracle_numeric_policy")
        numeric_readback = isinstance(expected, dict) and bool(expected)
        compact_opt_in = os.environ.get(
            "MERLIN_MUON_TRUSTED_COMPACT_NUMERIC", "").strip().lower() in (
                "1", "true", "yes", "on")
        host_dump_numeric = compact_opt_in and numeric_readback and muon.is_mlir_artifact(kernel_src)
        derived = _mh.args_from_cb(cb) if numeric_readback else None
        if numeric_readback and derived is None:
            raise muon.MuonUnavailable(
                "GSIM numeric readback requested but the harness cannot derive declared outputs")
        if compact_opt_in and numeric_readback and not host_dump_numeric:
            raise muon.MuonUnavailable(
                "trusted GSIM host readback requires an instrumentable MLIR artifact")
        # Build the graded ELF identically to the Verilator/cyclotron adapters (fork-free thesis path when
        # the artifact is LLVM-dialect MLIR; otherwise the runner-owned harness + oracle compile).
        if muon.is_mlir_artifact(kernel_src):
            elf, toolchain = muon.compile_mlir_forkfree(
                kernel_src, cb, workdir, target=target,
                result_page=numeric_readback and not host_dump_numeric,
                host_dump_outputs=host_dump_numeric,
                host_dump_done_marker=not host_dump_numeric), "fork-free"
        else:
            program = _mh.program_from_cb(
                cb, kernel_src, muon._model_for(target), result_page=numeric_readback)
            if numeric_readback and program is None:
                raise muon.MuonUnavailable(
                    "GSIM numeric readback cannot instrument an artifact that owns its main program")
            program = program or kernel_src
            elf, toolchain = muon.compile_for_oracle(program, workdir, target=target)
        result_manifest = outcome_symbols = None
        dump_spec = dump_manifest = None
        if host_dump_numeric:
            dump_spec, dump_manifest = _gsim_host_dump_plan(
                elf, derived[1], workdir, soc_offset=muon.soc_fuse_offset())
            soc = muon.fuse_soc_elf(Path(elf), Path(workdir))
        elif numeric_readback:
            result_manifest = _rp.manifest_from_elf(
                elf, derived[1], soc_offset=muon.soc_fuse_offset())
            import json
            manifest_path = Path(workdir) / "result_page.json"
            manifest_path.write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
            carrier = Path(workdir) / "result_carrier.c"
            elements = sum(int(spec["elements"]) for spec in result_manifest["outputs"])
            carrier_source = _rp.render_carrier(result_manifest, expected, numeric_policy)
            carrier.write_text(carrier_source, encoding="utf-8")
            soc = muon.fuse_soc_elf(Path(elf), Path(workdir), carrier_source=carrier)
            outcome_symbols = _rp.symbol_addresses(
                soc, (_rp.PASS_SYMBOL, _rp.FAIL_SYMBOL))
        else:
            soc = muon.fuse_soc_elf(Path(elf), Path(workdir))
        t1 = time.perf_counter()
        maxcyc, cycle_budget = _gsim_cycle_budget(cb, target=target, flops=flops)

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
        run_env = None
        if dump_spec is not None:
            address, byte_length, dump_path = dump_spec
            resolved_workdir = Path(workdir).resolve()
            resolved_dump = Path(dump_path).resolve()
            if resolved_dump.parent != resolved_workdir or resolved_dump.name != "gsim.output.bin":
                raise muon.MuonUnavailable(
                    "GSIM output dump must be directly inside the GSIM work directory")
            resolved_dump.unlink(missing_ok=True)
            run_env = os.environ.copy()
            # Override rather than inherit these evaluator-owned controls. A
            # caller's environment must not redirect or resize the oracle dump.
            run_env.update({
                "GSIM_STOP_ON_FINISH": "1",
                "GSIM_DUMP_FILE": str(resolved_dump),
                "GSIM_DUMP_ADDR": f"0x{address:x}",
                "GSIM_DUMP_LOCAL_ADDR": f"0x{dump_manifest['output']['muon_address']:x}",
                "GSIM_DUMP_LEN": str(byte_length),
            })
        try:
            with log.open("wb") as fh:
                completed = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                           preexec_fn=_unlimited_stack, env=run_env)
        except subprocess.TimeoutExpired as e:
            raise muon.MuonUnavailable(f"GSIM emu wall-timed out after {timeout}s") from e
        t2 = time.perf_counter()
        console, console_bytes, console_truncated = muon._read_console(log)
        if completed.returncode != 0:
            raise muon.MuonUnavailable(
                f"GSIM emu exited nonzero ({completed.returncode}); refusing completion markers from "
                f"a failed process. tail:\n{console[-600:]}")
        if dump_spec is not None and dump_manifest is not None:
            outputs = _gsim_host_dump_outputs(
                console, dump_spec[2], dump_manifest, workdir=workdir)
            cycles = _gsim_cycles_from_console(console)
            if cycles is None:
                raise muon.MuonUnavailable(
                    "GSIM binary dump completed without one parseable cycle witness")
            return {
                "outputs": outputs,
                "cycles": cycles,
                "oracle": {"kind": "rtl_gsim_muon_numeric", "derived_from_rtl": True,
                           "fidelity": "elaborated_rtl"},
                "console": console,
                "console_spool": {"path": str(log), "bytes_on_disk": console_bytes,
                                  "truncated": console_truncated,
                                  "markers_preserved": list(muon._GSIM_MARKERS)},
                "toolchain": toolchain,
                "timing": _timing(t1 - t0, t2 - t1),
                "gflops": muon.gflops(flops, cycles, target=target),
                "pct_fp_peak": muon.pct_fp_peak(flops, cycles, target=target),
                "host_gmem_dump": {
                    **dump_manifest,
                    "trust_scope": "frozen_non_adversarial_derived_evaluation_only",
                },
                "bounded_observation": cycle_budget,
            }
        if numeric_readback:
            outcome = _rp.outcome_from_console(console, outcome_symbols)
            if outcome is None:
                raise muon.MuonUnavailable(
                    "GSIM numeric result mailbox was instrumented but Rocket's final PC reached neither "
                    f"{_rp.PASS_SYMBOL} nor {_rp.FAIL_SYMBOL}; the cycle cap is not a numeric verdict. "
                    f"tail:\n{console[-600:]}")
            elements = sum(int(spec["elements"]) for spec in result_manifest["outputs"])
            return {
                "outputs": {},
                "cycles": None,
                "oracle": {"kind": "rtl_gsim_muon_numeric", "derived_from_rtl": True,
                           "fidelity": "elaborated_rtl"},
                "console": console,
                "console_spool": {"path": str(log), "bytes_on_disk": console_bytes,
                                  "truncated": console_truncated,
                                  "markers_preserved": list(muon._GSIM_MARKERS)},
                "toolchain": toolchain,
                "timing": _timing(t1 - t0, t2 - t1),
                "gflops": None,
                "pct_fp_peak": None,
                "numeric_verdict": {
                    "status": outcome,
                    "elements_checked": elements,
                    "policy": dict(numeric_policy or {}),
                    "witness": "final_rocket_pc",
                    "pass_symbol": f"0x{outcome_symbols[_rp.PASS_SYMBOL]:x}",
                    "fail_symbol": f"0x{outcome_symbols[_rp.FAIL_SYMBOL]:x}",
                },
                "result_page": result_manifest,
                "bounded_observation": cycle_budget,
            }
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
            "gflops": muon.gflops(flops, cycles, target=target),
            "pct_fp_peak": muon.pct_fp_peak(flops, cycles, target=target),
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
