"""Generic ``external_backend`` PROGRAM oracle — run an agent-emitted target-ISA program on the target's
own cosim, using the target's OWN assembler. The counterpart to the ``command_buffer`` oracle
(``mlc_bridge.arc_run_command_buffer``, RoCC/OPU) for accelerators that are a self-hosted ISA core
(their deliverable is an assembled program, e.g. a ``kernel.S`` → IMEM words), NOT a RoCC ``.insn``
host stream and NOT an ISA-less command buffer.

Design (HW-agnostic; nothing target-specific is hardcoded here):
  * the agent's emitted ``kernel.S`` is a stream of ``.word``/``.insn`` directives (the target's encoded
    instructions, grounded on the target's shipped ISA definition) — assembled to IMEM words by the
    PREBUILT stock LLVM (``llvm-mc`` + ``llvm-objcopy`` from the MLIR install), NOT by a per-target
    bespoke assembler and NOT by a forked toolchain. merlin holds no opcode table; the encoding lives in
    the emitted ``.word`` directives, and stock ``.insn`` gives the generic RISC-V-shaped formats. This
    is the target-agnostic assembly path: any HW whose instructions fit ``.word``/``.insn`` assembles the
    same way (see AW3);
  * the input tensors' DRAM byte layout comes from the selected support package's declared emitter,
    run in the model's own venv as a subprocess) — its torch owns the fp8/bf16 dtype encodings merlin's
    venv lacks. A self-contained ``program`` (validation) still assembles in the model venv (it owns the
    ``Program`` classes);
  * the cosim comes from mlc, resolved by target (``artifact_paths`` + the per-target program backend);
  * DRAM bases + output physical-layout come from the command buffer's tensors (declared by the emitting
    backend, i.e. the generation process), never from constants here.

The adapter matches the oracle contract used by :mod:`capsule_runner` — ``run(cb, fourth_text, workdir,
timeout) -> {"outputs", "cycles", "oracle"}`` — and raises :class:`OracleUnavailable` when the model
venv / cosim is absent (so ``not_run_is_not_pass`` fails closed, never a false green).
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.analysis.perf.measurement import cycle_trace as _CYCLE_TRACE
from merlin.analysis.perf.measurement import observations as _OBS_KEYS
from merlin.common import external_tools as modelir
from merlin.common.paths import data_path, ext_path
from merlin.target.simulators.program_engines import (
    _IEEE_NATIVE,
    _LINEAGE_ENGINE,
    _RTL_ENGINES,
    _WRAPPER_FLAVOUR,
    OracleUnavailable,
    _decode_elements,
    _decode_output,
    _dtype_key,
    _element_bytes,
    _model_venv_python,
    _out_nbytes,
    _output_name,
    _resolve_out_spec,
    _resolve_out_specs,
    _rtl_engine_dir,
    _rtl_engine_probe,
    select_rtl_engine,
)

# ---- program_oracle ----------------------------------------------------------------------------


def _require_complete_functional_decode(target: str, result: dict[str, Any]) -> None:
    """Refuse model output when any submitted instruction was replaced during decode.

    Both grading and the agent-facing debugger consume this runner. A debugger
    trace from a program with substituted instructions is as misleading as a
    passing grade from that program, even when the model reports a halt.
    """
    unsupported = result.get("unsupported")
    if not isinstance(unsupported, list):
        raise OracleUnavailable(f"{target}: functional runner did not report instruction decode coverage")
    if unsupported:
        raise OracleUnavailable(
            f"{target}: functional runner substituted {len(unsupported)} unsupported instruction(s); "
            f"first: {unsupported[0]!r}"
        )


def run_program_functional_oracle(
    target: str,
    *,
    model_ext: str,
    cb: dict | None = None,
    kernel_s: Path | None = None,
    program: str | None = None,
    inputs: list[dict] | None = None,
    max_cycles: int = 20000,
    workdir,
    timeout: int = 600,
) -> dict[str, Any]:
    """Assemble (model venv / stock LLVM) → run on the target's FAST FUNCTIONAL core → read back. Mirrors
    :func:`run_program_oracle` (same ``emit_bundle`` words+preload, same output-layout resolution) but the
    ONLY difference is the runner: the target's mlc functional program runner (derived, invoked by file
    path in the model venv), instead of the cycle-exact arc cosim. Returns the SAME result shape
    ``{"outputs": {name: list}, "cycles": int, "oracle": f"{target}-functional"}``. Raises
    :class:`OracleUnavailable` if ``mlc_dir()`` / the func file / the model venv is absent (fail closed)."""
    from merlin.development.targetgen.oracles.program_oracle import (
        ProgramDidNotHalt,
        _bundle_preload,
        _func_program_helper,
        _run_func_helper,
        _sim_timing,
        emit_bundle,
    )

    _func_program_helper(target)  # fail closed early if the runner is absent

    bundle = emit_bundle(
        target=target,
        model_ext=model_ext,
        program=program,
        kernel_s=kernel_s,
        inputs=inputs,
        workdir=workdir,
        timeout=timeout,
    )
    words = bundle["words"]
    preload = _bundle_preload(bundle, cb)
    # EVERY declared output, exactly as the cosim path captures them (a module with two commit ops has
    # two results; reading only the first reports the second as never written however correct the kernel
    # is). The runner already takes a LIST of out_bases and keys its result by str(base).
    specs = _resolve_out_specs(target, cb, bundle)

    # The emitted kernel + the cb tensor bases address DRAM at the target's real region base (its ISA
    # memory map), but the functional model's DRAM aperture is 0-based. Derive that base from the target's
    # facts (green-card memory map; 0 when none) and hand it to the runner, which relocates every DRAM
    # index by it — so preload, the kernel's internal stores, and readback all land in the aperture. The
    # cycle-exact cosim (L3) needs no such field: its TileLinkSlave masks the address into its window.
    # A self-contained ``program`` carries its OWN (model-native, 0-based) layout, so it is never
    # relocated — only the capsule/agent kernel path, which addresses the target's real DRAM region, is.
    from merlin.target.isa.machine_model import dram_base_for

    reloc_base = 0 if program else int(dram_base_for(target))
    req = {
        "words": [int(w) & 0xFFFFFFFF for w in words],
        "preload": [{"base": int(b), "b64": base64.b64encode(d).decode()} for b, d in preload],
        "out_bases": [[int(s["base"]), _out_nbytes(s)] for s in specs.values()],
        "max_cycles": int(max_cycles),
        "dram_base": reloc_base,
    }
    _t0 = time.monotonic()
    res = _run_func_helper(target, model_ext, req, Path(workdir), timeout)
    _require_complete_functional_decode(target, res)
    if not res.get("halted"):
        raise ProgramDidNotHalt(f"{target} program did not halt within {max_cycles} instructions (functional)")

    outmap = res.get("outputs") or {}
    outputs = {}
    for name, spec in specs.items():
        key = str(int(spec["base"]))  # the CLI keys outputs by str(base)
        if key not in outmap:
            raise OracleUnavailable(f"{target}: functional runner returned no output {name!r} at base {spec['base']}")
        outputs[name] = _decode_output(
            base64.b64decode(outmap[key]), spec["shape"], spec["dtype"], spec["physical"], name=name
        ).tolist()
    return {
        "outputs": outputs,
        "cycles": int(res.get("cycles") or 0),
        "timing": _sim_timing(time.monotonic() - _t0),
        "oracle": {"kind": f"{target}-functional", "derived_from_rtl": False, "fidelity": "functional_model"},
    }


def program_functional_adapter(target: str, *, model_ext: str) -> Callable:
    """The FAST functional oracle adapter (``run(cb, fourth_text, workdir, timeout)`` shape) for an
    ``external_backend`` target — the per-round QA-loop correctness tier that mirrors
    :func:`program_oracle_adapter` but runs the emitted kernel on the functional core, not the cosim."""

    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        return run_program_functional_oracle(
            target, model_ext=model_ext, cb=cb, kernel_s=ks, workdir=wd, timeout=timeout
        )

    return run


_DEBUG_MAX_DUMP_BYTES = 4096  # per-request cap on total dumped DRAM (anti-scrape)


def build_debug_cb(target: str, capsule_dir) -> dict:
    """A minimal command buffer (``tensors``: role/shape/dtype/base + input ``preload_b64``) for ONE
    capsule, built straight from the capsule spec — WITHOUT building the agent's package. Enough to run the
    agent's ``kernel.S`` on the functional model for debugging. Target-agnostic: bases from
    :func:`capsule_dram.layout`, preload from :func:`capsule_golden.canonical_input_raws`. The output tensor
    carries base/shape/dtype (used only to compute the redaction region) but NOT ``physical`` — the debugger
    never decodes the output, so the agent-declared output layout is irrelevant here."""
    from merlin.development.targetgen.corpus import capsule_io as capsule_common
    from merlin.development.targetgen.grading import (
        golden_inputs as capsule_dram,
    )
    from merlin.development.targetgen.grading import (
        golden_inputs as capsule_inputs,
    )
    from merlin.target.isa.machine_model import dram_base_for

    cap = capsule_common.load_capsule(capsule_dir)
    tensors: dict[str, dict] = {}
    for t in cap.get("inputs") or []:
        if t.get("role") == "output":
            continue  # the output tensor is added below (canonical)
        tensors[t["name"]] = {
            "role": t.get("role", "input"),
            "shape": list(t.get("shape") or []),
            "dtype": t.get("dtype"),
        }
    ot = capsule_dram.output_tensor(cap)
    if ot is None:
        raise OracleUnavailable(f"{target}: capsule {capsule_dir} declares no resolvable output tensor")
    tensors[ot["name"]] = {"role": "output", "shape": list(ot["shape"]), "dtype": ot["dtype"]}
    cb = {"tensors": tensors}
    capsule_dram.inject_bases(cb, cap, base=dram_base_for(target) + capsule_dram.DEFAULT_BASE)
    raws = capsule_inputs.selected_canonical_input_raws(cap, capsule_dir)
    for name, t in tensors.items():
        if t["role"] in ("input", "weight", "bias") and name in raws:
            t["preload_b64"] = base64.b64encode(raws[name]).decode()
    return cb


def _split_dump_regions(dump_regions, out_base: int, out_nbytes: int):
    """(allowed, rejected) split of requested ``(base, nbytes)`` windows: reject any that overlap the output
    region (withheld — see the section note) or overrun the per-request byte cap; keep the rest in order."""
    allowed: list[tuple[int, int]] = []
    rejected: list[dict] = []
    o0, o1 = int(out_base), int(out_base) + int(out_nbytes)
    total = 0
    for r in dump_regions or []:
        b, n = int(r[0]), int(r[1])
        if n <= 0:
            rejected.append({"base": b, "nbytes": n, "reason": "non-positive length"})
        elif b < o1 and o0 < b + n:
            rejected.append(
                {
                    "base": b,
                    "nbytes": n,
                    "reason": "overlaps the OUTPUT region — withheld (it would hold your result / "
                    "the answer). Debug the INPUT and scratch regions instead.",
                }
            )
        elif total + n > _DEBUG_MAX_DUMP_BYTES:
            rejected.append(
                {"base": b, "nbytes": n, "reason": f"exceeds the per-request dump cap ({_DEBUG_MAX_DUMP_BYTES} B)"}
            )
        else:
            total += n
            allowed.append((b, n))
    return allowed, rejected


def run_program_debug(
    target: str,
    *,
    model_ext: str,
    cb: dict,
    kernel_s: Path,
    dump_regions,
    run_to: int | None = None,
    state_summary: bool = False,
    max_cycles: int = 20000,
    workdir,
    timeout: int = 600,
) -> dict[str, Any]:
    """Assemble the agent's ``kernel.S`` and run it on the target's FUNCTIONAL core to instruction ``run_to``
    (or to halt when None), then return committed scalar state + REDACTED DRAM windows. Same assemble +
    preload + ``dram_base`` relocation as :func:`run_program_functional_oracle`; the output region is never
    read back (only used to reject overlapping dump windows). Raises :class:`OracleUnavailable` when the
    model venv / functional runner is absent (fail closed)."""
    from merlin.development.targetgen.oracles.program_oracle import (
        _bundle_preload,
        _func_program_helper,
        _run_func_helper,
        emit_bundle,
    )

    _func_program_helper(target)  # fail closed early if the runner is absent
    bundle = emit_bundle(target=target, model_ext=model_ext, kernel_s=kernel_s, workdir=workdir, timeout=timeout)
    words = bundle["words"]
    preload = _bundle_preload(bundle, cb)
    out_spec = _resolve_out_spec(target, cb, bundle)
    out_nbytes = _out_nbytes(out_spec)
    from merlin.target.isa.machine_model import dram_base_for

    reloc_base = int(dram_base_for(target))
    allowed, rejected = _split_dump_regions(dump_regions, int(out_spec["base"]), out_nbytes)
    req = {
        "words": [int(w) & 0xFFFFFFFF for w in words],
        "preload": [{"base": int(b), "b64": base64.b64encode(d).decode()} for b, d in preload],
        "out_bases": [],  # the debugger never reads the output region
        "max_cycles": int(max_cycles),
        "dram_base": reloc_base,
        "run_to": None if run_to is None else int(run_to),
        "dump_regions": [[int(b), int(n)] for b, n in allowed],
        "state_summary": bool(state_summary),
    }
    res = _run_func_helper(target, model_ext, req, Path(workdir), timeout)
    _require_complete_functional_decode(target, res)
    dmap = res.get("dram_dumps") or {}
    regions = []
    for b, n in allowed:
        raw = base64.b64decode(dmap.get(str(int(b)), "")) if dmap.get(str(int(b))) else b""
        regions.append(
            {
                "base": int(b),
                "nbytes": int(n),
                "returned_bytes": len(raw),
                "hex": raw.hex(),
                "all_zero": (len(raw) > 0 and not any(raw)),
            }
        )
    return {
        "halted": bool(res.get("halted")),
        "halt_reason": res.get("halt_reason"),
        "instr_count": res.get("instr_count"),
        "cycles": res.get("cycles"),
        "pc": res.get("pc"),
        "regs": res.get("regs"),
        "program_words": len(words),
        "unsupported": res.get("unsupported"),
        "regions": regions,
        "rejected_regions": rejected,
        "on_chip": res.get("state_summary"),  # value-free populated-map (vmem/mrf/acc), or None
        "oracle": f"{target}-debug",
    }


_CB_DEBUG_REDACTED_KEYS = ("outputs", "correct", "reference", "elf", "console")


def _inject_canonical_inputs(target: str, cb: dict, capsule_dir) -> dict:
    """Return a COPY of ``cb`` with the capsule's canonical leaf operands attached as tensor ``data`` and
    resident handles resolved, so the mlc arc command-buffer runner can answer it. Target-agnostic: the
    leaves are the SAME deterministic operands the grader materializes (:func:`capsule_golden.
    materialize_capsule_leaves`) — the inputs the agent is given, never a golden. Resident-pack handles
    (a ``RES_PACK`` dst) are rewritten to their source tensor so the matmul operands name real tensors."""
    import copy

    import numpy as np

    from merlin.development.targetgen.corpus import capsule_io as capsule_common
    from merlin.development.targetgen.grading import golden_inputs as capsule_inputs

    cb = copy.deepcopy(cb)
    cap = capsule_common.load_capsule(capsule_dir)
    leaves = capsule_inputs.selected_materialize_capsule_leaves(cap)
    tensors = cb.get("tensors") or {}
    for name, t in tensors.items():
        if name in leaves and t.get("role") in ("input", "weight", "bias"):
            shape = t.get("shape") or list(leaves[name].shape)
            t["data"] = np.asarray(leaves[name].to_list(), dtype=np.int64).reshape(shape).tolist()
    # resolve RES_PACK handles (dst produced from a real src tensor) in the matmul operands
    handle = {
        c.get("operands", {}).get("dst"): c.get("operands", {}).get("src")
        for c in cb.get("commands", [])
        if c.get("opcode") == "RES_PACK"
    }
    for c in cb.get("commands", []):
        if c.get("opcode") in ("MATMUL", "MATMUL_RESIDENT"):
            for k, v in list(c.get("operands", {}).items()):
                if v in handle and handle[v] in tensors:
                    c["operands"][k] = handle[v]
    return cb


def run_command_buffer_debug(target: str, *, cb: dict, capsule_dir) -> dict[str, Any]:
    """Answer the agent's OWN command buffer on the target's RTL-derived mlc arc model and return the
    REDACTED per-op hardware-state effects. Target-agnostic (mlc infers the target/config from the cb).
    Returns ``{halted, metrics, raw_metrics, per_command, oracle, redacted, ...}`` — the output VALUES and
    the ``correct`` verdict are stripped (see the section note). Raises :class:`OracleUnavailable` when the
    mlc arc model for ``target`` is absent (fail closed)."""
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    if not _merlin_mlc_discovery_1.arc_available(target):
        raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")
    prepared = _inject_canonical_inputs(target, cb, capsule_dir)
    # arc_run_command_buffer calls require_mlc(); make mlc importable for just this call (same context the
    # program cosim uses — never a global sys.path insert).
    with modelir.modelir_importable(mlc_bridge.mlc_dir()):
        try:
            res = _merlin_mlc_discovery_1.arc_run_command_buffer(prepared, target)
        except Exception as e:  # noqa: BLE001 — a run fault is surfaced, never a fabricated pass
            raise OracleUnavailable(f"{target} arc command-buffer run failed: {type(e).__name__}: {e}")
    # REDACT the answer-bearing keys; keep only the RTL-derived effect counts + provenance.
    safe = {k: v for k, v in res.items() if k not in _CB_DEBUG_REDACTED_KEYS}
    n_out = len(res.get("outputs") or {})
    return {
        "halted": True,
        "metrics": safe.get("metrics"),
        "raw_metrics": safe.get("raw_metrics"),
        "per_command": safe.get("per_command"),  # per-op cycles + spad/acc/DRAM counts (no values)
        "oracle": safe.get("oracle"),
        "n_output_tensors": n_out,  # count only — the VALUES are withheld (answer key)
        "redacted": list(_CB_DEBUG_REDACTED_KEYS),
        "note": (
            "Per-op HARDWARE STATE for YOUR command buffer on the RTL-derived arc model (cycles + "
            "scratchpad/accumulator/DRAM-refill counts per command). The output VALUES and the pass/"
            "fail verdict are withheld (answer key). This runs your INTENDED computation; to inspect "
            "the emitted .insn ENCODING use the disassembler + instruction_trace.json."
        ),
    }


def _timing_block(res: Any) -> tuple[list[dict] | None, dict | None]:
    """``(timing_observations, timing_capability)`` an oracle result carries, VALIDATED -- or
    ``(None, None)`` when the oracle has no timing capability at all.

    The rules are :mod:`merlin.analysis.perf.measurement.observations`'; this only routes. Two things it does not do, both
    load-bearing: it never invents the block for an oracle that did not emit one (an oracle with no
    timing capability contributes nothing -- not the key, not zeros), and it never drops a block that
    was emitted but failed validation. A refused block still travels, as a capability record naming
    what was refused, because "the producer emitted a block we could not believe" is a fact about the
    instrument that has to be visible.
    """
    from merlin.analysis.perf.measurement import observations as _OBS

    raw = (
        res
        if isinstance(res, Mapping)
        else {
            k: getattr(res, k)
            for k in (
                _OBS.TIMING_OBSERVATIONS_KEY,
                _OBS.UNMEASURED_UNITS_KEY,
                _OBS.PARTITIONED_KEY,
                _OBS.ALIAS_COLLISIONS_KEY,
            )
            if hasattr(res, k)
        }
    )
    block = _OBS.validate_block(raw)
    if block is None:
        return None, None
    cap = block.to_dict()
    if isinstance(raw, Mapping) and raw.get("alias_wrapped_accesses") is not None:
        # Wrapping is not colliding: with one page in play every access wraps identically and nothing
        # is lost. Carried beside the collision count so the distinction survives.
        cap["alias_wrapped_accesses"] = raw["alias_wrapped_accesses"]
    if isinstance(raw, Mapping) and raw.get("unmeasured_note"):
        cap["unmeasured_note"] = raw["unmeasured_note"]
    return ([dict(o) for o in block.observations] or None), cap


def _load_rtl_runner(engine_dir: Path, filename: str):
    """Import an engine dir's conventional wrapper (``run_program`` symmetric with
    ``cosim_<target>.run_program``) BY PATH — target-agnostic (merlin never names the sim binary). Raises
    :class:`OracleUnavailable` if the wrapper is absent / exposes no ``run_program``."""
    wrapper = Path(engine_dir) / filename
    if not wrapper.is_file():
        raise OracleUnavailable(f"RTL runner absent: {wrapper}")
    spec = importlib.util.spec_from_file_location("_merlin_rtl_run", wrapper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_program"):
        raise OracleUnavailable(f"RTL runner {wrapper} exposes no run_program")
    return mod


def _load_verilator_runner(vsim_dir: Path):
    """Back-compat shim: the Verilator wrapper specifically."""
    return _load_rtl_runner(Path(vsim_dir), _RTL_ENGINES["verilator"][1])


def run_program_verilator_oracle(
    target: str,
    *,
    model_ext: str,
    vsim_dir,
    cb: dict | None = None,
    kernel_s: Path | None = None,
    program: str | None = None,
    inputs: list[dict] | None = None,
    max_cycles: int = 20000,
    workdir,
    timeout: int = 600,
    per_cycle_csv: str | Path | None = None,
    engine: str = "verilator",
) -> dict[str, Any]:
    """Assemble (model venv / stock LLVM) → run on the target's program-driven Verilator RTL sim → read
    back. Mirrors :func:`run_program_oracle` (same ``emit_bundle`` words+preload, same output-layout
    resolution) but the runner is the external Verilator ``run_program`` instead of the arc cosim. The
    sim's TileLink DRAM slave masks addresses into its window exactly like the arc cosim's
    ``TileLinkSlave`` (both mirror ``cosim_<target>``), so the real cb DRAM base is passed straight through
    as the read region — no functional-tier ``dram_base`` relocation. Returns the SAME result shape
    ``{"outputs": {name: list}, "cycles": int, "oracle": f"{target}-verilator-rtl"}``. Raises
    :class:`OracleUnavailable` if the vsim / wrapper is absent, or the program does not halt."""
    from merlin.development.targetgen.oracles.program_oracle import (
        ProgramDidNotHalt,
        _bundle_preload,
        _sim_timing,
        emit_bundle,
    )

    runner = _load_rtl_runner(Path(vsim_dir), _RTL_ENGINES[engine][1])
    bundle = emit_bundle(
        target=target,
        model_ext=model_ext,
        program=program,
        kernel_s=kernel_s,
        inputs=inputs,
        workdir=Path(workdir),
        timeout=timeout,
    )
    words = bundle["words"]
    preload = _bundle_preload(bundle, cb)
    # EVERY declared output, in declaration order — a module that commits twice has two results, and
    # reading only the first grades the second as never written whatever the kernel did (the same defect
    # the cosim path carried). The wrapper's ``reads`` is already a list of regions, so this needs no
    # change on the sim side; the returned regions come back in the order they were requested.
    specs = _resolve_out_specs(target, cb, bundle)
    _kw: dict[str, Any] = {}
    _derived_trace: Path | None = None
    _accepts_trace = False
    # OPTIONAL, and only if THIS runner accepts it. A runner built before the per-cycle dump
    # existed must keep working unchanged rather than raising a TypeError that would be recorded
    # as the submission's crash -- the same "a harness limit reported as the agent's bug" class.
    import inspect as _inspect

    try:
        _accepts_trace = "per_cycle_csv" in _inspect.signature(runner.run_program).parameters
    except (TypeError, ValueError):  # unintrospectable runner: skip the dump
        _accepts_trace = False
    if per_cycle_csv and _accepts_trace:
        Path(per_cycle_csv).parent.mkdir(parents=True, exist_ok=True)
        _kw["per_cycle_csv"] = str(per_cycle_csv)
    elif _accepts_trace and _CYCLE_TRACE.load_declaration(Path(vsim_dir)) is not None:
        # THE ENGINE DECLARED ITS ACTIVITY COLUMNS, so it HAS the timing capability -- it just
        # computes the decomposition outside the simulator instead of inside it. Ask for the trace
        # the reduction needs rather than reporting the engine as timing-blind, which is what made
        # adopting a faster engine silently cost the campaign its per-capsule occupancy. Written into
        # the run's own workdir and removed once folded, so this costs a bounded temporary file (one
        # row per cycle) and never a number the caller did not ask for.
        _derived_trace = Path(workdir) / "per_cycle_trace.csv"
        _derived_trace.parent.mkdir(parents=True, exist_ok=True)
        _kw["per_cycle_csv"] = str(_derived_trace)
    from merlin.target.simulators.engine_policy import simulation_runtime_slot

    slot = simulation_runtime_slot(engine, wait_timeout_s=timeout)
    with slot:
        _t0 = time.monotonic()
        res = runner.run_program(
            words,
            preload=preload,
            reads=[(int(s["base"]), _out_nbytes(s)) for s in specs.values()],
            max_cycles=max_cycles,
            timeout=timeout,
            **_kw,
        )
    if not res.get("halted"):
        raise ProgramDidNotHalt(f"{target} program did not halt within {max_cycles} cycles (verilator)")
    outs = res.get("outputs") or []
    if len(outs) < len(specs):
        raise OracleUnavailable(
            f"{target}: verilator oracle returned {len(outs)} region(s) for {len(specs)} declared "
            f"output(s) {sorted(specs)} — cannot read every result back"
        )
    outputs = {}
    for raw, (name, spec) in zip(outs, specs.items()):
        outputs[name] = _decode_output(bytes(raw), spec["shape"], spec["dtype"], spec["physical"], name=name).tolist()
    # The one tier here that runs the ELABORATED Verilog, so the only one entitled to say RTL.
    out: dict[str, Any] = {
        "outputs": outputs,
        "cycles": int(res.get("cycles") or 0),
        "timing": _sim_timing(time.monotonic() - _t0),
        "oracle": {
            "kind": f"{target}-{engine}-rtl",
            "derived_from_rtl": True,
            "fidelity": "elaborated_rtl",
            "engine": engine,
            "provenance": _engine_provenance(target, engine, Path(vsim_dir)),
        },
    }
    # WHATEVER FINER TIMING THIS ORACLE COULD SEE, carried instead of discarded. The run already paid
    # for it: the sim evaluated the RTL's own top-level activity ports on every cycle whether anyone
    # read them or not, so the marginal cost of keeping the decomposition is a load per cycle. A
    # runner with no timing capability sets nothing here and the result is byte-identical to before.
    _obs, _cap = _timing_block(res)
    if _obs is None:
        # The runner returned no block. If the engine DUMPED the same activity ports instead of
        # accumulating them, the measurement exists and only the fold is missing -- so do the fold,
        # from the engine's OWN declaration of what its columns are. Nothing is derived when the
        # engine declares no columns or wrote no trace: an absent instrument stays absent.
        _trace = _kw.get("per_cycle_csv")
        if _trace:
            _folded = _CYCLE_TRACE.block_from_trace(_trace, Path(vsim_dir))
            if _folded is not None:
                _obs, _cap = _timing_block(
                    dict(_folded, **{_OBS_KEYS.ALIAS_COLLISIONS_KEY: res.get(_OBS_KEYS.ALIAS_COLLISIONS_KEY)})
                )
    if _derived_trace is not None:
        try:
            _derived_trace.unlink()
        except OSError:
            pass
    if _obs:
        out["timing_observations"] = _obs
    if _cap:
        out["timing_capability"] = _cap
    return out


def _engine_provenance(target: str, engine: str, engine_dir: Path | None = None) -> dict[str, Any]:
    """Identify the engine BUILD that produced a number: its directory, its wrapper, and the digest of
    whatever binary sits beside the wrapper.

    A cert record pinned the RTL and the toolchain and said nothing about the simulator that actually
    ran — which for an out-of-tree engine build is the one input tying the verdict to an elaboration.
    Digest every non-source file in the engine home rather than naming one: merlin does not know an
    engine's binary name and must not learn one. Cheap (a handful of files) and never raises; provenance
    that cannot be established is recorded as absent, never invented.
    """
    rec: dict[str, Any] = {"engine": engine}
    try:
        # THE DIRECTORY THE RUNNER WAS ACTUALLY LOADED FROM, when the caller knows it. Re-resolving
        # here instead was a real defect, caught end-to-end: the caller had been handed one engine dir
        # explicitly while resolution returned a different registered one, and the cert recorded the
        # digests of binaries that did not produce the number. A provenance block naming the wrong build
        # is worse than none, because it reads as an answer.
        d = Path(engine_dir) if engine_dir is not None else _rtl_engine_dir(target, engine)
        if d is None:
            return rec
        rec["engine_dir"] = str(d)
        rec["wrapper"] = _RTL_ENGINES[engine][1]
        from merlin.common.provenance_claims import file_digest

        binaries = {}
        for f in sorted(d.iterdir()):
            if f.is_file() and os.access(f, os.X_OK):
                binaries[f.name] = file_digest(f)
        if binaries:
            rec["binaries"] = binaries
        adoption = d / "provenance.json"
        if adoption.is_file():
            rec["adoption_record"] = {"path": str(adoption), "sha256": file_digest(adoption)}
        else:
            rec["lineage"] = (
                "no adoption record or build receipt beside this engine build — these "
                "bytes are not attributed to any RTL revision"
            )
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def program_verilator_adapter(target: str, *, model_ext: str, selection: dict | None = None) -> Callable | None:
    """The L4 cycle-accurate Verilator oracle adapter (``run(cb, fourth_text, workdir, timeout)`` shape)
    for an ``external_backend`` target — mirrors :func:`program_oracle_adapter` but runs the emitted
    kernel on the target's program-driven Verilator RTL sim. Returns ``None`` (no L4) when the target
    registers no vsim (``MERLIN_EXT_<TARGET>_VSIM`` unset) or its ``verilator_run.py`` is absent — so the
    tier is honestly ADDITIVE and target-agnostic (any external_backend target that ships a vsim gets it,
    with no target literal here)."""
    from merlin.development.targetgen.oracles.program_oracle import derive_cycle_budget

    if selection is None:
        try:
            selection = select_rtl_engine(target)
        except OracleUnavailable:
            return None  # no elaborated-RTL engine at all -> the tier is honestly absent
    engine = selection["engine"]
    vsim_dir = _rtl_engine_dir(target, engine)

    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        # PER-CYCLE OCCUPANCY, opt-in. It is ~34 B/cycle of PURGEABLE, replay-regenerable detail, so
        # it goes to the cache root and never into a product, and it is off unless asked for.
        _csv = None
        if os.environ.get("MERLIN_PERF_PER_CYCLE") == "1":
            from merlin.common.artifacts import cache_dir

            _csv = cache_dir("occupancy") / target / f"{wd.parent.name or wd.name}.csv"
        # Size the budget from the workload exactly as the arc tier does. Omitting it left every
        # capsule on run_program's 20000 default while the arc tier scaled with the command buffer, so
        # a correct kernel that simply needed longer raised ProgramDidNotHalt -- which the runner books
        # as the SUBMISSION's defect. A harness limit must not be reported as an agent's bug, and the
        # asymmetry between two tiers of the same target had no reason behind it.
        res = run_program_verilator_oracle(
            target,
            model_ext=model_ext,
            vsim_dir=vsim_dir,
            cb=cb,
            engine=engine,
            kernel_s=ks,
            workdir=wd,
            timeout=timeout,
            max_cycles=derive_cycle_budget(cb),
            per_cycle_csv=_csv,
        )
        # CARRY THE CHOICE, not just its outcome. `selection` records every engine considered and the
        # reason each slower/faster one was passed over. Without it a cert that ran on Verilator because
        # the fast engine was unregistered is indistinguishable from one that ran on Verilator because it
        # was the only engine there is — and the first is a thing to fix while the second is not.
        if isinstance(res.get("oracle"), dict):
            res["oracle"]["selection"] = dict(selection)
        return res

    return run
