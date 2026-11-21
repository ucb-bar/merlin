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


class ProgramDidNotHalt(OracleUnavailable):
    """The oracle RAN the submitted program and it never reached the ISA's halt/terminate instruction
    inside the cycle/instruction budget.

    This is a VERDICT about the artifact, not an absent oracle — the two were collapsed into
    ``OracleUnavailable`` and the grade reported "the oracle did not run" for a program that had run and
    hung. An agent cannot act on "the oracle is unavailable"; it can act on "your program never halts".
    Subclasses ``OracleUnavailable`` so existing fail-closed handlers keep working; handlers that want
    the distinction catch THIS first."""


_COSIM_HELPER = Path(__file__).resolve().parent / "oracle_helpers" / "program_cosim.py"


_FUNC_ERRATA_HELPER = Path(__file__).resolve().parent / "oracle_helpers" / "functional_errata_runner.py"


def _assemble_kernel_words(kernel_s: Path, workdir: Path, inst_width: int = 32) -> list[int]:
    """Assemble an agent-emitted ``.word``/``.insn``/``.quad`` kernel into little-endian IMEM words using the
    PREBUILT stock LLVM (``llvm-mc`` + ``llvm-objcopy`` from the MLIR install; override via
    ``MERLIN_MLIR_INSTALL``). Target-agnostic: the encoding lives in the emitted directives (grounded on
    the target's shipped ISA definition), so merlin needs no per-target assembler and holds no opcode
    table. ``inst_width`` (32 or 64) selects the word size the ``.text`` bytes are grouped into — a
    fixed-width wide-word ISA (a SIMT core's 64-bit encoding) is read as u64, the default RoCC/word ISA as
    u32. ``llvm-mc`` accepts ``#``/``//`` comments + labels natively (labels emit no ``.text`` bytes).
    Raises :class:`OracleUnavailable` if the toolchain is absent or the kernel does not assemble."""
    from merlin.target.contract.interfaces.abi import mlir_bin

    mc, objcopy = mlir_bin("llvm-mc"), mlir_bin("llvm-objcopy")
    if not mc.is_file() or not objcopy.is_file():
        raise OracleUnavailable(f"prebuilt stock LLVM assembler absent ({mc} / {objcopy}); set MERLIN_MLIR_INSTALL")
    obj, binf = workdir / "kernel.o", workdir / "kernel.bin"
    a = subprocess.run(
        [str(mc), "-triple=riscv64", "-filetype=obj", "-o", str(obj), str(kernel_s)], capture_output=True, text=True
    )
    if a.returncode != 0:
        raise OracleUnavailable(f"llvm-mc failed to assemble kernel.S: {a.stderr[-500:]}")
    b = subprocess.run(
        [str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)], capture_output=True, text=True
    )
    if b.returncode != 0:
        raise OracleUnavailable(f"llvm-objcopy failed: {b.stderr[-400:]}")
    import numpy as np

    dtype = "<u8" if inst_width > 32 else "<u4"
    words = [int(w) for w in np.frombuffer(binf.read_bytes(), dtype=dtype)]
    if not words:
        raise OracleUnavailable("kernel.S assembled to zero .text words (empty/all-comment kernel)")
    return words


def _run_emit_helper(
    target: str, model_ext: str, extra_args: list[str], workdir: Path, timeout: int, out_name: str
) -> dict[str, Any]:
    """Run the selected support emitter in the model venv and read its JSON bundle."""
    helper, options = _program_emitter(target)
    py = _model_venv_python(model_ext)
    # The model's assembler needs its own cwd. Pin the output against the
    # caller's cwd first, so a relative run directory stays a run artifact.
    workdir = Path(workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / out_name
    cmd = [str(py), str(helper), *options, "--out", str(out), *extra_args]
    cwd = ext_path(model_ext)  # ASM_FOLDER is cwd-relative in the model
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise OracleUnavailable(f"program emitter failed rc={p.returncode}: {p.stderr[-400:]}")
    return json.loads(out.read_text())


def _program_emitter(target: str) -> tuple[Path, list[str]]:
    """Resolve a subprocess helper without importing target code into the host interpreter."""
    from merlin.compiler.backend.providers import ProviderRole, contained_resource
    from merlin.target.registry.identity import resolve

    try:
        selected = resolve(target)
        provider = selected.provider
        if selected.kind != "external" or provider is None or provider.role != ProviderRole.SUPPORT:
            raise ValueError("an explicitly selected OOT support provider is required")
        declaration = (selected.load_contract().get("runner") or {}).get("program_emitter")
        if not isinstance(declaration, dict) or set(declaration) - {"path", "args"}:
            raise ValueError("runner.program_emitter must declare path and optional args")
        path = declaration.get("path")
        options = declaration.get("args", [])
        if not isinstance(path, str) or not path:
            raise ValueError("runner.program_emitter.path must be a nonempty relative filename")
        if not isinstance(options, list) or any(not isinstance(arg, str) for arg in options):
            raise ValueError("runner.program_emitter.args must be a list of strings")
        return contained_resource(provider.root, path), list(options)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise OracleUnavailable(f"{target!r}: program emitter unavailable: {exc}") from exc


def emit_bundle(
    *,
    target: str,
    model_ext: str,
    program: str | None = None,
    kernel_s: Path | None = None,
    inputs: list[dict] | None = None,
    workdir: Path,
    timeout: int,
) -> dict[str, Any]:
    """Produce the oracle bundle ``{words, inputs:[{base,b64}], output, golden}``. Two paths:

    * ``program`` (self-contained validation): assemble a named provider ``Program`` + lay out its
      memory regions/golden — done in the MODEL venv (it owns the ``Program`` classes + assembler).
    * ``kernel_s`` + ``inputs`` (the capsule/agent path): assemble the agent's ``.word``/``.insn`` kernel
      with PREBUILT stock LLVM merlin-side (target-agnostic — no model assembler), and lay out the input
      tensors' DRAM bytes in the model venv (its torch owns the fp8/bf16 dtypes)."""
    if program:
        args = ["--program", program]
        return _run_emit_helper(target, model_ext, args, workdir, timeout, "program_bundle.json")
    if kernel_s is None:
        raise ValueError("emit_bundle: need program or kernel_s")
    words = _assemble_kernel_words(kernel_s, workdir)
    laid: list[dict] = []
    if inputs:
        b = _run_emit_helper(
            target, model_ext, ["--inputs", json.dumps(inputs)], workdir, timeout, "program_inputs.json"
        )
        laid = b.get("inputs", [])
    return {"words": words, "inputs": laid, "output": None, "golden": None}


def _preload_from_cb(cb: dict) -> list[tuple[int, bytes]]:
    """DRAM preload (base, bytes) for the leaf inputs the grader attached to the cb's tensors as
    ``preload_b64`` — the capsule's canonical operands (see :func:`capsule_golden.canonical_input_raws`).
    Base is the agent's cb-declared address (where its kernel reads the operand); the bytes are the
    ground-truth operands the independent golden used. Target-agnostic: no operand values live here."""
    pre: list[tuple[int, bytes]] = []
    for t in (cb.get("tensors") or {}).values():
        if t.get("role") in ("input", "weight", "bias") and t.get("preload_b64") and t.get("base") is not None:
            pre.append((int(t["base"]), base64.b64decode(t["preload_b64"])))
    return pre


def _declare_executed_program(workdir: Path, bundle: dict) -> None:
    """Record the exact program+stimulus this oracle is about to execute. Never raises.

    THIS TARGET EXECUTES NO LINKED ELF, so nothing on disk identified what ran and its verdict could
    not be content-addressed: measured across its runs, 89% of emitted programs are byte-identical to a
    previous grade's and 84% of simulation time (12,376 s of 14,821 s) re-derives a verdict already
    earned. Writing what it runs makes those carryable.

    BOTH halves, because either alone would be a false identity. ``words`` is the assembled program;
    the inputs are the stimulus it is run on, and the same program on different operands is a different
    execution. A digest over one of them would let a verdict be carried across a change in the other.

    Best-effort: an unwritable artifact means the tier is measured afresh, which is the safe direction.
    """
    from merlin.development.targetgen.grading.elf_lanes import PROGRAM_ARTIFACT_SUFFIX

    try:
        payload = {
            "words": bundle.get("words"),
            "inputs": [
                {"base": i.get("base"), "b64": i.get("b64")}
                for i in (bundle.get("inputs") or [])
                if isinstance(i, dict)
            ],
        }
        (workdir / f"oracle{PROGRAM_ARTIFACT_SUFFIX}").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 -- a missing declaration only costs a re-run
        return


def _bundle_preload(bundle: dict, cb: dict | None) -> list[tuple[int, bytes]]:
    """DRAM preload ``(base, bytes)`` from the emit bundle's laid-out inputs, falling back to the cb's
    canonical leaf operands (``preload_b64``) exactly like the cosim path (see AW5)."""
    preload = [(int(x["base"]), base64.b64decode(x["b64"])) for x in bundle["inputs"]]
    if not preload and cb:
        preload = _preload_from_cb(cb)
    return preload


def run_raw_program(target: str, *, words: list[int], preload: list[tuple[int, bytes]], max_cycles: int = 20000):
    """Run already-assembled words on a target's discovered program cosim.

    This narrow public seam is also used by descriptor-owned assembly preflight adapters.  Assembly,
    memory expectations, and operation identities remain outside this function; backend discovery stays
    identical to the grading oracle and contains no target-name branch.
    """
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    if not _merlin_mlc_discovery_1.arc_available(target):
        raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")

    modeling = mlc_bridge.mlc_dir()
    with _merlin_mlc_discovery_1._mlc_cwd(), modelir.modelir_importable(modeling):
        try:
            from mlc.backends.cosim_core import large_stack_call
            from mlc.discover import fingerprint

            arc_name = mlc_bridge._arc_target(target)
            backend_name = fingerprint.cosim_backend(arc_name)
            backend = importlib.import_module(f"mlc.backends.{backend_name}")
        except Exception as exc:  # noqa: BLE001
            raise OracleUnavailable(f"mlc program-cosim import failed: {type(exc).__name__}: {exc}") from exc
        if not hasattr(backend, "run_program"):
            raise OracleUnavailable(
                f"mlc backend mlc.backends.{backend_name} for target {target!r} exposes no run_program "
                "(not a self-hosted-ISA program cosim)"
            )
        artifacts = fingerprint.artifact_paths(arc_name, base=modeling)
        return large_stack_call(
            backend.run_program,
            str(artifacts["so"]),
            str(artifacts["man"]),
            words,
            preload=preload,
            max_cycles=max_cycles,
        )


def _run_raw_program_helper(
    target: str,
    *,
    words: list[int],
    preload: list[tuple[int, bytes]],
    reads: dict[str, tuple[int, int]],
    max_cycles: int,
    workdir: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run an Arc program in a disposable process and return JSON-safe observations.

    Some Arc backends need :func:`large_stack_call`, whose implementation waits on a Python thread.
    A thread cannot be safely cancelled when the backend wedges, so a timeout around the caller was
    previously fictitious: the grader could remain inside ``join()`` indefinitely.  The process is the
    cancellation boundary.  This is shared by every self-hosted-ISA target; target discovery and all ISA
    facts remain in :func:`run_raw_program`.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    request = wd / "program_cosim_request.json"
    result = wd / "program_cosim_result.json"
    request.write_text(
        json.dumps(
            {
                "target": target,
                "words": [int(word) for word in words],
                "preload": [{"base": int(base), "b64": base64.b64encode(raw).decode()} for base, raw in preload],
                "reads": [
                    {"name": name, "base": int(base), "nbytes": int(nbytes)} for name, (base, nbytes) in reads.items()
                ],
                "max_cycles": int(max_cycles),
            }
        )
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(_COSIM_HELPER), "--request", str(request), "--out", str(result)],
            capture_output=True,
            text=True,
            timeout=max(0.001, float(timeout)),
        )
    except subprocess.TimeoutExpired as exc:
        raise ProgramDidNotHalt(
            f"{target} program exceeded the {float(timeout):g}s wall timeout; the isolated cosim worker was terminated"
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "cosim worker emitted no diagnostic")[-800:]
        raise OracleUnavailable(f"{target} program-cosim worker failed rc={proc.returncode}: {detail}")
    try:
        payload = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleUnavailable(f"{target} program-cosim worker returned no valid result: {exc}") from exc
    if not isinstance(payload, dict) or "halted" not in payload or "cycles" not in payload:
        raise OracleUnavailable(f"{target} program-cosim worker returned an incomplete result")
    return payload


def _sim_timing(seconds: float, *, build_s: float = 0.0) -> dict:
    """The timing block the cost model reads. ``oracle_wait_s`` is zero here by construction: these
    adapters call the simulator in-process, so there is no queue wait to attribute."""
    return {"build_s": float(build_s), "sim_active_s": float(seconds), "oracle_wait_s": 0.0}


def run_program_oracle(
    target: str,
    *,
    model_ext: str,
    cb: dict | None = None,
    kernel_s: Path | None = None,
    program: str | None = None,
    inputs: list[dict] | None = None,
    max_cycles: int = 20000,
    workdir: Path,
    timeout: int = 600,
) -> dict[str, Any]:
    """Assemble (model venv) → run on the target's mlc arc cosim (``large_stack_call``) → read back.
    Returns ``{outputs, cycles, oracle}``. ``cb`` supplies the output tensor {base, shape, dtype,
    physical}; for self-contained validation ``program`` provides its own golden/output instead."""
    deadline = time.monotonic() + float(timeout)
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
    _declare_executed_program(Path(workdir), bundle)
    # Capsule path: preload arc DRAM with the capsule's CANONICAL leaf inputs, attached to the cb's
    # leaf tensors by the grader (``preload_b64`` = the exact bytes the independent golden used — the
    # float target's operand palette, not the integer 0..3 fill). Base comes from the agent's cb tensor
    # (where its kernel reads the operand); the bytes are the ground-truth operands (see AW5).
    preload = _bundle_preload(bundle, cb)

    specs = _resolve_out_specs(target, cb, bundle)
    reads = {name: (spec["base"], _out_nbytes(spec)) for name, spec in specs.items()}
    _t0 = time.monotonic()
    res = _run_raw_program_helper(
        target,
        words=words,
        preload=preload,
        reads=reads,
        max_cycles=max_cycles,
        workdir=workdir,
        timeout=max(0.001, deadline - time.monotonic()),
    )
    if not res["halted"]:
        raise ProgramDidNotHalt(f"{target} program did not halt within {max_cycles} cycles")
    _obs = res.get("timing_observations")
    _cap = res.get("timing_capability")

    # resolve EVERY output tensor spec from the cb (generation-declared) or the program's own golden and
    # read each one back: a module that commits twice has two results, and capturing only the first
    # grades the second as never written whatever the kernel did.
    outputs = {}
    captured = res.get("captured") or {}
    for name, spec in specs.items():
        if name not in captured:
            raise OracleUnavailable(f"{target} program-cosim worker did not capture output {name!r}")
        raw = base64.b64decode(captured[name])
        outputs[name] = _decode_output(raw, spec["shape"], spec["dtype"], spec["physical"], name=name).tolist()
    # RTL-DERIVED IS NOT RTL, AND THE TIER NAME CANNOT TELL THEM APART. This cosim runs the arc MODEL
    # elaborated from the target's RTL -- authoritative about the ISA and the datapath, but not the
    # elaborated Verilog. It lands on the tier named L3, which on a target whose bespoke sim IS verilator
    # means genuinely-RTL, so classifying by tier NAME credited a model as RTL certification.
    # Declaring `derived_from_rtl` here is the seam capsule_runner already reads (it defaults to the tier
    # name only when the adapter stays silent) and the shape a SIMT target's gsim adapter already returns.
    arc_out: dict[str, Any] = {
        "outputs": outputs,
        "cycles": int(res["cycles"]),
        "timing": _sim_timing(time.monotonic() - _t0),
        "oracle": {"kind": f"{target}-arc-arcilator-cosim", "derived_from_rtl": False, "fidelity": "rtl_derived_model"},
    }
    # Same pass-through as the Verilator tier, and for the same reason: if this cosim ever reports a
    # decomposition it is harvested the moment it does. Today it reports none, so nothing is added
    # here -- which is the correct output for an oracle without the capability, and is NOT zeros.
    if _obs:
        arc_out["timing_observations"] = _obs
    if _cap:
        arc_out["timing_capability"] = _cap
    return arc_out


def derive_cycle_budget(cb: dict, *, floor: int = 20000, per_element: int = 64) -> int:
    """A halt budget sized to THIS program's declared work, read off the command buffer's own tensor
    extents — not a fixed cap.

    The default 20000 is a constant chosen for tile-sized programs. A real model layer is much larger: a
    32x352x128 matmul needs far more than that, so the oracle stopped it mid-flight and raised
    ``ProgramDidNotHalt`` — which callers collapsed into "no reachable oracle", i.e. a correct program on a
    working oracle was reported as a missing oracle. The budget must therefore GROW with the work.

    Derived from the total element count the buffer declares (every element must at minimum be moved once,
    and a contraction's work is monotone in operand size), scaled by ``per_element`` for slack and floored
    at the original constant so small programs are unaffected. This is a HANG backstop, not a performance
    model — over-budgeting a healthy program costs nothing because it halts on its own, while the wall-clock
    ``timeout`` remains the outer backstop. Opcode-agnostic by construction: it reads tensor shapes only,
    never a command spelling."""
    total = 0
    for spec in (cb.get("tensors") or {}).values():
        shape = spec.get("shape") or []
        n = 1
        for d in shape:
            n *= max(1, int(d))
        total += n
    return max(int(floor), int(floor) + per_element * total)


def program_oracle_adapter(target: str, *, model_ext: str) -> Callable:
    """An oracle adapter (the ``run(cb, fourth_text, workdir, timeout)`` shape ``capsule_runner`` expects)
    for an ``external_backend`` target. ``fourth_text`` is the agent's emitted ``kernel.S``."""

    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        return run_program_oracle(
            target,
            model_ext=model_ext,
            cb=cb,
            kernel_s=ks,
            workdir=wd,
            timeout=timeout,
            max_cycles=derive_cycle_budget(cb),
        )

    return run


def run_program_oracle_smoke(
    target: str,
    *,
    model_ext: str,
    program: str,
    workdir,
    max_cycles: int = 20000,
    timeout: int = 600,
    required_instruction_classes: tuple[str, ...] = (),
    isa_model=None,
) -> dict[str, Any]:
    """END-TO-END pre-flight oracle smoke for a self-hosted-ISA (``external_backend``) target: run a
    KNOWN-GOOD, self-contained model ``program`` (one that ships its OWN ``golden_result``) through the
    FULL grading path — assemble (the model's assembler) → arc cosim → read back the output region — and
    compare the read-back result to that golden BIT-EXACT.

    ``arc_available`` + adapter routing only prove the oracle's pieces are wired; they do NOT prove the
    path computes the right answer. This does: if a real run cannot be graded to a correct verdict, the
    launcher must learn that BEFORE spending tokens (the atlas 0/11-at-\\$43 lesson). ``program`` is the
    caller's PARAMETER — the concrete known-good program name is a per-target SETUP fact declared in the
    descriptor, never a literal here (this module is HW-agnostic).

    ``required_instruction_classes`` optionally binds a capability smoke to the instruction semantics it
    claims to exercise. The words are decoded through the target's derived :class:`IsaModel`; a fixture
    missing a required class fails before the expensive cosim. Both the class names and program come from
    the descriptor -- this generic runner does not name an instruction or target.

    Returns ``{ok, program, cycles, oracle, shape, mismatches, reason, instruction_coverage?}``. Raises
    :class:`OracleUnavailable` when the model venv / cosim is absent or the program does not halt (so the
    caller decides NO_GO vs. a clean skip — never a silent pass)."""
    import numpy as np

    wd = Path(workdir)
    gwd, rwd = wd / "golden", wd / "run"
    gwd.mkdir(parents=True, exist_ok=True)
    rwd.mkdir(parents=True, exist_ok=True)
    # 1) the program's OWN golden — computed by the model's fp8/bf16 datapath (the INDEPENDENT reference,
    #    not the cosim), so a bit-exact match certifies the whole assemble→run→readback chain.
    gb = emit_bundle(target=target, model_ext=model_ext, program=program, workdir=gwd, timeout=timeout)
    g = gb.get("golden")
    if not g:
        raise OracleUnavailable(
            f"{program!r}: model program ships no golden_result — cannot form a bit-exact smoke verdict"
        )
    golden = _decode_output(
        base64.b64decode(g["b64"]), list(g["shape"]), str(g["dtype"]), None, name=f"{program}:golden_result"
    )
    instruction_coverage = None
    if required_instruction_classes:
        if isa_model is None:
            from merlin.target.isa.machine_model import isa_model_for_target

            isa_model = isa_model_for_target(target)
        from merlin.target.isa import isa_disasm

        records = isa_disasm.disassemble(isa_model, list(gb.get("words") or []))
        instruction_coverage = isa_disasm.coverage(isa_model, records, required=list(required_instruction_classes))
        if instruction_coverage["missing"] or instruction_coverage["n_illegal"]:
            problems = []
            if instruction_coverage["missing"]:
                problems.append(f"missing {instruction_coverage['missing']}")
            if instruction_coverage["n_illegal"]:
                problems.append(f"{instruction_coverage['n_illegal']} undecodable word(s)")
            return {
                "ok": False,
                "program": program,
                "cycles": 0,
                "oracle": None,
                "shape": list(golden.shape),
                "mismatches": None,
                "instruction_coverage": instruction_coverage,
                "reason": (
                    f"{program}: capability fixture does not exercise its descriptor-declared "
                    f"instruction classes ({'; '.join(problems)})"
                ),
            }
    # 2) the FULL oracle path: assemble → arc cosim → read back the declared output region.
    res = run_program_oracle(
        target, model_ext=model_ext, program=program, max_cycles=max_cycles, workdir=rwd, timeout=timeout
    )
    got = np.array(next(iter(res["outputs"].values())))
    shape_ok = tuple(got.shape) == tuple(golden.shape)
    mism = None if not shape_ok else int(np.count_nonzero(got != golden))
    ok = bool(shape_ok and mism == 0)
    if ok:
        reason = f"{program} matches its golden bit-exact on the {res.get('oracle')}"
    elif not shape_ok:
        reason = f"{program}: output shape {tuple(got.shape)} != golden {tuple(golden.shape)}"
    else:
        reason = f"{program}: diverged from its golden ({mism} element(s) differ) on {res.get('oracle')}"
    result = {
        "ok": ok,
        "program": program,
        "cycles": int(res.get("cycles") or 0),
        "oracle": res.get("oracle"),
        "shape": list(got.shape),
        "mismatches": mism,
        "reason": reason,
    }
    if instruction_coverage is not None:
        result["instruction_coverage"] = instruction_coverage
    return result


def run_capability_probe(*, te, probe, workdir, timeout: int = 600) -> dict[str, Any]:
    """Generic-protocol adapter for a model-owned named self-hosted-ISA program.

    This is one adapter behind :mod:`preflight_probes`, not the global probe runner. Other adapters may
    lower RVV source or target-dialect IR through the exact same interface. The descriptor supplies the
    operation identities; this adapter accepts only instruction-domain identities and verifies each exact
    operation mnemonic occurs in the derived disassembly before the bit-exact result counts.
    """
    fixture = probe.fixture
    if fixture.get("kind") != "named_program" or not fixture.get("name"):
        raise OracleUnavailable("program-oracle probe requires fixture {kind: named_program, name: ...}")
    operations = list(probe.requirements.get("operations") or [])
    if any(operation.get("domain") != "instruction" for operation in operations):
        raise OracleUnavailable("program-oracle probe accepts instruction-domain operations only")
    from merlin.target.simulators.oracle_policy import selected_endpoint

    endpoint_kind, model_ext = selected_endpoint(te.target)
    if endpoint_kind != "external_backend" or not model_ext:
        raise OracleUnavailable(f"{te.target!r} does not expose an external_backend runner with model_ext")
    from merlin.target.isa.machine_model import isa_model_for_target

    result = run_program_oracle_smoke(
        te.target,
        model_ext=model_ext,
        program=str(fixture["name"]),
        workdir=workdir,
        timeout=timeout,
        required_instruction_classes=tuple(str(operation["operation"]) for operation in operations),
        isa_model=isa_model_for_target(te.target),
    )
    status = "supported" if result["ok"] else "unsupported"
    return {
        "reason": result["reason"],
        "observations": [
            {**operation, "status": status, "evidence": {"kind": "rtl_preflight", "detail": result["reason"]}}
            for operation in operations
        ],
        "program_result": result,
    }


def _func_program_helper(target: str) -> Path:
    """The mlc FUNCTIONAL program runner file for ``target``, invoked BY PATH in the model venv (the runner
    imports only the target's own model package, never ``mlc.*``, so ``mlc.backends.__init__`` is not
    triggered in the subprocess). The filename is DERIVED from the target's registered cosim backend (mlc's
    ``cosim_<stem>`` -> ``func_program_<stem>``), mirroring how the cosim backend itself is derived — so NO
    target-name literal lives in merlin. Raises :class:`OracleUnavailable` if ``mlc_dir()`` is None, the
    target has no registered program backend, or the file is absent (fail closed)."""
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_2
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    d = mlc_bridge.mlc_dir()
    if d is None:
        raise OracleUnavailable("mlc dir unavailable (set MERLIN_MLC_DIR) — no functional program runner")
    try:
        with _merlin_mlc_discovery_2._mlc_cwd(), modelir.modelir_importable(str(d)):
            from mlc.discover import fingerprint

            cosim = str(fingerprint.cosim_backend(mlc_bridge._arc_target(target)))  # e.g. "cosim_<stem>"
    except Exception as e:  # noqa: BLE001 — no registered program backend -> honestly unavailable
        raise OracleUnavailable(f"no functional program runner derivable for {target!r}: {type(e).__name__}: {e}")
    stem = cosim[len("cosim_") :] if cosim.startswith("cosim_") else cosim
    f = Path(d) / "mlc" / "backends" / f"func_program_{stem}.py"
    if not f.is_file():
        raise OracleUnavailable(f"mlc functional program runner absent: {f}")
    return f


def _run_func_helper(target: str, model_ext: str, req: dict, workdir: Path, timeout: int) -> dict[str, Any]:
    """Run the mlc functional program runner in the MODEL venv (same venv+cwd as ``_run_emit_helper``),
    marshalling a JSON+base64 request/result across the venv gap (mirrors its ``__main__`` CLI).

    A shipped ISA model is evidence and is never edited in place.  Reviewed RTL-authoritative
    corrections are instead passed to a process-local overlay helper.  This makes the functional
    decoder execute the same instruction words as hardware while retaining both the original spec and
    an auditable receipt of every field changed for this invocation.
    """
    py = _model_venv_python(model_ext)
    func = _func_program_helper(target)
    infile, outfile = workdir / "func_in.json", workdir / "func_out.json"
    from merlin.target.isa.machine_model import _reviewed_errata

    corrections = {
        name: row
        for name, row in _reviewed_errata(target).items()
        if str(row.get("authoritative") or "").lower() == "rtl"
        and row.get("declared") not in (None, "")
        and row.get("hardware") not in (None, "")
    }
    request = dict(req)
    request["reviewed_isa_errata"] = corrections
    functional_registry = data_path("contract", "functional_model_errata.yaml")
    functional_corrections = {}
    functional_isa_module = ""
    if functional_registry.is_file():
        import yaml

        functional_doc = yaml.safe_load(functional_registry.read_text()) or {}
        functional_section = dict(((functional_doc.get("errata") or {}).get(target)) or {})
        functional_isa_module = str(functional_section.get("isa_module") or "")
        functional_corrections = dict(functional_section.get("corrections") or {})
    request["reviewed_functional_model_errata"] = functional_corrections
    request["functional_model_isa_module"] = functional_isa_module
    infile.write_text(json.dumps(request))
    if corrections or functional_corrections:
        if not functional_isa_module:
            raise OracleUnavailable(f"{target}: reviewed functional corrections name no model ISA module")
        cmd = [str(py), str(_FUNC_ERRATA_HELPER), "--runner", str(func), "--in", str(infile), "--out", str(outfile)]
    else:
        request.pop("reviewed_isa_errata")
        request.pop("reviewed_functional_model_errata")
        request.pop("functional_model_isa_module")
        infile.write_text(json.dumps(request))
        cmd = [str(py), str(func), "--in", str(infile), "--out", str(outfile)]
    cwd = ext_path(model_ext)  # ASM_FOLDER is cwd-relative in the model
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise OracleUnavailable(f"functional program runner failed rc={p.returncode}: {p.stderr[-400:]}")
    return json.loads(outfile.read_text())
