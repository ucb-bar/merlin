"""Generic ``external_backend`` PROGRAM oracle — run an agent-emitted target-ISA program on the target's
own cosim, using the target's OWN assembler. The counterpart to the ``command_buffer`` oracle
(``mlc_bridge.arc_run_command_buffer``, gemmini/OPU) for accelerators that are a self-hosted ISA core
(their deliverable is an assembled program, e.g. atlas ``kernel.S`` → IMEM words), NOT a RoCC ``.insn``
host stream and NOT an ISA-less command buffer.

Design (HW-agnostic; nothing atlas-specific is hardcoded here):
  * the agent's emitted ``kernel.S`` is a stream of ``.word``/``.insn`` directives (the target's encoded
    instructions, grounded on the target's shipped ISA definition) — assembled to IMEM words by the
    PREBUILT stock LLVM (``llvm-mc`` + ``llvm-objcopy`` from the MLIR install), NOT by a per-target
    bespoke assembler and NOT by a forked toolchain. merlin holds no opcode table; the encoding lives in
    the emitted ``.word`` directives, and stock ``.insn`` gives the generic RISC-V-shaped formats. This
    is the target-agnostic assembly path: any HW whose instructions fit ``.word``/``.insn`` assembles the
    same way (see AW3);
  * the input tensors' DRAM byte layout comes from the MODEL project (via :mod:`oracle_helpers.npu_emit`,
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
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from merlin.common.paths import ext_path
from merlin.perf import cycle_trace as _CYCLE_TRACE
from merlin.perf import observations as _OBS_KEYS


class OracleUnavailable(RuntimeError):
    """Raised when the program oracle cannot run (model venv / cosim / arc artifacts absent)."""


class ProgramDidNotHalt(OracleUnavailable):
    """The oracle RAN the submitted program and it never reached the ISA's halt/terminate instruction
    inside the cycle/instruction budget.

    This is a VERDICT about the artifact, not an absent oracle — the two were collapsed into
    ``OracleUnavailable`` and the grade reported "the oracle did not run" for a program that had run and
    hung. An agent cannot act on "the oracle is unavailable"; it can act on "your program never halts".
    Subclasses ``OracleUnavailable`` so existing fail-closed handlers keep working; handlers that want
    the distinction catch THIS first."""


@contextmanager
def _mlc_importable(mlc_dir):
    """Make ``mlc`` importable for the duration of THIS call ONLY, by inserting ``mlc_dir`` on
    ``sys.path`` iff ``mlc`` is not already importable (restored on exit). Deliberately NOT a global /
    module-level insert: a permanent insert flips ``mlc_bridge.mlc_available()`` True process-wide and
    un-skips the heavy mlc-gated tests (a known regression). Kept local to the oracle call so a machine
    with mlc as an external checkout (not pip-installed) can still run the program cosim."""
    added = None
    if mlc_dir is not None and importlib.util.find_spec("mlc") is None:
        added = str(mlc_dir)
        sys.path.insert(0, added)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


_EMIT_HELPER = Path(__file__).resolve().parent / "oracle_helpers" / "npu_emit.py"


def _model_venv_python(model_ext: str) -> Path:
    """The model project's own interpreter (its ISA/assembler live there), from the ``.env`` registry."""
    root = ext_path(model_ext)                       # e.g. MERLIN_EXT_NPU_MODEL
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise OracleUnavailable(f"model venv python absent: {py} (run `uv sync` in {root})")
    return py


def _assemble_kernel_words(kernel_s: Path, workdir: Path, inst_width: int = 32) -> list[int]:
    """Assemble an agent-emitted ``.word``/``.insn``/``.quad`` kernel into little-endian IMEM words using the
    PREBUILT stock LLVM (``llvm-mc`` + ``llvm-objcopy`` from the MLIR install; override via
    ``MERLIN_MLIR_INSTALL``). Target-agnostic: the encoding lives in the emitted directives (grounded on
    the target's shipped ISA definition), so merlin needs no per-target assembler and holds no opcode
    table. ``inst_width`` (32 or 64) selects the word size the ``.text`` bytes are grouped into — a
    fixed-width wide-word ISA (a SIMT core's 64-bit encoding) is read as u64, the default RoCC/word ISA as
    u32. ``llvm-mc`` accepts ``#``/``//`` comments + labels natively (labels emit no ``.text`` bytes).
    Raises :class:`OracleUnavailable` if the toolchain is absent or the kernel does not assemble."""
    from merlin.targetgen.contract.toolchain import mlir_bin
    mc, objcopy = mlir_bin("llvm-mc"), mlir_bin("llvm-objcopy")
    if not mc.is_file() or not objcopy.is_file():
        raise OracleUnavailable(
            f"prebuilt stock LLVM assembler absent ({mc} / {objcopy}); set MERLIN_MLIR_INSTALL")
    obj, binf = workdir / "kernel.o", workdir / "kernel.bin"
    a = subprocess.run([str(mc), "-triple=riscv64", "-filetype=obj", "-o", str(obj), str(kernel_s)],
                       capture_output=True, text=True)
    if a.returncode != 0:
        raise OracleUnavailable(f"llvm-mc failed to assemble kernel.S: {a.stderr[-500:]}")
    b = subprocess.run([str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)],
                       capture_output=True, text=True)
    if b.returncode != 0:
        raise OracleUnavailable(f"llvm-objcopy failed: {b.stderr[-400:]}")
    import numpy as np
    dtype = "<u8" if inst_width > 32 else "<u4"
    words = [int(w) for w in np.frombuffer(binf.read_bytes(), dtype=dtype)]
    if not words:
        raise OracleUnavailable("kernel.S assembled to zero .text words (empty/all-comment kernel)")
    return words


def _run_emit_helper(model_ext: str, extra_args: list[str], workdir: Path, timeout: int,
                     out_name: str) -> dict[str, Any]:
    """Run the model-venv helper (:mod:`oracle_helpers.npu_emit`) and read back its JSON bundle."""
    py = _model_venv_python(model_ext)
    out = workdir / out_name
    cmd = [str(py), str(_EMIT_HELPER), "--out", str(out), *extra_args]
    cwd = ext_path(model_ext)                          # ASM_FOLDER is cwd-relative in the model
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise OracleUnavailable(f"npu_emit helper failed rc={p.returncode}: {p.stderr[-400:]}")
    return json.loads(out.read_text())


def emit_bundle(*, model_ext: str, program: str | None = None, kernel_s: Path | None = None,
                inputs: list[dict] | None = None, fix_itype_rd: bool, workdir: Path,
                timeout: int) -> dict[str, Any]:
    """Produce the oracle bundle ``{words, inputs:[{base,b64}], output, golden}``. Two paths:

    * ``program`` (self-contained validation): assemble a named npu_model ``Program`` + lay out its
      memory regions/golden — done in the MODEL venv (it owns the ``Program`` classes + assembler).
    * ``kernel_s`` + ``inputs`` (the capsule/agent path): assemble the agent's ``.word``/``.insn`` kernel
      with PREBUILT stock LLVM merlin-side (target-agnostic — no model assembler), and lay out the input
      tensors' DRAM bytes in the model venv (its torch owns the fp8/bf16 dtypes)."""
    if program:
        args = ["--program", program]
        if fix_itype_rd:
            args.append("--fix-itype-rd")
        return _run_emit_helper(model_ext, args, workdir, timeout, "npu_bundle.json")
    if kernel_s is None:
        raise ValueError("emit_bundle: need program or kernel_s")
    words = _assemble_kernel_words(kernel_s, workdir)
    laid: list[dict] = []
    if inputs:
        b = _run_emit_helper(model_ext, ["--inputs", json.dumps(inputs)], workdir, timeout,
                             "npu_inputs.json")
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


def _decode_output(raw: bytes, shape: list[int], dtype: str, physical: dict | None):
    import numpy as np
    if dtype in ("bf16", "torch.bfloat16"):
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif dtype in ("i32", "int32", "torch.int32"):
        arr = np.frombuffer(raw, dtype="<i4")
    else:
        arr = np.frombuffer(raw, dtype=np.int8)
    arr = arr.reshape(shape)
    # physical->logical layout, DECLARED by the emitting backend (not a constant here). The atlas MXU
    # writes an [2R, C] tensor as two stacked R-row banks; ``{"unstack_row_halves": 2}`` un-stacks it.
    halves = (physical or {}).get("unstack_row_halves")
    if halves and shape[0] % halves == 0:
        h = shape[0] // halves
        import numpy as np
        arr = np.concatenate([arr[i * h:(i + 1) * h] for i in range(halves)], axis=1)
    return arr


def _resolve_out_specs(target: str, cb: dict | None, bundle: dict) -> dict[str, dict]:
    """EVERY output tensor spec, ``{name: {base, shape, dtype, physical}}`` — from the cb
    (generation-declared) or the program's own golden, in declaration order.

    A command buffer may declare more than one output: an interface module with two commit ops (one
    resident weight, two activations) produces two result tensors, and capturing only the first reports
    the second as never written no matter what the kernel did. Each output DRAM base is the harness-owned
    address (stamped by ``capsule_dram.inject_bases``, the same map the agent's kernel was told to store
    to). A missing base is an actionable grading error (the layout could not be applied), NOT a bare
    ``KeyError``. Target-agnostic: the names and layouts are whatever the emitting backend declared."""
    specs: dict[str, dict] = {}
    declared = [(n, t) for n, t in ((cb or {}).get("tensors") or {}).items()
                if t.get("role") == "output"]
    for name, t in declared:
        # An output the submission declared but gave no address to cannot be captured. That is not a
        # tool error: it is exactly the "you never wrote this output" verdict the numeric compare
        # reports, and reporting it there names the tensor. So it is skipped, not raised on — unless NO
        # declared output has an address, which really is a layout failure the caller must hear about.
        if t.get("base") is None:
            continue
        specs[name] = {"base": int(t["base"]), "shape": list(t["shape"]),
                       "dtype": t.get("dtype", "bf16"), "physical": t.get("physical")}
    if declared and not specs:
        raise OracleUnavailable(
            f"{target}: no declared output tensor has a DRAM base — the harness DRAM layout was not "
            f"applied (capsule_dram.inject_bases); cannot read the result back")
    if not specs and bundle.get("output"):
        o = bundle["output"]
        specs[_output_name(cb)] = {"base": int(o["base"]), "shape": list(o["shape"]),
                                   "dtype": o["dtype"], "physical": o.get("physical")}
    if not specs:
        raise OracleUnavailable(f"{target}: no output tensor declared in cb or program golden")
    return specs


def _resolve_out_spec(target: str, cb: dict | None, bundle: dict) -> dict:
    """The FIRST output tensor spec ``{base, shape, dtype, physical}``. A runner that captures one memory
    region reads this; the cosim path captures every declared output via :func:`_resolve_out_specs`."""
    return next(iter(_resolve_out_specs(target, cb, bundle).values()))


def _out_nbytes(out_spec: dict) -> int:
    import numpy as np
    return int(np.prod(out_spec["shape"])) * (2 if "16" in out_spec["dtype"] else
                                              (4 if "32" in out_spec["dtype"] else 1))


def _output_name(cb: dict | None) -> str:
    return next((n for n, t in (cb.get("tensors") or {}).items()
                 if t.get("role") == "output"), "Y0") if cb else "Y0"


def _bundle_preload(bundle: dict, cb: dict | None) -> list[tuple[int, bytes]]:
    """DRAM preload ``(base, bytes)`` from the emit bundle's laid-out inputs, falling back to the cb's
    canonical leaf operands (``preload_b64``) exactly like the cosim path (see AW5)."""
    preload = [(int(x["base"]), base64.b64decode(x["b64"])) for x in bundle["inputs"]]
    if not preload and cb:
        preload = _preload_from_cb(cb)
    return preload


def run_program_oracle(target: str, *, model_ext: str, cb: dict | None = None,
                       kernel_s: Path | None = None, program: str | None = None,
                       inputs: list[dict] | None = None, fix_itype_rd: bool = True,
                       max_cycles: int = 20000, workdir: Path, timeout: int = 600) -> dict[str, Any]:
    """Assemble (model venv) → run on the target's mlc arc cosim (``large_stack_call``) → read back.
    Returns ``{outputs, cycles, oracle}``. ``cb`` supplies the output tensor {base, shape, dtype,
    physical}; for self-contained validation ``program`` provides its own golden/output instead."""
    from merlin.targetgen.rtl import mlc_bridge
    if not mlc_bridge.arc_available(target):
        raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")

    bundle = emit_bundle(model_ext=model_ext, program=program, kernel_s=kernel_s, inputs=inputs,
                         fix_itype_rd=fix_itype_rd, workdir=workdir, timeout=timeout)
    words = bundle["words"]
    # Capsule path: preload arc DRAM with the capsule's CANONICAL leaf inputs, attached to the cb's
    # leaf tensors by the grader (``preload_b64`` = the exact bytes the independent golden used — the
    # float target's operand palette, not the integer 0..3 fill). Base comes from the agent's cb tensor
    # (where its kernel reads the operand); the bytes are the ground-truth operands (see AW5).
    preload = _bundle_preload(bundle, cb)

    modeling = mlc_bridge.mlc_dir()
    # mlc.backends.__init__ eagerly loads the gemmini cache cwd-relative — run inside mlc's cwd; keep mlc
    # importable for just this call (context-managed, never a global sys.path insert — see _mlc_importable).
    with mlc_bridge._mlc_cwd(), _mlc_importable(modeling):
        try:
            from mlc.backends.cosim_core import large_stack_call
            from mlc.discover import fingerprint
            # The cosim backend module is DERIVED from the target by mlc's own registry (atlas ->
            # cosim_atlas, ...), so no target-cosim literal lives in merlin. A target with no registered
            # program cosim (or a backend that isn't a self-hosted-ISA program runner) is honestly
            # unavailable, never a fabricated fallback.
            arc_name = mlc_bridge._arc_target(target)   # composite target -> its mlc arc key (else identity)
            backend_name = fingerprint.cosim_backend(arc_name)
            backend = importlib.import_module(f"mlc.backends.{backend_name}")
        except Exception as e:  # noqa: BLE001
            raise OracleUnavailable(f"mlc program-cosim import failed: {type(e).__name__}: {e}")
        if not hasattr(backend, "run_program"):
            raise OracleUnavailable(
                f"mlc backend mlc.backends.{backend_name} for target {target!r} exposes no run_program "
                f"(not a self-hosted-ISA program cosim)")
        ap = fingerprint.artifact_paths(arc_name, base=modeling)
        res = large_stack_call(backend.run_program, str(ap["so"]), str(ap["man"]),
                               words, preload=preload, max_cycles=max_cycles)
    if not res.halted:
        raise ProgramDidNotHalt(f"{target} program did not halt within {max_cycles} cycles")
    _obs, _cap = _timing_block(res)      # only if THIS oracle grows the capability; never invented

    # resolve EVERY output tensor spec from the cb (generation-declared) or the program's own golden and
    # read each one back: a module that commits twice has two results, and capturing only the first
    # grades the second as never written whatever the kernel did.
    outputs = {}
    for name, spec in _resolve_out_specs(target, cb, bundle).items():
        raw = bytes(res.slave.captured(spec["base"], _out_nbytes(spec)))
        outputs[name] = _decode_output(raw, spec["shape"], spec["dtype"], spec["physical"]).tolist()
    # RTL-DERIVED IS NOT RTL, AND THE TIER NAME CANNOT TELL THEM APART. This cosim runs the arc MODEL
    # elaborated from the target's RTL -- authoritative about the ISA and the datapath, but not the
    # elaborated Verilog. It lands on the tier named L3, which on a target whose bespoke sim IS verilator
    # (gemmini) means genuinely-RTL, so classifying by tier NAME credited a model as RTL certification.
    # Declaring `derived_from_rtl` here is the seam capsule_runner already reads (it defaults to the tier
    # name only when the adapter stays silent) and the shape muon's gsim adapter already returns.
    arc_out: dict[str, Any] = {"outputs": outputs, "cycles": int(res.cycles),
                               "oracle": {"kind": f"{target}-arc-arcilator-cosim",
                                          "derived_from_rtl": False,
                                          "fidelity": "rtl_derived_model"}}
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
        return run_program_oracle(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                  workdir=wd, timeout=timeout,
                                  max_cycles=derive_cycle_budget(cb))
    return run


def run_program_oracle_smoke(target: str, *, model_ext: str, program: str, workdir,
                             max_cycles: int = 20000, timeout: int = 600) -> dict[str, Any]:
    """END-TO-END pre-flight oracle smoke for a self-hosted-ISA (``external_backend``) target: run a
    KNOWN-GOOD, self-contained model ``program`` (one that ships its OWN ``golden_result``) through the
    FULL grading path — assemble (the model's assembler) → arc cosim → read back the output region — and
    compare the read-back result to that golden BIT-EXACT.

    ``arc_available`` + adapter routing only prove the oracle's pieces are wired; they do NOT prove the
    path computes the right answer. This does: if a real run cannot be graded to a correct verdict, the
    launcher must learn that BEFORE spending tokens (the atlas 0/11-at-\\$43 lesson). ``program`` is the
    caller's PARAMETER — the concrete known-good program name is a per-target SETUP fact declared in the
    descriptor, never a literal here (this module is HW-agnostic).

    Returns ``{ok, program, cycles, oracle, shape, mismatches, reason}``. Raises
    :class:`OracleUnavailable` when the model venv / cosim is absent or the program does not halt (so the
    caller decides NO_GO vs. a clean skip — never a silent pass)."""
    import numpy as np
    wd = Path(workdir)
    gwd, rwd = wd / "golden", wd / "run"
    gwd.mkdir(parents=True, exist_ok=True)
    rwd.mkdir(parents=True, exist_ok=True)
    # 1) the program's OWN golden — computed by the model's fp8/bf16 datapath (the INDEPENDENT reference,
    #    not the cosim), so a bit-exact match certifies the whole assemble→run→readback chain.
    gb = emit_bundle(model_ext=model_ext, program=program, fix_itype_rd=True, workdir=gwd, timeout=timeout)
    g = gb.get("golden")
    if not g:
        raise OracleUnavailable(
            f"{program!r}: model program ships no golden_result — cannot form a bit-exact smoke verdict")
    golden = _decode_output(base64.b64decode(g["b64"]), list(g["shape"]), str(g["dtype"]), None)
    # 2) the FULL oracle path: assemble → arc cosim → read back the declared output region.
    res = run_program_oracle(target, model_ext=model_ext, program=program, max_cycles=max_cycles,
                             workdir=rwd, timeout=timeout)
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
    return {"ok": ok, "program": program, "cycles": int(res.get("cycles") or 0),
            "oracle": res.get("oracle"), "shape": list(got.shape), "mismatches": mism, "reason": reason}


# --------------------------------------------------------------------------------------------------
# FAST FUNCTIONAL tier — the same assembled program run on the target model's high-level FUNCTIONAL
# core (pure Python, no arc .so), for per-round QA feedback. The cycle-exact cosim (above) stays the
# gold checkpoint. Only the RUNNER differs: emit_bundle (words+preload) and the output layout are shared.
# --------------------------------------------------------------------------------------------------

def _func_program_helper(target: str) -> Path:
    """The mlc FUNCTIONAL program runner file for ``target``, invoked BY PATH in the model venv (the runner
    imports only the target's own model package, never ``mlc.*``, so ``mlc.backends.__init__`` is not
    triggered in the subprocess). The filename is DERIVED from the target's registered cosim backend (mlc's
    ``cosim_<stem>`` -> ``func_program_<stem>``), mirroring how the cosim backend itself is derived — so NO
    target-name literal lives in merlin. Raises :class:`OracleUnavailable` if ``mlc_dir()`` is None, the
    target has no registered program backend, or the file is absent (fail closed)."""
    from merlin.targetgen.rtl import mlc_bridge
    d = mlc_bridge.mlc_dir()
    if d is None:
        raise OracleUnavailable(
            "mlc dir unavailable (set MERLIN_MLC_DIR) — no functional program runner")
    try:
        with mlc_bridge._mlc_cwd(), _mlc_importable(str(d)):
            from mlc.discover import fingerprint
            cosim = str(fingerprint.cosim_backend(mlc_bridge._arc_target(target)))  # e.g. "cosim_<stem>"
    except Exception as e:  # noqa: BLE001 — no registered program backend -> honestly unavailable
        raise OracleUnavailable(
            f"no functional program runner derivable for {target!r}: {type(e).__name__}: {e}")
    stem = cosim[len("cosim_"):] if cosim.startswith("cosim_") else cosim
    f = Path(d) / "mlc" / "backends" / f"func_program_{stem}.py"
    if not f.is_file():
        raise OracleUnavailable(f"mlc functional program runner absent: {f}")
    return f


def _run_func_helper(target: str, model_ext: str, req: dict, workdir: Path, timeout: int) -> dict[str, Any]:
    """Run the mlc functional program runner in the MODEL venv (same venv+cwd as ``_run_emit_helper``),
    marshalling a JSON+base64 request/result across the venv gap (mirrors its ``__main__`` CLI)."""
    py = _model_venv_python(model_ext)
    func = _func_program_helper(target)
    infile, outfile = workdir / "func_in.json", workdir / "func_out.json"
    infile.write_text(json.dumps(req))
    cmd = [str(py), str(func), "--in", str(infile), "--out", str(outfile)]
    cwd = ext_path(model_ext)                           # ASM_FOLDER is cwd-relative in the model
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise OracleUnavailable(
            f"functional program runner failed rc={p.returncode}: {p.stderr[-400:]}")
    return json.loads(outfile.read_text())


def run_program_functional_oracle(target: str, *, model_ext: str, cb: dict | None = None,
                                  kernel_s: Path | None = None, program: str | None = None,
                                  inputs: list[dict] | None = None, fix_itype_rd: bool = True,
                                  max_cycles: int = 20000, workdir, timeout: int = 600) -> dict[str, Any]:
    """Assemble (model venv / stock LLVM) → run on the target's FAST FUNCTIONAL core → read back. Mirrors
    :func:`run_program_oracle` (same ``emit_bundle`` words+preload, same output-layout resolution) but the
    ONLY difference is the runner: the target's mlc functional program runner (derived, invoked by file
    path in the model venv), instead of the cycle-exact arc cosim. Returns the SAME result shape
    ``{"outputs": {name: list}, "cycles": int, "oracle": f"{target}-functional"}``. Raises
    :class:`OracleUnavailable` if ``mlc_dir()`` / the func file / the model venv is absent (fail closed)."""
    _func_program_helper(target)                        # fail closed early if the runner is absent

    bundle = emit_bundle(model_ext=model_ext, program=program, kernel_s=kernel_s, inputs=inputs,
                         fix_itype_rd=fix_itype_rd, workdir=workdir, timeout=timeout)
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
    from .dram_facts import dram_base_for
    reloc_base = 0 if program else int(dram_base_for(target))
    req = {
        "words": [int(w) & 0xFFFFFFFF for w in words],
        "preload": [{"base": int(b), "b64": base64.b64encode(d).decode()} for b, d in preload],
        "out_bases": [[int(s["base"]), _out_nbytes(s)] for s in specs.values()],
        "max_cycles": int(max_cycles),
        "dram_base": reloc_base,
    }
    res = _run_func_helper(target, model_ext, req, Path(workdir), timeout)
    if not res.get("halted"):
        raise ProgramDidNotHalt(
            f"{target} program did not halt within {max_cycles} instructions (functional)")

    outmap = res.get("outputs") or {}
    outputs = {}
    for name, spec in specs.items():
        key = str(int(spec["base"]))                    # the CLI keys outputs by str(base)
        if key not in outmap:
            raise OracleUnavailable(
                f"{target}: functional runner returned no output {name!r} at base {spec['base']}")
        outputs[name] = _decode_output(base64.b64decode(outmap[key]), spec["shape"], spec["dtype"],
                                       spec["physical"]).tolist()
    return {"outputs": outputs, "cycles": int(res.get("cycles") or 0),
            "oracle": {"kind": f"{target}-functional", "derived_from_rtl": False,
                       "fidelity": "functional_model"}}


def program_functional_adapter(target: str, *, model_ext: str) -> Callable:
    """The FAST functional oracle adapter (``run(cb, fourth_text, workdir, timeout)`` shape) for an
    ``external_backend`` target — the per-round QA-loop correctness tier that mirrors
    :func:`program_oracle_adapter` but runs the emitted kernel on the functional core, not the cosim."""
    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        return run_program_functional_oracle(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                             workdir=wd, timeout=timeout)
    return run


# --------------------------------------------------------------------------------------------------
# LITE DEBUGGER — run the agent's kernel on the FUNCTIONAL core to instruction N, then read back the
# committed scalar state (PC, scalar registers, halt reason) + caller-named DRAM windows. This is the
# observability the agent lacks when its kernel assembles + halts but computes zeros: it can watch a
# region fill (or NOT) as the DMA/compute sequence advances, instead of only seeing a wrong final output.
#
# NOT oracle-free (it runs the program on the model), but it leaks NO golden: the functional core runs the
# AGENT'S kernel, so its DRAM holds only the canonical INPUTS (which the agent is given) + whatever the
# agent's own kernel wrote — never the reference golden (computed on a separate path). The ONE region whose
# post-run contents equal the golden IFF the kernel is correct is the declared OUTPUT region, so that
# window is REFUSED (a correct kernel would otherwise let the agent read its own answer back and, across
# capsules, scrape the answer key). Total dumped bytes are capped to prevent region-scraping.
# --------------------------------------------------------------------------------------------------

_DEBUG_MAX_DUMP_BYTES = 4096            # per-request cap on total dumped DRAM (anti-scrape)


def build_debug_cb(target: str, capsule_dir) -> dict:
    """A minimal command buffer (``tensors``: role/shape/dtype/base + input ``preload_b64``) for ONE
    capsule, built straight from the capsule spec — WITHOUT building the agent's package. Enough to run the
    agent's ``kernel.S`` on the functional model for debugging. Target-agnostic: bases from
    :func:`capsule_dram.layout`, preload from :func:`capsule_golden.canonical_input_raws`. The output tensor
    carries base/shape/dtype (used only to compute the redaction region) but NOT ``physical`` — the debugger
    never decodes the output, so the agent-declared output layout is irrelevant here."""
    from . import capsule_common, capsule_dram, capsule_golden
    from .dram_facts import dram_base_for
    cap = capsule_common.load_capsule(capsule_dir)
    tensors: dict[str, dict] = {}
    for t in (cap.get("inputs") or []):
        if t.get("role") == "output":
            continue                                        # the output tensor is added below (canonical)
        tensors[t["name"]] = {"role": t.get("role", "input"), "shape": list(t.get("shape") or []),
                              "dtype": t.get("dtype")}
    ot = capsule_dram.output_tensor(cap)
    if ot is None:
        raise OracleUnavailable(f"{target}: capsule {capsule_dir} declares no resolvable output tensor")
    tensors[ot["name"]] = {"role": "output", "shape": list(ot["shape"]), "dtype": ot["dtype"]}
    cb = {"tensors": tensors}
    capsule_dram.inject_bases(cb, cap, base=dram_base_for(target) + capsule_dram.DEFAULT_BASE)
    raws = capsule_golden.canonical_input_raws(cap, capsule_dir)
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
    for r in (dump_regions or []):
        b, n = int(r[0]), int(r[1])
        if n <= 0:
            rejected.append({"base": b, "nbytes": n, "reason": "non-positive length"})
        elif b < o1 and o0 < b + n:
            rejected.append({"base": b, "nbytes": n,
                             "reason": "overlaps the OUTPUT region — withheld (it would hold your result / "
                                       "the answer). Debug the INPUT and scratch regions instead."})
        elif total + n > _DEBUG_MAX_DUMP_BYTES:
            rejected.append({"base": b, "nbytes": n,
                             "reason": f"exceeds the per-request dump cap ({_DEBUG_MAX_DUMP_BYTES} B)"})
        else:
            total += n
            allowed.append((b, n))
    return allowed, rejected


def run_program_debug(target: str, *, model_ext: str, cb: dict, kernel_s: Path, dump_regions,
                      run_to: int | None = None, state_summary: bool = False, max_cycles: int = 20000,
                      workdir, timeout: int = 600) -> dict[str, Any]:
    """Assemble the agent's ``kernel.S`` and run it on the target's FUNCTIONAL core to instruction ``run_to``
    (or to halt when None), then return committed scalar state + REDACTED DRAM windows. Same assemble +
    preload + ``dram_base`` relocation as :func:`run_program_functional_oracle`; the output region is never
    read back (only used to reject overlapping dump windows). Raises :class:`OracleUnavailable` when the
    model venv / functional runner is absent (fail closed)."""
    _func_program_helper(target)                            # fail closed early if the runner is absent
    bundle = emit_bundle(model_ext=model_ext, kernel_s=kernel_s, fix_itype_rd=True, workdir=workdir,
                         timeout=timeout)
    words = bundle["words"]
    preload = _bundle_preload(bundle, cb)
    out_spec = _resolve_out_spec(target, cb, bundle)
    out_nbytes = _out_nbytes(out_spec)
    from .dram_facts import dram_base_for
    reloc_base = int(dram_base_for(target))
    allowed, rejected = _split_dump_regions(dump_regions, int(out_spec["base"]), out_nbytes)
    req = {
        "words": [int(w) & 0xFFFFFFFF for w in words],
        "preload": [{"base": int(b), "b64": base64.b64encode(d).decode()} for b, d in preload],
        "out_bases": [],                                    # the debugger never reads the output region
        "max_cycles": int(max_cycles),
        "dram_base": reloc_base,
        "run_to": None if run_to is None else int(run_to),
        "dump_regions": [[int(b), int(n)] for b, n in allowed],
        "state_summary": bool(state_summary),
    }
    res = _run_func_helper(target, model_ext, req, Path(workdir), timeout)
    dmap = res.get("dram_dumps") or {}
    regions = []
    for b, n in allowed:
        raw = base64.b64decode(dmap.get(str(int(b)), "")) if dmap.get(str(int(b))) else b""
        regions.append({"base": int(b), "nbytes": int(n), "returned_bytes": len(raw),
                        "hex": raw.hex(), "all_zero": (len(raw) > 0 and not any(raw))})
    return {
        "halted": bool(res.get("halted")),
        "halt_reason": res.get("halt_reason"),
        "instr_count": res.get("instr_count"),
        "cycles": res.get("cycles"),
        "pc": res.get("pc"),
        "regs": res.get("regs"),
        "program_words": len(words),
        "regions": regions,
        "rejected_regions": rejected,
        "on_chip": res.get("state_summary"),      # value-free populated-map (vmem/mrf/acc), or None
        "oracle": f"{target}-debug",
    }


# --------------------------------------------------------------------------------------------------
# COMMAND-BUFFER LITE DEBUGGER — the RoCC / command-buffer counterpart of run_program_debug (above).
# A self-hosted-ISA target ships a kernel.S we run to instruction N; a command-buffer target (gemmini
# RoCC, OPU) instead ships a COMMAND BUFFER, which we answer on the RTL-derived mlc arc model and read
# back the per-op HARDWARE-STATE effects (cycles + scratchpad/accumulator/DRAM-refill counts per command,
# plus the RTL fingerprint). This gives the agent the observability it lacked: watch its OWN intended
# computation's traffic on the compiled-from-RTL model, per command, instead of only a redacted verdict.
#
# INTEGRITY (same contract as run_program_debug): the model runs the AGENT'S OWN command buffer over the
# capsule's CANONICAL INPUTS (which the agent is given), so nothing here is a golden. Two things are
# REFUSED so the answer key never leaks: (1) the OUTPUT tensor values (mlc computes them; a correct cb
# would otherwise hand the agent its own answer, scrape-able across capsules), and (2) the `correct`
# verdict (that is the withheld oracle result the redacted self-check exists to keep back). Only the
# RTL-derived effect COUNTS (which carry no output value) and the fingerprint cross back. NOTE: this
# answers the command buffer (the intended computation) on the arc model — it is the complement of the
# static artifact decode (rocc_decode / divergence_localizer), which inspects the emitted .insn ENCODING;
# use both (state here, encoding there).
# --------------------------------------------------------------------------------------------------

_CB_DEBUG_REDACTED_KEYS = ("outputs", "correct", "reference", "elf", "console")


def _inject_canonical_inputs(target: str, cb: dict, capsule_dir) -> dict:
    """Return a COPY of ``cb`` with the capsule's canonical leaf operands attached as tensor ``data`` and
    resident handles resolved, so the mlc arc command-buffer runner can answer it. Target-agnostic: the
    leaves are the SAME deterministic operands the grader materializes (:func:`capsule_golden.
    materialize_capsule_leaves`) — the inputs the agent is given, never a golden. Resident-pack handles
    (a ``RES_PACK`` dst) are rewritten to their source tensor so the matmul operands name real tensors."""
    import copy
    import numpy as np
    from . import capsule_common, capsule_golden
    cb = copy.deepcopy(cb)
    cap = capsule_common.load_capsule(capsule_dir)
    leaves = capsule_golden.materialize_capsule_leaves(cap)
    tensors = cb.get("tensors") or {}
    for name, t in tensors.items():
        if name in leaves and t.get("role") in ("input", "weight", "bias"):
            shape = t.get("shape") or list(leaves[name].shape)
            t["data"] = np.asarray(leaves[name].to_list(), dtype=np.int64).reshape(shape).tolist()
    # resolve RES_PACK handles (dst produced from a real src tensor) in the matmul operands
    handle = {c.get("operands", {}).get("dst"): c.get("operands", {}).get("src")
              for c in cb.get("commands", []) if c.get("opcode") == "RES_PACK"}
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
    from merlin.targetgen.rtl import mlc_bridge
    if not mlc_bridge.arc_available(target):
        raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")
    prepared = _inject_canonical_inputs(target, cb, capsule_dir)
    # arc_run_command_buffer calls require_mlc(); make mlc importable for just this call (same context the
    # program cosim uses — never a global sys.path insert).
    with _mlc_importable(mlc_bridge.mlc_dir()):
        try:
            res = mlc_bridge.arc_run_command_buffer(prepared, target)
        except Exception as e:  # noqa: BLE001 — a run fault is surfaced, never a fabricated pass
            raise OracleUnavailable(f"{target} arc command-buffer run failed: {type(e).__name__}: {e}")
    # REDACT the answer-bearing keys; keep only the RTL-derived effect counts + provenance.
    safe = {k: v for k, v in res.items() if k not in _CB_DEBUG_REDACTED_KEYS}
    n_out = len((res.get("outputs") or {}))
    return {
        "halted": True,
        "metrics": safe.get("metrics"),
        "raw_metrics": safe.get("raw_metrics"),
        "per_command": safe.get("per_command"),        # per-op cycles + spad/acc/DRAM counts (no values)
        "oracle": safe.get("oracle"),
        "n_output_tensors": n_out,                      # count only — the VALUES are withheld (answer key)
        "redacted": list(_CB_DEBUG_REDACTED_KEYS),
        "note": ("Per-op HARDWARE STATE for YOUR command buffer on the RTL-derived arc model (cycles + "
                 "scratchpad/accumulator/DRAM-refill counts per command). The output VALUES and the pass/"
                 "fail verdict are withheld (answer key). This runs your INTENDED computation; to inspect "
                 "the emitted .insn ENCODING use the disassembler + instruction_trace.json."),
    }


# --------------------------------------------------------------------------------------------------
# CYCLE-ACCURATE VERILATOR tier (L4) — the SAME assembled program run on a program-driven Verilator
# sim of the target's RTL top (a bare-core TileLink harness), for RTL-CERTIFIED outputs. This is the
# first truly RTL-grounded atlas oracle: the arc cosim (L3) is the RTL-DERIVED functional gold; this
# runs the elaborated Verilog itself. Additive — arc (L3) stays the required tier. Only emit_bundle
# (words+preload) + the output-layout resolution are shared; the runner is the external Verilator sim
# resolved from the target's registered vsim dir (MERLIN_EXT_<TARGET>_VSIM), never a literal here.
# --------------------------------------------------------------------------------------------------

def _timing_block(res: Any) -> "tuple[list[dict] | None, dict | None]":
    """``(timing_observations, timing_capability)`` an oracle result carries, VALIDATED -- or
    ``(None, None)`` when the oracle has no timing capability at all.

    The rules are :mod:`merlin.perf.observations`'; this only routes. Two things it does not do, both
    load-bearing: it never invents the block for an oracle that did not emit one (an oracle with no
    timing capability contributes nothing -- not the key, not zeros), and it never drops a block that
    was emitted but failed validation. A refused block still travels, as a capability record naming
    what was refused, because "the producer emitted a block we could not believe" is a fact about the
    instrument that has to be visible.
    """
    from merlin.perf import observations as _OBS
    raw = res if isinstance(res, Mapping) else {
        k: getattr(res, k) for k in (_OBS.TIMING_OBSERVATIONS_KEY, _OBS.UNMEASURED_UNITS_KEY,
                                     _OBS.PARTITIONED_KEY, _OBS.ALIAS_COLLISIONS_KEY)
        if hasattr(res, k)}
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


#: Elaborated-RTL engines, each as (ext_path key suffix, conventional wrapper filename). Every engine
#: exposes the SAME ``run_program``; merlin never names a binary or a target. Adding an engine is one row
#: here plus a wrapper beside its build — the priority among them lives in ``rtl_engine_policy``.
#: The engine whose home layout -- and therefore whose build receipt / adoption record --
#: :mod:`gsim_emulator` owns. Named here because the lineage question is only answerable for the homes
#: that module lays out; every other engine is registered and judged by whoever built it.
_LINEAGE_ENGINE = "gsim"

#: The flavour this program-driven path can actually import and call. The other flavour
#: (a standalone binary taking a linked ELF) is a real build of the same engine that this
#: particular route cannot drive -- a distinction the probe must state rather than flatten.
_WRAPPER_FLAVOUR = "wrapper"

_RTL_ENGINES: dict[str, tuple[str, str]] = {
    "vcs": ("vcs", "vcs_run.py"),
    "gsim": ("gsim", "gsim_run.py"),
    "verilator": ("vsim", "verilator_run.py"),
}


def _rtl_engine_dir(target: str, engine: str) -> Path | None:
    """Where ``engine``'s build for ``target`` lives, or None when nothing registers one.

    Two sources, in precedence order:

    1. ``MERLIN_EXT_<TARGET>_<SUFFIX>`` (process env or ``.env``) — the machine-specific registration.
    2. The DERIVED home ``out/build/rtl_engines/<target>/<engine>/``.

    (2) is new and it is the point. Registration by env var alone means an engine that IS BUILT AND
    WORKING on this machine is invisible to the policy until somebody adds a line to a gitignored file
    — which is exactly what happened: a cycle-exact, 32x-faster GSIM engine sat beside its conventional
    wrapper, fully built, and every cert ran on Verilator because no ``MERLIN_EXT_<TARGET>_GSIM`` line
    existed. Nothing reported that; the engine was simply never considered. A derived home makes
    "install it where it belongs" the way to register an engine, and the env var the exception.
    """
    suffix, _ = _RTL_ENGINES[engine]
    try:
        return Path(ext_path(f"{target}_{suffix}"))
    except KeyError:
        pass
    from .gsim_emulator import engine_home
    derived = engine_home(target, engine)
    return derived if derived.is_dir() else None


def _rtl_engine_probe(target: str, engine: str):
    """``() -> (available, reason)`` for one engine: its dir must be registered AND hold its wrapper."""
    def probe():
        from .gsim_emulator import engine_home
        _, fname = _RTL_ENGINES[engine]
        d = _rtl_engine_dir(target, engine)
        if d is None:
            # BOTH places, named. "not registered" alone sent every reader to the env var and none of
            # them to the directory an engine can simply be installed into.
            return False, (f"no MERLIN_EXT_{target.upper()}_{_RTL_ENGINES[engine][0].upper()} "
                           f"registered and no build at {engine_home(target, engine)}")
        w = d / fname
        if not w.is_file():
            # PRESENCE IS THE HOME-LAYOUT MODULE'S FACT TOO, not just lineage. Statting one filename
            # made a home holding a BUILT engine of the other flavour report the engine as ABSENT --
            # the same two-modules-disagree defect as the blind probe and the bypassed lineage gate,
            # in a third place. The answer stays False, because it must: this path calls
            # ``run_program(words, preload, reads)`` on an imported wrapper, and the binary flavour
            # takes a LINKED ELF instead (``<emu> <elf> +loadmem=<elf> +max-cycles=N``). Assembling an
            # ELF out of program words is the compiler's job upstream, not a probe's, so a home
            # without the wrapper genuinely cannot be driven from here and passing it would trade a
            # clean unavailable for a crash inside the runner import.
            #
            # What changes is the REASON. "absent" sent every reader looking for a missing build;
            # naming the flavour that IS there sends them to the right question -- which is whether
            # this target's own backend drives that engine, as it does for the binary flavour today.
            if engine == _LINEAGE_ENGINE:
                from .gsim_emulator import resolve as _built
                r = _built(target)
                if r.ok and r.flavour and r.flavour != _WRAPPER_FLAVOUR:
                    return False, (
                        f"{engine} IS built for {target}, as the {r.flavour} flavour at {r.path} -- "
                        f"but NOT as {fname}. This program-driven path imports {fname} and calls "
                        f"run_program(words, preload, reads); the {r.flavour} flavour is driven with a "
                        f"linked ELF instead, so it cannot be run from assembled program words here. "
                        f"The engine is not missing: the target's own backend runs it.")
            return False, f"{fname} absent under {d}"
        # FINDING THE WRAPPER PROVES AN ENGINE IS BUILT, NOT THAT ITS BYTES MAY CERTIFY. The home-layout
        # module owns the lineage question for the homes it lays out, and asking it here is what stops
        # two modules disagreeing about one fact with the OPTIMISTIC one having the last word.
        #
        # Measured 2026-09-04: with MERLIN_GSIM_REQUIRE_RECEIPT=1 set, a target whose engine carries only
        # an adoption record -- lineage `adopted`, never built-and-bound -- had gsim_emulator.probe()
        # answer False and STILL certified on it, because this probe asked only whether a file existed.
        # A provenance gate that the selection path routes around is not a gate.
        #
        # Only asked of the engine whose home this module describes, and only when the dir IS that
        # derived home: an engine registered elsewhere, or laid out by someone else, is not this
        # module's to judge, and a refusal it did not author would be worse than none.
        if engine == _LINEAGE_ENGINE:
            from .gsim_emulator import resolve as _resolve
            r = _resolve(target)
            if r.refused:
                return False, r.reason
        return True, f"{fname} at {d}"
    return probe


def select_rtl_engine(target: str) -> dict:
    """Pick this target's elaborated-RTL engine (see :mod:`rtl_engine_policy`). Raises
    :class:`OracleUnavailable` when none can run, so the tier reports unavailable rather than silently
    resolving to a lower-fidelity oracle."""
    from . import rtl_engine_policy as _pol
    try:
        return _pol.select(target, {e: _rtl_engine_probe(target, e) for e in _RTL_ENGINES})
    except _pol.NoEngineAvailable as exc:
        raise OracleUnavailable(str(exc)) from exc


def _load_rtl_runner(engine_dir: Path, filename: str):
    """Import an engine dir's conventional wrapper (``run_program`` symmetric with
    ``cosim_atlas.run_program``) BY PATH — target-agnostic (merlin never names the sim binary). Raises
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


def run_program_verilator_oracle(target: str, *, model_ext: str, vsim_dir, cb: dict | None = None,
                                 kernel_s: Path | None = None, program: str | None = None,
                                 inputs: list[dict] | None = None, fix_itype_rd: bool = True,
                                 max_cycles: int = 20000, workdir, timeout: int = 600,
                                 per_cycle_csv: "str | Path | None" = None,
                                 engine: str = "verilator") -> dict[str, Any]:
    """Assemble (model venv / stock LLVM) → run on the target's program-driven Verilator RTL sim → read
    back. Mirrors :func:`run_program_oracle` (same ``emit_bundle`` words+preload, same output-layout
    resolution) but the runner is the external Verilator ``run_program`` instead of the arc cosim. The
    sim's TileLink DRAM slave masks addresses into its window exactly like the arc cosim's
    ``TileLinkSlave`` (both mirror ``cosim_atlas``), so the real cb DRAM base is passed straight through
    as the read region — no functional-tier ``dram_base`` relocation. Returns the SAME result shape
    ``{"outputs": {name: list}, "cycles": int, "oracle": f"{target}-verilator-rtl"}``. Raises
    :class:`OracleUnavailable` if the vsim / wrapper is absent, or the program does not halt."""
    runner = _load_rtl_runner(Path(vsim_dir), _RTL_ENGINES[engine][1])
    bundle = emit_bundle(model_ext=model_ext, program=program, kernel_s=kernel_s, inputs=inputs,
                         fix_itype_rd=fix_itype_rd, workdir=Path(workdir), timeout=timeout)
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
    except (TypeError, ValueError):                           # unintrospectable runner: skip the dump
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
    res = runner.run_program(words, preload=preload,
                             reads=[(int(s["base"]), _out_nbytes(s)) for s in specs.values()],
                             max_cycles=max_cycles, timeout=timeout, **_kw)
    if not res.get("halted"):
        raise ProgramDidNotHalt(f"{target} program did not halt within {max_cycles} cycles (verilator)")
    outs = res.get("outputs") or []
    if len(outs) < len(specs):
        raise OracleUnavailable(
            f"{target}: verilator oracle returned {len(outs)} region(s) for {len(specs)} declared "
            f"output(s) {sorted(specs)} — cannot read every result back")
    outputs = {}
    for raw, (name, spec) in zip(outs, specs.items()):
        outputs[name] = _decode_output(bytes(raw), spec["shape"], spec["dtype"],
                                       spec["physical"]).tolist()
    # The one tier here that runs the ELABORATED Verilog, so the only one entitled to say RTL.
    out: dict[str, Any] = {"outputs": outputs, "cycles": int(res.get("cycles") or 0),
                           "oracle": {"kind": f"{target}-{engine}-rtl", "derived_from_rtl": True,
                                      "fidelity": "elaborated_rtl", "engine": engine,
                                      "provenance": _engine_provenance(target, engine,
                                                                       Path(vsim_dir))}}
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
                _obs, _cap = _timing_block(dict(_folded, **{
                    _OBS_KEYS.ALIAS_COLLISIONS_KEY: res.get(_OBS_KEYS.ALIAS_COLLISIONS_KEY)}))
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


def _engine_provenance(target: str, engine: str,
                       engine_dir: "Path | None" = None) -> dict[str, Any]:
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
        from merlin.common import provenance as _prov
        binaries = {}
        for f in sorted(d.iterdir()):
            if f.is_file() and os.access(f, os.X_OK):
                binaries[f.name] = _prov.file_digest(f)
        if binaries:
            rec["binaries"] = binaries
        adoption = d / "provenance.json"
        if adoption.is_file():
            rec["adoption_record"] = {"path": str(adoption), "sha256": _prov.file_digest(adoption)}
        else:
            rec["lineage"] = ("no adoption record or build receipt beside this engine build — these "
                              "bytes are not attributed to any RTL revision")
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def program_verilator_adapter(target: str, *, model_ext: str) -> Callable | None:
    """The L4 cycle-accurate Verilator oracle adapter (``run(cb, fourth_text, workdir, timeout)`` shape)
    for an ``external_backend`` target — mirrors :func:`program_oracle_adapter` but runs the emitted
    kernel on the target's program-driven Verilator RTL sim. Returns ``None`` (no L4) when the target
    registers no vsim (``MERLIN_EXT_<TARGET>_VSIM`` unset) or its ``verilator_run.py`` is absent — so the
    tier is honestly ADDITIVE and target-agnostic (any external_backend target that ships a vsim gets it,
    with no target literal here)."""
    try:
        selection = select_rtl_engine(target)
    except OracleUnavailable:
        return None                       # no elaborated-RTL engine at all -> the tier is honestly absent
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
        res = run_program_verilator_oracle(target, model_ext=model_ext, vsim_dir=vsim_dir, cb=cb,
                                           engine=engine,
                                           kernel_s=ks, workdir=wd, timeout=timeout,
                                           max_cycles=derive_cycle_budget(cb), per_cycle_csv=_csv)
        # CARRY THE CHOICE, not just its outcome. `selection` records every engine considered and the
        # reason each slower/faster one was passed over. Without it a cert that ran on Verilator because
        # the fast engine was unregistered is indistinguishable from one that ran on Verilator because it
        # was the only engine there is — and the first is a thing to fix while the second is not.
        if isinstance(res.get("oracle"), dict):
            res["oracle"]["selection"] = dict(selection)
        return res
    return run
