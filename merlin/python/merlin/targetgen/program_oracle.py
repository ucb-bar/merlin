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
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from merlin.common.paths import ext_path


class OracleUnavailable(RuntimeError):
    """Raised when the program oracle cannot run (model venv / cosim / arc artifacts absent)."""


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


def _assemble_kernel_words(kernel_s: Path, workdir: Path) -> list[int]:
    """Assemble an agent-emitted ``.word``/``.insn`` kernel into little-endian IMEM u32 words using the
    PREBUILT stock LLVM (``llvm-mc`` + ``llvm-objcopy`` from the MLIR install; override via
    ``MERLIN_MLIR_INSTALL``). Target-agnostic: the encoding lives in the emitted directives (grounded on
    the target's shipped ISA definition), so merlin needs no per-target assembler and holds no opcode
    table. ``llvm-mc`` accepts ``#``/``//`` comments + labels natively (labels emit no ``.text`` bytes).
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
    words = [int(w) for w in np.frombuffer(binf.read_bytes(), dtype="<u4")]
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


def _resolve_out_spec(target: str, cb: dict | None, bundle: dict) -> dict:
    """The output tensor spec ``{base, shape, dtype, physical}`` — from the cb (generation-declared) or
    the program's own golden. The output DRAM base is the harness-owned address (stamped by
    ``capsule_dram.inject_bases``, the same map the agent's kernel was told to store to). A missing base is
    an actionable grading error (the layout could not be applied), NOT a bare ``KeyError``. Shared by the
    cosim and functional runners so both read the result from the same physical layout."""
    out_spec = None
    if cb:
        outs = [t for t in (cb.get("tensors") or {}).values() if t.get("role") == "output"]
        if outs:
            t = outs[0]
            if t.get("base") is None:
                raise OracleUnavailable(
                    f"{target}: output tensor has no DRAM base — the harness DRAM layout was not applied "
                    f"(capsule_dram.inject_bases); cannot read the result back")
            out_spec = {"base": int(t["base"]), "shape": list(t["shape"]),
                        "dtype": t.get("dtype", "bf16"), "physical": t.get("physical")}
    if out_spec is None and bundle.get("output"):
        o = bundle["output"]
        out_spec = {"base": int(o["base"]), "shape": list(o["shape"]),
                    "dtype": o["dtype"], "physical": o.get("physical")}
    if out_spec is None:
        raise OracleUnavailable(f"{target}: no output tensor declared in cb or program golden")
    return out_spec


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
            backend_name = fingerprint.cosim_backend(target)
            backend = importlib.import_module(f"mlc.backends.{backend_name}")
        except Exception as e:  # noqa: BLE001
            raise OracleUnavailable(f"mlc program-cosim import failed: {type(e).__name__}: {e}")
        if not hasattr(backend, "run_program"):
            raise OracleUnavailable(
                f"mlc backend mlc.backends.{backend_name} for target {target!r} exposes no run_program "
                f"(not a self-hosted-ISA program cosim)")
        ap = fingerprint.artifact_paths(target, base=modeling)
        res = large_stack_call(backend.run_program, str(ap["so"]), str(ap["man"]),
                               words, preload=preload, max_cycles=max_cycles)
    if not res.halted:
        raise OracleUnavailable(f"{target} program did not halt within {max_cycles} cycles")

    # resolve the output tensor spec from the cb (generation-declared) or the program's own golden.
    out_spec = _resolve_out_spec(target, cb, bundle)
    nbytes = _out_nbytes(out_spec)
    raw = bytes(res.slave.captured(out_spec["base"], nbytes))
    logical = _decode_output(raw, out_spec["shape"], out_spec["dtype"], out_spec["physical"])

    oname = _output_name(cb)
    return {"outputs": {oname: logical.tolist()}, "cycles": int(res.cycles),
            "oracle": f"{target}-arc-arcilator-cosim"}


def program_oracle_adapter(target: str, *, model_ext: str) -> Callable:
    """An oracle adapter (the ``run(cb, fourth_text, workdir, timeout)`` shape ``capsule_runner`` expects)
    for an ``external_backend`` target. ``fourth_text`` is the agent's emitted ``kernel.S``."""
    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        return run_program_oracle(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                  workdir=wd, timeout=timeout)
    return run


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
            cosim = str(fingerprint.cosim_backend(target))     # e.g. "cosim_<stem>"
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
    out_spec = _resolve_out_spec(target, cb, bundle)
    nbytes = _out_nbytes(out_spec)

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
        "out_bases": [[int(out_spec["base"]), int(nbytes)]],
        "max_cycles": int(max_cycles),
        "dram_base": reloc_base,
    }
    res = _run_func_helper(target, model_ext, req, Path(workdir), timeout)
    if not res.get("halted"):
        raise OracleUnavailable(
            f"{target} program did not halt within {max_cycles} instructions (functional)")

    outmap = res.get("outputs") or {}
    key = str(int(out_spec["base"]))                    # the CLI keys outputs by str(base)
    if key not in outmap:
        raise OracleUnavailable(
            f"{target}: functional runner returned no output at base {out_spec['base']}")
    raw = base64.b64decode(outmap[key])
    logical = _decode_output(raw, out_spec["shape"], out_spec["dtype"], out_spec["physical"])
    return {"outputs": {_output_name(cb): logical.tolist()}, "cycles": int(res.get("cycles") or 0),
            "oracle": f"{target}-functional"}


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
            res = mlc_bridge.arc_run_command_buffer(prepared)
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

def _load_verilator_runner(vsim_dir: Path):
    """Import the vsim dir's conventional ``verilator_run.py`` (``run_program`` symmetric with
    ``cosim_atlas.run_program``) BY PATH — target-agnostic (merlin never names the sim binary). Raises
    :class:`OracleUnavailable` if the wrapper is absent / exposes no ``run_program``."""
    wrapper = Path(vsim_dir) / "verilator_run.py"
    if not wrapper.is_file():
        raise OracleUnavailable(f"verilator runner absent: {wrapper} (build the vsim + verilator_run.py)")
    spec = importlib.util.spec_from_file_location("_merlin_verilator_run", wrapper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_program"):
        raise OracleUnavailable(f"verilator runner {wrapper} exposes no run_program")
    return mod


def run_program_verilator_oracle(target: str, *, model_ext: str, vsim_dir, cb: dict | None = None,
                                 kernel_s: Path | None = None, program: str | None = None,
                                 inputs: list[dict] | None = None, fix_itype_rd: bool = True,
                                 max_cycles: int = 20000, workdir, timeout: int = 600) -> dict[str, Any]:
    """Assemble (model venv / stock LLVM) → run on the target's program-driven Verilator RTL sim → read
    back. Mirrors :func:`run_program_oracle` (same ``emit_bundle`` words+preload, same output-layout
    resolution) but the runner is the external Verilator ``run_program`` instead of the arc cosim. The
    sim's TileLink DRAM slave masks addresses into its window exactly like the arc cosim's
    ``TileLinkSlave`` (both mirror ``cosim_atlas``), so the real cb DRAM base is passed straight through
    as the read region — no functional-tier ``dram_base`` relocation. Returns the SAME result shape
    ``{"outputs": {name: list}, "cycles": int, "oracle": f"{target}-verilator-rtl"}``. Raises
    :class:`OracleUnavailable` if the vsim / wrapper is absent, or the program does not halt."""
    runner = _load_verilator_runner(vsim_dir)
    bundle = emit_bundle(model_ext=model_ext, program=program, kernel_s=kernel_s, inputs=inputs,
                         fix_itype_rd=fix_itype_rd, workdir=Path(workdir), timeout=timeout)
    words = bundle["words"]
    preload = _bundle_preload(bundle, cb)
    out_spec = _resolve_out_spec(target, cb, bundle)
    nbytes = _out_nbytes(out_spec)
    res = runner.run_program(words, preload=preload,
                             reads=[(int(out_spec["base"]), int(nbytes))],
                             max_cycles=max_cycles, timeout=timeout)
    if not res.get("halted"):
        raise OracleUnavailable(f"{target} program did not halt within {max_cycles} cycles (verilator)")
    outs = res.get("outputs") or []
    if not outs:
        raise OracleUnavailable(f"{target}: verilator oracle returned no output at base {out_spec['base']}")
    raw = bytes(outs[0])
    logical = _decode_output(raw, out_spec["shape"], out_spec["dtype"], out_spec["physical"])
    return {"outputs": {_output_name(cb): logical.tolist()}, "cycles": int(res.get("cycles") or 0),
            "oracle": f"{target}-verilator-rtl"}


def program_verilator_adapter(target: str, *, model_ext: str) -> Callable | None:
    """The L4 cycle-accurate Verilator oracle adapter (``run(cb, fourth_text, workdir, timeout)`` shape)
    for an ``external_backend`` target — mirrors :func:`program_oracle_adapter` but runs the emitted
    kernel on the target's program-driven Verilator RTL sim. Returns ``None`` (no L4) when the target
    registers no vsim (``MERLIN_EXT_<TARGET>_VSIM`` unset) or its ``verilator_run.py`` is absent — so the
    tier is honestly ADDITIVE and target-agnostic (any external_backend target that ships a vsim gets it,
    with no target literal here)."""
    try:
        vsim_dir = ext_path(f"{target}_vsim")
    except KeyError:
        return None
    if not (Path(vsim_dir) / "verilator_run.py").is_file():
        return None

    def run(cb, fourth_text, workdir, timeout):
        wd = Path(workdir)
        ks = wd / "kernel.S"
        if fourth_text:
            ks.write_text(fourth_text)
        return run_program_verilator_oracle(target, model_ext=model_ext, vsim_dir=vsim_dir, cb=cb,
                                            kernel_s=ks, workdir=wd, timeout=timeout)
    return run
