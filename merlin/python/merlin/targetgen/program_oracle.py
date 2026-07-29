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
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from merlin.common.paths import ext_path


class OracleUnavailable(RuntimeError):
    """Raised when the program oracle cannot run (model venv / cosim / arc artifacts absent)."""


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
    preload = [(int(x["base"]), base64.b64decode(x["b64"])) for x in bundle["inputs"]]
    # Capsule path: preload arc DRAM with the capsule's CANONICAL leaf inputs, attached to the cb's
    # leaf tensors by the grader (``preload_b64`` = the exact bytes the independent golden used — the
    # float target's operand palette, not the integer 0..3 fill). Base comes from the agent's cb tensor
    # (where its kernel reads the operand); the bytes are the ground-truth operands (see AW5).
    if not preload and cb:
        preload = _preload_from_cb(cb)

    modeling = mlc_bridge.mlc_dir()
    # mlc.backends.__init__ eagerly loads the gemmini cache cwd-relative — run inside mlc's cwd.
    with mlc_bridge._mlc_cwd():
        try:
            from mlc.discover import fingerprint
            from mlc.backends import cosim_atlas
            from mlc.backends.cosim_core import large_stack_call
        except Exception as e:  # noqa: BLE001
            raise OracleUnavailable(f"mlc program-cosim import failed: {type(e).__name__}: {e}")
        ap = fingerprint.artifact_paths(target, base=modeling)
        res = large_stack_call(cosim_atlas.run_program, str(ap["so"]), str(ap["man"]),
                               words, preload=preload, max_cycles=max_cycles)
    if not res.halted:
        raise OracleUnavailable(f"{target} program did not halt within {max_cycles} cycles")

    # resolve the output tensor spec from the cb (generation-declared) or the program's own golden.
    # The output DRAM base is the harness-owned address (stamped onto the cb by capsule_dram.inject_bases,
    # the same map the agent's kernel was told to store to) — read it defensively, exactly like the input
    # bases at :128. A missing base is an actionable grading error (the layout could not be applied), NOT
    # a bare KeyError that would surface as the opaque 'L3 crash: base'.
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

    import numpy as np
    nbytes = int(np.prod(out_spec["shape"])) * (2 if "16" in out_spec["dtype"] else
                                                (4 if "32" in out_spec["dtype"] else 1))
    raw = bytes(res.slave.captured(out_spec["base"], nbytes))
    logical = _decode_output(raw, out_spec["shape"], out_spec["dtype"], out_spec["physical"])

    oname = next((n for n, t in (cb.get("tensors") or {}).items()
                  if t.get("role") == "output"), "Y0") if cb else "Y0"
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
