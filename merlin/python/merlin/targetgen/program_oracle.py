"""Generic ``external_backend`` PROGRAM oracle — run an agent-emitted target-ISA program on the target's
own cosim, using the target's OWN assembler. The counterpart to the ``command_buffer`` oracle
(``mlc_bridge.arc_run_command_buffer``, gemmini/OPU) for accelerators that are a self-hosted ISA core
(their deliverable is an assembled program, e.g. atlas ``kernel.S`` → IMEM words), NOT a RoCC ``.insn``
host stream and NOT an ISA-less command buffer.

Design (HW-agnostic; nothing atlas-specific is hardcoded here):
  * the ISA / assembler comes from the MODEL project (via :mod:`oracle_helpers.npu_emit`, run in the
    model's own venv as a subprocess) — merlin holds no opcode table;
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


def emit_bundle(*, model_ext: str, program: str | None = None, kernel_s: Path | None = None,
                inputs: list[dict] | None = None, fix_itype_rd: bool, workdir: Path,
                timeout: int) -> dict[str, Any]:
    """Assemble + lay out DRAM bytes in the MODEL's venv (subprocess). Returns the JSON bundle
    {words, inputs:[{base,b64}], output, golden}. ``program`` (self-contained) XOR ``kernel_s``+``inputs``."""
    py = _model_venv_python(model_ext)
    out = workdir / "npu_bundle.json"
    cmd: list[str] = [str(py), str(_EMIT_HELPER), "--out", str(out)]
    if fix_itype_rd:
        cmd.append("--fix-itype-rd")
    if program:
        cmd += ["--program", program]
    else:
        if kernel_s is None:
            raise ValueError("emit_bundle: need program or kernel_s")
        cmd += ["--kernel-s", str(kernel_s), "--inputs", json.dumps(inputs or [])]
    cwd = ext_path(model_ext)                          # ASM_FOLDER is cwd-relative in the model
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise OracleUnavailable(f"model assembler failed rc={p.returncode}: {p.stderr[-400:]}")
    return json.loads(out.read_text())


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
    out_spec = None
    if cb:
        outs = [t for t in (cb.get("tensors") or {}).values() if t.get("role") == "output"]
        if outs:
            t = outs[0]
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
