"""Capture bundles for the RVV whole-model lane, and the scalar datatype a bundle's IR carries.

``_bundle_dir`` resolves the bundle ``merlin-compile`` compiles, ``_capture_python`` picks the interpreter a
model is captured with, and ``ir_scalar_dtype`` reads the element type the captured ``model.mlir``
actually carries -- which, not the bundle's name, decides the scalar datapath.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bundle_dir(workload: str, dtype: str):
    """The capture bundle `merlin-compile` uses, preferring the FULL-FIDELITY recapture.

    Delegates to ``baselines.bundle.resolve``, which prefers ``<w>_<dtype>_full`` (the real/native
    architecture) and falls back to ``<w>_<dtype>_consistent``. This used to hard-code the
    ``_consistent`` suffix, which for tiny_llama is a 2-layer RANDOM-INIT stand-in rather than the
    real 22-layer TinyLlama-1.1B — so `merlin-compile` silently compiled a toy while every baseline
    arm (which already goes through ``resolve``) compiled the real model.
    """
    from ..baselines.bundle import resolve
    return resolve(workload, dtype).root


def ir_scalar_dtype(bundle: "Path") -> str | None:
    """The dtype the bundle's IR ACTUALLY carries, read off ``model.mlir`` — the fact that decides which
    scalar/RVV datapath is correct.

    A bundle's NAME states how the model was quantized; it does not state what the compiled IR contains.
    A weight-only fake-quant capture stores fp8/int8 weights but emits f32 tensors end to end, so the
    correct scalar package for it is the f32 one — choosing by the name instead asks for a datapath the IR
    has no operands for. Returns ``None`` when no known element type dominates, so the caller fails closed
    rather than guessing.

    Selection is by PRESENCE of the narrowest datapath, not by majority. Every weight-only capture is
    mostly f32 by count -- an int8 bundle measures 46 i8 tensors against 1190 f32 ones -- so a majority vote
    hands back f32 and silently drops the int8 datapath the i8 operands require. One tensor of a narrow type
    means the narrow datapath is needed; only a bundle with none of them is an f32 model.

    Structural, not pattern-matched: it counts occurrences of each known tensor element-type spelling.
    (No regex — see the repo's no-regex rule.)"""
    mlir = bundle / "model.mlir"
    if not mlir.is_file():
        return None
    text = mlir.read_text(encoding="utf-8", errors="replace")
    # spelling in the IR -> the --dtype token that selects its datapath, narrowest datapath first
    present = [tok for tok in _IR_ELEMENT_ORDER
               if text.count(f"x{_IR_ELEMENT_SPELLING[tok]}>")]
    return present[0] if present else None


def _capture_python(m2m_dir: Path, workload: str) -> Path:
    """The interpreter to capture ``workload`` with: its own venv when its capture.toml names one.

    A workload's upstream stack is pinned per model (model2MLIR's `capture.toml [venv]`), so the
    capture must not run under merlin's interpreter just because that is what is executing.
    Falls back to model2MLIR's own venv, then to this interpreter, and never fails here — the
    caller surfaces a capture failure with the worker's output attached.
    """
    toml = m2m_dir / "workloads" / workload / "capture.toml"
    if toml.is_file():
        try:
            import tomllib
            cfg = tomllib.loads(toml.read_text())
        except Exception:                                          # noqa: BLE001
            cfg = {}
        venv = cfg.get("venv")
        if venv:
            cand = Path(venv)
            if not cand.is_absolute():
                cand = m2m_dir / "workloads" / workload / venv
            if (cand / "bin" / "python").is_file():
                return cand / "bin" / "python"
    fallback = m2m_dir / ".venv" / "bin" / "python"
    return fallback if fallback.is_file() else Path(sys.executable)

#: MLIR tensor element-type spelling for each ``--dtype`` token, used to read a bundle's ACTUAL IR dtype
#: back off ``model.mlir`` (:func:`ir_scalar_dtype`). Keys must stay a subset of ``_DTYPE_STRATEGY``.
_IR_ELEMENT_SPELLING = {"int8": "i8", "fp32": "f32", "fp16": "f16", "bf16": "bf16"}
#: narrowest datapath first — the first spelling PRESENT in the IR decides (see :func:`ir_scalar_dtype`).
_IR_ELEMENT_ORDER = ("int8", "fp16", "bf16", "fp32")
