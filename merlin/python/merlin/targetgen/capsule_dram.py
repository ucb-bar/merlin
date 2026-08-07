"""Deterministic DRAM address map for a capsule's tensors — harness-owned, target-agnostic.

An ``external_backend`` program oracle preloads each input at a DRAM base and reads the output back from
a base; the agent's emitted kernel must load/store at the SAME addresses. If the agent and the oracle
each pick addresses independently they diverge (the oracle preloads operand A at 0x0 but the kernel reads
it from wherever it guessed) — which is exactly why an atlas run graded 0/11 with the output tensor
carrying no base at all. So the HARNESS owns one canonical layout, computed as a PURE FUNCTION of the
capsule spec (tensor names/shapes/dtypes, known identically before the agent runs and at grade time), and
BOTH sides consume it:

  * it is surfaced to the agent in its task (the authoritative address map its kernel must target),
  * injected into the emitted command buffer's tensors before the oracle runs, and
  * used by the oracle to preload inputs and capture the output.

Nothing here is target-specific: no target name, no ISA, no agent output — just shape x dtype-size with
alignment, ordered by the capsule's declared tensors. Works for any external_backend target.
"""
from __future__ import annotations

from typing import Any

# Bytes per element per dtype token (capsule dtypes + the MLIR/torch spellings that show up in interfaces).
_DTYPE_BYTES: dict[str, int] = {
    "i8": 1, "int8": 1, "u8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "f8E4M3FN": 1, "f8E5M2": 1,
    "i16": 2, "int16": 2, "f16": 2, "float16": 2, "bf16": 2, "bfloat16": 2, "torch.bfloat16": 2,
    "i32": 4, "int32": 4, "torch.int32": 4, "f32": 4, "float32": 4,
    "i64": 8, "int64": 8, "f64": 8, "float64": 8,
    # microscaling (MX) block-float: the ELEMENT is whole-byte (its shared scale is stored per block,
    # separately) — an 8-bit mx element is 1 byte in the tensor's DRAM footprint.
    "mxfp8": 1, "mxint8": 1,
}

DEFAULT_BASE = 0x1000      # start above a small guard region (never address 0)
DEFAULT_ALIGN = 64         # 64-byte alignment is safe for any vector/tile datapath


def _canonical_dtype(dtype: str) -> str:
    """Fold equivalent width spellings to the table's canonical token so a byte-width lookup never crashes
    on a synonym: capsule specs write ``fp16``/``fp32`` where MLIR interfaces write ``f16``/``f32`` for the
    SAME width. Only the bare ``fp<N>`` family folds to ``f<N>`` (``fp8_e4m3``, ``bf16``, ``mx*`` keep their
    own explicit tokens). A DRAM layout must be robust to the spelling, not fail-closed on a synonym."""
    key = str(dtype).strip()
    if key.startswith("fp") and key[2:].isdigit():   # fp16 -> f16, fp32 -> f32, fp64 -> f64
        return "f" + key[2:]
    return key


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a capsule/interface dtype token. Raises on a genuinely unknown dtype
    (fail-closed — a silent wrong size would mis-place every following tensor); spelling synonyms
    (``fp16`` vs ``f16``) are folded first so they never trip that guard."""
    key = _canonical_dtype(dtype)
    if key not in _DTYPE_BYTES:
        raise KeyError(f"capsule_dram: unknown dtype {dtype!r}; add it to _DTYPE_BYTES")
    return _DTYPE_BYTES[key]


def tensor_nbytes(shape: list[int], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * dtype_bytes(dtype)


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def output_tensor(capsule: dict) -> dict[str, Any] | None:
    """The capsule's OUTPUT tensor spec ``{name, shape, dtype}`` — target-agnostic. Sources, in order:
    an ``inputs`` entry with ``role == "output"``; else the operation's ``out`` name + ``output_dtype``
    (shape resolved from the op, e.g. matmul ``[M, N]`` from the lhs/weight input shapes). Returns None if
    no output can be resolved (the oracle then raises an actionable error rather than crashing)."""
    for t in capsule.get("inputs", []) or []:
        if t.get("role") == "output":
            return {"name": t["name"], "shape": list(t["shape"]), "dtype": t["dtype"]}
    op = capsule.get("operation", {}) or {}
    attrs = op.get("attributes", {}) or {}
    name = attrs.get("out") or attrs.get("output") or "Y0"
    dtype = attrs.get("output_dtype") or (capsule.get("numeric_policy", {}) or {}).get("dtype", "bf16")
    by_name = {t["name"]: t for t in (capsule.get("inputs", []) or [])}
    shape = None
    if op.get("op") in ("matmul", "linear"):
        lhs = by_name.get(attrs.get("lhs"))
        rhs = by_name.get(attrs.get("weight") or attrs.get("rhs"))
        if lhs and rhs and len(lhs["shape"]) == 2 and len(rhs["shape"]) == 2:
            shape = [int(lhs["shape"][0]), int(rhs["shape"][1])]   # [M,K]x[K,N] -> [M,N]
    if shape is None:                                              # movement / elementwise: mirror an input
        ins = [t for t in (capsule.get("inputs", []) or []) if t.get("role") in ("input", "weight")]
        if ins:
            shape = list(ins[0]["shape"])
    if shape is None:
        return None
    return {"name": name, "shape": shape, "dtype": dtype}


def layout(capsule: dict, *, base: int = DEFAULT_BASE, align: int = DEFAULT_ALIGN) -> dict[str, int]:
    """Canonical ``{tensor_name: dram_base}`` for a capsule — a PURE function of the capsule spec, so the
    same map is produced when told to the agent and when grading. Order: the capsule's declared inputs
    (in listed order), then the output tensor. Each base is ``align``-ed; the output is placed after all
    inputs (its own size never affects an input's address). Deterministic across processes."""
    out: dict[str, int] = {}
    cur = int(base)
    for t in capsule.get("inputs", []) or []:
        if t.get("role") == "output":
            continue                                               # placed with the output below
        cur = _align_up(cur, align)
        out[t["name"]] = cur
        cur += tensor_nbytes(list(t["shape"]), t["dtype"])
    ot = output_tensor(capsule)
    if ot is not None:
        cur = _align_up(cur, align)
        out[ot["name"]] = cur
    return out


def inject_bases(cb: dict, capsule: dict, *, base: int = DEFAULT_BASE, align: int = DEFAULT_ALIGN) -> dict:
    """Fill in a canonical DRAM base for any command-buffer tensor that DIDN'T declare one, matched by
    name, from :func:`layout`. In-place on ``cb`` (also returned).

    The agent's kernel owns its memory map, so a ``base`` the agent DECLARED on a tensor is authoritative
    and left untouched (the oracle preloads inputs / reads the output at exactly the agent's addresses).
    This only supplies a deterministic default where the agent omitted one — so a partially-declaring (or
    non-declaring) submission still grades against a consistent layout instead of crashing on a missing
    base. A cb tensor whose name is not in the capsule layout is left as-is (the oracle raises an
    actionable error if a required base is still missing) — never a silent guess."""
    lay = layout(capsule, base=base, align=align)
    for name, t in (cb.get("tensors") or {}).items():
        if t.get("base") is None and name in lay:
            t["base"] = lay[name]
    return cb
