"""Shared helpers for reading a ``workload_region`` dict.

A workload region (see ``merlin/schemas/workload_region.schema.yaml``) is intentionally
loose: ``ops``, a ``tensors`` mapping, and optional ``reuse`` / ``op_sequence`` blocks. The
design-pressure metrics need a few derived facts from it (which tensor is the weight/RHS,
the matmul ``M``/``K``/``N``, dtype byte widths, whether there is a fused epilogue). This
module centralises that interpretation so every ``metric_*`` agrees on it.

Tensor roles are inferred from a small, documented heuristic (immutability, rank, and
name prefix) that is sufficient for the synthetic ``vla_action_chunk_decode`` region and the
``semantic_memory`` benchmarks. The MLIR-ingest path (``ingest/mlir_m2m.py``) classifies by
``m2m.*`` operand metadata instead and feeds the same downstream code.
"""
from __future__ import annotations

from typing import Any

# Byte width per element for the dtypes that appear in the benchmarks / synthetic regions.
_DTYPE_BYTES: dict[str, int] = {
    "i4": 1, "int4": 1,  # sub-byte packed; modelled as 1 byte for footprint purposes
    "i8": 1, "int8": 1, "u8": 1, "uint8": 1, "fp8": 1, "f8": 1, "f8e4m3fn": 1,
    "i16": 2, "int16": 2, "bf16": 2, "f16": 2, "fp16": 2, "half": 2,
    "i32": 4, "int32": 4, "f32": 4, "fp32": 4, "float32": 4,
    "i64": 8, "f64": 8,
}

# Ops that produce an accumulator (contraction ops).
CONTRACTION_OPS = ("matmul", "gemm", "gemv", "conv", "conv2d", "depthwise_conv")
# Epilogue ops that may run while the accumulator is still live.
EPILOGUE_OPS = ("bias_add", "bias", "requant", "dequant", "relu", "silu", "gelu",
                "activation", "add", "scale")


def dtype_bytes(dtype: str | None) -> int:
    """Byte width of one element of ``dtype`` (defaults to 1 for unknown dtypes)."""
    if not dtype:
        return 1
    return _DTYPE_BYTES.get(str(dtype).strip().lower(), 1)


def _shape(t: dict) -> list[int]:
    return list(t.get("shape", []) or [])


def _is_false(v: Any) -> bool:
    """True iff ``v`` denotes boolean false (handles bool and the string 'false')."""
    if isinstance(v, bool):
        return v is False
    return str(v).strip().lower() == "false"


def tensors(region: dict) -> dict[str, dict]:
    return dict(region.get("tensors", {}) or {})


def contraction_op(region: dict) -> str | None:
    """The primary contraction op name, or None if the region has none."""
    for op in region.get("ops", []) or []:
        if str(op).lower() in CONTRACTION_OPS:
            return str(op).lower()
    return None


def op_sequence(region: dict) -> list[str]:
    """The op sequence (explicit ``op_sequence`` if present, else ``ops``)."""
    seq = region.get("op_sequence") or region.get("ops") or []
    return [str(o).lower() for o in seq]


def has_epilogue(region: dict) -> bool:
    """True iff a contraction is followed by at least one epilogue op."""
    seq = op_sequence(region)
    seen_contraction = False
    for op in seq:
        if op in CONTRACTION_OPS:
            seen_contraction = True
        elif seen_contraction and op in EPILOGUE_OPS:
            return True
    # Fall back to: more than one op and at least one epilogue op present.
    return len(seq) > 1 and any(o in EPILOGUE_OPS for o in seq)


def classify_tensors(region: dict) -> dict[str, str | None]:
    """Return the role -> tensor-name mapping {lhs, rhs, bias, out}.

    Heuristic (sufficient for the benchmarks and the synthetic region):
      * bias  : a rank-1 immutable tensor (name often 'bias').
      * rhs   : an immutable (mutable:false) rank-2 tensor — the reused weight.
      * out   : a tensor named like the result ('Y'*) or, failing that, the remaining 2D one.
      * lhs   : the remaining rank-2 tensor (the activation / single-use operand).
    Name prefixes (A*/W*/Y*) break ties when immutability alone is ambiguous.
    """
    ts = tensors(region)
    roles: dict[str, str | None] = {"lhs": None, "rhs": None, "bias": None, "out": None}

    # bias: rank-1 tensor.
    for name, t in ts.items():
        if len(_shape(t)) == 1:
            roles["bias"] = name
            break

    rank2 = {n: t for n, t in ts.items() if len(_shape(t)) == 2}

    # rhs/weight: prefer an immutable, reused rank-2 tensor; tie-break by 'W' prefix.
    weight_candidates = [
        n for n, t in rank2.items()
        if _is_false(t.get("mutable")) and t.get("lifetime") != "single_use"
    ]
    if not weight_candidates:
        weight_candidates = [n for n, t in rank2.items() if _is_false(t.get("mutable"))]
    if weight_candidates:
        roles["rhs"] = sorted(
            weight_candidates, key=lambda n: (not n.upper().startswith("W"), n)
        )[0]

    # out: name starts with 'Y', else a single-use rank-2 tensor that is not the rhs.
    out = next((n for n in rank2 if n.upper().startswith("Y")), None)
    if out is None:
        out = next(
            (n for n, t in rank2.items()
             if n != roles["rhs"] and t.get("lifetime") == "single_use"
             and not n.upper().startswith("A")),
            None,
        )
    roles["out"] = out

    # lhs: remaining rank-2 tensor that is not rhs/out; tie-break by 'A' prefix.
    lhs_candidates = [n for n in rank2 if n not in (roles["rhs"], roles["out"])]
    if lhs_candidates:
        roles["lhs"] = sorted(
            lhs_candidates, key=lambda n: (not n.upper().startswith("A"), n)
        )[0]
    return roles


def mnk(region: dict) -> dict[str, int | None]:
    """Derive matmul (M, K, N) from the classified lhs/rhs shapes.

    For ``Y = A @ W`` with ``A:[M,K]``, ``W:[K,N]``: M = lhs[0], K = lhs[-1], N = rhs[-1].
    Returns Nones when the region has no recognisable matmul operands.
    """
    ts = tensors(region)
    roles = classify_tensors(region)
    lhs = ts.get(roles["lhs"]) if roles["lhs"] else None
    rhs = ts.get(roles["rhs"]) if roles["rhs"] else None
    M = K = N = None
    if lhs is not None and len(_shape(lhs)) == 2:
        M, K = _shape(lhs)[0], _shape(lhs)[-1]
    if rhs is not None and len(_shape(rhs)) == 2:
        # rhs is [K, N]; prefer rhs[-1] for N and rhs[0] for K if lhs was missing.
        if K is None:
            K = _shape(rhs)[0]
        N = _shape(rhs)[-1]
    return {"M": M, "K": K, "N": N}


def reuse_block(region: dict) -> dict:
    return dict(region.get("reuse", {}) or {})


def rhs_reuse_count(region: dict) -> int:
    """Reuse count of the weight/RHS across the region.

    Prefers the explicit ``reuse.rhs_reuse_count``; falls back to the rhs tensor's
    ``reuse_count``; defaults to 1 (no reuse).
    """
    rb = reuse_block(region)
    if "rhs_reuse_count" in rb:
        return int(rb["rhs_reuse_count"])
    roles = classify_tensors(region)
    if roles["rhs"]:
        t = tensors(region)[roles["rhs"]]
        if "reuse_count" in t:
            return int(t["reuse_count"])
    return 1


def rhs_mutable(region: dict) -> bool:
    """Whether the weight/RHS is mutable (defaults to False for an unmarked weight)."""
    rb = reuse_block(region)
    if "rhs_mutable" in rb:
        return not _is_false(rb["rhs_mutable"])
    roles = classify_tensors(region)
    if roles["rhs"]:
        return not _is_false(tensors(region)[roles["rhs"]].get("mutable", False))
    return False


def distinct_weights(region: dict) -> int:
    """Number of distinct resident weights competing for resident storage (default 1)."""
    rb = reuse_block(region)
    if "distinct_weights" in rb:
        return int(rb["distinct_weights"])
    return 1
