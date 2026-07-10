"""Extract a workload region from model2MLIR (linalg-on-tensors) MLIR.

A *validation* path, not the experiment driver: it proves the pressure pass extracts the same
per-invocation facts (op, shapes, dtypes, epilogue) from real VLA model MLIR as from the
synthetic builder. The action-chunk loop's H-fold reuse is host-side (not in single-pass model
export), so ``H`` is injected as the host-loop multiplier — documented, not IR-derived.

xDSL-gated: requires ``uv sync --extra xdsl``. Parses with the standard dialects loaded
(builtin/func/linalg/tensor/arith/math/scf/cf) and ``allow_unregistered`` for stray ops. The
quantized exports use model2MLIR's custom ``quant_ext`` dialect and need its own loader; the
fp32 exports parse with stock xDSL, so the parity path targets those.
"""
from __future__ import annotations

from merlin.common import mlir_query


def available() -> bool:
    """True iff xDSL is importable."""
    try:
        import xdsl  # noqa: F401
        return True
    except Exception:
        return False


def _parse_module(mlir_text: str):
    """Parse a model2MLIR linalg-on-tensors module for DSE analysis.

    ONE grounded ingestion point: delegate to :func:`merlin.common.mlir_query.parse` (which uses the
    inference frontend's xDSL parser — Builtin/Func/Arith/Linalg/Tensor/Scf/Math/Cf,
    ``allow_unregistered``, plus the ``} -> (T1,T2)`` normalizer), so inference and DSE never drift."""
    return mlir_query.parse(mlir_text)


def _attr(op, key: str) -> str | None:
    return mlir_query.attr_str(op, key)


def _shape_dtype(t) -> tuple[list[int], str]:
    """``([shape], dtype)`` for a tensor **type object** (regex-free; delegates to mlir_query). Pass
    ``op.operands[i].type`` directly — never a stringified type."""
    return mlir_query.type_shape_dtype(t)


def region_from_mlir(mlir_path: str, region_id: str | None = None, H: int = 8) -> dict:
    """Build a ``workload_region`` dict from one matmul region of a model2MLIR file.

    Picks the matmul named ``region_id`` (e.g. 'matmul_0'), or the first contraction op.
    ``H`` is the injected host-loop reuse (the IR captures a single forward pass).
    """
    module = _parse_module(open(mlir_path, encoding="utf-8").read())
    matmuls = [op for op in module.walk() if op.name == "linalg.matmul"]
    if not matmuls:
        raise ValueError(f"no linalg.matmul ops in {mlir_path}")
    op = None
    if region_id is not None:
        op = next((o for o in matmuls if _attr(o, "m2m.region_id") == region_id), None)
    op = op or matmuls[0]

    lhs_shape, lhs_dt = mlir_query.type_shape_dtype(op.operands[0].type)
    rhs_shape, rhs_dt = mlir_query.type_shape_dtype(op.operands[1].type)
    out_shape, out_dt = mlir_query.type_shape_dtype(op.results[0].type)
    if not (lhs_shape and rhs_shape):
        raise ValueError("could not read matmul operand shapes")

    # addmm = matmul + bias -> an epilogue is present.
    m2m_op = _attr(op, "m2m.op") or "matmul"
    epilogue = m2m_op == "addmm"
    ops = ["matmul"] + (["bias_add"] if epilogue else [])

    region_name = _attr(op, "m2m.region_id") or "mlir_matmul"
    tensors = {
        "A": {"shape": lhs_shape, "dtype": lhs_dt, "lifetime": "single_use"},
        "W": {"shape": rhs_shape, "dtype": rhs_dt, "mutable": False,
              "lifetime": "reused_across_region", "reuse_count": H},
        "Y": {"shape": out_shape or [lhs_shape[0], rhs_shape[-1]], "dtype": out_dt or rhs_dt,
              "lifetime": "single_use"},
    }
    if epilogue:
        tensors["bias"] = {"shape": [rhs_shape[-1]], "dtype": "i32", "mutable": False}

    region = {
        "name": f"mlir_{region_name}",
        "description": f"Extracted from {mlir_path} region {region_name} (H={H} injected).",
        "ops": ops,
        "op_sequence": list(ops),
        "tensors": tensors,
        "reuse": {"rhs_reuse_count": H, "rhs_mutable": False, "distinct_weights": 1},
        "provenance": {"source": "model2MLIR", "file": mlir_path, "m2m_op": m2m_op,
                       "host_loop_H": H},
    }
    from merlin.common.schemas import validate_or_raise
    validate_or_raise(region, "workload_region")   # schemas/ rule: code validates what lives here
    return region
