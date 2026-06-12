"""Synthetic VLA action-chunk decode region (the M1 experiment driver).

Models the inner loop of a vision-language-action policy's action head:

    for h in 1..H:                       # action-chunk horizon
        Y_h = relu(requant(A_h @ W + bias))

with batch=1, small GEMV/GEMM, a reused immutable weight ``W``, and a quantized epilogue.
Because the action-chunk loop lives host-side (it is not captured by single-pass model export
such as model2MLIR), this synthetic builder is the experiment driver: it gives full, honest
control over the axes the thesis sweeps — horizon ``H``, weight reuse, contraction depth ``K``,
dtype, and epilogue presence.

``H`` and ``reuse_count`` are independent axes (default ``reuse_count == H``) so the phase
transition can separate "more steps" from "more reuse of the same W".
"""
from __future__ import annotations

from merlin.common.yaml import dump_yaml

_OUT_DTYPE = {"i8": "i8", "fp8": "fp8", "bf16": "bf16"}


def build_region(H: int = 16, reuse_count: int | None = None, dtype: str = "i8",
                 epilogue: bool = True, K: int = 256, M: int = 1, N: int = 256,
                 distinct_weights: int = 1, name: str = "vla_action_chunk_decode") -> dict:
    """Build a ``workload_region``-schema dict for the action-chunk decode region.

    Args:
      H: action-chunk horizon (number of decode steps).
      reuse_count: how many steps reuse the same W (defaults to H).
      dtype: operand dtype for A/W/Y (i8, fp8, bf16).
      epilogue: whether a bias->requant->relu epilogue follows the matmul.
      K, M, N: matmul dims for ``A:[M,K] @ W:[K,N]``.
      distinct_weights: number of distinct resident weights competing for resident storage.
    """
    reuse = H if reuse_count is None else reuse_count
    out_dtype = _OUT_DTYPE.get(dtype, "i8")
    ops = ["matmul"] + (["bias_add", "requant", "relu"] if epilogue else [])

    tensors = {
        "A": {"shape": [M, K], "dtype": dtype, "lifetime": "single_use"},
        "W": {"shape": [K, N], "dtype": dtype, "lifetime": "reused_across_region",
              "reuse_count": reuse, "mutable": False},
        "Y": {"shape": [M, N], "dtype": out_dtype, "lifetime": "single_use"},
    }
    if epilogue:
        tensors["bias"] = {"shape": [N], "dtype": "i32", "mutable": False}

    region = {
        "name": name,
        "description": (
            "Synthetic VLA action-chunk decode: reused immutable weight, small-batch "
            "GEMV/GEMM, quantized epilogue, repeated over the action horizon."),
        "ops": ops,
        "region": {
            "loop": f"h in 1..{H}",
            "body": ("Y_h = relu(requant(A_h @ W + bias))" if epilogue
                     else "Y_h = A_h @ W"),
        },
        "tensors": tensors,
        "op_sequence": list(ops),
        "reuse": {
            "rhs_reuse_count": reuse,
            "rhs_mutable": False,
            "distinct_weights": distinct_weights,
        },
        "parameters": {"H": H, "K": K, "dtype": dtype, "epilogue": epilogue},
    }
    return region


def sweep_axes() -> dict:
    """Canonical M1 sweep grid for the phase-transition experiment."""
    return {
        "H": [1, 2, 4, 8, 16, 32],
        "reuse_count": [1, 2, 4, 8, 16],
        "dtype": ["i8", "fp8", "bf16"],
        "epilogue": [True, False],
    }


def to_yaml(region: dict) -> str:
    """Deterministic YAML for a built region (for materializing a golden benchmark)."""
    return dump_yaml(region)
