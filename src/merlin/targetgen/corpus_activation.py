"""Build the typed source DAG for runtime activation matrix products."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def activation_matmul_program(entry: dict, binding: CorpusBinding) -> dict:
    """The typed DAG of an activation x activation contraction whose result feeds a reshape.

    Both operands are runtime ``input`` tensors in the operand dtype: no stored weight, so nothing can
    be packed or made resident ahead of the call, and no scale stream, because a host epilogue applies
    the scales. The product accumulates in the accumulator dtype and is published through a row-major
    reshape to ``[N, M]``, so the contraction's consumer is data movement rather than a readout stage.
    """
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, K, N)):
        raise ValueError(f"{entry.get('name')}: activation matmul needs resolved positive M, K and N")
    from .component_program import SCHEMA_V2

    lhs, rhs, out = entry.get("lhs", "A0"), entry.get("rhs", "B0"), entry.get("out", "Y0")
    return {
        "schema": SCHEMA_V2,
        "inputs": [
            {"name": lhs, "role": "input", "shape": [M, K], "dtype": "operand"},
            {"name": rhs, "role": "input", "shape": [K, N], "dtype": "operand"},
        ],
        "nodes": [
            {"name": "product", "op": "matmul", "inputs": [lhs, rhs]},
            {"name": "moved", "op": "reshape", "inputs": ["product"], "shape": [N, M]},
        ],
        "outputs": [{"name": out, "value": "moved"}],
    }
