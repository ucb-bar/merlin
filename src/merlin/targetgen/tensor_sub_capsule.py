"""The two-operand difference capsule: ``Y = X0 - X1`` over two int8 tensors, read out at accumulator width.

Written as a generic component program (``component_program.render``; the golden is the independent exact
integer evaluator ``component_numerics``). It is the standalone elementwise-map affine form with weights
``(1, -1)``: both operands are widened first, so no int8 value is negated at int8 width (``-(-128)`` is
representable in the accumulator dtype). The negative weight is a signed-int8 mesh weight, not a load-path
scale: the operand-sum load path saturates to int8 and is closed to negative multipliers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .component_program import SCHEMA_V2

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def tensor_sub_program(entry: dict, binding: CorpusBinding) -> dict:
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, N)):
        raise ValueError(f"{entry.get('name')}: tensor sub needs resolved positive M and N")
    lhs, rhs, out = entry.get("lhs", "A0"), entry.get("rhs", "A1"), entry.get("out", "Y0")
    return {
        "schema": SCHEMA_V2,
        "inputs": [
            {"name": lhs, "role": "input", "shape": [M, N], "dtype": "operand"},
            {"name": rhs, "role": "input", "shape": [M, N], "dtype": "operand"},
        ],
        "nodes": [{"name": "difference", "op": "sub", "inputs": [lhs, rhs], "dtype": "accumulator"}],
        "outputs": [{"name": out, "value": "difference"}],
    }


def build_tensor_sub(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A standalone two-operand difference (op == tensor_sub), written as a component program."""
    from .corpus_spec import build_component_program

    cap, source = build_component_program({**entry, "program": tensor_sub_program(entry, binding)}, binding)
    cap["expected"]["instruction_classes"] = binding.classes_for(
        op="tensor_sub", output_dtype=binding.cap_dtype(binding.accum_dtype)
    )
    return cap, source
