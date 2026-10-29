"""The scalar-immediate affine capsule: ``y = weight * x + bias`` over one operand, literals as immediates.

Written as a generic component program (``component_program.render``; the golden is the independent exact
integer evaluator ``component_numerics``). The immediate operand is a LITERAL in the loop body, so the
written region has exactly one tensor operand and an affine body: the standalone elementwise-map form the
contract states (``out = sat_int8(RNE(s_o * act(sum_i w_i * x_i + c)))`` with one broadcast constant c).
``weight`` is a signed int8 (a mesh / load-path multiplier); ``bias`` is the int32 accumulator-seed
constant. ``cast: true`` states the identity map instead (the widening cast). The result is read out at accumulator width (the identity map read at accumulator width is the
int8 -> int32 widening), so the comparison is exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .component_program import SCHEMA_V2

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def scalar_affine_program(entry: dict, binding: CorpusBinding) -> dict:
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, N)):
        raise ValueError(f"{entry.get('name')}: scalar affine needs resolved positive M and N")
    src, out = entry.get("lhs", "A0"), entry.get("out", "Y0")
    if entry.get("relu"):
        # max(x, 0), no immediates: the activation as a standalone elementwise map (aten.relu).
        return {
            "schema": SCHEMA_V2,
            "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": "operand"}],
            "nodes": [{"name": "mapped", "op": "relu", "inputs": [src], "dtype": "accumulator"}],
            "outputs": [{"name": out, "value": "mapped"}],
        }  # fmt: skip
    if entry.get("cast"):
        # The identity map read out at accumulator width: the int8 -> int32 sign-extending widening (aten._to_copy).
        return {
            "schema": SCHEMA_V2,
            "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": "operand"}],
            "nodes": [{"name": "mapped", "op": "cast", "inputs": [src], "dtype": "accumulator"}],
            "outputs": [{"name": out, "value": "mapped"}],
        }  # fmt: skip
    if "weight" not in entry or "bias" not in entry:
        raise ValueError(f"{entry.get('name')}: scalar affine states its weight and bias immediates")
    return {
        "schema": SCHEMA_V2,
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": "operand"}],
        "nodes": [
            {
                "name": "mapped", "op": "affine", "inputs": [src],
                "weight": entry["weight"], "bias": entry["bias"], "dtype": "accumulator",
            }
        ],
        "outputs": [{"name": out, "value": "mapped"}],
    }  # fmt: skip


def build_scalar_affine(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A standalone scalar-immediate affine map (op == scalar_affine), written as a component program."""
    from .corpus_spec import build_component_program

    cap, source = build_component_program({**entry, "program": scalar_affine_program(entry, binding)}, binding)
    cap["expected"]["instruction_classes"] = binding.classes_for(
        op="scalar_affine", output_dtype=binding.cap_dtype(binding.accum_dtype)
    )
    return cap, source
