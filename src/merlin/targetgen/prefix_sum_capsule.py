"""The prefix-sum capsule: an inclusive exact integer scan down the outer (row) axis, ``Y[i] = sum_{k<=i} X[k]``.

Written as a generic component program (``component_program.render``; golden = the independent exact integer
evaluator ``component_numerics``). The written region is a sum reduction whose reduced elements are gated by the
triangular index mask ``k <= i`` (a select against the literal zero), so zero stays the additive identity of every
padded element. The mask is the lower-triangular ones matrix: the scan is the integer sum the spec's standalone
sum-reduction row declares, with a constant 0/1 weight, so no scan primitive is assumed of the hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .component_program import SCHEMA_V2

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def prefix_sum_program(entry: dict, binding: CorpusBinding) -> dict:
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, N)):
        raise ValueError(f"{entry.get('name')}: prefix sum needs resolved positive M and N")
    src, out = entry.get("lhs", "A0"), entry.get("out", "Y0")
    return {
        "schema": SCHEMA_V2,
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": "operand"}],
        "nodes": [{"name": "scan", "op": "cumsum", "inputs": [src], "axis": 0, "dtype": "accumulator"}],
        "outputs": [{"name": out, "value": "scan"}],
    }


def build_prefix_sum(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A standalone prefix sum (op == prefix_sum), written as a component program."""
    from .corpus_spec import build_component_program

    cap, source = build_component_program({**entry, "program": prefix_sum_program(entry, binding)}, binding)
    cap["expected"]["instruction_classes"] = binding.classes_for(
        op="prefix_sum", output_dtype=binding.cap_dtype(binding.accum_dtype)
    )
    return cap, source
