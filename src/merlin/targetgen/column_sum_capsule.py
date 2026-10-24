"""The standalone sum-reduction capsule: an exact integer sum over one operand's outer (row) axis.

Written as a generic component program, so the writer (``component_program.render``) and the golden
(the independent exact integer evaluator, ``component_numerics``) are the component program's own; this
module only states the program and what the capsule owes the target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .component_program import SCHEMA_V2

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def column_sum_program(entry: dict, binding: CorpusBinding) -> dict:
    """The typed DAG of a standalone sum reduction over the OUTER (row) axis of one operand.

    One runtime ``input`` ``[M, N]`` in the operand dtype, summed down its rows into one ``[1, N]``
    total per column in the accumulator dtype. Exact integer arithmetic with no scale and no readout
    narrowing, so the comparison is exact. The summed axis is the outer one because that is the axis a
    sum-by-accumulation unit adds along (each row lands on the same accumulator row); a sum across
    the lanes of a row is a contraction against a ones operand, which the contraction capsules
    already demand.
    """
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, N)):
        raise ValueError(f"{entry.get('name')}: column sum needs resolved positive M and N")
    src, out = entry.get("lhs", "A0"), entry.get("out", "Y0")
    return {
        "schema": SCHEMA_V2,
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": "operand"}],
        "nodes": [{"name": "total", "op": "reduce_sum", "inputs": [src], "axis": 0, "dtype": "accumulator"}],
        "outputs": [{"name": out, "value": "total"}],
    }


def build_column_sum(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A standalone sum reduction (op == column_sum), written as a generic component program.

    The writer and the golden are the component program's own: Linalg source from
    :func:`component_program.render` and the independent exact integer evaluator
    (``component_numerics``) as the reference. The only thing added here is what the capsule owes the
    target: the classes a non-contraction op requires, asked of the target's own deriver with this op,
    so a reduction capsule is never told it must issue a multiply -- nor allowed to issue nothing.
    """
    from .corpus_spec import build_component_program

    cap, source = build_component_program({**entry, "program": column_sum_program(entry, binding)}, binding)
    cap["expected"]["instruction_classes"] = binding.classes_for(
        op="column_sum", output_dtype=binding.cap_dtype(binding.accum_dtype)
    )
    return cap, source
