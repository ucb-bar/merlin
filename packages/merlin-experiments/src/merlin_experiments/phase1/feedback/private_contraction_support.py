"""Source proof for zero initialization of a contraction output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def contraction_zero_init_fill(row: Mapping[str, Any], parsed: Sequence[Any]) -> bool:
    """Prove a fill is only the zero initialization of a contraction output.

    A standalone fill is tensor computation and still needs normal admission. This
    narrow shape of fill has no independent result: its sole consumer is the
    contraction's output operand, so the linked whole-program build is the
    remaining lowering obligation.
    """
    if row.get("disposition") != "support_required" or row.get("mlir_operation") != "linalg.fill":
        return False
    ordinals = row.get("ordinals")
    if not isinstance(ordinals, list) or len(ordinals) != row.get("count"):
        return False
    for ordinal in ordinals:
        if type(ordinal) is not int or ordinal < 0 or ordinal >= len(parsed):
            return False
        fill = parsed[ordinal]
        if fill.name != "linalg.fill" or len(fill.inputs) != 1 or len(fill.outputs) != 1 or len(fill.results) != 1:
            return False
        scalar = getattr(fill.inputs[0], "owner", None)
        value = getattr(getattr(scalar, "value", None), "value", None)
        if getattr(scalar, "name", None) != "arith.constant" or getattr(value, "data", None) != 0:
            return False
        empty = getattr(fill.outputs[0], "owner", None)
        if getattr(empty, "name", None) != "tensor.empty":
            return False
        empty_uses = list(fill.outputs[0].uses)
        if len(empty_uses) != 1 or empty_uses[0].operation is not fill:
            return False
        uses = list(fill.results[0].uses)
        if len(uses) != 1:
            return False
        contraction = uses[0].operation
        if contraction.name not in {"linalg.matmul", "linalg.batch_matmul"}:
            return False
        if len(contraction.outputs) != 1 or contraction.outputs[0] is not fill.results[0]:
            return False
    return True
