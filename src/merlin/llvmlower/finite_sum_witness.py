"""Typed finite-result witness for a complete ordinary same-tensor row sum.

A finite sum result certifies every input element of that row was finite:
ordinary IEEE addition cannot turn a NaN or infinity back into a finite value.
The witness grants no value bound, library, physical alias or timing permission.
"""

from __future__ import annotations


def _ordinary(operation, allowed=()):
    return all(
        name in allowed or name.startswith("prov.")
        or name == "fastmath" and str(value) == "#arith.fastmath<none>"
        for name, value in (*operation.attributes.items(), *operation.properties.items())
    )


def complete_same_row_sum(numerator, denominator) -> bool:
    from xdsl.dialects.builtin import Float32Type, NoneAttr, TensorType
    from xdsl.ir import Operation

    typ = numerator.type
    if (not isinstance(typ, TensorType) or not isinstance(typ.element_type, Float32Type)
            or not isinstance(typ.encoding, NoneAttr) or len(typ.get_shape()) != 2
            or any(d <= 0 for d in typ.get_shape())):
        return False
    rows, _ = typ.get_shape()
    if denominator.type != TensorType(Float32Type(), [rows, 1]):
        return False
    view = denominator.owner
    if (not isinstance(view, Operation) or view.name != "tensor.expand_shape"
            or len(view.operands) != 1 or len(view.results) != 1
            or not _ordinary(view, {"reassociation", "static_output_shape"})):
        return False
    reduced = view.operands[0]
    if reduced.type != TensorType(Float32Type(), [rows]):
        return False
    operation = reduced.owner
    if (not isinstance(operation, Operation) or operation.name != "linalg.reduce"
            or len(operation.operands) != 2 or len(operation.results) != 1
            or operation.operands[0] is not numerator
            or operation.operands[1].type != reduced.type
            or not _ordinary(operation, {"dimensions"})
            or tuple(operation.dimensions.get_values()) != (1,)
            or len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1):
        return False
    block = operation.regions[0].block
    body = tuple(block.ops)
    return (
        len(block.args) == 2 and len(body) == 2 and body[0].name == "arith.addf"
        and not body[0].regions and _ordinary(body[0])
        and tuple(body[0].operands) in (tuple(block.args), tuple(block.args)[::-1])
        and body[1].name == "linalg.yield" and _ordinary(body[1])
        and tuple(body[1].operands) == tuple(body[0].results)
    )
