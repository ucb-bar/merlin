"""Derive local modular arithmetic from typed HW SSA, without naming roles.

Only an observed output of a binary add fed by an exact signed product and a
low input slice is recognized. State, instances, muxes and unknown arithmetic
stop the proof. This is a bit-vector relation, never an accumulation protocol,
matrix contraction, resource assignment, iteration count or tensor-axis map.
"""

from __future__ import annotations

from xdsl.ir import BlockArgument, OpResult

from .hw_observations import _inputs, _integer, _name
from .hw_observations import _width as _observed_width

SCHEMA = "merlin.local_hw_arithmetic.v1"
LAW = "signed_multiply_add_modulo_result_width"
UNKNOWN = (
    "cross_state_and_cross_instance_accumulation",
    "contraction_iteration_count_and_order",
    "software_to_selected_datapath_correspondence",
    "physical_resource_and_tensor_axis_mapping",
    "four_state_and_protocol_behavior",
)


def _width(value):
    observed = _observed_width(value)
    return observed if observed is not None and observed > 0 else None


def _operation(value, name, count=None):
    if not isinstance(value, OpResult) or _name(value.owner) != name:
        return None
    op = value.owner
    if len(op.results) != 1 or value.index != 0 or (count is not None and len(op.operands) != count):
        return None
    return op


def _sign_repeat(value, base, seen=frozenset()):
    if value in seen or _width(value) is None:
        return False
    op = _operation(value, "comb.extract", 1)
    if op is not None:
        return op.operands[0] is base and _integer(op, "lowBit") == _width(base) - 1 and _width(value) == 1
    op = _operation(value, "comb.replicate", 1)
    return (
        op is not None
        and _width(op.operands[0]) is not None
        and _width(value) % _width(op.operands[0]) == 0
        and _sign_repeat(op.operands[0], base, seen | {value})
    )


def _signed_root(value, seen=frozenset()):
    """Peel only exact sign extension; no signedness comes from port labels."""
    if value in seen or _width(value) is None:
        return None
    concat = _operation(value, "comb.concat", 2)
    if concat is None:
        return value
    high, low = concat.operands
    if (
        _width(high) is None
        or _width(low) is None
        or _width(high) + _width(low) != _width(value)
        or not _sign_repeat(high, low)
    ):
        return None
    return _signed_root(low, seen | {value})


def _input_low(value, inputs):
    if isinstance(value, BlockArgument) and value in inputs and _width(value) is not None:
        return value
    extract = _operation(value, "comb.extract", 1)
    if extract is None or _integer(extract, "lowBit") != 0:
        return None
    original = extract.operands[0]
    if original in inputs and _width(original) is not None and _width(value) <= _width(original):
        return original
    return None


def _product(value, inputs):
    root = _signed_root(value)
    multiply = _operation(root, "comb.mul", 2)
    if multiply is None:
        return None
    args = tuple(_signed_root(operand) for operand in multiply.operands)
    if any(arg not in inputs or _width(arg) is None for arg in args):
        return None
    width = _width(root)
    if (
        width is None
        or any(_width(operand) != width for operand in multiply.operands)
        or sum(_width(arg) for arg in args) > width
        or _width(value) < width
    ):
        return None
    return args, width


def _relation(value, inputs, ordinal):
    add = _operation(value, "comb.add", 2)
    if add is None or _width(value) is None or any(_width(arg) != _width(value) for arg in add.operands):
        return None
    matches = []
    for product, addend in (tuple(add.operands), tuple(reversed(add.operands))):
        multiplied, original = _product(product, inputs), _input_low(addend, inputs)
        if multiplied is not None and original is not None:
            operands, product_bits = multiplied
            matches.append(
                {
                    "output_ordinal": ordinal,
                    "law": LAW,
                    "operands": [{"input_ordinal": arg.index, "width": _width(arg)} for arg in operands],
                    "addend": {"input_ordinal": original.index, "width": _width(original)},
                    "product_bits": product_bits,
                    "result_bits": _width(value),
                }
            )
    return matches[0] if len(matches) == 1 else None


def local_arithmetic(module):
    """Read all original module outputs, retaining an explicit incomplete scope.

    Each relation is an exact two-state modular expression of original input
    ordinals. It does not state which module is selected by a software operation.
    Unrecognized outputs remain unexamined, never inferred to be non-arithmetic.
    """
    records, module_count, output_count = [], 0, 0
    for op in module.walk():
        if _name(op) != "hw.module":
            continue
        module_count += 1
        inputs = _inputs(op)
        terminator = op.regions[0].block.last_op
        if terminator is None or _name(terminator) != "hw.output":
            raise ValueError("local arithmetic needs the exact original HW output terminator")
        output_count += len(terminator.operands)
        for ordinal, value in enumerate(terminator.operands):
            relation = _relation(value, inputs, ordinal)
            if relation is not None:
                records.append({"module": op.attributes["sym_name"].data, **relation})
    return {
        "schema": SCHEMA,
        "scope": "local combinational module-output two-state modular expression",
        "relations": records,
        "examined_modules": module_count,
        "examined_outputs": output_count,
        "unrecognized_outputs": output_count - len(records),
        "complete_arithmetic_domain": False,
        "unknowns": list(UNKNOWN),
    }
