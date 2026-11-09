"""Recognize a closed row-major floating prefix scan with a wide accumulator.

This source translation preserves the explicit binary64 accumulator. It grants
neither a target implementation nor permission to narrow the accumulation.
"""

from dataclasses import dataclass
from math import copysign

from xdsl.dialects.builtin import FloatAttr, IntegerAttr
from xdsl.ir import Operation

from .linalg_recognition import LinalgBridgeError, _type


@dataclass(frozen=True)
class OrderedScanSource:
    source: object
    stride: int
    extent: int
    prelude: frozenset


def recognize_ordered_scan(loop):
    """Check the complete scalar/control/dataflow pattern, ignoring provenance."""

    def require(condition, reason):
        if not condition:
            raise LinalgBridgeError("ordered floating scan: " + reason)

    def screen(op, name, properties=frozenset(), regions=0):
        require(
            op.name == name
            and not op.successors
            and len(op.regions) == regions
            and not (set(op.properties) - properties)
            and all(key.startswith("prov.") for key in op.attributes),
            "unknown operation, properties or effects",
        )
        flags = op.properties.get("fastmath")
        require(flags is None or not flags.data, "relaxed arithmetic is not admitted")

    def exact(op, name, operands, results=1, properties=frozenset()):
        screen(op, name, properties)
        require(tuple(op.operands) == tuple(operands) and len(op.results) == results, "scalar SSA connections differ")

    constants = set()

    def literal(value, dtype):
        owner = value.owner
        require(isinstance(owner, Operation), "loop bound or zero is not a constant")
        screen(owner, "arith.constant", {"value"})
        require(
            not owner.operands and tuple(owner.results) == (value,) and str(value.type) == dtype,
            "constant has a different signature",
        )
        attribute = owner.properties.get("value")
        require(
            isinstance(attribute, IntegerAttr if dtype == "index" else FloatAttr) and str(attribute.type) == dtype,
            "constant has a different type",
        )
        constants.add(owner)
        return attribute.value.data

    screen(loop, "scf.for", regions=1)
    require(
        len(loop.operands) == 5 and len(loop.results) == 2 and len(loop.regions[0].blocks) == 1,
        "expected induction, output and wide-accumulator tensors",
    )
    lower, upper, step, out_init, acc_init = loop.operands
    total = literal(upper, "index")
    require(
        literal(lower, "index") == 0 and literal(step, "index") == 1 and 0 < total < 1 << 63,
        "loop must visit the complete positive flat domain once",
    )
    output_type, accumulator_type = _type(out_init.type), _type(acc_init.type)
    require(
        output_type.shape == accumulator_type.shape == (total,)
        and output_type.dtype == "f32"
        and accumulator_type.dtype == "f64"
        and tuple(result.type for result in loop.results) == (out_init.type, acc_init.type),
        "only flat f32 output with a separate f64 accumulator is admitted",
    )
    for value in (out_init, acc_init):
        require(isinstance(value.owner, Operation), "loop init has no definition")
        exact(value.owner, "tensor.empty", ())
    require(not tuple(loop.results[1].uses), "wide accumulator escapes the recognized source operation")
    block = loop.regions[0].block
    require(
        len(block.args) == 3 and tuple(arg.type for arg in block.args) == (lower.type, out_init.type, acc_init.type),
        "loop-carried argument types differ",
    )
    iv, output, accumulator = block.args
    body = tuple(block.ops)
    require(len(body) == 11, "loop body has an additional or missing operation")
    read, widen, quotient, coordinate, first, prior, add, narrow, store_out, store_acc, yielded = body
    require(len(read.operands) == 2, "input read has a different rank")
    source = read.operands[0]
    require(
        source.owner is not block and _type(source.type) == output_type,
        "scan input must be an external flat f32 tensor",
    )
    exact(read, "tensor.extract", (source, iv))
    exact(widen, "arith.extf", (read.results[0],))
    require(str(read.results[0].type) == "f32" and str(widen.results[0].type) == "f64", "input widening differs")
    require(len(quotient.operands) == len(coordinate.operands) == 2, "lane coordinates have a different arity")
    stride, extent = literal(quotient.operands[1], "index"), literal(coordinate.operands[1], "index")
    require(
        stride > 0 and extent > 0 and stride * extent <= total and total % (stride * extent) == 0,
        "lane stride and extent do not partition the flat domain",
    )
    exact(quotient, "arith.divui", (iv, quotient.operands[1]))
    exact(coordinate, "arith.remui", (quotient.results[0], coordinate.operands[1]))
    exact(first, "arith.cmpi", (coordinate.results[0], lower), properties={"predicate"})
    predicate = first.properties.get("predicate")
    require(isinstance(predicate, IntegerAttr) and predicate.value.data == 0, "lane reset is not equality to zero")
    screen(prior, "scf.if", regions=2)
    require(
        tuple(prior.operands) == (first.results[0],)
        and len(prior.results) == 1
        and str(prior.results[0].type) == "f64"
        and all(len(region.blocks) == 1 for region in prior.regions),
        "previous accumulator selection differs",
    )
    first_block, rest_block = (region.block for region in prior.regions)
    require(not first_block.args and not rest_block.args, "reset branches have unexpected arguments")
    resets, rest = tuple(first_block.ops), tuple(rest_block.ops)
    require(
        len(resets) == 1 and len(resets[0].operands) == 1 and len(rest) == 3, "reset or continuation branch differs"
    )
    zero = resets[0].operands[0]
    zero_value = literal(zero, "f64")
    require(zero_value == 0 and copysign(1.0, zero_value) > 0, "reset must use positive binary64 zero")
    exact(resets[0], "scf.yield", (zero,), results=0)
    previous_index, previous_read, previous_yield = rest
    exact(previous_index, "arith.subi", (iv, quotient.operands[1]), properties={"overflowFlags"})
    flags = previous_index.properties.get("overflowFlags")
    require(flags is None or not flags.data, "prior-index arithmetic has overflow flags")
    exact(previous_read, "tensor.extract", (accumulator, previous_index.results[0]))
    exact(previous_yield, "scf.yield", (previous_read.results[0],), results=0)
    exact(add, "arith.addf", (prior.results[0], widen.results[0]), properties={"fastmath"})
    exact(narrow, "arith.truncf", (add.results[0],))
    exact(store_out, "tensor.insert", (narrow.results[0], output, iv))
    exact(store_acc, "tensor.insert", (add.results[0], accumulator, iv))
    exact(yielded, "scf.yield", (store_out.results[0], store_acc.results[0]), results=0)
    require(
        str(add.results[0].type) == "f64" and str(narrow.results[0].type) == "f32",
        "accumulator or output rounding differs",
    )
    descendants = set(loop.walk())
    siblings = tuple(loop.parent_block().ops)
    require(
        all(op in siblings and siblings.index(op) < siblings.index(loop) for op in constants),
        "constants do not dominate the loop",
    )
    require(
        all(use.operation in descendants for op in constants for result in op.results for use in result.uses),
        "loop scalar prelude escapes",
    )
    try:
        loop.verify()
    except Exception as error:  # noqa: BLE001 - xDSL validates source type/region invariants
        raise LinalgBridgeError(f"ordered floating scan failed source verification: {error}") from error
    return OrderedScanSource(source, stride, extent, frozenset(constants))
