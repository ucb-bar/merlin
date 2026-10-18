"""Closed source-only witness for a static f32 sum reduction.

This describes prepared Linalg source: one registered ``linalg.reduce`` whose
init is an exact +0.0 f32 splat and whose body is one fast-math-free f32
``arith.addf`` of the element into the accumulator. It does not admit host work,
establish frontend equivalence, fix the summation order (``linalg.reduce`` leaves
it to the lowering, as the frontend reference does), or certify a linked lowering.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.frontends.linalg_patterns import InvalidLinalgPattern

# Frontend identity -> (registered Linalg root, reduction kind). Only exact forms.
F32_REDUCTION_TARGETS = {"aten.sum.dim_IntList": ("linalg.reduce", "f32_sum")}
F32_POSITIVE_ZERO_BITS = "0x00000000"


@dataclass(frozen=True)
class StaticF32ReductionPattern:
    """Prepared f32 reduction shape/body under an unbound index-width premise."""

    operation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: tuple[int, ...]
    accumulator_type: str
    init_bits: str
    index_bits_premise: int


def _static_tensor(value, *, allow_scalar: bool = False) -> tuple[int, ...]:
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    if not isinstance(typ, TensorType) or str(typ.element_type) != "f32" or not isinstance(typ.encoding, NoneAttr):
        raise InvalidLinalgPattern("f32 reduction requires a default-encoded f32 tensor")
    shape = tuple(typ.get_shape())
    if (not shape and not allow_scalar) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise InvalidLinalgPattern("f32 reduction requires positive static tensor extents")
    return shape


def _checked_index_bounds(index_bits: int, *shapes: tuple[int, ...]) -> None:
    """Record a caller-supplied width premise, not a selected-build observation."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise InvalidLinalgPattern("f32 reduction requires an explicit finite signed index-width premise")
    maximum = (1 << (index_bits - 1)) - 1
    if any(extent > maximum for shape in shapes for extent in shape):
        raise InvalidLinalgPattern("f32 reduction loop extent exceeds the signed index-width premise")


def _closed(op) -> bool:
    return not (op.regions or op.successors or any(not name.startswith("prov.") for name in op.attributes))


def _checked_positive_zero_init(value) -> str:
    from struct import pack, unpack

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import FloatAttr

    splat = value.owner
    if (
        type(splat) is not tensor.SplatOp
        or value is not splat.result
        or tuple(splat.operands) != (splat.input,)
        or len(splat.results) != 1
        or splat.dynamicSizes
        or splat.properties
        or not _closed(splat)
    ):
        raise InvalidLinalgPattern("f32 reduction init is not a static tensor.splat")
    constant = splat.input.owner
    if (
        type(constant) is not arith.ConstantOp
        or splat.input is not constant.result
        or constant.operands
        or len(constant.results) != 1
        or set(constant.properties) != {"value"}
        or not _closed(constant)
    ):
        raise InvalidLinalgPattern("f32 reduction init is not a closed constant")
    literal = constant.properties["value"]
    if not isinstance(literal, FloatAttr) or str(literal.type) != "f32" or str(constant.result.type) != "f32":
        raise InvalidLinalgPattern("f32 reduction init is not an f32 literal")
    bits = f"0x{unpack('<I', pack('<f', literal.value.data))[0]:08x}"
    if bits != F32_POSITIVE_ZERO_BITS:
        raise InvalidLinalgPattern("f32 sum init is not exactly +0.0")
    try:
        constant.verify()
        splat.verify()
    except Exception as exc:  # noqa: BLE001 - owner verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"f32 reduction init failed xDSL verification: {exc}") from exc
    return bits


def _checked_addf_and_yield(add, yld, element, accumulator) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.traits import Pure

    if type(add) is not arith.AddfOp or not type(add).has_trait(Pure) or not _closed(add):
        raise InvalidLinalgPattern("f32 sum combiner is not one registered pure arith.addf")
    if not set(add.properties) <= {"fastmath"}:
        raise InvalidLinalgPattern("f32 sum combiner has unknown properties")
    flags = add.properties.get("fastmath")
    if flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data):
        raise InvalidLinalgPattern("f32 sum combiner has fast-math flags")
    if (
        tuple(add.operands) != (element, accumulator)
        or len(add.results) != 1
        or str(add.results[0].type) != "f32"
        or type(yld) is not YieldOp
        or tuple(yld.operands) != (add.results[0],)
        or yld.results
        or yld.properties
        or not _closed(yld)
    ):
        raise InvalidLinalgPattern("f32 sum does not yield its exact accumulator update")


def recognize_static_f32_sum(op, *, index_bits: int) -> StaticF32ReductionPattern:
    """Recognize one closed prepared f32 sum, without admission or numeric claims."""
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.dialects.linalg.ops import ReduceOp

    if (
        type(op) is not ReduceOp
        or set(op.properties) != {"dimensions"}
        or any(not name.startswith("prov.") for name in op.attributes)
        or op.successors
        or len(op.operands) != 2
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise InvalidLinalgPattern("f32 sum is not a registered one-result linalg.reduce")
    input_shape = _static_tensor(op.input)
    output_shape = _static_tensor(op.init, allow_scalar=True)
    if _static_tensor(op.results[0], allow_scalar=True) != output_shape or op.results[0].type != op.init.type:
        raise InvalidLinalgPattern("f32 sum output differs from its initialized tensor")
    _checked_index_bounds(index_bits, input_shape, output_shape)
    dimensions = op.properties["dimensions"]
    if not isinstance(dimensions, DenseArrayBase) or str(dimensions.elt_type) != "i64":
        raise InvalidLinalgPattern("f32 sum reduction dimensions are not typed i64")
    axis = tuple(dimensions.iter_values())
    if (
        not axis
        or any(type(dim) is not int or dim < 0 or dim >= len(input_shape) for dim in axis)
        or tuple(sorted(set(axis))) != axis
        or tuple(size for dim, size in enumerate(input_shape) if dim not in axis) != output_shape
    ):
        raise InvalidLinalgPattern("f32 sum has invalid reduction axes or static output geometry")
    init_bits = _checked_positive_zero_init(op.init)
    block = op.regions[0].block
    args, body = tuple(block.args), tuple(block.ops)
    if tuple(str(arg.type) for arg in args) != ("f32", "f32") or len(body) != 2:
        raise InvalidLinalgPattern("f32 sum body is not a two-argument f32 addition")
    _checked_addf_and_yield(body[0], body[1], args[0], args[1])
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL is the fail-closed verification boundary
        raise InvalidLinalgPattern(f"f32 sum failed xDSL verification: {exc}") from exc
    return StaticF32ReductionPattern("f32_sum", input_shape, output_shape, axis, "f32", init_bits, index_bits)


def serialized_f32_reduction_pattern(pattern: StaticF32ReductionPattern) -> dict:
    """Use the same strict JSON shape before and after a receipt is saved."""
    from dataclasses import asdict

    return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(pattern).items()}
