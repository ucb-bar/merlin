"""Portable scheduling decision IR shared with standalone target compilers."""

from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, i64
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, traits_def
from xdsl.traits import NoMemoryEffect
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class DenseScheduleOp(IRDLOperation):
    """A selected contraction block, with explicit byte and command bounds.

    This records a compile-time choice and executes no command. The target
    adapter verifies its realization before emitting this decision.
    """

    name = "schedule.block_contraction"
    problem = attr_def(DenseArrayBase)
    primitive_tile = attr_def(DenseArrayBase)
    block_tiles = attr_def(DenseArrayBase)
    element_bytes = attr_def(DenseArrayBase)
    storage_bytes = attr_def(DenseArrayBase)
    storage_capacity = attr_def(DenseArrayBase)
    static_commands = attr_def(IntegerAttr)
    command_budget = attr_def(IntegerAttr)
    traits = traits_def(NoMemoryEffect())

    def verify_(self):
        values = {}
        for name, count in (
            ("problem", 3),
            ("primitive_tile", 3),
            ("block_tiles", 3),
            ("element_bytes", 3),
            ("storage_bytes", 2),
            ("storage_capacity", 2),
        ):
            value = getattr(self, name)
            data = tuple(value.get_values())
            if value.elt_type != i64 or len(data) != count or any(v <= 0 for v in data):
                raise VerifyException(f"{name} requires {count} positive i64 values")
            values[name] = data
        tm, tk, tn = values["primitive_tile"]
        bm, bk, bn = values["block_tiles"]
        a, b, c = values["element_bytes"]
        expected = (bk * (bm * tm * tk * a + bn * tk * tn * b), bm * bn * tm * tn * c)
        if expected != values["storage_bytes"]:
            raise VerifyException("dense schedule storage differs from its tile representation")
        if any(used > capacity for used, capacity in zip(expected, values["storage_capacity"])):
            raise VerifyException("dense schedule exceeds available storage")
        sites, budget = self.static_commands, self.command_budget
        if sites.type != i64 or budget.type != i64 or not 0 <= sites.value.data <= budget.value.data:
            raise VerifyException("dense schedule exceeds its static command budget")


def dense_schedule_op(problem, block, resources, static_commands, budget):
    """Materialize a checked decision without importing a particular planner copy."""
    arrays = {
        "problem": (problem.m, problem.k, problem.n),
        "primitive_tile": (problem.tile_m, problem.tile_k, problem.tile_n),
        "block_tiles": (block.m, block.k, block.n),
        "element_bytes": (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes),
        "storage_bytes": (block.operand_bytes, block.accumulator_bytes),
        "storage_capacity": (resources.operand_bytes, resources.accumulator_bytes),
    }
    op = DenseScheduleOp.create(
        attributes={
            **{name: DenseArrayBase.from_list(i64, values) for name, values in arrays.items()},
            "static_commands": IntegerAttr(static_commands, i64),
            "command_budget": IntegerAttr(budget, i64),
        }
    )
    op.verify()
    return op


def _trace_attributes(attributes) -> bool:
    """Accept only typed capture attribution; never use its values for legality.

    These existing frontend fields name taxonomy, original dtype or source
    ancestry. They remain on the exact source. Placement, dispatch, numerical
    policy and unknown ``prov.*`` fields are deliberately outside this reader.
    """
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    strings = {
        "prov.region_id",
        "prov.fqn",
        "prov.module",
        "prov.family",
        "prov.op",
        "prov._pattern_hint",
        "prov.aten",
        "prov.orig_dtype",
        "prov.trace_role",
    }
    lists = {"prov.source_node_ids", "prov.origin_node_ids"}
    for name, value in attributes.items():
        if name in strings:
            if type(value) is not StringAttr:
                return False
        elif name in lists:
            if type(value) is not ArrayAttr or any(type(item) is not StringAttr for item in value):
                return False
        else:
            return False
    return True


def recognize_signed_integer_matmul(op, dtypes=("i8", "i8", "i32")) -> tuple:
    """Read exact tensor matmul maps and signed modular multiply/add semantics.

    This establishes the scalar recurrence, not zero initialization, ownership,
    physical layout, target support or a numerical/qualification receipt.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType, NoneAttr, Signedness, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(op) is not GenericOp:
        raise ValueError("dense source requires a typed linalg.generic")
    if type(dtypes) is not tuple or len(dtypes) != 3 or any(t not in ("i8", "i16", "i32", "i64") for t in dtypes):
        raise ValueError("dense source requires explicit supported signed integer formats")
    if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise ValueError("linalg.generic is not a two-input, one-init, one-result contraction")
    lhs, rhs = op.inputs
    init = op.outputs[0]
    values = (lhs, rhs, init, op.results[0])
    if not all(isinstance(v.type, TensorType) for v in values) or [str(v.type.element_type) for v in values] != [
        *dtypes,
        dtypes[2],
    ]:
        raise ValueError("linalg.generic element types differ from the selected signed matmul formats")
    if any(
        type(v.type.element_type) is not IntegerType or v.type.element_type.signedness.data != Signedness.SIGNLESS
        for v in values
    ):
        raise ValueError("dense source requires signless integer storage with explicit signed widening")
    if any(type(v.type.encoding) is not NoneAttr for v in values):
        raise ValueError("dense source requires ordinary unencoded tensor layouts")
    if op.results[0].type != init.type:
        raise ValueError("linalg.generic result type differs from its initialized output")
    if not _trace_attributes(op.attributes) or set(op.properties) - {
        "indexing_maps",
        "iterator_types",
        "operandSegmentSizes",
    }:
        raise ValueError("dense source has unknown operation attributes or executable library calls")
    maps = tuple(attribute.data for attribute in op.indexing_maps)
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (AffineMap(3, 0, (d0, d2)), AffineMap(3, 0, (d2, d1)), AffineMap(3, 0, (d0, d1)))
    if maps != expected:
        raise ValueError("linalg.generic indexing maps are not rank-2 matmul maps")
    if tuple(attribute.data for attribute in op.iterator_types) != (
        IteratorType.PARALLEL,
        IteratorType.PARALLEL,
        IteratorType.REDUCTION,
    ):
        raise ValueError("linalg.generic iterators are not two parallel and one reduction")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("linalg.generic requires a single scalar body block")
    block = op.regions[0].block
    args = list(block.args)
    if len(args) != 3 or [str(arg.type) for arg in args] != list(dtypes):
        raise ValueError("linalg.generic body argument types do not match signed matmul")
    body = list(block.ops)
    widened, cursor = [], 0
    for argument, dtype in zip(args[:2], dtypes[:2]):
        if int(dtype[1:]) > int(dtypes[2][1:]):
            raise ValueError("dense source cannot narrow an input before multiplication")
        if dtype == dtypes[2]:
            widened.append(argument)
            continue
        if cursor >= len(body) or type(body[cursor]) is not arith.ExtSIOp:
            raise ValueError("linalg.generic body is not signed widen, multiply, add, yield")
        extension = body[cursor]
        if (
            list(extension.operands) != [argument]
            or len(extension.results) != 1
            or str(extension.results[0].type) != dtypes[2]
            or not _trace_attributes(extension.attributes)
            or extension.properties
        ):
            raise ValueError("linalg.generic body does not widen the ordered signed input elements")
        widened.append(extension.results[0])
        cursor += 1
    if [type(inner) for inner in body[cursor:]] != [arith.MuliOp, arith.AddiOp, YieldOp]:
        raise ValueError("linalg.generic body is not signed widen, multiply, add, yield")
    mul, add, yld = body[cursor:]
    if set(mul.operands) != set(widened) or len(mul.operands) != 2:
        raise ValueError("linalg.generic body product does not use both widened inputs")
    if set(add.operands) != {mul.results[0], args[2]} or len(add.operands) != 2:
        raise ValueError("linalg.generic body sum does not use product and accumulator")
    if list(yld.operands) != [add.results[0]] or not _trace_attributes(yld.attributes) or yld.properties:
        raise ValueError("linalg.generic body yields a different value or has unknown yield effects")
    for arithmetic in (mul, add):
        if (
            str(arithmetic.results[0].type) != dtypes[2]
            or arithmetic.overflow_flags.data
            or not _trace_attributes(arithmetic.attributes)
            or set(arithmetic.properties) - {"overflowFlags"}
        ):
            raise ValueError("linalg.generic body has an unsupported arithmetic width, attribute or overflow flag")
    return lhs, rhs, init


def static_zero_integer_matmul_geometry(operation, dtypes=("i8", "i8", "i32"), *, max_source_elements=64 << 20):
    """Read the exact initialized integer dot and its complete static geometry.

    This source-only check establishes no target, packing, effect, numerical
    admission or execution capability. The original source is never rewritten.
    """
    from xdsl.dialects import arith, builtin, tensor
    from xdsl.dialects.linalg.ops import FillOp, YieldOp

    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("dense source reading needs a positive original element budget")
    lhs, rhs, init = recognize_signed_integer_matmul(operation, dtypes)
    shapes = tuple(tuple(v.type.get_shape()) for v in (lhs, rhs, init))
    if any(len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes):
        raise ValueError("dense source requires positive static rank-two tensor shapes")
    (m, k), (rhs_k, n), output = shapes
    if rhs_k != k or output != (m, n):
        raise ValueError("dense source operand extents disagree with the complete reduction domain")
    if m * k + k * n + m * n > max_source_elements:
        raise ValueError("dense source exceeds the selected logical element budget")
    fill = init.owner
    if (
        type(fill) is not FillOp
        or len(fill.inputs) != 1
        or len(fill.outputs) != 1
        or len(fill.results) != 1
        or fill.results[0] is not init
        or not _trace_attributes(fill.attributes)
        or set(fill.properties) - {"operandSegmentSizes"}
    ):
        raise ValueError("dense source requires an explicit pure zero fill initializer")
    constant = fill.inputs[0].owner
    if (
        type(constant) is not arith.ConstantOp
        or not _trace_attributes(constant.attributes)
        or set(constant.properties) != {"value"}
        or type(constant.value) is not builtin.IntegerAttr
        or constant.value.value.data != 0
        or constant.results[0].type != init.type.element_type
    ):
        raise ValueError("dense source initializer is not an exact integer zero")
    if len(fill.regions) != 1 or len(fill.regions[0].blocks) != 1 or len(fill.regions[0].block.args) != 2:
        raise ValueError("dense source zero fill has an unsupported body")
    body = list(fill.regions[0].block.ops)
    if (
        len(body) != 1
        or type(body[0]) is not YieldOp
        or list(body[0].operands) != [fill.regions[0].block.args[0]]
        or not _trace_attributes(body[0].attributes)
        or body[0].properties
    ):
        raise ValueError("dense source zero fill does not yield its zero input")
    empty = fill.outputs[0].owner
    if (
        type(empty) is not tensor.EmptyOp
        or not _trace_attributes(empty.attributes)
        or empty.properties
        or empty.operands
    ):
        raise ValueError("dense source requires a fresh tensor destination before initialization")
    return m, k, n


class InvalidLinalgPattern(ValueError):
    pass


@dataclass(frozen=True)
class StaticLinalgShell:
    shape: tuple[int, ...]
    args: tuple[Any, ...]
    body: tuple[Any, ...]
    ordered_types: tuple[str, ...]
    input_maps: tuple[Any, ...]


def validate_singleton_projection_map(mapping, output_shape: tuple[int, ...], input_shape: tuple[int, ...]) -> None:
    """Require an ascending dimension projection or zero at a singleton input axis."""
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    if mapping.num_dims != len(output_shape) or mapping.num_symbols != 0 or len(mapping.results) != len(input_shape):
        raise InvalidLinalgPattern("pointwise input map is not a static singleton projection")
    last_dim = -1
    for extent, expr in zip(input_shape, mapping.results, strict=True):
        if isinstance(expr, AffineConstantExpr) and type(expr.value) is int and expr.value == 0 and extent == 1:
            continue
        if (
            not isinstance(expr, AffineDimExpr)
            or type(expr.position) is not int
            or expr.position <= last_dim
            or expr.position >= len(output_shape)
            or extent != output_shape[expr.position]
        ):
            raise InvalidLinalgPattern("pointwise input map is not a static singleton projection")
        last_dim = expr.position


def checked_static_linalg_shell(op, *, singleton_projection_inputs: bool = False) -> StaticLinalgShell:
    """Check the common typed tensor/map shell; never admit a scalar body here."""

    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineMap

    if type(op) is not GenericOp or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("expected registered one-result linalg.GenericOp")
    if op.successors:
        raise InvalidLinalgPattern("linalg.generic has an unexpected successor")
    if set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("unrecognized linalg.generic property")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (len(op.inputs), 1)
    ):
        raise InvalidLinalgPattern("pointwise operand segments differ from input/output arity")
    if any(not name.startswith("prov.") for name in op.attributes):
        raise InvalidLinalgPattern("unrecognized linalg.generic attribute")
    values = (*op.inputs, *op.outputs, *op.results)
    if not all(isinstance(value.type, TensorType) for value in values):
        raise InvalidLinalgPattern("pointwise operands and result must be tensors")
    shape = tuple(op.results[0].type.get_shape())
    if not shape or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise InvalidLinalgPattern("pointwise shape is not positive-rank static")
    if not singleton_projection_inputs and any(tuple(value.type.get_shape()) != shape for value in values):
        raise InvalidLinalgPattern("pointwise tensor shapes differ")
    if singleton_projection_inputs and (
        any(dim <= 0 for dim in shape)
        or any(not value.type.has_static_shape() or any(dim <= 0 for dim in value.type.get_shape()) for value in values)
    ):
        raise InvalidLinalgPattern("projected pointwise tensors require positive static extents")
    if any(not isinstance(value.type.encoding, NoneAttr) for value in values):
        raise InvalidLinalgPattern("pointwise tensor encoding is not default")
    if op.results[0].type != op.outputs[0].type:
        raise InvalidLinalgPattern("pointwise result type differs from output init")
    if not isinstance(op.indexing_maps, ArrayAttr) or any(
        not isinstance(attribute, AffineMapAttr) for attribute in op.indexing_maps
    ):
        raise InvalidLinalgPattern("pointwise indexing maps are malformed")
    maps = tuple(attribute.data for attribute in op.indexing_maps)
    if len(maps) != len(op.inputs) + 1 or maps[-1] != AffineMap.identity(len(shape)):
        raise InvalidLinalgPattern("pointwise destination indexing map is not identity")
    if not singleton_projection_inputs:
        if maps != (AffineMap.identity(len(shape)),) * len(maps):
            raise InvalidLinalgPattern("pointwise indexing maps are not identity")
    else:
        for input_value, input_map in zip(op.inputs, maps[:-1], strict=True):
            input_shape = tuple(input_value.type.get_shape())
            validate_singleton_projection_map(input_map, shape, input_shape)
    if not isinstance(op.iterator_types, ArrayAttr) or any(
        not isinstance(attribute, IteratorTypeAttr) for attribute in op.iterator_types
    ):
        raise InvalidLinalgPattern("pointwise iterator metadata is malformed")
    if tuple(attribute.data for attribute in op.iterator_types) != (IteratorType.PARALLEL,) * len(shape):
        raise InvalidLinalgPattern("pointwise iterators are not all parallel")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("pointwise region must have one block")
    block = op.regions[0].block
    args = tuple(block.args)
    ordered_types = tuple(str(value.type.element_type) for value in values)
    if tuple(str(arg.type) for arg in args) != ordered_types[:-1]:
        raise InvalidLinalgPattern("pointwise block argument types differ from tensor elements")
    body = tuple(block.ops)
    return StaticLinalgShell(shape, args, body, ordered_types, maps[:-1])
