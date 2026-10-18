"""Portable scheduling decision IR shared with standalone target compilers."""

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
    if set(op.attributes) - {"prov.region_id", "prov.fqn"} or set(op.properties) - {
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
            or extension.attributes
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
    if list(yld.operands) != [add.results[0]] or yld.attributes or yld.properties:
        raise ValueError("linalg.generic body yields a different value or has unknown yield effects")
    for arithmetic in (mul, add):
        if (
            str(arithmetic.results[0].type) != dtypes[2]
            or arithmetic.overflow_flags.data
            or arithmetic.attributes
            or set(arithmetic.properties) - {"overflowFlags"}
        ):
            raise ValueError("linalg.generic body has an unsupported arithmetic width, attribute or overflow flag")
    return lhs, rhs, init
