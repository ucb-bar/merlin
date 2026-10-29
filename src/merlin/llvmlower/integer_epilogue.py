"""Source-bound unary integer epilogues for capability-selected readout fusion."""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin, func
from xdsl.ir import Block, Operation, Region

from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import checked_static_linalg_shell
from merlin.verify.scalar_ir import verify_scalar_transform


def _plain(operation):
    return not any(not key.startswith("prov.") for key in operation.attributes)


def _constant(value):
    owner = value.owner
    if not isinstance(owner, Operation) or not _plain(owner) or owner.successors or len(owner.results) != 1:
        return None
    if (
        isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr)
        and set(owner.properties) == {"value"} and not owner.regions
        and owner.value.type == value.type
    ):
        return owner.value
    if (
        owner.name == "tensor.splat" and len(owner.operands) == 1
        and not owner.properties and not owner.regions
        and isinstance(value.type, builtin.TensorType)
        and value.type.element_type == owner.operands[0].type
    ):
        return _constant(owner.operands[0])
    if (
        owner.name == "linalg.fill" and len(owner.inputs) == len(owner.outputs) == 1
        and set(owner.properties) <= {"operandSegmentSizes"}
        and len(owner.regions) == 1 and len(owner.body.blocks) == 1
        and len(owner.body.block.args) == 2
    ):
        body = tuple(owner.body.block.ops)
        if (
            len(body) == 1 and body[0].name == "linalg.yield"
            and not body[0].properties and _plain(body[0])
            and tuple(body[0].operands) == (owner.body.block.args[0],)
        ):
            return _constant(owner.inputs[0])
    return None


def _setup(operation):
    """Only known pure tensor setup may be crossed when moving the readout."""
    if not _plain(operation) or operation.successors:
        return False
    if operation.name == "tensor.empty":
        return not operation.properties and not operation.regions and not operation.operands
    return len(operation.results) == 1 and _constant(operation.results[0]) is not None


@dataclass(frozen=True)
class IntegerEpilogue:
    operations: tuple
    scalar_source: str
    correspondence: object


def match_integer_epilogue(producer, readout, *, max_stages=8, max_scalar_operations=128, timeout_ms=1000):
    """Prove a sole-use same-coordinate chain equals the declared integer readout.

    Uniform tensor operands become exact scalar constants. Initializer reads,
    other observers, coordinate changes and unknown intervening effects retain
    the host path. The proof covers all accumulator bit patterns; it grants no
    implementation or physical readout qualification.
    """
    if type(readout) is not DenseIntegerReadout:
        raise ValueError("explicit declared integer readout required")
    if any(type(v) is not int or v <= 0 for v in (max_stages, max_scalar_operations, timeout_ms)):
        raise ValueError("positive integer epilogue proof budgets required")
    graph = readout.graph()
    if graph.input_bits != (32,) or graph.output_bits != 8 or readout.output_dtype != "i8":
        return None
    shape = tuple(producer.results[0].type.get_shape())
    block = Block(arg_types=[builtin.i32])
    current, scalar, stages = producer.results[0], block.args[0], []
    for _ in range(max_stages):
        uses = tuple(current.uses)
        if len(uses) != 1:
            return None
        operation = uses[0].operation
        if operation.name != "linalg.generic" or operation.parent_block() is not producer.parent_block():
            return None
        try:
            shell = checked_static_linalg_shell(operation)
        except ValueError:
            return None
        if shell.shape != shape:
            return None
        cursor = (stages[-1] if stages else producer).next_op
        while cursor is not operation:
            if cursor is None or not _setup(cursor):
                return None
            cursor = cursor.next_op
        arguments = {}
        for value, argument in zip(operation.inputs, operation.body.block.args):
            if value is current:
                arguments[argument] = scalar
            else:
                constant = _constant(value)
                if constant is None:
                    return None
                cloned = arith.ConstantOp(constant)
                block.add_op(cloned)
                arguments[argument] = cloned.results[0]
        body = tuple(operation.body.block.ops)
        for inner in body[:-1]:
            if not _plain(inner) or inner.regions or inner.successors:
                return None
            for operand in inner.operands:
                if operand not in arguments:
                    constant = _constant(operand)
                    if constant is None or isinstance(operand.type, builtin.TensorType):
                        return None
                    cloned = arith.ConstantOp(constant)
                    block.add_op(cloned)
                    arguments[operand] = cloned.results[0]
            cloned = inner.clone(value_mapper=arguments)
            for key in tuple(cloned.attributes):
                if key.startswith("prov."):
                    del cloned.attributes[key]
            block.add_op(cloned)
        if (
            not body or body[-1].name != "linalg.yield" or len(body[-1].operands) != 1
            or not _plain(body[-1]) or body[-1].properties
        ):
            return None
        scalar = arguments.get(body[-1].operands[0])
        if scalar is None or len(tuple(block.ops)) > max_scalar_operations:
            return None
        stages.append(operation)
        current = operation.results[0]
        if current.type.element_type == builtin.i8:
            block.add_op(func.ReturnOp(scalar))
            actual = ir_text(builtin.ModuleOp([func.FuncOp.from_region(
                "forward", [builtin.i32], [builtin.i8], Region(block),
            )]))
            expected = parse_mlir_text(readout.scalar_ir)
            function = next(op for op in expected.body.block.ops if isinstance(op, func.FuncOp))
            function.properties["sym_name"] = builtin.StringAttr("forward")
            proof = verify_scalar_transform(actual.encode(), ir_text(expected).encode(), timeout_ms=timeout_ms)
            return IntegerEpilogue(tuple(stages), actual, proof) if proof.verified else None
        if current.type.element_type != builtin.i32:
            return None
    return None
