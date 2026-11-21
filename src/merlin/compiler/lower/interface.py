"""Convert complete tensor functions to Merlin's canonical Interface IR.

Upstream source recognition and coverage belong to Merlin. OOT backends consume
``interface.matmul`` and ``interface.commit`` with explicit numerical contracts,
then choose their own target operations, schedules and physical ABI.

The initial domain is a static, from-zero signed i8 x i8 -> i32 dense or batched
contraction, with two function arguments and one complete returned tensor. Other
source work must use another admitted conversion; this pass never drops it.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass

from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, ModuleOp, NoneAttr, StringAttr, TensorType
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.printer import Printer

from merlin.compiler.dialects import interface
from merlin.compiler.lower.quant.requantization import _canonical_body, match_integer_gemm
from merlin.compiler.lower.interface_readout import match_readout


from merlin.compiler.ir.interface_contract import (
    BOUNDARY_ATTRIBUTE,
    NUMERICAL_ATTRIBUTE,
    SOURCE_KIND as SOURCE_KIND,
    InterfaceConversionError,
    _validate_attribute as _validate_attribute,
    _validate_numerical as _validate_numerical,
    parse_interface as parse_interface,
    validate_interface,
)


# Provenance is retained as evidence, never interpreted as a lowering request.
# Everything else must be understood by the operation-specific admission below.
_PROPERTIES = {
    "builtin.module": frozenset(),
    "func.func": frozenset({"sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"}),
    "func.return": frozenset(),
    "tensor.empty": frozenset(),
    "tensor.splat": frozenset(),
    "arith.constant": frozenset({"value"}),
    "linalg.fill": frozenset({"operandSegmentSizes"}),
    "linalg.matmul": frozenset({"operandSegmentSizes", "indexing_maps"}),
    "linalg.generic": frozenset({"operandSegmentSizes", "indexing_maps", "iterator_types"}),
    "arith.extsi": frozenset(),
    "arith.muli": frozenset({"overflowFlags"}),
    "arith.addi": frozenset({"overflowFlags"}),
    "arith.sitofp": frozenset(),
    "arith.mulf": frozenset({"fastmath"}),
    "arith.addf": frozenset({"fastmath"}),
    "math.roundeven": frozenset({"fastmath"}),
    "arith.maximumf": frozenset({"fastmath"}),
    "arith.minimumf": frozenset({"fastmath"}),
    "arith.fptosi": frozenset(),
    "arith.maxsi": frozenset(),
    "arith.minsi": frozenset(),
    "arith.trunci": frozenset(),
    "linalg.yield": frozenset(),
}


def _text(operation: Operation) -> str:
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(operation)
    return stream.getvalue() + "\n"


def _metadata(module: ModuleOp) -> list[dict]:
    records = []
    for ordinal, operation in enumerate(module.walk()):
        allowed = _PROPERTIES.get(operation.name)
        if allowed is None:
            raise InterfaceConversionError(f"source function contains unsupported operation {operation.name}")
        unknown = set(operation.properties) - allowed
        if unknown:
            raise InterfaceConversionError(f"{operation.name} has unhandled properties {sorted(unknown)}")
        if any(not key.startswith("prov.") for key in operation.attributes):
            raise InterfaceConversionError(f"{operation.name} has unhandled non-provenance attributes")
        if operation.successors:
            raise InterfaceConversionError("source conversion requires a single straight-line tensor block")
        overflow = operation.properties.get("overflowFlags")
        if overflow is not None and getattr(overflow, "data", None):
            raise InterfaceConversionError("integer overflow assertions require a separate preservation proof")
        fastmath = operation.properties.get("fastmath")
        if fastmath is not None and str(fastmath) != "#arith.fastmath<none>":
            raise InterfaceConversionError("source readout fastmath assumptions need an explicit preservation proof")
        if operation.attributes:
            records.append(
                {
                    "operation_ordinal": ordinal,
                    "operation": operation.name,
                    "attributes": {key: str(value) for key, value in operation.attributes.items()},
                }
            )
    return records


def _plain_tensor(value) -> None:
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr):
        raise InterfaceConversionError("source boundary requires ordinary dense ranked tensors")
    if any(dimension <= 0 for dimension in ty.get_shape()):
        raise InterfaceConversionError("source boundary requires positive static tensor dimensions")


def _initializer(contraction: Operation, block) -> set[Operation]:
    """Account for the complete zero initializer, including its empty storage."""
    initial = contraction.operands[2]
    producer = initial.owner
    if not isinstance(producer, Operation) or producer.parent is not block:
        raise InterfaceConversionError("accumulator initialization must be local to the selected function")
    if len(producer.results) != 1 or producer.results[0] is not initial:
        raise InterfaceConversionError("initializer has an unsupported result boundary")
    expected_operands = 2 if producer.name == "linalg.fill" else 1
    if producer.name not in ("linalg.fill", "tensor.splat") or len(producer.operands) != expected_operands:
        raise InterfaceConversionError("accumulator needs a static scalar-zero fill or splat")
    scalar = producer.operands[0]
    constant = scalar.owner
    if (
        not isinstance(constant, Operation)
        or constant.parent is not block
        or constant.name != "arith.constant"
        or constant.operands
        or constant.regions
        or len(constant.results) != 1
        or str(scalar.type) != "i32"
    ):
        raise InterfaceConversionError("initializer needs a local scalar i32 zero constant")
    value = constant.properties.get("value")
    if not isinstance(value, IntegerAttr) or value.value.data != 0:
        raise InterfaceConversionError("nonzero accumulator initialization is not admitted")
    covered = {producer, constant}
    if producer.name == "tensor.splat":
        if producer.regions:
            raise InterfaceConversionError("tensor splat cannot contain an executable region")
    else:
        empty_value = producer.operands[1]
        empty = empty_value.owner
        if (
            not isinstance(empty, Operation)
            or empty.parent is not block
            or empty.name != "tensor.empty"
            or empty.operands
            or empty.regions
            or len(empty.results) != 1
            or empty_value.type != initial.type
        ):
            raise InterfaceConversionError("fill needs a local static tensor.empty destination")
        if len(producer.regions) != 1 or len(producer.regions[0].blocks) != 1:
            raise InterfaceConversionError("fill has an unsupported scalar region")
        body = producer.regions[0].block
        operations = list(body.ops)
        if (
            len(body.args) != 2
            or [str(arg.type) for arg in body.args] != ["i32", "i32"]
            or len(operations) != 1
            or operations[0].name != "linalg.yield"
            or list(operations[0].operands) != [body.args[0]]
        ):
            raise InterfaceConversionError("fill region must yield its scalar zero unchanged")
        covered.add(empty)
    return covered


def _contraction_maps(operation: Operation) -> None:
    """Named matmul may carry maps too; do not assume they are canonical."""
    if operation.name != "linalg.matmul":
        return
    maps = operation.properties.get("indexing_maps")
    expected = ((0, 2), (2, 1), (0, 1))
    if maps is None or len(maps.data) != 3:
        raise InterfaceConversionError("named matmul requires canonical indexing maps")
    for attribute, indices in zip(maps.data, expected, strict=True):
        mapping = attribute.data
        if (
            mapping.num_dims != 3
            or mapping.num_symbols != 0
            or tuple(getattr(expr, "position", None) for expr in mapping.results) != indices
        ):
            raise InterfaceConversionError("named matmul indexing differs from dense GEMM")


def to_interface(module: ModuleOp, *, function: str | None = None) -> ModuleOp:
    """Return canonical Interface IR for one completely covered tensor function.

    The original module is unchanged. Its tensor function signature, argument
    order and result survive; pointer conversion belongs to the selected OOT
    backend. Constants cover the exact zero initializer and immutable readout
    parameters. Ordered-f32 scale/bias/readout or integer-saturating i8 readout
    is materialized only after shared source matching. Embedded weights, dynamic
    qparams, extra results and side consumers explicitly refuse. Scalar and
    final-axis channel parameters retain their exact constant SSA values.
    """
    module.verify()
    provenance = _metadata(module)
    roots = list(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], func.FuncOp):
        raise InterfaceConversionError("source conversion requires exactly one complete func.func")
    entry = roots[0]
    if function is not None and entry.sym_name.data != function:
        raise InterfaceConversionError(f"source has no sole function named {function!r}")
    if len(entry.body.blocks) != 1:
        raise InterfaceConversionError("source entry must have exactly one defined block")
    visibility = entry.properties.get("sym_visibility")
    if visibility is not None and visibility.data != "public":
        raise InterfaceConversionError("nonpublic source symbol visibility is not admitted by this boundary")
    for key in ("arg_attrs", "res_attrs"):
        attributes = entry.properties.get(key)
        if attributes is not None and any(attribute.data for attribute in attributes.data):
            raise InterfaceConversionError(f"source {key} need an explicit Interface IR attribute mapping")
    block = entry.body.block
    if len(block.args) != 2 or len(entry.function_type.outputs.data) != 1:
        raise InterfaceConversionError("source entry requires two tensor inputs and one tensor result")
    for argument in block.args:
        _plain_tensor(argument)
    operations = list(block.ops)
    contractions = [operation for operation in operations if match_integer_gemm(operation) is not None]
    if len(contractions) != 1:
        raise InterfaceConversionError("source needs one exact from-zero signed i8 x i8 -> i32 contraction")
    contraction = contractions[0]
    dimensions = match_integer_gemm(contraction)
    assert dimensions is not None
    if not _canonical_body(contraction):
        raise InterfaceConversionError(
            "contraction region must explicitly sign-extend i8 operands before i32 arithmetic"
        )
    _contraction_maps(contraction)
    lhs, rhs = contraction.operands[:2]
    if {lhs, rhs} != set(block.args):
        raise InterfaceConversionError(
            "both source arguments must be the contraction operands; embedded inputs are unsupported"
        )
    accumulator_result = contraction.results[0]
    _plain_tensor(accumulator_result)
    _plain_tensor(contraction.operands[2])
    returned = operations[-1] if operations else None
    if not isinstance(returned, func.ReturnOp) or len(returned.operands) != 1:
        raise InterfaceConversionError("the complete region must have one sole tensor return")
    result = returned.operands[0]
    _plain_tensor(result)
    try:
        readout = match_readout(contraction, result)
    except ValueError as exc:
        raise InterfaceConversionError(f"source readout is not admitted: {exc}") from exc
    covered = {contraction, returned, *_initializer(contraction, block)}
    if readout is not None:
        covered.update(readout.operations)
    if set(operations) != covered:
        raise InterfaceConversionError("source function contains additional computations or effects")
    covered_operations = {member for operation in covered for member in operation.walk()}
    for operation in operations:
        for value in operation.results:
            if any(use.operation not in covered_operations for use in value.uses):
                raise InterfaceConversionError("source value has a consumer outside the converted function boundary")
    # The Interface operations expose the proven tensor computation; no target
    # schedule, allocation, packing decision or instruction encoding is chosen.
    converted_entry = entry.clone()
    converted_block = converted_entry.body.block
    for operation in reversed(list(converted_block.ops)):
        converted_block.detach_op(operation)
        operation.erase()
    lhs_index, rhs_index = block.args.index(lhs), block.args.index(rhs)
    numerical_contract = {
        "lhs_signedness": "signed",
        "rhs_signedness": "signed",
        "accumulator_initialization": "zero",
        "accumulator_overflow": "wrap",
        "layout": "dense_row_major",
        "contraction": "batch_matmul" if len(lhs.type.get_shape()) == 3 else "matmul",
    }
    matmul = interface.MatmulOp(
        operands=[converted_block.args[lhs_index], converted_block.args[rhs_index]],
        result_types=[interface.AccumulatorType(accumulator_result.type)],
        attributes={
            NUMERICAL_ATTRIBUTE: DictionaryAttr({key: StringAttr(value) for key, value in numerical_contract.items()})
        },
    )
    matmul.location = contraction.location
    commit_properties = {
        "epilogue": ArrayAttr([StringAttr(stage) for stage in readout.stages] if readout is not None else []),
        "output_dtype": StringAttr("i8" if readout is not None else "i32"),
    }
    if readout is not None:
        commit_properties["readout"] = readout.contract
    constant_operations = []
    constant_values = []
    if readout is not None:
        for value in readout.parameters:
            constant = value.owner.clone()
            constant_operations.append(constant)
            constant_values.append(constant.results[0])
    commit = interface.CommitOp(
        operands=[matmul.acc, constant_values],
        result_types=[result.type],
        properties=commit_properties,
    )
    commit.location = result.owner.location
    final_return = func.ReturnOp(commit.out)
    final_return.location = returned.location
    converted_block.add_ops([*constant_operations, matmul, commit, final_return])
    converted = ModuleOp([converted_entry], attributes=dict(module.attributes))
    converted.location = module.location
    boundary = {
        "schema": "merlin.source_function.v1",
        "canonical_source_sha256": hashlib.sha256(_text(module).encode()).hexdigest(),
        "source_symbol": entry.sym_name.data,
        "source_signature": str(entry.function_type),
        "source_function_properties": {key: str(value) for key, value in entry.properties.items()},
        "tensor_abi": {
            "arguments": [
                {
                    "source_argument": index,
                    "role": "lhs" if argument is lhs else "rhs",
                    "tensor_type": str(argument.type),
                }
                for index, argument in enumerate(block.args)
            ],
            "results": [{"source_result": 0, "tensor_type": str(result.type)}],
            "physical_abi": "selected by backend; not materialized here",
        },
        "dimensions": asdict(dimensions),
        "numerical_contract": numerical_contract,
        "readout_source_proof": readout.proof if readout is not None else None,
        "readout_parameters": [
            {
                "parameter": index,
                "type": str(value.type),
                "constant_sha256": hashlib.sha256(_text(value.owner).encode()).hexdigest(),
            }
            for index, value in enumerate(readout.parameters if readout is not None else ())
        ],
        "source_contraction_ordinal": operations.index(contraction),
        "initializer": {"value": 0, "type": "i32", "coverage": "complete consumed local zero initializer"},
        "provenance": provenance,
        "coverage": "complete single tensor function",
        "execution": "not_run",
        "qualification": "not_established",
    }
    converted.attributes[BOUNDARY_ATTRIBUTE] = StringAttr(json.dumps(boundary, sort_keys=True, allow_nan=False))
    validate_interface(converted)
    return converted


@dataclass(frozen=True)
class ToInterfacePass(ModulePass):
    """Materialize an admitted complete tensor function as Interface IR."""

    name = "merlin-to-interface"
    function: str | None = None

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        converted = to_interface(op, function=self.function)
        for operation in reversed(list(op.body.block.ops)):
            op.body.block.detach_op(operation)
            operation.erase()
        for operation in list(converted.body.block.ops):
            converted.body.block.detach_op(operation)
            op.body.block.add_op(operation)
        op.attributes.clear()
        op.attributes.update(converted.attributes)
