"""Conservative parsed-Linalg entry to Merlin's typed semantic kernel graph.

The admitted region is a single static rank-two integer contraction function.
Every source operation is accounted for; unsupported bodies and initializers
raise instead of being silently reinterpreted as a matmul or zero tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import prod

from .model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from .reference import TensorValue


class LinalgBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class LinalgTranslation:
    request: KernelRequest
    constants: dict[str, TensorValue]
    source_operations: tuple[str, ...]


def _maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _type(attribute) -> TensorType:
    from xdsl.dialects.builtin import TensorType as XdslTensorType

    if not isinstance(attribute, XdslTensorType):
        raise LinalgBridgeError("only ranked tensor values are admitted")
    shape = tuple(int(dimension) for dimension in attribute.get_shape())
    dtype = str(attribute.element_type)
    if dtype == "i8":
        policy = "signed-i8"
    elif dtype == "i32":
        policy = "i32-wrap-k-ascending"
    else:
        raise LinalgBridgeError(f"no admitted numerical policy for {dtype}")
    try:
        return TensorType(shape, dtype, policy)
    except ValueError as exc:
        raise LinalgBridgeError(f"unadmitted tensor type {attribute}: {exc}") from exc


def _scalar_integer(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i32":
        raise LinalgBridgeError("fill scalar must be an explicit i32 integer constant")
    if not -(1 << 31) <= value < (1 << 31):
        raise LinalgBridgeError("fill constant is outside signed i32")
    return value


def _named_matmul_maps_are_default(op) -> bool:
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d2, d1)),
        AffineMap(3, 0, (d0, d1)),
    )
    return tuple(attr.data for attr in op.get_indexing_maps()) == expected and tuple(
        attr.data for attr in op.get_iterator_types()
    ) == (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)


def translate_linalg_text(
    source: str,
    *,
    entry: str,
    target_identity: str,
    lowering_policy: str = "strict-native",
    boundary_storage: str = "external",
) -> LinalgTranslation:
    """Translate one admitted function, preserving its init and ordered returns.

    The source is parsed by Merlin's established xDSL frontend. The shared
    generic-matmul recognizer checks maps, iterators, scalar body,
    signed extension, widths, and operand connections before translation.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern, recognize_signed_i8_i32_matmul

    module = parse_mlir_text(source)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise LinalgBridgeError(f"expected exactly one entry function @{entry}")
    function = functions[0]
    if len(function.body.blocks) != 1:
        raise LinalgBridgeError("entry function must have one block")
    block = function.body.block
    operations = tuple(block.ops)
    if not operations or operations[-1].name != "func.return":
        raise LinalgBridgeError("entry function must end in func.return")

    nodes: list[SemanticNode] = []
    env: dict[object, str | None] = {}
    scalars: dict[object, int] = {}
    constants: dict[str, TensorValue] = {}
    constant_bindings: list[ConstantBinding] = []
    boundaries: list[tuple[str, str]] = []
    source_operations: list[str] = []
    for index, arg in enumerate(block.args):
        type_ = _type(arg.type)
        node_id = f"arg{index}"
        nodes.append(SemanticNode(node_id, "input", (), type_, effect="input"))
        env[arg] = node_id
        boundaries.append((node_id, boundary_storage))

    for ordinal, op in enumerate(operations):
        source_operations.append(op.name)
        if op.name == "arith.constant":
            if len(op.results) != 1:
                raise LinalgBridgeError("scalar constant has an unsupported result count")
            scalars[op.results[0]] = _scalar_integer(op)
        elif op.name == "tensor.empty":
            if len(op.results) != 1:
                raise LinalgBridgeError("tensor.empty has an unsupported result count")
            _type(op.results[0].type)
            env[op.results[0]] = None
        elif op.name == "linalg.fill":
            if len(op.operands) != 2 or len(op.results) != 1:
                raise LinalgBridgeError("linalg.fill requires one scalar and one destination")
            scalar, destination = op.operands
            if scalar not in scalars or destination not in env:
                raise LinalgBridgeError("linalg.fill requires a declared scalar and destination")
            type_ = _type(op.results[0].type)
            if type_ != _type(destination.type) or type_.dtype != "i32":
                raise LinalgBridgeError("linalg.fill result/destination has unsupported type")
            node_id = f"op{ordinal}r0"
            value = scalars[scalar]
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=(("value", value),), effect="constant"))
            constants[node_id] = TensorValue(type_, (value,) * prod(type_.shape))
            constant_bindings.append(ConstantBinding(
                node_id, "i32-le", (value.to_bytes(4, "little", signed=True) * prod(type_.shape)).hex(),
            ))
            env[op.results[0]] = node_id
        elif op.name in {"linalg.generic", "linalg.matmul"}:
            if op.name == "linalg.generic":
                try:
                    lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
                except InvalidLinalgPattern as exc:
                    raise LinalgBridgeError(str(exc)) from exc
            else:
                if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
                    raise LinalgBridgeError("named matmul has an unsupported signature")
                if not _named_matmul_maps_are_default(op):
                    raise LinalgBridgeError("named matmul has nonstandard indexing or iterators")
                lhs, rhs, init = (*op.inputs, op.outputs[0])
            if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
                raise LinalgBridgeError("matmul reads an undefined or uninitialized operand")
            lhs_type, rhs_type, init_type, result_type = (
                _type(value.type) for value in (lhs, rhs, init, op.results[0])
            )
            if lhs_type.dtype != rhs_type.dtype or lhs_type.numerical_policy != rhs_type.numerical_policy:
                raise LinalgBridgeError("matmul input dtypes or numerical policies differ")
            if op.name == "linalg.matmul" and lhs_type.dtype != "i32":
                raise LinalgBridgeError("named matmul is admitted only for i32 operands")
            if len(lhs_type.shape) != 2 or len(rhs_type.shape) != 2 or (
                lhs_type.shape[1] != rhs_type.shape[0]
                or result_type.shape != (lhs_type.shape[0], rhs_type.shape[1])
                or init_type != result_type
                or result_type.dtype != "i32"
            ):
                raise LinalgBridgeError("matmul shape or initialized result type differs")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "matmul_accumulate", tuple(env[value] for value in (lhs, rhs, init)),
                result_type, index_maps=_maps(),
            ))
            env[op.results[0]] = node_id
        elif op.name == "func.return":
            if ordinal != len(operations) - 1 or not op.operands:
                raise LinalgBridgeError("func.return must be final and return tensors")
            if tuple(value.type for value in op.operands) != tuple(function.function_type.outputs.data):
                raise LinalgBridgeError("returned tensor types differ from the entry signature")
            outputs = tuple(env.get(value) for value in op.operands)
            if any(output is None for output in outputs):
                raise LinalgBridgeError("function returns an undefined or uninitialized tensor")
            request = KernelRequest(
                nodes=tuple(nodes), outputs=outputs, output_storages=(boundary_storage,) * len(outputs),
                input_storages=tuple(boundaries), target_identity=target_identity,
                lowering_policy=lowering_policy, source_identity=sha256(source.encode()).hexdigest(),
                constants=tuple(constant_bindings),
            )
            return LinalgTranslation(request, constants, tuple(source_operations))
        else:
            raise LinalgBridgeError(f"no admitted translation for source operation {op.name}")
    raise LinalgBridgeError("entry function has no return")
