"""Independent evaluator for the explicitly admitted exact-i32 graph subset.

This is a verification tool. Native selection never calls it, and unknown
numerical policies or operations fail instead of acquiring guessed semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .model import IndexMap, KernelRequest, TensorType

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _i32(value: int) -> int:
    if type(value) is not int or not _I32_MIN <= value <= _I32_MAX:
        raise ValueError("exact-i32 value is outside its admitted domain")
    return value


def _wrap_i32(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def _matmul_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


@dataclass(frozen=True)
class TensorValue:
    type: TensorType
    elements: tuple[int, ...]

    def __post_init__(self) -> None:
        admitted = {("i32", "exact-i32"), ("i32", "i32-wrap-k-ascending"), ("i8", "signed-i8")}
        if (self.type.dtype, self.type.numerical_policy) not in admitted:
            raise ValueError("reference evaluator has no admitted numerical policy")
        if len(self.elements) != prod(self.type.shape):
            raise ValueError("reference tensor size differs from its type")
        for element in self.elements:
            if self.type.dtype == "i8":
                if type(element) is not int or not -128 <= element <= 127:
                    raise ValueError("signed-i8 value is outside its admitted domain")
            else:
                _i32(element)


def evaluate_graph(
    request: KernelRequest,
    inputs: dict[str, TensorValue],
    *,
    constants: dict[str, TensorValue] | None = None,
) -> tuple[TensorValue, ...]:
    """Evaluate the full ordered output tuple with no selector or target code."""
    constants = constants or {}
    expected_inputs = {node.id for node in request.nodes if node.effect == "input"}
    expected_constants = {node.id for node in request.nodes if node.effect == "constant"}
    if set(inputs) != expected_inputs or set(constants) != expected_constants:
        raise ValueError("reference inputs or declared constants differ from the graph boundary")
    bindings = {binding.node_id: binding for binding in request.constants}
    values: dict[str, TensorValue] = {}
    for node in request.nodes:
        if node.effect in {"input", "constant"}:
            value = inputs[node.id] if node.effect == "input" else constants[node.id]
            if value.type != node.type:
                raise ValueError(f"reference boundary type differs for {node.id}")
            if node.effect == "constant":
                binding = bindings[node.id]
                width = 1 if binding.encoding == "i8" else 4
                expected = b"".join(element.to_bytes(width, "little", signed=True) for element in value.elements)
                if expected.hex() != binding.data_hex:
                    raise ValueError(f"reference constant differs from declared compiler bytes for {node.id}")
            values[node.id] = value
            continue
        if node.effect != "pure":
            raise ValueError(f"reference evaluator has no semantics for {node.id}")
        operands = tuple(values[child] for child in node.inputs)
        if node.op == "matmul_accumulate":
            if node.attrs or node.index_maps != _matmul_maps() or len(operands) != 3:
                raise ValueError("reference matmul has unsupported attributes or indexing maps")
            lhs, rhs, init = operands
            if node.type.dtype != "i32" or node.type.numerical_policy != "i32-wrap-k-ascending":
                raise ValueError("reference matmul has an unsupported numerical policy")
            expected_input_policy = "signed-i8" if lhs.type.dtype == "i8" else "i32-wrap-k-ascending"
            if (lhs.type.dtype not in {"i8", "i32"}
                    or lhs.type.numerical_policy != expected_input_policy
                    or rhs.type.dtype != lhs.type.dtype
                    or rhs.type.numerical_policy != expected_input_policy
                    or init.type != node.type):
                raise ValueError("reference matmul operand type or policy mismatch")
            if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
                raise ValueError("reference matmul needs rank-two operands")
            m, k = lhs.type.shape
            right_k, n = rhs.type.shape
            if right_k != k or node.type.shape != (m, n):
                raise ValueError("reference matmul shape mismatch")
            result = []
            for row in range(m):
                for col in range(n):
                    acc = init.elements[row * n + col]
                    for reduction in range(k):
                        product = _wrap_i32(lhs.elements[row * k + reduction] * rhs.elements[reduction * n + col])
                        acc = _wrap_i32(acc + product)
                    result.append(acc)
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.index_maps or node.attrs:
            raise ValueError(f"reference evaluator has no semantics for {node.id}")
        if node.type.dtype != "i32" or node.type.numerical_policy != "exact-i32":
            raise ValueError(f"reference evaluator has no numerical policy for {node.id}")
        if node.op == "identity" and len(operands) == 1 and operands[0].type == node.type:
            result = operands[0].elements
        elif node.op in {"add", "multiply"} and len(operands) == 2 and all(
            operand.type == node.type for operand in operands
        ):
            operation = int.__add__ if node.op == "add" else int.__mul__
            result = tuple(_i32(operation(left, right)) for left, right in zip(
                operands[0].elements, operands[1].elements,
            ))
        elif node.op == "matmul" and len(operands) == 2:
            left, right = operands
            if len(left.type.shape) != 2 or len(right.type.shape) != 2:
                raise ValueError("exact-i32 matmul needs rank-two operands")
            m, k = left.type.shape
            right_k, n = right.type.shape
            if right_k != k or node.type.shape != (m, n):
                raise ValueError("exact-i32 matmul shape mismatch")
            if any(operand.type.dtype != "i32" or operand.type.numerical_policy != "exact-i32"
                   for operand in operands):
                raise ValueError("exact-i32 matmul operand policy mismatch")
            result = tuple(
                _i32(sum(left.elements[row * k + reduction] * right.elements[reduction * n + col]
                         for reduction in range(k)))
                for row in range(m) for col in range(n)
            )
        else:
            raise ValueError(f"reference evaluator has no semantics for {node.op}")
        values[node.id] = TensorValue(node.type, result)
    return tuple(values[output] for output in request.outputs)
