"""Independent evaluator for the explicitly admitted exact-i32 graph subset.

This is a verification tool. Native selection never calls it, and unknown
numerical policies or operations fail instead of acquiring guessed semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .model import KernelRequest, TensorType

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _i32(value: int) -> int:
    if type(value) is not int or not _I32_MIN <= value <= _I32_MAX:
        raise ValueError("exact-i32 value is outside its admitted domain")
    return value


@dataclass(frozen=True)
class TensorValue:
    type: TensorType
    elements: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.type.dtype != "i32" or self.type.numerical_policy != "exact-i32":
            raise ValueError("reference evaluator admits only exact-i32 tensors")
        if len(self.elements) != prod(self.type.shape):
            raise ValueError("reference tensor size differs from its type")
        for element in self.elements:
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
    values: dict[str, TensorValue] = {}
    for node in request.nodes:
        if node.effect in {"input", "constant"}:
            value = inputs[node.id] if node.effect == "input" else constants[node.id]
            if value.type != node.type:
                raise ValueError(f"reference boundary type differs for {node.id}")
            values[node.id] = value
            continue
        if node.effect != "pure" or node.index_maps or node.attrs:
            raise ValueError(f"reference evaluator has no semantics for {node.id}")
        if node.type.dtype != "i32" or node.type.numerical_policy != "exact-i32":
            raise ValueError(f"reference evaluator has no numerical policy for {node.id}")
        operands = tuple(values[child] for child in node.inputs)
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
