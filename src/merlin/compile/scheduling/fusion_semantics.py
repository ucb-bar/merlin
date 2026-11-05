"""Typed scalar MLIR correspondence for exact elementwise composition.

This is an immutable verification DAG, not another arithmetic language. The
operation meanings and widths remain those of the actual scalar MLIR. Only
total integer operations are admitted; no reassociation, approximation or
floating-environment policy is inferred from a target capability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache


class UnsupportedElementwise(ValueError):
    """Keep the original operation: its scalar semantics are outside this family."""


class ScalarBudgetExceeded(UnsupportedElementwise):
    """A candidate exceeds a selected compile-time bound; retain its source."""


@dataclass(frozen=True)
class ElementwiseSemantics:
    """Qualified contiguous elementwise execution with ordinary byte memory.

    Every read access is one function argument, in access order; the sole final
    write is the scalar return. Every access has exactly `elements` values.
    All arguments of an element are read before writing that element. There
    are no hidden effects, volatile accesses, state, traps or cross-element
    dependencies. Exact in-place aliases are allowed; shifted aliases are not.
    These execution properties require independent adapter qualification.
    """

    function_ir: bytes
    elements: int
    stages: tuple[str, ...]
    byte_orders: tuple[str, ...]
    readout: str | None = None

    def __post_init__(self):
        if (
            type(self.function_ir) is not bytes
            or not self.function_ir
            or type(self.elements) is not int
            or self.elements <= 0
            or type(self.stages) is not tuple
            or not self.stages
            or any(not isinstance(stage, str) or not stage for stage in self.stages)
            or type(self.byte_orders) is not tuple
            or not self.byte_orders
            or any(not isinstance(order, str) or order not in {"little", "big"} for order in self.byte_orders)
            or self.readout is not None
            and (not isinstance(self.readout, str) or not self.readout)
        ):
            raise ValueError(
                "elementwise semantics require typed scalar IR, extent, ordered stages and explicit byte layouts"
            )


@dataclass(frozen=True)
class ScalarGraph:
    input_bits: tuple[int, ...]
    output_bits: int
    # Typed MLIR applications with SSA operands numbered after the arguments.
    operations: tuple[tuple[str, str, tuple, tuple[int, ...]], ...]
    returned: int

    def expression(self, arguments):
        if len(arguments) != len(self.input_bits):
            raise ValueError("scalar expression argument count differs from its actual typed function")
        values = list(arguments)
        for name, dtype, properties, operands in self.operations:
            values.append(term((name, dtype, properties, tuple(values[index] for index in operands))))
        return values[self.returned]


def term(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=128)
def scalar_graph(source: bytes, max_operations: int, max_ir_bytes: int) -> ScalarGraph:
    """Reopen actual scalar IR; cache only an immutable, byte-bound typed DAG."""
    if type(max_operations) is not int or max_operations <= 0 or type(max_ir_bytes) is not int or max_ir_bytes <= 0:
        raise ValueError("scalar fusion requires positive explicit operation and source-byte budgets")
    if len(source) > max_ir_bytes:
        raise ScalarBudgetExceeded("scalar function exceeds its selected source-byte budget")
    from xdsl.dialects import arith, builtin, func

    from merlin.transforms.certificates import parse_ir
    from merlin.transforms.pure_scalar import eligible_scalar_operation

    module = parse_ir(source.decode("utf-8"))
    functions = tuple(module.body.block.ops)
    if module.attributes or module.properties or len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise UnsupportedElementwise("elementwise computation requires exactly one plain scalar function")
    function = functions[0]
    if (
        function.attributes
        or set(function.properties) - {"sym_name", "function_type", "sym_visibility"}
        or len(function.body.blocks) != 1
        or len(function.function_type.outputs) != 1
    ):
        raise UnsupportedElementwise("elementwise function requires a plain single-block single-result definition")
    block = function.body.block

    def bits(typ, *, memory=False):
        if (
            not isinstance(typ, builtin.IntegerType)
            or typ.signedness.data != builtin.Signedness.SIGNLESS
            or not 1 <= typ.width.data <= 256
            or memory
            and typ.width.data % 8
        ):
            raise UnsupportedElementwise(
                "scalar fusion supports total fixed-width integers with byte-sized memory elements"
            )
        return int(typ.width.data)

    inputs = tuple(bits(arg.type, memory=True) for arg in block.args)
    output = bits(function.function_type.outputs.data[0], memory=True)
    ops = tuple(block.ops)
    if len(ops) > max_operations + 1:
        raise ScalarBudgetExceeded("scalar function exceeds its selected operation budget")
    if not ops or type(ops[-1]) is not func.ReturnOp or len(ops[-1].operands) != 1:
        raise UnsupportedElementwise("elementwise scalar function needs one explicit return")
    if ops[-1].attributes or ops[-1].properties:
        raise UnsupportedElementwise("scalar return has unmodeled metadata")
    values = {value: index for index, value in enumerate(block.args)}
    applications = []
    extra = (
        arith.CmpiOp,
        arith.AndIOp,
        arith.OrIOp,
        arith.ShRSIOp,
        arith.SelectOp,
        arith.MinSIOp,
        arith.MaxSIOp,
        arith.MinUIOp,
        arith.MaxUIOp,
        arith.ExtSIOp,
        arith.ExtUIOp,
        arith.TruncIOp,
    )
    # Some xDSL versions omit effect traits on these standard scalar casts.
    # Admit only these exact, total builtin classes, never an operation merely
    # sharing their namespace. This does not broaden the frozen scalar passes.
    for op in ops[:-1]:
        admitted = eligible_scalar_operation(op)
        if type(op) in extra:
            admitted = (
                not op.attributes
                and not op.regions
                and not op.successors
                and len(op.results) == 1
                and set(op.properties) <= ({"predicate"} if type(op) is arith.CmpiOp else set())
            )
            if type(op) is arith.CmpiOp:
                admitted = (
                    admitted and isinstance(op.predicate, builtin.IntegerAttr) and 0 <= op.predicate.value.data <= 9
                )
        if type(op) is arith.ShRSIOp:
            # A shift is total only when its actual SSA amount is a constant
            # within the operand width; dynamic/oversized shifts can be poison.
            owner = op.operands[1].owner
            admitted = (admitted and type(owner) is arith.ConstantOp
                        and eligible_scalar_operation(owner)
                        and isinstance(owner.value, builtin.IntegerAttr)
                        and 0 <= owner.value.value.data < bits(op.operands[0].type))
        if not admitted:
            raise UnsupportedElementwise(f"unmodeled or non-total scalar operation {op.name}")
        for value in (*op.operands, *op.results):
            bits(value.type)
        if any(value not in values for value in op.operands):
            raise UnsupportedElementwise("scalar function reads an external or undefined SSA value")
        properties = tuple(sorted((name, str(value)) for name, value in op.properties.items()))
        applications.append(
            (op.name, str(op.results[0].type), properties, tuple(values[value] for value in op.operands))
        )
        values[op.results[0]] = len(values)
    returned = ops[-1].operands[0]
    if returned not in values or bits(returned.type, memory=True) != output:
        raise UnsupportedElementwise("scalar return differs from its declared result")
    return ScalarGraph(inputs, output, tuple(applications), values[returned])
