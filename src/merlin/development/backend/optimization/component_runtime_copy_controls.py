"""Evaluator-only tensor-copy source/callee controls; no helper semantics or seed.

The source and LLVM checks bind ordered static tensor inputs and returns to an
explicitly selected external pointer ABI. The separately selected OOT helper
owns device instructions and effects. A matching call never proves that helper.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyProgram:
    shape: tuple[int, ...]
    dtype: str
    inputs: int
    outputs: tuple[int, ...]


def _symbol(value):
    if not isinstance(value, str) or not value.isascii() or not value.isidentifier():
        raise ValueError("copy control requires an explicit plain external ABI symbol")
    return value


def parse_copy(source: str) -> CopyProgram:
    from xdsl.context import Context
    from xdsl.dialects import builtin, func, linalg, tensor
    from xdsl.dialects.linalg.ops import CopyOp, YieldOp
    from xdsl.parser import Parser

    context = Context()
    for dialect in (builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if module.attributes or module.properties or len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise ValueError("copy control requires one original closed tensor function")
    function = functions[0]
    if (
        function.attributes
        or set(function.properties) != {"sym_name", "function_type"}
        or len(function.body.blocks) != 1
        or not function.body.block.args
    ):
        raise ValueError("copy control has unproved source function properties")
    types = (*function.function_type.inputs.data, *function.function_type.outputs.data)
    if not types or any(
        not isinstance(value, builtin.TensorType)
        or not isinstance(value.encoding, builtin.NoneAttr)
        or not isinstance(value.get_element_type(), builtin.IntegerType)
        for value in types
    ):
        raise ValueError("copy control requires explicit unencoded integer tensors")
    shape, dtype = tuple(types[0].get_shape()), str(types[0].get_element_type())
    if not shape or any(dim < 1 for dim in shape) or any(value != types[0] for value in types):
        raise ValueError("copy control requires equal positive static tensor types")
    inputs = {value: index for index, value in enumerate(function.body.block.args)}
    empty, copies = set(), {}
    ops = tuple(function.body.block.ops)
    for op in ops[:-1]:
        if op.attributes:
            raise ValueError("copy control has unproved source operation attributes")
        if type(op) is tensor.EmptyOp and not op.operands and not op.properties and len(op.results) == 1:
            empty.add(op.results[0])
        elif type(op) is CopyOp:
            if (
                set(op.properties) != {"operandSegmentSizes"}
                or len(op.inputs) != 1
                or len(op.outputs) != 1
                or len(op.res) != 1
                or op.inputs[0] not in inputs
                or op.outputs[0] not in empty
                or len(op.regions) != 1
                or len(op.regions[0].blocks) != 1
            ):
                raise ValueError("copy control changes original source/destination ownership")
            body = op.regions[0].block
            body_ops = tuple(body.ops)
            if (
                len(body.args) != 2
                or len(body_ops) != 1
                or type(body_ops[0]) is not YieldOp
                or tuple(body_ops[0].operands) != (body.args[0],)
                or body_ops[0].attributes
                or body_ops[0].properties
            ):
                raise ValueError("copy control changes the registered scalar copy relation")
            copies[op.res[0]] = inputs[op.inputs[0]]
        else:
            raise ValueError("copy control contains an unsupported source operation")
    if (
        not ops
        or type(ops[-1]) is not func.ReturnOp
        or ops[-1].attributes
        or ops[-1].properties
        or any(value not in copies for value in ops[-1].operands)
    ):
        raise ValueError("copy control omits its original ordered tensor returns")
    returned = tuple(copies[value] for value in ops[-1].operands)
    if not returned or set(ops[-1].operands) != set(copies):
        raise ValueError("copy control has incomplete or unobserved copy results")
    return CopyProgram(shape, dtype, len(inputs), returned)


def copy_buffer(program: CopyProgram, target: str):
    inputs = ["arg" + str(i) for i in range(program.inputs)]
    outputs = ["out" + str(i) for i in range(len(program.outputs))]
    return {
        "abi_version": "0.1",
        "target": target,
        "commands": [],
        "operand_naming": "positional",
        "tensors": {
            name: {
                "shape": list(program.shape),
                "dtype": program.dtype,
                "role": "input" if name in inputs else "output",
            }
            for name in (*inputs, *outputs)
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "read" if name in inputs else "write"} for name in (*inputs, *outputs)],
            "outputs": outputs,
        },
    }


def emit_copy_llvm(program: CopyProgram, *, entry_symbol: str, callee_symbol: str):
    _symbol(entry_symbol)
    _symbol(callee_symbol)
    if entry_symbol == callee_symbol:
        raise ValueError("copy control entry and helper must be distinct")
    args = ["%p" + str(i) for i in range(program.inputs + len(program.outputs))]
    lines = [
        "module {",
        "  llvm.func @" + callee_symbol + "(!llvm.ptr, !llvm.ptr)",
        "  llvm.func @" + entry_symbol + "(" + ", ".join(p + ": !llvm.ptr" for p in args) + ") {",
    ]
    for ordinal, original in enumerate(program.outputs):
        lines.append(
            "    llvm.call @"
            + callee_symbol
            + "("
            + args[original]
            + ", "
            + args[program.inputs + ordinal]
            + ") : (!llvm.ptr, !llvm.ptr) -> ()"
        )
    return "\n".join((*lines, "    llvm.return", "  }", "}")) + "\n"


def verify_copy_llvm(source, lowered, *, entry_symbol, callee_symbol):
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    program = parse_copy(source)
    expected = Parser(
        _llvm_context(), emit_copy_llvm(program, entry_symbol=entry_symbol, callee_symbol=callee_symbol)
    ).parse_module()
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    actual = Parser(context, lowered).parse_module()
    actual.verify()
    # Structural equivalence tolerates SSA names but preserves all properties,
    # declarations, call operands, arithmetic flags and complete ordered returns.
    if not actual.is_structurally_equivalent(expected):
        raise ValueError("copy LLVM does not preserve the original complete pointer/callee relation")
    return {
        "shape": list(program.shape),
        "dtype": program.dtype,
        "inputs": program.inputs,
        "outputs": len(program.outputs),
        "ordered_return_inputs": list(program.outputs),
        "external_callee": callee_symbol,
        "helper_semantics": "UNKNOWN",
        "scope": "registered source/LLVM call relation only; helper/object/ISA/effects/runtime unqualified",
    }


def _llvm_context():
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm

    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    return context


def main():
    command, filename, target, entry, callee, *rest = sys.argv[1:]
    source = Path(filename).read_text()
    program = parse_copy(source)
    if command == "parse":
        return
    if command == "lower_interface_to_target":
        sys.stdout.write(source)
    elif command == "emit_command_buffer" and len(rest) == 1:
        Path(rest[0]).write_text(json.dumps(copy_buffer(program, target), sort_keys=True) + "\n")
    elif command == "lower_target_to_llvm":
        sys.stdout.write(emit_copy_llvm(program, entry_symbol=entry, callee_symbol=callee))
    else:
        raise ValueError("unknown private ordinary copy control entrypoint")


if __name__ == "__main__":
    main()
