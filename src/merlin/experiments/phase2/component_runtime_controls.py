"""Private primitive transport controls, never a Phase 1 compiler seed.

This small independent author produces static f32 identity/add programs solely
to test the shared source/build/readback path. Symbolic source correspondence is
checked independently against the parsed LLVM SSA memory operations. Unsupported
operations refuse. No accelerator operation, ISA rule, schedule or hardware
qualification is supplied here, and this module is withheld from author views.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrimitiveProgram:
    shape: tuple[int, ...]
    inputs: int
    outputs: tuple[tuple, ...]

    @property
    def count(self):
        return math.prod(self.shape)


def parse_primitive(source: str) -> PrimitiveProgram:
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func
    from xdsl.parser import Parser

    context = Context()
    for dialect in (builtin.Builtin, func.Func, arith.Arith):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if (
        module.attributes
        or module.properties
        or len(functions) != 1
        or type(functions[0]) is not func.FuncOp
        or len(functions[0].body.blocks) != 1
    ):
        raise ValueError("private primitive control requires one closed straight-line tensor function")
    function = functions[0]
    if function.attributes or set(function.properties) != {"sym_name", "function_type"}:
        raise ValueError("private primitive control has unproved function properties")
    types = (*function.function_type.inputs.data, *function.function_type.outputs.data)
    if not types or any(
        not isinstance(value, builtin.TensorType)
        or value.get_element_type() != builtin.f32
        or not isinstance(value.encoding, builtin.NoneAttr)
        for value in types
    ):
        raise ValueError("private primitive control accepts only explicit f32 tensor types")
    shape = tuple(types[0].get_shape())
    if not shape or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("private primitive control requires positive static extents")
    if any(tuple(value.get_shape()) != shape for value in types):
        raise ValueError("private primitive control requires equal explicit tensor shapes")
    expressions = {value: ("input", index) for index, value in enumerate(function.body.block.args)}
    operations = tuple(function.body.block.ops)
    for op in operations[:-1]:
        if (
            type(op) is not arith.AddfOp
            or op.attributes
            or set(op.properties) != {"fastmath"}
            or op.fastmath.data
            or any(value not in expressions for value in op.operands)
        ):
            raise ValueError("private primitive source contains an unsupported operation or arithmetic flag")
        expressions[op.result] = ("add", *(expressions[value] for value in op.operands))
    if (
        not operations
        or type(operations[-1]) is not func.ReturnOp
        or operations[-1].attributes
        or operations[-1].properties
    ):
        raise ValueError("private primitive control has no complete tensor return")
    returned = tuple(expressions[value] for value in operations[-1].operands)
    if not returned:
        raise ValueError("private primitive control must expose every declared output")
    return PrimitiveProgram(shape, len(function.body.block.args), returned)


def primitive_buffer(program: PrimitiveProgram, target: str) -> dict:
    inputs = ["arg" + str(index) for index in range(program.inputs)]
    outputs = ["out" + str(index) for index in range(len(program.outputs))]
    return {
        "abi_version": "0.1",
        "target": target,
        "commands": [],
        "operand_naming": "positional",
        "tensors": {
            name: {"shape": list(program.shape), "dtype": "f32", "role": "input" if name in inputs else "output"}
            for name in (*inputs, *outputs)
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "read" if name in inputs else "write"} for name in (*inputs, *outputs)],
            "outputs": outputs,
        },
    }


def emit_primitive_llvm(program: PrimitiveProgram, entry_symbol: str) -> str:
    if (
        not entry_symbol
        or not entry_symbol.isidentifier()
        or not entry_symbol.isascii()
        or type(program) is not PrimitiveProgram
    ):
        raise ValueError("private primitive LLVM requires a typed program and explicit plain entry symbol")
    args = ["%ptr" + str(index) for index in range(program.inputs + len(program.outputs))]
    lines = [
        "module {",
        "  llvm.func @" + entry_symbol + "(" + ", ".join(name + ": !llvm.ptr" for name in args) + ") {",
    ]
    serial = 0

    def fresh():
        nonlocal serial
        serial += 1
        return "%v" + str(serial)

    def expression(expr, element):
        value = fresh()
        if expr[0] == "input":
            address = fresh()
            lines.append(
                f"    {address} = llvm.getelementptr {args[expr[1]]}[{element}] : (!llvm.ptr) -> !llvm.ptr, f32"
            )
            lines.append(f"    {value} = llvm.load {address} : !llvm.ptr -> f32")
        elif expr[0] == "add":
            lhs, rhs = expression(expr[1], element), expression(expr[2], element)
            lines.append(f"    {value} = llvm.fadd {lhs}, {rhs} : f32")
        else:
            raise ValueError("private primitive LLVM cannot lower this source expression")
        return value

    for element in range(program.count):
        for output, expr in enumerate(program.outputs):
            value, address = expression(expr, element), fresh()
            lines.append(
                f"    {address} = llvm.getelementptr {args[program.inputs + output]}[{element}] : "
                "(!llvm.ptr) -> !llvm.ptr, f32"
            )
            lines.append(f"    llvm.store {value}, {address} : f32, !llvm.ptr")
    return "\n".join((*lines, "    llvm.return", "  }", "}")) + "\n"


def verify_primitive_llvm(source: str, lowered: str, *, entry_symbol: str) -> dict:
    """Symbolically join each scalar read, arithmetic expression and output store.

    The accepted closed subset contains no calls, allocations, loops, globals,
    inline instructions or asynchronous operations. This is a source/LLVM proof
    for that subset; it grants no object, target ISA or physical runtime proof.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    program = parse_primitive(source)
    context = Context()
    for dialect in (builtin.Builtin, llvm.LLVM):
        context.load_dialect(dialect)
    module = Parser(context, lowered).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if (
        module.attributes
        or module.properties
        or len(functions) != 1
        or type(functions[0]) is not llvm.FuncOp
        or functions[0].sym_name.data != entry_symbol
        or len(functions[0].body.blocks) != 1
        or len(functions[0].body.block.args) != program.inputs + len(program.outputs)
    ):
        raise ValueError("private primitive LLVM has no exact single pointer entry")
    block = functions[0].body.block
    function = functions[0]
    if (
        function.attributes
        or set(function.properties) != {"unnamed_addr", "sym_name", "function_type", "CConv", "linkage", "visibility_"}
        or function.CConv.convention.data != "ccc"
        or function.linkage.linkage.data != "external"
        or function.function_type.output != llvm.LLVMVoidType()
        or not isinstance(function.function_type.variadic, builtin.NoneAttr)
    ):
        raise ValueError("private primitive LLVM has unproved entry properties")
    if any(value.type != llvm.LLVMPointerType() for value in block.args):
        raise ValueError("private primitive LLVM changes the pointer entry ABI")
    values = {value: ("pointer", index, 0) for index, value in enumerate(block.args)}
    stores = {}
    operations = tuple(block.ops)
    for op in operations:
        if op.attributes:
            raise ValueError("private primitive LLVM contains unproved operation attributes")
        if op.name == "llvm.getelementptr":
            indices = tuple(op.rawConstantIndices.iter_values())
            if (
                op.ssa_indices
                or len(indices) != 1
                or op.elem_type != builtin.f32
                or set(op.properties) != {"rawConstantIndices", "elem_type", "noWrapFlags"}
                or op.noWrapFlags.value.data
            ):
                raise ValueError("private primitive LLVM requires exact scalar constant addressing")
            pointer = values.get(op.ptr)
            if not pointer or pointer[0] != "pointer":
                raise ValueError("private primitive LLVM loses a pointer's source ownership")
            address = pointer[2] + indices[0]
            if not 0 <= address < program.count:
                raise ValueError("private primitive LLVM accesses beyond the original logical extent")
            values[op.result] = ("pointer", pointer[1], address)
        elif op.name == "llvm.load":
            pointer = values.get(op.ptr)
            if (
                not pointer
                or pointer[0] != "pointer"
                or pointer[1] >= program.inputs
                or op.dereferenced_value.type != builtin.f32
                or set(op.properties) != {"ordering"}
                or op.ordering.value.data
            ):
                raise ValueError("private primitive LLVM reads undefined or unowned mutable storage")
            values[op.dereferenced_value] = ("input", pointer[1], pointer[2])
        elif op.name == "llvm.fadd":
            if (
                op.fastmathFlags.data
                or set(op.properties) != {"fastmathFlags"}
                or op.res.type != builtin.f32
                or any(value not in values for value in op.operands)
            ):
                raise ValueError("private primitive LLVM has unproved arithmetic flags or operands")
            values[op.res] = ("add", *(values[value] for value in op.operands))
        elif op.name == "llvm.store":
            pointer = values.get(op.ptr)
            if (
                not pointer
                or pointer[0] != "pointer"
                or pointer[1] < program.inputs
                or op.value not in values
                or set(op.properties) != {"ordering"}
                or op.ordering.value.data
            ):
                raise ValueError("private primitive LLVM writes an original input or an unowned address")
            slot = (pointer[1] - program.inputs, pointer[2])
            if slot in stores:
                raise ValueError("private primitive LLVM gives one output multiple source writers")
            stores[slot] = values[op.value]
        elif op.name == "llvm.return" and op is operations[-1] and not op.operands and not op.properties:
            pass
        else:
            raise ValueError("private primitive LLVM contains an unqualified instruction: " + op.name)

    def at(expr, element):
        return (
            ("input", expr[1], element) if expr[0] == "input" else ("add", at(expr[1], element), at(expr[2], element))
        )

    expected = {
        (index, element): at(expr, element)
        for index, expr in enumerate(program.outputs)
        for element in range(program.count)
    }
    if stores != expected:
        raise ValueError("private primitive LLVM does not preserve every original output expression")
    return {
        "scope": "closed primitive source/LLVM correspondence only; object/ISA/runtime/effects unqualified",
        "inputs": program.inputs,
        "outputs": len(program.outputs),
        "shape": list(program.shape),
        "logical_output_stores": len(stores),
        "operations": len(operations),
    }


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    command, source_path, target, entry_symbol, *rest = arguments
    source = Path(source_path).read_text()
    program = parse_primitive(source)
    if command == "parse":
        return
    if command == "lower_interface_to_target":
        sys.stdout.write(source)
    elif command == "emit_command_buffer" and len(rest) == 1:
        Path(rest[0]).write_text(json.dumps(primitive_buffer(program, target), sort_keys=True) + "\n")
    elif command == "lower_target_to_llvm":
        sys.stdout.write(emit_primitive_llvm(program, entry_symbol))
    else:
        raise ValueError("private primitive control received an unsupported ABI command")


if __name__ == "__main__":
    main()
