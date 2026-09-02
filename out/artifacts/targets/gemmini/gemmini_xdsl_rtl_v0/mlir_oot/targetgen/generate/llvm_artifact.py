"""Emit canonical LLVM-dialect MLIR with pointer-derived RoCC operands."""

from __future__ import annotations

from ir_ingest import InterfaceProgram
from lowering.isa import Address, Instruction, build_trace


def _normalized_artifact_program(program: InterfaceProgram) -> InterfaceProgram:
    """Keep artifact lowering on interface tensors and interface commands.

    Derived command-buffer tensors are an L0 semantic device, not kernel ABI
    arguments.  The hardware emitter gathers layouts itself from the original
    pointer arguments.
    """
    return program


def _kernel_argument_order(program: InterfaceProgram) -> list[str]:
    # Tensor specs are inserted while walking interface declarations, with
    # produced outputs appended at their defining operation.  Preserve that
    # order exactly: the bare-metal harness passes interface pointers, not an
    # instruction-selection-specific weight/lhs ordering.
    return [name for name, spec in program.tensors.items()
            if spec.role in ("input", "weight", "bias", "output")]


def emit_llvm_artifact(program: InterfaceProgram) -> str:
    program = _normalized_artifact_program(program)
    tensors = _kernel_argument_order(program)
    args = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(len(tensors)))
    tensor_arg = {name: i for i, name in enumerate(tensors)}
    lines = ["module {", f"  llvm.func @gemmini_kernel({args}) {{"]
    bases: dict[str, str] = {}
    value_id = 0

    def new_value(prefix: str = "c") -> str:
        nonlocal value_id
        name = f"%{prefix}{value_id}"
        value_id += 1
        return name

    for name, index in tensor_arg.items():
        value = new_value("p")
        lines.append(f"    {value} = llvm.ptrtoint %arg{index} : !llvm.ptr to i64")
        bases[name] = value

    def operand(value: int | Address | None) -> str:
        if isinstance(value, Address):
            base = bases[value.tensor]
            if value.offset == 0:
                return base
            const = new_value()
            result = new_value("a")
            lines.append(f"    {const} = llvm.mlir.constant({value.offset} : i64) : i64")
            lines.append(f"    {result} = llvm.add {base}, {const} : i64")
            return result
        const = new_value()
        lines.append(f"    {const} = llvm.mlir.constant({int(value or 0)} : i64) : i64")
        return const

    for ins in build_trace(program):
        if ins.name == "FENCE":
            lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        rs1, rs2 = operand(ins.rs1), operand(ins.rs2)
        lines.append(
            f'    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 0x{ins.funct:x}, x0, $0, $1", '
            f'"r,r" {rs1}, {rs2} : (i64, i64) -> ()'
        )
    lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')

    lines.append("    llvm.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
