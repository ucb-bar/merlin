"""``merlin-lower-inline-asm`` — custom ISA with no LLVM fork.

The Merlin thesis for accelerator/custom instructions: a Merlin op lowers **1:1** to an
``llvm.inline_asm`` (or ``llvm.call_intrinsic``), so a custom instruction the compiler has
no intrinsic for still lands in the binary — without modifying llvm-project. The escape
hatch for a truly novel encoding (e.g. a Saturn vcix instruction) is the assembler ``.insn``
directive inside the inline asm: the toolchain emits the exact word it is told to, even
though it cannot name the mnemonic.

This module provides:

- :func:`lower_inline_asm` — the pass: rewrite each ``merlin.inline_asm`` marker op (carried
  through the dialects as an unregistered op with ``asm_string`` / ``constraints`` attrs)
  into a real ``llvm.InlineAsmOp`` at the LLVM edge.
- :func:`inline_asm_function` / :func:`build_rvv_object` — build a standalone llvm-dialect
  function around one ``llvm.inline_asm`` and compile it to an rv64gcv object, so the
  emitted (custom) instruction can be confirmed in the disassembly.

Standard rv64gcv needs none of this (clang auto-vectorizes); it is the on-ramp for Saturn
custom instructions and hand-placed RVV sequences.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import codegen
from .pipeline import lower_to_llvm_ir

MERLIN_INLINE_ASM = "merlin.inline_asm"


def lower_inline_asm(module) -> int:
    """Rewrite every ``merlin.inline_asm`` marker op into a real ``llvm.inline_asm``.

    The marker carries ``asm_string`` and ``constraints`` string attributes (and an optional
    ``has_side_effects`` unit attr); operands and result types pass through unchanged.
    Returns the number rewritten.
    """
    from xdsl.dialects.llvm import InlineAsmOp

    rewrites = []
    for op in module.walk():
        name = getattr(op, "op_name", None)
        if op.name == "builtin.unregistered" and name is not None \
                and name.data == MERLIN_INLINE_ASM:
            rewrites.append(op)

    for op in rewrites:
        asm = op.attributes["asm_string"].data
        cons = op.attributes["constraints"].data
        side = "has_side_effects" in op.attributes
        res_types = [r.type for r in op.results]
        new = InlineAsmOp(asm, cons, list(op.operands), res_types or None,
                          has_side_effects=side)
        block = op.parent_block()
        block.insert_op_before(new, op)
        for old, fresh in zip(op.results, new.results):
            old.replace_all_uses_with(fresh)
        block.detach_op(op)
    return len(rewrites)


def inline_asm_function(name: str, asm: str, constraints: str,
                        arg_types: list[str], res_type: str | None,
                        has_side_effects: bool = True) -> str:
    """An llvm-dialect MLIR module: ``@name`` computed by one ``llvm.inline_asm`` (1:1)."""
    args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(arg_types))
    operands = ", ".join(f"%a{i}" for i in range(len(arg_types)))
    side = "has_side_effects " if has_side_effects else ""
    in_tys = ", ".join(arg_types)
    if res_type is None:
        body = (f'    llvm.inline_asm {side}"{asm}", "{constraints}" {operands} '
                f': ({in_tys}) -> ()\n'
                f'    llvm.return\n')
        sig = f"({args})"
    else:
        body = (f'    %r = llvm.inline_asm {side}"{asm}", "{constraints}" {operands} '
                f': ({in_tys}) -> {res_type}\n'
                f'    llvm.return %r : {res_type}\n')
        sig = f"({args}) -> {res_type}"
    return f"module {{\n  llvm.func @{name}{sig} {{\n{body}  }}\n}}\n"


def build_rvv_object(name: str, asm: str, constraints: str, arg_types: list[str],
                     res_type: str | None, workdir: str | Path,
                     has_side_effects: bool = True) -> Path:
    """Lower an inline-asm function to LLVM IR and compile it to an rv64gcv object."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = inline_asm_function(name, asm, constraints, arg_types, res_type, has_side_effects)
    ll = lower_to_llvm_ir(text, workdir=workdir)
    (workdir / f"{name}.ll").write_text(ll, encoding="utf-8")
    return Path(codegen.compile_ll(workdir / f"{name}.ll", workdir / f"{name}.o", "riscv"))


def disassemble(obj: str | Path) -> str:
    """objdump -d of an rv64gcv object (via the chipyard toolchain)."""
    from ..runtime.backends import spike

    objdump = spike.gcc_path().with_name("riscv64-unknown-elf-objdump")
    return subprocess.run([str(objdump), "-d", str(obj)],
                          capture_output=True, text=True).stdout
