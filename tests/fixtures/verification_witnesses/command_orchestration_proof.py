from __future__ import annotations

from types import SimpleNamespace

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.development.targetgen.rocc import decode
from merlin.verification import host_compute_audit as HA

_ISA = {
    "CUSTOM_OPCODE": 123,
    "FUNCT_CLASS": {0: "CONFIG", 2: "MVIN", 7: "FLUSH", 14: "MVIN3"},
    "CONFIG_CLASS": "CONFIG",
}
_COMMAND = (
    '"llvm.inline_asm"(OPS) <{asm_string = "TEMPLATE", constraints = "CONSTRAINTS", has_side_effects}> : (TYPES) -> ()'
)


def _asm(template: str, *operands: str, constraints: str | None = None) -> str:
    spelled = constraints if constraints is not None else ",".join(["r"] * len(operands) + ["~{memory}"])
    return (
        _COMMAND.replace("OPS", ", ".join(operands))
        .replace("TEMPLATE", template)
        .replace("CONSTRAINTS", spelled)
        .replace("TYPES", ", ".join(("i64" for _ in operands)))
    )


_MVIN = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1"


def _kernel(prologue: str = "", body: str = "", epilogue: str = "", *, tiles: int = 16) -> str:
    """Address math and one transfer per tile; ``body`` runs per tile and may use %t and %c1."""
    return f"""module {{\n      llvm.func @entry(%scale: !llvm.ptr, %input: !llvm.ptr, %output: !llvm.ptr) {{\n        %c0 = llvm.mlir.constant(0 : i64) : i64\n        %c1 = llvm.mlir.constant(1 : i64) : i64\n        %width = llvm.mlir.constant(56 : i64) : i64\n        %tiles = llvm.mlir.constant({tiles} : i64) : i64\n        {prologue}\n        llvm.br ^loop(%c0 : i64)\n      ^loop(%t: i64):\n        %more = llvm.icmp "ult" %t, %tiles : i64\n        llvm.cond_br %more, ^tile, ^done\n      ^tile:\n        %x = llvm.urem %t, %width : i64\n        %y = llvm.udiv %t, %width : i64\n        %row = llvm.mul %y, %width : i64\n        %offset = llvm.add %row, %x : i64\n        %src = llvm.getelementptr %input[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n        %addr = llvm.ptrtoint %src : !llvm.ptr to i64\n        {_asm(_MVIN, "%addr", "%t")}\n        {body}\n        %next = llvm.add %t, %c1 : i64\n        llvm.br ^loop(%next : i64)\n      ^done:\n        {epilogue}\n        llvm.return\n      }}\n    }}"""


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setattr(decode, "isa_constants", lambda target: dict(_ISA))
    monkeypatch.setattr(decode, "_semantics", lambda target: SimpleNamespace(operand_observer=None))

    def run(source: str, elements: int = 4) -> dict:
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, llvm.LLVM):
            context.load_dialect(dialect)
        module = Parser(context, source).parse_module()
        module.verify()
        site = HA.GroupSite(group=0, placement="unit0", symbol="entry", elements=elements, element_bytes=1)
        return HA.audit(module, [site], target="fixture")["groups"][0]

    return run


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "Context",
    "HA",
    "Parser",
    "SimpleNamespace",
    "_COMMAND",
    "_ISA",
    "_MVIN",
    "_asm",
    "_kernel",
    "builtin",
    "decode",
    "judge",
    "llvm",
    "pytest",
]
