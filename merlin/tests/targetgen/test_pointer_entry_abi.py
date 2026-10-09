"""The shared pointer harness requires the actual declared C entry signature.

These structural tests issue no source, device, effect or runtime authority.
Matching argument counts cannot replace actual parameter and convention checks.
"""

from io import StringIO

import pytest

from merlin.targetgen.bundle_harness import emitted_entry_arity
from merlin.targetgen.contract.compile_only import require_pointer_entry

_ENTRY = "module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) { llvm.return } }"


@pytest.mark.parametrize(
    "entry",
    [
        _ENTRY,
        _ENTRY.replace("llvm.func @entry", "llvm.func hidden @entry"),
        _ENTRY.replace("{ llvm.return }", "attributes {dso_local, nobuiltins = []} { llvm.return }"),
    ],
)
def test_plain_pointer_entry_and_standard_metadata_are_structural_only(entry):
    assert emitted_entry_arity(entry, entry_symbol="entry") == 2
    require_pointer_entry(entry, entry_symbol="entry", pointer_arity=2)


@pytest.mark.parametrize(
    "entry",
    [
        _ENTRY.replace("%a: !llvm.ptr", "%a: i64"),
        _ENTRY.replace("%a: !llvm.ptr", "%a: !llvm.ptr<1>"),
        _ENTRY.replace("%b: !llvm.ptr)", "%b: !llvm.ptr, ...)"),
        _ENTRY.replace("llvm.func @entry", "llvm.func fastcc @entry"),
        _ENTRY.replace("llvm.func @entry", "llvm.func internal @entry"),
        _ENTRY.replace("%a: !llvm.ptr", "%a: !llvm.ptr {llvm.byval = i8}"),
        "module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) -> i32 {"
        " %zero = llvm.mlir.constant(0 : i32) : i32 llvm.return %zero : i32 } }",
    ],
)
def test_equal_arity_cannot_admit_changed_pointer_calling_contract(entry):
    assert emitted_entry_arity(entry, entry_symbol="entry") == 2
    with pytest.raises(ValueError, match="pointer ABI"):
        require_pointer_entry(entry, entry_symbol="entry", pointer_arity=2)


@pytest.mark.parametrize(
    "entry, symbol, count",
    [
        (_ENTRY, "missing", 2),
        (_ENTRY, "entry", 1),
        (_ENTRY, "entry", True),
        (_ENTRY, "entry", 0),
        ("module { llvm.func @entry(!llvm.ptr, !llvm.ptr) }", "entry", 2),
        (_ENTRY.replace("module {", "module { llvm.func @entry(!llvm.ptr, !llvm.ptr) "), "entry", 2),
    ],
)
def test_missing_ambiguous_or_malformed_pointer_entry_refuses(entry, symbol, count):
    with pytest.raises(ValueError, match="entry|pointer ABI"):
        require_pointer_entry(entry, entry_symbol=symbol, pointer_arity=count)


def test_generic_function_type_cannot_hide_a_different_body_argument_type():
    from xdsl.dialects import builtin, llvm
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer

    body = Block([llvm.ReturnOp()], arg_types=[builtin.i64, llvm.LLVMPointerType()])
    entry = llvm.FuncOp(
        "entry",
        llvm.LLVMFunctionType([llvm.LLVMPointerType()] * 2),
        linkage=llvm.LinkageAttr("external"),
        body=Region(body),
    )
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(builtin.ModuleOp([entry]))
    with pytest.raises(ValueError, match="pointer ABI"):
        require_pointer_entry(output.getvalue(), entry_symbol="entry", pointer_arity=2)
