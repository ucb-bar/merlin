"""Equivalent byte-address spellings retain source identity; unknown addresses do not."""

import pytest

from merlin.targetgen.rocc import decode as RD


def _pointer(body, *, globals=""):
    module = RD._parse_module(
        f"module {{ {globals}\n llvm.func @kernel(%base: !llvm.ptr, %index: i64) {{\n"
        + body + "\n llvm.return\n } }"
    )
    assert module is not None
    address = next(op for op in module.walk() if op.name == "llvm.ptrtoint")
    return RD._operand(RD._resolve(address.results[0], _word_bits=64))


def test_static_and_constant_ssa_byte_offsets_identify_the_same_argument():
    static = _pointer("""
      %p = llvm.getelementptr %base[16] : (!llvm.ptr) -> !llvm.ptr, i8
      %address = llvm.ptrtoint %p : !llvm.ptr to i64
    """)
    dynamic = _pointer("""
      %offset = llvm.mlir.constant(16 : i64) : i64
      %p = llvm.getelementptr %base[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %address = llvm.ptrtoint %p : !llvm.ptr to i64
    """)
    assert static == dynamic == {"kind": "argbase", "arg_index": 0, "offset": 16, "raw": None}


@pytest.mark.parametrize("body", [
    "%p = llvm.getelementptr %base[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8",
    "%p = llvm.getelementptr %base[16] : (!llvm.ptr) -> !llvm.ptr, i32",
    "llvm.br ^next(%base : !llvm.ptr)\n ^next(%p: !llvm.ptr):",
])
def test_runtime_index_unproved_layout_and_cfg_arguments_stay_unknown(body):
    assert _pointer(body + "\n %address = llvm.ptrtoint %p : !llvm.ptr to i64")["kind"] == "unknown"


def test_narrowed_pointer_cannot_supply_original_source_identity():
    assert _pointer("%address = llvm.ptrtoint %base : !llvm.ptr to i32")["kind"] == "unknown"


_GLOBAL = """llvm.mlir.global private constant @padding(dense<0> : tensor<16xi8>)
 {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<16 x i8>"""
_ADDRESS = """
 %storage = llvm.mlir.addressof @padding : !llvm.ptr
 %p = llvm.getelementptr %storage[8] : (!llvm.ptr) -> !llvm.ptr, i8
 %address = llvm.ptrtoint %p : !llvm.ptr to i64
"""


def test_closed_readonly_global_has_symbol_relative_identity():
    assert _pointer(_ADDRESS, globals=_GLOBAL) == {
        "kind": "global", "symbol": "padding", "offset": 8, "raw": None, "arg_index": None,
    }


@pytest.mark.parametrize("declaration", [
    "", _GLOBAL.replace(" constant", ""), _GLOBAL.replace("private", "external"),
    _GLOBAL + "\n" + _GLOBAL,
])
def test_missing_mutable_external_and_ambiguous_globals_stay_unknown(declaration):
    assert _pointer(_ADDRESS, globals=declaration)["kind"] == "unknown"
