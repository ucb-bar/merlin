"""Complete typed emitted-site observations never become physical prices."""

import struct

import pytest

from merlin.perf import compiled_static_features as F
from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable


def elf():
    # A structural ELF fixture, with opaque executable bytes, not ISA answers.
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, 64)
    struct.pack_into("<HHH", header, 0x3A, 64, 3, 1)
    empty = bytes(64)
    names = b"\x00.names\x00.code\x00"
    strings = struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 256, len(names), 0, 0, 1, 0)
    code = struct.pack("<IIQQQQIIQQ", 8, 1, 4, 1024, 256 + len(names), 8, 0, 0, 1, 0)
    return bytes(header) + empty + strings + code + names + b"opaque00"


def observe(operations, **changes):
    kwargs = {
        "llvm_mlir": "module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {\n" + operations + "\nllvm.return } }",
        "elf_bytes": elf(),
        "entry_symbol": "entry",
        "pointer_bits": 64,
        "retained_entry": {"symbol": "entry", "address": 1024, "size_bytes": 8, "scope": "static linked definition"},
        "limits": F.StaticFeatureLimits(8192, 8192, 64),
        **changes,
    }
    return F.derive_compiled_static_features(**kwargs)


def test_full_memory_site_width_alignment_and_volatile_roster():
    result = observe("""
%v = llvm.load %a {alignment = 1 : i64} : !llvm.ptr -> i16
llvm.store volatile %v, %b : i16, !llvm.ptr
""")
    memory = result["emitted"]["memory"]
    assert memory["load"] == {
        "sites": 1,
        "declared_access_bits": 16,
        "width_sites": {"16": 1},
        "alignment_sites": {"1": 1},
        "volatile_sites": 0,
    }
    assert memory["store"]["declared_access_bits"] == 16
    assert memory["store"]["alignment_sites"] == {"undeclared": 1}
    assert memory["store"]["volatile_sites"] == 1
    assert result["linked"]["executable_bytes"] == 8
    assert result["authority"] == "none" and "physical_memory_traffic" in result["unknown"]


def test_opaque_assembly_and_pointer_operations_do_not_invent_command_semantics():
    result = observe("""
%p = llvm.ptrtoint %a : !llvm.ptr to i64
llvm.inline_asm has_side_effects "opaque $0", "r,~{memory}" %p : (i64) -> ()
""")
    assert result["emitted"]["opaque_assembly_sites"] == 1
    assert result["emitted"]["pointer_transform_sites"] == 1
    assert result["emitted"]["memory"]["load"]["sites"] == 0
    assert "instruction_semantics_and_count" in result["unknown"]


@pytest.mark.parametrize(
    "operations",
    [
        "%v = llvm.load %a : !llvm.ptr -> f32",
        "llvm.call @other(%a) : (!llvm.ptr) -> ()",
        "%v = llvm.load %a {ordering = 2 : i64} : !llvm.ptr -> i8",
        "llvm.br ^next\n^next:",
    ],
)
def test_unsupported_work_refuses_instead_of_omitting_sites(operations):
    with pytest.raises((DataflowUnavailable, ValueError)):
        observe(operations)


@pytest.mark.parametrize("field", ["max_source_bytes", "max_elf_bytes", "max_operations"])
def test_explicit_byte_and_operation_limits(field):
    values = {"max_source_bytes": 8192, "max_elf_bytes": 8192, "max_operations": 64, field: 1}
    with pytest.raises(ValueError, match="limit|bound"):
        observe("%v = llvm.load %a : !llvm.ptr -> i8", limits=F.StaticFeatureLimits(**values))


@pytest.mark.parametrize(
    "change",
    [
        {"address": 1023},
        {"size_bytes": 9},
        {"size_bytes": True},
        {"symbol": "other"},
        {"address": -1},
    ],
)
def test_entry_must_be_a_complete_structurally_contained_definition(change):
    entry = {"symbol": "entry", "address": 1024, "size_bytes": 8, "scope": "static linked definition", **change}
    with pytest.raises(ValueError, match="extent"):
        observe("", retained_entry=entry)
