"""Source-format gates retain complete declarations and refuse disguised inputs."""

import pytest
from merlin_experiments.corpus.storage_budget_selection import _dram_parameters, _linker_memory, _memory_region
from merlin_experiments.phase1 import public_runtime_abi as abi
from merlin_experiments.phase1.gemmini_gsim_alignment import _dts_identity
from merlin_experiments.phase1.neutral_program_inputs import UnsupportedProgram, encode_source


def test_firrtl_memory_owner_and_parameters_are_unique():
    source = (
        "circuit Chip :\n  extmodule SimDRAM :\n"
        "    parameter MEM_BASE = 4096\n    parameter MEM_SIZE = 32768\n"
        "  module Chip :\n    skip\n"
    )
    assert _dram_parameters(source) == (4096, 32768)
    with pytest.raises(ValueError, match="ambiguous"):
        _dram_parameters(source.replace("    parameter MEM_BASE", "    parameter MEM_BASE = 0\n    parameter MEM_BASE"))
    with pytest.raises(ValueError):
        _dram_parameters(source.replace("4096", "4096 + 1"))
    with pytest.raises(ValueError):
        _dram_parameters(source + "  extmodule SimDRAM :\n")


def test_linker_literals_ignore_comment_decoys_and_accept_whitespace():
    source = "/* . = 0xBAD; INCLUDE private.ld */ SECTIONS { .\n= 0x1000; . += 256; .text : { *(.text*) } }"
    assert _linker_memory(source) == ([4096], [256])
    assert _memory_region("-m0x1000:65536") == (4096, 65536)


@pytest.mark.parametrize(
    "source",
    [
        "SECTIONS { . = 0x1000; . += SIZE; }",
        "SECTIONS { . = 0x1000; . += 4 + 4; }",
        "SECTIONS { . = LOAD; }",
        "SECTIONS { . = 0x1000; } INCLUDE private.ld",
        "SECTIONS { . = 0x1000; } /* open",
    ],
)
def test_linker_dynamic_or_indirect_loads_refuse(source):
    with pytest.raises(ValueError):
        _linker_memory(source)


@pytest.mark.parametrize("argument", ["-m1:2,3:4", "-m:4", "-m0x:4", "-m+1:4", "-m1:-4"])
def test_memory_argument_requires_two_complete_unsigned_literals(argument):
    with pytest.raises(ValueError):
        _memory_region(argument)


def test_complete_pointer_declaration_and_protocol_prose_are_distinct():
    name, arguments = abi._extern_declaration("extern\tvoid entry( void *a, void* );")
    assert name == "entry" and abi._pointer_parameters(arguments)
    assert abi._void_signature("void {target}_kernel(void *arg_0) -- every argument a data pointer", "{target}_kernel")
    assert not abi._pointer_parameters(["void", "*", "*", "a"])
    for source in ("extern void entry(void); extern void other(void);", "extern void entry(void (*callback)());"):
        with pytest.raises(ValueError):
            abi._extern_declaration(source)


def test_generic_mlir_cannot_be_admitted_by_operation_names_in_a_comment():
    with pytest.raises(UnsupportedProgram):
        encode_source(b"// func.func and tensor.cast\nmodule {}")


def test_dts_root_cells_are_not_taken_from_a_nested_bus(tmp_path):
    path = tmp_path / "chip.dts"
    path.write_text("""/dts-v1/;
/ {
  #address-cells = <1>;
  #size-cells = <1>;
  bus { #address-cells = <2>; #size-cells = <2>; };
  cpus { cpu@0 { device_type = "cpu"; riscv,isa = "rv64imac"; }; };
  // memory@0 { device_type = "memory"; reg = <0 1>; };
  memory@1000 { device_type = "memory"; reg = <0x1000 0x4000>; };
};""")
    assert _dts_identity(path) == dict(dram_base=4096, dram_bytes=16384, host_isa="rv64imac", harts=1)
    path.write_text(path.read_text().replace("<0x1000 0x4000>", "<0x1000 0x4000>, <0x8000 0x1000>"))
    with pytest.raises(ValueError, match="single register span"):
        _dts_identity(path)


def test_dts_ambiguous_or_incomplete_source_refuses(tmp_path):
    path = tmp_path / "chip.dts"
    complete = (
        "/dts-v1/; / { #address-cells = <1>; #size-cells = <1>; "
        'memory@1000 { device_type = "memory"; reg = <0x1000 0x4000>; }; '
        'cpus { cpu@0 { riscv,isa = "rv64imac"; }; }; };'
    )
    for source in (complete.replace("#size-cells = <1>;", "#size-cells = <1>; #size-cells = <2>;"), complete[:-2]):
        path.write_text(source)
        with pytest.raises(ValueError):
            _dts_identity(path)
