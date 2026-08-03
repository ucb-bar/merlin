"""runtime_build derives the bare-metal linker load address from the RTL memory map instead of baking it.

Hermetic: the origin-rewrite and the fallback are pure logic (no RTL build needed); the chipyard memmap
read is exercised only when a build is present (skipped otherwise), so this stays target/host-agnostic.
"""
from __future__ import annotations

from pathlib import Path

from merlin.targetgen import runtime_build as RB


def test_rebase_replaces_the_first_absolute_origin():
    ld = "OUTPUT_ARCH(\"riscv\")\nSECTIONS {\n  . = 0x80000000;\n  .text : { *(.text*) }\n}\n"
    out = RB._rebase_ld(ld, 0x40000000)
    assert ". = 0x40000000;" in out
    assert "0x80000000" not in out          # the baked origin is gone
    assert ".text : { *(.text*) }" in out    # the rest of the layout is untouched


def test_rebase_returns_none_when_no_origin_to_rewrite():
    # a script with no absolute `. = 0x..;` is left to the caller (copied through unchanged, never broken)
    assert RB._rebase_ld("SECTIONS { .text : { *(.text*) } }\n", 0x80000000) is None


def test_derived_link_script_writes_the_derived_base(tmp_path: Path):
    tmpl = tmp_path / "test.ld"
    tmpl.write_text("ENTRY(_start)\nSECTIONS {\n  . = 0x80000000;\n  .text : {}\n}\n")
    out = RB.derived_link_script(0x70000000, tmpl, tmp_path / "gen")
    assert out.is_file()
    body = out.read_text()
    assert ". = 0x70000000;" in body and "0x80000000" not in body


def test_platform_dram_base_falls_back_for_unknown_build_tool():
    # an unknown/absent RTL build tool yields the documented platform default, never a crash or a guess
    assert RB.platform_dram_base("any_target", None) == RB.DEFAULT_PLATFORM_DRAM_BASE
    assert RB.platform_dram_base("any_target", "some_other_sim") == RB.DEFAULT_PLATFORM_DRAM_BASE
