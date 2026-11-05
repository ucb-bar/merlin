"""Optional AOT derivation, symbol/DWARF correspondence and independent coverage; no hardware."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from merlin.compile.analysis.disassembly import RawInsn
from merlin.mining.reference_database_analysis import independent_site_census, symbol_overlap
from merlin.mining.reference_database_aot import (
    UNAVAILABLE,
    compile_declared,
    decoded_lines,
    dependency_paths,
    subprogram_ranges,
    symbol_correspondence,
)
from merlin.mining.reference_database_audit import audit


def test_accelerator_region_analysis_gains_derived_effects_without_changing_cca():
    from merlin.mining.reference_database_analysis import region_effects

    report = dict(
        cca=dict(kind="toy"),
        evidence=dict(evidence_id="toy-region"),
        mechanism_projection=dict(instruction_classes={"LOAD": 1}),
    )
    rows = [RawInsn(addr=0, mnemonic="x", operands="", hexcode=f"{(2 << 25) | 0x2B:08x}")]
    effects = region_effects(report, rows, {}, {}, "artifact:region")
    assert effects["schema"] == "merlin.derived_effects.v1"
    assert report["cca"] == dict(kind="toy") and "derived_effects" not in report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_symbol_overlap_keeps_exact_partial_ambiguous_and_absent_distinct():
    symbols = [dict(name="kernel", address=0x100, size_bytes=0x40), dict(name="other", address=0x200, size_bytes=8)]
    assert symbol_overlap(symbols, 0x100, 0x140)["symbol_status"] == "exact"
    partial = symbol_overlap(symbols, 0x100, 0x160)
    assert partial["symbol"] == "kernel" and partial["symbol_status"] == "partial"
    assert partial["overlapping_symbols"][0]["overlap_bytes"] == 0x40
    ambiguous = symbol_overlap(symbols, 0x100, 0x208)
    assert ambiguous["symbol"] is None and ambiguous["symbol_status"] == "ambiguous"
    assert symbol_overlap(symbols, 0x300, 0x310) == dict(symbol=None, symbol_status="none", overlapping_symbols=[])


def test_independent_census_counts_every_selected_extension_word():
    opcode = 0x2B
    rows = [
        RawInsn(addr=0, mnemonic="x", operands="", hexcode=f"{(3 << 25) | opcode:08x}"),
        RawInsn(addr=4, mnemonic="x", operands="", hexcode=f"{(3 << 25) | opcode:08x}"),
        RawInsn(addr=8, mnemonic="x", operands="", hexcode=f"{(1 << 25) | opcode:08x}"),
        RawInsn(addr=12, mnemonic="c", operands="", hexcode=f"{opcode:04x}"),
        RawInsn(addr=14, mnemonic="y", operands="", hexcode=f"{(3 << 25) | 0x13:08x}"),
    ]
    census = independent_site_census(rows, opcode)
    assert census["sites"] == 3 and census["by_funct7"] == {"1": 1, "3": 2} and census["instructions"] == 5


def test_dependency_and_debug_listings_parse_structurally():
    assert dependency_paths("out.o: a.c inc/b.h \\\n  /sys/c.h a.c\n") == ["a.c", "inc/b.h", "/sys/c.h"]
    lines = decoded_lines(
        "Contents of the .debug_line section:\n\nCU: ./k.c:\n"
        "File name    Line number    Starting address    View    Stmt\n"
        "k.c                   3          0x80000000               x\n"
        "k.c                   4          0x80000004       1       x\n"
    )
    assert lines == [
        dict(address=0x80000000, file="k.c", line=3, unit="./k.c"),
        dict(address=0x80000004, file="k.c", line=4, unit="./k.c"),
    ]
    ranges = subprogram_ranges(
        " <1><2d>: Abbrev Number: 5 (DW_TAG_subprogram)\n"
        "    <2e>   DW_AT_name        : (strp) (offset: 0x10): kernel\n"
        "    <35>   DW_AT_low_pc      : (addr) 0x80000000\n"
        "    <3d>   DW_AT_high_pc     : (data8) 0x40\n"
        " <1><50>: Abbrev Number: 6 (DW_TAG_variable)\n"
        "    <51>   DW_AT_name        : data\n"
    )
    assert ranges == [
        dict(
            tag="DW_TAG_subprogram",
            name="kernel",
            origin=None,
            address_range=[0x80000000, 0x80000040],
            decl_line=None,
        )
    ]


def recipe(driver, digest="0" * 64):
    return dict(
        toolchain=dict(driver=dict(path=str(driver), sha256=digest)),
        builds=[
            dict(
                id="aot-unit",
                sources=["unit-c"],
                include_flag="-I",
                include_dirs=[],
                compile_argv=["{driver}", "-c", "{source}", "-o", "{object}"],
                link_argv=["{driver}", "{objects}", "-o", "{output}"],
            )
        ],
    )


def staged(tmp_path, text=b"int kernel(int x) { return x + 1; }\n"):
    source = tmp_path / "originals" / "unit.c"
    source.parent.mkdir(parents=True)
    source.write_bytes(text)
    return [dict(id="unit-c", path="originals/unit.c", format="c_source", sha256=sha(source))]


def test_absent_toolchain_is_a_structured_diagnostic_not_a_silent_skip(tmp_path):
    artifacts = staged(tmp_path)
    before = (tmp_path / "originals" / "unit.c").read_bytes()
    derived, records, diagnostics = compile_declared(tmp_path, recipe(tmp_path / "missing" / "cc"), artifacts)
    assert derived == [] and records == []
    assert diagnostics == [
        dict(
            kind=UNAVAILABLE,
            reason="toolchain_driver_absent",
            path=str(tmp_path / "missing" / "cc"),
            build="aot-unit",
            sources=["unit-c"],
        )
    ]
    assert (tmp_path / "derived" / "aot" / "diagnostics.json").is_file()
    assert (tmp_path / "originals" / "unit.c").read_bytes() == before


def test_toolchain_that_differs_from_its_pin_is_refused(tmp_path):
    driver = tmp_path / "cc"
    driver.write_bytes(b"#!/bin/sh\nexit 1\n")
    with pytest.raises(ValueError, match="pinned hash"):
        compile_declared(tmp_path, recipe(driver), staged(tmp_path))


def test_symbol_correspondence_joins_only_unique_unit_definitions():
    records = [
        dict(
            id="aot",
            status="compiled",
            units=[
                dict(source="a-c", defined_functions=["kernel", "shared"]),
                dict(source="b-c", defined_functions=["shared"]),
            ],
        )
    ]
    occurrences = [
        dict(id="m0", artifact_id="aot", symbol="kernel", symbol_status="partial"),
        dict(id="m1", artifact_id="aot", symbol="shared", symbol_status="exact"),
        dict(id="m2", artifact_id="aot", symbol=None, symbol_status="ambiguous"),
        dict(id="s0", artifact_id="a-c", symbol="kernel", boundary_status="parsed_source_definition"),
        dict(id="s1", artifact_id="a-c", symbol="shared", boundary_status="parsed_source_definition"),
    ]
    (match,) = symbol_correspondence(records, occurrences)
    assert (match["machine_occurrence"], match["source_occurrence"], match["symbol_status"]) == ("m0", "s0", "partial")
    assert [row["source_correspondence"] for row in occurrences[:3]] == ["aot_symbol_definition", "unknown", "unknown"]


def test_boundary_audit_refuses_derived_products_inside_originals(tmp_path):
    (tmp_path / "a.json").write_text("{}\n")
    rows = [dict(analysis="a.json", numerical_equivalence="unknown", performance="unknown")]
    good = dict(path="derived/aot/x/x.elf", origin=dict(kind="aot_compiled_derivative"))
    assert audit(tmp_path, [], rows, derived=[good])["derived_products"] == 1
    with pytest.raises(ValueError, match="derived product"):
        audit(tmp_path, [], rows, derived=[dict(good, path="originals/x.elf")])


DRIVER = os.environ.get("MERLIN_TEST_AOT_DRIVER")


@pytest.mark.skipif(not DRIVER or not shutil.which(DRIVER), reason="no pinned cross compiler selected")
def test_real_toolchain_build_records_provenance_symbols_and_line_table(tmp_path):
    driver = Path(shutil.which(DRIVER)).resolve()
    artifacts = staged(tmp_path, b"int kernel(int x)\n{\n  return x + 1;\n}\nvoid _start(void) {}\n")
    selection = recipe(driver, sha(driver))
    build = selection["builds"][0]
    build["compile_argv"] = ["{driver}", "-O2", "-g", "-c", "{source}", "-o", "{object}", "-MD", "-MF", "{depfile}"]
    build["link_argv"] = ["{driver}", "-nostdlib", "-nostartfiles", "-static", "{objects}", "-o", "{output}"]
    readelf = os.environ.get("MERLIN_TEST_AOT_READELF")
    if readelf:
        selection["toolchain"]["components"] = dict(readelf=dict(path=readelf, sha256=sha(readelf)))
        build["debug_dumps"] = dict(
            line_table=["{readelf}", "--debug-dump=decodedline", "-W", "{output}"],
            subprograms=["{readelf}", "--debug-dump=info", "-W", "{output}"],
        )
    before = (tmp_path / "originals" / "unit.c").read_bytes()
    derived, (record,), diagnostics = compile_declared(tmp_path, selection, artifacts)
    assert diagnostics == [] and record["status"] == "compiled"
    (product,) = derived
    assert product["path"].startswith("derived/aot/") and sha(tmp_path / product["path"]) == product["sha256"]
    (unit,) = record["units"]
    assert unit["source_sha256"] == artifacts[0]["sha256"]
    assert {"kernel", "_start"} <= set(unit["defined_functions"])
    assert any(row["path"] == "originals/unit.c" for row in unit["dependencies"])
    if readelf:
        dumps = record["debug_correspondence"]["dumps"]
        assert dumps["line_table"]["rows"] > 0
        assert dumps["subprograms"]["rows"] >= 2
    assert (tmp_path / "originals" / "unit.c").read_bytes() == before
