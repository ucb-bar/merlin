"""A lane engine's element format is read off the RTL, and an accumulator format is never an operand.

Two defects, one shape. The capability generator gave a synthesized vector unit the union of the
array's formats (there was nothing else to read), and the array's declared formats included its
ACCUMULATOR format, so a bf16 accumulator became a bf16 operand of the array's contraction: a target
whose multiply-accumulate cell consumes only fp8 was declared able to contract bf16 tensors.

The fix has three parts, each pinned here: the facts reader names the lane engine's format from its
own per-lane arithmetic (``rtl.datapaths.lane_datapaths``), the generator takes a synthesized lane
unit's formats from that fact, and a format the cell facts ground only as the accumulator is removed
from the unit's operand formats (recorded, not silent). Role evidence carries the formats of the
engine the role belongs to, not every engine's.
"""

from __future__ import annotations

import external_sources
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import capability_derive as CD
from merlin.targetgen import capability_manifests as CM
from merlin.targetgen.rtl import datapaths as D

# A minimal elaboration: an array of two MAC cells, and beside it a vector unit whose lane box
# replicates one hardfloat rounding primitive per lane (four lanes) next to a per-lane multiplier.
_FIR = """FIRRTL version 4.0.0
circuit Top :
  module Cell :
    input act : UInt<8>
    input addend : UInt<16>
    output mac : UInt<16>
  module Cell_1 :
    input act : UInt<8>
    input addend : UInt<16>
    output mac : UInt<16>
  module Mesh :
    input clock : Clock
    inst c0 of Cell
    inst c1 of Cell_1
  module RoundRawFNToRecFN_e8_s8 :
    input in : UInt<20>
    output out : UInt<17>
  module RoundRawFNToRecFN_e8_s8_1 :
    input in : UInt<20>
    output out : UInt<17>
  module RoundRawFNToRecFN_e8_s8_2 :
    input in : UInt<20>
    output out : UInt<17>
  module RoundRawFNToRecFN_e8_s8_3 :
    input in : UInt<20>
    output out : UInt<17>
  module MulLane :
    input a : UInt<16>
    output o : UInt<16>
  module MulLane_1 :
    input a : UInt<16>
    output o : UInt<16>
  module MulLane_2 :
    input a : UInt<16>
    output o : UInt<16>
  module MulLane_3 :
    input a : UInt<16>
    output o : UInt<16>
  module LaneBox :
    input clock : Clock
    inst r0 of RoundRawFNToRecFN_e8_s8
    inst r1 of RoundRawFNToRecFN_e8_s8_1
    inst r2 of RoundRawFNToRecFN_e8_s8_2
    inst r3 of RoundRawFNToRecFN_e8_s8_3
    inst m0 of MulLane
    inst m1 of MulLane_1
    inst m2 of MulLane_2
    inst m3 of MulLane_3
  module VecUnit :
    input clock : Clock
    inst box of LaneBox
  module Core :
    input clock : Clock
    inst mesh of Mesh
    inst vec of VecUnit
  public module Top :
    input clock : Clock
    inst core of Core
"""


def _facts() -> dict:
    return {"arrays": [{"name": "mesh", "element": "Cell"}], "census": {"unit_root": "Top"}}


def test_hardfloat_suffix_names_a_registry_format_by_its_split():
    assert D.hardfloat_formats("RoundRawFNToRecFN_e8_s8_3") == ("bf16",)
    assert D.hardfloat_formats("RoundRawFNToRecFN_e8_s24") == ("fp32",)
    assert D.hardfloat_formats("RoundRawFNToRecFN_e4_s4") == ("fp8_e4m3",)
    assert D.hardfloat_formats("MulRawFN") == ()


def test_the_reader_names_the_lane_engine_format_beside_the_array(tmp_path):
    fir = tmp_path / "design.fir"
    fir.write_text(_FIR, encoding="utf-8")
    lanes, notes = D.lane_datapaths(_facts(), [fir])
    assert [(r["unit_module"], r["lanes"], r["dtype"]) for r in lanes] == [("VecUnit", 4, "bf16")], notes
    assert lanes[0]["source"] == D.LANE_SOURCE
    assert "e8/s8" in lanes[0]["evidence"] and "17-bit" in lanes[0]["evidence"]


def test_two_named_formats_per_lane_fail_closed(tmp_path):
    fir = tmp_path / "design.fir"
    text = _FIR.replace("module MulLane", "module E4M3Mul").replace("of MulLane", "of E4M3Mul")
    text = text.replace("input a : UInt<16>", "input a : UInt<8>")
    fir.write_text(text, encoding="utf-8")
    lanes, notes = D.lane_datapaths(_facts(), [fir])
    assert lanes and lanes[0]["dtype"] is None
    assert "bf16" in lanes[0]["dtype_unknown"] and "fp8_e4m3" in lanes[0]["dtype_unknown"]
    assert any("VecUnit" in n for n in notes)


def _cell_body() -> dict:
    return {
        "datapaths": [
            {"name": "input", "dtype": "fp8_e4m3", "elem_bits": 8, "evidence": "cell consumes 8-bit e4m3"},
            {"name": "accumulator", "dtype": "bf16", "elem_bits": 16, "evidence": "cell accumulates in bf16"},
        ]
    }


def test_an_accumulator_format_is_not_kept_as_an_operand_format():
    unit = {
        "name": "mxu",
        "kind": "systolic",
        "dtypes": ["fp8_e4m3", "bf16"],
        "semantic_capabilities": [{"family": "contraction", "dtypes": ["fp8_e4m3", "bf16"]}],
    }
    kept = CM._drop_accumulator_only(unit, list(unit["dtypes"]), _cell_body())
    assert kept == ["fp8_e4m3"]
    assert unit["semantic_capabilities"][0]["dtypes"] == ["fp8_e4m3"]
    (correction,) = unit["dtype_corrections"]
    assert correction["dtype"] == "bf16" and "accumulator" in correction["reason"]


def test_a_format_that_is_also_an_input_is_kept():
    body = _cell_body()
    body["datapaths"][1]["dtype"] = "fp8_e4m3"
    unit = {"name": "mxu", "dtypes": ["fp8_e4m3"]}
    assert CM._drop_accumulator_only(unit, ["fp8_e4m3"], body) == ["fp8_e4m3"]
    assert "dtype_corrections" not in unit


def test_role_evidence_carries_its_own_engines_formats():
    units = [
        {"name": "mxu", "kind": "systolic", "dtypes": ["fp8_e4m3"]},
        {"name": "v", "kind": "vector", "dtypes": ["bf16"]},
    ]
    union = ("fp8_e4m3", "bf16")
    assert CD._role_dtypes("matmul", units, union) == ("fp8_e4m3",)
    assert CD._role_dtypes("tensor_compute_binary", units, union) == ("bf16",)
    assert CD._role_dtypes("memory", units, union) == union


def test_a_synthesized_lane_unit_takes_the_lane_fact():
    facts = {
        "facts": {
            "lane_datapaths": [{"unit_module": "VecUnit", "dtype": "bf16", "source": D.LANE_SOURCE, "evidence": "e"}]
        }
    }
    formats, evidence = CM._lane_formats_for_kind("vector", facts)
    assert formats == ("bf16",) and "VecUnit" in evidence
    assert CM._lane_formats_for_kind("systolic", facts) is None


@external_sources.requires_rtl("atlas")
@external_sources.requires_ext("npu_model")
def test_the_atlas_reference_vector_unit_is_the_derived_one():
    """The reference contract records the derived unit so a host without the RTL can grade against it;
    with the RTL present, re-derive and compare, and confirm no unit claims a bf16 contraction."""
    derived = CM.manifest_for("atlas")
    reference = yaml.safe_load((repo_root() / "examples/atlas/target/contracts/target_contract.yaml").read_text())
    pick = {u["name"]: u for u in derived["compute_units"]}
    ref = {u["name"]: u for u in reference["compute_units"]}
    for key in ("kind", "dtypes", "ops", "semantic_capabilities"):
        assert pick["vector_unit"][key] == ref["vector_unit"][key], key
    for unit in derived["compute_units"]:
        for cap in unit.get("semantic_capabilities") or []:
            if cap["family"] == "contraction":
                assert "bf16" not in cap["dtypes"], unit["name"]
    contraction = [c for c in derived.get("semantic_capabilities_derived") or [] if c["family"] == "contraction"]
    assert contraction and all("bf16" not in c["dtypes"] for c in contraction)


@external_sources.requires_rtl("atlas")
def test_the_atlas_facts_name_a_bf16_lane_engine():
    from merlin.targetgen.rtl.facts import load_facts

    lanes = (load_facts("atlas").get("facts") or {}).get("lane_datapaths") or []
    named = {r["dtype"] for r in lanes if r.get("dtype")}
    assert named == {"bf16"}, lanes
