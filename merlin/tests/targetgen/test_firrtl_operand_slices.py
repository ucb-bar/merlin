"""Actual typed operand slices never inherit reserved-slot widths or missing logic."""

import copy
import hashlib

import pytest

from merlin.targetgen.rtl.firrtl_operand_slices import (
    apply_selected_layouts,
    derive_operand_layouts,
    source_flag_offsets,
)

SOURCE = """FIRRTL version 4.2.0
circuit Top:
  module Words:
    output io : { deq : { bits : { rs2 : UInt<64> } } }
  module Top:
    input control : UInt<1>
    inst q of Words
    wire raw : UInt<64>
    connect raw, q.io.deq.bits.rs2
    wire unpacked : { rows : UInt<5>, local : { acc : UInt<1>, full : UInt<1>, data : UInt<30> } }
    node rows = bits(raw, 52, 48)
    connect unpacked.rows, rows
    node addr = bits(raw, 31, 0)
    node acc = bits(addr, 31, 31)
    node full = bits(addr, 30, 30)
    node data = bits(addr, 29, 0)
    connect unpacked.local.acc, acc
    connect unpacked.local.full, full
    connect unpacked.local.data, data
"""


def _selection(tmp_path, source=SOURCE):
    path = tmp_path / "design.fir"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [{"bundle": "Packed", "name": name, "module": "Top",
             "field": "unpacked." + field, "input": "q.io.deq.bits.rs2"}
            for name, field in (("num_rows", "rows"), ("local_addr", "local"))]
    return path, digest, derive_operand_layouts(path, expected_sha256=digest, selections=rows)


def _facts(path, digest):
    return {"inputs": {"firrtl_inputs": [{"path": str(path), "sha256": digest}]},
            "facts": {"interfaces": [{"name": "register_bundle_layouts", "bundles": {
                "Packed": {"width": 64, "fields": {
                    "num_rows": {"offset": 48, "width": None, "slot_width": 16},
                    "local_addr": {"offset": 0, "width": None, "slot_width": 32},
                }}}}]}}


def test_actual_nested_slice_and_typed_instance_export(tmp_path):
    path, digest, selection = _selection(tmp_path)
    fields = {row["name"]: row for row in selection["fields"]}
    assert fields["num_rows"]["width"] == 5
    assert fields["num_rows"]["offset"] == 48
    assert fields["local_addr"]["width"] == 32
    assert fields["local_addr"]["offset"] == 0
    facts = _facts(path, digest)
    updated = apply_selected_layouts(facts, selection)
    bundle = updated["facts"]["interfaces"][0]["bundles"]["Packed"]
    assert bundle["fields"]["num_rows"]["width"] == 5
    assert bundle["fields"]["num_rows"]["slot_width"] == 16
    assert facts["facts"]["interfaces"][0]["bundles"]["Packed"]["fields"]["num_rows"]["width"] is None
    assert source_flag_offsets(selection, bundle="Packed", name="local_addr",
                               flags={"accumulator": "acc", "read_full": "full"}) == {
                                   "accumulator": {"msb_offset": 0}, "read_full": {"msb_offset": 1}}
    with pytest.raises(ValueError, match="one-bit"):
        source_flag_offsets(selection, bundle="Packed", name="local_addr", flags={"bad": "data"})


@pytest.mark.parametrize("change", [
    lambda s: s.replace("connect unpacked.rows, rows", "when control :\n      connect unpacked.rows, rows"),
    lambda s: s.replace("connect unpacked.rows, rows", "connect unpacked.rows, rows\n    connect unpacked.rows, rows"),
    lambda s: s.replace("node rows = bits(raw, 52, 48)", "node rows = mux(control, bits(raw, 52, 48), UInt<5>(0))"),
    lambda s: s.replace("wire raw : UInt<64>", "wire raw : UInt<32>"),
    lambda s: s.replace("output io : { deq : { bits : { rs2 : UInt<64> } } }",
                        "output io : { deq : { bits : { rs2 : UInt<32> } } }"),
])
def test_missing_conditional_ambiguous_or_width_changing_sources_stay_unavailable(tmp_path, change):
    path, digest, selection = _selection(tmp_path, change(SOURCE))
    row = next(row for row in selection["fields"] if row["name"] == "num_rows")
    assert row["status"] == "unavailable"
    updated = apply_selected_layouts(_facts(path, digest), selection)
    assert updated["facts"]["interfaces"][0]["bundles"]["Packed"]["fields"]["num_rows"]["width"] is None


def test_aggregate_overlap_is_not_an_address_layout(tmp_path):
    _, _, selection = _selection(tmp_path, SOURCE.replace("bits(addr, 30, 30)", "bits(addr, 31, 31)"))
    row = next(row for row in selection["fields"] if row["name"] == "local_addr")
    assert row["status"] == "unavailable"
    with pytest.raises(ValueError, match="complete selected address"):
        source_flag_offsets(selection, bundle="Packed", name="local_addr", flags={"full": "full"})


def test_edited_source_forged_record_and_foreign_fact_pin_refuse(tmp_path):
    path, digest, selection = _selection(tmp_path)
    facts = _facts(path, digest)
    forged = copy.deepcopy(selection)
    forged["fields"][0]["width"] = 16
    with pytest.raises(ValueError, match="actual source observation"):
        apply_selected_layouts(facts, forged)
    foreign = copy.deepcopy(facts)
    foreign["inputs"]["firrtl_inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another selected FIRRTL"):
        apply_selected_layouts(foreign, selection)
    facts["facts"]["interfaces"][0]["bundles"]["Packed"]["fields"]["num_rows"]["offset"] = 47
    with pytest.raises(ValueError, match="contradicts"):
        apply_selected_layouts(facts, selection)
    path.write_text(SOURCE + "\n")
    with pytest.raises(ValueError, match="bytes changed"):
        apply_selected_layouts(_facts(path, digest), selection)


def test_duplicate_definition_and_duplicate_request_refuse(tmp_path):
    with pytest.raises(ValueError, match="duplicate source definitions"):
        _selection(tmp_path, SOURCE.replace("node rows =", "node rows = UInt<5>(0)\n    node rows ="))
    path, digest, selection = _selection(tmp_path)
    with pytest.raises(ValueError, match="duplicate fields"):
        derive_operand_layouts(path, expected_sha256=digest,
                               selections=[selection["selections"][0], selection["selections"][0]])
