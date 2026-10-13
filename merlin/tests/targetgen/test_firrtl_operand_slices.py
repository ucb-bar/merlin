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


def test_original_subfield_declaration_is_never_overwritten(tmp_path):
    path, digest, observation = _selection(tmp_path)
    original = _facts(path, digest)
    address = original["facts"]["interfaces"][0]["bundles"]["Packed"]["fields"]["local_addr"]
    address["source_subfields"] = [{"field": "different", "offset": 0, "width": 32}]
    with pytest.raises(ValueError, match="subfields contradict"):
        apply_selected_layouts(original, observation)


def _project(tmp_path, text, rows):
    path = tmp_path / "selected.fir"
    path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, derive_operand_layouts(path, expected_sha256=digest, selections=rows)


def test_selected_guarded_packing_keeps_exact_conditional_premise(tmp_path):
    text = SOURCE.replace("connect unpacked.rows, rows", "when control :\n      connect unpacked.rows, rows")
    row = {"bundle": "Packed", "name": "num_rows", "module": "Top", "field": "unpacked.rows",
           "input": "q.io.deq.bits.rs2", "packing_context": {"guards": ["control"]}}
    path, digest, selected = _project(tmp_path, text, [row])
    assert selected["schema"].endswith(".v2")
    assert selected["fields"][0]["width"] == 5
    assert selected["command_acceptance_and_semantic_roles"] == "unproven"
    updated = apply_selected_layouts(_facts(path, digest), selected)
    assert updated["facts"]["interfaces"][0]["bundles"]["Packed"]["fields"]["num_rows"]["width"] == 5
    for guards in (["not control"], ["control", "unrelated"]):
        wrong = copy.deepcopy(row)
        wrong["packing_context"]["guards"] = guards
        _, _, failed = _project(tmp_path, text, [wrong])
        assert failed["fields"][0]["status"] == "unavailable"
    # An alternative driver is still ambiguous, even if one selected guard matches.
    text += "    when control :\n      connect unpacked.rows, UInt<5>(0)\n"
    _, _, failed = _project(tmp_path, text, [row])
    assert failed["fields"][0]["status"] == "unavailable"


DYNAMIC_SOURCE = """FIRRTL version 4.2.0
circuit Top:
  module Top:
    input selector : UInt<2>
    input words : UInt<64>[3]
    wire address : UInt<32>
    connect address, words[selector]
    node cols = bits(words[selector], 36, 32)
    node selected_data = bits(address, 13, 0)
    wire local : { upper : UInt<18>, data : UInt<14> }
    connect local.data, selected_data
    node flags = bits(address, 31, 14)
    connect local.upper, flags
"""


def test_dynamic_root_node_and_unsigned_truncation_are_explicit_packing_only(tmp_path):
    rows = [{"bundle": "Packed", "name": "num_rows", "module": "Top", "field": "cols",
             "input": "words[selector]", "packing_context": {"dynamic_input": True, "node_field": True}},
            {"bundle": "Packed", "name": "local_addr", "module": "Top", "field": "local",
             "input": "words[selector]", "packing_context": {"dynamic_input": True, "truncate_aliases": True}}]
    _, _, selected = _project(tmp_path, DYNAMIC_SOURCE, rows)
    assert [(row["offset"], row["width"]) for row in selected["fields"]] == [(32, 5), (0, 32)]
    assert selected["selections"] == rows  # the selector remains explicit, never a queue-slot guess
    assert selected["command_acceptance_and_semantic_roles"] == "unproven"
    for name in ("dynamic_input", "node_field", "truncate_aliases"):
        changed = copy.deepcopy(rows)
        for row in changed:
            row["packing_context"][name] = False
        _, _, failed = _project(tmp_path, DYNAMIC_SOURCE, changed)
        assert any(row["status"] == "unavailable" for row in failed["fields"])
    for changed in (
        DYNAMIC_SOURCE.replace("words[selector]", "words[another]", 1),
        DYNAMIC_SOURCE.replace("wire address : UInt<32>", "reg address : UInt<32>, clock"),
        DYNAMIC_SOURCE.replace("input words : UInt<64>[3]", "input words : SInt<64>[3]"),
        DYNAMIC_SOURCE.replace("node cols = bits(words[selector], 36, 32)",
                               "node cols = mux(selector, bits(words[selector], 36, 32), UInt<5>(0))"),
    ):
        _, _, failed = _project(tmp_path, changed, rows)
        assert any(row["status"] == "unavailable" for row in failed["fields"])


@pytest.mark.parametrize("context", [{}, {"guards": []}, {"guards": [1]}, {"guards": "control"},
                                      {"dynamic_input": 1}, {"unknown": True}])
def test_packing_context_has_closed_explicit_premises(tmp_path, context):
    row = {"bundle": "Packed", "name": "num_rows", "module": "Top", "field": "cols",
           "input": "words[selector]", "packing_context": context}
    with pytest.raises(ValueError, match="packing"):
        _project(tmp_path, DYNAMIC_SOURCE, [row])
