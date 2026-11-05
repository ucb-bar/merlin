"""Detached normalized protocol projections cannot invent unresolved RTL widths."""
import copy

import pytest

from merlin.targetgen.rocc import semantics as S


def selection():
    contract = {
        "name": "independent_array", "rtl_checks": ["decode_clean", "legal_funct"],
        "rocc_operand_roles": {"version": 1, "word_bits": 64, "instructions": [
            {"funct": 0, "class": "CONTROL", "operands": {}},
            {"funct": 4, "class": "INPUT", "operands": {}},
            {"funct": 9, "class": "OUTPUT", "operands": {}},
        ]},
        "rocc_operand_protocol": {
            "version": 1, "encoding": {"semantic_class": {0: "CONFIG", 4: "MVIN", 9: "MVOUT"},
                                        "config_subtype": {1: "CONFIG_LD", 2: "CONFIG_ST"}},
            "address_field": {"bundle": "Extent", "field": "address"},
            "roles": {"config": {"class": "CONFIG", "selector": {
                "operand": "rs1", "bundle": "Config", "field": "selector"}},
                "local_address_flags": {"full_width": {"msb_offset": 1}},
                "classes": {
                    "CONFIG_LD": {"fields": [
                        {"name": "state_id", "operand": "rs1", "bundle": "Config", "field": "state"},
                        {"name": "stride", "operand": "rs2", "as": "raw"}]},
                    "CONFIG_ST": {"fields": [{"name": "out_stride_bytes", "operand": "rs2", "as": "raw"}]},
                    "MVIN": {"fields": [
                        {"name": "dram", "operand": "rs1", "as": "operand"},
                        {"name": "rows", "operand": "rs2", "bundle": "Extent", "field": "rows"}]},
                    "MVOUT": {"fields": [
                        {"name": "dram", "operand": "rs1", "as": "operand"},
                        {"name": "readout", "operand": "rs2", "bundle": "Extent", "field": "address",
                         "as": "flag_choice", "flag": "full_width", "set": "i32", "clear": "i8"}]},
                }},
        },
    }
    facts = {"facts": {"interfaces": [
        {"name": "funct_decode_table", "legal_funct": [0, 4, 9], "custom_opcode": 43, "funct3": 7},
        {"name": "register_bundle_layouts", "bundles": {
            "Config": {"width": 64, "fields": {"selector": {"offset": 0, "width": 2},
                                                  "state": {"offset": 7, "width": 2}}},
            "Extent": {"width": 64, "fields": {"address": {"offset": 0, "width": 11},
                                                  "rows": {"offset": 20, "width": 5, "slot_width": 16}}},
        }},
    ]}}
    return contract, facts


def owner(contract, facts):
    return S.bind(target=contract["name"], contract=contract, facts=facts)


def test_normalized_config_and_dma_keep_raw_roster_and_original_pointer(monkeypatch):
    monkeypatch.setattr(S, "_load_manifest", lambda *args: pytest.fail("ambient manifest discovery"))
    monkeypatch.setattr(S, "_load_facts", lambda *args: pytest.fail("ambient facts discovery"))
    contract, facts = selection()
    raw = owner(contract, facts)
    isa = raw.isa_constants(contract["name"])
    assert isa["FUNCT_CLASS"] == {0: "CONTROL", 4: "INPUT", 9: "OUTPUT"}
    assert raw.decode_instruction(0, {"raw": 1}, {"raw": 32}, isa) == ("CONTROL", {})
    observed = raw.operand_observer
    assert observed.word_bits == 64
    selected = observed.isa_constants(contract["name"])
    assert observed.decode_instruction(0, {"kind": "const", "raw": 1}, {"kind": "const", "raw": 32}, selected) == (
        "CONFIG_LD", {"state_id": 0, "stride": 32})
    pointer = {"kind": "arg", "raw": None, "arg_index": 3, "offset": 12}
    assert observed.decode_instruction(4, pointer, {"kind": "const", "raw": 7 << 20}, selected) == (
        "MVIN", {"dram": pointer, "rows": 7})
    assert observed.decode_instruction(9, pointer, {"kind": "const", "raw": 1 << 9}, selected)[1]["readout"] == "i32"
    contract["rocc_operand_protocol"]["encoding"]["semantic_class"][4] = "CHANGED"
    facts["facts"]["interfaces"][1]["bundles"]["Config"]["fields"]["state"]["offset"] = 2
    decoded = observed.decode_instruction(0, {"kind": "const", "raw": 1}, {"kind": "const", "raw": 32}, selected)
    assert decoded[1]["state_id"] == 0


def test_reserved_slot_is_not_actual_field_width_or_readout_proof():
    contract, facts = selection()
    fields = facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]
    fields["rows"]["width"] = None
    fields["address"]["width"] = None
    fields["address"]["slot_width"] = 11
    obs = owner(contract, facts).operand_observer
    isa = obs.isa_constants(contract["name"])
    assert obs.decode_instruction(4, {"kind": "arg"}, {"kind": "const", "raw": 7 << 20}, isa)[1]["rows"] is None
    assert obs.decode_instruction(9, {"kind": "arg"}, {"kind": "const", "raw": 1 << 9}, isa)[1]["readout"] is None


def test_missing_selector_width_and_nonconstant_operands_remain_unknown():
    contract, facts = selection()
    facts["facts"]["interfaces"][1]["bundles"]["Config"]["fields"]["selector"].update(width=None, slot_width=2)
    obs = owner(contract, facts).operand_observer
    assert obs.decode_instruction(0, {"kind": "const", "raw": 1}, {"kind": "const", "raw": 32},
                                  obs.isa_constants(contract["name"])) == ("UNKNOWN", {})
    contract, facts = selection()
    obs = owner(contract, facts).operand_observer
    assert obs.decode_instruction(0, {"kind": "arg", "raw": None}, {},
                                  obs.isa_constants(contract["name"])) == ("UNKNOWN", {})


@pytest.mark.parametrize("mutation", ["unselected_code", "overlap", "invented_numeric", "foreign_program"])
def test_unbound_or_malformed_protocol_refuses(mutation):
    contract, facts = selection()
    if mutation == "unselected_code":
        contract["rocc_operand_protocol"]["encoding"]["semantic_class"][5] = "EXTRA"
    elif mutation == "overlap":
        facts["facts"]["interfaces"][1]["bundles"]["Config"]["fields"]["state"]["offset"] = 1
    elif mutation == "invented_numeric":
        contract["rocc_operand_protocol"]["roles"]["classes"]["MVIN"]["fields"] = [{"name": "rows", "const": 16}]
    else:
        obs = owner(contract, facts).operand_observer
        isa = obs.isa_constants(contract["name"])
        wrong = copy.deepcopy(isa)
        wrong["FUNCT_CLASS"][4] = "OTHER"
        with pytest.raises(ValueError):
            obs.decode_instruction(4, {}, {}, wrong)
        return
    with pytest.raises(ValueError):
        owner(contract, facts)


def test_unset_protocol_keeps_legacy_raw_selection_exactly():
    contract, facts = selection()
    contract.pop("rocc_operand_protocol")
    raw = owner(contract, facts)
    assert raw.operand_observer is None
    assert set(raw.isa_constants(contract["name"])) == {"CUSTOM_OPCODE", "FUNCT3", "FUNCT_CLASS"}


def compound_selection():
    contract, facts = selection()
    address = facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]["address"]
    address["source_subfields"] = [
        {"field": "data", "offset": 0, "width": 7},
        {"field": "garbage_bit", "offset": 7, "width": 1},
        {"field": "accumulate", "offset": 8, "width": 1},
        {"field": "full", "offset": 9, "width": 1},
        {"field": "accumulator", "offset": 10, "width": 1},
    ]
    fields = [{"name": name, "operand": "rs2", "bundle": "Extent", "field": "address." + name,
               "as": "all_ones"} for name in ("data", "garbage_bit", "accumulate", "full", "accumulator")]
    fields += [
        {"name": "garbage", "all": [{"from": name, "equals": True}
                                     for name in ("data", "garbage_bit", "accumulate", "full", "accumulator")]},
        {"name": "readout", "all": [{"from": "garbage", "equals": False},
                                      {"from": "accumulator", "equals": True},
                                      {"from": "full", "equals": True}], "set": "wide", "clear": "small"},
    ]
    contract["rocc_operand_protocol"]["roles"]["classes"]["MVOUT"]["fields"] = fields
    return contract, facts


@pytest.mark.parametrize("address,readout", [(0, "small"), (1 << 9, "small"),
    ((1 << 10) | (1 << 9), "wide"), ((1 << 11) - 1, "small"), ((1 << 11) - 2, "wide")])
def test_source_subfields_and_compound_readout_preserve_all_actual_predicates(address, readout):
    contract, facts = compound_selection()
    observed = owner(contract, facts).operand_observer
    result = observed.decode_instruction(9, {}, {"kind": "const", "raw": address},
                                         observed.isa_constants(contract["name"]))
    assert result[1]["readout"] == readout
    assert set(facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]) == {"address", "rows"}


def test_compound_unresolved_fields_stay_unknown_and_owner_program_remains_detached():
    contract, facts = compound_selection()
    observed = owner(contract, facts).operand_observer
    isa = observed.isa_constants(contract["name"])
    assert observed.decode_instruction(9, {}, {"kind": "arg", "raw": None}, isa)[1]["readout"] is None
    fields = contract["rocc_operand_protocol"]["roles"]["classes"]["MVOUT"]["fields"]
    fields[-1]["all"][0]["equals"] = True
    assert observed.decode_instruction(9, {}, {"kind": "const", "raw": 1536}, isa)[1]["readout"] == "wide"
    facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]["address"]["width"] = None
    unknown = owner(contract, facts).operand_observer
    assert unknown.decode_instruction(9, {}, {"kind": "const", "raw": 1536},
                                      unknown.isa_constants(contract["name"]))[1]["readout"] is None


@pytest.mark.parametrize("mutation", ["gap", "overlap", "incomplete", "duplicate", "forward", "mixed",
                                      "numeric_choice"])
def test_subfield_and_compound_forged_or_unsupported_projections_refuse(mutation):
    contract, facts = compound_selection()
    children = facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]["address"]["source_subfields"]
    field = contract["rocc_operand_protocol"]["roles"]["classes"]["MVOUT"]["fields"][-1]
    if mutation == "gap":
        children[1]["offset"] += 1
    elif mutation == "overlap":
        children[1]["offset"] -= 1
    elif mutation == "incomplete":
        children.pop()
    elif mutation == "duplicate":
        children[1]["field"] = "data"
    elif mutation == "forward":
        field["all"][0]["from"] = "readout"
    elif mutation == "mixed":
        field["operand"] = "rs2"
    else:
        field["set"] = 32
    with pytest.raises(ValueError):
        owner(contract, facts)
