"""An explicit sidecar observes fields without replacing original facts/roles."""
import copy
import hashlib
import json
import os

import pytest

from merlin.targetgen.rocc import selected_operand_protocol as P
from merlin.targetgen.rocc import semantics as S
from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts

FIRRTL = """FIRRTL version 4.2.0
circuit Array:
  module Array:
    input rs2 : UInt<64>
    wire extent : { rows : UInt<5> }
    node rows = bits(rs2, 24, 20)
    connect extent.rows, rows
"""


def selected(tmp_path):
    source = tmp_path / "array.fir"
    source.write_text(FIRRTL)
    isa = tmp_path / "isa.scala"
    isa.write_text("val INPUT = 4\n")
    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    facts = {"inputs": {"firrtl_inputs": [pin(source)], "isa_path": str(isa), "isa_sha256": pin(isa)["sha256"]},
             "facts": {"interfaces": [
                 {"name": "funct_decode_table", "legal_funct": [4], "custom_opcode": 43, "funct3": 7},
                 {"name": "register_bundle_layouts", "bundles": {
                     "Extent": {"width": 64, "fields": {"rows": {"offset": 20, "width": None, "slot_width": 16}}}}}]}}
    contract = {"name": "array", "rtl_checks": ["decode_clean", "legal_funct"],
                "rocc_operand_roles": {"version": 1, "word_bits": 64, "instructions": [
                    {"funct": 4, "class": "INPUT", "operands": {}}]}}
    original = S.bind(target="array", contract=contract, facts=facts)
    fact_bytes = json.dumps(facts).encode()
    layouts = derive_operand_layouts(source, expected_sha256=pin(source)["sha256"], selections=[
        {"bundle": "Extent", "name": "rows", "module": "Array", "field": "extent.rows", "input": "rs2"}])
    sidecar = {"schema": P.SCHEMA, "target": "array", "original_facts_sha256": hashlib.sha256(fact_bytes).hexdigest(),
               "source_pins": [pin(isa)], "layouts": layouts,
               "protocol": {"version": 1, "encoding": {"semantic_class": {4: "MVIN"}},
                            "roles": {"classes": {"MVIN": {"fields": [
                                {"name": "rows", "operand": "rs2", "bundle": "Extent", "field": "rows"}]}}}}}
    return original, facts, fact_bytes, sidecar, source, isa


def load(original, fact_bytes, sidecar):
    blob = json.dumps(sidecar, allow_nan=False).encode()
    return P.bind_original_operand_selection(original, original_facts_bytes=fact_bytes, selection_bytes=blob,
                                              expected_selection_sha256=hashlib.sha256(blob).hexdigest())


def test_detached_selection_rederives_fields_and_preserves_original_owner_and_facts(tmp_path):
    original, facts, blob, sidecar, _source, _isa = selected(tmp_path)
    before = copy.deepcopy(original._facts)
    observer = load(original, blob, sidecar)
    assert original.operand_observer is None
    assert original._facts == before == facts
    assert original.isa_constants("array")["FUNCT_CLASS"] == {4: "INPUT"}
    assert observer.decode_instruction(4, {}, {"kind": "const", "raw": 17 << 20},
                                       observer.isa_constants("array")) == ("MVIN", {"rows": 17})


def test_partial_scalar_bits_only_recover_complete_selected_source_fields(tmp_path):
    original, _facts, blob, sidecar, _source, _isa = selected(tmp_path)
    observer = load(original, blob, sidecar)
    operand = {"kind": "unknown", "raw": None,
               "known_bits": {"width": 64, "mask": 31 << 20, "value": 17 << 20}}
    assert observer.decode_instruction(4, {}, operand, observer.isa_constants("array")) == ("MVIN", {"rows": 17})
    operand["known_bits"]["mask"] ^= 1 << 21
    assert observer.decode_instruction(4, {}, operand, observer.isa_constants("array")) == ("MVIN", {"rows": None})


@pytest.mark.parametrize("change", [{"width": 32}, {"mask": 1 << 64}, {"value": 1}, {"mask": True}])
def test_partial_register_facts_cannot_change_selected_width_or_unknown_bits(tmp_path, change):
    original, _facts, blob, sidecar, _source, _isa = selected(tmp_path)
    observer = load(original, blob, sidecar)
    known = {"width": 64, "mask": 31 << 20, "value": 17 << 20, **change}
    with pytest.raises(ValueError, match="width or mask"):
        observer.decode_instruction(4, {}, {"kind": "unknown", "raw": None, "known_bits": known},
                                    observer.isa_constants("array"))


@pytest.mark.parametrize("mutation", ["facts", "foreign_owner", "foreign_target", "wrong_fact_pin",
    "forged_layout", "isa_source", "firrtl_source", "duplicate_pin", "missing_isa", "already_selected"])
def test_foreign_forged_or_mutated_selection_refuses(tmp_path, mutation):
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    if mutation == "facts":
        original._facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]["rows"]["offset"] = 19
    elif mutation == "foreign_owner":
        original = object()
    elif mutation == "foreign_target":
        sidecar["target"] = "other"
    elif mutation == "wrong_fact_pin":
        sidecar["original_facts_sha256"] = "0" * 64
    elif mutation == "forged_layout":
        sidecar["layouts"]["fields"][0]["width"] = 16
    elif mutation == "duplicate_pin":
        sidecar["source_pins"].append(sidecar["source_pins"][0])
    elif mutation == "missing_isa":
        sidecar["source_pins"] = []
    elif mutation == "already_selected":
        original.operand_observer = object()
    else:
        changed = source if mutation == "firrtl_source" else isa
        before = changed.stat()
        changed.write_bytes(changed.read_bytes().replace(b"4", b"5", 1))
        os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        load(original, blob, sidecar)


def test_selected_byte_pin_and_duplicate_json_cannot_be_bypassed(tmp_path):
    original, _facts, blob, sidecar, _source, _isa = selected(tmp_path)
    data = json.dumps(sidecar).encode()
    with pytest.raises(ValueError, match="selected byte pin"):
        P.bind_original_operand_selection(original, original_facts_bytes=blob, selection_bytes=data,
                                            expected_selection_sha256="0" * 64)
    data = b'{"schema":0,"schema":0}'
    with pytest.raises(ValueError, match="duplicate JSON"):
        P.bind_original_operand_selection(original, original_facts_bytes=blob, selection_bytes=data,
                                            expected_selection_sha256=hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("which", ["controller", "firrtl", "owner"])
def test_source_or_owner_mutation_at_final_boundary_refuses_before_return(tmp_path, monkeypatch, which):
    original, _facts, blob, sidecar, source, isa = selected(tmp_path)
    construct = P.BoundOperandObserver
    def interrupted(**kwargs):
        observed = construct(**kwargs)
        if which == "owner":
            original._isa["FUNCT_CLASS"][4] = "CHANGED"
        else:
            changed = source if which == "firrtl" else isa
            before = changed.stat()
            changed.write_bytes(changed.read_bytes().replace(b"4", b"5", 1))
            os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
        return observed
    monkeypatch.setattr(P, "BoundOperandObserver", interrupted)
    with pytest.raises(ValueError, match="changed"):
        load(original, blob, sidecar)
    assert original.operand_observer is None
