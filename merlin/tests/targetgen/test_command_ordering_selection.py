"""Owner-selected ordering supplements preserve original facts, contract, and source authority."""

import copy
import json

import pytest
import yaml

from merlin.targetgen.rtl import command_ordering as CO
from merlin.targetgen.rtl import command_ordering_selection as S
from merlin.targetgen.rtl import facts as RF


@pytest.fixture
def selected(tmp_path, monkeypatch):
    hw = tmp_path / "original.generic.mlir"
    hw.write_text("original selected hardware")
    facts = tmp_path / "facts.json"
    original = {
        "inputs": {"target": "t", "generic_hw_path": str(hw), "generic_hw_sha256": S._sha(hw)},
        "source_consistency": {"status": "verified"},
        "facts": {"original_geometry": 17},
    }
    facts.write_text(json.dumps(original))
    contract, declarations = (tmp_path / name for name in ("contract.yaml", "decl.yaml"))
    base = {
        "encoding": {"unchanged": 1},
        "rtl_extraction": {"timing_structure": {"dispatch": {"module": "Dispatch", "classes": {"load": "ld_"}}}},
    }
    supplement = copy.deepcopy(base)
    supplement["encoding"] = {"untrusted_other_fields": 9}  # Never used to derive semantics.
    supplement["rtl_extraction"]["timing_structure"]["dispatch"]["ordering"] = {"reset": "reset"}
    contract.write_text(yaml.safe_dump(base))
    declarations.write_text(yaml.safe_dump(supplement))
    monkeypatch.setattr(RF, "find_facts", lambda *_: facts)
    monkeypatch.setattr(RF, "target_contract_path", lambda *_: contract)
    calls = []

    def derive(target, generic, body, *, contract):
        calls.append(contract)
        assert contract["encoding"] == base["encoding"]
        assert body == original["facts"] and generic == hw
        dispatch = contract["rtl_extraction"]["timing_structure"]["dispatch"]
        return {
            "schema": CO.SCHEMA,
            "status": "derived",
            "provenance": {
                "core_hw_generic": {"path": str(hw), "sha256": S._sha(hw)},
                "declarations_sha256": S._digest(dispatch),
            },
        }

    monkeypatch.setattr(CO, "derive_from_paths", derive)
    output = tmp_path / "selection.json"
    pin = S.prepare(target="t", facts=facts, contract=contract, declarations=declarations, output=output)
    return dict(
        pin=pin,
        original=original,
        facts=facts,
        hw=hw,
        contract=contract,
        declarations=declarations,
        output=output,
        calls=calls,
    )


def test_selection_reopens_all_original_inputs_without_changing_them(selected):
    s = selected
    record, identity = S.read(s["pin"], target="t", document=s["original"])
    assert record["status"] == "derived" and identity["selection"] == s["pin"]
    assert json.loads(s["facts"].read_text()) == s["original"]
    assert "ordering" not in yaml.safe_load(s["contract"].read_text())["rtl_extraction"]["timing_structure"]["dispatch"]
    assert len(s["calls"]) == 1  # Reopening a source-bound selection does not rerun extraction.


@pytest.mark.parametrize("changed", ["facts", "hw", "contract", "declarations", "output"])
def test_mutated_inputs_and_records_refuse(selected, changed):
    s = selected
    s[changed].write_text(s[changed].read_text() + " ")
    with pytest.raises(ValueError, match="changed|selected"):
        S.read(s["pin"], target="t", document=s["original"])


def test_other_selected_facts_do_not_inherit_the_sidecar(selected):
    document = copy.deepcopy(selected["original"])
    document["facts"]["original_geometry"] = 18
    with pytest.raises(ValueError, match="complete selected base facts"):
        S.read(selected["pin"], target="t", document=document)


def test_modified_readers_are_not_accepted(selected, monkeypatch):
    monkeypatch.setattr(S, "_readers", lambda: {"unknown": "reader"})
    with pytest.raises(ValueError, match="reader identity"):
        S.read(selected["pin"], target="t", document=selected["original"])


def test_supplement_cannot_replace_dispatch_or_existing_order():
    base = {"rtl_extraction": {"timing_structure": {"dispatch": {"module": "Original"}}}}
    extra = copy.deepcopy(base)
    extra["rtl_extraction"]["timing_structure"]["dispatch"] = {"module": "Other", "ordering": {}}
    with pytest.raises(ValueError, match="module or classes"):
        S._effective_contract(base, extra)
    extra["rtl_extraction"]["timing_structure"]["dispatch"]["module"] = "Original"
    base["rtl_extraction"]["timing_structure"]["dispatch"]["ordering"] = {"original": True}
    with pytest.raises(ValueError, match="conflict"):
        S._effective_contract(base, extra)


def test_normalized_probe_projection_uses_actual_bound_source_fields(tmp_path):
    from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts

    source = tmp_path / "array.fir"
    source.write_text("""FIRRTL version 4.2.0
circuit Array:
  module Array:
    input rs2 : UInt<64>
    wire extent : { rows : UInt<5>, local_addr : UInt<8> }
    connect extent.rows, bits(rs2, 24, 20)
    connect extent.local_addr, bits(rs2, 7, 0)
""")
    isa = tmp_path / "isa.scala"
    isa.write_text("val INPUT = 4\n")
    original = {
        "inputs": {"firrtl_inputs": [S._pin(source)], "isa_path": str(isa), "isa_sha256": S._sha(isa)},
        "facts": {
            "interfaces": [
                {"name": "funct_decode_table", "legal_funct": [4], "custom_opcode": 43, "funct3": 7},
                {
                    "name": "register_bundle_layouts",
                    "bundles": {
                        "Extent": {
                            "width": 64,
                            "fields": {
                                "rows": {"offset": 20, "width": None, "slot_width": 16},
                                "local_addr": {"offset": 0, "width": None, "slot_width": 20},
                            },
                        }
                    },
                },
            ]
        },
    }
    layouts = derive_operand_layouts(
        source,
        expected_sha256=S._sha(source),
        selections=[
            {"bundle": "Extent", "name": name, "module": "Array", "field": f"extent.{name}", "input": "rs2"}
            for name in ("rows", "local_addr")
        ],
    )
    selected = {
        "schema": "merlin.selected_operand_protocol.v1",
        "target": "array",
        "original_facts_sha256": S._digest(original),
        "source_pins": [S._pin(isa)],
        "layouts": layouts,
        "protocol": {
            "version": 1,
            "encoding": {"semantic_class": {"4": "MVIN"}},
            "roles": {
                "classes": {
                    "MVIN": {
                        "fields": [
                            {"name": "spad_addr", "bundle": "Extent", "field": "local_addr", "operand": "rs2"},
                            {"name": "rows", "bundle": "Extent", "field": "rows", "operand": "rs2"},
                        ]
                    }
                }
            },
        },
    }
    facts_file, selection_file = (tmp_path / n for n in ("original.json", "operand.json"))
    facts_file.write_text(json.dumps(original, sort_keys=True))
    selection_file.write_text(json.dumps(selected))
    contract = {
        "name": "array",
        "encoding": {"addr_len": 8},
        "rtl_checks": ["decode_clean"],
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": [{"funct": 4, "class": "INPUT", "operands": {}}],
        },
        "rocc_operand_selection": {"original_facts": S._pin(facts_file), "sidecar": S._pin(selection_file)},
    }
    before = copy.deepcopy((contract, original))
    projected, observed, files = S._projection("array", contract, original)
    assert (contract, original) == before
    assert projected["encoding"]["semantic_class"] == {"4": "MVIN"}
    assert projected["rtl_checks"]["local_addresses"] == [{"role": "MVIN", "address": "spad_addr"}]
    fields = observed["interfaces"][1]["bundles"]["Extent"]["fields"]
    assert fields["local_addr"]["width"] == 8 and fields["rows"]["width"] == 5
    assert {p["path"] for p in files} >= {str(source), str(isa), str(facts_file), str(selection_file)}


def test_command_line_readers_match_the_imported_module(monkeypatch):
    """A selection prepared with ``python -m`` must verify under the grader's ordinary import."""
    import importlib.util
    import sys
    from pathlib import Path

    # What ``python -m`` runs: the module source as ``__main__``, its spec keeping the import name.
    source = Path(S.__file__)
    namespace = {
        "__name__": "__main__",
        "__spec__": importlib.util.find_spec(S.__name__),
        "__file__": str(source),
        "__package__": S.__package__,
    }
    monkeypatch.setattr(sys, "argv", ["prepare", "--help"])
    with pytest.raises(SystemExit):
        exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert namespace["_readers"]() == S._readers()
