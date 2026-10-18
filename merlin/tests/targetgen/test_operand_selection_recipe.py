"""Regeneration uses actual selected source slices, never reserved packing slots."""

import copy
import hashlib
import json
import os

import pytest
import yaml

from merlin.targetgen.rocc import operand_selection_recipe as R
from merlin.targetgen.rocc import selected_operand_protocol as P
from merlin.targetgen.rocc.semantics import bind


@pytest.fixture
def selection(tmp_path):
    source = tmp_path / "array.fir"
    source.write_text("""FIRRTL version 4.2.0
circuit Array:
  module Array:
    input rs2 : UInt<64>
    wire extent : { rows : UInt<5>, addr : { flag : UInt<1>, data : UInt<7> } }
    node rows = bits(rs2, 24, 20)
    connect extent.rows, rows
    node data = bits(rs2, 6, 0)
    connect extent.addr.data, data
    node flag = bits(rs2, 7, 7)
    connect extent.addr.flag, flag
""")
    isa = tmp_path / "isa.scala"
    isa.write_text("val CONFIG_CMD = 0.U\nval INPUT = 4.U\nval CONFIG_KIND = 0.U\n")

    def pin(p):
        return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    facts = {
        "inputs": {
            "target": "array",
            "isa_path": str(isa),
            "isa_sha256": pin(isa)["sha256"],
            "firrtl_inputs": [pin(source)],
        },
        "facts": {
            "interfaces": [
                {"name": "funct_decode_table", "legal_funct": [0, 4], "custom_opcode": 43, "funct3": 7},
                {
                    "name": "register_bundle_layouts",
                    "bundles": {
                        "Extent": {
                            "width": 64,
                            "fields": {
                                "rows": {"offset": 20, "width": None, "slot_width": 16},
                                "addr": {"offset": 0, "width": None, "slot_width": 16},
                            },
                        }
                    },
                },
            ]
        },
    }
    contract = {
        "name": "array",
        "rtl_checks": ["decode_clean", "legal_funct"],
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": [
                {"funct": 0, "class": "CONFIG_CMD", "operands": {}},
                {"funct": 4, "class": "INPUT", "operands": {}},
            ],
        },
    }
    declaration = {
        "encoding": {"config_subtype": {0: "CONFIG_ST"}},
        "rocc_operand_roles": {
            "config": {"class": "CONFIG", "selector": {"operand": "rs2", "bundle": "Extent", "field": "rows"}},
            "classes": {
                "MVIN": {
                    "fields": [
                        {"name": "rows", "operand": "rs2", "bundle": "Extent", "field": "rows"},
                        {
                            "name": "selected",
                            "operand": "rs2",
                            "bundle": "Extent",
                            "field": "addr",
                            "as": "flag",
                            "flag": "selected",
                        },
                    ]
                },
                "CONFIG_ST": {"fields": [{"name": "subtype", "const": "ST"}]},
            },
        },
    }
    (tmp_path / "semantics.yaml").write_text(yaml.safe_dump(declaration))
    recipe = {
        "schema": "merlin.operand_selection_recipe.v1",
        "semantic_declaration": "semantics.yaml",
        "aliases": {"CONFIG_CMD": "CONFIG", "INPUT": "MVIN"},
        "config_subtype_sources": {"CONFIG_KIND": "CONFIG_ST"},
        "isa_sibling_sources": [],
        "bundles": {"Extent": {"module": "Array", "field": "extent", "input": "rs2"}},
        "fields": [],
        "load_state_selection": None,
        "address": {"bundle": "Extent", "name": "addr", "flags": {"selected": "flag"}, "agree": []},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    original = bind(target="array", contract=contract, facts=facts)
    return original, json.dumps(facts).encode(), path, recipe, source, isa


def test_recipe_regenerates_exact_widths_flags_and_preserves_original_facts(selection):
    original, blob, path, *_ = selection
    before = copy.deepcopy(original._facts)
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    assert R.derive(original, original_facts_bytes=blob, recipe_path=path) == sidecar
    rows = {(row["bundle"], row["name"]): row for row in sidecar["layouts"]["fields"]}
    assert rows["Extent", "rows"]["width"] == 5
    assert rows["Extent", "addr"]["width"] == 8
    data = json.dumps(sidecar).encode()
    observer = P.bind_original_operand_selection(
        original,
        original_facts_bytes=blob,
        selection_bytes=data,
        expected_selection_sha256=hashlib.sha256(data).hexdigest(),
    )
    isa = observer.isa_constants("array")
    assert observer.decode_instruction(4, {}, {"kind": "const", "raw": (17 << 20) | 128}, isa) == (
        "MVIN",
        {"rows": 17, "selected": True},
    )
    assert original._facts == before and original.operand_observer is None


@pytest.mark.parametrize("change", ["isa", "source", "alias", "config", "field", "facts"])
def test_recipe_refuses_changed_sources_unselected_roles_and_unproved_flags(selection, change):
    original, blob, path, recipe, source, isa = selection
    if change in {"source", "isa"}:
        p = source if change == "source" else isa
        stat = p.stat()
        p.write_bytes(p.read_bytes().replace(b"4", b"5", 1))
        os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "alias":
        recipe["aliases"]["UNSELECTED"] = "NEW_CLASS"
    elif change == "config":
        recipe["config_subtype_sources"]["CONFIG_KIND"] = "CONFIG_UNSELECTED"
    elif change == "field":
        recipe["address"]["flags"]["selected"] = "missing"
    else:
        original._facts["inputs"]["target"] = "other"
    if change in {"alias", "config", "field"}:
        path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        R.derive(original, original_facts_bytes=blob, recipe_path=path)
