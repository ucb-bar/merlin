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
    input clock : Clock
    input rs2 : UInt<64>
    reg rows_reg : UInt<5>, clock
    reg unrelated_reg : UInt<5>, clock
    wire extent : { rows : UInt<5>, addr : { flag : UInt<1>, data : UInt<7> } }
    node rows = bits(rs2, 24, 20)
    connect extent.rows, rows
    node data = bits(rs2, 6, 0)
    connect extent.addr.data, data
    node flag = bits(rs2, 7, 7)
    connect extent.addr.flag, flag
    connect rows_reg, extent.rows
    connect unrelated_reg, rows
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


def test_recipe_projects_every_derived_field_of_a_bound_bundle(selection):
    original, blob, path, recipe, *_ = selection
    # The configuration consumer reads the same rs2 packing; bind it explicitly.
    recipe["class_operand_bundles"] = {"CONFIG_ST": {"rs2": "Extent"}}
    path.write_text(yaml.safe_dump(recipe))
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    fields = sidecar["protocol"]["roles"]["classes"]["CONFIG_ST"]["fields"]
    assert [row["name"] for row in fields] == ["subtype", "rows", "addr"]
    assert all(row.get("bundle") == "Extent" for row in fields[1:])
    assert R.layout_field_roster(sidecar) == {"status": "complete", "derived_fields": 2, "unprojected_fields": []}
    data = json.dumps(sidecar).encode()
    observer = P.bind_original_operand_selection(
        original,
        original_facts_bytes=blob,
        selection_bytes=data,
        expected_selection_sha256=hashlib.sha256(data).hexdigest(),
    )
    klass, decoded = observer.decode_instruction(0, {}, {"kind": "const", "raw": 5}, observer.isa_constants("array"))
    assert klass == "CONFIG_ST" and decoded["addr"] == 5 and decoded["rows"] == 0


def test_layout_roster_reports_unprojected_derived_fields(selection):
    original, blob, path, *_ = selection
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    sidecar["protocol"]["roles"]["classes"]["MVIN"]["fields"].pop(0)
    roster = R.layout_field_roster(sidecar)
    assert roster["status"] == "declared_subset"
    assert roster["unprojected_fields"] == [{"bundle": "Extent", "field": "rows"}]


@pytest.mark.parametrize(
    "binding",
    [{"UNDECLARED": {"rs2": "Extent"}}, {"CONFIG_ST": {"rs2": "Missing"}}, {"CONFIG_ST": {"rd": "Extent"}}],
)
def test_recipe_refuses_unselected_class_bundle_bindings(selection, binding):
    original, blob, path, recipe, *_ = selection
    recipe["class_operand_bundles"] = binding
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        R.derive(original, original_facts_bytes=blob, recipe_path=path)


_ROWS_KIND = {"label": "MVIN.rows", "kind": "extent", "consumer": {"module": "Array", "sink": "rows_reg"}}


def test_field_kinds_are_verified_against_their_firrtl_consumer(selection):
    original, blob, path, recipe, *_ = selection
    recipe["field_kinds"] = [_ROWS_KIND]
    path.write_text(yaml.safe_dump(recipe))
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    assert R.derive(original, original_facts_bytes=blob, recipe_path=path) == sidecar
    assert sidecar["operand_field_kinds"] == {"MVIN.rows": "extent"}
    [source] = sidecar["operand_field_kind_sources"]
    assert source["source"] == "extent.rows" and source["sink"] == "rows_reg" and source["firrtl_lines"]
    assert source["sink_declaration"]["keyword"] == "reg"
    data = json.dumps(sidecar).encode()
    P.bind_original_operand_selection(
        original,
        original_facts_bytes=blob,
        selection_bytes=data,
        expected_selection_sha256=hashlib.sha256(data).hexdigest(),
    )
    structure = P.decoder_structure(sidecar)
    assert set(structure) == {"operand_field_kinds", "operand_layout_roster"}
    assert structure["operand_layout_roster"]["status"] == "complete"
    # A projected field with no declared consumer stays absent, hence unknown.
    assert "MVIN.selected" not in structure["operand_field_kinds"]


def test_undeclared_field_kinds_stay_absent(selection):
    original, blob, path, *_ = selection
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    assert "operand_field_kinds" not in sidecar
    assert P.decoder_structure(sidecar)["operand_field_kinds"] == {}


@pytest.mark.parametrize(
    "entry",
    [
        {**_ROWS_KIND, "kind": "shift"},
        {**_ROWS_KIND, "consumer": {"module": "Array", "sink": "unrelated_reg"}},
        {**_ROWS_KIND, "consumer": {"module": "Array", "sink": "rows"}},
        {**_ROWS_KIND, "consumer": {"module": "Other", "sink": "rows_reg"}},
        {**_ROWS_KIND, "label": "CONFIG_ST.subtype"},
        {**_ROWS_KIND, "label": "MVIN.absent"},
        {**_ROWS_KIND, "note": "observed value"},
    ],
)
def test_field_kinds_refuse_unverified_consumers(selection, entry):
    original, blob, path, recipe, *_ = selection
    recipe["field_kinds"] = [entry]
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        R.derive(original, original_facts_bytes=blob, recipe_path=path)


def test_field_kinds_refuse_duplicate_labels(selection):
    original, blob, path, recipe, *_ = selection
    recipe["field_kinds"] = [_ROWS_KIND, _ROWS_KIND]
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        R.derive(original, original_facts_bytes=blob, recipe_path=path)


def test_decoder_structure_refuses_uncited_kinds(selection):
    original, blob, path, recipe, *_ = selection
    recipe["field_kinds"] = [_ROWS_KIND]
    path.write_text(yaml.safe_dump(recipe))
    sidecar = R.derive(original, original_facts_bytes=blob, recipe_path=path)
    sidecar["operand_field_kinds"]["MVIN.selected"] = "scale"
    with pytest.raises(ValueError):
        P.decoder_structure(sidecar)


def test_target_recipe_entries_cite_rtl_sources():
    from merlin.common.paths import repo_root

    contracts = repo_root() / "examples" / "gemmini" / "target" / "contracts"
    recipe = yaml.safe_load((contracts / "operand_selection_recipe.yaml").read_text())
    derivation = (contracts / "OPERAND_RECIPE_DERIVATION.md").read_text()
    for source in recipe["bundles"].values():
        assert {"module", "field", "input"} <= set(source)
    for row in recipe["fields"]:
        assert {"bundle", "name", "module", "field", "input"} <= set(row)
    for klass, operands in recipe.get("class_operand_bundles", {}).items():
        assert klass in derivation
        for bundle in operands.values():
            assert bundle in recipe["bundles"] and bundle in derivation
    labels = [entry["label"] for entry in recipe.get("field_kinds", [])]
    assert len(labels) == len(set(labels))
    for entry in recipe.get("field_kinds", []):
        assert set(entry) == {"label", "kind", "consumer"} and entry["kind"] in R.FIELD_KINDS
        assert set(entry["consumer"]) == {"module", "sink"}
        assert entry["label"] in derivation and entry["consumer"]["sink"] in derivation


def test_recipe_kinds_match_the_machine_evidence_consumer():
    from merlin.kernels.machine_evidence import FIELD_KINDS

    assert R.FIELD_KINDS == FIELD_KINDS
