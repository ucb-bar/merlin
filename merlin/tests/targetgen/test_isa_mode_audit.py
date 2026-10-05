"""A mode ledger cannot shrink the selected decoder population or certify itself."""

from __future__ import annotations

import copy
import json

import yaml

from merlin.targetgen.cli import main as targetgen_main
from merlin.targetgen.isa_mode_audit import audit_mode_inventory


def _inputs():
    controls = ["N"] * 17
    census = {
        "schema": "merlin.isa_source_census.v1",
        "rtl_revision": "a" * 40,
        "source_revision_verification": {
            "status": "verified", "rtl_revision": "a" * 40, "model_revision": "b" * 40,
        },
        "summary": {
            "patterns_not_decoded": [], "decoder_rows_without_pattern": [],
            "model_classes_without_compatible_pattern": [], "overlapping_patterns": [],
            "dma_kind_conflicts": [],
        },
        "rows": [
            {"name": "MOVE_A", "pattern_bits": "0" * 32, "decode_controls": controls},
            {"name": "MOVE_B", "pattern_bits": "1" * 32, "decode_controls": controls},
        ],
    }
    inventory = {
        "selected_sources": {"rtl_revision": "a" * 40, "model_revision": "b" * 40},
        "parameter_domains": {"registers": {
            "kind": "integer", "unit": "register_index", "reviewed": True,
            "evidence_sources": ["rtl/register_file.sv"],
            "intervals": [{"min": 0, "max": 7, "step": 1}],
        }},
        "variants": [
            {
                "id": name, "rtl_bitpat": bit * 32,
                "family": "transfer",
                "rtl_decode_controls": ",".join(controls),
                "required": True, "dialect_op": "demo.move", "mode_attrs": {"kind": kind},
                "parameter_domains": ["registers"], "software_admitted": True, "blocked": [],
                "parameter_bindings": {"registers": [{"kind": "attribute", "name": "register_index"}]},
            }
            for name, bit, kind in (("MOVE_A", "0", "a"), ("MOVE_B", "1", "b"))
        ],
    }
    plan = {
        "target": "demo", "dialect_name": "demo", "types": [], "lowering": [],
        "ops": [{"name": "move", "signature": {
            "operands": [{"name": "src", "type": "i32"}],
            "results": [{"name": "dst", "type": "i32"}],
            "attributes": [
                {"name": "kind", "type": "string", "role": "mode", "choices": ["a", "b"]},
                {"name": "register_index", "type": "i32", "role": "binding", "unit": "register_index", "min": 0, "max": 7},
            ],
            "effects": [],
        }}],
    }
    return census, inventory, plan


def test_parameterized_operation_accounts_for_every_decoder_mode():
    census, inventory, plan = _inputs()
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["mode_inventory_ready"]
    assert report["typed_mode_binding_ready"]
    assert report["counts"]["typed_mode_bindings"] == 2
    assert report["counts"]["required_modes"] == 2
    assert report["counts"]["selected_decoder_modes"] == 2
    assert report["qualification"].startswith("phase1_mode_scope_ready")


def test_phase1_scope_is_ready_before_dialect_authoring_or_software_admission():
    census, inventory, _ = _inputs()
    for row in inventory["variants"]:
        row["software_admitted"] = False
        row["blocked"] = ["numerical_semantics_requires_review"]
        row.pop("dialect_op")
        row.pop("mode_attrs")
    report = audit_mode_inventory(census, inventory)
    assert report["phase1_mode_scope_ready"]
    assert not report["mode_inventory_ready"]
    assert report["counts"]["required_modes"] == 2
    assert report["counts"]["phase1_scope_problem_kinds"] == {}


def test_machine_parameter_domains_require_reviewed_finite_structure():
    census, inventory, _ = _inputs()
    inventory["parameter_domains"]["registers"] = "registers 0 through 7"
    report = audit_mode_inventory(census, inventory)
    assert report["phase1_mode_scope_ready"]
    assert not report["phase1_parameter_domains_ready"]
    assert report["counts"]["phase1_parameter_domain_problem_kinds"] == {
        "parameter_domain_unstructured": 2,
    }

    inventory["parameter_domains"]["registers"] = {
        "kind": "integer", "unit": "register_index", "reviewed": True,
        "evidence_sources": ["rtl/register_file.sv"],
        "intervals": [{"min": 0, "max": 6, "step": 2}],
    }
    assert audit_mode_inventory(census, inventory)["phase1_parameter_domains_ready"]
    inventory["parameter_domains"]["registers"]["intervals"][0]["max"] = 7
    assert audit_mode_inventory(census, inventory)["counts"]["phase1_parameter_domain_problem_kinds"] == {
        "parameter_domain_malformed": 2,
    }
    inventory["parameter_domains"]["registers"] = {
        "kind": "enum", "unit": "register_index", "reviewed": True,
        "evidence_sources": ["rtl/register_file.sv"], "values": [0, 2, 4],
    }
    assert audit_mode_inventory(census, inventory)["phase1_parameter_domains_ready"]
    inventory["parameter_domains"]["registers"]["evidence_sources"] = ["../other_file"]
    assert audit_mode_inventory(census, inventory)["counts"]["phase1_parameter_domain_problem_kinds"] == {
        "parameter_domain_evidence_missing": 2,
    }
    inventory["parameter_domains"]["registers"]["evidence_sources"] = ["rtl/register_file.sv"]
    inventory["parameter_domains"]["registers"]["values"] = [0, True]
    assert not audit_mode_inventory(census, inventory)["phase1_parameter_domains_ready"]


def test_phase1_scope_refuses_unaccounted_modes_and_silent_exclusions():
    census, inventory, _ = _inputs()
    inventory["variants"].pop()
    report = audit_mode_inventory(census, inventory)
    assert not report["phase1_mode_scope_ready"]
    assert report["counts"]["phase1_scope_problem_kinds"] == {"selected_mode_missing_from_inventory": 1}

    census, inventory, _ = _inputs()
    inventory["variants"][0]["required"] = False
    report = audit_mode_inventory(census, inventory)
    assert not report["phase1_mode_scope_ready"]
    assert report["counts"]["phase1_scope_problem_kinds"] == {"scope_exclusion_unreviewed": 1}
    inventory["variants"][0]["scope_exclusion"] = {
        "reason": "not selected in the executable configuration",
        "evidence": "reviewed architecture selection",
        "reviewed": True,
    }
    report = audit_mode_inventory(census, inventory)
    assert report["phase1_mode_scope_ready"]
    assert report["counts"]["required_modes"] == 1


def test_phase1_scope_requires_reviewed_per_mode_model_encoding_decision():
    census, inventory, plan = _inputs()
    inventory["variants"][0]["model_classes"] = ["MODEL_A"]
    census["rows"][0]["model_candidates"] = []
    report = audit_mode_inventory(census, inventory)
    assert not report["source_reconciled"]
    assert report["unresolved_source_discrepancies"] == [
        {"kind": "mode_model_encoding_disagrees", "item": "MOVE_A"},
    ]
    inventory["source_resolutions"] = [{
        "kind": "mode_model_encoding_disagrees", "item": "MOVE_A",
        "authority": "selected_rtl", "reviewed": True, "evidence": "public mode discriminator",
    }]
    report = audit_mode_inventory(census, inventory)
    assert report["source_reconciled"]
    assert report["phase1_mode_scope_ready"]
    assert not report["mode_inventory_ready"]
    completed = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert completed["mode_inventory_ready"]
    assert completed["counts"]["problem_kinds"] == {"model_encoding_disagrees": 1}
    assert completed["counts"]["qualification_problem_kinds"] == {}
    assert completed["counts"]["modes_with_open_obligations"] == 0


def test_mode_binding_checks_typed_attribute_domain_and_role():
    census, inventory, plan = _inputs()
    inventory["variants"][0]["mode_attrs"]["kind"] = "other"
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["mode_inventory_ready"]
    assert report["counts"]["typed_binding_problem_kinds"] == {"mode_attribute_out_of_domain": 1}

    inventory["variants"][0]["mode_attrs"] = {}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"required_mode_attribute_missing": 1}

    inventory["variants"][0]["mode_attrs"] = {"kind": "a", "register_index": 2}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"mode_attribute_not_in_plan": 1}

    plan["ops"][0]["signature"]["attributes"][0]["role"] = "binding"
    inventory["variants"][0]["mode_attrs"] = {"kind": "a"}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {
        "mode_attribute_not_in_plan": 2, "unbound_physical_field": 2,
    }


def test_physical_domain_binding_is_checked_against_typed_fields():
    census, inventory, plan = _inputs()
    plan["ops"][0]["signature"]["attributes"][1]["max"] = 6
    assert audit_mode_inventory(census, inventory, dialect_plan=plan)["counts"]["typed_binding_problem_kinds"] == {
        "parameter_binding_domain_mismatch": 2,
    }
    plan["ops"][0]["signature"]["attributes"][1]["max"] = 7
    plan["ops"][0]["signature"]["attributes"][1]["unit"] = "byte"
    assert audit_mode_inventory(census, inventory, dialect_plan=plan)["counts"]["typed_binding_problem_kinds"] == {
        "parameter_binding_domain_mismatch": 2,
    }
    plan["ops"][0]["signature"]["attributes"][1]["unit"] = "register_index"
    inventory["variants"][0].pop("parameter_bindings")
    assert audit_mode_inventory(census, inventory, dialect_plan=plan)["counts"]["typed_binding_problem_kinds"] == {
        "parameter_binding_missing": 1,
    }
    inventory["variants"][0]["parameter_bindings"] = {
        "registers": [{"kind": "attribute", "name": "nonexistent"}],
    }
    assert audit_mode_inventory(census, inventory, dialect_plan=plan)["counts"]["typed_binding_problem_kinds"] == {
        "parameter_binding_field_missing": 1, "unbound_physical_field": 1,
    }


def test_stepped_type_parameter_can_bind_reviewed_domain():
    census, inventory, plan = _inputs()
    inventory["parameter_domains"]["registers"]["intervals"] = [{"min": 0, "max": 6, "step": 2}]
    plan["types"] = [{"name": "pair", "parameters": [{
        "name": "first", "kind": "unsigned", "unit": "register_index",
        "intervals": [{"min": 0, "max": 6, "step": 2}],
    }]}]
    signature = plan["ops"][0]["signature"]
    signature["operands"][0]["type"] = "!demo.pair"
    signature["attributes"][1]["role"] = "policy"
    for row in inventory["variants"]:
        row["parameter_bindings"] = {"registers": [{
            "kind": "type_parameter", "value": "src", "name": "first",
        }]}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["typed_mode_binding_ready"]
    plan["types"][0]["parameters"][0]["intervals"][0]["step"] = 3
    plan["types"][0]["parameters"][0]["intervals"][0]["max"] = 6
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["typed_mode_binding_ready"]
    assert report["counts"]["typed_binding_problem_kinds"] == {"parameter_binding_domain_mismatch": 2}


def test_signed_attribute_intervals_bind_exact_physical_units():
    census, inventory, plan = _inputs()
    inventory["parameter_domains"]["registers"] = {
        "kind": "integer", "unit": "byte", "reviewed": True,
        "evidence_sources": ["rtl/offsets.sv"],
        "intervals": [{"min": -8, "max": -4, "step": 4}, {"min": 0, "max": 16, "step": 8}],
    }
    plan["ops"][0]["signature"]["attributes"][1].update({
        "unit": "byte", "intervals": copy.deepcopy(inventory["parameter_domains"]["registers"]["intervals"]),
    })
    del plan["ops"][0]["signature"]["attributes"][1]["min"]
    del plan["ops"][0]["signature"]["attributes"][1]["max"]
    assert audit_mode_inventory(census, inventory, dialect_plan=plan)["typed_mode_binding_ready"]
    plan["ops"][0]["signature"]["attributes"][1]["intervals"][1]["step"] = 4
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"parameter_binding_domain_mismatch": 2}


def test_name_only_plan_cannot_complete_machine_mode_binding():
    census, inventory, plan = _inputs()
    plan["ops"][0].pop("signature")
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_reconciled"]
    assert not report["typed_mode_binding_ready"]
    assert not report["mode_inventory_ready"]
    assert report["typed_plan_error"] == "dialect plan has no reviewed typed signatures"


def test_revision_string_without_verified_selected_bytes_is_not_source_bound():
    census, inventory, plan = _inputs()
    census["source_revision_verification"] = {"status": "unverified"}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["source_discrepancies"] == ["selected_source_revisions_unverified"]
    census["source_revision_verification"] = {
        "status": "verified", "rtl_revision": "a" * 40, "model_revision": "c" * 40,
    }
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_discrepancies"] == ["selected_source_revision_disagrees"]


def test_missing_changed_and_duplicate_modes_do_not_shrink_denominator():
    census, inventory, plan = _inputs()
    inventory["variants"].pop()
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["counts"]["selected_decoder_modes"] == 2
    assert report["counts"]["required_modes"] == 2
    assert report["modes"][1]["problems"] == ["selected_mode_missing_from_inventory"]

    census, inventory, plan = _inputs()
    inventory["variants"][0]["rtl_bitpat"] = "1" * 32
    inventory["variants"][1]["rtl_decode_controls"] = ",".join(["Y"] + ["N"] * 16)
    inventory["variants"].append(copy.deepcopy(inventory["variants"][1]))
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["counts"]["problem_kinds"] == {
        "duplicate_mode_id": 1, "rtl_decode_controls_changed": 1, "rtl_pattern_changed": 1,
    }


def test_admission_plan_and_parameter_unknowns_remain_visible():
    census, inventory, plan = _inputs()
    inventory["variants"][0]["software_admitted"] = False
    inventory["variants"][0]["blocked"] = ["numerical_semantics"]
    inventory["variants"][1]["parameter_domains"] = ["unknown"]
    inventory["variants"][1]["dialect_op"] = "demo.missing"
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_bound"]
    assert not report["mode_inventory_ready"]
    assert report["counts"]["problem_kinds"] == {
        "dialect_operation_not_in_plan": 1,
        "mode_qualification_open": 1,
        "parameter_domain_unresolved": 1,
        "software_admission_missing": 1,
    }


def test_model_disagreement_is_not_hidden_by_exact_rtl_mode_match():
    census, inventory, plan = _inputs()
    census["summary"]["model_classes_without_compatible_pattern"] = ["MODEL_MODE"]
    inventory["variants"][0]["model_classes"] = ["MODEL_MODE"]
    census["rows"][0]["model_candidates"] = []
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_bound"]
    assert not report["source_reconciled"]
    assert not report["mode_inventory_ready"]
    assert report["census_discrepancies"] == {"model_classes_without_compatible_pattern": ["MODEL_MODE"]}
    assert report["counts"]["problem_kinds"] == {"model_encoding_disagrees": 1}


def test_selected_rtl_resolution_requires_exact_reviewed_discrepancy():
    census, inventory, plan = _inputs()
    census["summary"]["model_classes_without_compatible_pattern"] = ["MODEL_MODE"]
    inventory["source_resolutions"] = [{
        "kind": "model_classes_without_compatible_pattern", "item": "MODEL_MODE",
        "authority": "selected_rtl", "reviewed": True, "evidence": "selected-core/mode-test",
    }]
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_reconciled"]
    assert report["mode_inventory_ready"]
    assert report["unresolved_source_discrepancies"] == []
    inventory["source_resolutions"][0]["item"] = "OLD_MODE"
    import pytest

    with pytest.raises(ValueError, match="stale, invented, or duplicated"):
        audit_mode_inventory(census, inventory, dialect_plan=plan)
    inventory["source_resolutions"][0]["item"] = "MODEL_MODE"
    inventory["source_resolutions"][0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed selected_rtl"):
        audit_mode_inventory(census, inventory, dialect_plan=plan)


def test_cli_refuses_open_modes_and_clears_stale_output_on_malformed_input(tmp_path, capsys):
    census, inventory, plan = _inputs()
    census_file = tmp_path / "census.json"
    inventory_file = tmp_path / "inventory.json"
    plan_file = tmp_path / "plan.yaml"
    output = tmp_path / "audit.json"
    census_file.write_text(json.dumps(census))
    inventory_file.write_text(json.dumps(inventory))
    plan_file.write_text(yaml.safe_dump(plan))
    args = [
        "audit-dialect-modes", "--census", str(census_file), "--inventory", str(inventory_file),
        "--dialect-plan", str(plan_file), "--out", str(output),
    ]
    assert targetgen_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "MODE_INVENTORY_READY"
    first_inventory_digest = json.loads(output.read_text())["inputs_sha256"]["inventory"]
    inventory["variants"][0]["software_admitted"] = False
    inventory_file.write_text(json.dumps(inventory))
    assert targetgen_main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "MODE_OBLIGATIONS_OPEN"
    assert output.is_file()
    assert json.loads(output.read_text())["inputs_sha256"]["inventory"] != first_inventory_digest
    inventory_file.write_text("invalid json")
    assert targetgen_main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert not output.exists()


def test_cli_phase1_input_scope_succeeds_without_preexisting_dialect(tmp_path, capsys):
    census, inventory, _ = _inputs()
    for row in inventory["variants"]:
        row["software_admitted"] = False
        row["blocked"] = ["phase1_implementation"]
        row.pop("dialect_op")
        row.pop("mode_attrs")
    census_file = tmp_path / "census.json"
    inventory_file = tmp_path / "inventory.json"
    output = tmp_path / "scope.json"
    census_file.write_text(json.dumps(census))
    inventory_file.write_text(json.dumps(inventory))
    command = [
        "audit-dialect-modes", "--census", str(census_file), "--inventory", str(inventory_file),
        "--require", "phase1-inputs", "--out", str(output),
    ]
    assert targetgen_main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PHASE1_INPUTS_READY"
    assert json.loads(output.read_text())["phase1_parameter_domains_ready"]
    assert not json.loads(output.read_text())["mode_inventory_ready"]
    inventory["parameter_domains"]["registers"] = "unbounded prose"
    inventory_file.write_text(json.dumps(inventory))
    assert targetgen_main(command) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "PARAMETER_DOMAINS_OPEN"
    assert json.loads(output.read_text())["counts"]["phase1_parameter_domain_problem_kinds"] == {
        "parameter_domain_unstructured": 2,
    }
    inventory["parameter_domains"]["registers"] = _inputs()[1]["parameter_domains"]["registers"]
    inventory["variants"].pop()
    inventory_file.write_text(json.dumps(inventory))
    assert targetgen_main(command) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_MISMATCH"
