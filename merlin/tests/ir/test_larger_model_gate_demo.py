"""Small protocol guards for opt-in model demos; no compiler/proof/native run."""

import json
from pathlib import Path

import pytest
from larger_model_gate_demo import (
    SCHEMA,
    _functions,
    _tensor_detail,
    audit_coverage,
    bound,
    file_ref,
    load_manifest,
    require_selectors,
)


def original_manifest(tmp_path):
    source = tmp_path / "complete-model.mlir"
    source.write_text("builtin.module {}\n")
    definition = tmp_path / "model-definition.py"
    definition.write_text("# Small manifest protocol control only; no model or array computation.\n")
    manifest = {
        "schema": SCHEMA,
        "id": "metadata_control",
        "source": file_ref(source),
        "provenance": [file_ref(definition)],
        "source_kind": "explicit_model_definition",
        "options": {},
        "textual": False,
        "required_changed_selectors": ["lower_quant_ext"],
        "required_verified_selectors": ["lower_quant_ext"],
        "native_contract": None,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_original_source_and_definition_are_bound_without_execution(tmp_path):
    path, expected = original_manifest(tmp_path)
    actual, source, options, refs = load_manifest(path)
    assert actual == expected and source == "builtin.module {}\n" and not options.enabled
    assert len(refs) == 3 and refs[2] == expected["provenance"][0]
    definition = Path(refs[2]["path"])
    definition.write_text("changed definition")
    with pytest.raises(ValueError, match="bound file bytes changed"):
        bound(refs[2])


def test_normalized_native_abi_is_descriptive_without_generating_inputs(tmp_path):
    path, value = original_manifest(tmp_path)
    value["native_contract"] = {
        "entry": "forward",
        "arguments": [{"name": "arg0", "canonical_input_name": "arg0", "shape": [2, 3], "dtype": "float16"}],
        "outputs": [{"name": "result0", "shape": [2, 4], "dtype": "float16"}],
        "scope": "finite deterministic baseline-versus-gated execution, no modelaccuracyclaim",
    }
    path.write_text(json.dumps(value))
    loaded, _, _, _ = load_manifest(path)
    assert loaded["native_contract"] == value["native_contract"]


@pytest.mark.parametrize(
    "fault", ["source_bytes", "native_contract", "derivative", "no_proof_selector", "unknown_selector"]
)
def test_invalid_or_unbound_demo_manifest_refuses_before_compilation(tmp_path, fault):
    path, value = original_manifest(tmp_path)
    if fault == "source_bytes":
        value["source"]["sha256"] = "0" * 64
    elif fault == "native_contract":
        value["native_contract"] = {"invented_inputs": True}
    elif fault == "derivative":
        value["source_kind"] = "derived_compiler_ir"
    elif fault == "no_proof_selector":
        value["required_verified_selectors"] = []
    else:
        value["required_verified_selectors"] = ["nonexistent_gate"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("status,changed", [("unchanged", False), ("unverified", True), ("not_observed", False)])
def test_changed_or_excluded_work_cannot_satisfy_required_proof_metadata(status, changed):
    manifest = {"required_changed_selectors": ["lower_quant_ext"], "required_verified_selectors": ["lower_quant_ext"]}
    coverage = {
        "boundaries": []
        if status == "not_observed"
        else [
            {
                "pass": "lower_quant_ext",
                "status": status,
                "proof_credit": False,
                "source_sha256": "a" * 64,
                "final_sha256": ("b" if changed else "a") * 64,
            }
        ]
    }
    with pytest.raises(ValueError):
        require_selectors(manifest, coverage)


@pytest.mark.parametrize("fault", ["missing_chain", "duplicate_function", "wrong_theorem", "nonstandard_axiom"])
def test_handler_acceptance_strings_cannot_replace_complete_tensor_evidence(fault):
    from merlin.verify.gate_checks import THEOREMS

    detail = {"status": "accepted", "functions": ["forward"], "theorems": list(THEOREMS), "axioms": [], "chains": {}}
    if fault == "duplicate_function":
        detail["functions"] *= 2
    elif fault == "wrong_theorem":
        detail["theorems"] = ["invented"]
    elif fault == "nonstandard_axiom":
        detail["axioms"] = ["invented"]
    with pytest.raises(ValueError, match="complete functions/theorems/strict refinement chains"):
        _tensor_detail(detail, [], {})


def test_complete_function_roster_retains_other_entries_and_excludes_declarations():
    def function(name, defined):
        return {
            "name": "func.func",
            "attrs": {"sym_name": {"value": name}},
            "props": {},
            "regions": [{"blocks": [{}] if defined else []}],
        }

    value = {"module": {"ops": [function("forward", True), function("other", True), function("external", False)]}}
    assert _functions(value) == ["forward", "other"]
    value["module"]["ops"].append(function("forward", True))
    with pytest.raises(ValueError):
        _functions(value)


@pytest.mark.parametrize("fault", ["duplicate_job", "unchanged_credit", "identity_proof", "aggregate", "pending"])
def test_journal_metadata_corruption_refuses_before_reading_any_proof_files(tmp_path, fault):
    from merlin.verify import pass_validation

    row = {
        "pass": "lower_quant_ext",
        "status": "verified",
        "proof_credit": True,
        "job": "job-control",
        "source_sha256": "a" * 64,
        "final_sha256": "b" * 64,
    }
    coverage = {
        "finalized": True,
        "compiler_completed": True,
        "requested_passes": sorted(pass_validation.CHECKS),
        "boundaries": [row],
        "verified_boundaries": 1,
        "unverified_boundaries": 0,
        "unchanged_boundaries": 0,
    }
    if fault == "duplicate_job":
        coverage["boundaries"].append(dict(row))
        coverage["verified_boundaries"] = 2
    elif fault == "unchanged_credit":
        row.update(status="unchanged", final_sha256=row["source_sha256"])
    elif fault == "identity_proof":
        row["final_sha256"] = row["source_sha256"]
    elif fault == "aggregate":
        coverage["verified_boundaries"] = 2
    else:
        row["status"] = "pending"
    with pytest.raises(ValueError):
        audit_coverage(coverage, [{"verdict": "accepted", "job": "job-control"}], tmp_path, None)
