"""Versioned original reference selections are source inputs, never authority."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase0 import declared_run as D
from merlin_experiments.phase0 import original_reference_flow as F
from merlin_experiments.phase0.component_source_performance import SCHEMA as PERFORMANCE_SCHEMA


def _fixtures(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("declared_reference_flow_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures("test_declared_phase0_run.py")
declared = fixtures.declared


def _request(declared, tmp_path):
    request, path = declared
    request = copy.deepcopy(request)
    request.update(schema=D.REFERENCE_SCHEMA, release_purpose="performance_campaign")
    request["source_performance"] = {
        "schema": PERFORMANCE_SCHEMA,
        "objectives": F.pin(path),
        "sweeps": F.pin(path),
    }
    policies = _fixtures("original_reference_fixtures.py").selection(
        type("Identity", (), {"sha256": "a" * 64})(),
        type("Basis", (), {"source": type("Identity", (), {"sha256": "b" * 64})()})(),
    )
    policies.pop("operator_schema_intake_sha256")
    policies.pop("semantic_basis_sha256")
    policies.update(schema=F.REFERENCE_SELECTION, native_observations="batch.v1")
    policies["cohorts"] = {"functional_guard": [1, 2], "withheld_transfer": [3]}
    reference = tmp_path / "reference-selection.json"
    reference.write_text(json.dumps(policies))
    checkout = tmp_path / "capture"
    (checkout / "m2m").mkdir(parents=True)
    (checkout / "m2m/__init__.py").write_text("# independently owned source inventory\n")
    for arguments in (
        ["init", "--quiet"],
        ["add", "m2m"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "Source"],
    ):
        subprocess.run(["git", "-C", str(checkout), *arguments], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    standard = tmp_path / "standard-selection.json"
    standard.write_text(
        json.dumps(
            {
                "schema": F.STANDARD_SELECTION,
                "capture_checkout": str(checkout),
                "capture_commit": commit,
                "mlir_opt": F.pin(path),
                "budget": {
                    "max_members": 100,
                    "max_source_bytes": 10000,
                    "max_total_source_bytes": 10000000,
                    "max_observation_bytes": 100000,
                    "max_nesting": 64,
                    "max_integer_bits": 512,
                    "max_dense_elements": 1000,
                    "max_dense_payload_bytes": 8000,
                    "timeout_s": 180,
                },
                "execution_budget": policies["execution_budget"],
            }
        )
    )
    request["original_references"] = {"reference": F.pin(reference), "standard_ir": F.pin(standard)}
    return request, reference, standard


def test_new_request_selects_exact_fixed_sources_without_reinterpreting_old_versions(declared, tmp_path):
    request, _, _ = _request(declared, tmp_path)
    assert D.validate(request) is request
    for schema in (D.SCHEMA, D.BRIDGE_SCHEMA, D.REQUIREMENT_SCHEMA, D.PERFORMANCE_SCHEMA):
        request["schema"] = schema
        with pytest.raises(ValueError):
            D.validate(request)


@pytest.mark.parametrize("change", ["missing", "saved_status", "factory", "bare_path", "other_purpose"])
def test_versioned_request_cannot_select_saved_authority_or_unbounded_factory(declared, tmp_path, change):
    request, _, _ = _request(declared, tmp_path)
    if change == "missing":
        del request["original_references"]["standard_ir"]
    elif change == "bare_path":
        request["original_references"]["reference"] = request["original_references"]["reference"]["path"]
    elif change == "other_purpose":
        request["release_purpose"] = "source_preparation"
    else:
        request["original_references"][change] = "accepted"
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize("change", ["schema", "saved_identity", "missing_cohort", "swapped_cohorts", "budget"])
def test_protected_reference_contract_is_closed_and_requires_all_original_cohorts(declared, tmp_path, change):
    request, reference, _ = _request(declared, tmp_path)
    value = json.loads(reference.read_bytes())
    if change == "schema":
        value["schema"] = "merlin.original_reference_selection.v1"
    elif change == "saved_identity":
        value["operator_schema_intake_sha256"] = "a" * 64
    elif change == "missing_cohort":
        value["cohorts"]["functional_guard"] = [1]
    elif change == "swapped_cohorts":
        value["cohorts"]["functional_guard"] = [2, 1]
    else:
        del value["execution_budget"]
    reference.write_text(json.dumps(value))
    request["original_references"]["reference"] = F.pin(reference)
    with pytest.raises(ValueError):
        F.read_selection(request["original_references"], forbidden=())


@pytest.mark.parametrize("change", ["changed", "forbidden", "alias", "saved_roster", "parser_bytes"])
def test_selected_files_and_parser_are_reopened_before_live_issuance(declared, tmp_path, change):
    request, reference, standard = _request(declared, tmp_path)
    forbidden = ()
    if change == "changed":
        reference.write_text("replacement")
    elif change == "forbidden":
        forbidden = (reference,)
    elif change == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(reference)
        request["original_references"]["reference"]["path"] = str(alias)
    else:
        value = json.loads(standard.read_bytes())
        if change == "saved_roster":
            value["reference_roster_sha256"] = "a" * 64
        else:
            value["mlir_opt"]["sha256"] = "a" * 64
        standard.write_text(json.dumps(value))
        request["original_references"]["standard_ir"] = F.pin(standard)
    with pytest.raises(ValueError):
        F.read_selection(request["original_references"], forbidden=forbidden)


def test_valid_inputs_are_explicit_data_and_saved_objects_cannot_prepare(declared, tmp_path):
    request, _, _ = _request(declared, tmp_path)
    selected = F.read_selection(request["original_references"], forbidden=())
    selected.verify()
    with pytest.raises(ValueError, match="live"):
        F.prepare(selected, schema_intake={}, semantic_basis=None, destination=tmp_path / "native")
    assert not (tmp_path / "native").exists()
