"""Ordinary original schema→factory→full reference→standard source controls."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_reference_flow as F
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_products as V
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S

from merlin.common import invocation_record as I
from merlin.targetgen.frontend_original_call import call_contracts

_spec = importlib.util.spec_from_file_location(
    "original_transpose_fixtures", Path(__file__).with_name("original_transpose_fixtures.py")
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
live_transpose_originals = G.live_transpose_originals


def standard():
    if not all(os.environ.get(name) for name in ("MERLIN_TEST_MLIR_OPT", "MERLIN_TEST_M2M_COMMIT")):
        pytest.skip("transpose ordinary standard sources need their explicit clean upstream/parser selections")
    return {
        "schema": F.TRANSPOSE_STANDARD_SELECTION,
        "capture_checkout": os.environ["MERLIN_TEST_M2M_ROOT"],
        "capture_commit": os.environ["MERLIN_TEST_M2M_COMMIT"],
        "mlir_opt": F.pin(os.environ["MERLIN_TEST_MLIR_OPT"]),
        "budget": {
            "max_members": 30,
            "max_source_bytes": 200000,
            "max_total_source_bytes": 20000000,
            "max_observation_bytes": 2000000,
            "max_nesting": 64,
            "max_integer_bits": 512,
            "max_dense_elements": 1000,
            "max_dense_payload_bytes": 8000,
            "timeout_s": 180,
        },
        "execution_budget": G.F.selection(
            type("Identity", (), {"sha256": "a" * 64})(),
            type("Basis", (), {"source": type("Identity", (), {"sha256": "b" * 64})()})(),
        )["execution_budget"],
    }


@pytest.fixture(scope="module")
def observed(live_transpose_originals, tmp_path_factory):
    intake, basis, spec = live_transpose_originals
    owner = tmp_path_factory.mktemp("ordinary-transpose-reference-flow")
    selection = G.selection(intake, basis)
    selection.pop("operator_schema_intake_sha256")
    selection.pop("semantic_basis_sha256")
    selection["schema"] = F.TRANSPOSE_REFERENCE_SELECTION
    reference, upstream = owner / "reference.json", owner / "standard.json"
    G.write(reference, selection)
    G.write(upstream, standard())
    selected = F.read_selection({"reference": F.pin(reference), "standard_ir": F.pin(upstream)}, forbidden=())
    before = spec.read_bytes()
    result = F.prepare(selected, schema_intake=intake, semantic_basis=basis, destination=owner / "products")
    assert spec.read_bytes() == before
    return result


def test_all_original_guard_and_private_slots_preserve_exact_alias_axes_and_unknowns(observed):
    reference = json.loads(observed.references.receipt_json)
    record = json.loads(observed.receipt_json)
    assert reference["schema"] == R.TRANSPOSE_SCHEMA and record["schema"] == S.TRANSPOSE_SCHEMA
    assert len(reference["members"]) == len(record["members"]) == 21
    checked = [row for row in reference["members"] if row["state"] == "reference_checked"]
    assert len(checked) == 18, [(row["state"], row.get("reason")) for row in reference["members"]]
    assert sum(row["state"] == "source_reference_ir_checked" for row in record["members"]) == 18, record["members"]
    assert sum(row["state"] == "unavailable" for row in reference["members"]) == 3
    originals = {(row["original_member_id"], row["node"]) for row in reference["members"]}
    assert len(originals) == 7
    assert {row["form"]["operand_dtypes"][0] for row in checked} == {"float32", "int8", "int16", "int32", "int64"}
    for identity in originals:
        assert [
            (row["cohort"], row["extent"])
            for row in reference["members"]
            if (row["original_member_id"], row["node"]) == identity
        ] == list(C.required_source_cohorts())
    for row in checked:
        assert row["form"]["operand_dtypes"] == row["form"]["result_dtypes"]
        assert row["call"]["arguments"][0]["alias"] == row["call"]["schema_returns"][0]["alias"]
        assert row["form"]["parameters"] == {arg["name"]: arg["value"]["value"] for arg in row["call"]["arguments"][1:]}
        metadata = json.loads(Path(row["products"]["metadata"]["path"]).read_bytes())
        assert metadata["inputs"][0]["shape"] == [row["extent"], row["extent"] + 1]
        comparison = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
        assert comparison["passed"] and comparison["checked_elements"] == row["extent"] * (row["extent"] + 1)
        alias = comparison["finite_alias_observation"]
        assert alias["same_storage_base"] is True
        assert alias["input_bytes_sha256_before"] == alias["input_bytes_sha256_after"]
        assert set(R._UNKNOWN) <= set(row["required_unknowns"])
        assert I.require_environment(Path(row["invocation"]["path"]), environment=R.D.ENVIRONMENT)["returncode"] == 0
    for row in record["members"]:
        assert set(R._UNKNOWN) | set(S._UNKNOWN) <= set(row["required_unknowns"])
    assert F.summary(observed)["original_numerical_admissions"] == 0
    assert F.summary(observed)["release_authority"] == "not_issued"


def test_actual_schema_binding_reaches_new_automatic_source_factory_without_admission(observed):
    references = observed.references
    record = json.loads(references.receipt_json)
    schema = references.schema_intake.record()
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    requested, members = 21, []
    for defaults in record["defaults"]:
        trace, schemas, original_defaults = R.D.verify_member(
            defaults, schema_record=schema, version=2, transport="batch.v1"
        )
        calls = call_contracts(trace, schemas, original_defaults)
        legacy = C._forms(trace, schemas, original_defaults, numerical_semantics=None, version=3)
        assert not legacy
        forms = C._forms(trace, schemas, original_defaults, numerical_semantics=None, version=4)
        members += C._sources(
            calls,
            forms,
            budget=G.selection(references.schema_intake, references.basis)["source_budget"],
            total=total,
            requested=requested,
            version=4,
        )
    assert len(members) == 21
    assert sum(row["status"] == "source_constructed" for row, _ in members) == 18
    assert sum(row["status"] == "unknown" for row, _ in members) == 3
    assert A._original_source_version({"schema": A.TRANSPOSE_POLICY_SCHEMA}) == 4
    assert A._original_source_version({"schema": A.POINTWISE_POLICY_SCHEMA}) == 3
    assert "merlin.targetgen.original_transpose_sources" in C.reader_modules(4)
    assert "merlin.targetgen.original_transpose_sources" not in C.reader_modules(3)


@pytest.mark.parametrize("defect", ["alias_bool", "alias_float", "stride", "offset", "input_write", "missing"])
def test_value_equality_cannot_supply_a_missing_or_changed_native_alias_observation(observed, tmp_path, defect):
    record = json.loads(observed.references.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "reference_checked")
    contract = S._contract(row, observed.references)
    inputs = R._tensors(json.loads(Path(row["products"]["inputs"]["path"]).read_bytes()))
    value = json.loads(Path(row["products"]["actual"]["path"]).read_bytes())
    if defect == "missing":
        value.pop("alias_observation")
    else:
        alias = value["alias_observation"]
        if defect == "alias_bool":
            alias["same_storage_base"] = 1
        elif defect == "alias_float":
            alias["input_index"] = 0.0
        elif defect == "stride":
            alias["output_strides"] = alias["input_strides"]
        elif defect == "offset":
            alias["output_storage_offset"] = 1
        else:
            alias["input_bytes_sha256_after"] = "a" * 64
    path = tmp_path / "changed-native.json"
    G.write(path, value)
    if defect != "missing":
        assert contract.compare(inputs, V.native_outputs(contract, path))["passed"]
    with pytest.raises(ValueError):
        V.finite_transpose_alias(contract, inputs, path)


def test_legacy_policy_and_declared_versions_never_acquire_new_transpose_contract(observed):
    row = next(
        row for row in json.loads(observed.references.receipt_json)["members"] if row["state"] == "reference_checked"
    )
    for schema in (P.SCHEMA, P.BATCH_SCHEMA, P.POINTWISE_SCHEMA):
        selected = G.selection(observed.references.schema_intake, observed.references.basis)
        selected["schema"] = schema
        if schema == P.SCHEMA:
            selected.pop("native_observations")
        with pytest.raises(ValueError):
            P.validate(selected)
    for schema in (F.REFERENCE_SELECTION, F.POINTWISE_REFERENCE_SELECTION):
        selected = G.selection(observed.references.schema_intake, observed.references.basis)
        selected.pop("operator_schema_intake_sha256")
        selected.pop("semantic_basis_sha256")
        selected["schema"] = schema
        with pytest.raises(ValueError):
            F._reference(selected, "a" * 64)
    selected = P.policy(row["policy"], transpose=True)
    assert selected.record() == row["policy"]
    with pytest.raises(ValueError):
        P.policy(row["policy"], transpose=1)


def test_new_complete_roster_replay_refuses_lost_private_slot_before_native_replay(observed):
    references = observed.references
    record = json.loads(references.receipt_json)
    record["members"] = [row for row in record["members"] if row["cohort"] != "withheld_transfer"]
    with pytest.raises(ValueError, match="required original slots"):
        R.verify(record, schema_intake=references.schema_intake, basis=references.basis, selection=references.selection)


def test_alias_measurement_work_is_in_complete_source_budget(observed):
    references = observed.references
    row = next(row for row in json.loads(references.receipt_json)["members"] if row["state"] == "reference_checked")
    roles = {allocation["role"] for allocation in row["cost"]["allocations"]}
    assert {"native_alias_input_bytes_before", "native_alias_input_bytes_after", "native_alias_geometry"} <= roles
    selected = copy.deepcopy(json.loads(references.selection.read_bytes()))
    selected["source_budget"]["max_sources"] = 1
    rows, contracts, _ = R._drafts(
        json.loads(references.receipt_json)["defaults"],
        schema=references.schema_intake.record(),
        basis=references.basis,
        selection=selected,
    )
    assert len(rows) == 21 and not contracts and all(row["state"] == "unavailable" for row in rows)
