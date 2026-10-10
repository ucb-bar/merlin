"""Fresh ordinary schema→broadcast source→full reference→standard-IR controls.

The five-call independent fixture is separate from the protected corpus. Its
fifteen requested source slots establish no mandatory numerical/domain coverage.
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_reference_flow as F
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S

from merlin.common import invocation_record as I
from merlin.targetgen.frontend_original_call import call_contracts

_spec = importlib.util.spec_from_file_location(
    "original_broadcast_fixtures", Path(__file__).with_name("original_broadcast_add_fixtures.py")
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
live_broadcast_originals = G.live_broadcast_originals


def standard():
    if not all(os.environ.get(name) for name in ("MERLIN_TEST_MLIR_OPT", "MERLIN_TEST_M2M_COMMIT")):
        pytest.skip("original broadcast standard sources need explicit clean upstream/parser selections")
    return {
        "schema": F.BROADCAST_STANDARD_SELECTION,
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
def observed(live_broadcast_originals, tmp_path_factory):
    intake, basis, spec = live_broadcast_originals
    owner = tmp_path_factory.mktemp("ordinary-broadcast-reference-flow")
    selection = G.selection(intake, basis)
    selection.pop("operator_schema_intake_sha256")
    selection.pop("semantic_basis_sha256")
    selection["schema"] = F.BROADCAST_REFERENCE_SELECTION
    reference, upstream = owner / "reference.json", owner / "standard.json"
    G.write(reference, selection)
    G.write(upstream, standard())
    selected = F.read_selection({"reference": F.pin(reference), "standard_ir": F.pin(upstream)}, forbidden=())
    before = spec.read_bytes()
    result = F.prepare(selected, schema_intake=intake, semantic_basis=basis, destination=owner / "products")
    assert spec.read_bytes() == before
    return result


def test_every_original_fixture_slot_preserves_storage_broadcast_and_complete_native_outputs(observed):
    reference = json.loads(observed.references.receipt_json)
    record = json.loads(observed.receipt_json)
    assert reference["schema"] == R.BROADCAST_SCHEMA and record["schema"] == S.BROADCAST_SCHEMA
    assert len(reference["members"]) == len(record["members"]) == 15
    checked = [row for row in reference["members"] if row["state"] == "reference_checked"]
    assert len(checked) == 9, [(row["state"], row.get("reason")) for row in reference["members"]]
    assert sum(row["state"] == "source_reference_ir_checked" for row in record["members"]) == 9, record["members"]
    unknown = [row for row in reference["members"] if row["state"] == "unavailable"]
    assert len(unknown) == 6 and {row["original_member_id"] for row in unknown} == {"unsupported_f16", "nonunit"}
    for identity in {row["original_member_id"] for row in reference["members"]}:
        assert [
            (row["cohort"], row["extent"]) for row in reference["members"] if row["original_member_id"] == identity
        ] == list(C.required_source_cohorts())
    for row in checked:
        contract = S._contract(row, observed.references)
        metadata = contract.verify()
        assert row["form"]["operand_dtypes"] == [row["form"]["result_dtypes"][0]] * 2
        assert metadata["parameters"]["alpha"] == 1 and metadata["parameters"]["broadcasting"] == "right_aligned"
        comparison = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
        assert comparison["passed"] and comparison["checked_elements"] == math.prod(metadata["outputs"][0]["shape"])
        assert I.require_environment(Path(row["invocation"]["path"]), environment=R.D.ENVIRONMENT)["returncode"] == 0
        assert set(R._UNKNOWN) <= set(row["required_unknowns"])
    for row in record["members"]:
        assert set(R._UNKNOWN) | set(S._UNKNOWN) <= set(row["required_unknowns"])
    assert F.summary(observed)["original_numerical_admissions"] == 0


def test_actual_original_schema_reaches_only_the_new_automatic_broadcast_factory(observed):
    references = observed.references
    record, schema = json.loads(references.receipt_json), references.schema_intake.record()
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    members = []
    for defaults in record["defaults"]:
        trace, schemas, original_defaults = R.D.verify_member(
            defaults, schema_record=schema, version=2, transport="batch.v1"
        )
        calls = call_contracts(trace, schemas, original_defaults)
        forms = C._forms(trace, schemas, original_defaults, numerical_semantics=None, version=5)
        members += C._sources(
            calls,
            forms,
            budget=G.selection(references.schema_intake, references.basis)["source_budget"],
            total=total,
            requested=15,
            version=5,
        )
        legacy = C._forms(trace, schemas, original_defaults, numerical_semantics=None, version=4)
        if calls[0]["node"] == forms[0]["node"] and forms[0]["status"] == "supported":
            assert legacy[0]["status"] == "unknown"
    assert len(members) == 15 and sum(row["status"] == "source_constructed" for row, _ in members) == 12
    assert sum(row["status"] == "unknown" for row, _ in members) == 3
    assert A._original_source_version({"schema": A.BROADCAST_POLICY_SCHEMA}) == 5
    assert A._original_source_version({"schema": A.TRANSPOSE_POLICY_SCHEMA}) == 4


@pytest.mark.parametrize("defect", ["missing_private", "policy", "broadcast", "type"])
def test_re_signed_rows_cannot_change_original_membership_policy_or_broadcast_relations(observed, defect):
    record = json.loads(observed.references.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "reference_checked")
    if defect == "missing_private":
        record["members"] = [row for row in record["members"] if row["cohort"] != "withheld_transfer"]
    elif defect == "policy":
        row["policy"]["atol"] = 1.0
    elif defect == "broadcast":
        row["form"]["parameters"]["operand_axes"][1][-1] = "singleton"
    else:
        row["form"]["arguments"][2]["value"]["value"] = True
    with pytest.raises(ValueError):
        R.verify(
            record,
            schema_intake=observed.references.schema_intake,
            basis=observed.references.basis,
            selection=observed.references.selection,
        )


def test_missing_policy_retains_every_original_requested_slot(observed, tmp_path):
    references = observed.references
    selected = G.selection(references.schema_intake, references.basis)
    selected["policies"] = []
    selected = P.validate(selected)
    rows, contracts, _ = R._drafts(
        json.loads(references.receipt_json)["defaults"],
        schema=references.schema_intake.record(),
        basis=references.basis,
        selection=selected,
    )
    assert len(rows) == 15 and contracts == {} and all(row["state"] == "unavailable" for row in rows)
