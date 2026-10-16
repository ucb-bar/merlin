"""Pure original declaration controls never grant native or mandatory coverage."""

import copy
import json
from types import SimpleNamespace

import pytest
import test_declared_phase0_run as DECLARED
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import declared_run as D
from merlin_experiments.phase0 import original_call_sources as C
from test_original_integer_scalar_binary_plan import independent_sources
from test_original_scalar_binary_plan import budget

declared = DECLARED.declared


def request_for(declared):
    from test_packing_memory_intake import selection

    request, _ = declared
    request = copy.deepcopy(request)
    request.update(
        schema=D.INTEGER_SCALAR_SCHEMA,
        packing=selection(),
        release_purpose="performance_campaign",
        source_performance={
            "schema": "merlin.phase0.source_performance_preparation.v1",
            "objectives": request["inputs"]["descriptor"],
            "sweeps": request["inputs"]["descriptor"],
        },
        original_references={
            "reference": request["inputs"]["descriptor"],
            "standard_ir": request["inputs"]["descriptor"],
        },
        original_scalar_conversion=request["inputs"]["descriptor"],
    )
    request["automatic"]["schema"] = A.INTEGER_SCALAR_POLICY_SCHEMA
    request["operator_schemas"].update(
        schema=D.S.TENSOR_SELECTION_SCHEMA,
        tensor_arguments={"compiler": request["inputs"]["descriptor"]},
    )
    return request


def test_explicit_v14_selects_v7_without_widening_any_prior_automatic_policy(declared):
    request = request_for(declared)
    policy = {
        **request["automatic"],
        "hardware": {},
        **dict.fromkeys(
            (
                "software_spec_sha256",
                "numerical_semantics_sha256",
                "semantic_basis_sha256",
                "operator_schema_intake_sha256",
                "arithmetic_intake_sha256",
                "packing_intake_sha256",
            ),
            "a" * 64,
        ),
        "original_source_budget": budget(),
    }
    for version, schema in enumerate(
        (
            A.ORIGINAL_POLICY_SCHEMA,
            A.LINEAR_POLICY_SCHEMA,
            A.POINTWISE_POLICY_SCHEMA,
            A.TRANSPOSE_POLICY_SCHEMA,
            A.BROADCAST_POLICY_SCHEMA,
            A.SCALAR_BINARY_POLICY_SCHEMA,
            A.INTEGER_SCALAR_POLICY_SCHEMA,
        ),
        start=1,
    ):
        selected = {**policy, "schema": schema}
        assert A._closed_policy(selected) == selected
        assert A._original_source_version(selected) == version
        assert A._selected_effects(selected) and A._selected_arithmetic(selected)
    assert D.validate(request) is request


@pytest.mark.parametrize("zero", [False, True])
def test_new_caller_requires_same_original_tensor_facet_and_retains_zero_result_selection(declared, zero):
    request = request_for(declared)
    if zero:
        request["operator_schemas"]["schema"] = D.S.ZERO_SELECTION_SCHEMA
        request["operator_schemas"]["zero_returns"] = copy.deepcopy(request["operator_schemas"]["tensor_arguments"])
    assert D.validate(request) is request


@pytest.mark.parametrize(
    "change",
    ["old_request", "old_automatic", "old_schema", "missing_converter", "saved_owner", "bare_path"],
)
def test_old_versions_or_missing_original_native_inputs_cannot_acquire_v7(declared, change):
    request = request_for(declared)
    if change == "old_request":
        request["schema"] = D.PACKING_SCHEMA
    elif change == "old_automatic":
        request["automatic"]["schema"] = A.SCALAR_BINARY_POLICY_SCHEMA
    elif change == "old_schema":
        request["operator_schemas"]["schema"] = D.S.SELECTION_SCHEMA
        del request["operator_schemas"]["tensor_arguments"]
    elif change == "missing_converter":
        del request["original_scalar_conversion"]
    elif change == "saved_owner":
        request["original_scalar_conversion"]["source_record_sha256"] = "a" * 64
    elif change == "bare_path":
        request["original_scalar_conversion"] = request["original_scalar_conversion"]["path"]
    else:
        raise AssertionError(change)
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize("native_schema", [D.S.TENSOR_SCHEMA, D.S.ZERO_SCHEMA])
def test_same_exact_tensor_binding_supports_original_v7_on_both_declared_intake_versions(
    monkeypatch, tmp_path, native_schema
):
    record, basis, policy, schema = independent_sources(monkeypatch, tmp_path)
    schema["schema"] = native_schema
    assert C.verify(record, schema_record=schema, basis=basis, numerical_semantics=policy) == record
    missing = C.required_unknowns(record, basis=basis, unknown=A.P._unknown)
    assert len([row for row in missing if row["kind"] == "original_operator_admission"]) == 3
    assert len([row for row in missing if row["kind"] == "original_operator_factory"]) == 3
    assert sum(len(row["source_members"]) for row in record["members"]) == 9


@pytest.mark.parametrize("change", ["binding", "native_false", "missing_graph", "v1"])
def test_v3_tensor_facet_drift_never_grants_integer_source_slots(monkeypatch, tmp_path, change):
    record, basis, policy, schema = independent_sources(monkeypatch, tmp_path)
    schema["schema"] = D.S.ZERO_SCHEMA
    if change == "binding":
        schema["members"][1]["tensor_bindings"] = []
    elif change == "native_false":
        schema["members"][1]["tensor_bindings"][0]["native"]["wrapped_number"] = False
    elif change == "missing_graph":
        schema["members"].pop(1)
    else:
        schema["schema"] = D.S.SCHEMA
    with pytest.raises(ValueError):
        C.verify(record, schema_record=schema, basis=basis, numerical_semantics=policy)


def test_connected_caller_passes_exact_generated_original_sources_to_registered_conversion(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import original_scalar_conversion_flow as F

    generated = {"automatic_derivation": {"original_call_sources": {"schema": C.INTEGER_SCALAR_SCHEMA}}}
    schema, basis, selected = object(), object(), object()
    calls = []
    monkeypatch.setattr(F, "prepare", lambda *args, **kwargs: calls.append((args, kwargs)) or "actual-owner")
    monkeypatch.setattr(F, "summary", lambda owner: {"actual": owner, "numerical_or_mandatory_admission": False})
    # The ordinary helper is the same call used after actual generate_target.
    owner, report = D._issue_scalar_conversion(
        selected,
        schemas=schema,
        basis=basis,
        coverage=generated,
        numerical_semantics={"bounded_exact_original": True},
        output=tmp_path,
    )
    assert owner == "actual-owner" and report["actual"] == owner
    assert report["numerical_or_mandatory_admission"] is False
    assert calls == [
        (
            (selected,),
            {
                "schema_intake": schema,
                "basis": basis,
                "source_record": generated["automatic_derivation"]["original_call_sources"],
                "numerical_semantics": {"bounded_exact_original": True},
                "destination": tmp_path / "original-scalar-conversion",
            },
        )
    ]


@pytest.mark.parametrize("references,packing", [(False, False), (True, False), (False, True), (True, True)])
def test_v7_source_construction_does_not_require_a_new_reference_or_memory_policy(declared, references, packing):
    request = request_for(declared)
    if not references:
        del request["original_references"]
    if not packing:
        del request["packing"]
    assert D.validate(request) is request
    assert D._has_original_references(request) is references
    assert D._has_memory_packing(request) is packing
    # Existing v6 still requires both original fields and cannot inherit new absence.
    request.pop("original_scalar_conversion")
    request["schema"] = D.PACKING_SCHEMA
    request["automatic"]["schema"] = A.SCALAR_BINARY_POLICY_SCHEMA
    if references and packing:
        assert D.validate(request) is request
    else:
        with pytest.raises(ValueError):
            D.validate(request)


def test_ordinary_v7_generation_keeps_complete_original_missing_selectors_and_blocked_phases(
    declared, tmp_path, monkeypatch
):
    """Run the real caller with explicit native seams; no capability is issued."""
    from merlin_experiments.phase0 import original_reference_flow as R
    from merlin_experiments.phase0 import original_scalar_conversion_flow as F
    from merlin_experiments.phase0 import source_requirement_ledger as L
    from test_declared_original_scalar_conversion_flow import inputs

    original, basis, policy, _ = independent_sources(monkeypatch, tmp_path)
    historical = [
        A.P._unknown(kind, {"original": "independent-source"}, "missing independent source premise")
        for kind in ("packing_mapping", "numeric_domain", "physical_interaction", "effect_domain")
    ]
    missing = C.merge_unknowns(historical, original, basis=basis, unknown=A.P._unknown)
    assert {row["id"] for row in historical} <= {row["id"] for row in missing}
    assert sum(row["kind"] == "original_operator_admission" for row in missing) == 3
    request = request_for(declared)
    del request["original_references"], request["packing"]
    pin, _, _ = inputs(declared, tmp_path)
    request["original_scalar_conversion"] = pin
    request_path = tmp_path / "connected-request.json"
    request_path.write_text(json.dumps(request))
    hardware = SimpleNamespace(public_facts=lambda: {"unknowns": ["physical-mapping-pending"]})
    software = SimpleNamespace(
        sha256="1" * 64,
        receipt_json=json.dumps({"semantic_basis_sha256": request["inputs"]["semantic_basis"]["sha256"]}),
        public_facts=lambda: {"numerical_semantics": policy},
    )
    schemas = SimpleNamespace(sha256="2" * 64, record=lambda: {"unknowns": ["sdk-build-correspondence-pending"]})
    monkeypatch.setattr(D, "issue_independent_hardware_intake", lambda **kwargs: hardware)
    monkeypatch.setattr(D, "issue_independent_software_intake", lambda **kwargs: software)
    monkeypatch.setattr(D.S, "issue_independent_operator_schema_intake", lambda **kwargs: schemas)
    monkeypatch.setattr(D, "issue_independent_arithmetic_intake", lambda **kwargs: SimpleNamespace(sha256="3" * 64))
    packing_calls = []
    monkeypatch.setattr(
        D,
        "issue_independent_packing_intake",
        lambda **kwargs: packing_calls.append(kwargs) or SimpleNamespace(sha256="4" * 64),
    )
    monkeypatch.setattr(
        D,
        "select_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            derivation_identity={"contract_sha256": "5" * 64, "raw_facts_sha256": "6" * 64}
        ),
    )
    monkeypatch.setattr(
        D,
        "_source_performance_inputs",
        lambda *args, **kwargs: ({"objectives": request_path, "sweeps": request_path}, [], {"sweeps": []}),
    )

    def generation(target, **kwargs):
        assert target == request["target"]
        assert json.loads(kwargs["component_coverage"].read_bytes())["schema"] == A.INTEGER_SCALAR_POLICY_SCHEMA
        assert kwargs["operator_schema_intake"] is schemas
        root = kwargs["output_root"]
        coverage = {
            "status": "source_prepared_incomplete",
            "automatic_derivation": {
                "original_call_sources": original,
                "required_unknowns": missing,
            },
            "obligations": [{**row, "mandatory": True, "state": "unavailable"} for row in missing],
        }
        path = root / "_evidence/coverage"
        path.mkdir(parents=True)
        (root / "MANIFEST.yaml").write_text("generated: []\n")
        (path / "component-coverage.json").write_text(json.dumps(coverage))
        (path / "generation.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.phase0_generation.v1",
                    "mode": "diagnostic",
                    "qualification": "not_established",
                    "failures": [
                        {
                            "capsule": "component coverage",
                            "reason": "mandatory independent obligations unavailable; inspect private coverage report",
                        }
                    ],
                    "corpus_manifest": str(root / "MANIFEST.yaml"),
                    "capsule_commitments": [],
                }
            )
        )
        raise RuntimeError("mandatory independent obligations unavailable")

    monkeypatch.setattr(D, "generate_target", generation)
    monkeypatch.setattr(
        D,
        "_verify_source_performance_products",
        lambda *args, **kwargs: (0, {"source_checked_counts": {}, "requested_members": []}, request_path),
    )
    ledger_joins = []
    monkeypatch.setattr(
        L, "prepare_requirement_ledger", lambda **kwargs: pytest.fail("v7 selected failure-only ledger")
    )
    monkeypatch.setattr(
        L,
        "prepare_prerequisite_ledger",
        lambda **kwargs: ledger_joins.append(kwargs) or SimpleNamespace(record=lambda: {}),
    )
    monkeypatch.setattr(R, "prepare", lambda *args, **kwargs: pytest.fail("implicit numerical reference selected"))
    from merlin_experiments.phase0.component_semantic_basis import ComponentSemanticBasis

    monkeypatch.setattr(ComponentSemanticBasis, "from_recipe", lambda *args, **kwargs: basis)
    joined, replays = [], []
    owner = SimpleNamespace(record=lambda: replays.append("fresh registered owner replay"))
    monkeypatch.setattr(F, "prepare", lambda *args, **kwargs: joined.append(kwargs) or owner)
    monkeypatch.setattr(F, "summary", lambda actual: {"numerical_or_mandatory_admission": False})
    report = D.run(request_path, output=tmp_path / "connected-run")
    assert len(joined) == 1 and joined[0]["schema_intake"] is schemas
    assert len(ledger_joins) == 1 and ledger_joins[0]["schema_intake"] is schemas
    assert ledger_joins[0]["semantic_basis"] is basis
    assert ledger_joins[0]["coverage"]["automatic_derivation"]["original_call_sources"] == original
    assert joined[0]["source_record"] == original and joined[0]["numerical_semantics"] == policy
    assert "memory_selection" not in packing_calls[0]
    assert replays == ["fresh registered owner replay"]
    assert {row["id"] for row in report["required_unknowns"]} == {row["id"] for row in missing}
    assert report["mandatory_obligations"] == report["mandatory_unavailable"] == len(missing)
    assert report["original_calls"] == 3 and report["original_source_members"] == 9
    assert report["original_numerical_admissions"] == report["candidate_comparisons"] == 0
    assert report["coverage_status"] == "source_prepared_incomplete"
    assert report["phases"]["0"]["coverage_gate"] == "incomplete"
    assert report["phases"]["1"]["handoff_accepted"] is False
    assert "original_reference_preparation" not in report
