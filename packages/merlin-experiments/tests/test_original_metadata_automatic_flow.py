"""Explicit v15/v8 ordinary caller controls; native seams are diagnostic substitutes."""

import json
from types import SimpleNamespace

import pytest
import test_original_integer_automatic_flow as OLD
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import declared_run as D
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion as V
from merlin_experiments.phase0 import original_scalar_conversion_flow as F
from test_declared_original_scalar_conversion_flow import inputs as converter_inputs
from test_original_integer_scalar_binary_plan import conversion_budget
from test_original_integer_scalar_binary_plan import independent_sources as scalar_sources
from test_original_metadata_source_flow import independent_sources

declared = OLD.declared


def request_for(declared):
    request = OLD.request_for(declared)
    request.update(schema=D.METADATA_SCHEMA)
    request["automatic"]["schema"] = A.METADATA_POLICY_SCHEMA
    return request


@pytest.mark.parametrize("references,packing", [(False, False), (True, False), (False, True), (True, True)])
def test_v8_keeps_explicit_reference_memory_choices_and_mandatory_scalar_conversion(declared, references, packing):
    request = request_for(declared)
    if not references:
        del request["original_references"]
    if not packing:
        del request["packing"]
    assert D.validate(request) is request
    assert D._has_original_references(request) is references
    assert D._has_memory_packing(request) is packing
    request.pop("original_scalar_conversion")
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize("change", ["old_request", "old_policy", "old_schema", "saved_converter", "default_factory"])
def test_new_vocabularies_cannot_widen_prior_callers_or_accept_saved_authority(declared, change):
    request = request_for(declared)
    if change == "old_request":
        request["schema"] = D.INTEGER_SCALAR_SCHEMA
    elif change == "old_policy":
        request["automatic"]["schema"] = A.INTEGER_SCALAR_POLICY_SCHEMA
    elif change == "old_schema":
        request["operator_schemas"]["schema"] = D.S.SELECTION_SCHEMA
        del request["operator_schemas"]["tensor_arguments"]
    elif change == "saved_converter":
        request["original_scalar_conversion"]["status"] = "checked"
    else:
        request["automatic"]["factory"] = "implicit"
    with pytest.raises(ValueError):
        D.validate(request)


def test_automatic_policy_selects_only_the_new_source_version_and_keeps_complete_facets(declared):
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
    }
    assert A._closed_policy(policy) == policy and A._original_source_version(policy) == 8
    assert A._selected_effects(policy) and A._selected_arithmetic(policy)
    prior = {**policy, "schema": A.INTEGER_SCALAR_POLICY_SCHEMA}
    assert A._closed_policy(prior) == prior and A._original_source_version(prior) == 7


def test_full_v8_source_keeps_exact_old_scalar_subroster_request_and_payload_budgets(monkeypatch, tmp_path):
    source, basis, policy, schemas = scalar_sources(monkeypatch, tmp_path, version=8)
    assert source["schema"] == C.METADATA_SCHEMA
    assert C.verify(source, schema_record=schemas, basis=basis, numerical_semantics=policy) == source
    members, total = V.required_members(source, basis=basis, budget=conversion_budget())
    assert len(members) == 9 and sum(loader is not None for _, loader in members) == 6
    assert total["promotion_tensor_elements"] == 63 and total["reserved_product_bytes"] == 1900000
    getter = tmp_path / "getter"
    getter.write_bytes(b"unloaded diagnostic fixture")
    request = V._request(members, [], conversion_budget(), version=2, getter={"getter": str(getter)})
    assert request["schema"] == "merlin.original_scalar_conversion_request.v2"
    assert len(request["members"]) == 6  # unavailable original slots remain in the owner's complete members
    assert "original_numerical_policy_and_reference" in V._UNKNOWN


def test_new_declared_converter_pins_full_v8_record_and_old_selection_refuses_it(declared, tmp_path, monkeypatch):
    pin, path, raw = converter_inputs(declared, tmp_path)
    raw["schema"] = F.METADATA_SCHEMA
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        F.read_selection(F._pin(path), forbidden=())
    selected = F.read_selection(F._pin(path), forbidden=(), version=2)

    class Schema:
        sha256 = "a" * 64

    class Basis:
        source = SimpleNamespace(sha256="b" * 64)

    monkeypatch.setattr(V, "IndependentOperatorSchemaIntake", Schema)
    monkeypatch.setattr(V, "ComponentSemanticBasis", Basis)
    source = {"schema": C.METADATA_SCHEMA, "members": []}
    calls = []
    monkeypatch.setattr(V, "prepare", lambda **kwargs: calls.append(kwargs) or "diagnostic-owner")
    owner = F.prepare(
        selected,
        schema_intake=Schema(),
        basis=Basis(),
        source_record=source,
        numerical_semantics={"unchanged": "original policy"},
        destination=tmp_path / "fresh",
    )
    assert owner == "diagnostic-owner" and calls[0]["source_record"] is source
    actual = json.loads(calls[0]["selection"].read_bytes())
    assert actual["schema"] == V.METADATA_SELECTION_SCHEMA and actual["source_record_sha256"] == V._digest(source)
    actual["schema"] = V.INTEGER_SELECTION_SCHEMA
    with pytest.raises(ValueError, match="widen"):
        V.validate_selection(actual, source_record=source, schema_intake=Schema(), basis=Basis())


def test_actual_v8_declared_caller_keeps_complete_missing_rows_and_invokes_versioned_fixed_producers(
    declared, tmp_path, monkeypatch
):
    from merlin_experiments.phase0 import original_reference_flow as R
    from merlin_experiments.phase0 import source_requirement_ledger as L
    from merlin_experiments.phase0.component_semantic_basis import ComponentSemanticBasis

    original = independent_sources(monkeypatch, tmp_path / "original")
    source, basis, policy = (original[key] for key in ("source_record", "basis", "numerical"))
    historical = [
        A.P._unknown(kind, {"independent": "source"}, "missing")
        for kind in ("packing_mapping", "numeric_domain", "physical_interaction", "effect_domain")
    ]
    missing = C.merge_unknowns(historical, source, basis=basis, unknown=A.P._unknown)
    request = request_for(declared)
    del request["original_references"], request["packing"]
    _, converter, selection = converter_inputs(declared, tmp_path)
    selection["schema"] = F.METADATA_SCHEMA
    converter.write_text(json.dumps(selection))
    request["original_scalar_conversion"] = F._pin(converter)
    request_path = tmp_path / "connected-request.json"
    request_path.write_text(json.dumps(request))
    hardware = SimpleNamespace(public_facts=lambda: {"unknowns": ["hardware pending"]})
    software = SimpleNamespace(
        sha256="1" * 64,
        receipt_json=json.dumps({"semantic_basis_sha256": request["inputs"]["semantic_basis"]["sha256"]}),
        public_facts=lambda: {"numerical_semantics": policy},
    )
    schemas = SimpleNamespace(sha256="2" * 64, record=lambda: {"unknowns": ["native correspondence pending"]})
    monkeypatch.setattr(D, "issue_independent_hardware_intake", lambda **kwargs: hardware)
    monkeypatch.setattr(D, "issue_independent_software_intake", lambda **kwargs: software)
    monkeypatch.setattr(D.S, "issue_independent_operator_schema_intake", lambda **kwargs: schemas)
    monkeypatch.setattr(D, "issue_independent_arithmetic_intake", lambda **kwargs: SimpleNamespace(sha256="3" * 64))
    monkeypatch.setattr(D, "issue_independent_packing_intake", lambda **kwargs: SimpleNamespace(sha256="4" * 64))
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

    def generate(target, **kwargs):
        assert json.loads(kwargs["component_coverage"].read_bytes())["schema"] == A.METADATA_POLICY_SCHEMA
        assert kwargs["operator_schema_intake"] is schemas
        root = kwargs["output_root"]
        path = root / "_evidence/coverage"
        path.mkdir(parents=True)
        (root / "MANIFEST.yaml").write_text("generated: []\n")
        coverage = {
            "status": "source_prepared_incomplete",
            "automatic_derivation": {"original_call_sources": source, "required_unknowns": missing},
            "obligations": [{**row, "mandatory": True, "state": "unavailable"} for row in missing],
        }
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

    monkeypatch.setattr(D, "generate_target", generate)
    monkeypatch.setattr(
        D,
        "_verify_source_performance_products",
        lambda *args, **kwargs: (0, {"source_checked_counts": {}, "requested_members": []}, request_path),
    )
    ledger_calls, converter_calls, replays = [], [], []
    monkeypatch.setattr(L, "prepare_requirement_ledger", lambda **kwargs: pytest.fail("old ledger"))
    monkeypatch.setattr(L, "prepare_prerequisite_ledger", lambda **kwargs: pytest.fail("v7 ledger"))
    monkeypatch.setattr(
        L,
        "prepare_metadata_prerequisite_ledger",
        lambda **kwargs: ledger_calls.append(kwargs) or SimpleNamespace(record=lambda: {}),
    )
    monkeypatch.setattr(R, "prepare", lambda *args, **kwargs: pytest.fail("implicit reference"))
    monkeypatch.setattr(ComponentSemanticBasis, "from_recipe", lambda *args, **kwargs: basis)
    monkeypatch.setattr(
        F,
        "prepare",
        lambda selected, **kwargs: (
            converter_calls.append((selected, kwargs))
            or SimpleNamespace(record=lambda: replays.append("actual selected owner replay"))
        ),
    )
    monkeypatch.setattr(F, "summary", lambda owner: {"numerical_or_mandatory_admission": False})
    report = D.run(request_path, output=tmp_path / "ordinary")
    assert len(converter_calls) == len(ledger_calls) == 1 and converter_calls[0][0].version == 2
    assert (
        converter_calls[0][1]["source_record"]
        == ledger_calls[0]["coverage"]["automatic_derivation"]["original_call_sources"]
        == source
    )
    assert replays == ["actual selected owner replay"]
    assert report["original_calls"] == 4 and report["original_source_members"] == 12
    assert report["mandatory_obligations"] == report["mandatory_unavailable"] == len(missing)
    assert {row["id"] for row in report["required_unknowns"]} == {row["id"] for row in missing}
    assert report["original_numerical_admissions"] == report["candidate_comparisons"] == 0
    assert (
        report["phases"]["1"]["handoff_accepted"] is False and report["coverage_status"] == "source_prepared_incomplete"
    )
