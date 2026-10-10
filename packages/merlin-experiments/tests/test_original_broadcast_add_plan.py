"""Pure complete-cohort/version checks; declarations confer no source authority."""

import copy
import json
from dataclasses import asdict

import pytest
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_reference_flow as F
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S

from merlin.targetgen import original_broadcast_add_sources as B
from merlin.targetgen.frontend_original_call import _literal
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, OriginalReferencePolicy


def policy():
    return OriginalReferencePolicy(
        B.TARGET,
        ("float32", "float32"),
        ("float32",),
        "float32",
        "finite_f32",
        "accumulator_format",
        "elementwise",
        "per_step",
        "rne",
        True,
        False,
        "after_reduction",
        0.0,
        0.0,
        "preserve",
    )


def budget():
    return {"schema": C.BUDGET_SCHEMA, **dict.fromkeys(C._LIMITS, 100000)}


def form(rank=2):
    def tensor(name, rank):
        return {
            "id": name,
            "kind": "tensor",
            "rank": rank,
            "dtype": "float32",
            "storage_dtype": "float32",
            "layout": "torch.strided",
            "device": "cpu",
        }

    return {
        "node": "add" + str(rank),
        "target": B.TARGET,
        "form_schema": B.FORM_SCHEMA,
        "status": "supported",
        "arguments": [
            {
                "name": name,
                "type": "Tensor",
                "alias": None,
                "value": {"kind": "ssa", "value": tensor(name, operand_rank)},
            }
            for name, operand_rank in (("self", rank), ("other", 1 if rank == 2 else rank))
        ]
        + [{"name": "alpha", "type": "Scalar", "alias": None, "value": _literal(1)}],
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_roster": [tensor("Y", rank)],
        "operand_dtypes": ["float32", "float32"],
        "result_dtypes": ["float32"],
        "parameters": {
            "alpha": 1,
            "broadcasting": "right_aligned",
            "operand_axes": [["varying"] * rank, ["varying"] * (1 if rank == 2 else rank)],
            "output_axes": ["varying"] * rank,
        },
        "source_numerical_semantics": policy().record(),
    }


def selection():
    return {
        "schema": P.BROADCAST_SCHEMA,
        "native_observations": "batch.v1",
        "operator_schema_intake_sha256": "a" * 64,
        "semantic_basis_sha256": "b" * 64,
        "source_budget": budget(),
        "execution_budget": {"schema": R.E.SCHEMA, **dict.fromkeys(R.E._LIMITS, 100000)},
        "reference_budget": asdict(OriginalReferenceBudget(100000, 100000, 100000, 100000)),
        "policies": [policy().record()],
        "input_palettes": [{"dtype": "float32", "values": [-3.0, -0.0, 0.0, 5.0]}],
        "cohorts": {
            cohort: [extent for name, extent in C.required_source_cohorts() if name == cohort] for cohort in P.COHORTS
        },
        "byteorder": "little",
    }


def source_members(*, limits=None, version=5):
    forms = [form(2), form(4)]
    calls = [
        *[{**row, "status": "bound"} for row in forms],
        {"node": "unsupported", "target": "aten.unknown.default", "status": "bound"},
    ]
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    result = C._sources(calls, forms, budget=limits or budget(), total=total, requested=9, version=version)
    return calls, result, total


def test_new_factory_retains_complete_original_cohorts_and_all_numerical_admissions():
    calls, members, total = source_members()
    assert len(members) == 9 and sum(row["status"] == "source_constructed" for row, _ in members) == 6
    for call in calls:
        assert [(row["cohort"], row["extent"]) for row, _ in members if row["node"] == call["node"]] == list(
            C.required_source_cohorts()
        )
    for metric, count in total.items():
        assert count == sum(row["costs"][metric] for row, _ in members if "costs" in row)
    record = {"members": [{"calls": calls, "source_members": [row for row, _ in members]}]}
    basis = type("Declaration", (), {"declaration_json": json.dumps({"members": [{"id": "independent"}]})})()
    unknowns = C.required_unknowns(record, basis=basis, unknown=lambda kind, selector, reason: (kind, selector, reason))
    assert [kind for kind, *_ in unknowns].count("original_operator_factory") == 3
    assert [kind for kind, *_ in unknowns].count("original_operator_admission") == 3
    assert all(
        row["metadata"]["source_numerical_semantics"] == policy().record() for row, _ in members if "metadata" in row
    )


def test_legacy_factory_does_not_silently_select_broadcast_forms():
    _, members, total = source_members(version=4)
    assert len(members) == 9 and all(row["status"] == "unknown" and loader is None for row, loader in members)
    assert total == {"tensor_elements": 0, "scalar_products": 0, "source_bytes": 0}
    for version, schema in (
        (1, A.ORIGINAL_POLICY_SCHEMA),
        (2, A.LINEAR_POLICY_SCHEMA),
        (3, A.POINTWISE_POLICY_SCHEMA),
        (4, A.TRANSPOSE_POLICY_SCHEMA),
        (5, A.BROADCAST_POLICY_SCHEMA),
    ):
        assert A._original_source_version({"schema": schema}) == version
        assert ("merlin.targetgen.original_broadcast_add_sources" in C.reader_modules(version)) is (version == 5)


@pytest.mark.parametrize(
    "limit",
    [
        "max_sources",
        "max_tensor_elements",
        "max_total_tensor_elements",
        "max_total_scalar_products",
        "max_total_source_bytes",
    ],
)
def test_all_requested_slots_survive_per_member_and_complete_roster_budget_refusal(limit, monkeypatch):
    limits = budget()
    limits[limit] = 1
    if limit == "max_sources":
        monkeypatch.setattr(
            B, "broadcast_add_source", lambda *args, **kwargs: pytest.fail("member denial allocated a loader")
        )
    _, members, totals = source_members(limits=limits)
    assert len(members) == 9 and all(row["status"] == "unknown" and loader is None for row, loader in members)
    assert not any(totals.values())


def test_new_reference_and_standard_versions_keep_the_same_explicit_add_policy_and_unknowns():
    selected = selection()
    assert P.validate(selected) is selected and P.transport(selected) == "batch.v1"
    assert P.selected_policy(selected, form()).record() == policy().record()
    assert R.record_schema(selected) == R.BROADCAST_SCHEMA
    owner = type("ReferenceSchema", (), {"record_without_verification": lambda self: {"schema": R.BROADCAST_SCHEMA}})()
    assert S.schemas(owner) == (
        S.BROADCAST_SCHEMA,
        "merlin.original_standard_ir_request.v4",
        "merlin.native_original_standard_ir.v4",
    )
    assert {
        "original_numerical_domain",
        "physical_effects",
        "target_support",
        "original_operation_correspondence",
    } <= set(R._UNKNOWN)
    assert {"source_effect_test_contract", "resource_axis_tail_mapping", "mandatory_numerical_stress_coverage"} <= set(
        S._UNKNOWN
    )
    declared = copy.deepcopy(selected)
    for key in ("operator_schema_intake_sha256", "semantic_basis_sha256"):
        declared.pop(key)
    declared["schema"] = F.BROADCAST_REFERENCE_SELECTION
    assert F._reference(declared, "c" * 64)["schema"] == P.BROADCAST_SCHEMA


@pytest.mark.parametrize(
    "defect", ["policy_missing", "dtype_promotion", "tolerance_int", "extra_geometry", "cohort_bool"]
)
def test_source_construction_cannot_supply_missing_policy_or_change_closed_selection(defect):
    selected = selection()
    if defect == "policy_missing":
        selected["policies"] = []
    elif defect == "dtype_promotion":
        selected["policies"][0]["readout_dtypes"] = ["int8"]
    elif defect == "tolerance_int":
        selected["policies"][0]["atol"] = 0
    elif defect == "extra_geometry":
        selected["shapes"] = [7, 9]
    else:
        selected["cohorts"]["functional_guard"] = [True]
    with pytest.raises(ValueError):
        P.selected_policy(P.validate(selected), form())
