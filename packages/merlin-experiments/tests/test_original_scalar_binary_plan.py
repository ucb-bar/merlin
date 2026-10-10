"""Ordinary source-only owner controls over independent minimal declarations.

The native schema seam is explicitly mocked. These controls issue no original
conversion, software/numerical ownership or protected-corpus coverage.
"""

import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import component_execution_budget as E
from merlin_experiments.phase0 import original_call_sources as C
from test_original_scalar_binary_sources import declarations


def budget():
    return {"schema": C.BUDGET_SCHEMA, **dict.fromkeys(C._LIMITS, 100000)}


def independent_sources(monkeypatch, tmp_path, *, version=6, limits=None):
    documents = [
        declarations("aten.mul.Tensor", 1.0),
        declarations("aten.div.Tensor", 2.23606797749979),
        declarations(scalar=1),
    ]
    # A complete third unsupported call stays in all three required cohorts.
    basis = SimpleNamespace(
        graph_sources=[SimpleNamespace(path=str(tmp_path / f"graph-{index}.json")) for index in range(3)],
        declaration_json=json.dumps({"members": [{"id": f"independent-{index}"} for index in range(3)]}),
    )
    rows = []
    for index, source in enumerate(basis.graph_sources):
        directory = tmp_path / "sources" / str(index)
        directory.mkdir(parents=True)
        rows.append(
            {
                "graph_path": source.path,
                "request": {},
                "observation": str(directory / "observed.json"),
                "invocation": {},
            }
        )
    monkeypatch.setattr(C.D, "observe_members", lambda **kwargs: copy.deepcopy(rows))
    monkeypatch.setattr(
        C.D,
        "verify_member",
        lambda row, **kwargs: copy.deepcopy(
            documents[[item.path for item in basis.graph_sources].index(row["graph_path"])]
        ),
    )
    policy = {"pending_original_numeric_policy": True}
    record = C.observe(
        schema_record={},
        basis=basis,
        numerical_semantics=policy,
        budget=budget() if limits is None else limits,
        destination=tmp_path / "sources",
        version=version,
    )
    return record, basis, policy


def test_opt_in_observer_preserves_all_calls_cohorts_and_policy_without_new_admissions(monkeypatch, tmp_path):
    record, basis, policy = independent_sources(monkeypatch, tmp_path)
    assert record["schema"] == C.SCALAR_BINARY_SCHEMA
    assert sum(len(row["calls"]) for row in record["members"]) == 3
    members = [member for row in record["members"] for member in row["source_members"]]
    assert len(members) == 9
    assert sum(row["status"] == "source_constructed" for row in members) == 6
    assert sum(row["status"] == "unknown" for row in members) == 3
    assert all(
        [(member["cohort"], member["extent"]) for member in row["source_members"]] == list(C.required_source_cohorts())
        for row in record["members"]
    )
    missing = C.required_unknowns(record, basis=basis, unknown=A.P._unknown)
    assert sum(row["kind"] == "original_operator_factory" for row in missing) == 3
    assert sum(row["kind"] == "original_operator_admission" for row in missing) == 3
    for member in members[:6]:
        assert member["metadata"]["source_numerical_semantics"] == policy
        assert len(member["metadata"]["inputs"]) == len(member["metadata"]["outputs"]) == 1
    assert all(row["status"] == "unknown" for member in record["members"] for row in member["policy_compatibility"])


def test_legacy_v5_keeps_all_scalar_factory_gaps_and_exact_old_reader_roster(monkeypatch, tmp_path):
    record, basis, _ = independent_sources(monkeypatch, tmp_path, version=5)
    assert record["schema"] == C.BROADCAST_SCHEMA
    assert all(member["status"] == "unknown" for row in record["members"] for member in row["source_members"])
    assert C.reader_modules(6) == C.reader_modules(5) + ("merlin.targetgen.original_scalar_binary_sources",)
    assert (
        sum(
            row["kind"] == "original_operator_factory"
            for row in C.required_unknowns(record, basis=basis, unknown=A.P._unknown)
        )
        == 9
    )


@pytest.mark.parametrize(
    "field,value,constructed",
    [("max_sources", 8, 0), ("max_tensor_elements", 11, 2), ("max_total_tensor_elements", 79, 5)],
)
def test_complete_original_roster_and_payload_budgets_retain_every_denied_slot(
    monkeypatch, tmp_path, field, value, constructed
):
    limits = budget()
    limits[field] = value
    record, basis, _ = independent_sources(monkeypatch, tmp_path, limits=limits)
    members = [member for row in record["members"] for member in row["source_members"]]
    assert len(members) == 9 and sum(member["status"] == "source_constructed" for member in members) == constructed
    missing = C.required_unknowns(record, basis=basis, unknown=A.P._unknown)
    assert sum(row["kind"] == "original_operator_factory" for row in missing) == 9 - constructed
    assert sum(row["kind"] == "original_operator_admission" for row in missing) == 3


@pytest.mark.parametrize(
    "change",
    [
        "cohort",
        "literal",
        "rank",
        "storage",
        "second_ssa",
        "cost_bool",
        "ordinal_bool",
        "loader",
        "member_drop",
        "version",
    ],
)
def test_live_reconstruction_refuses_original_source_row_substitution(monkeypatch, tmp_path, change):
    record, basis, policy = independent_sources(monkeypatch, tmp_path)
    row = record["members"][0]
    if change == "cohort":
        row["source_members"][0]["cohort"] = "development"
    elif change == "literal":
        row["forms"][0]["parameters"]["other"]["value_hex"] = (-1.0).hex()
    elif change == "rank":
        row["forms"][0]["rank"] = 1
    elif change == "storage":
        row["calls"][0]["result_roster"][0]["storage_dtype"] = "float64"
    elif change == "second_ssa":
        row["calls"][0]["arguments"][1]["value"] = copy.deepcopy(row["calls"][0]["arguments"][0]["value"])
    elif change == "cost_bool":
        row["source_members"][0]["costs"]["scalar_products"] = True
    elif change == "ordinal_bool":
        row["forms"][0]["arguments"][1]["ordinal"] = True
    elif change == "loader":
        from pathlib import Path

        Path(row["source_members"][0]["source"]["path"]).write_text("substituted\n")
    elif change == "member_drop":
        row["source_members"].pop()
    else:
        record["schema"] = C.BROADCAST_SCHEMA
    with pytest.raises(ValueError):
        C.verify(record, schema_record={}, basis=basis, numerical_semantics=policy)


def test_automatic_new_version_preserves_closed_legacy_policy_fields_and_facets():
    policy = {
        "schema": A.SCALAR_BINARY_POLICY_SCHEMA,
        "status": "reviewed",
        "hardware": {"target": "unit", "source_sha256": "a" * 64},
        **dict.fromkeys(
            (
                "software_spec_sha256",
                "numerical_semantics_sha256",
                "semantic_basis_sha256",
                "operator_schema_intake_sha256",
                "arithmetic_intake_sha256",
                "packing_intake_sha256",
            ),
            "b" * 64,
        ),
        "budget": {"max_members": 100000, "max_interaction_cells": 100000},
        "execution_budget": {"schema": E.SCHEMA, **dict.fromkeys(E._LIMITS, 100000)},
        "original_source_budget": budget(),
    }
    assert A._closed_policy(policy) == policy
    assert A._original_source_version(policy) == 6
    assert A._selected_effects(policy) and A._selected_arithmetic(policy)
    prior = {**policy, "schema": A.BROADCAST_POLICY_SCHEMA}
    assert A._closed_policy(prior) == prior and A._original_source_version(prior) == 5
    with pytest.raises(ValueError):
        A._closed_policy({**policy, "scalar_tensor_promotion": True})
