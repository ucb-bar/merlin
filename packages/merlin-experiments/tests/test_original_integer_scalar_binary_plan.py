"""Complete source roster controls use declared native seams, never native credit."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion as V
from test_original_integer_scalar_binary_sources import tensor_binding
from test_original_scalar_binary_plan import budget
from test_original_scalar_binary_sources import declarations
from test_original_scalar_conversion_plan import budget as legacy_conversion_budget

from merlin.targetgen.frontend_original_call import call_contracts


def conversion_budget():
    return {
        **legacy_conversion_budget(),
        "max_promotion_tensor_elements": 100,
        "max_total_promotion_tensor_elements": 1000,
    }


def independent_sources(monkeypatch, tmp_path, *, version=7, native=True):
    documents = [declarations(scalar=1.0), declarations(scalar=1), declarations(scalar=2**63)]
    basis = SimpleNamespace(
        graph_sources=[SimpleNamespace(path=str(tmp_path / f"graph-{index}.json")) for index in range(3)],
        declaration_json=json.dumps({"members": [{"id": f"declared-{index}"} for index in range(3)]}),
    )
    original, rows = [], []
    for index, source in enumerate(basis.graph_sources):
        owner = tmp_path / "sources" / str(index)
        owner.mkdir(parents=True)
        rows.append(
            {"graph_path": source.path, "request": {}, "observation": str(owner / "observation.json"), "invocation": {}}
        )
        original.append(
            {
                "graph_path": source.path,
                "tensor_bindings": [] if index == 0 else [tensor_binding(call_contracts(*documents[index])[0])],
            }
        )
    schema = {
        "schema": "merlin.independent_operator_schema_intake.v2"
        if native
        else "merlin.independent_operator_schema_intake.v1",
        "members": original,
    }
    monkeypatch.setattr(C.D, "observe_members", lambda **kwargs: copy.deepcopy(rows))
    monkeypatch.setattr(
        C.D,
        "verify_member",
        lambda row, **kwargs: copy.deepcopy(
            documents[[source.path for source in basis.graph_sources].index(row["graph_path"])]
        ),
    )
    policy = {"original_numerical_policy_pending": True}
    record = C.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=policy,
        budget=budget(),
        destination=tmp_path / "sources",
        version=version,
    )
    return record, basis, policy, schema


def test_v7_adds_only_bound_signed64_source_slots_and_keeps_every_unsupported_cohort(monkeypatch, tmp_path):
    record, basis, policy, schema = independent_sources(monkeypatch, tmp_path)
    members, totals = V.required_members(record, basis=basis, budget=conversion_budget())
    assert record["schema"] == C.INTEGER_SCALAR_SCHEMA
    assert len(members) == 9 and sum(source is not None for _, source in members) == 6
    assert [row["cohort"] for row, _ in members].count("withheld_transfer") == 3
    assert all(row["state"] == "unavailable" for row, _ in members[6:])
    assert totals["tensor_elements"] == 80 and totals["reserved_product_bytes"] == 1900000
    assert totals["promotion_tensor_elements"] == 63
    assert C.verify(record, schema_record=schema, basis=basis, numerical_semantics=policy) == record
    assert "original_numerical_policy_and_reference" in V._UNKNOWN
    assert "mandatory_coverage_and_phase_admission" in V._UNKNOWN
    assert C.reader_modules(7) == C.reader_modules(6)


@pytest.mark.parametrize("version,native,constructed", [(6, True, 3), (7, False, 3)])
def test_old_version_or_missing_bridge_cannot_grant_integer_slots(monkeypatch, tmp_path, version, native, constructed):
    record, basis, _, _ = independent_sources(monkeypatch, tmp_path, version=version, native=native)
    members, _ = V.required_members(
        record, basis=basis, budget=conversion_budget() if version == 7 else legacy_conversion_budget()
    )
    assert len(members) == 9 and sum(source is not None for _, source in members) == constructed
    assert all(row["state"] == "unavailable" for row, _ in members[3:])


@pytest.mark.parametrize("change", ["schema", "row_drop", "binding", "float_kind", "cohort", "extent", "source"])
def test_original_membership_and_live_schema_bindings_are_replayed_without_integer_aliases(
    monkeypatch, tmp_path, change
):
    record, basis, policy, schema = independent_sources(monkeypatch, tmp_path)
    if change == "schema":
        schema["schema"] = "merlin.independent_operator_schema_intake.v1"
    elif change == "row_drop":
        schema["members"].pop(1)
    elif change == "binding":
        schema["members"][1]["tensor_bindings"][0]["native"]["wrapped_number"] = False
    elif change == "float_kind":
        record["members"][1]["forms"][0]["parameters"]["other"] = {"kind": "float", "value_hex": (1.0).hex()}
    elif change == "cohort":
        record["members"][1]["source_members"][0]["cohort"] = "development"
    elif change == "extent":
        record["members"][1]["source_members"][0]["extent"] = True
    else:
        Path(record["members"][1]["source_members"][0]["source"]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        C.verify(record, schema_record=schema, basis=basis, numerical_semantics=policy)


def test_v2_request_preserves_exact_integer_native_row_and_original_unavailable_slots(monkeypatch, tmp_path):
    record, basis, _, _ = independent_sources(monkeypatch, tmp_path)
    members, _ = V.required_members(record, basis=basis, budget=conversion_budget())
    getter = tmp_path / "original-getter"
    getter.write_bytes(b"explicit data fixture, never loaded as native")
    request = V._request(members, [], conversion_budget(), version=2, getter={"getter": str(getter)})
    assert request["schema"] == "merlin.original_scalar_conversion_request.v2"
    assert [row["index"] for row in request["members"]] == [0, 1, 2, 3, 4, 5]
    assert all(row["original_tensor_binding"] is None for row in request["members"][:3])
    assert all(
        row["original_tensor_binding"]["request"]["literal"] == {"type": "int", "value": "1"}
        for row in request["members"][3:]
    )
    with pytest.raises(ValueError, match="live schema Tensor getter"):
        V._request(members, [], conversion_budget(), version=2)


@pytest.mark.parametrize("change", ["selection_v1", "source_v6", "extra"])
def test_explicit_selection_version_cannot_widen_v1_or_relax_original_policy(monkeypatch, tmp_path, change):
    record, basis, _, _ = independent_sources(monkeypatch, tmp_path)
    basis.source = SimpleNamespace(sha256="a" * 64)
    intake = SimpleNamespace(sha256="b" * 64)
    tool = tmp_path / "stock-parser"
    tool.write_bytes(b"explicit parser fixture")
    selected = {
        "schema": V.INTEGER_SELECTION_SCHEMA,
        "source_record_sha256": V._digest(record),
        "operator_schema_intake_sha256": intake.sha256,
        "semantic_basis_sha256": basis.source.sha256,
        "capture_checkout": str(tmp_path),
        "capture_commit": "c" * 40,
        "mlir_opt": str(tool),
        "budget": conversion_budget(),
    }
    assert V.validate_selection(selected, source_record=record, schema_intake=intake, basis=basis) == selected
    if change == "selection_v1":
        selected["schema"] = V.SELECTION_SCHEMA
    elif change == "source_v6":
        record["schema"] = C.SCALAR_BINARY_SCHEMA
        selected["source_record_sha256"] = V._digest(record)
    else:
        selected["quantize_int8"] = True
    with pytest.raises(ValueError):
        V.validate_selection(selected, source_record=record, schema_intake=intake, basis=basis)


@pytest.mark.parametrize(
    "field,value,constructed,total",
    [("max_promotion_tensor_elements", 18, 4, 7), ("max_total_promotion_tensor_elements", 62, 5, 26)],
)
def test_extra_native_readouts_are_reserved_before_allocation_and_denied_slots_remain(
    monkeypatch, tmp_path, field, value, constructed, total
):
    record, basis, _, _ = independent_sources(monkeypatch, tmp_path)
    members, totals = V.required_members(record, basis=basis, budget={**conversion_budget(), field: value})
    assert len(members) == 9 and sum(source is not None for _, source in members) == constructed
    assert totals["promotion_tensor_elements"] == total
    assert members[5][0]["state"] == "unavailable"
