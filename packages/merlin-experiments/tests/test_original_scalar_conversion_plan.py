"""Source-only original roster preflight; native seams are explicitly mocked."""

import copy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_scalar_conversion as C
from test_original_scalar_binary_plan import independent_sources


def budget():
    return {
        "max_source_bytes": 100000,
        "max_nesting": 32,
        "max_operations": 20,
        "max_tensor_elements": 100,
        "max_members": 20,
        "max_total_tensor_elements": 1000,
        "max_total_source_bytes": 10000000,
        "timeout_s": 180,
    }


def test_original_unsupported_slot_and_complete_three_cohorts_remain_with_no_admission(monkeypatch, tmp_path):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    members, totals = C.required_members(sources, basis=basis, budget=budget())
    assert len(members) == 9
    assert sum(source is not None for _, source in members) == 6
    assert sum(row["state"] == "unavailable" for row, _ in members) == 3
    assert totals["tensor_elements"] == 80 and totals["reserved_product_bytes"] == 1900000
    assert {row["cohort"] for row, _ in members} == {"functional_guard", "withheld_transfer"}
    assert "original_numerical_policy_and_reference" in C._UNKNOWN
    assert "mandatory_coverage_and_phase_admission" in C._UNKNOWN


@pytest.mark.parametrize("change", ["drop", "reorder", "cohort", "extent_bool", "legacy"])
def test_original_slot_denominator_and_old_vocabulary_cannot_be_changed(monkeypatch, tmp_path, change):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    rows = sources["members"][0]["source_members"]
    if change == "drop":
        rows.pop()
    elif change == "reorder":
        rows.reverse()
    elif change == "cohort":
        rows[0]["cohort"] = "development"
    elif change == "extent_bool":
        rows[0]["extent"] = True
    else:
        sources["schema"] = "merlin.original_call_sources.v5"
    with pytest.raises(ValueError):
        C.required_members(sources, basis=basis, budget=budget())


@pytest.mark.parametrize("field,value", [("max_members", 8), ("max_total_source_bytes", 1000000)])
def test_full_member_and_future_product_bounds_refuse_before_native_expansion(monkeypatch, tmp_path, field, value):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        C.required_members(sources, basis=basis, budget={**budget(), field: value})


def test_aggregate_logical_denial_preserves_original_request_slots(monkeypatch, tmp_path):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    members, totals = C.required_members(sources, basis=basis, budget={**budget(), "max_total_tensor_elements": 79})
    assert len(members) == 9 and sum(source is not None for _, source in members) == 5
    assert totals["tensor_elements"] == 56
    assert sum(row["state"] == "unavailable" for row, _ in members) == 4


@pytest.mark.parametrize("change", ["literal", "storage", "loader", "metadata_alias"])
def test_preflight_cannot_construct_a_substituted_original_source(monkeypatch, tmp_path, change):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    graph = sources["members"][0]
    if change == "literal":
        graph["forms"][0]["parameters"]["other"]["value_hex"] = (-1.0).hex()
    elif change == "storage":
        graph["source_members"][0]["metadata"]["inputs"][0]["dtype"] = "float64"
    elif change == "metadata_alias":
        graph["source_members"][0]["metadata"]["parameters"]["other"]["value_hex"] = "changed"
    else:
        from pathlib import Path

        Path(graph["source_members"][0]["source"]["path"]).write_text("changed")
    members, _ = C.required_members(sources, basis=basis, budget=budget())
    assert members[0][0]["state"] == "unavailable" and members[0][1] is None


@pytest.mark.parametrize("change", ["source", "schema", "basis", "extra", "bool", "deadline"])
def test_closed_independent_selection_retains_exact_owners_and_budgets(monkeypatch, tmp_path, change):
    sources, basis, _ = independent_sources(monkeypatch, tmp_path)
    basis.source = SimpleNamespace(sha256="a" * 64)
    intake = SimpleNamespace(sha256="b" * 64)
    tool = tmp_path / "selected-tool"
    tool.write_text("explicit selected tool fixture")
    selected = {
        "schema": C.SELECTION_SCHEMA,
        "source_record_sha256": C._digest(sources),
        "operator_schema_intake_sha256": intake.sha256,
        "semantic_basis_sha256": basis.source.sha256,
        "capture_checkout": str(tmp_path),
        "capture_commit": "c" * 40,
        "mlir_opt": str(tool),
        "budget": budget(),
    }
    assert C.validate_selection(selected, source_record=sources, schema_intake=intake, basis=basis) == selected
    mutated = copy.deepcopy(selected)
    if change in {"source", "schema", "basis"}:
        key = {
            "source": "source_record_sha256",
            "schema": "operator_schema_intake_sha256",
            "basis": "semantic_basis_sha256",
        }[change]
        mutated[key] = "d" * 64
    elif change == "extra":
        mutated["admit_numerics"] = True
    elif change == "bool":
        mutated["budget"]["max_members"] = True
    else:
        mutated["budget"]["timeout_s"] = 181
    with pytest.raises(ValueError):
        C.validate_selection(mutated, source_record=sources, schema_intake=intake, basis=basis)


def test_saved_or_copied_records_cannot_mint_live_execution_authority(tmp_path):
    owner = C.OriginalScalarConversion(None, None, b"{}", b"{}", tmp_path / "selection", b"{}")
    with pytest.raises(ValueError, match="actual live"):
        owner.record()
