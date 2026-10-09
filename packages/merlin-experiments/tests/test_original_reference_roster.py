"""Complete original obligations, fixed native execution and independent answers."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R

from merlin.common import invocation_record as I

_spec = importlib.util.spec_from_file_location(
    "private_original_reference_fixtures", Path(__file__).with_name("original_reference_fixtures.py")
)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
live_originals = F.live_originals


@pytest.fixture(scope="module")
def observed(live_originals, tmp_path_factory):
    intake, basis, spec = live_originals
    owner = tmp_path_factory.mktemp("full-original-reference-roster")
    selection = owner / "selection.json"
    F.write(selection, F.selection(intake, basis))
    before = spec.read_bytes()
    result = R.prepare(schema_intake=intake, basis=basis, selection=selection, destination=owner / "observations")
    assert spec.read_bytes() == before
    return result


def test_actual_source_native_reference_all_original_slots_and_cohorts(observed):
    record = observed.record()
    assert len(record["members"]) == 16
    assert {row["cohort"] for row in record["members"]} == set(P.COHORTS)
    checked = [row for row in record["members"] if row["state"] == "reference_checked"]
    assert len(checked) == 12
    assert {row["target"] for row in checked} == {"aten.matmul.default", "aten.add.Tensor", "aten.conv2d.default"}
    unknown = [row for row in record["members"] if row["state"] == "unavailable"]
    assert {row["original_member_id"] for row in unknown} == {"matmul_f16", "add_nonunit"}
    for row in checked:
        assert set(row["products"]) == {"source", "metadata", "inputs", "reference", "actual", "comparison"}
        comparison = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
        assert comparison["passed"] and comparison["mismatches"] == []
        process = I.require_environment(Path(row["invocation"]["path"]), environment=R.D.ENVIRONMENT)
        assert process["returncode"] == 0
        assert "original_numerical_domain" in row["required_unknowns"]
        assert "phase1_candidate_execution" in row["required_unknowns"]
    assert "admission" not in record or record.get("admission") is None
    assert all(Path(pin["path"]).stat().st_mode & 0o077 == 0 for row in checked for pin in row["products"].values())


@pytest.mark.parametrize(
    "defect",
    [
        "missing_private",
        "cohort",
        "call",
        "policy",
        "cost",
        "unknown",
        "status",
        "missing_product",
        "missing_invocation",
        "output_name",
        "owner",
        "reader_pin",
    ],
)
def test_re_signed_saved_claims_cannot_drop_or_replace_actual_original_products(observed, defect):
    record = observed.record()
    row = next(row for row in record["members"] if row["state"] == "reference_checked")
    if defect == "missing_private":
        record["members"] = [row for row in record["members"] if row["cohort"] != "withheld_transfer"]
    elif defect == "cohort":
        row["cohort"] = "withheld_transfer" if row["cohort"] == "functional_guard" else "functional_guard"
    elif defect == "call":
        row["call"]["target"] = "aten.clone.default"
    elif defect == "policy":
        row["policy"]["atol"] = 1.0
    elif defect == "cost":
        row["cost"]["reference_work"] = 0
    elif defect == "unknown":
        row["required_unknowns"] = []
    elif defect == "status":
        row["state"] = "comparison_unavailable"
        row["reason"] = "original complete comparison differs from independent replay"
    elif defect == "missing_product":
        row["products"].pop("comparison")
    elif defect == "missing_invocation":
        row.pop("invocation")
    elif defect == "output_name":
        row["form"]["result_dtypes"] = ["int32"]
    elif defect == "owner":
        record["destination"] = str(Path(record["destination"]).parent)
    else:
        record["source_pins"].pop()
    with pytest.raises((ValueError, KeyError)):
        R.verify(record, schema_intake=observed.schema_intake, basis=observed.basis, selection=observed.selection)


def test_saved_roster_does_not_recreate_live_source_owner(observed):
    with pytest.raises(ValueError, match="live source preparation"):
        replace(observed).verify()


@pytest.mark.parametrize("limit", ["max_sources", "max_tensor_elements", "max_total_source_bytes"])
def test_complete_source_budget_denominator_remains_without_tensor_execution(
    live_originals, tmp_path, monkeypatch, limit
):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["source_budget"][limit] = 1
    path = tmp_path / "selection.json"
    F.write(path, selected)
    monkeypatch.setattr(R, "_stimulus", lambda *args: pytest.fail("over-budget source allocated shaped input"))
    roster = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "results")
    rows = roster.record()["members"]
    assert len(rows) == 16 and all(row["state"] == "unavailable" for row in rows)
    assert not any("invocation" in row for row in rows)


@pytest.mark.parametrize("metric", R.E._METRICS)
def test_all_native_and_reference_materializations_count_before_execution(
    live_originals, tmp_path, monkeypatch, metric
):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["execution_budget"]["max_" + metric] = 1
    path = tmp_path / "selection.json"
    F.write(path, selected)
    monkeypatch.setattr(R, "_stimulus", lambda *args: pytest.fail("budget denial reached stimulus allocation"))
    roster = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "results")
    record = roster.record()
    assert len(record["members"]) == 16
    assert all(row["state"] == "unavailable" for row in record["members"])
    assert record["totals"]["execution"][metric] == 0


def test_missing_operation_policy_keeps_every_original_source_obligation(live_originals, tmp_path):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["policies"] = []
    path = tmp_path / "selection.json"
    F.write(path, selected)
    roster = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "results")
    assert len(roster.record()["members"]) == 16
    assert all(row["state"] == "unavailable" for row in roster.record()["members"])


@pytest.mark.parametrize(
    "defect",
    [
        "tolerance_absent",
        "implicit_byteorder",
        "ambiguous_policy",
        "positive_palette",
        "nan",
        "shape_selector",
        "bool_extent",
        "unbound_basis",
    ],
)
def test_selection_cannot_hide_missing_choices_or_select_original_shapes(live_originals, defect):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    if defect == "tolerance_absent":
        selected["policies"][0].pop("atol")
    elif defect == "implicit_byteorder":
        selected.pop("byteorder")
    elif defect == "ambiguous_policy":
        selected["policies"].append(copy.deepcopy(selected["policies"][0]))
    elif defect == "positive_palette":
        selected["input_palettes"][0]["values"] = [1.0, 2.0]
    elif defect == "nan":
        selected["input_palettes"][0]["values"] = [-1.0, float("nan"), 2.0]
    elif defect == "shape_selector":
        selected["shapes"] = [5, 7]
    elif defect == "bool_extent":
        selected["cohorts"]["functional_guard"] = [True]
    else:
        selected["semantic_basis_sha256"] = "not-an-original-identity"
    with pytest.raises(ValueError):
        P.validate(selected)


def test_readout_overflow_is_not_a_source_reference_pass(live_originals, tmp_path):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["policies"] = [row for row in selected["policies"] if row["operand_dtypes"] == ["int8", "int8"]]
    for row in selected["policies"]:
        row["arithmetic"] = "bounded_exact"
    selected["input_palettes"] = [{"dtype": "int8", "values": [-128, 127, 126]}]
    path = tmp_path / "selection.json"
    F.write(path, selected)
    roster = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "results")
    rows = roster.record()["members"]
    assert len(rows) == 16
    overflow = [row for row in rows if row["original_member_id"] in {"add_i8", "matmul_i8"}]
    assert len(overflow) == 4 and all(row["state"] == "reference_unavailable" for row in overflow)
    assert all("overflow" in row["reason"] and "invocation" not in row for row in overflow)
    assert all(set(row["products"]) == {"source", "metadata", "inputs"} for row in overflow)


def test_exact_saved_full_comparison_cannot_hide_changed_final_element(observed):
    record = observed.record()
    row = next(row for row in record["members"] if row["state"] == "reference_checked")
    contract = R._drafts(
        record["defaults"],
        schema=observed.schema_intake.record(),
        basis=observed.basis,
        selection=json.loads(observed.selection.read_bytes()),
    )[1][record["members"].index(row)]
    inputs = R._tensors(json.loads(Path(row["products"]["inputs"]["path"]).read_bytes()))
    actual = json.loads(Path(row["products"]["actual"]["path"]).read_bytes())["outputs"]
    tensor = R._tensors(actual)[0]
    values = list(tensor.values())
    values[-1] += 1
    damaged = R.T.from_values(tensor.name, tensor.dtype, tensor.shape, values, byteorder=tensor.byteorder)
    comparison = contract.compare(inputs, (damaged,))
    assert not comparison["passed"]
    assert comparison["checked_elements"] == len(values)
    assert comparison["mismatches"][-1]["index"] == len(values) - 1


def test_f32_full_comparison_accounts_for_rational_intermediates(observed):
    record = observed.record()
    floats = [row for row in record["members"] if row.get("policy", {}).get("arithmetic") == "finite_f32"]
    assert floats
    for row in floats:
        assert row["cost"]["scalar_bits"] > 64
        assert any(item["role"] == "comparison_rational_parts" for item in row["cost"]["allocations"])
        assert any(item["role"] == "native_input_hex" for item in row["cost"]["allocations"])


def test_original_format_and_explicit_tolerance_drive_rational_width(live_originals):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    policy = P.policy(selected["policies"][0])
    assert P.comparison_bits(policy) > 64
    tiny_tolerance = replace(policy, atol=float.fromhex("0x0.0000000000001p-1022"))
    assert P.comparison_bits(tiny_tolerance) > 1074
    assert P.comparison_bits(P.policy(selected["policies"][1])) == 0


def test_rational_width_budget_denies_before_tensor_reference_execution(live_originals, tmp_path, monkeypatch):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["execution_budget"]["max_scalar_bits"] = 64
    selected["policies"] = [row for row in selected["policies"] if row["arithmetic"] == "finite_f32"]
    path = tmp_path / "selection.json"
    F.write(path, selected)
    monkeypatch.setattr(R, "_stimulus", lambda *args: pytest.fail("rational budget denial allocated input tensor"))
    roster = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "results")
    rows = roster.record()["members"]
    assert len(rows) == 16 and all(row["state"] == "unavailable" for row in rows)
    assert any("scalar_bits" in row["reason"] for row in rows)
