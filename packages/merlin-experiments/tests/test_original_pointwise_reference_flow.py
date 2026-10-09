"""Full ordinary original pointwise rosters, native values and standard sources."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_flow as F
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S
from merlin_experiments.phase0 import original_standard_ir_plan as SP
from merlin_experiments.phase0.original_call_sources import required_source_cohorts

from merlin.common import invocation_record as I
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_operator_sources import _argument
from merlin.targetgen.original_pointwise_sources import pointwise_source
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T

_spec = importlib.util.spec_from_file_location(
    "pointwise_original_fixtures", Path(__file__).with_name("original_pointwise_reference_fixtures.py")
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
live_pointwise_originals = G.live_pointwise_originals


def case_name(row):
    original = _argument(row["call"]["arguments"][0])["value"]
    target, dtype, rank = row["target"], original["dtype"], original["rank"]
    if target == "aten.clamp.default" and dtype == "int8" and row["call"]["result_roster"][0]["dtype"] == "float32":
        return "clamp_promoted"
    if dtype != "float32":
        return target.split(".")[1] + "_" + dtype
    if target == "aten.relu.default":
        return "relu_scalar" if rank == 0 else "relu_f32"
    if target == "aten.round.default":
        return "round_f32"
    parameters = row["form"]["parameters"]
    return {
        R._json({"min": -0.0, "max": 1.0}): "clamp_zero",
        R._json({"min": 7, "max": -3}): "clamp_inverted",
        R._json({"min": -0.0, "max": None}): "clamp_min",
        R._json({"min": None, "max": 0.0}): "clamp_max",
        R._json({"min": 1.0e-50, "max": -0.0}): "clamp_coerced",
    }[R._json(parameters)]


def standard(references=None):
    if not all(os.environ.get(key) for key in ("MERLIN_TEST_MLIR_OPT", "MERLIN_TEST_M2M_COMMIT")):
        pytest.skip("pointwise standard source controls need the selected stock parser")
    return {
        "schema": F.POINTWISE_STANDARD_SELECTION if references is None else SP.POINTWISE_SCHEMA,
        **({} if references is None else {"reference_roster_sha256": references.sha256}),
        "capture_checkout": os.environ["MERLIN_TEST_M2M_ROOT"],
        "capture_commit": os.environ["MERLIN_TEST_M2M_COMMIT"],
        "mlir_opt": F.pin(os.environ["MERLIN_TEST_MLIR_OPT"])
        if references is None
        else os.environ["MERLIN_TEST_MLIR_OPT"],
        "budget": {
            "max_members": 100,
            "max_source_bytes": 200000,
            "max_total_source_bytes": 60000000,
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
def observed(live_pointwise_originals, tmp_path_factory):
    intake, basis, spec = live_pointwise_originals
    owner = tmp_path_factory.mktemp("full-pointwise-reference-flow")
    value = G.selection(intake, basis)
    value.pop("operator_schema_intake_sha256")
    value.pop("semantic_basis_sha256")
    value["schema"] = F.POINTWISE_REFERENCE_SELECTION
    reference, upstream = owner / "reference.json", owner / "standard.json"
    G.write(reference, value)
    G.write(upstream, standard())
    selected = F.read_selection({"reference": F.pin(reference), "standard_ir": F.pin(upstream)}, forbidden=())
    before = spec.read_bytes()
    result = F.prepare(selected, schema_intake=intake, semantic_basis=basis, destination=owner / "products")
    assert spec.read_bytes() == before
    return result


def test_all_original_calls_cohorts_defaults_types_and_unknowns_remain(observed):
    record = json.loads(observed.references.receipt_json)
    assert record["schema"] == R.POINTWISE_SCHEMA and len(record["members"]) == 72
    checked = [row for row in record["members"] if row["state"] == "reference_checked"]
    unavailable = [row for row in record["members"] if row["state"] == "unavailable"]
    assert len(checked) == 60, [
        (row["original_member_id"], row["state"], row.get("reason")) for row in record["members"]
    ]
    assert len(unavailable) == 12
    assert {case_name(row) for row in unavailable} == {
        "round_float16",
        "round_bfloat16",
        "round_float64",
        "clamp_promoted",
    }
    originals = {(row["original_member_id"], row["node"]) for row in record["members"]}
    assert len(originals) == 24 and len({row["original_member_id"] for row in record["members"]}) == 6
    for original in originals:
        assert [
            (row["cohort"], row["extent"])
            for row in record["members"]
            if (row["original_member_id"], row["node"]) == original
        ] == list(required_source_cohorts())
    for row in checked:
        assert row["form"]["operand_dtypes"] == row["form"]["result_dtypes"]
        assert (
            len(row["call"]["schema_returns"]) == row["call"]["result_arity"] == len(row["call"]["result_roster"]) == 1
        )
        assert set(R._UNKNOWN) <= set(row["required_unknowns"])
        comparison = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
        assert comparison["passed"] and comparison["checked_elements"] >= 1
        assert I.require_environment(Path(row["invocation"]["path"]), environment=R.D.ENVIRONMENT)["returncode"] == 0
    assert F.summary(observed)["original_numerical_admissions"] == 0


def test_real_upstream_standard_sources_and_complete_reference_outputs(observed):
    record = json.loads(observed.receipt_json)
    assert record["schema"] == S.POINTWISE_SCHEMA and len(record["members"]) == 72
    checked = [row for row in record["members"] if row["state"] == "source_reference_ir_checked"]
    assert len(checked) == 60, [
        (row["original"]["original_member_id"], row["state"], row.get("reason")) for row in record["members"]
    ]
    assert {row["original"]["target"] for row in checked} == {
        "aten.relu.default",
        "aten.round.default",
        "aten.clamp.default",
    }
    for row in checked:
        assert row["comparison"]["passed"] and row["parse_invocation"]
        assert set(R._UNKNOWN) | set(S._UNKNOWN) <= set(row["required_unknowns"])
    assert I.require_environment(Path(record["invocation"]["path"]), environment=R.D.ENVIRONMENT)["returncode"] == 0
    assert F.summary(observed)["release_authority"] == "not_issued"


@pytest.mark.parametrize(
    "name",
    [
        "relu_f32",
        "round_f32",
        "clamp_zero",
        "clamp_inverted",
        "clamp_min",
        "clamp_max",
        "clamp_coerced",
        "relu_scalar",
        "relu_int64",
        "round_int64",
        "clamp_int64",
    ],
)
def test_varied_full_native_scalar_and_long_array_signed_zero_rne_subnormal_and_bounds(observed, tmp_path, name):
    references = observed.references
    row = next(
        row
        for row in json.loads(references.receipt_json)["members"]
        if row["state"] == "reference_checked" and case_name(row) == name
    )
    extent = 129 if row["form"]["rank"] < 2 else 13
    source = pointwise_source(row["form"], extent=extent, max_tensor_elements=10000)
    selected = P.policy(row["policy"], pointwise=True)
    contract = prepare_original_reference(
        row["form"],
        source,
        extent=extent,
        policy=selected,
        budget=OriginalReferenceBudget(10000, 100000, 100000, 30000),
        output_byteorder="little",
    )
    owner = tmp_path / "native"
    owner.mkdir(mode=0o700)
    chosen = json.loads(references.selection.read_bytes())
    if name == "relu_scalar":
        chosen["input_palettes"][0]["values"] = [-0.0]
    process_row = {}
    R._evaluate(process_row, contract, selection=chosen, python=os.environ["MERLIN_TEST_TORCH_PYTHON"], owner=owner)
    assert process_row["state"] == "reference_checked", process_row
    comparison = json.loads((owner / "comparison.json").read_bytes())
    count = 1 if name == "relu_scalar" else 182 if name == "clamp_inverted" else 129
    assert comparison["checked_elements"] == count
    if name == "relu_scalar":
        assert (
            bytes.fromhex(json.loads((owner / "actual.json").read_bytes())["outputs"][0]["data_hex"])
            == T.from_values("Y", "float32", [], [-0.0], byteorder="little").data
        )


@pytest.mark.parametrize(
    "defect", ["private", "call", "unknown", "policy", "bool_alias", "float_alias", "source_pin", "schema"]
)
def test_complete_v3_roster_replay_refuses_changed_required_membership_or_fields(observed, defect):
    references = observed.references
    record = json.loads(references.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "reference_checked")
    if defect == "private":
        record["members"] = [row for row in record["members"] if row["cohort"] != "withheld_transfer"]
    elif defect == "call":
        row["call"]["result_roster"] = []
    elif defect == "unknown":
        row["required_unknowns"] = []
    elif defect == "policy":
        row["policy"]["zero_sign"] = "ignore"
    elif defect == "bool_alias":
        row["extent"] = True
    elif defect == "float_alias":
        record["totals"]["source"]["tensor_elements"] = float(record["totals"]["source"]["tensor_elements"])
    elif defect == "source_pin":
        record["source_pins"].pop()
    else:
        record["schema"] = R.BATCH_SCHEMA
    with pytest.raises((ValueError, KeyError)):
        R.verify(record, schema_intake=references.schema_intake, basis=references.basis, selection=references.selection)


@pytest.mark.parametrize("metric", ["max_sources", "max_total_source_bytes", *R.E._METRICS])
def test_complete_budget_denials_preserve_all_original_slots_before_stimulus(
    live_pointwise_originals, tmp_path, monkeypatch, metric
):
    intake, basis, _ = live_pointwise_originals
    selected = G.selection(intake, basis)
    key = metric if metric.startswith("max_") else "max_" + metric
    budget = selected["source_budget"] if key in selected["source_budget"] else selected["execution_budget"]
    budget[key] = 1
    path = tmp_path / "selection.json"
    G.write(path, selected)
    monkeypatch.setattr(R, "_stimulus", lambda *args: pytest.fail("denied pointwise roster allocated stimulus"))
    result = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "denied")
    record = json.loads(result.receipt_json)
    assert len(record["members"]) == 72 and all(row["state"] == "unavailable" for row in record["members"])
    assert not any("invocation" in row for row in record["members"])


def test_legacy_selection_never_acquires_pointwise_policy_or_factory(live_pointwise_originals, tmp_path):
    intake, basis, _ = live_pointwise_originals
    selected = G.selection(intake, basis)
    selected["schema"] = P.BATCH_SCHEMA
    with pytest.raises(ValueError):
        P.validate(selected)
    selected["policies"] = []
    path = tmp_path / "selection.json"
    G.write(path, selected)
    result = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "legacy")
    assert len(json.loads(result.receipt_json)["members"]) == 72
    assert all(row["state"] == "unavailable" for row in json.loads(result.receipt_json)["members"])
    with pytest.raises(ValueError):
        SP.validate(standard(result), result)


def test_pointwise_declared_versions_cannot_be_relabelled_as_legacy(live_pointwise_originals):
    intake, basis, _ = live_pointwise_originals
    value = G.selection(intake, basis)
    value.pop("operator_schema_intake_sha256")
    value.pop("semantic_basis_sha256")
    value["schema"] = F.REFERENCE_SELECTION
    with pytest.raises(ValueError):
        F._reference(value, "a" * 64)
    value["schema"] = F.POINTWISE_REFERENCE_SELECTION
    assert F._reference(value, "a" * 64)["schema"] == P.POINTWISE_SCHEMA
