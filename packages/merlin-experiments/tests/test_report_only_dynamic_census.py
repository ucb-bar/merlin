"""Explicit report-only float quality never relaxes exact integer-reference custody."""

from __future__ import annotations

import copy
import json

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from test_sealed_dynamic_census import _attach_census, _census, _empty_dynamic, _integer_meta, _receipt, _stage

from merlin.capture.integerization import dynamic_integerization_count
from merlin.targetgen.application_inventory import verified_static_integerization, verify_capture_receipt
from merlin.targetgen.quant_recipe import digest


def _report_only(meta):
    meta = copy.deepcopy(meta)
    recipe = meta["recipe"]
    recipe["dynamic"].pop("site_agreement", None)
    recipe["software_numerical_engine"] = "integer_reference"
    recipe["numerical_selection"] = {
        "schema": "numerical_recipe_selection_v1",
        "policy": {
            "schema": "numerical_recipe_policy_v1",
            "weight_granularity": "tensor",
            "native_outputs": {"operators": ["linear"], "reader_policy": "compute"},
            "dynamic_activation_planes": 1,
            "float_quality": "report_only",
            "integer_extensions": {},
        },
        "source_recipe_sha256": "a" * 64,
        "source_methodology_sha256": "b" * 64,
        "source_output_path_sha256": "c" * 64,
    }
    return _redigest(meta)


def _redigest(meta):
    meta["recipe_sha256"] = meta["recipe"]["recipe_sha256"] = digest(meta["recipe"])
    return meta


def _observation(stage, meta):
    _receipt(stage, _integer_meta(meta, exported=7))
    observed = verify_capture_receipt(stage / "model.mlir")
    assert observed["status"] == "verified_materialized", observed
    assert observed["source_closure_verified"] is False
    return observed["capture_integerization"]


def test_report_only_records_finite_below_bar_metrics_without_claiming_quality(tmp_path):
    stage, meta = _stage(tmp_path)
    meta = _report_only(meta)
    meta["recipe"]["why"] = {"private": "/private/checkpoint"}
    _redigest(meta)
    for site in meta["quantization_stats"]["dynamic_contractions"]["sites"]:
        site["verification"].update(cosine_min=0.2, cosine_mean=0.3, rel_rmse_max=3.0)
    assert _census(stage, meta) == 5
    projection = _observation(stage, meta)
    assert verified_static_integerization(projection)
    evidence = projection["dynamic_integerization"]
    assert evidence["schema"] == "merlin.dynamic_integerization.v2"
    quality = evidence["float_quality"]
    assert quality["mode"] == "report_only" and quality["qualified"] is False
    assert quality["selection_sha256"] == digest(meta["recipe"]["numerical_selection"])
    assert quality["policy_sha256"] == digest(meta["recipe"]["numerical_selection"]["policy"])
    assert evidence["site_agreement"] is None
    public = json.dumps(evidence)
    assert "/private/" not in public and "matmul_qk" not in public and "native_outputs" not in public


@pytest.mark.parametrize("field,value", [("max_abs", 0.01), ("atol", 0.01), ("rtol", 0.01), ("finite", False)])
def test_report_only_does_not_relax_independent_integer_reference(tmp_path, field, value):
    stage, meta = _stage(tmp_path)
    projection = _observation(stage, _report_only(meta))
    projection["integerization_receipt"]["golden_agreement"][field] = value
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["numerical_selection"].update(schema="unreviewed"),
        lambda r: r["numerical_selection"].update(extra=True),
        lambda r: r["numerical_selection"].pop("source_recipe_sha256"),
        lambda r: r["numerical_selection"].update(source_methodology_sha256="invalid"),
        lambda r: r["numerical_selection"]["policy"].update(float_quality="pass"),
        lambda r: r["numerical_selection"]["policy"].update(extra=True),
        lambda r: r.update(software_numerical_engine="portable"),
        lambda r: r["dynamic"].update(
            site_agreement={
                "schema": "dynamic_site_agreement_v1",
                "cosine_min": 0.99,
                "rel_rmse_max": 0.1,
            }
        ),
    ],
)
def test_report_only_requires_closed_validated_selection_and_unambiguous_engine(tmp_path, edit):
    stage, meta = _stage(tmp_path)
    meta = _report_only(meta)
    edit(meta["recipe"])
    _redigest(meta)
    with pytest.raises(sealed_m2m.SealedM2MError):
        _census(stage, meta)


@pytest.mark.parametrize(
    "field,value",
    [
        ("finite", False),
        ("cosine_min", float("nan")),
        ("cosine_mean", float("inf")),
        ("rel_rmse_max", -1),
        ("samples", True),
        ("samples", 2),
        ("max_abs_error", -1),
    ],
)
def test_report_only_keeps_finite_measured_site_requirements(tmp_path, field, value):
    stage, meta = _stage(tmp_path)
    meta = _report_only(meta)
    meta["quantization_stats"]["dynamic_contractions"]["sites"][0]["verification"][field] = value
    with pytest.raises(sealed_m2m.SealedM2MError):
        _census(stage, meta)


@pytest.mark.parametrize(
    "edit",
    [
        lambda e: e.update(schema="merlin.dynamic_integerization.v1"),
        lambda e: e.update(schema=[]),
        lambda e: e["float_quality"].update(qualified=True),
        lambda e: e["float_quality"].update(mode="passed"),
        lambda e: e["float_quality"].pop("policy_sha256"),
        lambda e: e["float_quality"].update(selection_sha256="invalid"),
        lambda e: e["dynamic_owners"][0].update(calls=1),
        lambda e: e["contractions"]["sites"][0].update(lineage=[]),
        lambda e: e.update(program_sha256="d" * 64),
    ],
)
def test_report_only_projection_still_rechecks_identity_lineage_and_census(tmp_path, edit):
    stage, meta = _stage(tmp_path)
    projection = _observation(stage, _report_only(meta))
    evidence = projection["dynamic_integerization"]
    program_sha256 = evidence["program_sha256"]
    edit(evidence)
    with pytest.raises(ValueError):
        dynamic_integerization_count(evidence, emitted=2, exported=7, program_sha256=program_sha256)


def test_legacy_missing_bar_and_threshold_semantics_remain_unchanged(tmp_path):
    stage, meta = _stage(tmp_path)
    meta["quantization_stats"]["dynamic_contractions"]["sites"][0]["verification"]["cosine_min"] = 0.2
    with pytest.raises(sealed_m2m.SealedM2MError, match="declared agreement bar"):
        _census(stage, meta)
    meta["recipe"]["dynamic"].pop("site_agreement")
    _redigest(meta)
    with pytest.raises(sealed_m2m.SealedM2MError, match="no per-site agreement bar"):
        _census(stage, meta)


def test_report_only_empty_domain_cannot_hide_traced_dynamic_work(tmp_path):
    stage, meta = _empty_dynamic(tmp_path)
    meta = _report_only(meta)
    assert _census(stage, meta, exported=2) == 0
    meta["quantization_stats"]["dynamic_contractions"]["host_elements"] = 1
    with pytest.raises(sealed_m2m.SealedM2MError):
        _census(stage, meta, exported=2)


def test_selected_recipe_file_plan_and_metadata_remain_independent_joins(tmp_path, monkeypatch):
    """Keep the actual v2 plan/file join; stub only the unrelated worker bootstrap."""
    stage, meta = _stage(tmp_path)
    meta = _attach_census(stage, _integer_meta(_report_only(meta), exported=7))
    meta["recipe"].update(activation={"dtype": "int8", "mode": "static"}, weight={"dtype": "int8"})
    _redigest(meta)
    meta.update(dtype="int8", scheme="int8_static_act_int8_weight")
    meta["quantization_stats"]["recipe_sha256"] = meta["recipe_sha256"]
    source = tmp_path / "source"
    (source / "inputs").mkdir(parents=True)
    selected = source / "inputs/quant_recipe.json"
    selected.write_text(json.dumps(meta["recipe"]))
    plan = {"dtype": "int8", "recipe": sealed_m2m._recipe_selection(selected, dtype="int8")}
    monkeypatch.setattr(
        sealed_m2m,
        "_materialized",
        lambda *_args, **_kwargs: {
            "status": "verified_materialized",
            "receipt_sha256": "a" * 64,
        },
    )
    (stage / "meta.json").write_text(json.dumps(meta))
    assert sealed_m2m._materialized_v3(stage, source, stage, plan)["integer_contractions"] == 7
    # The actual issuer still requires zero error/tolerance in aggregate and every output.
    for scope in ("aggregate", "output"):
        for field in ("max_abs", "max_rel", "atol", "rtol"):
            changed = copy.deepcopy(meta)
            agreement = changed["integerization_receipt"]["golden_agreement"]
            row = agreement if scope == "aggregate" else agreement["outputs"][0]
            row[field] = 0.01
            (stage / "meta.json").write_text(json.dumps(changed))
            with pytest.raises(sealed_m2m.SealedM2MError, match="independently verified"):
                sealed_m2m._materialized_v3(stage, source, stage, plan)
    (stage / "meta.json").write_text(json.dumps(meta))
    selected.write_text(selected.read_text() + " ")
    with pytest.raises(sealed_m2m.SealedM2MError, match="snapshot differs"):
        sealed_m2m._materialized_v3(stage, source, stage, plan)
    selected.write_text(json.dumps(meta["recipe"]))
    meta["recipe_sha256"] = "d" * 64
    (stage / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(sealed_m2m.SealedM2MError, match="selected recipe"):
        sealed_m2m._materialized_v3(stage, source, stage, plan)
