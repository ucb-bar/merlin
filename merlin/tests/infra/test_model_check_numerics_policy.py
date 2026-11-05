"""The selected models YAML owns runtime tolerances; historical files keep their policy."""
from argparse import Namespace

import numpy as np
import pytest
import yaml

from merlin.model_check import cli
from merlin.model_check import full_quantized as FQ
from merlin.model_check import models as M


def declaration(atol=0.125, rtol=0.0):
    return {"version": M.NUMERICS_SCHEMA, "report_only": True, "functional": {
        "atol": atol, "rtol": rtol, "equal_nan": False, "finite_required": True,
        "integer_boolean": "exact", "outputs": "all_declared_every_step",
    }}


def config(tmp_path, numerics):
    doc = {"schema": M.MODELS_SCHEMA, "target": "fixture", "models": {"toy": {
        "workload": "toy", "chain": {"kind": "single"}, "programs": [{"name": "p"}],
        "judged": {"kind": "logits", "outputs": [{"program": "p", "output": 0}]},
    }}}
    if numerics is not None:
        doc["numerics"] = numerics
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_alternate_selected_yaml_drives_runtime_and_retains_full_policy(tmp_path):
    path = config(tmp_path, declaration())
    loaded = M.load(path)
    assert loaded.numerics == declaration()
    policy = FQ.policy(Namespace(models=path))
    assert policy == {"atol": 0.125, "rtol": 0.0, "equal_nan": False, "finite_required": True}
    expected = np.array([1.0], np.float32)
    assert FQ.compare_outputs([expected + .1], [expected], [{}], policy)[0]["pass"]
    assert not FQ.compare_outputs([expected + .2], [expected], [{}], policy)[0]["pass"]
    # Returned policy mutation never changes the loaded declaration.
    policy["atol"] = 10.0
    assert M.selected_float_policy(loaded)["atol"] == .125


def test_cli_omitted_tolerances_select_yaml_and_conflicting_flags_refuse(tmp_path):
    path = config(tmp_path, declaration())
    argv = ["check", "--submission", "s", "--model", "toy", "--models", str(path),
            "--split", "visible", "--mode", "full", "--out", "report.json"]
    args = cli.build_parser().parse_args(argv)
    assert args.quantized_reference_atol is None and args.quantized_reference_rtol is None
    assert FQ.policy(args)["atol"] == .125
    args.quantized_reference_atol = .125
    assert FQ.policy(args)["atol"] == .125
    args.quantized_reference_atol = 100.0
    with pytest.raises(M.ModelsFileError, match="conflicts with selected models"):
        FQ.policy(args)


def test_historical_missing_policy_keeps_defaults_and_explicit_legacy_selection(tmp_path):
    path = config(tmp_path, None)
    assert M.load(path).numerics is None
    assert FQ.policy(Namespace(models=path)) == M.LEGACY_FLOAT_POLICY
    assert FQ.policy(Namespace(models=path, quantized_reference_atol=.25))["atol"] == .25


@pytest.mark.parametrize("change", ["negative", "nan", "infinity", "huge_int", "bool_atol", "integer",
                                    "nan_equal", "finite", "outputs", "missing", "version", "null"])
def test_invalid_declared_policy_never_falls_back(tmp_path, change):
    doc = declaration()
    if change in {"negative", "nan", "infinity", "huge_int", "bool_atol"}:
        doc["functional"]["atol"] = {"negative": -1, "nan": float("nan"), "infinity": float("inf"),
                                       "huge_int": 10**400, "bool_atol": True}[change]
    elif change == "version":
        doc["version"] = "unknown"
    elif change == "missing":
        del doc["functional"]["rtol"]
    elif change == "null":
        path = config(tmp_path, None)
        path.write_text(path.read_text() + "numerics: null\n")
        with pytest.raises(M.ModelsFileError):
            M.load(path)
        return
    else:
        key, value = {"integer": ("integer_boolean", "approximate"), "nan_equal": ("equal_nan", True),
                      "finite": ("finite_required", False), "outputs": ("outputs", "judged_only")}[change]
        doc["functional"][key] = value
    with pytest.raises(M.ModelsFileError):
        M.load(config(tmp_path, doc))


def test_declared_float_tolerance_never_relaxes_integer_bool_shape_dtype_or_nonfinite(tmp_path):
    policy = FQ.policy(Namespace(models=config(tmp_path, declaration(100., 100.))))
    for expected, actual in [
        (np.array([1], np.int64), np.array([2], np.int64)),
        (np.array([True]), np.array([False])),
        (np.array([1], np.float32), np.array([1], np.float64)),
        (np.array([1], np.float32), np.array([[1]], np.float32)),
        (np.array([np.nan], np.float32), np.array([np.nan], np.float32)),
        (np.array([np.inf], np.float32), np.array([np.inf], np.float32)),
    ]:
        assert not FQ.compare_outputs([actual], [expected], [{}], policy)[0]["pass"]
