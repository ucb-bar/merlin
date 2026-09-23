"""A parametric readout stage states its parameter, or the capsule cannot fail on it."""

from __future__ import annotations

import importlib.util

import yaml

from merlin.common.paths import repo_root


def _gate():
    path = repo_root() / "build_tools" / "scripts" / "check_stage_parameters.py"
    spec = importlib.util.spec_from_file_location("check_stage_parameters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, name, attributes, label="public"):
    directory = root / name
    directory.mkdir(parents=True)
    capsule = {"name": name, "label": label, "operation": {"op": "matmul", "attributes": attributes}}
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule), encoding="utf-8")


def test_a_scale_stage_without_its_multiplier_is_an_offender(tmp_path) -> None:
    _write(tmp_path, "states_it", {"epilogue": ["acc_scale", "relu"], "acc_scale": 0.0625})
    _write(tmp_path, "assumes_it", {"epilogue": ["acc_scale"]})
    _write(tmp_path, "shift_missing", {"epilogue": ["requant"]})
    _write(tmp_path, "per_command", {"matmuls": [{"epilogue": []}, {"epilogue": ["acc_scale"]}]})
    _write(tmp_path, "not_graded", {"epilogue": ["acc_scale"]}, label="dev")
    _write(tmp_path, "no_parametric_stage", {"epilogue": ["relu"]})
    assert _gate().offenders(tmp_path) == ["assumes_it", "per_command", "shift_missing"]


def test_the_tracked_ledger_is_exactly_the_tracked_debt() -> None:
    gate = _gate()
    ledger = {
        line.strip()
        for line in gate.LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert ledger == set(gate.offenders())
