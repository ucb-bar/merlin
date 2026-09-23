"""Actual repeat worker body with a managed AF_UNIX service, no Ray or providers."""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import test_managed_native as native_tests
from merlin_experiments.execution.chia_native import Session, cleanup, setup

from merlin.common.paths import repo_root

service = native_tests.service


@pytest.mark.parametrize("returncode", [0, 17])
def test_actual_repeat_body_preserves_native_outcome_and_telemetry(service, tmp_path, monkeypatch, returncode):
    functions = types.ModuleType("chia.base.ChiaFunction")
    functions.ChiaFunction = lambda **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "chia", types.ModuleType("chia"))
    monkeypatch.setitem(sys.modules, "chia.base", types.ModuleType("chia.base"))
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", functions)
    context = SimpleNamespace(REPO=tmp_path, EXP=tmp_path / "target", RUNS=tmp_path / "runs", TARGET="fixture")
    monkeypatch.setitem(sys.modules, "_common", context)
    monkeypatch.setitem(sys.modules, "run_repeatability", types.ModuleType("run_repeatability"))
    source = repo_root() / "merlin/experiments/capsule_bench/harness/chia_repeatability.py"
    spec = importlib.util.spec_from_file_location("managed_repeat_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    original_paths = sys.path[:]
    try:
        spec.loader.exec_module(module)
        marker = tmp_path / "native-result"
        program = (
            "from pathlib import Path; "
            f"assert Path.cwd() == Path({str(tmp_path)!r}); "
            f"Path({str(marker)!r}).write_text('native'); raise SystemExit({returncode})"
        )
        with Session(service[0]) as session:
            invitation = session.reserve()
            setup(invitation)
            try:
                result = module.run_repeat([sys.executable, "-c", program], str(tmp_path), "repeat-id")
            finally:
                cleanup(invitation)
            receipt = session.receipt(invitation)
            assert result["run_id"] == "repeat-id" and result["returncode"] == returncode
            assert isinstance(result["wall_s"], float) and result["wall_s"] >= 0
            assert marker.read_text() == "native"
            assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
            assert receipt["guardian"]["native_started"] and receipt["guardian"]["returncode"] == returncode
    finally:
        sys.path[:] = original_paths
