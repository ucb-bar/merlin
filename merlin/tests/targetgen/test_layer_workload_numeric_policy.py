"""The authored workload edge passes declared policy into generation and measurement."""

import importlib.util
import json
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def edge():
    path = repo_root() / "merlin/experiments/performance_contract/layer_workload.py"
    spec = importlib.util.spec_from_file_location("numeric_workload_edge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_plan_passes_complete_recipe(monkeypatch, tmp_path, edge):
    policy = {"compare": "tolerance_float", "subnormal_operand_flush": True, "atol": 0.25, "rtol": 0.02}
    monkeypatch.setattr(edge, "_numeric_inputs", lambda target: (policy, {"scope": "declared"}))
    monkeypatch.setattr(edge, "facts_for", lambda target: SimpleNamespace(operand_dtype="f32"))
    monkeypatch.setattr(edge, "kernel_ops", lambda target: object())
    monkeypatch.setattr(edge, "load_contract", lambda *args: (object(), object()))
    monkeypatch.setattr(edge.WG, "plan_matmul", lambda *args, **kwargs: kwargs)
    result = edge.build_plan("synthetic", "fixture", "1x1x1", tmp_path)
    assert result["numeric_policy"] is policy
    assert result["A"].shape == (1, 1)


def test_settle_probe_binds_recipe_and_invalidates_changed_recipe(monkeypatch, tmp_path, edge):
    policy = {"compare": "tolerance_float", "subnormal_operand_flush": True, "atol": 0.25, "rtol": 0.02}
    source = {"scope": "declared-numerical-assumptions", "sha256": "first", "hardware_verified": False}
    monkeypatch.setattr(edge, "_numeric_inputs", lambda target: (policy, dict(source)))
    cache = tmp_path / "contract.json"
    monkeypatch.setattr(edge, "contract_path", lambda *args: cache)
    facts = SimpleNamespace(tile=SimpleNamespace(rows=1, cols=1), operand_dtype="f32", accum_dtype="f32", dram_base=0)
    monkeypatch.setattr(edge, "facts_for", lambda target: facts)
    monkeypatch.setattr(edge, "kernel_ops", lambda target: object())
    monkeypatch.setattr(edge, "make_runner", lambda *args: None)
    monkeypatch.setattr(edge.WG, "probe_control_flow", lambda *args: edge.WG.ControlFlow(1, 0, "synthetic"))
    observed = []

    def settle(*args, **kwargs):
        observed.append(kwargs["numeric_policy"])
        return edge.WG.Settle.uniform(1, "synthetic")

    monkeypatch.setattr(edge.WG, "probe_settle", settle)
    first = edge.probe("synthetic", "fixture", tmp_path)
    assert first["numeric_policy_source"] == source
    assert edge.probe("synthetic", "fixture", tmp_path) == first
    assert len(observed) == 1
    source["sha256"] = "second"
    second = edge.probe("synthetic", "fixture", tmp_path)
    assert second["numeric_policy_source"] == source
    assert observed == [policy, policy]
    assert json.loads(cache.read_text())["numeric_policy_source"]["sha256"] == "second"
