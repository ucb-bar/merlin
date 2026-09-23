"""Engine reports are read-only metadata with stable legacy report shapes."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import python_import_roots
from merlin.targetgen import capsule_runner as evaluator
from merlin.targetgen import evaluation_cohort, oracle_policy, program_oracle


def _selection():
    return {
        "engine": "fixture-engine",
        "fidelity": "elaborated_rtl",
        "reason": "fixture build receipt",
        "considered": [{"engine": "fixture-engine", "available": True}],
        "passed_over": ["slower-engine"],
    }


@pytest.fixture
def reports(monkeypatch):
    discoveries = []
    monkeypatch.setattr(evaluator, "_ensure_sim_oracles_discovered", lambda: discoveries.append("legacy"))
    monkeypatch.setattr(evaluator, "_bespoke_sim_via", lambda target: "chipyard")
    monkeypatch.setattr(evaluator, "chipyard_l3_selection", lambda target: _selection())
    monkeypatch.setattr(program_oracle, "select_rtl_engine", lambda target: _selection())
    monkeypatch.setattr(evaluator, "_SIM_ORACLES", {})
    return discoveries


@pytest.mark.parametrize("via", [None, "chipyard", "", "legacy-unknown", "plugin"])
@pytest.mark.parametrize("available", [True, False])
def test_shared_report_preserves_legacy_selection_and_error_shapes(reports, monkeypatch, via, available):
    def select(target):
        if not available:
            raise ValueError("fixture unavailable")
        return _selection()

    monkeypatch.setattr(evaluator, "chipyard_l3_selection", select)
    monkeypatch.setattr(program_oracle, "select_rtl_engine", select)
    monkeypatch.setitem(
        evaluator._SIM_ORACLES,
        "plugin",
        oracle_policy._SimOracle(lambda target: {}, lambda target: (False, "unused"), True, l3_selection=select),
    )
    actual_via = "chipyard" if via is None else via
    expected = (
        {
            **_selection(),
            "available": True,
            "target": "fixture",
            "sim_via": actual_via,
            "summary": "fixture-engine [elaborated_rtl] (over slower-engine)",
        }
        if available
        else {
            "available": False,
            "target": "fixture",
            "sim_via": actual_via,
            "reason": "ValueError: fixture unavailable",
        }
    )
    assert evaluator.describe_l3_engine is oracle_policy.describe_l3_engine
    assert oracle_policy.selected_l3_engine_report("fixture", via) == expected
    assert reports == ["legacy"]


def test_exclusive_plugin_without_selection_keeps_exact_unavailable_report(reports, monkeypatch):
    monkeypatch.setitem(
        evaluator._SIM_ORACLES,
        "plugin",
        oracle_policy._SimOracle(lambda target: {}, lambda target: (False, "unused"), True),
    )
    assert evaluator.describe_l3_engine("fixture", "plugin") == {
        "available": False,
        "target": "fixture",
        "sim_via": "plugin",
        "reason": "the 'plugin' sim oracle owns this target's cert tier and reports no engine selection",
    }


def test_cohort_preflight_retains_legacy_reporter_override(monkeypatch):
    selected = {**_selection(), "available": True}
    monkeypatch.setattr(evaluator, "describe_l3_engine", lambda target, sim_via: selected)
    descriptor = SimpleNamespace(
        target="fixture",
        sim_via="plugin",
        evaluation_cohort=lambda name: {"oracle_engine": "fixture-engine", "oracle_tier": "L3"},
    )
    report = evaluation_cohort.engine_preflight(descriptor, "cert")
    assert report["selected"] is selected
    assert report["ok"] is True


def test_core_reporter_override_is_not_shadowed_by_legacy_reexport(monkeypatch):
    expected = {"available": False, "reason": "core override"}
    monkeypatch.setattr(oracle_policy, "describe_l3_engine", lambda *args: expected)
    assert oracle_policy.selected_l3_engine_report("fixture") is expected


def test_core_only_reports_do_not_import_evaluation():
    code = """
import importlib.abc, sys
from types import SimpleNamespace
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'aet' or fullname.startswith('aet.') or fullname in {
            'merlin.targetgen.capsule_runner', 'merlin.targetgen.capsule_grade',
            'merlin.targetgen.capsule_golden', 'merlin.targetgen.program_oracle',
            'merlin.targetgen.evaluation_cohort',
        }:
            raise AssertionError('core report imported evaluation: ' + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen import oracle_policy as policy, program_engine_policy as program
policy._ensure_sim_metadata_discovered = lambda: None
policy._bespoke_sim_via = lambda target: 'chipyard'
selection = {'engine': 'fixture-engine', 'fidelity': 'elaborated_rtl', 'reason': 'fixture build receipt'}
policy.chipyard_l3_selection = lambda target: selection
program.select_rtl_engine = lambda target: selection
assert policy.selected_sim_via('fixture') == 'chipyard'
assert policy.describe_l3_engine('fixture')['available'] is True
assert policy.describe_l3_engine('fixture', '')['engine'] == 'fixture-engine'
missing = policy.describe_l3_engine('fixture', 'unregistered-plugin')
assert missing['available'] is False and 'plugin.sim_oracle_metadata' in missing['reason']
assert 'merlin.targetgen.capsule_runner' not in sys.modules
assert 'merlin.targetgen.program_oracle' not in sys.modules
assert 'merlin.targetgen.evaluation_cohort' not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()))
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
