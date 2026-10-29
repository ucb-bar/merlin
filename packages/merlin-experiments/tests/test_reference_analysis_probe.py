"""A missing optional broker dependency degrades analysis, and preflight refuses it before authoring."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from merlin.targetgen.sandbox import bwrap
from merlin_experiments.phase1 import preflight as P
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_analysis_probe as probe
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_analysis_tool import _request, _world

_MINING = "merlin.mining.reconstruction_experiments"


def _without_mining(monkeypatch):
    # A None entry makes the import raise ImportError, as an absent package does.
    monkeypatch.setitem(sys.modules, _MINING, None)


def test_probe_lifts_one_visible_reference_input(tmp_path):
    fixture = _world(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = probe.probe(workspace=tmp_path, facts=fixture["facts"])
    assert result["status"] == "analyzed" and result["operation"] == "lift"
    assert result["input"].startswith("reference-kernels/")
    # The probe's unbound compiler digest never publishes a current-analysis binding.
    assert not list(fixture["path"].parent.glob(".reference-analysis-*"))


def test_probe_is_inert_without_reference_guidance(tmp_path):
    fixture = _world(tmp_path)
    assert probe.probe(workspace=tmp_path, facts=fixture["facts"]) is None


def test_missing_experiment_dependency_degrades_one_call_instead_of_failing_it(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    _without_mining(monkeypatch)
    with reference.invocation(fixture["path"], fixture["sha"]):
        degraded = reference.record_analysis({}, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        lifted = analysis.handle(_request("lift", "reference-kernels/trace.txt"), workspace=tmp_path,
                                 facts=fixture["facts"], compiler_sha256=fixture["compiler"])
    assert degraded == dict(status="unknown", reason=reference.DEPENDENCY_UNAVAILABLE, tasks=[], requests=[])
    assert lifted["status"] == "analyzed" and lifted["report"]["listed_instructions"] == 4
    assert lifted["report"]["experiment"]["reason"] == reference.DEPENDENCY_UNAVAILABLE


def test_probe_refuses_an_environment_without_the_experiment_dependency(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    _without_mining(monkeypatch)
    with reference.invocation(fixture["path"], fixture["sha"]):
        with pytest.raises(ValueError, match="merlin-mining"):
            probe.probe(workspace=tmp_path, facts=fixture["facts"])


def test_probe_refuses_when_nothing_visible_can_be_lifted(tmp_path):
    fixture = _world(tmp_path)
    for path in (tmp_path / "reference-kernels").iterdir():
        path.unlink()
    with reference.invocation(fixture["path"], fixture["sha"]):
        with pytest.raises(ValueError, match="no visible reference input"):
            probe.probe(workspace=tmp_path, facts=fixture["facts"])


def test_preflight_runs_the_probe_with_the_broker_selection_and_records_it(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    selected = []
    monkeypatch.setattr(bwrap, "frozen_selected_rtl_facts",
                        lambda ws, bundle, *, repo: selected.append((ws, repo)) or fixture["facts"])
    prepared = SimpleNamespace(workspace=tmp_path, run_dir=run, bundle={"bundle_id": "b"},
                               request=SimpleNamespace(context=SimpleNamespace(repo=tmp_path, target="t")))
    assert probe.for_prepared(prepared) is None and not selected
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = probe.for_prepared(prepared)
    assert result["status"] == "analyzed" and selected == [(tmp_path, tmp_path)]


def test_preflight_result_carries_the_probe_and_a_failed_probe_stops_it(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "verify_prepared_inputs", lambda prepared: "m" * 64)
    monkeypatch.setattr("merlin_experiments.phase1.session.validate_preflight_options", lambda options: None)
    run = tmp_path / "run"
    run.mkdir()
    prepared = SimpleNamespace(
        request=SimpleNamespace(options=SimpleNamespace(preflight_only=True, codex_binary="")),
        resuming=False, reviewed_roots=(tmp_path,), run_dir=run,
        environment={"bundle_input_snapshot": {"content_sha256": "s" * 64}})
    monkeypatch.setattr(probe, "for_prepared", lambda selected: {"status": "analyzed", "input": "x"})
    assert P.complete(prepared) == 0
    record = json.loads((run / "preflight_result.json").read_text())
    assert record["reference_analysis_probe"] == {"status": "analyzed", "input": "x"}

    def refused(selected):
        raise ValueError("reference analysis lacks a dependency in this environment")

    (run / "preflight_result.json").unlink()
    monkeypatch.setattr(probe, "for_prepared", refused)
    with pytest.raises(ValueError, match="lacks a dependency"):
        P.complete(prepared)
    assert not (run / "preflight_result.json").exists()
