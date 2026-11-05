"""Canary pre-dispatch checks; no sandbox, provider, credential, or API runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from merlin.common.paths import repo_root


class ProviderBoundaryReached(RuntimeError):
    """Stop the positive comparison before a provider can execute."""


@pytest.fixture
def canary(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CODEX_CANARY", "1")
    source = repo_root() / "merlin/experiments/capsule_bench/harness/codex_canary.py"
    spec = importlib.util.spec_from_file_location("owned_codex_canary_mask_control", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / "owned-canary-output"
    monkeypatch.setattr(module, "_now", lambda: "source-control")
    runner = ModuleType("run_baseline_qa_loop")
    corpus = tmp_path / "own-corpus"
    golden = corpus / "x/y/golden.json"
    golden.parent.mkdir(parents=True)
    golden.write_text("owned mask-control answer\n")
    runner.C = SimpleNamespace(CONTEXT=SimpleNamespace(descriptor="owned-descriptor"))
    monkeypatch.setattr(module, "load_target_experiment", lambda _path: SimpleNamespace(capsule_corpus=corpus))
    monkeypatch.setitem(sys.modules, "run_baseline_qa_loop", runner)
    bundle = ModuleType("run_agent_experiment")
    bundle._load_bundle = lambda _arm, *, bundle_id: {
        "bundle_id": bundle_id,
        "allowed": [],
        "denied": [],
    }
    monkeypatch.setitem(sys.modules, "run_agent_experiment", bundle)
    from merlin.common import artifacts

    monkeypatch.setattr(artifacts, "cache_dir", lambda _name: work)
    monkeypatch.setattr(module.WT, "assemble_copy_workspace", lambda *_args, **_kwargs: {})
    calls = []

    def blocked_provider(*_args, **_kwargs):
        calls.append("provider_dispatch")
        raise ProviderBoundaryReached("test stops before any provider or API")

    monkeypatch.setattr(module.CA, "run_round", blocked_provider)
    return module, work / "source-control", calls


@pytest.mark.parametrize(
    "observation",
    [
        {"pilot_golden_visible_to_agent": "LEAK"},
        {"pilot_golden_visible_to_agent": "UNPROVEN"},
        {},
        None,
        [],
        {"pilot_golden_visible_to_agent": True},
        {"pilot_golden_visible_to_agent": 1},
        {"pilot_golden_visible_to_agent": "ok"},
        {"ok": True},
        {"pilot_golden_visible_to_agent": "OK", "unserializable": object()},
    ],
    ids=("leak", "unproven", "missing", "null", "list", "bool", "int", "wrong-case", "other-field", "malformed"),
)
def test_non_ok_mask_refuses_before_provider_and_retains_no_go(canary, monkeypatch, observation):
    module, work, calls = canary
    monkeypatch.setattr(module.WT, "probe", lambda *_args, **_kwargs: observation)
    rc, report = module.run_canary(arm="synthetic", bundle_id="owned-selection")
    assert rc == 1 and report["verdict"] == "NO-GO"
    assert calls == [] and report["provider_started"] is False
    assert report["transcript"] is None and report["codex_summary"] == {} and report["agent_report"] == ""
    assert len(report["checks"]) == 1 and report["checks"][0]["check"] == "mask_selftest"
    assert report["checks"][0]["ok"] is False
    assert json.loads((work / "canary_report.json").read_bytes()) == report
    assert (work / "ws/probe.sh").is_file() and (work / "ws/TASK.md").is_file()
    assert not tuple(work.rglob("auth.json")) and not tuple(work.rglob("*.transcript.jsonl"))


def test_mask_exception_refuses_before_provider_and_retains_observation(canary, monkeypatch):
    module, work, calls = canary

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("owned mask observation unavailable")

    monkeypatch.setattr(module.WT, "probe", unavailable)
    rc, report = module.run_canary(arm="synthetic", bundle_id="owned-selection")
    assert rc == 1 and calls == [] and report["provider_started"] is False
    assert report["checks"] == [
        {"check": "mask_selftest", "ok": False, "detail": "RuntimeError: owned mask observation unavailable"}
    ]
    assert json.loads((work / "canary_report.json").read_bytes()) == report


def test_exact_ok_mask_reaches_blocked_original_provider_comparison(canary, monkeypatch):
    module, work, calls = canary
    monkeypatch.setattr(module.WT, "probe", lambda *_args, **_kwargs: {"pilot_golden_visible_to_agent": "OK"})
    with pytest.raises(ProviderBoundaryReached, match="before any provider"):
        module.run_canary(arm="synthetic", bundle_id="owned-selection")
    assert calls == ["provider_dispatch"]
    assert not (work / "canary_report.json").exists()
    assert not tuple(work.rglob("auth.json")) and not tuple(work.rglob("*.transcript.jsonl"))


def test_cli_propagates_unproven_no_go_without_provider(canary, monkeypatch, capsys):
    module, work, calls = canary
    monkeypatch.setattr(module.WT, "probe", lambda *_args, **_kwargs: {"pilot_golden_visible_to_agent": "UNPROVEN"})
    assert module.main(["--arm", "synthetic", "--bundle", "owned-selection"]) == 1
    assert calls == [] and "NO-GO" in capsys.readouterr().out
    assert json.loads((work / "canary_report.json").read_bytes())["provider_started"] is False
