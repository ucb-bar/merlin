"""Preserved releases keep source commitments and never invent Phase 0 lineage."""

import json
import tomllib

import pytest
from merlin_experiments.corpus.preserved import verify_preserved
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError


def test_preserved_review_rejects_source_drift_and_fabricated_lineage(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("original: true\n")
    manifest = tmp_path / "release/payload/corpus/MANIFEST.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("generated: []\n")
    prepared = {
        "source_run": None,
        "generation_lineage": None,
        "preserved_sources": [{"path": str(source), "sha256": fingerprint(source)}],
    }
    verify_preserved(tmp_path / "release", prepared)
    prepared["source_run"] = "invented-phase0"
    with pytest.raises(SpecError, match="cannot claim Phase 0"):
        verify_preserved(tmp_path / "release", prepared)
    prepared["source_run"] = None
    source.write_text("original: false\n")
    with pytest.raises(SpecError, match="source changed"):
        verify_preserved(tmp_path / "release", prepared)


def test_codex_service_tier_keeps_auth_out_of_isolated_home(tmp_path, monkeypatch):
    from merlin_experiments.phase1.providers import codex_agent

    real = tmp_path / "real"
    real.mkdir()
    (real / "auth.json").write_text(json.dumps({"fixture": "must-never-be-copied"}))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(real))
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    isolated = tmp_path / "isolated"
    report = codex_agent.prepare_codex_home(isolated, model="gpt-6.1-sol", effort="high", workspace=workspace)
    config = tomllib.loads((isolated / "config.toml").read_text())
    assert config["service_tier"] == "fast"
    assert config["model"] == "gpt-6.1-sol"
    assert report["auth_present"] and report["auth_copied"] is False
    assert not (isolated / "auth.json").exists()
    assert "must-never-be-copied" not in (isolated / "config.toml").read_text()


def test_preserved_missing_weights_stay_negative_without_allowing_indirection(tmp_path):
    from merlin_experiments.phase1.corpus_inputs import _resources

    capsule = {
        "name": "deferred",
        "kind": "model",
        "__dir__": str(tmp_path),
        "operation": {"attributes": {"weights": "weights.safetensors"}},
    }
    with pytest.raises(ValueError, match="absent or indirect"):
        _resources([capsule])
    assert _resources([capsule], preserve_missing_weights=True) == [
        {"capsule": "deferred", "field": "weights", "resource": "weights.safetensors"}
    ]
    (tmp_path / "weights.safetensors").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="absent or indirect"):
        _resources([capsule], preserve_missing_weights=True)
    capsule["operation"]["attributes"]["weights"] = "../outside"
    with pytest.raises(ValueError, match="non-capsule-local"):
        _resources([capsule], preserve_missing_weights=True)
