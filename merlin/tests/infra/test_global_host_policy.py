"""Real policy and snapshot verification across native static-cache handoffs."""

import copy
import importlib.util
import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments import source_snapshot
from merlin_experiments.phase2 import host_policy as HP
from test_global_perf_experiment import (
    EA,
    P2_CONTRACTS,
    _make_static_cache_experiment,
    _make_test_source_snapshot,
    _static_cache_analyzer,
    _test_host_policy,
)

from merlin.common.paths import repo_root


@pytest.fixture(autouse=True)
def refuse_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("host policy integration must not launch processes or bind sockets")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def captured_policy(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("# Synthetic controller policy identity, never executed.\n")
    resources = tmp_path / "resources"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return HP.build_record(controller_source=controller, contract_root=resources)


def archived_policy(tmp_path, name, captured):
    """Copy captured owners into a different layout, then seal with the real snapshot API."""
    source = tmp_path / f"{name}-source"
    payload = source / "payload" / name
    closure_locations = {
        captured["identities"][identity]: Path("closures") / namespace / relative
        for namespace, closure in captured["closures"].items()
        for relative, identity in closure["members"].items()
    }
    locations = {}
    for index, original in enumerate(captured["sources"]):
        original_path = Path(original)
        relative = closure_locations.get(original, Path("owners") / f"{index:04d}" / original_path.name)
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, destination)
        locations[original] = destination.relative_to(source)
    snapshot = tmp_path / f"{name}-snapshot"
    source_snapshot.create(
        source,
        snapshot,
        output_root=tmp_path / "unused-output",
        source_roots=("payload",),
        python_roots=(),
        legacy_roots=(),
    )
    receipt = source_snapshot.verify(snapshot)
    rebound = copy.deepcopy(captured)
    rebound["sources"] = {str(snapshot / locations[path]): digest for path, digest in captured["sources"].items()}
    rebound["identities"] = {
        identity: str(snapshot / locations[path]) for identity, path in captured["identities"].items()
    }
    for namespace, closure in rebound["closures"].items():
        directory = str(snapshot / "payload" / name / "closures" / namespace)
        if "roots" in closure:
            closure["roots"] = [directory]
        else:
            closure["root"] = directory
    rebound["location_sha256"] = HP.location_sha256(rebound)
    assert HP.content_sha256(rebound, source_root=snapshot) == captured["sha256"]
    return snapshot, P2_CONTRACTS.document_sha256(receipt["files"]), rebound


def make_experiment(tmp_path, name, archive, active, monkeypatch):
    snapshot, digest, policy = archive
    active[0] = policy
    calls = []
    analyzer = _static_cache_analyzer(calls)
    monkeypatch.setattr(EA, "analyze_whole_model_emission", analyzer)
    experiment, candidate = _make_static_cache_experiment(tmp_path, name, snapshot, digest, analyzer)
    return experiment, candidate, calls


@pytest.fixture
def policy_provider(monkeypatch):
    active = [None]
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: copy.deepcopy(active[0]))
    return active


def test_v3_real_snapshots_reuse_identical_policy_across_different_layouts(
    tmp_path, captured_policy, policy_provider, monkeypatch
):
    first = archived_policy(tmp_path, "first", captured_policy)
    second = archived_policy(tmp_path, "nested-second", captured_policy)
    assert first[2]["sha256"] == second[2]["sha256"]
    assert first[2]["location_sha256"] != second[2]["location_sha256"]
    seed, candidate, seed_calls = make_experiment(tmp_path, "seed", first, policy_provider, monkeypatch)
    seed.analysis.analyze(candidate, hypothesis="Produce verified static evidence")
    checkpoint = seed.revision_session.seal(candidate)
    current, candidate, current_calls = make_experiment(tmp_path, "current", second, policy_provider, monkeypatch)
    receipt = current.static_analysis_import.import_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=P2_CONTRACTS.sha256_file(checkpoint)
    )
    assert receipt["status"] == "hit"
    assert len(seed_calls) == 1 and current_calls == []
    assert current.revisions.iterations[0]["analysis_reuse"]["probe_or_timing_receipts_reused"] is False
    assert current.revisions.iterations[0]["readiness"]["status"] == "ready_for_probe_admission"


@pytest.mark.parametrize("changed", ["seed", "current"])
def test_v3_tampered_source_snapshot_refuses_before_reuse(
    tmp_path, captured_policy, policy_provider, monkeypatch, changed
):
    first = archived_policy(tmp_path, "first", captured_policy)
    second = archived_policy(tmp_path, "second", captured_policy)
    seed, candidate, _ = make_experiment(tmp_path, "seed", first, policy_provider, monkeypatch)
    seed.analysis.analyze(candidate, hypothesis="Produce verified static evidence")
    checkpoint = seed.revision_session.seal(candidate)
    current, candidate, calls = make_experiment(tmp_path, "current", second, policy_provider, monkeypatch)
    archive = first if changed == "seed" else second
    path = Path(archive[2]["identities"]["controller/global"])
    path.chmod(0o644)
    path.write_text("# Changed controller bytes.\n")
    path.chmod(0o444)
    with pytest.raises((ValueError, source_snapshot.SnapshotError)):
        current.static_analysis_import.import_checkpoint(
            candidate, checkpoint=checkpoint, checkpoint_sha256=P2_CONTRACTS.sha256_file(checkpoint)
        )
    assert current.revisions.iterations == [] and calls == []


def test_v1_checkpoint_under_v3_is_explicit_version_miss(tmp_path, captured_policy, policy_provider, monkeypatch):
    snapshot, digest = _make_test_source_snapshot(tmp_path, "legacy", "POLICY = 1\n")
    seed, candidate, _ = make_experiment(
        tmp_path, "seed", (snapshot, digest, _test_host_policy(snapshot)), policy_provider, monkeypatch
    )
    seed.analysis.analyze(candidate, hypothesis="Produce historical v1 static evidence")
    checkpoint = seed.revision_session.seal(candidate)
    current, candidate, calls = make_experiment(
        tmp_path, "current", archived_policy(tmp_path, "v3", captured_policy), policy_provider, monkeypatch
    )
    receipt = current.static_analysis_import.import_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=P2_CONTRACTS.sha256_file(checkpoint)
    )
    assert receipt["status"] == "miss"
    assert receipt["reason"] == "host_verification_policy_version_changed"
    assert current.revisions.iterations == [] and calls == []


def test_v3_checkpoint_cannot_infer_missing_source_snapshot_from_paths(
    tmp_path, captured_policy, policy_provider, monkeypatch
):
    first = archived_policy(tmp_path, "first", captured_policy)
    second = archived_policy(tmp_path, "second", captured_policy)
    seed, candidate, _ = make_experiment(tmp_path, "seed", first, policy_provider, monkeypatch)
    seed.analysis.analyze(candidate, hypothesis="Produce v3 static evidence")
    checkpoint = seed.revision_session.seal(candidate)
    document = json.loads(checkpoint.read_text())
    document.pop("source_snapshot")
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(P2_CONTRACTS.canonical_json(document))
    checkpoint.chmod(0o444)
    current, candidate, calls = make_experiment(tmp_path, "current", second, policy_provider, monkeypatch)
    with pytest.raises(ValueError, match="host policy requires an explicit source_snapshot"):
        current.static_analysis_import.import_checkpoint(
            candidate, checkpoint=checkpoint, checkpoint_sha256=P2_CONTRACTS.sha256_file(checkpoint)
        )
    assert current.revisions.iterations == [] and calls == []


def test_historical_v2_checkpoint_under_v3_is_explicit_version_miss(
    tmp_path, captured_policy, policy_provider, monkeypatch
):
    helper_path = repo_root() / "packages/merlin-experiments/tests/host_policy_fixtures.py"
    spec = importlib.util.spec_from_file_location("historical_policy_fixture", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old = helper.historical_v2(tmp_path / "historical-v2")
    assert HP.content_sha256(old, source_root=tmp_path) == helper.V2_CONTENT_SHA256
    seed, candidate, _ = make_experiment(
        tmp_path, "seed", archived_policy(tmp_path, "old", old), policy_provider, monkeypatch
    )
    seed.analysis.analyze(candidate, hypothesis="Produce historical V2 static evidence")
    checkpoint = seed.revision_session.seal(candidate)
    current, candidate, calls = make_experiment(
        tmp_path, "current", archived_policy(tmp_path, "v3", captured_policy), policy_provider, monkeypatch
    )
    receipt = current.static_analysis_import.import_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=P2_CONTRACTS.sha256_file(checkpoint)
    )
    assert receipt["status"] == "miss"
    assert receipt["reason"] == "host_verification_policy_version_changed"
    assert current.revisions.iterations == [] and calls == []
