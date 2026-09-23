"""Lifecycle protection survives planning races, frozen outputs, and unknown historical state."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest

from merlin import benchharness
from merlin.common import paths
from merlin.common import storage_cli as storage
from merlin.common import storage_lifecycle as lifecycle


@pytest.fixture
def output(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.delenv("MERLIN_BUNDLE_CAS", raising=False)
    declared = dict(storage.contract(), scan_roots=[])
    monkeypatch.setattr(storage, "contract", lambda: declared)
    return root


def _finished(path: Path) -> Path:
    with lifecycle.lease(path, owner="test producer"):
        path.mkdir(parents=True)
        (path / "result.json").write_text("{}")
    return path


def _campaign(root: Path) -> tuple[Path, Path]:
    group = root / "runs" / "test-target" / "compiler"
    return tuple(_finished(group / f"2026090{day}T010000Z_agent_seed000_abc1234") for day in (1, 2))


def test_harness_outputs_follow_override_without_moving_resource_or_machine_paths(output, monkeypatch):
    resources = paths.schemas_dir()
    machine = paths.tracked_build_dir()
    monkeypatch.delenv("MERLIN_OUT_ROOT")
    assert paths.schemas_dir() == resources
    assert paths.tracked_build_dir() == machine
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(output))
    assert benchharness.runs_root("test-target", "compiler") == output / "runs" / "test-target" / "compiler"
    assert benchharness.reports_root("comparison") == output / "artifacts" / "comparison"
    assert paths.schemas_dir() == resources
    assert paths.tracked_build_dir() == machine
    assert not output.exists()


def test_reporting_empty_workspace_never_creates_state(output):
    report = storage.collect()
    assert report["lifecycle"] == {"status": "known", "paths": []}
    assert not output.exists(), "a read-only report created output/control/cache state"


def test_lease_and_pin_do_not_edit_frozen_evidence(output):
    run = output / "runs" / "frozen"
    run.mkdir(parents=True)
    evidence = run / "manifest.json"
    evidence.write_text('{"frozen":true}\n')
    evidence.chmod(0o400)
    run.chmod(0o500)
    try:
        with lifecycle.lease(run, owner="grader"):
            assert any("running" in reason for reason in lifecycle.blockers(run))
            token = lifecycle.pin(run, reason="published experiment")
        assert evidence.read_text() == '{"frozen":true}\n'
        assert sorted(p.name for p in run.iterdir()) == ["manifest.json"]
        assert any("pin" in reason for reason in lifecycle.blockers(run))
        lifecycle.unpin(run, token)
        assert not lifecycle.blockers(run, require_terminal=True)
    finally:
        run.chmod(0o700)


def test_exception_records_failure_and_releases_only_its_own_lease(output):
    run = output / "runs" / "one"
    other = lifecycle.acquire(run, owner="second consumer")
    with pytest.raises(RuntimeError), lifecycle.lease(run, owner="failing worker"):
        raise RuntimeError("worker failed")
    assert any("running" in reason for reason in lifecycle.blockers(run))
    other.close("failed")
    assert lifecycle.inventory()["paths"][0]["state"] == "failed"


def test_old_live_or_pinned_runs_survive_retention(output):
    old, _new = _campaign(output)
    with lifecycle.lease(old, owner="resumed run"):
        plan = storage.retention_plan(keep=1)
        assert plan["drop_units"] == 0
        assert storage.main(["retain", "--keep", "1", "--apply"]) == 0
        assert old.exists()
    token = lifecycle.pin(old, reason="paper evidence")
    assert storage.retention_plan(keep=1)["drop_units"] == 0
    lifecycle.unpin(old, token)
    assert storage.retention_plan(keep=1)["drop_units"] == 1


def test_historical_run_is_not_finished_just_because_its_name_is_old(output):
    old, _new = _campaign(output)
    historical = old.parent / "20260801T010000Z_historical_seed000_abc1234"
    historical.mkdir()
    plan = storage.retention_plan(keep=0)
    protected = [row for group in plan["groups"].values() for row in group["protected"]]
    assert any(row["path"] == str(historical) for row in protected)


def test_deletion_rechecks_pin_acquired_after_planning(output):
    old, _new = _campaign(output)
    assert storage.retention_plan(keep=1)["drop_units"] == 1
    lifecycle.pin(old / "result.json", reason="new citation")
    with pytest.raises(PermissionError, match="pin"):
        storage._remove(old, require_terminal=True)
    assert (old / "result.json").is_file()


def test_retention_rechecks_latest_pointer_changed_after_planning(output):
    old, _new = _campaign(output)
    assert storage.retention_plan(keep=1)["drop_units"] == 1
    (old.parent / "latest").symlink_to(old.name, target_is_directory=True)
    with pytest.raises(PermissionError, match="latest"):
        storage._remove(old, require_terminal=True)
    assert old.exists()


def test_pending_suffix_never_proves_abandonment(output):
    pending = output / "runs" / "copy" / "bundle_inputs.pending"
    pending.mkdir(parents=True)
    (pending / "partial.bin").write_bytes(b"partial")
    assert storage.pending_snapshots() == ([], 0)
    with lifecycle.lease(pending.parent, owner="copying input closure"):
        assert storage.pending_snapshots() == ([], 0)
        assert storage.main(["prune", "--apply", "pending-snapshots"]) == 0
        assert pending.exists()
    assert storage.pending_snapshots() == ([pending], 7)


def test_foreign_host_lease_is_unknown_not_expired(output, monkeypatch):
    monkeypatch.setattr(lifecycle.socket, "gethostname", lambda: "remote-host")
    held = lifecycle.acquire(output / "runs" / "remote", owner="remote worker")
    monkeypatch.setattr(lifecycle.socket, "gethostname", lambda: "local-host")
    assert any("unknown" in reason for reason in lifecycle.blockers(held.path))
    with pytest.raises(PermissionError, match="unknown"):
        with lifecycle.removal_guard(held.path):
            pytest.fail("unknown ownership allowed removal")


def test_dead_owner_needs_explicit_acknowledgement_of_stopped_workers(output, monkeypatch):
    held = lifecycle.acquire(output / "runs" / "dead", owner="worker")
    monkeypatch.setattr(lifecycle, "_process_identity", lambda pid: {"boot": "different", "start": "1"})
    assert lifecycle.inventory()["paths"][0]["state"] == "owner-dead"
    assert lifecycle.blockers(held.path, require_terminal=True)
    lifecycle.acknowledge_abandoned(held.path, reason="operator verified all child workers stopped")
    assert not lifecycle.blockers(held.path, require_terminal=True)


def test_abandonment_cannot_override_live_or_missing_ownership(output):
    with lifecycle.lease(output / "runs" / "live", owner="worker") as held:
        with pytest.raises(PermissionError, match="live"):
            lifecycle.acknowledge_abandoned(held.path, reason="attempted override")
    with pytest.raises(ValueError, match="no crashed"):
        lifecycle.acknowledge_abandoned(output / "runs" / "historical", reason="old timestamp")


def test_active_or_pinned_cache_child_protects_namespace(output):
    cache = output / "artifacts" / "cache" / "compilation"
    child = _finished(cache / "one")
    with lifecycle.lease(child, owner="compiler"):
        assert cache not in storage.purgeable_caches()[0]
        with pytest.raises(PermissionError, match="running"):
            storage._remove(cache)
    lifecycle.pin(child, reason="frozen toolchain")
    assert cache not in storage.purgeable_caches()[0]


def test_corrupt_registry_blocks_cleanup_and_is_reported(output):
    cache = output / "artifacts" / "cache" / "one"
    cache.mkdir(parents=True)
    (output / ".merlin-storage.json").write_text("not-json")
    assert lifecycle.inventory()["status"] == "unknown"
    with pytest.raises(OSError, match="ownership"):
        storage._remove(cache)
    assert cache.exists()


def test_tracked_evidence_is_never_removed(output):
    subprocess.run(["git", "init", "-q", str(output)], check=True)
    run = _finished(output / "runs" / "tracked")
    subprocess.run(["git", "-C", str(output), "add", "--", str(run / "result.json")], check=True)
    with pytest.raises(PermissionError, match="tracked"):
        storage._remove(run, require_terminal=True)
    assert (run / "result.json").is_file()


def test_index_inspection_failure_is_not_no_tracked_files(output):
    path = output / "runs" / "broken"
    path.mkdir(parents=True)
    (output / ".git").write_text("gitdir: missing-index-directory\n")
    assert storage._holds_tracked_files(path)
    with pytest.raises(PermissionError, match="tracked"):
        storage._remove(path)


def test_store_object_linked_after_planning_is_not_deleted(output):
    store = storage.store_root()
    store.mkdir(parents=True)
    obj = store / "object"
    obj.write_bytes(b"frozen")
    survivor = output / "consumer"
    os.link(obj, survivor)
    with pytest.raises(PermissionError, match="reference"):
        storage._remove(obj, store_object=True)
    assert survivor.read_bytes() == b"frozen"


def test_storage_contract_uses_package_resource_resolution(tmp_path, monkeypatch):
    resource = tmp_path / "storage.yaml"
    resource.write_text("out_roots: [runs, artifacts, build]\n")
    calls = []

    def resolve(*parts):
        calls.append(parts)
        return resource

    monkeypatch.setattr(paths, "data_path", resolve)
    assert storage.contract()["out_roots"] == ["runs", "artifacts", "build"]
    assert calls == [("contract", "storage.yaml")]


def test_registry_snapshot_is_json_serializable_and_read_only(output):
    with lifecycle.lease(output / "runs" / "one", owner="worker"):
        before = (output / ".merlin-storage.json").read_bytes()
        assert json.loads(json.dumps(lifecycle.inventory()))["paths"][0]["state"] == "running"
        assert (output / ".merlin-storage.json").read_bytes() == before


def test_lifecycle_control_files_cannot_be_removed(output):
    _finished(output / "runs" / "one")
    for name in (".merlin-storage.json", ".merlin-storage.lock"):
        with pytest.raises(PermissionError, match="control state"):
            storage._remove(output / name)


def test_fold_rechecks_lease_acquired_after_planning(output):
    source = _finished(output / "old-artifacts")
    destination = output / "artifacts" / "reports" / "old"
    with lifecycle.lease(source, owner="report consumer"):
        with pytest.raises(PermissionError, match="running"):
            storage.fold(source, destination)
    assert (source / "result.json").exists()
    assert not destination.exists()


def test_dedup_skips_pins_and_rechecks_after_planning(output):
    one = _finished(output / "runs" / "one") / "result.json"
    two = _finished(output / "runs" / "two") / "result.json"
    groups = storage.dedup_candidates([one.parent, two.parent], min_bytes=1)["groups"]
    assert len(groups) == 1
    lifecycle.pin(one, reason="mode-sensitive published evidence")
    assert not storage.dedup_candidates([one.parent, two.parent], min_bytes=1)["groups"]
    with pytest.raises(PermissionError, match="pin"):
        storage.dedup(groups)
    assert one.stat().st_ino != two.stat().st_ino


def test_writer_cannot_acquire_between_final_check_and_removal(output):
    run = _finished(output / "runs" / "one")
    worker = """
import sys
from merlin.common import storage_lifecycle as lifecycle
print("ready", flush=True)
with lifecycle.lease(sys.argv[1], owner="subprocess writer"):
    print("held", flush=True)
    sys.stdin.readline()
"""
    process = None
    try:
        with selectors.DefaultSelector() as readable:
            with lifecycle.removal_guard(run, require_terminal=True):
                process = subprocess.Popen(
                    [sys.executable, "-c", worker, str(run)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
                )
                readable.register(process.stdout, selectors.EVENT_READ)
                assert readable.select(timeout=10), "worker never reached the ownership check"
                assert process.stdout.readline().strip() == "ready"
                assert not readable.select(timeout=0.1), "writer bypassed the cleanup lock"
            assert readable.select(timeout=10), "writer did not acquire the released lock"
            assert process.stdout.readline().strip() == "held"
            with pytest.raises(PermissionError, match="running"):
                storage._remove(run, require_terminal=True)
        _stdout, stderr = process.communicate(input="\n", timeout=10)
        assert process.returncode == 0, stderr
        assert not lifecycle.blockers(run, require_terminal=True)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
