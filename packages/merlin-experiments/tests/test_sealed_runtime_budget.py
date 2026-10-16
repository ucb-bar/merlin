"""A verified same-filesystem runtime cache changes disk cost, never the selected bytes."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.capture_execution import runtime_store, sealed_m2m


class _ReachedSandbox(Exception):
    pass


def _plan(tmp_path: Path) -> dict:
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    (venv / "lib/module.bin").write_bytes(b"v" * (16 * 1024 * 1024))
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python").write_bytes(b"interpreter")
    library = tmp_path / "library.so"
    library.write_bytes(b"library")
    trees = {
        "venv": sealed_m2m._source_tree(venv, skip_lib64=True),
        "base": sealed_m2m._source_tree(base),
    }
    runtime_bytes = trees["venv"]["bytes"] + trees["base"]["bytes"] + library.stat().st_size
    return {
        "schema": sealed_m2m.SCHEMA,
        "status": "plan_only",
        "m2m_root": str(tmp_path / "m2m"),
        "workload_root": str(tmp_path / "workload"),
        "worker": str(tmp_path / "merlin/targetgen/_m2m_capture_worker.py"),
        "merlin_root": str(tmp_path / "merlin"),
        "schemas_root": str(tmp_path / "merlin/_data/schemas"),
        "venv": str(venv),
        "base": str(base),
        "dtype": "fp32",
        "recipe": None,
        "max_snapshot_bytes": 32 * 1024 * 1024,
        "system_libs": [str(library)],
        "selected_trees": trees,
        "estimate_bytes": runtime_bytes + 1_000_000,
    }


def _issue_preflight(plan: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free: int) -> Path:
    run_parent = tmp_path / "runs"
    run_parent.mkdir()
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(sealed_m2m.shutil, "disk_usage", lambda _path: SimpleNamespace(free=free))
    monkeypatch.setattr(sealed_m2m, "_bwrap_binary", lambda _path: tmp_path / "bwrap")
    return run_parent / "capture"


def test_verified_cache_hit_admits_incremental_source_cost(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    runtime_store.store_entry(plan)
    run = _issue_preflight(plan, tmp_path, monkeypatch, free=8_000_000)
    cache = runtime_store.verified_cached_entry(plan, run.parent)
    assert cache is not None and cache.runtime_bytes > 16_000_000
    monkeypatch.setattr(sealed_m2m, "_probe_sandbox", lambda _bwrap: (_ for _ in ()).throw(_ReachedSandbox()))
    with pytest.raises(_ReachedSandbox):
        sealed_m2m.issue(plan, run)
    assert not run.exists()


def test_missing_corrupt_or_extra_cache_members_never_earn_discount(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    destination = tmp_path / "runs"
    destination.mkdir()
    assert runtime_store.verified_cached_entry(plan, destination) is None
    full_cost, cache = runtime_store.snapshot_space_requirement(plan, destination)
    assert cache is None and full_cost > plan["estimate_bytes"]
    entry = runtime_store.store_entry(plan)
    assert runtime_store.verified_cached_entry(plan, destination) is not None
    (entry / "opt/capture-venv/lib/module.bin").write_bytes(b"changed")
    assert runtime_store.verified_cached_entry(plan, destination) is None
    full_cost, cache = runtime_store.snapshot_space_requirement(plan, destination)
    assert cache is None and full_cost > plan["estimate_bytes"]
    (entry / "opt/capture-venv/lib/module.bin").write_bytes(b"v" * (16 * 1024 * 1024))
    (entry / "extra.bin").write_bytes(b"unselected")
    assert runtime_store.verified_cached_entry(plan, destination) is None


def test_only_the_exact_selected_plan_and_library_bytes_earn_a_discount(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    runtime_store.store_entry(plan)
    destination = tmp_path / "runs"
    destination.mkdir()
    library = Path(plan["system_libs"][0])
    selected = [{"path": str(library), "bytes": library.stat().st_size, "sha256": sealed_m2m._file_digest(library)}]
    assert runtime_store.verified_cached_entry(plan, destination, selected) is not None
    changed = deepcopy(plan)
    changed["selected_trees"]["venv"]["sha256"] = "0" * 64
    assert runtime_store.verified_cached_entry(changed, destination, selected) is None
    wrong_library = [{**selected[0], "sha256": "0" * 64}]
    assert runtime_store.verified_cached_entry(plan, destination, wrong_library) is None


def test_other_filesystem_cannot_claim_a_hardlink_discount(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    runtime_store.store_entry(plan)
    if not Path("/dev/shm").is_dir() or Path("/dev/shm").stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("no distinct writable memory filesystem")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as destination:
        assert runtime_store.verified_cached_entry(plan, Path(destination)) is None
        full_cost, cache = runtime_store.snapshot_space_requirement(plan, Path(destination))
        assert cache is None and full_cost > plan["estimate_bytes"]


def test_discounted_issue_refuses_cache_mutation_before_link(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    runtime_store.store_entry(plan)
    run = _issue_preflight(plan, tmp_path, monkeypatch, free=8_000_000)
    monkeypatch.setattr(sealed_m2m, "_probe_sandbox", lambda _bwrap: None)
    marker = tmp_path / "store" / f"{runtime_store._identity(plan)}.complete.json"

    def change_cache(_plan, _source):
        marker.write_text(json.dumps({"identity": "changed"}))

    monkeypatch.setattr(sealed_m2m, "_stage_source", change_cache)
    with pytest.raises(sealed_m2m.SealedM2MError, match="cached runtime changed"):
        sealed_m2m.issue(plan, run)
    assert not (run / "sealed_m2m_pending.json").exists()


def test_discounted_issue_refuses_cache_mutation_during_link(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    runtime_store.store_entry(plan)
    run = _issue_preflight(plan, tmp_path, monkeypatch, free=8_000_000)
    monkeypatch.setattr(sealed_m2m, "_probe_sandbox", lambda _bwrap: None)
    monkeypatch.setattr(sealed_m2m, "_stage_source", lambda _plan, _source: None)
    marker = tmp_path / "store" / f"{runtime_store._identity(plan)}.complete.json"
    original_link = os.link
    linked = 0

    def changed_link(source, destination):
        nonlocal linked
        original_link(source, destination)
        linked += 1
        if linked == 1:
            marker.write_text(json.dumps({"identity": "changed during link"}))

    monkeypatch.setattr(runtime_store.os, "link", changed_link)
    with pytest.raises(sealed_m2m.SealedM2MError, match="cached runtime changed during hard-link snapshot"):
        sealed_m2m.issue(plan, run)
    assert linked > 0
    assert not (run / "sealed_m2m_pending.json").exists()
    assert not (run / "capture").exists()


def test_concurrent_runtime_snapshot_waits_for_inventory_and_keeps_exact_cached_identity(tmp_path, monkeypatch):
    """A second capture's link waits for the first one's exclusive inventory instead of racing it."""
    import threading
    import time

    from merlin_experiments.capture_execution import sealed_static

    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    entry = runtime_store.store_entry(plan)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    cache = runtime_store.verified_cached_entry(plan, first)
    assert cache is not None
    source = entry / "opt/capture-venv/lib/module.bin"
    original_digest = sealed_static._file_digest
    inventories, observations, errors = 0, [], []
    concurrent: list[threading.Thread] = []

    def link_second():
        try:
            runtime_store.link_runtime(plan, second, verified_cache=cache)
        except BaseException as exc:  # noqa: BLE001 -- reported below
            errors.append(exc)

    def interleaved_digest(path):
        nonlocal inventories
        if path == source:
            inventories += 1
            if inventories == 2:
                # The second capture starts linking the same immutable inode while the first
                # caller's post-link inventory is reading that file.
                before = source.stat()
                thread = threading.Thread(target=link_second)
                thread.start()
                concurrent.append(thread)
                time.sleep(0.3)
                observations.append((before, source.stat()))
        return original_digest(path)

    monkeypatch.setattr(sealed_static, "_file_digest", interleaved_digest)
    runtime_store.link_runtime(plan, first, verified_cache=cache)
    concurrent[0].join(timeout=60)
    assert not concurrent[0].is_alive() and errors == []
    assert len(observations) == 1
    before, during = observations[0]
    # The inventory saw a stable inode: the second link waited for the exclusive verification.
    assert (before.st_nlink, before.st_ctime_ns) == (during.st_nlink, during.st_ctime_ns)
    assert source.stat().st_nlink == before.st_nlink + 1
    monkeypatch.setattr(sealed_static, "_file_digest", original_digest)
    assert runtime_store.verified_cached_entry(plan, first) == cache
    for destination in (first, second):
        linked = destination / "opt/capture-venv/lib/module.bin"
        assert linked.stat().st_ino == source.stat().st_ino
        assert original_digest(linked) == original_digest(source)


@pytest.mark.parametrize("change", ["bytes_restored_mtime", "mode", "continuous_link_churn"])
def test_link_count_rehash_never_admits_changed_content_modes_or_unstable_reads(tmp_path, monkeypatch, change):
    from merlin_experiments.capture_execution import sealed_static

    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _plan(tmp_path)
    entry = runtime_store.store_entry(plan)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    cache = runtime_store.verified_cached_entry(plan, runtime)
    assert cache is not None
    source = entry / "opt/capture-venv/lib/module.bin"
    original_digest = sealed_static._file_digest
    inventories = 0

    def changed_digest(path):
        nonlocal inventories
        if path == source:
            inventories += 1
            if inventories == 2 or (inventories > 2 and change == "continuous_link_churn"):
                before = source.stat()
                if change == "bytes_restored_mtime":
                    with source.open("r+b") as stream:
                        stream.write(b"x")
                    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                elif change == "mode":
                    source.chmod(before.st_mode ^ 0o100)
                os.link(source, tmp_path / f"other_snapshot_{inventories}")
        return original_digest(path)

    monkeypatch.setattr(sealed_static, "_file_digest", changed_digest)
    with pytest.raises(sealed_m2m.SealedM2MError, match="cached runtime changed during hard-link snapshot"):
        runtime_store.link_runtime(plan, runtime, verified_cache=cache)
    assert inventories <= 4  # Initial admission plus at most three post-link reads.
