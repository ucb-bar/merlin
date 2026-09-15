"""The bundle-input snapshot stores each distinct byte-string once.

Every agent run freezes its declared input closure so the bytes it ran on cannot change under it.
That freeze was a deep copy, so a campaign whose runs all granted the same toolchain and the same
model weights wrote those identical bytes once per run -- 12.8 GB per run, 235 GiB across one
campaign. These tests hold the replacement to both halves of the claim: the disk cost of the
second run is a directory entry, and the frozen bytes are still frozen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW


def _repo(tmp_path: Path, payload: bytes) -> Path:
    repo = tmp_path / "repo"
    (repo / "inputs").mkdir(parents=True)
    (repo / "inputs" / "weights.bin").write_bytes(payload)
    (repo / "inputs" / "nested").mkdir()
    (repo / "inputs" / "nested" / "isa.txt").write_text("encoding\n", encoding="utf-8")
    return repo


def _store(tmp_path: Path, monkeypatch) -> Path:
    cas = tmp_path / "cas"
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(cas))
    return cas


BUNDLE = {"allowed": [{"path": "inputs"}]}


def test_a_second_run_granting_the_same_inputs_adds_no_bytes(tmp_path, monkeypatch):
    """The point of the change: identical inputs cost disk once, not once per run."""
    _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"w" * 4096)

    first = BW.materialize_bundle_inputs(tmp_path / "run1" / "workspace", BUNDLE, repo=repo)
    second = BW.materialize_bundle_inputs(tmp_path / "run2" / "workspace", BUNDLE, repo=repo)

    # Same declared closure => same content digest, so the two runs are the same treatment.
    assert first["content_sha256"] == second["content_sha256"]
    assert first["n_bytes"] == second["n_bytes"] == 4096 + len("encoding\n")

    weights = [BW.bundle_snapshot_root(tmp_path / f"run{n}" / "workspace")
               / "repo" / "inputs" / "weights.bin" for n in (1, 2)]
    assert weights[0].read_bytes() == weights[1].read_bytes()
    assert weights[0].stat().st_ino == weights[1].stat().st_ino, "the bytes were stored twice"


def test_the_frozen_bytes_still_do_not_follow_an_in_place_source_edit(tmp_path, monkeypatch):
    """The property the deep copy existed for. A link to the SOURCE would lose it; a link to a
    store object does not, because the store holds its own copy of the bytes."""
    _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"before")
    ws = tmp_path / "run" / "workspace"

    manifest = BW.materialize_bundle_inputs(ws, BUNDLE, repo=repo)
    frozen = BW.bundle_snapshot_root(ws) / "repo" / "inputs" / "weights.bin"
    assert not (frozen.stat().st_mode & 0o222)

    # `open("wb")` truncates the source IN PLACE -- same inode, no rename.
    (repo / "inputs" / "weights.bin").write_bytes(b"after!")

    assert frozen.read_bytes() == b"before"
    assert BW.verify_bundle_snapshot(ws, BUNDLE, repo=repo) == manifest


def test_a_store_object_that_is_not_the_bytes_it_claims_is_never_served(tmp_path, monkeypatch):
    """A corrupted object must not become a run's input: the snapshot would then digest the wrong
    bytes and certify them as the frozen treatment, attributing a result to inputs it never ran on."""
    cas = _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"genuine")

    BW.materialize_bundle_inputs(tmp_path / "run1" / "workspace", BUNDLE, repo=repo)
    objects = [p for p in cas.rglob("*") if p.is_file()]
    assert objects, "nothing was stored"
    poisoned = next(p for p in objects if p.stat().st_size == len(b"genuine"))
    poisoned.chmod(0o644)
    poisoned.write_bytes(b"forged!")

    BW.materialize_bundle_inputs(tmp_path / "run2" / "workspace", BUNDLE, repo=repo)
    frozen = BW.bundle_snapshot_root(tmp_path / "run2" / "workspace") / "repo" / "inputs" / "weights.bin"
    assert frozen.read_bytes() == b"genuine"
    assert poisoned.read_bytes() == b"genuine", "the bad object was left in place to be served again"


def test_disabling_the_store_produces_the_same_snapshot(tmp_path, monkeypatch):
    """The escape hatch must change only the disk cost, never the treatment."""
    repo = _repo(tmp_path, b"w" * 128)

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    shared = BW.materialize_bundle_inputs(tmp_path / "shared" / "workspace", BUNDLE, repo=repo)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    copied = BW.materialize_bundle_inputs(tmp_path / "copied" / "workspace", BUNDLE, repo=repo)

    for key in ("content_sha256", "n_files", "n_bytes"):
        assert shared[key] == copied[key]
    plain = BW.bundle_snapshot_root(tmp_path / "copied" / "workspace") / "repo" / "inputs" / "weights.bin"
    assert plain.stat().st_nlink == 1 and not (plain.stat().st_mode & 0o222)


def test_removing_one_snapshot_leaves_its_neighbour_frozen(tmp_path, monkeypatch):
    """Teardown unlinks a shared inode; it must not reach into the other run that shares it, and
    must not leave that run's copy writable."""
    _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"shared payload")
    keep = tmp_path / "keep" / "workspace"
    drop = tmp_path / "drop" / "workspace"

    manifest = BW.materialize_bundle_inputs(keep, BUNDLE, repo=repo)
    BW.materialize_bundle_inputs(drop, BUNDLE, repo=repo)
    BW.remove_bundle_snapshot(drop)

    assert not BW.bundle_snapshot_root(drop).exists()
    survivor = BW.bundle_snapshot_root(keep) / "repo" / "inputs" / "weights.bin"
    assert survivor.read_bytes() == b"shared payload"
    assert not (survivor.stat().st_mode & 0o222)
    assert BW.verify_bundle_snapshot(keep, BUNDLE, repo=repo) == manifest


def test_a_symlinked_directory_is_dereferenced_and_a_cycle_is_refused(tmp_path, monkeypatch):
    """Dereferencing is why the copy followed links; the walk must still not run away."""
    _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"x")
    (repo / "outside").mkdir()
    (repo / "outside" / "rtl.txt").write_text("facts\n", encoding="utf-8")
    (repo / "inputs" / "link").symlink_to(repo / "outside")

    BW.materialize_bundle_inputs(tmp_path / "run" / "workspace", BUNDLE, repo=repo)
    landed = BW.bundle_snapshot_root(tmp_path / "run" / "workspace") / "repo" / "inputs" / "link"
    assert not landed.is_symlink() and (landed / "rtl.txt").read_text() == "facts\n"

    (repo / "outside" / "loop").symlink_to(repo / "inputs")
    with pytest.raises(RuntimeError, match="symlink cycle"):
        BW.materialize_bundle_inputs(tmp_path / "loop" / "workspace", BUNDLE, repo=repo)
    assert not BW.bundle_snapshot_root(tmp_path / "loop" / "workspace").exists()


def test_the_store_location_is_purgeable_by_convention(tmp_path, monkeypatch):
    """The store lives under the declared-purgeable cache root, and purging it is safe precisely
    because a live snapshot holds its own link to the bytes."""
    monkeypatch.delenv("MERLIN_BUNDLE_CAS", raising=False)
    from merlin.common import content_store
    from merlin.common.paths import artifacts_dir
    assert content_store.default_root() == artifacts_dir() / "cache" / content_store.NAMESPACE

    _store(tmp_path, monkeypatch)
    repo = _repo(tmp_path, b"purge me")
    ws = tmp_path / "run" / "workspace"
    BW.materialize_bundle_inputs(ws, BUNDLE, repo=repo)
    cas = Path(os.environ["MERLIN_BUNDLE_CAS"])
    for obj in [p for p in cas.rglob("*") if p.is_file()]:
        obj.chmod(0o644)
        obj.unlink()
    assert (BW.bundle_snapshot_root(ws) / "repo" / "inputs" / "weights.bin").read_bytes() == b"purge me"


def test_an_executable_grant_keeps_its_bit_and_never_shares_a_plain_object(tmp_path, monkeypatch):
    """A mode belongs to the inode, so two grants of the same bytes at different modes must not
    share one. A bundle may grant compilers and scripts that have to stay runnable in the box."""
    _store(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    (repo / "bin").mkdir(parents=True)
    tool = repo / "bin" / "frozen-tool"
    tool.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    tool.chmod(0o755)
    plain = repo / "bin" / "notes"
    plain.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")       # identical bytes, not runnable
    plain.chmod(0o644)

    bundle = {"allowed": [{"path": "bin"}]}
    BW.materialize_bundle_inputs(tmp_path / "run" / "workspace", bundle, repo=repo)
    root = BW.bundle_snapshot_root(tmp_path / "run" / "workspace") / "repo" / "bin"

    assert (root / "frozen-tool").stat().st_mode & 0o111, "the grant stopped being runnable"
    assert not (root / "notes").stat().st_mode & 0o111
    assert (root / "frozen-tool").stat().st_ino != (root / "notes").stat().st_ino
    for name in ("frozen-tool", "notes"):
        assert not (root / name).stat().st_mode & 0o222
