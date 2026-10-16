"""Fresh private inode discovery avoids repeated walks without weakening masks."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import host_surfaces as HS


def _record(path: Path) -> tuple[str, Path, Path]:
    return str(path), path, path


def _masked_aliases(runtime: Path, records) -> set[Path]:
    destination = runtime.parent / "view"
    argv = ["--ro-bind", str(runtime), str(destination)]
    surfaces = HS._host_hardlink_surfaces(argv, records, set())
    masked = BW.apply_answer_masks(argv, surfaces)
    for surface in surfaces:
        assert not BW.is_exposed(masked, surface.path)
    return {surface.path.relative_to(destination) for surface in surfaces}


def test_overlapping_private_roots_walk_each_directory_once_and_mask_all_aliases(tmp_path, monkeypatch):
    private = tmp_path / "private"
    nested = private / "nested"
    deeper = nested / "deeper"
    deeper.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    selected = [private / "one", nested / "two", deeper / "three"]
    for index, path in enumerate(selected):
        path.write_text("synthetic private input")
        os.link(path, runtime / str(index))
    records = [_record(deeper), _record(nested), *map(_record, selected), _record(private)] * 3
    scans = Counter()
    original_scandir = os.scandir

    def counted_scan(path):
        scans[Path(path)] += 1
        return original_scandir(path)

    monkeypatch.setattr(HS.os, "scandir", counted_scan)
    assert _masked_aliases(runtime, records) == {Path(str(index)) for index in range(3)}
    assert scans == Counter({private: 1, nested: 1, deeper: 1, runtime: 1})


def test_private_leaf_symlink_retains_identity_without_following_directory_cycle(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    referent = tmp_path / "referent"
    referent.write_text("synthetic private input")
    (private / "leaf").symlink_to(referent)
    (private / "cycle").symlink_to(private, target_is_directory=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(referent, runtime / "alias")
    original_scandir = os.scandir

    def refuse_cycle(path):
        assert Path(path) != private / "cycle", "unselected directory symlinks must not recurse"
        return original_scandir(path)

    monkeypatch.setattr(HS.os, "scandir", refuse_cycle)
    assert _masked_aliases(runtime, [_record(private)]) == {Path("alias")}


def test_explicit_directory_symlink_root_still_discovers_its_private_children(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    secret = external / "secret"
    secret.write_text("synthetic private input")
    selected = private / "selected"
    selected.symlink_to(external, target_is_directory=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(secret, runtime / "alias")
    assert _masked_aliases(runtime, [_record(private)]) == set()
    assert _masked_aliases(runtime, [_record(private), _record(selected)]) == {Path("alias")}


def test_each_policy_call_rediscovers_late_private_members_and_hardlinks(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    records = [_record(private)]
    assert _masked_aliases(runtime, records) == set()
    secret = private / "late"
    secret.write_text("synthetic private input")
    os.link(secret, runtime / "late-alias")
    assert _masked_aliases(runtime, records) == {Path("late-alias")}
    (runtime / "late-alias").unlink()
    assert _masked_aliases(runtime, records) == set()


@pytest.mark.parametrize("damage", ["unreadable", "broken_leaf"])
def test_private_inventory_uncertainty_refuses(tmp_path, monkeypatch, damage):
    private = tmp_path / "private"
    private.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    if damage == "broken_leaf":
        (private / "leaf").symlink_to(tmp_path / "absent")
    else:
        original_scandir = os.scandir

        def unreadable(path):
            if Path(path) == private:
                raise PermissionError("synthetic denied private directory")
            return original_scandir(path)

        monkeypatch.setattr(HS.os, "scandir", unreadable)
    with pytest.raises(RuntimeError, match="private input hardlink identities cannot be verified"):
        _masked_aliases(runtime, [_record(private)])


def test_missing_live_source_preserves_frozen_private_alias_mask(tmp_path):
    original = tmp_path / "absent"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    secret = frozen / "secret"
    secret.write_text("synthetic private input")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(secret, runtime / "alias")
    assert _masked_aliases(runtime, [("private", original, frozen)]) == {Path("alias")}
