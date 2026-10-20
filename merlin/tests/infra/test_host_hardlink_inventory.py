"""Fresh private inode discovery avoids repeated walks without weakening masks."""

from __future__ import annotations

import os
import stat
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


def _path_object_inventory(records) -> set[tuple[int, int]]:
    """The inventory exactly as it was written with a ``Path`` per member (the reference result)."""
    inodes, members, directories, indirect = set(), set(), set(), set()
    roots = {root for _, original, frozen in records for root in (original, frozen)}

    def remember(path, info):
        members.add(path)
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            inodes.add((info.st_dev, info.st_ino))

    for root in sorted(roots, key=lambda path: (len(path.parts), str(path))):
        if root in members and root not in indirect:
            continue
        if not root.exists():
            continue
        info = root.stat()
        remember(root, info)
        if not stat.S_ISDIR(info.st_mode):
            continue
        pending = [root]
        while pending:
            directory = pending.pop()
            if directory in directories:
                continue
            directories.add(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    member = Path(entry.path)
                    if member in members:
                        continue
                    info = entry.stat()
                    remember(member, info)
                    if stat.S_ISDIR(info.st_mode):
                        if entry.is_symlink():
                            indirect.add(member)
                        else:
                            pending.append(member)
    return inodes


def test_string_member_inventory_matches_the_path_object_inventory(tmp_path, monkeypatch):
    """Tracking members by normalized strings changes the cost, never the discovered identities."""
    private = tmp_path / "private"
    nested = private / "nested" / "deeper"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "tree").mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    linked = []
    for index, directory in enumerate((private, private / "nested", nested, outside / "tree")):
        shared = directory / f"shared-{index}"
        shared.write_text("synthetic private input")
        os.link(shared, runtime / f"alias-{index}")
        linked.append(shared)
        (directory / f"single-{index}").write_text("one name only")
    (private / "leaf-to-outside").symlink_to(outside / "tree" / "shared-3")
    (private / "unselected-dir").symlink_to(outside / "tree", target_is_directory=True)
    selected_dir = private / "nested" / "selected-dir"
    selected_dir.symlink_to(outside / "tree", target_is_directory=True)
    (private / "cycle").symlink_to(private, target_is_directory=True)
    file_root = private / "single-0"
    shapes = {
        "one root": [_record(private)],
        "overlapping roots": [_record(nested), _record(private), _record(private / "nested")],
        "file and live/frozen pair": [_record(file_root), ("pair", tmp_path / "absent-live", private)],
        "selected directory symlink": [_record(private), _record(selected_dir)],
        "linked file root": [_record(linked[1]), _record(linked[3])],
        "nothing present": [_record(tmp_path / "absent")],
    }
    argv = ["--ro-bind", str(runtime), str(tmp_path / "view")]
    discovered = set()
    for name, records in shapes.items():
        expected = _path_object_inventory(records)
        assert HS._private_hardlink_inodes(records) == expected, name
        discovered |= expected
        actual = HS._host_hardlink_surfaces(argv, records, set())
        with monkeypatch.context() as reference:
            reference.setattr(HS, "_private_hardlink_inodes", _path_object_inventory)
            assert actual == HS._host_hardlink_surfaces(argv, records, set()), name
    assert len(discovered) == len(linked)  # every shape class above found something to compare
    everything = [_record(private), _record(selected_dir), _record(outside)]
    assert HS._private_hardlink_inodes(everything) == {(path.stat().st_dev, path.stat().st_ino) for path in linked}
