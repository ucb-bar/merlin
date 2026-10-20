"""V3 composes complete nested metadata; old flat overlays keep their meaning."""

import copy
import json
import os
import shutil
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import readonly_runtime as R
from merlin_experiments.capture_execution import runtime_store as STORE
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution.sealed_static import _file_digest


def fixture(tmp_path):
    entry = tmp_path / "store/initial"
    site = "opt/capture-venv/lib/python3.12/site-packages"
    for name in ["opt/capture-venv/bin/python", site + "/retained.py", "opt/base/id"]:
        p = entry / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"unchanged original runtime")
    old = dict(
        base="/opt/base",
        system_libs=[],
        selected_trees={
            "venv": M._snapshot_tree(entry / "opt/capture-venv"),
            "base": M._snapshot_tree(entry / "opt/base"),
        },
    )
    target = entry.parent / STORE._identity(old)
    entry.rename(target)
    entry = target
    marker = entry.parent / (entry.name + ".complete.json")
    marker.write_text(json.dumps(dict(identity=entry.name, base=old["base"], system_libs=[])))
    source = tmp_path / "task-venv"
    shutil.copytree(entry / "opt/capture-venv", source)
    relative = [
        "pkg/__init__.py",
        "pkg-1.dist-info/METADATA",
        "pkg-1.dist-info/licenses/LICENSE",
        "pkg-1.dist-info/sboms/dependencies.json",
    ]
    overlays = {}
    for index, name in enumerate(relative):
        path = source / "lib/python3.12/site-packages" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("selected leaf " + str(index)).encode())
        overlays[site + "/" + name] = path
    plan = {**old, "selected_trees": {**old["selected_trees"], "venv": M._snapshot_tree(source)}}
    return plan, entry, marker, source, overlays


def select(tmp_path, *, version=3, limits=None):
    plan, entry, marker, source, overlays = fixture(tmp_path)
    selected = R.select(
        plan,
        entry=entry,
        marker_sha256=_file_digest(marker),
        destination=tmp_path,
        overlays=overlays,
        overlay_version=version,
        overlay_limits=limits or dict(max_leaves=4, max_total_bytes=1024),
    )
    return plan, entry, marker, source, overlays, selected


def test_complete_nested_namespace_stage_and_mount_are_explicit_and_readonly(tmp_path):
    plan, entry, marker, source, overlays, selected = select(tmp_path)
    assert selected["schema"] == R.NESTED_OVERLAY_SCHEMA
    assert selected["max_created_directories"] == len(selected["created_directories"]) == 4
    assert selected["source_cas_identity"] != selected["effective_runtime_identity"]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_nlink) for p in entry.rglob("*")}
    assert R.verify(plan, selected, tmp_path) == selected
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    R.stage(selected, runtime)
    assert all(p.stat().st_size == 0 for p in runtime.rglob("*") if p.is_file())
    mounts = R.mounts(selected, runtime)
    assert str(source) not in mounts
    assert all(str(p) in mounts for p in overlays.values())
    assert before == {p: (p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_nlink) for p in entry.rglob("*")}
    assert R._file_digest(marker) == selected["marker_sha256"]


def test_default_v2_still_refuses_nested_missing_directories(tmp_path):
    with pytest.raises(ValueError, match="existing direct CAS ancestor"):
        select(tmp_path, version=2)


@pytest.mark.parametrize("version", [True, 3.0, 1, 4, "3"])
def test_overlay_version_is_exact_and_explicit(tmp_path, version):
    plan, entry, marker, _, overlays = fixture(tmp_path)
    with pytest.raises(ValueError, match="version"):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays=overlays,
            overlay_version=version,
        )
    with pytest.raises(ValueError, match="version"):
        R.select(plan, entry=entry, marker_sha256=_file_digest(marker), destination=tmp_path, overlay_version=3)


def test_created_directories_have_independent_bound(tmp_path):
    plan, entry, marker, source, _ = fixture(tmp_path)
    leaf = source / "new/a/b/deep.py"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("bounded")
    with pytest.raises(ValueError, match="directory budget"):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays={"opt/capture-venv/new/a/b/deep.py": leaf},
            overlay_version=3,
            overlay_limits=dict(max_leaves=2, max_total_bytes=1024),
        )


@pytest.mark.parametrize(
    "mutation", ["traversal", "shadow", "file_ancestor", "symlink_source", "symlink_parent", "missing", "extra"]
)
def test_namespace_and_source_escape_refusals(tmp_path, mutation):
    plan, entry, marker, source, overlays = fixture(tmp_path)
    first = next(iter(overlays))
    if mutation == "traversal":
        overlays["opt/capture-venv/../escape"] = overlays.pop(first)
    elif mutation == "shadow":
        overlays["opt/capture-venv/bin/python"] = overlays.pop(first)
    elif mutation == "file_ancestor":
        overlays["opt/capture-venv/bin/python/extra"] = overlays.pop(first)
    elif mutation == "symlink_source":
        link = tmp_path / "indirect.py"
        link.symlink_to(overlays[first])
        overlays[first] = link
    elif mutation == "symlink_parent":
        parent = overlays[first].parent
        moved = parent.with_name("original-pkg")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    elif mutation == "missing":
        overlays.pop(first)
    else:
        extra = tmp_path / "extra.py"
        extra.write_text("unselected")
        overlays["opt/capture-venv/lib/python3.12/site-packages/extra.py"] = extra
    with pytest.raises(ValueError):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays=overlays,
            overlay_version=3,
            overlay_limits=dict(max_leaves=16, max_total_bytes=1024),
        )


@pytest.mark.parametrize("mutation", ["leaf", "parent_mode", "cas_leaf", "cas_directory"])
def test_mid_selection_mutations_cannot_return_an_authority(tmp_path, monkeypatch, mutation):
    plan, entry, marker, _, overlays = fixture(tmp_path)
    original = R._overlay_directories

    def mutate(selected):
        result = original(selected)
        if mutation == "leaf":
            next(iter(overlays.values())).write_bytes(b"changed")
        elif mutation == "parent_mode":
            next(iter(overlays.values())).parent.chmod(0o700)
        elif mutation == "cas_leaf":
            (entry / "opt/capture-venv/bin/python").write_bytes(b"changed CAS")
        else:
            (entry / "opt/capture-venv/lib/python3.12/site-packages/pkg").mkdir()
        return result

    monkeypatch.setattr(R, "_overlay_directories", mutate)
    with pytest.raises(ValueError, match="changed"):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays=overlays,
            overlay_version=3,
            overlay_limits=dict(max_leaves=4, max_total_bytes=1024),
        )


def test_reopening_refuses_extra_leaves_and_mutated_directory_cap(tmp_path):
    plan, _, _, _, _, selected = select(tmp_path)
    changed = copy.deepcopy(selected)
    changed["max_created_directories"] = 64
    with pytest.raises(ValueError, match="identity changed"):
        R.verify(plan, changed, tmp_path)
    changed = copy.deepcopy(selected)
    changed["overlays"].pop()
    with pytest.raises(ValueError):
        R.verify(plan, changed, tmp_path)


def test_shared_guest_parent_modes_must_agree(tmp_path):
    plan, entry, marker, source, overlays = fixture(tmp_path)
    extra = tmp_path / "foreign/pkg-1.dist-info/other"
    extra.parent.mkdir(parents=True)
    extra.parent.chmod(0o700)
    extra.write_text("different parent mode")
    overlays["opt/capture-venv/lib/python3.12/site-packages/pkg-1.dist-info/other"] = extra
    with pytest.raises(ValueError, match="modes disagree"):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays=overlays,
            overlay_version=3,
            overlay_limits=dict(max_leaves=8, max_total_bytes=1024),
        )


@pytest.mark.parametrize("during", [False, True])
def test_owned_parents_and_post_inventory_cas_ownership_are_required(tmp_path, monkeypatch, during):
    plan, entry, marker, _, overlays = fixture(tmp_path)
    target = (entry / "opt/capture-venv/bin/python") if during else next(iter(overlays.values())).parent
    original_stat, original_directories = Path.stat, R._overlay_directories
    active = not during

    def changed_owner(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if active and path == target:
            fields = list(result)
            fields[4] = os.getuid() + 1
            return os.stat_result(fields)
        return result

    def finish_inventory(selected):
        nonlocal active
        result = original_directories(selected)
        active = True
        return result

    monkeypatch.setattr(Path, "stat", changed_owner)
    if during:
        monkeypatch.setattr(R, "_overlay_directories", finish_inventory)
    with pytest.raises(ValueError, match="owned|ownership"):
        R.select(
            plan,
            entry=entry,
            marker_sha256=_file_digest(marker),
            destination=tmp_path,
            overlays=overlays,
            overlay_version=3,
            overlay_limits=dict(max_leaves=4, max_total_bytes=1024),
        )
