"""Closed private root records preserve full pins and frozen descendant custody."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from merlin.common import bundle_snapshot as Core
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import host_surfaces as HS
from merlin.targetgen.tool_profile_toolchain import attach_selected_toolchain


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    library = repo / "library"
    (library / "nested").mkdir(parents=True)
    members = [library / "First.olean", library / "nested/Second.lean"]
    executable, solver = repo / "lean", repo / "cadical"
    for index, path in enumerate([*members, executable, solver]):
        path.write_bytes(f"original selected member {index}".encode())

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    document = {"schema": "merlin.lean_toolchain.v1", "executable": pin(executable),
                "files": [pin(path) for path in [*members, solver]], "library_roots": [str(library)],
                "sat_solver": pin(solver)}
    manifest = repo / "toolchain.json"
    manifest.write_text(json.dumps(document))
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    bundle = {"allowed": [], "denied": [], "host_inputs": []}
    yield repo, workspace, bundle, library, members, manifest, document
    BW.remove_bundle_snapshot(workspace)


def _attach(fixture, *, tools=()):
    _, _, bundle, _, _, manifest, _ = fixture
    attach_selected_toolchain(bundle, tools, manifest)
    return bundle


def _grouped(bundle):
    return [row for row in bundle["host_inputs"] if "closed_toolchain_root" in row]


def test_exact_private_tree_groups_but_retains_original_manifest_and_external_leaves(selected):
    repo, _, _, library, members, manifest, document = selected
    before = manifest.read_bytes()
    bundle = _attach(selected)
    assert [row["path"] for row in _grouped(bundle)] == [str(library)]
    assert {row["path"] for row in bundle["host_inputs"]} == {
        str(library), str(manifest), str(repo / "lean"), str(repo / "cadical")
    }
    assert not bundle["allowed"]
    assert manifest.read_bytes() == before
    assert json.loads(before)["files"] == document["files"]
    assert not any(row["path"] == str(path) for row in bundle["host_inputs"] for path in members)


@pytest.mark.parametrize("extra", ["file", "symlink", "fifo"])
def test_nonclosed_parent_retains_its_leaf_and_groups_only_closed_descendants(selected, extra):
    _, _, _, library, members, _, _ = selected
    addition = library / "unselected"
    if extra == "file":
        addition.write_bytes(b"not selected")
    elif extra == "symlink":
        addition.symlink_to(members[0])
    else:
        import os

        os.mkfifo(addition)
    bundle = _attach(selected)
    assert [row["path"] for row in _grouped(bundle)] == [str(library / "nested")]
    assert str(members[0]) in {row["path"] for row in bundle["host_inputs"]}
    assert str(addition) not in {row["path"] for row in bundle["host_inputs"]}


@pytest.mark.parametrize("conflict", ["public", "alias", "host", "unknown"])
def test_existing_source_overlap_or_unknown_resolution_retains_leaves(selected, conflict):
    repo, _, bundle, library, _, _, _ = selected
    if conflict == "alias":
        public = repo / "public-alias"
        public.symlink_to(library, target_is_directory=True)
    else:
        public = library if conflict != "unknown" else repo / "missing-source"
    bundle["host_inputs" if conflict == "host" else "allowed"].append({"path": str(public)})
    _attach(selected)
    assert not _grouped(bundle)


def test_authoring_level_keeps_all_exact_file_grants_and_no_private_root(selected):
    _, _, _, library, members, _, _ = selected
    bundle = _attach(selected, tools=("lean_proof_authoring",))
    assert not bundle["host_inputs"]
    assert str(library) not in {row["path"] for row in bundle["allowed"]}
    assert {str(path) for path in members} <= {row["path"] for row in bundle["allowed"]}


def test_frozen_descendants_reopen_same_bytes_and_runtime_alias_is_private(selected):
    repo, workspace, _, library, members, _, _ = selected
    bundle = _attach(selected)
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    legacy = BW.snapshot_input_paths(workspace, bundle, members, repo=repo)
    core = Core.snapshot_input_paths(workspace, bundle, members, repo=repo)
    assert core == legacy
    assert [path.read_bytes() for path in core] == [path.read_bytes() for path in members]
    alias = repo / "runtime-alias"
    argv = ["--ro-bind", str(library), str(alias)]
    assert BW.is_exposed(argv, alias / members[0].relative_to(library))
    private = HS.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert not BW.is_exposed(BW.apply_answer_masks(argv, private), alias / members[0].relative_to(library))
    # Frozen inputs remain independent of subsequent changes to live sources.
    members[0].write_bytes(b"later live edit")
    assert BW.snapshot_input_paths(workspace, bundle, members, repo=repo)[0].read_bytes() != b"later live edit"


@pytest.mark.parametrize("mutation", ["extra", "bytes", "manifest"])
def test_mutation_after_declaration_refuses_before_copy(selected, mutation):
    repo, workspace, _, library, members, manifest, document = selected
    bundle = _attach(selected)
    if mutation == "extra":
        (library / "extra.txt").write_bytes(b"unselected")
    elif mutation == "bytes":
        members[0].write_bytes(b"changed")
    else:
        document["dependency_revision"] = "changed"
        manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="private toolchain"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


@pytest.mark.parametrize("mutation", ["extra", "bytes"])
def test_actual_frozen_copy_is_checked_before_sealing(selected, monkeypatch, mutation):
    repo, workspace, _, library, members, _, _ = selected
    bundle = _attach(selected)
    original = BW.content_store.place_tree

    def copy_tree(source, target, *args, **kwargs):
        result = original(source, target, *args, **kwargs)
        if source == library:
            if mutation == "extra":
                target.chmod(0o700)
                (target / "extra").write_bytes(b"unselected")
            else:
                changed = target / members[0].relative_to(library)
                changed.chmod(0o600)
                changed.write_bytes(b"wrong frozen bytes")
        return result

    monkeypatch.setattr(BW.content_store, "place_tree", copy_tree)
    with pytest.raises(ValueError, match="private toolchain"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


def test_marked_root_cannot_select_an_unrelated_directory(selected):
    repo, workspace, _, _, _, _, _ = selected
    bundle = copy.deepcopy(_attach(selected))
    _grouped(bundle)[0]["path"] = str(repo)
    with pytest.raises(ValueError, match="outside.*library roots"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)


def test_manifest_cannot_change_between_initial_read_and_private_selection(selected, monkeypatch):
    from merlin.targetgen import tool_profile_toolchain as T

    _, _, _, _, _, manifest, document = selected
    original = T._private_root_records

    def changed(*args):
        result = original(*args)
        document["timeout_s"] = 999
        manifest.write_text(json.dumps(document))
        return result

    monkeypatch.setattr(T, "_private_root_records", changed)
    with pytest.raises(ValueError, match="manifest changed during"):
        _attach(selected)


def test_maximal_subtrees_are_disjoint_and_each_directory_is_scanned_once(selected, monkeypatch):
    from collections import Counter

    from merlin.targetgen import tool_profile_toolchain as T

    _, _, _, library, _, manifest, document = selected
    (library / "build.trace").write_bytes(b"unselected build metadata")
    nested = library / "nested"
    for parent in (nested, nested / "deeper", library / "other"):
        parent.mkdir(exist_ok=True)
        path = parent / "Selected.olean"
        path.write_bytes(b"selected")
        document["files"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(document))
    scanned = Counter()
    original = T.os.scandir

    def scan(path):
        scanned[path] += 1
        return original(path)

    monkeypatch.setattr(T.os, "scandir", scan)
    bundle = _attach(selected)
    assert [row["path"] for row in _grouped(bundle)] == [str(nested), str(library / "other")]
    assert set(scanned) == {library, nested, nested / "deeper", library / "other"}
    assert set(scanned.values()) == {1}


def test_partial_root_freeze_never_copies_unpinned_parent_files(selected):
    repo, workspace, _, library, members, _, _ = selected
    (library / "build.trace").write_bytes(b"unselected build metadata")
    bundle = _attach(selected)
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    selected_copy = Core.snapshot_input_paths(workspace, bundle, members, repo=repo)
    assert [path.read_bytes() for path in selected_copy] == [path.read_bytes() for path in members]
    assert not (selected_copy[0].parent / "build.trace").exists()


def test_closed_descendant_mutation_after_selection_refuses(selected):
    repo, workspace, _, library, _, _, _ = selected
    (library / "build.trace").write_bytes(b"unselected build metadata")
    bundle = _attach(selected)
    assert [row["path"] for row in _grouped(bundle)] == [str(library / "nested")]
    (library / "nested/extra").write_bytes(b"new member")
    with pytest.raises(ValueError, match="membership changed"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
