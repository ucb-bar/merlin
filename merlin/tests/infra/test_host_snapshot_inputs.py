"""Private host inputs are frozen without becoming candidate grants."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def private_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    hidden = repo / "inputs" / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "secret.bin").write_bytes(b"host only")
    (repo / "inputs" / "public.txt").write_text("public input")
    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    bundle = {"allowed": [{"path": "inputs"}], "host_inputs": [{"path": "inputs/hidden"}]}
    yield repo, workspace, bundle, hidden
    BW.remove_bundle_snapshot(workspace)


def test_private_inputs_are_frozen_but_never_granted(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert manifest["version"] == 4
    _, grants = BW._snapshot_grants(workspace, bundle, repo)
    assert [path for path, _, _ in grants] == ["inputs"]
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    assert (frozen / "secret.bin").read_bytes() == b"host only"
    assert not BW.bundle_snapshot_root(workspace).stat().st_mode & 0o077
    (hidden / "secret.bin").write_bytes(b"changed live input")
    assert BW.verify_bundle_snapshot(workspace, bundle, repo=repo) == manifest
    policy = BW.base_argv(workspace, bundle, repo=repo)
    assert BW.is_exposed(policy, repo / "inputs/public.txt")
    assert not BW.is_exposed(policy, hidden / "secret.bin")
    # The mask must matter: the same actual snapshot without the deny exposes it.
    unmasked = ["--ro-bind", str(frozen.parent), str(hidden.parent)]
    assert BW.is_exposed(unmasked, hidden / "secret.bin")
    assert "secret.bin" not in json.dumps(BW.snapshot_record(workspace))


def test_private_only_bundle_has_no_candidate_input_binds(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    bundle["allowed"] = []
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    _, grants = BW._snapshot_grants(workspace, bundle, repo)
    assert grants == []
    assert not BW.is_exposed(BW.base_argv(workspace, bundle, repo=repo), hidden / "secret.bin")
    assert BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)[0].is_dir()


@pytest.mark.parametrize("path", ["inputs/hidden", "inputs/hidden/secret.bin"])
def test_explicit_public_grant_cannot_reopen_private_subtree(private_bundle, path):
    repo, workspace, bundle, _ = private_bundle
    bundle["allowed"].append({"path": path})
    with pytest.raises(RuntimeError, match="overlaps a host-only"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "hardlink"])
def test_public_aliases_to_private_bytes_are_refused(private_bundle, kind):
    repo, workspace, bundle, hidden = private_bundle
    alias = repo / "inputs" / "public-alias"
    if kind == "hardlink":
        os.link(hidden / "secret.bin", alias)
    else:
        alias.symlink_to(hidden if kind == "directory_symlink" else hidden / "secret.bin")
    with pytest.raises(RuntimeError, match="alias to host-only"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


def test_indirect_directory_alias_cannot_copy_private_inputs(private_bundle, tmp_path):
    repo, workspace, bundle, hidden = private_bundle
    hop = tmp_path / "external-hop"
    hop.mkdir()
    (hop / "indirect").symlink_to(hidden, target_is_directory=True)
    (repo / "inputs/public-alias").symlink_to(hop, target_is_directory=True)
    with pytest.raises(RuntimeError, match="alias to host-only"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


def test_unrelated_directory_alias_is_still_supported(private_bundle, tmp_path):
    repo, workspace, bundle, hidden = private_bundle
    external = tmp_path / "public-headers"
    external.mkdir()
    (external / "header.h").write_text("public header")
    (repo / "inputs/headers").symlink_to(external, target_is_directory=True)
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [repo / "inputs/headers/header.h"], repo=repo)
    assert frozen.read_text() == "public header"


@pytest.mark.parametrize("relative", [".", "private", ".."])
def test_workspace_rebind_cannot_reexpose_host_inputs(private_bundle, relative):
    repo, workspace, bundle, _ = private_bundle
    private = (workspace / relative).resolve()
    private.mkdir(exist_ok=True)
    bundle["host_inputs"] = [{"path": str(private)}]
    with pytest.raises(RuntimeError, match="overlaps the writable workspace"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()


def test_resume_keeps_private_denial_after_source_disappears(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    (hidden / "secret.bin").unlink()
    hidden.rmdir()
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    assert (frozen / "secret.bin").read_bytes() == b"host only"
    assert not BW.is_exposed(BW.base_argv(workspace, bundle, repo=repo), hidden / "secret.bin")


def test_frozen_lookup_requires_declared_membership(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    outsider = repo / "undeclared"
    outsider.write_text("not frozen")
    with pytest.raises(RuntimeError, match="not declared"):
        BW.snapshot_input_paths(workspace, bundle, [outsider], repo=repo)


def test_host_declarations_cannot_be_added_to_legacy_snapshot(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    old = {"allowed": bundle["allowed"]}
    manifest = BW.materialize_bundle_inputs(workspace, old, repo=repo)
    # Historical decoder remains inspectable; new freezes never emit V2.
    manifest["version"] = 2
    for key in ("support_ownership", "host_inputs", "host_records"):
        manifest.pop(key)
    marker = BW.bundle_snapshot_root(workspace) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    assert manifest["version"] == 2
    assert "manifest_sha256" not in BW.snapshot_record(workspace)
    assert BW.verify_bundle_snapshot(workspace, old, repo=repo) == manifest
    with pytest.raises(RuntimeError, match="legacy snapshot"):
        BW.verify_bundle_snapshot(workspace, bundle, repo=repo)


def test_private_snapshot_record_pins_exact_manifest_bytes(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    marker = BW.bundle_snapshot_root(workspace) / "snapshot.json"
    record = BW.snapshot_record(workspace)
    assert record["manifest_sha256"] == hashlib.sha256(marker.read_bytes()).hexdigest()
    marker.chmod(0o600)
    marker.write_text(marker.read_text() + "\n")
    # Even semantically equivalent marker edits invalidate the arrival identity.
    assert BW.snapshot_record(workspace)["manifest_sha256"] != record["manifest_sha256"]


@pytest.mark.parametrize("scope", ["parent", "root", "file"])
def test_private_snapshot_marker_is_masked_through_runtime_alias(private_bundle, scope):
    repo, workspace, bundle, _ = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    snapshot = BW.bundle_snapshot_root(workspace)
    marker = snapshot / "snapshot.json"
    source = {"parent": snapshot.parent, "root": snapshot, "file": marker}[scope]
    alias = repo / "runtime-view"
    exposed = alias / marker.relative_to(source) if source != marker else alias
    argv = ["--ro-bind", str(source), str(alias)]
    assert BW.is_exposed(argv, exposed)
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    masked = BW.apply_answer_masks(argv, surfaces)
    assert not BW.is_exposed(masked, exposed)
    if scope == "root":
        assert BW.is_exposed(masked, alias / "repo/inputs/public.txt")


@pytest.mark.parametrize("scope", ["parent", "file", "hardlink"])
def test_host_provenance_files_are_masked_without_hiding_public_neighbors(tmp_path, scope):
    archive = tmp_path / "original-run"
    archive.mkdir()
    private = archive / "environment.yaml"
    private.write_text("hidden capsule identity")
    (archive / "public.txt").write_text("public input")
    source = archive if scope == "parent" else private
    if scope == "hardlink":
        source = tmp_path / "runtime-metadata-alias"
        os.link(private, source)
    destination = tmp_path / "runtime-view"
    exposed = destination / private.name if scope == "parent" else destination
    argv = ["--ro-bind", str(source), str(destination)]
    assert BW.is_exposed(argv, exposed)
    surfaces = BW.private_file_surfaces(argv, [private])
    masked = BW.apply_answer_masks(argv, surfaces)
    assert not BW.is_exposed(masked, exposed)
    if scope == "parent":
        assert BW.is_exposed(masked, destination / "public.txt")


@pytest.mark.parametrize("kind", ["absent", "directory", "symlink"])
def test_host_provenance_file_mask_refuses_invalid_sources(tmp_path, kind):
    source = tmp_path / "provenance"
    if kind == "directory":
        source.mkdir()
    elif kind == "symlink":
        referent = tmp_path / "referent"
        referent.write_text("metadata")
        source.symlink_to(referent)
    with pytest.raises(RuntimeError, match="private provenance file cannot be classified"):
        BW.private_file_surfaces([], [source])


def test_changed_or_removed_private_declaration_refuses_resume(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    with pytest.raises(RuntimeError, match="host-only input set"):
        BW.verify_bundle_snapshot(workspace, {"allowed": bundle["allowed"]}, repo=repo)


def test_snapshot_record_cannot_redirect_public_grant_to_private_bytes(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    manifest["grants"][0]["snapshot"] = manifest["host_records"][0]["snapshot"]
    marker = BW.bundle_snapshot_root(workspace) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="does not match its declared destination"):
        BW.verify_bundle_snapshot(workspace, bundle, repo=repo)


def test_private_marker_cannot_redirect_denial_to_an_alternate_spelling(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    alternate = repo / "merlin/inputs/hidden"
    alternate.mkdir(parents=True)
    (alternate / "secret.bin").write_text("other content")
    bundle["allowed"] = [{"path": "."}]
    manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    manifest["host_records"][0].update(destination=str(alternate), snapshot="repo/merlin/inputs/hidden")
    marker = BW.bundle_snapshot_root(workspace) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="destination is invalid"):
        BW.verify_bundle_snapshot(workspace, bundle, repo=repo)
    with pytest.raises(RuntimeError, match="destination is invalid"):
        BW.base_argv(workspace, bundle, repo=repo)


def test_host_input_cannot_select_a_shorthand_fallback(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    alternate = repo / "merlin/inputs"
    alternate.parent.mkdir()
    (repo / "inputs").rename(alternate)
    with pytest.raises(RuntimeError, match="canonical paths"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)


def test_clean_room_excludes_host_inputs_and_detects_an_injected_copy(private_bundle, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from merlin.targetgen.sandbox import cleanroom as CR

    repo, _, bundle, hidden = private_bundle
    monkeypatch.setattr(CR, "answer_surfaces", lambda te: [])
    descriptor = SimpleNamespace(target="fixture-target")
    home = tmp_path / "clean-run"
    clean = CR.build_clean_room(descriptor, home, bundle, repo=repo)
    try:
        assert clean.verdict.ok
        assert not (clean.inputs / "repo/inputs/hidden").exists()
        assert (clean.inputs / "repo/inputs/public.txt").read_text() == "public input"
        (clean.work / "injected.bin").write_bytes((hidden / "secret.bin").read_bytes())
        assert not CR.verify_clean_room(clean.root, descriptor, bundle=bundle, repo=repo).ok
    finally:
        CR.remove_clean_room(home)


@pytest.mark.parametrize("location", ["live", "frozen"])
@pytest.mark.parametrize("scope", ["parent", "root", "file"])
def test_final_toolchain_alias_cannot_expose_private_inputs(private_bundle, monkeypatch, location, scope):
    from types import SimpleNamespace

    from merlin.targetgen.sandbox import toolchain as TC

    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    source = hidden if location == "live" else frozen
    tail = Path("secret.bin")
    if scope == "parent":
        tail = Path(source.name) / tail
        source = source.parent
    elif scope == "file":
        source = source / "secret.bin"
        tail = Path(".")
    alias = repo / "runtime-alias"
    extra = ["--ro-bind", str(source), str(alias)]
    monkeypatch.setattr(BW, "repo_root", lambda: repo)
    monkeypatch.setattr(BW, "claude_runtime_binds", lambda: [])
    monkeypatch.setattr(TC, "toolchain_binds", lambda te: extra)
    monkeypatch.setattr(BW, "answer_surfaces", lambda te: [])
    assert BW.is_exposed(extra, alias / tail)  # Negative control uses actual source bytes.
    argv = BW.full_argv(SimpleNamespace(target="fixture-target"), workspace, bundle)
    assert not BW.is_exposed(argv, alias / tail)
    private = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert BW.coverage_gap(argv, private) == []


def test_private_frozen_payload_mutation_refuses_lookup(private_bundle):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    payload = frozen / "secret.bin"
    payload.chmod(0o600)
    payload.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="content verification failed"):
        BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)


@pytest.mark.parametrize("location", ["live", "frozen"])
@pytest.mark.parametrize("scope", ["directory", "file"])
def test_late_runtime_hardlink_alias_is_masked(private_bundle, tmp_path, location, scope):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link((hidden if location == "live" else frozen) / "secret.bin", runtime / "alias.bin")
    source = runtime if scope == "directory" else runtime / "alias.bin"
    destination = repo / "runtime-view"
    argv = ["--ro-bind", str(source), str(destination)]
    exposed = destination / "alias.bin" if scope == "directory" else destination
    assert BW.is_exposed(argv, exposed)
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert not BW.is_exposed(BW.apply_answer_masks(argv, surfaces), exposed)


def test_private_intersecting_root_never_enters_shared_cas(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    public = repo / "inputs/public.txt"
    public.write_bytes((hidden / "secret.bin").read_bytes())
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    frozen_public, frozen_private = BW.snapshot_input_paths(
        workspace, bundle, [public, hidden / "secret.bin"], repo=repo
    )
    assert frozen_public.stat().st_ino != frozen_private.stat().st_ino
    assert frozen_private.stat().st_nlink == 1
    assert frozen_public.stat().st_nlink == 1
    assert not list((tmp_path / "cas").rglob("*"))
    argv = BW._bundle_mount_args(workspace, bundle, repo)
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    masked = BW.apply_answer_masks(argv, surfaces)
    assert BW.is_exposed(masked, public)
    assert not BW.is_exposed(masked, hidden / "secret.bin")
    # Identical independently owned public bytes are not a private hardlink.
    alias = repo / "runtime-alias"
    argv += ["--ro-bind", str(frozen_public), str(alias)]
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert BW.is_exposed(BW.apply_answer_masks(argv, surfaces), alias)


def test_verified_relocated_public_grant_preserves_cas_dedup(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    public = repo / "inputs/public.txt"
    public.write_bytes((hidden / "secret.bin").read_bytes())
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [repo / "inputs"], repo=repo)
    # Compatibility with an already-frozen v3 snapshot from before private CAS
    # isolation: equivalent bytes may still share an inode in an existing run.
    private_file = frozen / "hidden/secret.bin"
    private_file.parent.chmod(0o700)
    private_file.unlink()
    os.link(frozen / "public.txt", private_file)
    private_file.parent.chmod(0o500)
    BW.verify_bundle_snapshot(workspace, bundle, repo=repo)
    relocated = tmp_path / "relocated"
    destination = relocated / "inputs"
    argv = ["--ro-bind", str(frozen), str(destination)]
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo, grant_repo=relocated)
    masked = BW.apply_answer_masks(argv, surfaces)
    assert BW.is_exposed(masked, destination / "public.txt")
    assert not BW.is_exposed(masked, destination / "hidden/secret.bin")
    # A second, undeclared destination is not covered by the verified relocation.
    alias = tmp_path / "arbitrary-alias"
    argv += ["--ro-bind", str(frozen / "public.txt"), str(alias)]
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo, grant_repo=relocated)
    assert not BW.is_exposed(BW.apply_answer_masks(argv, surfaces), alias)


def test_private_copy_retains_cas_for_disjoint_public_roots(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    public = repo / "inputs/public.txt"
    public.write_bytes((hidden / "secret.bin").read_bytes())
    bundle["allowed"] = [{"path": "inputs/public.txt"}]
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    frozen_public, frozen_private = BW.snapshot_input_paths(
        workspace, bundle, [public, hidden / "secret.bin"], repo=repo
    )
    assert frozen_public.stat().st_nlink > 1
    assert frozen_private.stat().st_nlink == 1
    assert frozen_public.stat().st_ino != frozen_private.stat().st_ino
    assert frozen_public.read_bytes() == frozen_private.read_bytes()


def test_every_snapshot_directory_is_readonly(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    snapshot = BW.bundle_snapshot_root(workspace)
    assert all(not path.stat().st_mode & 0o222 for path in [snapshot, *snapshot.rglob("*")] if path.is_dir())


def test_remove_readonly_nested_snapshot_directories(private_bundle):
    repo, workspace, bundle, _ = private_bundle
    task = repo / "inputs/task/nested"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("public task")
    task.chmod(0o500)
    task.parent.chmod(0o500)
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [task.parent], repo=repo)
    assert not frozen.stat().st_mode & 0o200
    # Directories naturally have multiple links, but are never shared CAS files.
    assert frozen.stat().st_nlink > 1
    BW.remove_bundle_snapshot(workspace)
    assert not BW.bundle_snapshot_root(workspace).exists()


def test_runtime_hardlink_scan_does_not_follow_directory_symlinks(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(hidden / "secret.bin", runtime / "alias.bin")
    (runtime / "cycle").symlink_to(runtime, target_is_directory=True)
    scans = []
    original_scandir = BW.os.scandir

    def record_scan(path):
        if not isinstance(path, int):
            scans.append(Path(path))
        return original_scandir(path)

    monkeypatch.setattr(BW.os, "scandir", record_scan)
    argv = ["--ro-bind", str(runtime), str(repo / "runtime-view")]
    BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert runtime in scans
    assert runtime / "cycle" not in scans


def test_unreadable_runtime_hardlink_scan_refuses(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.link(hidden / "secret.bin", runtime / "alias.bin")
    original_scandir = BW.os.scandir

    def denied_scan(path):
        if not isinstance(path, int) and Path(path) == runtime:
            raise PermissionError("synthetic unreadable runtime")
        return original_scandir(path)

    monkeypatch.setattr(BW.os, "scandir", denied_scan)
    argv = ["--ro-bind", str(runtime), str(repo / "runtime-view")]
    with pytest.raises(RuntimeError, match="hardlink privacy cannot be verified"):
        BW.host_input_surfaces(argv, workspace, bundle, repo=repo)


def test_nested_private_device_mount_is_scanned_beneath_disjoint_root(private_bundle, tmp_path, monkeypatch):
    repo, workspace, bundle, hidden = private_bundle
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    runtime = tmp_path / "runtime"
    mounted = runtime / "nested-mount"
    mounted.mkdir(parents=True)
    os.link(hidden / "secret.bin", mounted / "alias.bin")
    device = mounted.stat().st_dev
    real_stat = Path.stat
    real_scandir = os.scandir

    def foreign_root(path, *args, **kwargs):
        info = real_stat(path, *args, **kwargs)
        if path == runtime:
            fields = list(info)
            fields[2] = device + 1
            return os.stat_result(fields)
        return info

    def scan_only_relevant(path):
        assert path != runtime, "disjoint parent need not be traversed"
        return real_scandir(path)

    monkeypatch.setattr(Path, "stat", foreign_root)
    monkeypatch.setattr(BW, "_host_mount_devices", lambda: [(device, mounted)])
    monkeypatch.setattr(BW.os, "scandir", scan_only_relevant)
    destination = repo / "runtime-view"
    argv = ["--ro-bind", str(runtime), str(destination)]
    surfaces = BW.host_input_surfaces(argv, workspace, bundle, repo=repo)
    assert not BW.is_exposed(BW.apply_answer_masks(argv, surfaces), destination / "nested-mount/alias.bin")


def test_mount_topology_decodes_escaped_paths(monkeypatch):
    mountinfo = r"10 1 8:2 / /space\040tab\011line\012literal\134040 rw - ext4 /dev/sda2 rw"
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: mountinfo)
    assert BW._host_mount_devices() == [(os.makedev(8, 2), Path("/space tab\tline\nliteral\\040"))]


@pytest.mark.parametrize("mountinfo", ["", "broken", "1 0 8:2 / /bad\\099 rw - ext4 /dev/sda2 rw"])
def test_malformed_mount_topology_refuses(monkeypatch, mountinfo):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: mountinfo)
    with pytest.raises(RuntimeError, match="mount topology cannot be verified"):
        BW._host_mount_devices()


def test_unavailable_mount_topology_refuses(monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError("synthetic proc restriction")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(RuntimeError, match="mount topology cannot be verified"):
        BW._host_mount_devices()


@pytest.mark.parametrize("entries", [None, {}, ["private"], [{"path": ""}], [{"path": "../private"}]])
def test_malformed_host_input_declarations_fail_closed(private_bundle, entries):
    repo, workspace, bundle, _ = private_bundle
    bundle["host_inputs"] = entries
    with pytest.raises(RuntimeError, match="host_inputs"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(workspace).exists()
