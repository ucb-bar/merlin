"""A selected public facts leaf cannot be withheld by a new bundle's host inputs."""

import hashlib
import json

import pytest

from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def facts_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    public = repo / "inputs"
    selected = public / "selected" / "facts.json"
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps({"facts": {"target": "neutral", "selection_marker": "selected"}}))
    private = public / "private" / "host.bin"
    private.parent.mkdir()
    private.write_bytes(b"private runtime input")
    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    bundle = {
        "allowed": [{"path": str(public) + "/"}, {"path": str(selected.parent) + "/"}],
        "host_inputs": [],
        "selected_rtl_facts_file": str(selected),
    }
    yield repo, workspace, bundle, selected, private
    BW.remove_bundle_snapshot(workspace)


@pytest.mark.parametrize("private_scope", ["leaf", "ancestor"])
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("use_store", [False, True], ids=["copied", "shared-store"])
def test_new_freeze_refuses_private_selected_facts_before_copying(
    facts_bundle, monkeypatch, tmp_path, private_scope, relative, use_store
):
    repo, workspace, bundle, selected, _ = facts_bundle
    private = selected if private_scope == "leaf" else selected.parent
    bundle["host_inputs"] = [{"path": str(private.relative_to(repo) if relative else private)}]
    store = tmp_path / "cas"
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(store) if use_store else "")

    def unexpected_copy(*args, **kwargs):
        pytest.fail("contradictory selected public facts reached snapshot copying")

    monkeypatch.setattr(BW.content_store, "place_file", unexpected_copy)
    monkeypatch.setattr(BW.content_store, "place_tree", unexpected_copy)
    with pytest.raises(RuntimeError, match="selected public RTL facts leaf is operator-private"):
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    root = BW.bundle_snapshot_root(workspace)
    assert not root.exists()
    assert not root.with_name(root.name + ".pending").exists()
    assert not store.exists()


@pytest.mark.parametrize("private_scope", ["none", "file", "directory"])
def test_public_selection_preserves_unrelated_private_descendants(facts_bundle, private_scope):
    repo, workspace, bundle, selected, private = facts_bundle
    if private_scope != "none":
        bundle["host_inputs"] = [{"path": str(private if private_scope == "file" else private.parent)}]
    original = selected.read_bytes()
    manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert BW.verify_bundle_snapshot(workspace, bundle, repo=repo) == manifest
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [selected], repo=repo)
    assert frozen.read_bytes() == original
    policy = BW.base_argv(workspace, bundle, repo=repo)
    assert BW.is_exposed(policy, selected)
    assert BW.is_exposed(policy, private) == (private_scope == "none")


@pytest.mark.parametrize("semantic_pin", [False, True])
def test_private_facts_without_public_selection_remain_valid(facts_bundle, semantic_pin):
    repo, workspace, bundle, selected, _ = facts_bundle
    bundle.pop("selected_rtl_facts_file")
    bundle["host_inputs"] = [{"path": str(selected)}]
    if semantic_pin:
        # Semantic byte identity alone is not a promise of author-public access.
        bundle["compiler_tool_profile"] = {
            "semantic_inputs": {
                "rtl_facts": {"path": str(selected), "sha256": hashlib.sha256(selected.read_bytes()).hexdigest()}
            }
        }
    manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert BW.verify_bundle_snapshot(workspace, bundle, repo=repo) == manifest
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [selected], repo=repo)
    assert frozen.read_bytes() == selected.read_bytes()
    assert not BW.is_exposed(BW.base_argv(workspace, bundle, repo=repo), selected)


def test_existing_conflicting_snapshot_keeps_its_identity_and_private_mask(facts_bundle, monkeypatch):
    repo, workspace, bundle, selected, _ = facts_bundle
    bundle["host_inputs"] = [{"path": str(selected)}]
    # Reproduce a historical freeze before the new preflight existed.
    with monkeypatch.context() as previous:
        previous.setattr(BW, "_validate_selected_public_facts", lambda *_: None)
        manifest = BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    identity = BW.snapshot_record(workspace)
    marker = BW.bundle_snapshot_root(workspace) / "snapshot.json"
    marker_bytes = marker.read_bytes()
    assert BW.verify_bundle_snapshot(workspace, bundle, repo=repo) == manifest
    assert BW.materialize_bundle_inputs(workspace, bundle, repo=repo) == manifest
    assert BW.snapshot_record(workspace) == identity
    assert marker.read_bytes() == marker_bytes
    assert not BW.is_exposed(BW.base_argv(workspace, bundle, repo=repo), selected)
