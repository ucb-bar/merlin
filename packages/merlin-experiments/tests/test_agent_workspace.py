"""Snapshot membership and link refusal before sandbox mount construction."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as W


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))
    root = tmp_path / "snapshot"
    (root / "capsule").mkdir(parents=True)
    payload = b"public input"
    (root / "capsule/input.mlir").write_bytes(payload)
    row = {"path": "capsule/input.mlir", "sha256": hashlib.sha256(payload).hexdigest(), "n_bytes": len(payload)}
    digest = hashlib.sha256(f"{row['path']}\0{row['sha256']}\0{len(payload)}\n".encode()).hexdigest()
    manifest = root / "agent_input_manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "files": [row], "content_sha256": digest, "n_files": 1, "n_bytes": len(payload)}
        )
    )
    return W.AgentInputSnapshot(
        root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), digest, 1, len(payload)
    )


def test_valid_snapshot(snapshot):
    W.verify_answer_free_agent_inputs(snapshot)


@pytest.mark.parametrize(
    "mutation", ["added", "root_link", "parent_link", "manifest_link", "leaf_link", "missing", "changed"]
)
def test_snapshot_mutations_refused(snapshot, mutation, tmp_path):
    leaf = snapshot.root / "capsule/input.mlir"
    if mutation == "added":
        (snapshot.root / "answer.txt").write_text("unrecorded")
    elif mutation in {"root_link", "parent_link", "manifest_link", "leaf_link"}:
        original = {
            "root_link": snapshot.root,
            "parent_link": leaf.parent,
            "manifest_link": snapshot.manifest_path,
            "leaf_link": leaf,
        }[mutation]
        moved = tmp_path / "replacement"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=moved.is_dir())
    elif mutation == "missing":
        leaf.unlink()
    else:
        leaf.write_text("changed")
    with pytest.raises(W.StageGateError):
        W.verify_answer_free_agent_inputs(snapshot)


@pytest.mark.parametrize("policy", ["inner", "outer"])
@pytest.mark.parametrize("added", [False, True])
def test_policy_verifies_before_mount_construction(snapshot, tmp_path, monkeypatch, policy, added):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if added:
        (snapshot.root / "answer.txt").write_text("unrecorded")

    def reached(*args, **kwargs):
        raise AssertionError("mount construction reached")

    monkeypatch.setattr(W.BW, "base_argv", reached)
    selected = W.CAMPAIGN.PackageSandboxInputs(
        W.TC.ToolchainPaths(tmp_path, *[str(tmp_path / name) for name in ("venv", "llvm", "compat", "clang", "uv")]),
        W.TC.SimToolchain(),
        "",
        (),
    )
    expected = W.StageGateError if added else AssertionError
    with pytest.raises(expected):
        if policy == "inner":
            W.inner_execution_policy(SimpleNamespace(), workspace, snapshot, inputs=selected)
        else:
            W.outer_codex_policy(workspace, snapshot, (), SimpleNamespace(), inputs=selected)


def test_manifest_is_parsed_from_hashed_bytes(snapshot, monkeypatch):
    original = Path.read_text

    def no_second_manifest_read(path, *args, **kwargs):
        if path == snapshot.manifest_path:
            pytest.fail("manifest reopened after hashing")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_second_manifest_read)
    W.verify_answer_free_agent_inputs(snapshot)
