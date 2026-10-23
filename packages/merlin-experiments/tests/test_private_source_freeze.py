"""Frozen operator inputs retain their reviewed source ownership after authoring."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_source_freeze as freeze


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    software = tmp_path / "authored" / "software.yaml"
    host = tmp_path / "authored" / "host.yaml"
    software.parent.mkdir()
    software.write_bytes(b"reviewed software\n")
    host.write_bytes(b"reviewed host\n")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    selection = b'{"target":"fixture"}\n'
    (evidence / "software").mkdir()
    (evidence / "software" / "selection.json").write_bytes(selection)
    sources = []
    artifacts = {"software/selection.json": {"sha256": _sha(selection), "size_bytes": len(selection)}}
    for index, (path, role) in enumerate(((software, "software-spec"), (host, "host-capability-spec:profile"))):
        raw = path.read_bytes()
        member = f"software/source-snapshots/{index:04d}-{_sha(raw)}.bin"
        destination = evidence / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        artifacts[member] = {"sha256": _sha(raw), "size_bytes": len(raw)}
        sources.append({"source": str(path), "role": role, "path": member, "sha256": _sha(raw), "size_bytes": len(raw)})
    manifest = {
        "schema": "phase0_evidence_v1",
        "target": "fixture",
        "raw_facts_sha256": None,
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest_path = evidence / "evidence-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    spec = tmp_path / "private.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.phase1.private_full_models.v1",
                "target": "fixture",
                "models": [
                    {
                        "id": "model",
                        "recipe_derivation_root": str(evidence),
                        "recipe_derivation_manifest_sha256": _sha(manifest_path.read_bytes()),
                        "software_spec": str(software),
                        "software_spec_sha256": _sha(software.read_bytes()),
                        "host_capabilities": str(host),
                        "host_capabilities_sha256": _sha(host.read_bytes()),
                    }
                ],
            }
        )
    )
    return spec, software, host, manifest_path


def test_exact_archived_sources_survive_authored_changes_and_resume(tmp_path: Path):
    spec, software, host, _ = _fixture(tmp_path)
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    record = freeze.stage(spec, root, target="fixture")
    software.write_bytes(b"new software\n")
    host.write_bytes(b"new host\n")
    resolved = freeze.verify(spec, record, root=root, target="fixture")
    assert resolved[(str(software), "software_spec")].read_bytes() == b"reviewed software\n"
    assert resolved[(str(host), "host_capabilities")].read_bytes() == b"reviewed host\n"
    assert freeze.verify(spec, record, root=root, target="fixture") == resolved


def test_two_owned_derivations_share_exact_authored_bytes(tmp_path: Path):
    spec, software, host, manifest_path = _fixture(tmp_path)
    second_evidence = tmp_path / "second-evidence"
    shutil.copytree(manifest_path.parent, second_evidence)
    document = yaml.safe_load(spec.read_text())
    second_model = dict(document["models"][0])
    second_model["id"] = "second-model"
    second_model["recipe_derivation_root"] = str(second_evidence)
    document["models"].append(second_model)
    spec.write_text(yaml.safe_dump(document))
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    record = freeze.stage(spec, root, target="fixture")
    assert {row["original_root"] for row in record["archives"]} == {str(manifest_path.parent), str(second_evidence)}
    assert len(record["sources"]) == 4
    assert len({row["frozen_path"] for row in record["sources"]}) == 2
    assert freeze.verify(spec, record, root=root, target="fixture") == {
        (str(software), "software_spec"): root / f"software_spec-{_sha(software.read_bytes())}.bin",
        (str(host), "host_capabilities"): root / f"host_capabilities-{_sha(host.read_bytes())}.bin",
    }
    document["models"][1]["software_spec_sha256"] = "0" * 64
    spec.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="conflicting model byte pins"):
        freeze.stage(spec, tmp_path / "another-run" / "sources", target="fixture")


@pytest.mark.parametrize("mutation", ["wrong_role", "wrong_source", "wrong_digest", "wrong_manifest", "wrong_mapping"])
def test_frozen_source_rejects_changed_ownership(tmp_path: Path, mutation: str):
    spec, software, host, manifest_path = _fixture(tmp_path)
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    if mutation in {"wrong_role", "wrong_source", "wrong_digest"}:
        document = json.loads(manifest_path.read_text())
        source = document["sources"][0]
        if mutation == "wrong_role":
            source["role"] = "unrelated"
        elif mutation == "wrong_source":
            source["source"] = str(host)
        else:
            source["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(document))
        private = yaml.safe_load(spec.read_text())
        private["models"][0]["recipe_derivation_manifest_sha256"] = _sha(manifest_path.read_bytes())
        spec.write_text(yaml.safe_dump(private))
        with pytest.raises(ValueError):
            freeze.stage(spec, root, target="fixture")
        return
    record = freeze.stage(spec, root, target="fixture")
    if mutation == "wrong_manifest":
        frozen = Path(record["archives"][0]["frozen_path"])
        frozen.chmod(0o600)
        frozen.write_bytes(b"changed\n")
    else:
        record["sources"][0]["frozen_path"] = str(Path(record["sources"][1]["frozen_path"]))
    with pytest.raises(ValueError):
        freeze.verify(spec, record, root=root, target="fixture")


def test_failed_partial_stage_cannot_be_reused_as_a_frozen_source(tmp_path: Path, monkeypatch):
    spec, _, _, _ = _fixture(tmp_path)
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    original_copy = freeze.copy_input
    copied = 0

    def fail_after_manifest(*args, **kwargs):
        nonlocal copied
        copied += 1
        if copied == 2:
            raise RuntimeError("interrupted private copy")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(freeze, "copy_input", fail_after_manifest)
    with pytest.raises(RuntimeError, match="interrupted"):
        freeze.stage(spec, root, target="fixture")
    monkeypatch.setattr(freeze, "copy_input", original_copy)
    assert root.is_dir()
    with pytest.raises(ValueError, match="already exists"):
        freeze.stage(spec, root, target="fixture")


def test_linked_build_cannot_publish_after_frozen_manifest_changes(tmp_path: Path):
    spec, _, _, _ = _fixture(tmp_path)
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    record = freeze.stage(spec, root, target="fixture")
    assert freeze.valid_binding(
        freeze.claim_binding(spec, record, root, target="fixture")["authored_source_freeze"], _sha(spec.read_bytes())
    )
    # Model build runs between the initial selection and final report publication.
    manifest = Path(record["archives"][0]["frozen_path"])
    manifest.chmod(0o600)
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        freeze.claim_binding(spec, record, root, target="fixture")


@pytest.mark.parametrize("alias_target", ["authored_original", "frozen_copy"])
def test_public_parent_grant_cannot_freeze_private_hardlink_alias(tmp_path: Path, alias_target: str):
    from merlin.targetgen.sandbox import bwrap

    spec, software, host, _ = _fixture(tmp_path)
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    record = freeze.stage(spec, root, target="fixture")
    frozen_software = next(row for row in record["sources"] if row["field"] == "software_spec")
    selected = software if alias_target == "authored_original" else Path(frozen_software["frozen_path"])
    os.link(selected, tmp_path / "public-hardlink.yaml")
    bundle = {
        "allowed": [{"path": str(tmp_path)}],
        "host_inputs": [{"path": str(root)}],
        "private_validation_paths": [
            {"path": str(software), "kind": "file"},
            {"path": str(host), "kind": "file"},
            {"path": str(root), "kind": "dir"},
        ],
    }
    workspace = tmp_path.parent / f"{tmp_path.name}-workspace"
    with pytest.raises(RuntimeError, match="hardlink alias"):
        bwrap.materialize_bundle_inputs(workspace, bundle, repo=tmp_path)
    assert not bwrap.bundle_snapshot_root(workspace).exists()


def test_public_parent_with_direct_private_child_never_uses_shared_cas(tmp_path: Path, monkeypatch):
    from merlin.targetgen.sandbox import bwrap, host_surfaces

    spec, software, host, _ = _fixture(tmp_path)
    public = tmp_path / "public.txt"
    public.write_text("public control\n")
    root = tmp_path / "run" / "private_full_model_input" / "sources"
    freeze.stage(spec, root, target="fixture")
    bundle = {
        "allowed": [{"path": str(tmp_path)}],
        "host_inputs": [{"path": str(root)}],
        "private_validation_paths": [
            {"path": str(software), "kind": "file"},
            {"path": str(host), "kind": "file"},
            {"path": str(root), "kind": "dir"},
        ],
    }
    observed_store = []
    original_place_tree = bwrap.content_store.place_tree

    def observe_place_tree(source, destination, store, **kwargs):
        observed_store.append(store)
        return original_place_tree(source, destination, store, **kwargs)

    monkeypatch.setattr(bwrap.content_store, "place_tree", observe_place_tree)
    with TemporaryDirectory(prefix="bwrap-private-test-") as work:
        workspace = Path(work) / "workspace"
        bwrap.materialize_bundle_inputs(workspace, bundle, repo=tmp_path)
        workspace.mkdir()
        assert observed_store == [None]
        [frozen_parent] = bwrap.snapshot_input_paths(workspace, bundle, [tmp_path], repo=tmp_path)
        frozen_software = frozen_parent / software.relative_to(tmp_path)
        surfaces = host_surfaces.host_input_surfaces(
            ["--ro-bind", str(workspace), str(workspace)], workspace, bundle, repo=tmp_path
        )
        assert frozen_software in {surface.path for surface in surfaces}
        if shutil.which("bwrap"):
            argv = bwrap.base_argv(
                workspace, bundle, repo=tmp_path, include_claude_home=False, inherit_environment=False
            )
            argv = bwrap.apply_answer_masks(
                argv, host_surfaces.host_input_surfaces(argv, workspace, bundle, repo=tmp_path)
            )
            probe = subprocess.run(
                [
                    *argv,
                    "/bin/sh",
                    "-c",
                    f'test -s "{public}" && test ! -s "{software}" && test ! -s "{frozen_software}"',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if "Creating new namespace failed: Operation not permitted" in probe.stderr:
                pytest.skip("bubblewrap user namespace is disabled on this host")
            assert probe.returncode == 0, probe.stderr


def test_formal_uses_only_masked_run_owned_freeze(tmp_path: Path):
    from merlin_experiments.phase1.feedback import formal

    from merlin.targetgen.sandbox import bwrap

    spec, software, host, _ = _fixture(tmp_path)
    run = tmp_path / "run"
    root = run / "private_full_model_input" / "sources"
    record = freeze.stage(spec, root, target="fixture")
    private_spec = run / "private_full_model_input" / "spec.yaml"
    shutil.copy2(spec, private_spec)
    masks = [{"path": str(path), "kind": kind} for path, kind in ((root, "dir"), (software, "file"), (host, "file"))]
    effective = {"host_inputs": [{"path": str(root)}], "private_validation_paths": masks}
    effective_raw = yaml.safe_dump(effective).encode()
    (run / "input_bundle_manifest.yaml").write_bytes(effective_raw)
    private = {
        "source_sha256": _sha(spec.read_bytes()),
        "path": str(private_spec),
        "sha256": _sha(private_spec.read_bytes()),
        "frozen_path": str(private_spec),
        "frozen_sha256": _sha(private_spec.read_bytes()),
        "masked_paths": masks,
        "source_freeze": record,
    }
    environment = {
        "private_full_model_spec": private,
        "bundle_manifest_sha256": _sha(effective_raw),
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, effective, repo=tmp_path)
    environment["bundle_input_snapshot"] = bwrap.snapshot_record(workspace)
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    software.write_bytes(b"changed after run preparation\n")
    assert (
        formal._private_source_freeze_for_formal(run, private_spec, "fixture", workspace=workspace, repo=tmp_path)
        == record
    )
    [frozen_root] = bwrap.snapshot_input_paths(workspace, effective, [root], repo=tmp_path)
    copied_source = frozen_root / Path(record["sources"][0]["frozen_path"]).relative_to(root)
    preserved = copied_source.read_bytes()
    copied_source.chmod(0o600)
    copied_source.write_bytes(b"changed during linked builds\n")
    with pytest.raises(RuntimeError, match="snapshot content verification failed"):
        formal._private_source_freeze_for_formal(run, private_spec, "fixture", workspace=workspace, repo=tmp_path)
    copied_source.write_bytes(preserved)
    copied_source.chmod(0o400)
    effective["private_validation_paths"] = masks[:1]
    changed_raw = yaml.safe_dump(effective).encode()
    (run / "input_bundle_manifest.yaml").write_bytes(changed_raw)
    environment["bundle_manifest_sha256"] = _sha(changed_raw)
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(ValueError, match="masked"):
        formal._private_source_freeze_for_formal(run, private_spec, "fixture", workspace=workspace, repo=tmp_path)
    environment_path = run / "environment.yaml"
    original_environment = tmp_path / "indirect-environment.yaml"
    original_environment.write_text(yaml.safe_dump(environment))
    environment_path.unlink()
    environment_path.symlink_to(original_environment)
    with pytest.raises(ValueError, match="indirect"):
        formal._private_source_freeze_for_formal(run, private_spec, "fixture", workspace=workspace, repo=tmp_path)


@pytest.mark.parametrize("alias_kind", ["hardlink", "private_symlink", "ancestor_symlink", "ordinary"])
def test_large_private_roster_preserves_alias_denials(tmp_path, alias_kind):
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    private.mkdir()
    hosts = []
    for i in range(512):
        root = private / str(i)
        root.mkdir()
        hosts.append((str(root), root))
    secret = hosts[-1][1] / "answer"
    secret.write_text("same bytes are allowed; inode aliases are not")
    alias = public / "input"
    if alias_kind == "hardlink":
        os.link(secret, alias)
    elif alias_kind == "private_symlink":
        alias.symlink_to(secret)
    elif alias_kind == "ancestor_symlink":
        alias.symlink_to(private, target_is_directory=True)
    else:
        alias.write_bytes(secret.read_bytes())
    if alias_kind == "ordinary":
        _validate_host_sources([("public", public)], hosts)
    else:
        with pytest.raises(RuntimeError, match="alias"):
            _validate_host_sources([("public", public)], hosts)
