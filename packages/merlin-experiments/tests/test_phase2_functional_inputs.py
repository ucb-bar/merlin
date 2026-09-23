"""Installed functional-input transport uses frozen bytes and captured ownership."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import functional_inputs as FI

from merlin.benchharness import hash_tree
from merlin.common.paths import python_import_roots
from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture(params=[False, True], ids=["copied", "shared-store"])
def frozen_source(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas") if request.param else "")
    repo = tmp_path / "original"
    public = repo / "inputs"
    private = public / "private"
    host = public / "host"
    support = public / "support"
    for directory in (private, host, support / "contracts"):
        directory.mkdir(parents=True)
    (public / "public.h").write_text("public interface")
    (private / "secret.txt").write_text("withheld expected value")
    (host / "host.cc").write_text("host source")
    (support / "contracts/target_contract.yaml").write_text("name: synthetic_fixture\n")
    (support / "private.py").write_text("private implementation")
    (support / "contracts/alias.py").symlink_to(support / "private.py")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(support))
    external = tmp_path / "external.txt"
    external.write_text("external frozen bytes")
    run = tmp_path / "run"
    workspace = run / "authoring/workspace"
    workspace.mkdir(parents=True)
    bundle = {
        "allowed": [{"path": "inputs"}, {"path": "inputs/support/contracts"}, {"path": str(external)}],
        "host_inputs": [{"path": "inputs/private"}],
    }
    manifest = run / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    BW.verify_bundle_snapshot(workspace, bundle, repo=repo)
    root = BW.bundle_snapshot_root(workspace)
    snapshot = BW.snapshot_record(workspace)
    frozen_host = root / "repo/inputs/host"
    host_record = {
        "run_snapshot": snapshot,
        "package": "inputs/host",
        "resolved_package": str(frozen_host),
        **hash_tree(frozen_host),
    }
    host_record["package_sha256"] = host_record.pop("sha256")
    environment = {
        "workspace_path": str(workspace),
        "bundle_input_snapshot": snapshot,
        "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_host_lane_snapshot": host_record,
    }
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    baseline = SimpleNamespace(
        run_dir=run,
        submission_dir=run / "submission",
        run_id="synthetic",
        digest="a" * 64,
        public_capsules=1,
        hidden_capsules=1,
        public_score={},
        hidden_score={},
        frozen_at="fixture",
    )
    relocated = tmp_path / "relocated"
    monkeypatch.setattr(FI, "repo_root", lambda: relocated)
    try:
        yield SimpleNamespace(
            baseline=baseline,
            root=root,
            workspace=workspace,
            public=public,
            external=external,
            relocated=relocated,
            manifest=manifest,
            environment=environment,
            projection=tmp_path / "stage/public.json",
        )
    finally:
        BW.remove_bundle_snapshot(workspace)


def transport(f):
    admitted = FI._functional_input_snapshot(f.baseline)
    return admitted, FI.load_frozen_functional_inputs(admitted, public_manifest_path=f.projection)


def test_real_v4_admission_projection_relocation_and_private_masks(frozen_source, monkeypatch):
    f = frozen_source
    before = FI.CONTRACTS.exact_tree_record(f.root)
    # The admitted snapshot, not current provider discovery or live payload, owns execution.
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    (f.public / "public.h").write_text("changed live input")
    admitted, frozen = transport(f)
    assert admitted.bundle_input_snapshot["version"] == 4
    assert frozen.grants[0].destination == f.relocated / "inputs"
    external = next(grant for grant in frozen.grants if grant.declared_path == str(f.external))
    assert external.destination == f.external
    assert external.source.read_text() == "external frozen bytes"
    assert FI._frozen_path_for_destination(frozen, f.relocated / "inputs/public.h").read_text() == "public interface"
    assert FI.CONTRACTS.exact_tree_record(f.root) == before
    assert not f.projection.stat().st_mode & 0o222
    projection = json.loads(f.projection.read_text())
    assert projection["content_sha256"] == frozen.public_content_sha256
    assert all(word not in f.projection.read_text() for word in ("secret.txt", "withheld", "host_inputs"))
    contracts = next(row for row in projection["grants"] if row["path"] == "inputs/support/contracts")
    contract = f.root / "repo/inputs/support/contracts/target_contract.yaml"
    assert contracts["sha256"] == FI.CONTRACTS.document_sha256(
        [["target_contract.yaml", hashlib.sha256(contract.read_bytes()).hexdigest()]]
    )
    alias = f.projection.parent / "private-alias"
    argv = FI.frozen_grant_mounts(frozen) + [
        "--ro-bind",
        str(f.root / "repo/inputs/private"),
        str(alias),
    ]
    surfaces = FI._private_functional_surfaces(argv, frozen)
    masked = BW.apply_answer_masks(argv, surfaces)
    assert BW.coverage_gap(masked, surfaces) == []
    assert BW.is_exposed(masked, f.relocated / "inputs/public.h")
    assert not BW.is_exposed(masked, f.relocated / "inputs/private/secret.txt")
    assert not BW.is_exposed(masked, alias / "secret.txt")
    assert not BW.is_exposed(masked, f.relocated / "inputs/support/contracts/alias.py")
    with pytest.raises(FrozenInstanceError):
        frozen.root = f.root
    with pytest.raises(FileExistsError):
        FI.load_frozen_functional_inputs(admitted, public_manifest_path=f.projection)


@pytest.mark.parametrize("tamper", ["projection", "environment", "bundle", "payload"])
def test_private_mask_revalidates_bound_evidence(frozen_source, tamper):
    f = frozen_source
    _, frozen = transport(f)
    path = {
        "projection": f.projection,
        "environment": f.baseline.run_dir / "environment.yaml",
        "bundle": f.manifest,
        "payload": f.root / "repo/inputs/private/secret.txt",
    }[tamper]
    path.chmod(0o600)
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(FI.StageGateError):
        FI._private_functional_surfaces([], frozen)


def test_missing_projection_unbound_provenance_and_ungranted_paths_refused(frozen_source):
    f = frozen_source
    admitted = FI._functional_input_snapshot(f.baseline)
    with pytest.raises(FI.StageGateError, match="separate public manifest"):
        FI.load_frozen_functional_inputs(admitted)
    _, frozen = transport(f)
    with pytest.raises(FI.StageGateError, match="newly frozen"):
        FI._private_functional_surfaces([], replace(frozen, host_provenance=None))
    with pytest.raises(FI.StageGateError, match="did not grant"):
        FI._frozen_path_for_destination(frozen, f.relocated / "ungranted")
    with pytest.raises(FI.StageGateError, match="absent"):
        FI._frozen_path_for_destination(frozen, f.relocated / "inputs/missing.h")
    assert FI._private_functional_surfaces([], None) == []


def test_owner_import_does_not_load_native_controller(tmp_path):
    program = """import importlib.abc, subprocess, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"perf_agent_stage", "perf_campaign", "refreeze_functional_run", "_common"}:
            raise AssertionError("native import: " + fullname)
sys.meta_path.insert(0, NoNative())
def forbidden(*args, **kwargs):
    raise AssertionError("import launched a process")
subprocess.Popen = forbidden
from merlin_experiments.phase2 import functional_inputs
assert callable(functional_inputs.load_frozen_functional_inputs)
assert not any("gemmini_perf_bench" in str(getattr(m, "__file__", "")) for m in sys.modules.values())
"""
    env = {
        **os.environ,
        "MERLIN_REPO_ROOT": str(tmp_path),
        "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
    }
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
