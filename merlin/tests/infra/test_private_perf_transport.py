"""Real frozen V4 inputs cross the performance boundary without granting private bytes."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import functional_inputs as FI

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC

sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
PC = importlib.import_module("merlin_experiments.phase2.campaign")
RF = importlib.import_module("refreeze_functional_run")


@pytest.fixture(params=[False, True], ids=["copied", "shared-store"])
def private_functional(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas") if request.param else "")
    repo = tmp_path / "original-repo"
    public = repo / "inputs"
    private = public / "private-holdout"
    host = public / "host-lane"
    private.mkdir(parents=True)
    host.mkdir()
    (public / "public.h").write_text("public interface")
    (private / "secret-case.txt").write_text("unpublished expected value")
    (host / "host.cc").write_text("public host implementation")
    support = public / "selected-support"
    (support / "contracts").mkdir(parents=True)
    (support / "contracts/target_contract.yaml").write_text("name: frozen_fixture_device\n")
    (support / "private.py").write_text("# withheld support implementation\n")
    (support / "contracts/alias.py").symlink_to(support / "private.py")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(support))
    run_dir = tmp_path / "functional-run"
    workspace = run_dir / "authoring" / "workspace"
    workspace.mkdir(parents=True)
    bundle = {
        "allowed": [{"path": "inputs"}, {"path": "inputs/selected-support/contracts"}],
        "host_inputs": [{"path": "inputs/private-holdout"}],
    }
    manifest = run_dir / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    RF.materialize_snapshot(workspace, bundle, repo)
    snapshot = BW.snapshot_record(workspace)
    root = BW.bundle_snapshot_root(workspace)
    frozen_host = root / "repo/inputs/host-lane"
    host_record = {
        "run_snapshot": snapshot,
        "package": "inputs/host-lane",
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
    (run_dir / "environment.yaml").write_text(yaml.safe_dump(environment))
    functional = SimpleNamespace(
        run_dir=run_dir,
        bundle_input_snapshot=snapshot,
        model_host_package=frozen_host,
        submission_dir=run_dir / "submission",
        run_id="fixture",
        digest="a" * 64,
        public_capsules=1,
        hidden_capsules=1,
        public_score={},
        hidden_score={},
        frozen_at="fixture",
    )
    sealed_repo = tmp_path / "sealed-repo"
    sealed_repo.mkdir()
    monkeypatch.setattr(AW, "repo_root", lambda: sealed_repo)
    monkeypatch.setattr(FI, "repo_root", lambda: sealed_repo)
    yield SimpleNamespace(
        repo=repo,
        public=public,
        support=support,
        private=private,
        workspace=workspace,
        root=root,
        bundle=bundle,
        manifest=manifest,
        environment=environment,
        functional=functional,
        sealed_repo=sealed_repo,
        tmp_path=tmp_path,
    )
    BW.remove_bundle_snapshot(workspace)


def _transport(fixture):
    return FI.load_frozen_functional_inputs(
        fixture.functional, public_manifest_path=fixture.tmp_path / "stage/public-inputs.json"
    )


def test_v4_admission_and_public_projection_preserve_private_host_provenance(private_functional):
    f = private_functional
    verified = PC.verify_private_input_snapshot(f.functional.run_dir, f.environment)
    assert verified["bundle"] == f.bundle
    admitted = FI._functional_input_snapshot(f.functional)
    assert admitted.bundle_input_snapshot == f.environment["bundle_input_snapshot"]
    frozen = _transport(f)
    public_text = frozen.public_marker.read_text()
    projection = json.loads(public_text)
    assert projection["kind"] == "public_functional_input_projection"
    assert projection["content_sha256"] == frozen.public_content_sha256
    assert all(
        word not in public_text
        for word in ("private-holdout", "secret-case", "unpublished", "host_records", "host_inputs")
    )
    assert frozen.content_sha256 == f.environment["bundle_input_snapshot"]["content_sha256"]
    assert frozen.marker_sha256 == f.environment["bundle_input_snapshot"]["manifest_sha256"]
    assert frozen.host_provenance["bundle_manifest_sha256"] == f.environment["bundle_manifest_sha256"]
    assert frozen.grants[0].destination == f.sealed_repo / "inputs"


def test_projection_uses_captured_support_ownership_after_provider_unlink(private_functional, monkeypatch):
    f = private_functional
    shutil.rmtree(f.support)
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    frozen = _transport(f)
    projection = json.loads(frozen.public_marker.read_text())
    contracts = next(row for row in projection["grants"] if row["path"] == "inputs/selected-support/contracts")
    public = f.root / "repo/inputs/selected-support/contracts/target_contract.yaml"
    expected = [["target_contract.yaml", hashlib.sha256(public.read_bytes()).hexdigest()]]
    assert contracts["sha256"] == CONTRACTS.document_sha256(expected)
    # Negative control: including the formerly aliased private implementation
    # would materially change the actual public receipt, not merely its label.
    private = f.root / "repo/inputs/selected-support/contracts/alias.py"
    assert contracts["sha256"] != CONTRACTS.document_sha256(
        [["alias.py", hashlib.sha256(private.read_bytes()).hexdigest()], *expected]
    )


def test_policy_refuses_unbound_inputs_but_allows_absent_bundle(private_functional):
    from dataclasses import replace

    frozen = _transport(private_functional)
    with pytest.raises(CONTRACTS.StageGateError, match="newly frozen"):
        FI._private_functional_surfaces([], replace(frozen, host_provenance=None))
    assert FI._private_functional_surfaces([], None) == []


@pytest.mark.parametrize("tamper", ["bundle", "marker", "payload", "missing_bundle_pin", "workspace"])
def test_v4_transport_refuses_unbound_or_changed_host_inputs(private_functional, tamper):
    f = private_functional
    if tamper == "bundle":
        f.manifest.write_text(f.manifest.read_text() + "\n# changed policy bytes\n")
    elif tamper == "marker":
        marker = f.root / "snapshot.json"
        marker.chmod(0o600)
        document = json.loads(marker.read_text())
        document["host_records"] = []
        marker.write_text(json.dumps(document))
    elif tamper == "payload":
        private = f.root / "repo/inputs/private-holdout/secret-case.txt"
        private.chmod(0o600)
        private.write_text("changed private oracle")
    elif tamper == "missing_bundle_pin":
        del f.environment["bundle_manifest_sha256"]
    else:
        f.environment["workspace_path"] = str(f.tmp_path / "different-run/workspace")
    with pytest.raises(PC.CampaignGateError):
        PC.verify_private_input_snapshot(f.functional.run_dir, f.environment)


def test_v4_transport_never_falls_back_to_live_input_bytes(private_functional):
    f = private_functional
    (f.private / "secret-case.txt").unlink()
    f.private.rmdir()
    (f.public / "public.h").write_text("changed live interface")
    frozen = _transport(f)
    assert (frozen.grants[0].source / "public.h").read_text() == "public interface"
    assert PC.verify_private_input_snapshot(f.functional.run_dir, f.environment)["bundle"] == f.bundle


def test_only_private_bytes_change_full_identity_not_public_projection(private_functional):
    f = private_functional
    original = _transport(f)
    hidden = f.root / "repo/inputs/private-holdout/secret-case.txt"
    hidden.chmod(0o600)
    hidden.write_text("different privately graded expected value")
    hidden.chmod(0o444)
    marker = f.root / "snapshot.json"
    marker.chmod(0o600)
    document = json.loads(marker.read_text())
    digest, n_files, n_bytes = BW._snapshot_content(f.root)
    document.update(content_sha256=digest, n_files=n_files, n_bytes=n_bytes)
    marker.write_text(json.dumps(document))
    marker.chmod(0o444)
    changed_snapshot = BW.snapshot_record(f.workspace)
    f.environment["bundle_input_snapshot"] = changed_snapshot
    f.environment["model_host_lane_snapshot"]["run_snapshot"] = changed_snapshot
    f.functional.bundle_input_snapshot = changed_snapshot
    (f.functional.run_dir / "environment.yaml").write_text(yaml.safe_dump(f.environment))
    changed = FI.load_frozen_functional_inputs(
        f.functional, public_manifest_path=f.tmp_path / "stage/public-inputs-changed.json"
    )
    assert changed.content_sha256 != original.content_sha256
    assert changed.marker_sha256 != original.marker_sha256
    assert changed.public_content_sha256 == original.public_content_sha256
    assert changed.public_marker.read_bytes() == original.public_marker.read_bytes()


@pytest.mark.parametrize("plane", ["inner", "outer"])
def test_v3_agent_planes_mask_translated_private_and_raw_marker_aliases(private_functional, monkeypatch, plane):
    f = private_functional
    frozen = _transport(f)
    candidate = f.tmp_path / "candidate"
    candidate.mkdir()
    agent_root = f.tmp_path / "agent-inputs"
    agent_root.mkdir()
    agent_inputs = SimpleNamespace(root=agent_root)
    alias = f.tmp_path / "runtime-alias"
    raw_marker_alias = f.tmp_path / "raw-marker.json"
    run_alias = f.tmp_path / "run-alias"
    support_alias = f.tmp_path / "support-alias.py"
    runtime = [
        "--ro-bind",
        str(f.root / "repo/inputs/private-holdout"),
        str(alias),
        "--ro-bind",
        str(f.root / "snapshot.json"),
        str(raw_marker_alias),
        "--ro-bind",
        str(f.functional.run_dir),
        str(run_alias),
        "--bind",
        str(f.root / "repo/inputs/selected-support/contracts/alias.py"),
        str(support_alias),
    ]
    monkeypatch.setattr(AW, "verify_answer_free_agent_inputs", lambda inputs: None)
    monkeypatch.setattr(BW, "base_argv", lambda *args, **kwargs: ["bwrap"])
    monkeypatch.setattr(TC, "toolchain_binds", lambda te: runtime)
    monkeypatch.setattr(AW, "answer_surfaces", lambda te: [])
    if plane == "inner":
        policy = AW.inner_execution_policy(None, candidate, agent_inputs, frozen)
    else:
        policy = AW.outer_codex_policy(candidate, agent_inputs, runtime, None, frozen)
    argv = list(policy.argv)
    assert BW.is_exposed(argv, f.sealed_repo / "inputs/public.h")
    assert not BW.is_exposed(argv, f.sealed_repo / "inputs/private-holdout/secret-case.txt")
    assert not BW.is_exposed(argv, alias / "secret-case.txt")
    assert not BW.is_exposed(argv, raw_marker_alias)
    assert not BW.is_exposed(argv, run_alias / "environment.yaml")
    assert not BW.is_exposed(argv, run_alias / "input_bundle_manifest.yaml")
    assert not BW.is_exposed(argv, support_alias)
    assert not BW.is_exposed(argv, f.sealed_repo / "inputs/selected-support/contracts/alias.py")
    assert BW.is_exposed(argv, f.sealed_repo / "inputs/selected-support/contracts/target_contract.yaml")
    assert BW.is_exposed(argv, AW.FUNCTIONAL_INPUT_MANIFEST_MOUNT)
    index = argv.index(str(AW.FUNCTIONAL_INPUT_MANIFEST_MOUNT))
    assert argv[index - 1] == str(frozen.public_marker)


def test_v3_policy_rechecks_public_projection_bytes(private_functional):
    frozen = _transport(private_functional)
    frozen.public_marker.chmod(0o600)
    frozen.public_marker.write_text("changed projection")
    with pytest.raises(CONTRACTS.StageGateError, match="projection changed"):
        FI._private_functional_surfaces([], frozen)


def test_refreeze_absolute_hidden_root_uses_frozen_tree(private_functional):
    f = private_functional
    frozen_hidden = f.root / "repo/inputs/private-holdout"
    # A real absolute source inside the original repo must still select its snapshot counterpart.
    frozen_hidden.chmod(0o700)
    capsule = frozen_hidden / "hidden-case"
    capsule.mkdir()
    (capsule / "capsule.yaml").write_text("frozen declaration")
    destination = f.tmp_path / "refrozen-cohort"
    te = SimpleNamespace(hidden_corpus=lambda: str(f.private))
    assert RF.stage_hidden_cohort(te, f.root, f.repo, ["hidden-case"], destination) == ["hidden-case"]
    assert (destination / "hidden-case/capsule.yaml").read_text() == "frozen declaration"


def test_refreeze_oot_hidden_root_never_reads_live_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external-corpus"
    capsule = external / "heldout"
    capsule.mkdir(parents=True)
    declaration = capsule / "capsule.yaml"
    declaration.write_text("frozen external declaration")
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    bundle = {"allowed": [], "host_inputs": [{"path": str(external)}]}
    try:
        root, record = RF.materialize_snapshot(workspace, bundle, repo)
        assert record["version"] == 4
        declaration.write_text("changed live declaration")
        te = SimpleNamespace(hidden_corpus=lambda: str(external))
        destination = tmp_path / "cohort"
        assert RF.stage_hidden_cohort(te, root, repo, ["heldout"], destination) == ["heldout"]
        assert (destination / "heldout/capsule.yaml").read_text() == "frozen external declaration"
    finally:
        BW.remove_bundle_snapshot(workspace)
