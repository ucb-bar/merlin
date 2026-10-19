"""Actual independent preparation before source freeze; fixtures grant no target support."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import compiler_pass_preparation as P

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, copy_compiler_source


class OriginalPreparation:
    def __init__(self, evidence):
        self.evidence, self.identity = evidence, sha256_file(evidence)
        self.sources = []
        self.checked = []

    def revalidate(self):
        if sha256_file(self.evidence) != self.identity:
            raise ValueError("original independent evidence changed")

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        self.sources.append((source, policy))
        copy_compiler_source(source, destination)
        (destination / "selected-release.json").write_text(json.dumps({"source": self.identity}))
        (destination / "candidate_policy.py").write_text("ENABLED=" + repr(policy["enabled"]) + "\n")
        return destination

    def check_compiler_binding(self, package):
        self.revalidate()
        if json.loads((package / "selected-release.json").read_bytes()) != {"source": self.identity}:
            raise ValueError("foreign selected release")
        self.checked.append(package)


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "compiler.py").write_text("def compile(source): return source\n")
    original = tmp_path / "original"
    digest = PC.materialize_readonly_tree(source, original)
    evidence = tmp_path / "evidence"
    evidence.write_text("original independently checked neutral fixture\n")
    owner = OriginalPreparation(evidence)
    binding = PassCapabilityBinding("facts", "effects", "exact-i32", "native-fixture", "fixed-release")
    registry = ReviewedPassCapabilities((ReviewedPassCapability("composite", binding, owner),))
    selection = CompilerPassSelection(registry, binding, mechanisms=("composite",))
    choice = P.CompilerPassPreparation(
        "composite", baseline_policy={"enabled": False}, candidate_policy={"enabled": True}
    )
    return selection, choice, original, digest, owner


def prepared(setup, tmp_path):
    selection, choice, original, digest, _ = setup
    return P.prepare_pair(selection, choice, original=original, functional_sha256=digest, root=tmp_path / "prepared")


def test_both_arms_start_from_original_before_readonly_freeze(setup, tmp_path):
    selection, choice, original, digest, owner = setup
    record = prepared(setup, tmp_path)
    assert owner.sources == [(original, {"enabled": False}), (original, {"enabled": True})]
    assert hash_tree(original)["sha256"] == digest
    assert not (original / "candidate_policy.py").exists()
    assert record["baseline"]["sha256"] != record["candidate_seed"]["sha256"] != digest
    for role in ("baseline", "candidate_seed"):
        package = Path(record[role]["path"])
        assert package in owner.checked
        assert not package.stat().st_mode & 0o222
        assert not (package / "candidate_policy.py").stat().st_mode & 0o222
    assert P.verify_pair(record, selection, functional_sha256=digest) == record
    assert P.validate_pair(record, functional_sha256=digest) == record
    assert choice.record()["candidate_policy"] == {"enabled": True}


@pytest.mark.parametrize("role", ["original", "baseline", "candidate_seed"])
def test_mutated_original_or_prepared_snapshot_refuses(setup, tmp_path, role):
    selection, _, _, digest, _ = setup
    record = prepared(setup, tmp_path)
    package = Path(record[role]["path"])
    path = package / "compiler.py"
    path.chmod(0o644)
    path.write_text("def compile(source): return 'copied success'\n")
    with pytest.raises(P.StageGateError, match="writable|changed"):
        P.verify_pair(record, selection, functional_sha256=digest)


def test_preparation_record_cannot_issue_owner_or_relabel_original_parent(setup, tmp_path):
    selection, _, _, digest, _ = setup
    record = prepared(setup, tmp_path)
    with pytest.raises(P.StageGateError, match="live original"):
        P.verify_pair(record, {"status": "pass"}, functional_sha256=digest)
    with pytest.raises(P.StageGateError, match="different original"):
        P.verify_pair(record, selection, functional_sha256="0" * 64)
    changed = deepcopy(record)
    changed["candidate_seed"]["path"] = record["original"]["path"]
    changed["sha256"] = P._sha({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(P.StageGateError, match="disjoint"):
        P.verify_pair(changed, selection, functional_sha256=digest)


def test_missing_binding_checker_or_changed_original_evidence_never_prepares(setup, tmp_path):
    selection, choice, original, digest, owner = setup
    owner.check_compiler_binding = lambda _package: {"status": "pass"}
    with pytest.raises(P.StageGateError, match="original bound"):
        P.prepare_pair(selection, choice, original=original, functional_sha256=digest, root=tmp_path / "refused")
    owner.evidence.write_text("copied successful verdict\n")
    with pytest.raises(ValueError, match="evidence changed"):
        selection.record()


def test_unknown_policy_and_multiple_capability_chain_refuse(setup, tmp_path):
    selection, choice, original, digest, _ = setup
    with pytest.raises(P.StageGateError, match="explicit live"):
        P.prepare_pair(selection, choice.record(), original=original, functional_sha256=digest, root=tmp_path / "bad")
    selection = CompilerPassSelection(selection.registry, selection.binding, mechanisms=("composite", "other"))
    with pytest.raises(P.StageGateError, match="not chained"):
        P.prepare_pair(selection, choice, original=original, functional_sha256=digest, root=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_policy_read_requires_exact_ordinary_leaf_and_hash(setup, tmp_path):
    _, choice, _, _, _ = setup
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(choice.record()))
    digest = sha256_file(path)
    assert P.read_preparation(path, digest).record() == choice.record()
    with pytest.raises(P.StageGateError, match="hash differs"):
        P.read_preparation(path, "0" * 64)
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(P.StageGateError, match="ordinary"):
        P.read_preparation(linked, digest)
    path.write_text(json.dumps({**choice.record(), "status": "pass"}))
    with pytest.raises(P.StageGateError, match="unknown.*fields"):
        P.read_preparation(path, sha256_file(path))


def test_independent_original_host_policy_and_fresh_application_required(setup, tmp_path):
    selection, choice, original, _, owner = setup
    record = prepared(setup, tmp_path)
    replayed = P.rederive_pair(record, selection, choice, replay_root=tmp_path / "fresh-original-application")
    assert len(owner.sources) == 4
    assert all(source == original for source, _ in owner.sources)
    assert replayed["baseline"]["sha256"] == record["baseline"]["sha256"]
    assert replayed["candidate_seed"]["sha256"] == record["candidate_seed"]["sha256"]
    other = P.CompilerPassPreparation(
        "composite", baseline_policy={"enabled": True}, candidate_policy={"enabled": False}
    )
    with pytest.raises(P.StageGateError, match="independently selected"):
        P.rederive_pair(record, selection, other, replay_root=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_self_consistent_candidate_json_cannot_choose_host_preparation(setup, tmp_path):
    selection, choice, _, _, _ = setup
    record = prepared(setup, tmp_path)
    forged = deepcopy(record)
    forged["preparation"]["baseline_policy"]["enabled"] = True
    forged["sha256"] = P._sha({k: v for k, v in forged.items() if k != "sha256"})
    P.validate_pair(forged)
    with pytest.raises(P.StageGateError, match="independently selected"):
        P.rederive_pair(forged, selection, choice, replay_root=tmp_path / "unused")
    changed = deepcopy(record)
    changed["preparations"]["baseline"]["prepared_sha256"] = "0" * 64
    changed["sha256"] = P._sha({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(P.StageGateError, match="receipt has different"):
        P.verify_pair(changed, selection)
