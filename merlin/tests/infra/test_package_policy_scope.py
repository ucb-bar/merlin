"""One package-policy scope reuses parsed capsule metadata and changes no mount, mask or probe.

Every candidate entrypoint call composes a fresh bwrap policy, and classifying the live and frozen
corpora parses every ``capsule.yaml`` each time. A scope reuses only the parse of byte-identical
metadata (keyed by the SHA-256 of the bytes read); guarded filesystem facts may be shared,
while live host-input policy and mask derivation stay per call. These tests hold the reuse to three properties: the result equals a fresh
per-call derivation on actual frozen snapshots (hardlink aliases included), any changed input is
observed, and a candidate call's argv is byte-identical with and without the scope.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import external_sources
import pytest
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as PC

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_privacy
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, CapsuleMetadata

# The package re-exports the function under the submodule's name.
AS = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


def _closure(*members: str) -> dict:
    return {
        "schema": capsule_privacy.REFERENCE_SCHEMA,
        "members": [{"path": path, "role": "expected_output", "sha256": "0" * 64, "bytes": 1} for path in members],
    }


def _owner(tmp_path: Path, name: str) -> SimpleNamespace:
    """An actual frozen bundle snapshot with a private host input, a corpus and a public grant."""
    repo = tmp_path / name / "repo"
    hidden = repo / "inputs/hidden"
    hidden.mkdir(parents=True)
    (hidden / "secret.bin").write_bytes(b"host only")
    (repo / "inputs/public.txt").write_text("public input")
    corpus = repo / "merlin/contract/capsules/fixture"
    for capsule in ("member", "twin"):
        member = corpus / "isa" / capsule
        (member / "reference").mkdir(parents=True)
        (member / "reference/expected.bin").write_bytes(b"x")
        # A declared reference artifact is withheld only through the parsed capsule metadata. Both
        # capsules carry byte-identical metadata, so one parse serves both.
        document = {"name": "public", "reference_closure": _closure("reference/expected.bin")}
        (member / "capsule.yaml").write_text(json.dumps(document))
    member = corpus / "isa/member"
    (member / "golden.yaml").write_text("private golden\n")
    (member / "model.weights.safetensors").write_bytes(b"private weights")
    (corpus / "hidden/holdout").mkdir(parents=True)
    (corpus / "hidden/holdout/capsule.yaml").write_text("name: holdout\n")
    workspace = tmp_path / name / "run/workspace"
    workspace.mkdir(parents=True)
    bundle = {
        "allowed": [{"path": "inputs"}, {"path": "merlin/contract/capsules"}],
        "host_inputs": [{"path": "inputs/hidden"}],
    }
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    runtime = tmp_path / name / "runtime"
    runtime.mkdir()
    # A runtime tree carrying a hardlink to the frozen private payload: only the per-call hardlink
    # scan can see it.
    os.link(frozen / "secret.bin", runtime / "alias.bin")
    te = SimpleNamespace(target="fixture", capsule_corpus=corpus / "isa", corpus_siblings=lambda: [])
    return SimpleNamespace(
        root=tmp_path / name,
        repo=repo,
        workspace=workspace,
        bundle=bundle,
        hidden=hidden,
        frozen=frozen,
        runtime=runtime,
        corpus=corpus,
        golden=member / "golden.yaml",
        te=te,
    )


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    # Descriptor discovery needs a full descriptor; the live golden is the answer it would yield.
    monkeypatch.setattr(
        BW,
        "answer_surfaces",
        lambda te, **_: [AnswerSurface("golden", te.capsule_corpus / "member/golden.yaml", "file", "golden")],
    )
    selected = _owner(tmp_path, "owner")
    yield selected
    BW.remove_bundle_snapshot(selected.workspace)


@pytest.fixture
def parses(monkeypatch):
    """Count capsule metadata parses while keeping the actual loader."""
    counted = []
    loader = capsule_privacy.fast_safe_load

    def load(data):
        counted.append(data)
        return loader(data)

    monkeypatch.setattr(capsule_privacy, "fast_safe_load", load)
    return counted


def _call_prefix(owner: SimpleNamespace, call: Path) -> list[str]:
    """The mount list one candidate call hands to the final masks: its own workspace, a runtime
    tree bound under the repository, a broad installed-runtime ancestor (which also reaches the
    frozen snapshot storage), and the owner's frozen public grants."""
    call.mkdir(parents=True)
    return [
        "--bind",
        str(call),
        str(call),
        "--ro-bind",
        str(owner.runtime),
        str(owner.repo / "runtime-view"),
        "--ro-bind",
        str(owner.root),
        str(owner.root),
        *BW._bundle_mount_args(owner.workspace, owner.bundle, owner.repo),
    ]


def _masks(owner: SimpleNamespace, prefix: list[str], capsules=None) -> list[str]:
    return BW.apply_final_answer_masks(
        list(prefix), owner.te, owner.workspace, owner.bundle, repo=owner.repo, capsules=capsules
    )


def _frozen_root(owner: SimpleNamespace) -> Path:
    return BW.bundle_snapshot_root(owner.workspace) / "repo/merlin/contract/capsules/fixture"


def test_scoped_masks_equal_fresh_per_call_masks_on_actual_snapshots(owner, parses, tmp_path):
    capsules = CapsuleMetadata()
    frozen_root = _frozen_root(owner)
    scoped_parses = []
    for index in range(3):
        prefix = _call_prefix(owner, tmp_path / f"call-{index}")
        alias = owner.repo / "runtime-view/alias.bin"
        assert BW.is_exposed(prefix, alias)  # The hardlink alias is a real exposure before masking.
        assert BW.is_exposed(prefix, frozen_root / "isa/member/reference/expected.bin")
        before = len(parses)
        fresh = _masks(owner, prefix)
        fresh_parses = len(parses) - before
        scoped = _masks(owner, prefix, capsules)
        scoped_parses.append(len(parses) - before - fresh_parses)
        assert "\0".join(scoped).encode() == "\0".join(fresh).encode()
        assert not BW.is_exposed(scoped, alias)
        assert not BW.is_exposed(scoped, owner.frozen / "secret.bin")
        assert not BW.is_exposed(scoped, owner.golden)
        for relative in (
            "isa/member/golden.yaml",
            "isa/member/model.weights.safetensors",
            "isa/member/reference/expected.bin",
            "isa/twin/reference/expected.bin",
            "hidden/holdout",
        ):
            assert not BW.is_exposed(scoped, frozen_root / relative)
        assert BW.is_exposed(scoped, frozen_root / "isa/member/capsule.yaml")
    # A fresh call parses at least the three frozen capsule documents (possibly again for
    # overlapping roots); the scope parses the twins' shared bytes and the holdout once total.
    assert fresh_parses >= 3
    assert scoped_parses == [2, 0, 0]


def test_hardlink_created_mid_scope_is_still_masked(owner, tmp_path):
    """Host-input surfaces are never reused: a later call observes a newly created private link."""
    capsules = CapsuleMetadata()
    _masks(owner, _call_prefix(owner, tmp_path / "first"), capsules)
    os.link(owner.frozen / "secret.bin", owner.runtime / "late.bin")
    prefix = _call_prefix(owner, tmp_path / "second")
    late = owner.repo / "runtime-view/late.bin"
    assert BW.is_exposed(prefix, late)
    scoped = _masks(owner, prefix, capsules)
    assert not BW.is_exposed(scoped, late)
    assert scoped == _masks(owner, prefix)


def test_changed_capsule_metadata_or_new_capsule_is_observed_mid_scope(owner, parses, tmp_path):
    capsules = CapsuleMetadata()
    prefix = _call_prefix(owner, tmp_path / "call")
    _masks(owner, prefix, capsules)
    assert len(parses) == 2
    # The live corpus changes: one capsule declares another reference member and a new capsule
    # appears. The owner's snapshot is re-frozen over those bytes.
    member = owner.corpus / "isa/member"
    (member / "reference/late.bin").write_bytes(b"y")
    changed = {"name": "public", "reference_closure": _closure("reference/expected.bin", "reference/late.bin")}
    (member / "capsule.yaml").write_text(json.dumps(changed))
    appeared = owner.corpus / "isa/appeared"
    (appeared / "reference").mkdir(parents=True)
    (appeared / "reference/expected.bin").write_bytes(b"z")
    new = {"name": "appeared", "reference_closure": _closure("reference/expected.bin")}
    (appeared / "capsule.yaml").write_text(json.dumps(new))
    BW.remove_bundle_snapshot(owner.workspace)
    BW.materialize_bundle_inputs(owner.workspace, owner.bundle, repo=owner.repo)
    [refrozen] = BW.snapshot_input_paths(owner.workspace, owner.bundle, [owner.hidden], repo=owner.repo)
    os.link(refrozen / "secret.bin", owner.runtime / "refrozen.bin")
    before = len(parses)
    scoped = _masks(owner, prefix, capsules)
    # Only the changed and the newly appearing documents are parsed; the unchanged ones are reused.
    assert len(parses) == before + 2
    assert scoped == _masks(owner, prefix)
    frozen_root = _frozen_root(owner)
    for relative in ("isa/member/reference/late.bin", "isa/appeared/reference/expected.bin"):
        assert BW.is_exposed(prefix, frozen_root / relative)
        assert not BW.is_exposed(scoped, frozen_root / relative)
    assert not BW.is_exposed(scoped, owner.repo / "runtime-view/refrozen.bin")


def test_capsule_metadata_is_keyed_by_the_bytes_read(tmp_path, parses):
    capsules = CapsuleMetadata()
    first, second, absent = tmp_path / "first", tmp_path / "second", tmp_path / "absent"
    for directory in (first, second, absent):
        directory.mkdir()
    (first / "capsule.yaml").write_text("name: same\n")
    (second / "capsule.yaml").write_text("name: same\n")
    assert capsules.load(first) == capsules.load(second) == {"name": "same"}
    assert len(parses) == 1
    (second / "capsule.yaml").write_text("name: changed\n")
    assert capsules.load(second) == {"name": "changed"}
    assert len(parses) == 2
    assert capsules.load(absent) == {}
    (first / "capsule.yaml").write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="requires a mapping"):
        capsule_privacy.capsule_answer_members(first, capsule=capsules.load(first))
    with pytest.raises(ValueError, match="requires a mapping"):
        capsule_privacy.capsule_answer_members(first)


def test_original_executor_owns_one_scope_for_all_its_calls(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "tool.py").write_text("print('compiler')\n")
    seen = []
    monkeypatch.setattr(PC, "package_sandbox_policy", lambda *args, **kwargs: seen.append(kwargs["scope"]))
    first = R._Executor(None, None, candidate, tmp_path / "first")
    second = R._Executor(None, None, candidate, tmp_path / "second")
    first._policy(tmp_path, candidate)
    first._policy(tmp_path, candidate)
    second._policy(tmp_path, candidate)
    assert type(seen[0]) is PC.PackagePolicyScope
    assert seen[0] is seen[1] is first.policy_scope
    assert seen[2] is second.policy_scope is not first.policy_scope


@external_sources.requires_ext("chipyard")
def test_real_corpus_surfaces_and_candidate_argv_are_identical_with_and_without_scope(tmp_path, monkeypatch):
    """The real descriptor and repository corpus, package policy plus original public grants."""
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini" / "target_experiment.yaml"
    te = load_target_experiment(descriptor)
    capsules = CapsuleMetadata()
    fresh_answers = AS.answer_surfaces(te)
    assert AS.answer_surfaces(te, capsules=capsules) == fresh_answers  # Fills the scope.
    assert AS.answer_surfaces(te, capsules=capsules) == fresh_answers  # Served from it.
    assert capsules._documents, "the real corpus must contribute parsed capsule metadata"

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    owner = _owner(tmp_path, "owner")
    package = tmp_path / "frozen/submission"
    package.mkdir(parents=True)
    monkeypatch.setattr(R, "_public_python_roots", lambda _owner: (owner.repo / "src",))
    scope = PC.PackagePolicyScope()
    try:
        for index in range(2):
            workspace = tmp_path / f"call-{index}"
            workspace.mkdir()
            fresh = PC.package_sandbox_policy(te, workspace, package)
            scoped = PC.package_sandbox_policy(te, workspace, package, scope=scope)
            assert "\0".join(scoped.argv).encode() == "\0".join(fresh.argv).encode()
            assert (scoped.env_prefix, scoped.required_tools, scoped.coverage_gap) == (
                fresh.env_prefix,
                fresh.required_tools,
                fresh.coverage_gap,
            )
            assert scoped.selected_inputs == fresh.selected_inputs
            # The original-profile public grants and final masks over an actual owner snapshot.
            fresh_public = R.public_package_policy(fresh, target=te, owner=owner)
            scoped_public = R.public_package_policy(scoped, target=te, owner=owner, scope=scope)
            assert "\0".join(scoped_public.argv).encode() == "\0".join(fresh_public.argv).encode()
            assert scoped_public.env_prefix == fresh_public.env_prefix
            assert not BW.is_exposed(scoped_public.argv, owner.frozen / "secret.bin")
    finally:
        BW.remove_bundle_snapshot(owner.workspace)
