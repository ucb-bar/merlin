"""Candidate projections withhold typed references without executing any program."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import agent_workspace as workspace

from merlin.targetgen.capsule_privacy import (
    REFERENCE_SCHEMA,
    capsule_answer_members,
    capture_reference_closure,
)
from merlin.targetgen.sandbox import bwrap

answers = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


def _capsule(root, *, explicit=True):
    root.mkdir(parents=True)
    payload = b"protected full expected output"
    document = {"name": "program", "materialized_capture": {
        "schema": "merlin.materialized_public_capture.v2", "workload_role": "performance_scale"}}
    if explicit:
        document["reference_closure"] = capture_reference_closure({
            "golden.npy": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}})
        document["reference_closure"]["members"].append({
            "path": "references/complete.bin", "role": "expected_output",
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        (root / "references").mkdir()
        (root / "references/complete.bin").write_bytes(payload)
    (root / "capsule.yaml").write_text(yaml.safe_dump(document))
    (root / "source-capture-golden.npy").write_bytes(payload)
    (root / "golden.yaml").write_text("expected: private\n")
    (root / "capsule.interface.mlir").write_text("module {}\n")
    (root / "source-capture-inputs.npz").write_bytes(b"public operand bytes")
    return document


@pytest.mark.parametrize("explicit", [False, True])
def test_live_and_frozen_capture_references_are_masked_with_inputs_preserved(tmp_path, monkeypatch, explicit):
    repo = tmp_path / "repo"
    corpus = repo / "merlin/contract/capsules"
    original = corpus / "_perf/program"
    _capsule(original, explicit=explicit)
    target = SimpleNamespace(capsule_corpus=corpus / "isa", corpus_siblings=lambda: ["merlin/contract/capsules/_perf"])
    monkeypatch.setattr(answers, "repo_root", lambda: repo)
    monkeypatch.setattr(answers, "contract_resource_roots", lambda *_args: ())
    live = answers.golden_files(target)
    assert original / "source-capture-golden.npy" in live
    if explicit:
        assert original / "references/complete.bin" in live
    work = tmp_path / "work"
    bundle = {"allowed": [{"path": str(corpus)}]}
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    monkeypatch.setattr(answers, "_support_package_dirs", lambda: [])
    monkeypatch.setattr(bwrap, "contract_resource_roots", lambda *_args: ())
    manifest = bwrap.materialize_bundle_inputs(work, bundle, repo=repo)
    source = bwrap.bundle_snapshot_root(work) / manifest["grants"][0]["snapshot"]
    try:
        # Reopen pinned metadata after the original declaration and answers disappear.
        shutil.rmtree(original)
        bwrap.verify_bundle_snapshot(work, bundle, repo=repo)
        surfaces = bwrap._frozen_answer_surfaces(work, bundle, target, [], repo)
        argv = ["--ro-bind", str(source), str(corpus)]
        assert bwrap.is_exposed(argv, corpus / "_perf/program/source-capture-golden.npy")
        masked = bwrap.apply_answer_masks(argv, surfaces)
        assert not bwrap.is_exposed(masked, corpus / "_perf/program/source-capture-golden.npy")
        assert bwrap.is_exposed(masked, corpus / "_perf/program/source-capture-inputs.npz")
        assert bwrap.is_exposed(masked, corpus / "_perf/program/capsule.interface.mlir")
        if explicit:
            assert not bwrap.is_exposed(masked, corpus / "_perf/program/references/complete.bin")
        assert bwrap.coverage_gap(masked, surfaces) == []
    finally:
        bwrap.remove_bundle_snapshot(work)


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_phase2_projection_has_no_expected_values_even_without_live_sources(tmp_path, monkeypatch, explicit):
    original = tmp_path / "original/_perf/program"
    _capsule(original, explicit=explicit)
    frozen = tmp_path / "frozen/program"
    frozen.parent.mkdir()
    shutil.copytree(original, frozen)
    shutil.rmtree(original)
    corpus = SimpleNamespace(
        capsules=[SimpleNamespace(source_dir=frozen, source_relative_path="_perf/program")],
        manifest_sha256="a" * 64, capsules_sha256="b" * 64)
    target = SimpleNamespace(capsule_corpus=original.parents[1] / "isa")
    monkeypatch.setattr(workspace, "answer_surfaces", lambda _: [])
    result = workspace.build_answer_free_agent_inputs(corpus, target, tmp_path / "public-view")
    workspace.verify_answer_free_agent_inputs(result)
    public = result.root / "_perf/program"
    assert not (public / "source-capture-golden.npy").exists()
    assert not (public / "golden.yaml").exists()
    assert not (public / "references/complete.bin").exists()
    assert (public / "source-capture-inputs.npz").read_bytes() == b"public operand bytes"
    assert (public / "capsule.interface.mlir").read_text() == "module {}\n"
    rows = json.loads(result.manifest_path.read_bytes())["files"]
    assert all("golden" not in row["path"] and "complete.bin" not in row["path"] for row in rows)


@pytest.mark.parametrize("path", ["../answer.bin", "/answer.bin", "a/../answer.bin", "a\\answer.bin"])
def test_unsafe_declared_reference_paths_refuse_before_projection(tmp_path, path):
    root = tmp_path / "capsule"
    document = _capsule(root)
    document["reference_closure"]["members"][0]["path"] = path
    with pytest.raises(ValueError, match="safe relative"):
        capsule_answer_members(root, capsule=document)
    assert document["reference_closure"]["schema"] == REFERENCE_SCHEMA
