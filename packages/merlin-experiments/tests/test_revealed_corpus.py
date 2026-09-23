"""Installed reveal admission: immutable member evidence and one-read manifest binding."""

import json
from pathlib import Path

import pytest
from merlin_experiments.phase2 import revealed_corpus as RC


@pytest.fixture
def reveal(tmp_path):
    root = tmp_path / "reveal"
    member = root / "members" / "sample"
    member.mkdir(parents=True)
    descriptor = {
        "name": "sample",
        "inputs": [
            {"name": "lhs", "shape": [2, 3], "dtype": "i8"},
            {"name": "rhs", "shape": [3, 4], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {"lhs": "lhs", "weight": "rhs", "out": "result"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    (member / "capsule.yaml").write_text(json.dumps(descriptor))
    manifest = root / "holdout_manifest.json"
    tree = RC.tree_without_manifest(root, manifest)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "generated_performance_holdout_reveal",
                "domain": {"target": "fixture_device"},
                "cohorts": {"sample_cohort": {"family": "sample_family", "member_count": 1}},
                "members": [
                    {
                        "name": "sample",
                        "path": "members/sample",
                        "family": "sample_family",
                        "cohort": "sample_cohort",
                        "M": 2,
                        "N": 4,
                        "K": 3,
                    }
                ],
                "corpus": tree,
            }
        )
    )
    digest = RC.sha_file(manifest)
    for path in (root, *root.rglob("*")):
        path.chmod(0o555 if path.is_dir() else 0o444)
    return manifest, digest, tree["sha256"]


def test_installed_reader_accepts_exact_reveal(reveal):
    manifest, digest, corpus = reveal
    members = RC.load_revealed_members(
        manifest, expected_manifest_sha256=digest, expected_corpus_sha256=corpus, expected_target="fixture_device"
    )
    assert len(members) == 1
    assert members[0].name == "sample"
    assert members[0].workload["shape"] == {"m": 2, "n": 4, "k": 3}


@pytest.mark.parametrize("change", ["changed_member", "missing_member", "changed_manifest", "missing_manifest"])
def test_reveal_refuses_changed_or_missing_evidence(reveal, change):
    manifest, digest, corpus = reveal
    member = manifest.parent / "members/sample/capsule.yaml"
    target = manifest if change.endswith("manifest") else member
    target.parent.chmod(0o755)
    target.chmod(0o644)
    if change.startswith("missing"):
        target.unlink()
    else:
        target.write_text("{}")
        target.chmod(0o444)
    target.parent.chmod(0o555)
    with pytest.raises(RC.QualificationError):
        RC.load_revealed_members(manifest, expected_manifest_sha256=digest, expected_corpus_sha256=corpus)


def test_manifest_hash_and_parse_use_one_read(reveal, monkeypatch):
    manifest, digest, _ = reveal
    original = Path.read_bytes
    reads = []

    def read(path):
        payload = original(path)
        if path == manifest:
            reads.append(path)
            path.chmod(0o644)
            path.write_text("not valid JSON")
            path.chmod(0o444)
        return payload

    monkeypatch.setattr(Path, "read_bytes", read)
    assert RC.load_revealed_members(manifest, expected_manifest_sha256=digest)[0].name == "sample"
    assert reads == [manifest]
