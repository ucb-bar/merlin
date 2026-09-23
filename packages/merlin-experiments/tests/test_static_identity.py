"""Portable static identities using real bytes, without native controller imports."""

import hashlib

import pytest
from merlin_experiments.phase2 import static_identity as SI


def test_relocated_sources_preserve_identity_and_refuse_drift_or_escape(tmp_path):
    records = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        source = root / "policy.py"
        source.write_bytes(b"policy = 1\n")
        record = {
            "schema": "global_host_verification_policy_v1",
            "sources": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
        }
        records.append((root, source, record))
    first, second = records
    assert SI.host_policy_content_sha256(first[2], source_root=first[0]) == SI.host_policy_content_sha256(
        second[2], source_root=second[0]
    )
    with pytest.raises(ValueError, match="escaped"):
        SI.host_policy_content_sha256(first[2], source_root=second[0])
    first[1].write_bytes(b"policy = 2\n")
    with pytest.raises(ValueError, match="bytes changed"):
        SI.host_policy_content_sha256(first[2], source_root=first[0])


def test_static_copy_removes_nested_dynamic_evidence_without_mutating_input():
    original = {
        "nested": [{"shape": [2, 3], "probe_receipts": [1], "emission_execution": {"ok": True}}],
        "wall_seconds": 10,
    }
    copied = SI.static_only_copy(original)
    assert copied == {"nested": [{"shape": [2, 3]}]}
    copied["nested"][0]["shape"].append(4)
    assert original["nested"][0]["shape"] == [2, 3]
    assert original["nested"][0]["probe_receipts"] == [1]
    assert original["wall_seconds"] == 10
