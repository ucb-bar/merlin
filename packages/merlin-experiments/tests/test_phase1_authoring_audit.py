"""Invocation-local answer policy without importing native admission or providers."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
import yaml
from merlin_experiments.phase1.audit import AnswerAudit

from merlin.targetgen.target_experiment import load_target_experiment


def policy(root, name):
    root.mkdir()
    descriptor = root / "target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": name,
                "capsule_corpus": str(root / "public"),
                "answer_surfaces": {"prior_backends": [f"private/{name}_answer"]},
            }
        )
    )
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "allowed_files.txt").write_text("targetgen/generate/public_tool.py\n")
    return AnswerAudit.for_descriptor(load_target_experiment(descriptor), bundle)


def transcript(root, path):
    out = root / "transcript.jsonl"
    out.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read",
                            "name": "Read",
                            "input": {"file_path": path},
                        }
                    ]
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read",
                            "content": "nonempty evidence",
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    return out


def test_descriptor_tokens_remain_invocation_local_interleaved(tmp_path, monkeypatch):
    first, second = policy(tmp_path / "first", "first"), policy(tmp_path / "second", "second")
    evidence = transcript(tmp_path, "private/first_answer/code.py")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(tmp_path / "missing.yaml"))
    assert not first.audit_transcript(evidence)["clean"]
    assert second.audit_transcript(evidence)["clean"]
    assert not first.audit_transcript(evidence)["clean"]
    with pytest.raises(FrozenInstanceError):
        first.bundle_dir = second.bundle_dir


def test_grants_are_reread_and_denies_cannot_launder_answer_tokens(tmp_path):
    selected = policy(tmp_path / "selected", "selected")
    evidence = transcript(tmp_path, "targetgen/generate/public_tool.py")
    assert selected.audit_transcript(evidence)["granted_reads"] == 1
    (selected.bundle_dir / "allowed_files.txt").write_text("")
    assert not selected.audit_transcript(evidence)["clean"]
    (selected.bundle_dir / "allowed_files.txt").write_text("private/selected_answer/code.py\n")
    evidence = transcript(tmp_path, "private/selected_answer/code.py")
    assert not selected.audit_transcript(evidence)["clean"]
    with pytest.raises(ValueError, match="absolute"):
        selected.audit_transcript(evidence, bundle="ambient-bundle-id")
