"""Receipt admission consumes only pinned bytes and finite resource assignments."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2 import chia_launch as CL


@pytest.fixture
def launch(tmp_path, monkeypatch):
    source = tmp_path / "coordinator.py"
    source.write_text("# synthetic entrypoint\n")
    command = [sys.executable, str(source), "--experiment-id", "synthetic"]
    trace = tmp_path / "trace.py"
    trace.write_text("# synthetic source identity; never executed\n")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# synthetic wrapper\n")
    plan = {
        "command": command,
        "command_artifacts": CL.command_artifacts(command),
        "wrapper": {"path": str(wrapper.resolve()), "sha256": CL._sha_file(wrapper)},
        "chia_trace": {"path": str(trace.resolve()), "sha256": CL._sha_file(trace)},
        "launch_policy": CL.policy_identity(),
    }
    plan["sha256"] = hashlib.sha256(CL._canonical(plan)).hexdigest()
    receipt = {
        **plan,
        "schema": "merlin.chia-agentic-perf-launch.v2",
        "status": "assigned_before_coordinator",
        "plan": plan,
        "plan_sha256": plan["sha256"],
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"codex_slots": 1.0, "gsim_slots": 1.0},
    }
    path = tmp_path / "receipt.json"

    def save(document):
        if path.exists():
            path.chmod(0o644)
        # The untrusted JSON decoder accepts nonfinite numbers; admission must refuse them.
        path.write_text(json.dumps(document))
        path.chmod(0o444)
        monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", CL._sha_file(path))

    save(receipt)
    monkeypatch.setenv("MERLIN_CHIA_ENVELOPE_PLAN_SHA256", plan["sha256"])
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT", str(path))
    return {"command": command, "wrapper": wrapper, "environment": os.environ}, receipt, path, save


@pytest.mark.parametrize("assignment", [float("nan"), float("inf"), -float("inf"), True, "1", 0.5])
def test_nonfinite_or_invalid_assignment_refuses(launch, assignment):
    arguments, receipt, _, save = launch
    receipt["assigned_resources"]["gsim_slots"] = assignment
    save(receipt)
    with pytest.raises(CL.ExperimentError, match="exact assigned invocation"):
        CL.verify_launch_receipt(**arguments)


@pytest.mark.parametrize("field", [None, "plan", "required_resources", "assigned_resources", "wrapper", "chia_trace"])
def test_malformed_receipt_mapping_refuses(launch, field):
    arguments, receipt, _, save = launch
    if field is None:
        receipt = []
    else:
        receipt[field] = ["not a mapping"]
    save(receipt)
    with pytest.raises(CL.ExperimentError, match="malformed"):
        CL.verify_launch_receipt(**arguments)


def test_receipt_parses_the_single_hashed_read(launch, monkeypatch):
    arguments, _, path, _ = launch
    read_bytes, read_text = Path.read_bytes, Path.read_text
    reads = []

    def bytes_once(selected):
        if selected == path:
            reads.append(selected)
            assert len(reads) == 1, "receipt was read again after hashing"
        return read_bytes(selected)

    def no_text_reread(selected, *args, **kwargs):
        assert selected != path, "parsed an unpinned second receipt read"
        return read_text(selected, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", bytes_once)
    monkeypatch.setattr(Path, "read_text", no_text_reread)
    verified = CL.verify_launch_receipt(**arguments)
    assert verified["assigned_resources"]["gsim_slots"] == 1.0
    assert reads == [path]
