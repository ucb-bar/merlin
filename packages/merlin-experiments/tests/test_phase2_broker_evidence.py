"""Historical receipt policies retain distinct contracts under one evidence owner."""

import json
from pathlib import Path

import pytest
from merlin_experiments.phase2 import broker, broker_evidence, telemetry
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import whole_model as WM
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import access
from merlin.common.digest import sha256_bytes
from merlin.common.paths import repo_root


def test_evidence_owner_is_in_existing_recursive_source_identity():
    record = telemetry._package_source_record()
    for owner in (broker_evidence, BP, CF, WM):
        source = Path(owner.__file__)
        assert record["members"][source.name] == sha256_bytes(source.read_bytes())
        assert any(
            source.resolve().is_relative_to(location.resolve())
            for item in access.MODULE_ACCESS
            if item.origin == "grader"
            for location in access.module_locations(repo_root(), item)
        )
    for owner, names in (
        (CF, ("validate_redacted_feedback", "verify_broker_receipts")),
        (WM, ("verify_global_broker_receipts",)),
    ):
        for name in names:
            assert getattr(owner, name).__module__ == owner.__name__
            assert not hasattr(broker, name)


def test_stage_requires_feedback_but_global_does_not(tmp_path):
    path = tmp_path / "receipts.jsonl"
    row = {
        "receipt_schema_version": 1,
        "state": "complete",
        "index": 0,
        "action": "inspect",
        "returncode": 0,
        **{key: "a" * 64 for key in ("argv_sha256", "bindings_command_sha256", "stdout_sha256", "stderr_sha256")},
    }
    path.write_text(json.dumps(row) + "\n")
    actions = (broker.BrokerAction("inspect", ("inspect",), (), "inspection", True),)
    audit = {"broker_invocations": [{"action": "inspect", "bindings_sha256": "a" * 64}]}
    assert WM.verify_global_broker_receipts(path, actions=actions, audit=audit)["count"] == 1
    with pytest.raises(StageGateError, match="mandatory tuning GSIM feedback"):
        CF.verify_broker_receipts(path, actions, audit)
