"""Run navigation projects existing records without moving or requalifying outputs."""

import hashlib
import json

from merlin_experiments.runner import status


def test_status_exposes_each_phase_and_latest_segment_without_mutation(tmp_path):
    plan = {
        "experiment": "workflow",
        "target": "fixture",
        "evidence_authority": "phase engine; not process exit",
        "phases": {
            "0": {"adapter": "capsule_derivation", "engine_output": "/original/corpus", "resume_policy": "retry"},
            "1": {"adapter": "capsule_bench", "engine_output": "/original/compiler", "resume_policy": "native_flag"},
            "2": {
                "adapter": "model_portfolio",
                "engine_output": "/original/segment-0001",
                "resume_policy": "checkpoint_segment",
            },
        },
    }
    payload = json.dumps(plan).encode()
    record = {
        "plan_sha256": hashlib.sha256(payload).hexdigest(),
        "state": "interrupted",
        "attempts": [
            {"phase": "0", "state": "execution_succeeded", "log": "/original/generation.log"},
            {"phase": "2", "state": "failed", "engine_output": "/original/segment-0001"},
            {
                "phase": "2",
                "state": "interrupted",
                "engine_output": "/original/segment-0002",
                "log": "/original/optimization.log",
            },
        ],
    }
    (tmp_path / "resolved-plan.json").write_bytes(payload)
    (tmp_path / "orchestration.json").write_text(json.dumps(record))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result = status(tmp_path)
    assert result["phases"]["0"]["engine_output"] == "/original/corpus"
    assert result["phases"]["1"]["state"] == "not_started"
    assert result["phases"]["1"]["attempt_count"] == 0
    assert result["phases"]["2"]["engine_output"] == "/original/segment-0002"
    assert result["phases"]["2"]["latest_log"] == "/original/optimization.log"
    assert result["phases"]["2"]["attempt_count"] == 2
    assert result["attempts"] == record["attempts"]
    assert result["evidence_authority"] == plan["evidence_authority"]
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
