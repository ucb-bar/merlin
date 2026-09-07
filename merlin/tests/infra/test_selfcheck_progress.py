"""Incremental self-check feedback remains redacted and separate from the final verdict."""
from __future__ import annotations

import importlib.util
import json

from merlin.common.paths import merlin_dir


def _selfcheck_module():
    path = merlin_dir() / "experiments/capsule_bench/harness/agent_selfcheck.py"
    spec = importlib.util.spec_from_file_location("agent_selfcheck_progress", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_progress_snapshot_counts_only_durable_capsule_statuses(tmp_path):
    selfcheck = _selfcheck_module()
    root = tmp_path / "runs" / "fake-capsule-bench"
    for name, status in (("fast_pass", "pass"), ("numeric_bug", "fail")):
        result = root / name / "capsule_result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({
            "capsule": name,
            "status": status,
            "numeric": {"first_mismatch": {"expected": 123, "observed": 0}},
        }))
    partial = root / "still_writing" / "capsule_result.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"capsule":')

    progress = selfcheck._progress_snapshot(root, expected=4, started_ns=10)

    assert progress["n_finished"] == 2
    assert progress["n_remaining"] == 2
    assert progress["counts"] == {"fail": 1, "pass": 1}
    assert progress["per_capsule"] == [
        {"capsule": "fast_pass", "status": "pass"},
        {"capsule": "numeric_bug", "status": "fail"},
    ]
    encoded = json.dumps(progress)
    assert "first_mismatch" not in encoded and "123" not in encoded


def test_atomic_progress_publish_never_leaves_a_partial_file(tmp_path):
    selfcheck = _selfcheck_module()
    out = tmp_path / "progress.json"

    selfcheck._atomic_json(out, {"n_finished": 3})

    assert json.loads(out.read_text()) == {"n_finished": 3}
    assert not list(tmp_path.glob(".*.tmp"))
