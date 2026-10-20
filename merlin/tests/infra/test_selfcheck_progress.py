"""Incremental self-check feedback remains redacted and separate from the final verdict."""

from __future__ import annotations

import importlib.util
import json

from phase1_feedback import feedback_source


def _selfcheck_module():
    path = feedback_source("agent_selfcheck")
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
        result.write_text(
            json.dumps(
                {
                    "capsule": name,
                    "status": status,
                    "numeric": {"first_mismatch": {"expected": 123, "observed": 0}},
                }
            )
        )
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


def test_progress_publishes_only_observed_operator_tier_statuses(tmp_path):
    selfcheck = _selfcheck_module()
    root = tmp_path / "runs"
    records = {
        "operator": {
            "capsule": "operator", "kind": "matmul", "status": "fail",
            "tiers": {
                "L0": {"status": "pass"}, "L2": {"status": "pass", "numeric": {"expected": "ANSWER_SENTINEL"}},
                "L3": {"status": "fail", "console": "CONSOLE_SENTINEL"},
                "L4": {"status": "pass"}, "PRIVATE_SENTINEL": {"status": "pass"},
            },
        },
        "model": {"capsule": "model", "kind": "model", "status": "pass", "tiers": {"L3": {"status": "pass"}}},
        "model_marker": {
            "capsule": "model_marker", "status": "pass", "model_execution_check": {},
            "tiers": {"L3": {"status": "pass"}},
        },
        "overall_only": {"capsule": "overall_only", "status": "pass"},
        "malformed_tiers": {
            "capsule": "malformed_tiers", "status": "incomplete",
            "tiers": {
                "L0": "pass", "L1": {"status": "ANSWER_SENTINEL"}, "L2": {"status": None},
                "L3": {"status": {"expected": "ANSWER_SENTINEL"}},
            },
        },
        "not_a_record": [],
    }
    for name, record in records.items():
        path = root / name / "capsule_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
    progress = selfcheck._progress_snapshot(root, expected=5, started_ns=10)
    rows = {row["capsule"]: row for row in progress["per_capsule"]}
    assert rows["operator"]["tiers"] == {"L0": "pass", "L2": "pass", "L3": "fail"}
    assert all("tiers" not in rows[name] for name in ("model", "model_marker", "overall_only", "malformed_tiers"))
    assert progress["n_finished"] == 5
    assert "SENTINEL" not in json.dumps(progress)


def test_progress_republishes_tier_changes_with_same_finished_count(tmp_path, monkeypatch):
    selfcheck = _selfcheck_module()
    rows = [
        {"capsule": "operator", "status": "incomplete", "tiers": {"L2": "pass"}},
        {"capsule": "operator", "status": "pass", "tiers": {"L2": "pass", "L3": "pass"}},
    ]
    step = [0]
    published = []

    class Stop:
        def is_set(self):
            return step[0] >= 2

        def wait(self, timeout):
            step[0] += 1

    monkeypatch.setattr(selfcheck, "_progress_snapshot", lambda *args, **kwargs: {
        "n_finished": 1, "per_capsule": [dict(rows[min(step[0], 1)])],
    })
    monkeypatch.setattr(selfcheck, "_atomic_json", lambda path, doc: published.append(doc))
    selfcheck._progress_publisher(tmp_path / "progress.json", tmp_path, expected=1, started_ns=10, stop=Stop())
    assert len(published) == 3
    assert published[0]["per_capsule"][0]["tiers"] == {"L2": "pass"}
    assert published[1]["per_capsule"][0]["tiers"] == {"L2": "pass", "L3": "pass"}
    assert published[2]["status"] == "grading_complete"
