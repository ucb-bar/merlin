"""Incremental self-check feedback remains redacted and separate from the final verdict."""

from __future__ import annotations

import json


def _selfcheck_module():
    # Imported as the package module: loading the file standalone cannot resolve its relative imports.
    from merlin_experiments.phase1.feedback import selfcheck

    return selfcheck


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
    # Clock fields are dropped: a wall-clock nanosecond stamp may contain the golden digits.
    encoded = json.dumps({k: v for k, v in progress.items() if not k.endswith("_ns")})
    assert "first_mismatch" not in encoded and "123" not in encoded


def test_atomic_progress_publish_never_leaves_a_partial_file(tmp_path):
    selfcheck = _selfcheck_module()
    out = tmp_path / "progress.json"

    selfcheck._atomic_json(out, {"n_finished": 3})

    assert json.loads(out.read_text())["n_finished"] == 3
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

    from merlin_experiments.phase1.feedback import selfcheck_progress

    monkeypatch.setattr(selfcheck_progress, "progress_snapshot", lambda *args, **kwargs: {
        "n_finished": 1, "per_capsule": [dict(rows[min(step[0], 1)])],
    })
    monkeypatch.setattr(selfcheck_progress, "publish", lambda path, doc: published.append(doc))
    selfcheck._progress_publisher(tmp_path / "progress.json", tmp_path, expected=1, started_ns=10, stop=Stop())
    assert len(published) == 3
    assert published[0]["per_capsule"][0]["tiers"] == {"L2": "pass"}
    assert published[1]["per_capsule"][0]["tiers"] == {"L2": "pass", "L3": "pass"}
    assert published[2]["status"] == "grading_complete"


def _write_result(root, name, record):
    path = root / name / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def _write_numerics(root, name, status):
    report = root / name / "generated" / "numeric_report.yaml"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"policy: exact_int\nstatus: {status}\nmismatch_count: 0\nfirst_mismatch: null\n")
    return report


def test_numerics_are_published_as_soon_as_they_are_decided(tmp_path):
    """A case whose numerics are decided is reported before its later checks finish -- provisionally."""
    from merlin_experiments.phase1.feedback import selfcheck_progress as SP

    root = tmp_path / "runs"
    _write_numerics(root, "big_case", "pass")
    _write_numerics(root, "wrong_case", "fail")
    (root / "torn_case" / "generated").mkdir(parents=True)
    (root / "torn_case" / "generated" / "numeric_report.yaml").write_text("status: [")
    _write_numerics(root, "done_case", "pass")
    _write_result(root, "done_case", {
        "capsule": "done_case", "status": "fail", "numeric": {"status": "pass", "max_abs_diff": 77},
        "failure": {"plane": "host_compute", "category": "host_compute_in_accelerator_group",
                    "detail": "DETAIL_SENTINEL"},
    })

    progress = SP.progress_snapshot(root, expected=4, started_ns=1)

    assert progress["per_capsule"] == [
        {"capsule": "big_case", "status": "running", "numerics": "pass", "final": False},
        {"capsule": "done_case", "status": "fail", "numerics": "pass",
         "failure_plane": "host_compute", "failure_category": "host_compute_in_accelerator_group"},
        {"capsule": "wrong_case", "status": "running", "numerics": "fail", "final": False},
    ]
    # Final counts stay final-only; the decided numerics are counted separately.
    assert progress["n_finished"] == 1 and progress["counts"] == {"fail": 1}
    assert progress["n_numerics_decided"] == 3 and progress["numerics_counts"] == {"pass": 2, "fail": 1}
    encoded = json.dumps(progress["per_capsule"])
    assert "SENTINEL" not in encoded and "77" not in encoded


def test_each_result_is_parsed_once_per_version(tmp_path, monkeypatch):
    """The snapshot repeats every half second inside the grader; an unchanged result is not re-read."""
    from merlin_experiments.phase1.feedback import selfcheck_progress as SP

    root = tmp_path / "runs"
    path = _write_result(root, "case", {"capsule": "case", "status": "pass"})
    reads = []
    original = SP._finished_row
    monkeypatch.setattr(SP, "_finished_row", lambda p: reads.append(p) or original(p))
    cache: dict = {}
    for _ in range(3):
        SP.progress_snapshot(root, expected=1, started_ns=1, cache=cache)
    assert len(reads) == 1
    path.write_text(json.dumps({"capsule": "case", "status": "fail", "padding": "x"}))
    assert SP.progress_snapshot(root, expected=1, started_ns=1, cache=cache)["counts"] == {"fail": 1}
    assert len(reads) == 2


def test_the_agent_client_prints_each_case_when_decided_and_again_when_final():
    from merlin_experiments.phase1.tools import selfcheck as client

    seen: dict = {}
    first = {"per_capsule": [
        {"capsule": "big", "status": "running", "numerics": "pass", "final": False},
        {"capsule": "small", "status": "pass", "numerics": "pass", "tiers": {"L2": "pass"}},
    ]}
    lines = client._new_case_lines(seen, first)
    assert lines == [
        "[self-check] big: numerics pass (provisional: later checks still running and can still fail it)",
        "[self-check] small: pass; numerics pass",
    ]
    assert client._new_case_lines(seen, first) == []  # unchanged rows are not repeated
    final = {"per_capsule": [
        {"capsule": "big", "status": "fail", "numerics": "pass",
         "failure_plane": "host_compute", "failure_category": "host_compute_in_accelerator_group"},
        {"capsule": "small", "status": "pass", "numerics": "pass", "tiers": {"L2": "pass"}},
        "not a row",
    ]}
    assert client._new_case_lines(seen, final) == [
        "[self-check] big: fail (host_compute/host_compute_in_accelerator_group); numerics pass",
    ]
    assert client._case_rows({"per_capsule": None}) == [] and client._case_rows(None) == []


def test_a_check_that_ends_without_a_verdict_returns_the_cases_it_decided(tmp_path):
    from merlin_experiments.phase1.feedback import selfcheck_progress as SP

    root = tmp_path / "runs"
    _write_result(root, "done", {"capsule": "done", "status": "pass", "numeric": {"status": "pass"}})
    _write_numerics(root, "slow", "pass")
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps(SP.progress_snapshot(root, expected=3, started_ns=1)))

    partial = SP.partial_results(progress)["partial"]
    assert [row["capsule"] for row in partial["per_capsule"]] == ["done", "slow"]
    assert partial["n_expected"] == 3 and partial["n_numerics_decided"] == 2
    assert SP.partial_results(tmp_path / "absent.json") == {}


def test_published_progress_keeps_provisional_numerics_and_writes_no_case_details(tmp_path, monkeypatch):
    """The publisher writes its small rows as they are: a running case keeps ``numerics``/``final`` for the
    client's line, and no per-row detail file lands in the workspace's ``qa/cases``."""
    import threading

    from merlin_experiments.phase1.feedback import selfcheck_progress as SP

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    root = tmp_path / "runs"
    _write_numerics(root, "big_case", "pass")
    _write_result(root, "done_case", {"capsule": "done_case", "status": "pass"})
    out = workspace / ".qa_channel" / "progress_1.json"
    stop = threading.Event()
    stop.set()  # one final snapshot only
    SP.progress_publisher(out, root, expected=2, started_ns=1, stop=stop)

    published = json.loads(out.read_text())
    assert published["status"] == "grading_complete"
    assert published["per_capsule"] == [
        {"capsule": "big_case", "status": "running", "numerics": "pass", "final": False},
        {"capsule": "done_case", "status": "pass"},
    ]
    assert not (workspace / "qa").exists()
    assert not list(out.parent.glob(".*.tmp"))
