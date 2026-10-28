"""Family readiness combines completed baseline checks without borrowing stale or variant passes."""
from __future__ import annotations

import json

import pytest
from merlin_experiments.phase1.brokers.selfcheck import _submission_digest
from merlin_experiments.phase1.feedback import family_rollup as rollup
from merlin_experiments.phase1.feedback import reconcile, result_details
from test_family_rollup import EDGE, row, timed
from test_family_rollup import workspace as family_workspace  # noqa: F401 -- pytest fixture

from merlin.common import stat_memo


@pytest.fixture
def workspace(family_workspace):  # noqa: F811 -- pytest fixture injection
    return family_workspace


def _family(workspace, family="contraction"):
    return next(row for row in rollup.build(workspace, edge=EDGE)["families"] if row["family"] == family)


def _check(workspace, identifier, rows, *, stamp=100, digest=None, variant=None, request=None, complete=True):
    channel = workspace / ".qa_channel"
    channel.mkdir(exist_ok=True)
    payload = dict(selfcheck_request_id=identifier, selfcheck_submission="submission",
                   submission_sha256=digest or _submission_digest(workspace / "submission"),
                   graded_at_unix_ns=stamp, per_capsule=rows)
    if variant is not None:
        payload["variant"] = variant
    (channel / f"req_{identifier}.json").write_text(json.dumps(dict(submission="submission", **(request or {}))))
    (channel / f"resp_{identifier}.json").write_text(json.dumps(payload))
    if complete:
        (channel / f"done_{identifier}").write_text("ok")
    return payload


def test_selfchecks_accumulate_without_any_completed_background_grade(workspace):
    (workspace / "qa/latest_completed_grade.json").unlink()
    _check(workspace, "first", [row("D_big_tall", "pass", timed(9000, 1000))])
    found = _family(workspace)
    assert found["correct"] == 1 and found["timed"] == 1
    assert found["failing_count"] == 0 and found["unchecked_count"] == 4
    _check(workspace, "second", [row("D_small_tall", "pass")], stamp=200)
    found = _family(workspace)
    assert found["correct"] == 2 and found["unchecked_count"] == 3
    assert "0 failed, 3 unchecked" in rollup.build(workspace, edge=EDGE)["family_line"]
    assert not (workspace / "qa/latest_completed_grade.json").exists()


def test_newest_current_numerical_result_wins_and_unrun_does_not_erase_it(workspace):
    _check(workspace, "pass", [row("D_wide_fail", "pass")], stamp=100)
    assert _family(workspace)["correct"] == 5
    _check(workspace, "failed", [row("D_wide_fail", "fail")], stamp=200)
    _check(workspace, "gated", [dict(capsule="D_wide_fail", status="gated")], stamp=300)
    found = _family(workspace)
    assert found["correct"] == 4 and found["failing"] == ["D_wide_fail"]


@pytest.mark.parametrize("kind", ["variant", "overrides", "diagnostic", "compile", "action", "incomplete", "stale"])
def test_nonbaseline_or_unfinished_checks_cannot_supply_current_correctness(workspace, kind):
    (workspace / "qa/latest_completed_grade.json").unlink()
    kwargs = {}
    if kind == "variant":
        kwargs["variant"] = dict(applied=True, sha256="a" * 64)
    elif kind == "overrides":
        kwargs["request"] = dict(tuning_overrides=dict(block=2))
    elif kind == "diagnostic":
        kwargs["request"] = dict(candidate_diagnostics=True)
    elif kind == "compile":
        kwargs["request"] = dict(compile_only=True)
    elif kind == "action":
        kwargs["request"] = dict(action="reference-analysis")
    elif kind == "incomplete":
        kwargs["complete"] = False
    elif kind == "stale":
        kwargs["digest"] = "b" * 64
    _check(workspace, "ignored", [row("D_big_tall", "pass")], **kwargs)
    found = _family(workspace)
    assert found["correct"] == found["failing_count"] == 0 and found["unchecked_count"] == 5


def test_source_edit_invalidates_selfchecks_without_rewriting_their_receipts(workspace):
    (workspace / "qa/latest_completed_grade.json").unlink()
    _check(workspace, "earned", [row("D_big_tall", "pass")])
    response = workspace / ".qa_channel/resp_earned.json"
    original = response.read_bytes()
    assert _family(workspace)["correct"] == 1
    (workspace / "submission/backend.py").write_text("import math\n")
    found = _family(workspace)
    assert found["correct"] == 0 and found["unchecked_count"] == 5
    assert response.read_bytes() == original


def test_grade_currency_refreshes_when_source_changes_but_grade_file_does_not(workspace, monkeypatch):
    path = workspace / "qa/latest_completed_grade.json"
    grade = json.loads(path.read_text())
    for result in grade["per_capsule"]:
        result["graded"] = dict(currency="current", reach_sha256="original", graded_source_sha256="old")
    path.write_text(json.dumps(grade))
    monkeypatch.setattr(reconcile, "current_reach", lambda _: "original")
    assert _family(workspace)["correct"] == 4
    original = path.read_bytes()
    monkeypatch.setattr(reconcile, "current_reach", lambda _: "edited")
    found = _family(workspace)
    assert found["correct"] == 0 and found["correct_older_version"] == 4
    assert found["failing_count"] == 0 and found["unchecked_count"] == 5 and found["timed"] == 0
    assert found["all_correct"] is False
    assert not rollup.build(workspace, edge=EDGE)["bundles"]
    overlaid = rollup.build(workspace, rows=grade["per_capsule"], edge=EDGE)
    assert all(record["correct"] == 0 for record in overlaid["families"])
    _check(workspace, "new", [row("D_wide_fail", "pass")])
    assert _family(workspace)["correct"] == 1
    assert path.read_bytes() == original


def test_newer_current_grade_failure_beats_an_earlier_selfcheck_pass(workspace, monkeypatch):
    grade = dict(per_capsule=[{**row("D_big_tall", "fail"), "graded": dict(
        reach_sha256="same", graded_at="2026-10-27T16:00:00+00:00")}])
    (workspace / "qa/latest_completed_grade.json").write_text(json.dumps(grade))
    monkeypatch.setattr(reconcile, "current_reach", lambda _: "same")
    _check(workspace, "earlier", [row("D_big_tall", "pass")], stamp=1)
    found = _family(workspace)
    assert found["correct"] == 0 and found["failing"] == ["D_big_tall"]


def test_compact_selfcheck_requires_reopened_detail_and_preserves_numerical_flag(workspace):
    (workspace / "qa/latest_completed_grade.json").unlink()
    full = dict(capsule="D_big_tall", numerical_execution_passed=True, tiers=dict(L3="pass"))
    document = result_details.compact_document(dict(per_capsule=[full]), workspace=workspace)
    compact = document["per_capsule"][0]
    assert compact.get("detail")
    _check(workspace, "compact", [compact])
    assert _family(workspace)["correct"] == 1
    (workspace / compact["detail"]["path"]).unlink()
    result_details._CACHE.clear()
    assert _family(workspace)["correct"] == 0


@pytest.fixture
def cache_reads(monkeypatch):
    from merlin_experiments.phase1.feedback import idle_performance

    stat_memo.clear()
    monkeypatch.setattr(stat_memo, "_DISABLED", False)
    monkeypatch.setattr(stat_memo, "SETTLED_NS", -(10**12))
    reads = []
    original = idle_performance._load
    monkeypatch.setattr(idle_performance, "_load", lambda path: reads.append(path.name) or original(path))
    yield reads
    stat_memo.clear()


def test_warm_family_reader_does_not_reparse_response_bodies_and_new_completion_is_incremental(
        workspace, cache_reads):
    _check(workspace, "first", [row("D_big_tall", "pass")])
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]
    identity = rollup.selfcheck_identity(workspace)
    assert cache_reads == ["resp_first.json", "req_first.json"]
    cache_reads.clear()
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]
    assert rollup.selfcheck_identity(workspace) == identity and cache_reads == []
    _check(workspace, "second", [row("D_small_tall", "pass")], stamp=200, complete=False)
    assert set(rollup.selfcheck_rows(workspace)) == {"D_big_tall"}
    before_done = rollup.selfcheck_identity(workspace)
    (workspace / ".qa_channel/done_second").write_text("ok")
    assert set(rollup.selfcheck_rows(workspace)) == {"D_big_tall", "D_small_tall"}
    assert rollup.selfcheck_identity(workspace) != before_done
    assert cache_reads == ["resp_second.json", "req_second.json"]


def test_cached_join_invalidates_on_request_response_done_and_source_changes(workspace, cache_reads):
    _check(workspace, "joined", [row("D_big_tall", "pass")])
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]
    channel = workspace / ".qa_channel"
    identity = rollup.selfcheck_identity(workspace)
    (channel / "req_joined.json").write_text(json.dumps(dict(submission="submission", compile_only=True)))
    assert rollup.selfcheck_rows(workspace) == {}
    assert rollup.selfcheck_identity(workspace) != identity
    _check(workspace, "joined", [row("D_big_tall", "fail")], stamp=200)
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]["status"] == "fail"
    (channel / "done_joined").unlink()
    assert rollup.selfcheck_rows(workspace) == {}
    (channel / "done_joined").write_text("ok")
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]["status"] == "fail"
    before_edit = rollup.selfcheck_identity(workspace)
    (workspace / "submission/backend.py").write_text("import math\n")
    assert rollup.selfcheck_rows(workspace) == {}
    assert rollup.selfcheck_identity(workspace) != before_edit


def test_missing_or_changed_pinned_detail_is_retried_without_reparsing_response(workspace, cache_reads):
    full = dict(capsule="D_big_tall", numerical_execution_passed=True, tiers=dict(L2="pass"))
    compact = result_details.compact_document(dict(per_capsule=[full]), workspace=workspace)["per_capsule"][0]
    detail = workspace / compact["detail"]["path"]
    original = detail.read_bytes()
    detail.unlink()
    _check(workspace, "detail", [compact])
    assert rollup.selfcheck_rows(workspace) == {}
    missing = rollup.selfcheck_identity(workspace)
    detail.write_bytes(original)
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]
    assert rollup.selfcheck_identity(workspace) != missing
    detail.write_text("{}")
    assert rollup.selfcheck_rows(workspace) == {}
    detail.write_bytes(original)
    assert rollup.selfcheck_rows(workspace)["D_big_tall"]
    assert cache_reads.count("resp_detail.json") == 1
