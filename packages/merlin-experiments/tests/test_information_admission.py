"""A background grade admits by information value; an edit never withholds or closes its admission."""

import json
import os
import shutil

import pytest
from merlin_experiments.phase1.feedback import background_admission as admission

from merlin.targetgen.capsule_scheduling import (
    admission_relevance,
    information_rank,
    newest_outcomes,
    observe_admission,
    rolling_admission,
    run_pipelined,
    timing_history,
)

SOURCE = "a" * 64
OTHER = "b" * 64


def _cap(name, shape=(4,)):
    return {"name": name, "inputs": [{"name": "x", "shape": list(shape)}]}


def _result(root, name, *, mtime, status="pass", source=SOURCE, cost=10.0, document=None):
    path = root / "runs" / "fixture-capsule-bench" / name / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = document if document is not None else json.dumps({
        "capsule": name, "status": status,
        "failure": None if status == "pass" else {"category": "FUNCTIONAL_MISMATCH"},
        "compiler_source": {"candidate_sha256": source},
        "tiers": {"L2": {"status": status, "measured_now": True,
                         "timing": {"sim_active_s": cost, "build_s": 0.0}}},
    })
    path.write_text(body)
    os.utime(path, ns=(mtime, mtime))


@pytest.fixture()
def history(tmp_path):
    root = tmp_path / "fruns_1"
    _result(root, "current_pass", mtime=10**18, cost=1.0)            # passed on these exact bytes
    _result(root, "stale_pass", mtime=10**18, source=OTHER, cost=30.0)  # passed on other bytes
    _result(root, "failing_cheap", mtime=10**18, status="fail", cost=2.0)
    _result(root, "failing_long", mtime=10**18, status="fail", cost=500.0)
    return (root,)


def _order(caps, **kwargs):
    order = []
    rows = run_pipelined(caps, lambda c, n: order.append(c["name"]) or {"capsule": c["name"], "status": "pass"},
                         max_workers=1, **kwargs)
    return order, rows


NAMES = ("failing_long", "current_pass", "missing", "stale_pass", "failing_cheap")


def test_failing_and_missing_go_first_then_stale_passes_then_current_passes(monkeypatch, history):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "information-first")
    with timing_history(history), admission_relevance(source_sha256=SOURCE):
        order, rows = _order([_cap(n) for n in NAMES])
    # "missing" has no measured cost: it ranks as the median priced member (16 s), never as free.
    # A pass earned on older source waits behind every failing or never-graded member, however long.
    assert order == ["failing_cheap", "missing", "failing_long", "stale_pass", "current_pass"]
    assert [row["capsule"] for row in rows] == list(NAMES), "start order never moves a result"


def test_unknown_graded_source_makes_every_result_stale(monkeypatch, history):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "information-first")
    with timing_history(history), admission_relevance(source_sha256=None):
        order, _ = _order([_cap(n) for n in NAMES])
    assert order == ["failing_cheap", "missing", "failing_long", "current_pass", "stale_pass"]
    assert information_rank({"passed": True, "source_sha256": SOURCE}, SOURCE) == 2
    assert information_rank({"passed": True, "source_sha256": None}, SOURCE) == 1
    assert information_rank({"passed": True, "source_sha256": SOURCE}, None) == 1
    assert information_rank({"passed": False, "source_sha256": SOURCE}, SOURCE) == 0
    assert information_rank(None, SOURCE) == 0


def test_cheap_members_reaching_more_components_go_first(monkeypatch, tmp_path):
    root = tmp_path / "fruns_1"
    _result(root, "wide", mtime=10**18, status="fail", cost=10.0)
    _result(root, "narrow", mtime=10**18, status="fail", cost=4.0)
    reach = {"wide": frozenset({"parse", "lower", "emit", "link", "<unattributed>"}), "narrow": frozenset({"parse"})}
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "information-first")
    with timing_history((root,)), admission_relevance(source_sha256=SOURCE, reach=lambda c: reach[c["name"]]):
        order, _ = _order([_cap("narrow"), _cap("wide")])
    assert order == ["wide", "narrow"]  # 2 s per reached component against 4 s
    with timing_history((root,)), admission_relevance(source_sha256=SOURCE, reach=lambda c: 1 / 0):
        order, _ = _order([_cap("narrow"), _cap("wide")])
    assert order == ["narrow", "wide"]  # unknown reach orders as one component and skips nothing


def test_information_first_is_the_background_default_and_explicit_orders_override(monkeypatch, history):
    monkeypatch.delenv("MERLIN_CAPSULE_ORDER", raising=False)
    caps = [_cap("failing_long", shape=(64, 64)), _cap("failing_cheap", shape=(2,))]
    closed = []
    with timing_history(history), admission_relevance(source_sha256=SOURCE), observe_admission(closed.append):
        assert _order(caps)[0] == ["failing_cheap", "failing_long"]
        monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
        assert _order(caps)[0] == ["failing_long", "failing_cheap"]
        monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "legacy")
        assert _order(caps)[0] == ["failing_long", "failing_cheap"]  # declared operand volume
    monkeypatch.delenv("MERLIN_CAPSULE_ORDER")
    with timing_history(history):
        assert _order(caps)[0] == ["failing_long", "failing_cheap"]  # foreground keeps legacy
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "cheapest")
    with pytest.raises(ValueError, match="information-first"):
        _order(caps)


def test_caller_admission_priority_still_leads_the_information_order(monkeypatch, history):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "information-first")
    attempts = {"failing_cheap": 3, "failing_long": 1, "stale_pass": 1}
    with timing_history(history), admission_relevance(source_sha256=SOURCE), \
            rolling_admission(admission_priority=lambda c: attempts[c["name"]],
                              admission_costs={"failing_cheap": 2.0, "failing_long": 500.0, "stale_pass": 30.0}):
        order, _ = _order([_cap(n) for n in ("failing_long", "failing_cheap", "stale_pass")])
    # Priority leads (failing_cheap last); within one priority a failing member precedes a stale pass.
    assert order == ["failing_long", "stale_pass", "failing_cheap"]


def test_newest_outcome_skips_a_result_being_written_and_a_foreign_row(tmp_path):
    old, new = tmp_path / "fruns_1", tmp_path / "fruns_2"
    _result(old, "case", mtime=10**18, status="fail")
    _result(new, "case", mtime=2 * 10**18, document="{")  # still being written
    _result(new, "other", mtime=2 * 10**18, document=json.dumps({"capsule": "renamed", "status": "pass"}))
    outcomes = newest_outcomes((old, new))
    assert outcomes == {"case": {"passed": False, "source_sha256": SOURCE}}
    _result(new, "case", mtime=3 * 10**18)
    assert newest_outcomes((old, new))["case"] == {"passed": True, "source_sha256": SOURCE}


# --------------------------------------------------------------- per-member supersession


def test_a_superseded_member_is_withheld_while_unaffected_members_are_still_admitted():
    closed = []
    with observe_admission(closed.append), admission_relevance(
            superseded=lambda c: "background source superseded" if c["name"] == "b" else None):
        order, rows = _order([_cap(n) for n in "abc"], defer_pending=lambda: None, defer_running=False)
    assert order == ["a", "c"]
    assert [row["status"] for row in rows] == ["pass", "gated", "pass"]
    assert rows[1]["admission_deferred"] is True and rows[1]["execution_status"] == "not_started"
    assert rows[1]["failure"] == {"plane": "gate", "category": "ADMISSION_DEFERRED",
                                  "detail": "background source superseded"}
    assert closed == ["background source superseded"]


def test_an_unattributable_member_fails_closed_and_a_fully_superseded_queue_closes():
    def broken(cap):
        raise OSError("source unreadable")

    with admission_relevance(superseded=broken):
        order, rows = _order([_cap(n) for n in "ab"])
    assert order == []
    assert {row["failure"]["detail"] for row in rows} == {"background source dependencies unavailable"}
    with admission_relevance(superseded=lambda c: "background source superseded"):
        rows = run_pipelined([_cap(n) for n in "abc"], lambda c, n: pytest.fail("admitted"), max_workers=2)
    assert all(row["status"] == "gated" and row["admission_deferred"] for row in rows)


# --------------------------------------------------------------- guard: shared versus narrower reach


def _workspace(tmp_path):
    ws = tmp_path / "workspace"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text(
        "language: python\ncommands:\n  parse:\n    argv: [python3, compiler.py]\n"
        "  lower_interface_to_target:\n    argv: [python3, lower.py]\n"
        "components:\n  parse: [compiler.py]\n  lower_interface_to_target: [lower.py]\n")
    (source / "compiler.py").write_text("print(1)\n")
    (source / "lower.py").write_text("print(2)\n")
    (source / "NOTES.md").write_text("plan one\n")
    candidate = tmp_path / "snapshot" / "submission"
    shutil.copytree(source, candidate)
    return ws, source, candidate


def test_every_case_reaches_every_component_and_no_edit_closes_admission(tmp_path):
    ws, source, candidate = _workspace(tmp_path)
    guard = admission.BackgroundGuard(ws, tmp_path, None, admission.source_digest(ws))
    guard.bind_snapshot(candidate)
    from merlin_experiments.phase1.feedback import reconcile

    from merlin.targetgen.oracle_schedule import UNATTRIBUTED

    # The manifest no declared prefix claims is UNATTRIBUTED: every case depends on it too.
    assert guard.case_reach(_cap("x")) == {"parse", "lower_interface_to_target", UNATTRIBUTED}
    snapshot = guard.reach_identity()
    (source / "NOTES.md").write_text("plan two\n")
    assert guard.pending_reason() is None and reconcile.current_reach(ws) == snapshot  # inert: still current
    (source / "compiler.py").write_text("print(3)\n")  # the entry every case reaches
    assert guard.pending_reason() is None  # an edit never closes admission ...
    assert guard.reason() == "background source superseded" and not guard.current()  # ... it labels the grade
    assert reconcile.current_reach(ws) != snapshot  # and its rows reconcile as an older version


def test_continuous_edits_to_the_shared_entry_still_grade_every_case(tmp_path):
    ws, source, candidate = _workspace(tmp_path)
    guard = admission.BackgroundGuard(ws, tmp_path, None, admission.source_digest(ws))
    guard.bind_snapshot(candidate)
    edits = []

    def execute(cap, parallel):
        edits.append(cap["name"])
        (source / "compiler.py").write_text(f"print({len(edits)})\n")  # edited between every admission
        return {"capsule": cap["name"], "status": "pass"}

    rows = run_pipelined([_cap(n) for n in ("a", "b", "c", "d")], execute, max_workers=2,
                         defer_pending=guard.pending_reason, defer_running=False)
    assert sorted(edits) == ["a", "b", "c", "d"]
    assert [row["status"] for row in rows] == ["pass"] * 4
    assert (candidate / "compiler.py").read_text() == "print(1)\n"  # each result belongs to its snapshot


def test_the_joint_deadline_still_stops_new_admission(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control

    ws, source, candidate = _workspace(tmp_path)
    guard = admission.BackgroundGuard(ws, tmp_path, None, admission.source_digest(ws), finish_bounded_snapshot=True)
    guard.bind_snapshot(candidate)
    monkeypatch.setattr(joint_control, "stop_reason", lambda: None)
    monkeypatch.setattr(joint_control, "remaining", lambda **kwargs: 10.0)
    (source / "compiler.py").write_text("print(9)\n")
    assert guard.pending_reason() is None
    monkeypatch.setattr(joint_control, "remaining", lambda **kwargs: 0.0)
    assert guard.pending_reason() == "joint campaign deadline reached"
    monkeypatch.setattr(joint_control, "stop_reason", lambda: "final handoff")
    assert guard.pending_reason() == "final handoff"
