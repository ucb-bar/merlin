"""A stale or non-loading background snapshot stops spending grade capacity, and says so honestly."""

import json
import threading
from types import SimpleNamespace

import pytest

from merlin.common import cpu_pool
from merlin.targetgen import capsule_scheduling as CS
from merlin_experiments.phase1.feedback import background_admission as admission
from merlin_experiments.phase1.feedback import snapshot_publication as SP
from merlin_experiments.phase1.feedback import snapshot_smoke as SM


def _cap(name):
    return {"name": name, "inputs": [{"name": "x", "shape": [4, 4]}]}


# ---- claim file: newer snapshot of different source supersedes; same source complements ------------

def test_only_a_newer_snapshot_of_different_source_supersedes(tmp_path):
    SP.claim(tmp_path, "100", "src-a")
    assert SP.superseding_snapshot(tmp_path, "100") is None
    SP.claim(tmp_path, "200", "src-a")  # same-source overlap grades the complement: no supersession
    assert SP.superseding_snapshot(tmp_path, "100") is None
    SP.claim(tmp_path, "300", "src-b")
    assert SP.superseding_snapshot(tmp_path, "100") == "300"
    assert SP.superseding_snapshot(tmp_path, "200") == "300"
    assert SP.superseding_snapshot(tmp_path, "300") is None  # the newest is never stale


def test_unknown_source_never_supersedes_and_a_snapshot_that_never_graded_is_released(tmp_path):
    SP.claim(tmp_path, "100", "src-a")
    SP.claim(tmp_path, "200", None)
    assert SP.superseding_snapshot(tmp_path, "100") is None
    SP.claim(tmp_path, "300", "src-b")
    SP.release(tmp_path, "300", fallback="100")
    assert SP.superseding_snapshot(tmp_path, "100") is None
    SP.release(tmp_path, "999", fallback="1")  # not the claimant: no change
    assert json.loads((tmp_path / "qa/.snapshot-publication.json").read_text())["token"] == "100"


# ---- scheduler: pending skipped, queued members never start, rows are 'gated' not pass/fail --------

def test_superseded_snapshot_skips_pending_and_records_unknown_not_pass_or_fail():
    newer = threading.Event()
    reason = "superseded by newer snapshot 300 (different source)"
    ran = []

    def run(cap, workers):
        ran.append(cap["name"])
        return {"capsule": cap["name"], "status": "pass"}

    def completed(row):
        newer.set()

    with CS.observe_supersession(lambda: reason if newer.is_set() else None):
        rows = CS.run_pipelined([_cap(n) for n in "abcd"], run, max_workers=1, on_complete=completed,
                                defer_pending=lambda: reason if newer.is_set() else None)
    assert ran == ["a"] or sorted(ran) == ["a"]
    assert [r["status"] for r in rows] == ["pass", "gated", "gated", "gated"]
    for row in rows[1:]:
        assert row["background_deferred"] == reason and row["correctness"] == "unknown"
        assert row["failure"]["category"] == "BACKGROUND_DEFERRED"


def test_member_queued_behind_the_pool_is_skipped_at_its_start_not_run():
    reason = "superseded by newer snapshot 9 (different source)"
    gate, started = threading.Event(), []

    def run(cap, workers):
        started.append(cap["name"])
        gate.wait(2)
        return {"capsule": cap["name"], "status": "pass"}

    # run_one is entered for 'b' only after the snapshot went stale: the safe point before work skips it.
    with CS.observe_supersession(lambda: reason if gate.is_set() else None):
        gate.set()
        wrapped = {"n": 0}

        def pending():
            wrapped["n"] += 1
            return None

        rows = CS.run_pipelined([_cap("a"), _cap("b")], run, max_workers=2, defer_pending=pending)
    assert started == []
    assert all(r["status"] == "gated" and r["background_deferred"] == reason for r in rows)


def test_running_member_sees_the_stale_reason_through_the_defer_probe():
    reason = "superseded by newer snapshot 9 (different source)"
    stale, seen = threading.Event(), []

    def run(cap, workers):
        stale.set()
        probe = CS.active_defer_probe()
        seen.append(probe() if probe else None)
        return {"capsule": cap["name"], "status": "gated"}

    with CS.observe_supersession(lambda: reason if stale.is_set() else None):
        CS.run_pipelined([_cap("a")], run, max_workers=1, defer_pending=lambda: None)
    assert seen == [reason]


def test_no_supersession_scope_changes_nothing():
    assert CS.supersession_reason() is None
    rows = CS.run_pipelined([_cap("a")], lambda c, w: {"capsule": "a", "status": "pass"}, max_workers=1,
                            defer_pending=lambda: None)
    assert rows[0]["status"] == "pass"


# ---- compiler command boundary ---------------------------------------------------------------------

def test_next_compiler_command_of_a_stale_snapshot_is_gated_before_custody():
    from merlin.targetgen import package_runtime as P
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R

    reason = "superseded by newer snapshot 5 (different source)"
    with CS.observe_admission(lambda why: None), CS.observe_supersession(lambda: reason):
        with pytest.raises(P.NotMeasuredFailure) as caught:
            R._checkpoint_command_admission()
    assert reason in str(caught.value)
    with CS.observe_admission(lambda why: None):
        R._checkpoint_command_admission()  # a current snapshot proceeds


# ---- newest snapshot has priority for grade slots -------------------------------------------------

def test_stale_snapshot_leases_queue_behind_the_newest_one():
    stale = {"v": False}
    with cpu_pool.cpu_priority(cpu_pool.GRADE), cpu_pool.demote_when_stale(lambda: stale["v"]):
        assert cpu_pool.lease_priority() == cpu_pool.GRADE
        stale["v"] = True
        assert cpu_pool.lease_priority() == cpu_pool.PREPARATION
    with cpu_pool.cpu_priority(cpu_pool.AUTHOR), cpu_pool.demote_when_stale(lambda: True):
        assert cpu_pool.lease_priority() == cpu_pool.AUTHOR  # author work is never demoted
    assert cpu_pool.lease_priority(environ={}) == cpu_pool.GRADE


# ---- guard: labels and accounting stay honest -------------------------------------------------------

def _guard(tmp_path, monkeypatch, token, source, policy="interrupt"):
    monkeypatch.setenv("MERLIN_SNAPSHOT_STALE_POLICY", policy)
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True, exist_ok=True)
    (ws / "submission/compiler.py").write_text("x = 1\n")
    SP.claim(ws, token, source)
    with SP.scope(ws, token):
        return ws, admission.BackgroundGuard(ws, tmp_path, SimpleNamespace(compiler_profile_service=None),
                                             admission.source_digest(ws), finish_bounded_snapshot=True)


def test_guard_labels_a_superseded_snapshot_and_a_finished_one_stays_complete(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    monkeypatch.setattr(joint_control, "stop_reason", lambda: None)
    monkeypatch.setattr(joint_control, "remaining", lambda *a, **k: 600)
    ws, guard = _guard(tmp_path, monkeypatch, "100", "a")
    assert guard.pending_reason() is None and not guard.ended_early()
    SP.claim(ws, "200", "b")
    assert guard.ended_early() is False  # nothing was skipped yet: a complete roster is not relabelled
    reason = guard.pending_reason()
    assert "superseded by newer snapshot 200" in reason
    assert guard.ended_early() and guard.terminal_status() == "superseded_snapshot"
    assert guard.ended_reason() == reason and guard.interrupt_reason() == reason


def test_drain_policy_lets_started_work_finish_but_still_skips_pending(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    monkeypatch.setattr(joint_control, "stop_reason", lambda: None)
    monkeypatch.setattr(joint_control, "remaining", lambda *a, **k: 600)
    ws, guard = _guard(tmp_path, monkeypatch, "100", "a", policy="drain")
    SP.claim(ws, "200", "b")
    assert guard.pending_reason() is not None
    assert guard.interrupt_reason() is None


# ---- smoke ------------------------------------------------------------------------------------------

def test_static_smoke_catches_the_run1_skeleton_and_passes_a_loadable_tree(tmp_path):
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    (skeleton / "compiler.py").write_text("from frontend import parse\nimport graph_lowering\n")
    reason = SM.static_smoke(skeleton)
    assert reason and "frontend" in reason and "graph_lowering" in reason

    good = tmp_path / "good"
    good.mkdir()
    (good / "frontend.py").write_text("import os, json\ntry:\n    import optional_dep\nexcept ImportError:\n    pass\n")
    (good / "compiler.py").write_text("from frontend import parse\nimport json\n")
    assert SM.static_smoke(good) is None

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "compiler.py").write_text("def f(:\n")
    assert "does not parse" in SM.static_smoke(broken)


def test_load_failure_screen_trips_only_when_all_first_rows_fail_to_load(monkeypatch):
    load = {"status": "fail", "failure_detail": "parse rc=1 ModuleNotFoundError: No module named 'frontend'"}
    screen = SM.LoadFailureScreen(rows=3)
    for i in range(2):
        screen.observe({"capsule": str(i), **load})
    assert screen.reason is None
    screen.observe({"capsule": "2", **load})
    assert screen.reason and "skipped" in screen.reason

    mixed = SM.LoadFailureScreen(rows=3)
    for row in (load, {"status": "pass"}, load):
        mixed.observe(row)
    assert mixed.reason is None  # one pass: the compiler loads

    other = SM.LoadFailureScreen(rows=2)
    for _ in range(2):
        other.observe({"status": "fail", "failure_detail": "numeric mismatch"})
    assert other.reason is None  # an ordinary failure is a result, not a skip

    skipped = SM.LoadFailureScreen(rows=1)
    skipped.observe({"status": "gated", "background_deferred": "x"})
    assert skipped.reason is None


def test_guard_prescreen_gates_the_roster_and_is_labelled_not_failed(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    monkeypatch.setattr(joint_control, "stop_reason", lambda: None)
    monkeypatch.setattr(joint_control, "remaining", lambda *a, **k: 600)
    ws, guard = _guard(tmp_path, monkeypatch, "100", "a")
    tree = tmp_path / "cand"
    tree.mkdir()
    (tree / "compiler.py").write_text("import frontend\n")
    assert "frontend" in guard.smoke(tree)
    assert guard.pending_reason().startswith("smoke:")
    assert guard.ended_early() and guard.terminal_status() == "prescreen_failed"
    monkeypatch.setenv("MERLIN_SNAPSHOT_SMOKE", "0")
    _, off = _guard(tmp_path, monkeypatch, "101", "a")
    assert off.smoke(tree) is None
