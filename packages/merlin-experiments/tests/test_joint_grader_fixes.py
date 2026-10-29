"""Joint grader fixes: stale-snapshot preemption, freeze-boundary revocation, guaranteed final
coverage budget, drain checkpoint, and the Spike-only scalability tier-floor exemption.

Every behavior here is joint-only; the default (non-joint) paths are pinned by the existing
dual-snapshot, lifecycle and final-coverage tests.
"""

import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
from merlin_experiments.phase1 import joint_control as JC
from merlin_experiments.phase1 import joint_final_coverage as F
from merlin_experiments.phase1.feedback import dual_snapshots, snapshot_publication
from merlin_experiments.phase1.feedback.background_admission import BackgroundGuard

from merlin.targetgen import tier_policy
from merlin.targetgen.capsule_scheduling import BackgroundAbort, check_abort, rolling_admission, run_pipelined


@pytest.fixture(autouse=True)
def _fast_scheduler_poll(monkeypatch):
    from merlin.targetgen import capsule_scheduling

    monkeypatch.setattr(capsule_scheduling, "ADMISSION_POLL_S", 0.02)


_real_remaining = JC.remaining


def _state(**policy):
    base = {"duration_s": 7200, "drain_s": 1200, "final_model_s": 3000}
    return {"started_at_epoch": time.time(), "policy_sha256": "policy", "policy": {**base, **policy}}


def _workspace(tmp_path):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    return ws, run


def _owners(run):
    rows = {}
    for path in (run / "_qa_work/background_snapshot_owners").glob("*.json"):
        if path.name.endswith(".next.json"):
            continue
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rows[row["tick"]] = row
    return rows


class _Harness:
    """Fake snapshot grades through the real scheduler: blocked members poll the cooperative checkpoint.

    ``window`` is the scheduler's admission window (production: 300 s), the event that closes a snapshot's
    admission and makes it preemptable; ``prep_s`` is queue preparation before the scheduler runs.
    """

    def __init__(self, cooperative=True, names=6, workers=2, window=0.1, prep_s=0.0, member_s=None):
        self.cooperative, self.names, self.workers = cooperative, names, workers
        self.window, self.prep_s, self.member_s = window, prep_s, member_s
        self.started, self.release = [], threading.Event()
        self.lock = threading.Lock()
        self.live, self.peak = 0, 0
        self.verdicts = {}

    def grade(self, tick):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        caps = [{"name": f"t{tick % 1000}_c{i}"} for i in range(self.names)]

        def execute(cap, parallel):
            with self.lock:
                self.started.append(cap["name"])
            began = time.monotonic()
            while not self.release.is_set():
                if self.member_s is not None and time.monotonic() - began >= self.member_s:
                    break
                if self.cooperative:
                    check_abort()
                time.sleep(0.005)
            return {"capsule": cap["name"], "status": "pass"}

        try:
            time.sleep(self.prep_s)
            scope = rolling_admission(admission_window_s=self.window) if self.window else nullcontext()
            with scope:
                rows = run_pipelined(caps, execute, max_workers=self.workers)
        finally:
            with self.lock:
                self.live -= 1
        self.verdicts[tick] = rows
        return {"all_pass": False, "rows": rows}


def _run(tmp_path, monkeypatch, harness, state, sources, *, stop, poll=0.005, interval=0.3):
    ws, run = _workspace(tmp_path)
    monkeypatch.setattr(JC, "selected", lambda: state)
    manager = threading.Thread(
        target=lambda: dual_snapshots.run(
            ws,
            run,
            stop,
            interval_s=interval,
            first_grade=harness.grade,
            grade=harness.grade,
            source_identity=lambda: sources[0],
            poll_s=poll,
        ),
        daemon=True,
    )
    manager.start()
    return manager, run


def _wait(predicate, seconds=5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_abort_preemption_cancels_queue_revokes_in_flight_and_starts_newest(tmp_path, monkeypatch):
    harness, stop, sources = _Harness(workers=2, names=6), threading.Event(), ["a" * 64]
    state = _state(snapshot_preemption={"mode": "abort", "min_age_s": 0, "abort_grace_s": 0})
    manager, run = _run(tmp_path, monkeypatch, harness, state, sources, stop=stop)
    assert _wait(lambda: len(harness.started) >= 2)
    sources[0] = "b" * 64  # the author moved on; the next snapshot is due
    assert _wait(lambda: len(_owners(run)) >= 2 and len(harness.started) >= 4)
    first, second = sorted(_owners(run))[:2]
    assert _wait(lambda: harness.verdicts.get(first) is not None)
    rows = harness.verdicts[first]
    # Only the two in-flight members ever started; they were revoked, the other four never admitted.
    assert [r.get("execution_status") for r in rows].count("aborted") == 2
    assert sum(r["status"] == "pass" for r in rows) == 0
    assert all(r["status"] == "gated" for r in rows)
    assert len([n for n in harness.started if n.startswith(f"t{first % 1000}_")]) == 2
    owner = _owners(run)[first]
    assert owner["preemption"]["by_source"] == "b" * 64 and owner["admission_closed"] is True
    assert owner["preemption"]["abort_reason"].startswith("snapshot preempted")
    assert harness.peak <= 2
    stop.set()
    harness.release.set()
    manager.join(5)
    assert not manager.is_alive()


def test_drain_mode_stops_admission_but_lets_in_flight_finish(tmp_path, monkeypatch):
    harness, stop, sources = _Harness(workers=2, names=6), threading.Event(), ["a" * 64]
    state = _state(snapshot_preemption={"mode": "drain", "min_age_s": 0})
    manager, run = _run(tmp_path, monkeypatch, harness, state, sources, stop=stop)
    assert _wait(lambda: len(harness.started) >= 2)
    sources[0] = "b" * 64
    assert _wait(lambda: len(_owners(run)) >= 2)  # newest started while the old one still drains
    first = sorted(_owners(run))[0]
    assert first not in harness.verdicts, "drain mode must not revoke started members"
    harness.release.set()
    assert _wait(lambda: first in harness.verdicts)
    rows = harness.verdicts[first]
    assert sum(r["status"] == "pass" for r in rows) == 2  # in-flight finished
    assert sum(r["status"] == "gated" for r in rows) == 4  # queue was cancelled
    stop.set()
    manager.join(5)


def test_never_more_than_two_live_snapshots_and_slot_pressure_revokes_oldest(tmp_path, monkeypatch):
    harness, stop, sources = _Harness(workers=2, names=4), threading.Event(), ["a" * 64]
    # Large abort grace: only slot pressure (a third snapshot is due while two exist) may revoke early.
    state = _state(snapshot_preemption={"mode": "abort", "min_age_s": 0, "abort_grace_s": 900})
    manager, run = _run(tmp_path, monkeypatch, harness, state, sources, stop=stop)
    assert _wait(lambda: len(harness.started) >= 1)
    sources[0] = "b" * 64
    assert _wait(lambda: len(_owners(run)) >= 2)
    sources[0] = "c" * 64
    assert _wait(lambda: len(_owners(run)) >= 3)
    assert harness.peak <= 3  # a revoked owner unwinds beside the two live ones, never two stuck
    live = [
        o
        for o in _owners(run).values()
        if o["state"] in {"admitting", "draining"} and not (o["preemption"] and o["preemption"]["abort_reason"])
    ]
    assert len(live) <= 2
    stop.set()
    harness.release.set()
    manager.join(5)
    assert not manager.is_alive()


def test_freeze_boundary_revokes_and_bounds_the_join(tmp_path, monkeypatch):
    # A non-cooperative member (blocked in a call that never reaches a checkpoint) must not hold the
    # freeze/drain window: the join gives up after freeze_grace_s and records the owner as abandoned.
    harness, stop, sources = _Harness(cooperative=False, workers=2, names=2), threading.Event(), ["a" * 64]
    state = _state(freeze_grace_s=1, snapshot_preemption={"mode": "off"})
    state["started_at_epoch"] = time.time() - 7200 + 600  # inside the drain: the freeze boundary passed
    manager, run = _run(tmp_path, monkeypatch, harness, state, sources, stop=stop)
    assert _wait(lambda: len(harness.started) >= 2)
    started = time.monotonic()
    stop.set()
    manager.join(10)
    assert not manager.is_alive() and time.monotonic() - started < 5
    owner = next(iter(_owners(run).values()))
    assert owner["state"] == "abandoned"
    assert owner["preemption"]["abort_reason"].startswith("joint campaign freeze boundary")
    harness.release.set()


def test_freeze_boundary_with_cooperative_members_unwinds_without_waiting_for_grades(tmp_path, monkeypatch):
    harness, stop, sources = _Harness(cooperative=True, workers=2, names=6), threading.Event(), ["a" * 64]
    state = _state(freeze_grace_s=5)
    state["started_at_epoch"] = time.time() - 7200 + 600
    manager, run = _run(tmp_path, monkeypatch, harness, state, sources, stop=stop)
    assert _wait(lambda: len(harness.started) >= 2)
    stop.set()
    manager.join(10)
    assert not manager.is_alive()
    ((tick, rows),) = harness.verdicts.items()
    assert all(r["status"] == "gated" for r in rows) and len(harness.started) == 2
    assert next(iter(_owners(run).values()))["state"] == "complete"
    harness.release.set()


def test_non_joint_stop_still_joins_without_cancelling(tmp_path, monkeypatch):
    harness, stop, sources = _Harness(workers=2, names=6, window=None), threading.Event(), ["a" * 64]
    ws, run = _workspace(tmp_path)
    monkeypatch.setattr(JC, "selected", lambda: None)
    manager = threading.Thread(
        target=lambda: dual_snapshots.run(
            ws,
            run,
            stop,
            interval_s=0,
            first_grade=harness.grade,
            grade=harness.grade,
            source_identity=lambda: sources[0],
            poll_s=0.005,
        )
    )
    manager.start()
    assert _wait(lambda: len(harness.started) >= 2)
    sources[0] = "b" * 64
    time.sleep(0.1)
    assert len(_owners(run)) == 1, "no preemption outside a joint campaign"
    stop.set()
    time.sleep(0.05)
    assert manager.is_alive()
    harness.release.set()
    manager.join(5)
    # Exactly the pre-existing stop semantics: started members finish, the unstarted queue is cancelled.
    assert [r["status"] for r in next(iter(harness.verdicts.values()))] == ["pass"] * 2 + ["gated"] * 4


def test_policy_preemption_validation_and_resume_safety(tmp_path):
    base = {"version": 1, "duration_s": 7200, "drain_s": 1200, "final_model_s": 3000}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(base))
    stored = JC.policy(path)
    assert "snapshot_preemption" not in stored and "final_coverage_s" not in stored  # nothing injected
    for bad in (
        {"snapshot_preemption": {"mode": "kill"}},
        {"snapshot_preemption": {"min_age_s": -1}},
        {"final_coverage_s": 0},
        {"final_coverage_s": 1201},
        {"freeze_grace_s": 301},
        {"drain_checkpoint": "yes"},
        {"snapshot_preemption": {"bogus": 1}},
    ):
        path.write_text(json.dumps({**base, **bad}))
        with pytest.raises(ValueError):
            JC.policy(path)
    path.write_text(json.dumps({**base, "final_coverage_s": 900, "snapshot_preemption": {"mode": "drain"}}))
    assert JC.policy(path)["final_coverage_s"] == 900


def test_guard_reports_superseded_snapshot_with_partial_counts():
    guard = BackgroundGuard("ws", "public", None, "a" * 64, finish_bounded_snapshot=True)
    with snapshot_publication.scope("ws", "tok", preempted=lambda: "snapshot preempted: newer snapshot"):
        assert guard.pending_reason() == "snapshot preempted: newer snapshot"
        record = guard.superseded_record(
            {
                "per_capsule": [
                    {"status": "pass"},
                    {"status": "fail"},
                    {"status": "gated", "execution_status": "aborted"},
                    {"status": "gated"},
                ]
            }
        )
    assert record["status"] == "superseded_snapshot" and record["preempted"] is True
    assert record["partial_counts"] == {
        "n_rows": 4,
        "n_pass": 1,
        "n_fail": 1,
        "n_not_run": 2,
        "n_other": 0,
        "n_aborted_in_flight": 1,
    }
    assert guard.reason() == "snapshot preempted: newer snapshot"


def test_abort_checkpoint_is_free_outside_a_revocable_snapshot_and_is_not_an_ordinary_error():
    check_abort()  # no owner: no-op
    assert not issubclass(BackgroundAbort, Exception)
    rows = run_pipelined([{"name": "x"}], lambda cap, n: {"capsule": "x", "status": "pass"}, max_workers=2)
    assert rows[0]["status"] == "pass"


# ---------------------------------------------------------------- final coverage budget


def test_final_coverage_window_guarantees_budget_after_a_drain_overrun(monkeypatch):
    state = _state(final_coverage_s=600)
    state["started_at_epoch"] = time.time() - 7200 - 30  # the drain overran the hard deadline by 30 s
    monkeypatch.setattr(JC, "selected", lambda: state)
    assert JC.remaining() == 0.0 and JC.remaining(coverage=True) == 0.0
    with JC.final_coverage_window() as window:
        assert window["extended_past_original_deadline"] is True and window["basis"] == "reserved_floor"
        assert 595 <= JC.remaining(coverage=True) <= 600
        assert JC.remaining() == 0.0  # the author-side/gSIM clock never moves
        assert 595 <= JC.require_full_model(engine="spike") <= 600  # the model branch uses the window too
    assert JC.remaining(coverage=True) == 0.0  # the window does not outlive final coverage


def test_final_coverage_window_does_not_shorten_a_healthy_budget(monkeypatch):
    state = _state(final_coverage_s=300)
    state["started_at_epoch"] = time.time() - 7200 + 900  # 900 s to the deadline
    monkeypatch.setattr(JC, "selected", lambda: state)
    with JC.final_coverage_window() as window:
        assert window["basis"] == "original_deadline" and window["extended_past_original_deadline"] is False
        assert 890 <= window["window_s"] <= 900


def test_drain_overrun_still_grades_core_and_scalability(tmp_path, monkeypatch):
    from test_joint_final_coverage import _native_fixture

    real_remaining = JC.remaining
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    state = F.joint_control.selected()
    state["policy"].update(duration_s=7200, drain_s=200, final_model_s=3000, final_coverage_s=100)
    state["started_at_epoch"] = time.time() - 7200 - 45  # 45 s past the hard deadline (reserve 100 s)
    monkeypatch.setattr(F.joint_control, "remaining", real_remaining)
    monkeypatch.setattr(F.joint_control, "require_full_model", lambda **kw: 100)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert report["budget"]["extended_past_original_deadline"] is True
    assert report["budget"]["window_s"] == 100
    errors = {k: g.get("error") for k, g in report["groups"].items() if g.get("error")}
    assert not errors, errors
    assert report["n_passed"] == 246 and report["passed"] is True, "an overrun must not zero the coverage"
    assert [c[0] for c in calls].count("capsules") == 2


def test_scalability_group_keeps_its_l0_l2_ladder_under_an_operator_floor(tmp_path, monkeypatch):
    from test_joint_final_coverage import _native_fixture

    from merlin.targetgen import capsule_grade as CG

    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(tier_policy.TIER_FLOOR_ENV, "L3")
    seen = {}

    def grade(candidate, **kw):
        root = Path(kw["capsules_root"])
        name = "scale0" if root.name == "cases" else "core0"
        capsule = {"required_oracle_tiers": ["L0", "L1", "L2"], "__dir__": str(root / name)}
        seen[root.name] = tier_policy.required_tiers(capsule)
        from test_joint_final_coverage import _score

        names = [f"scale{i}" for i in range(67)] if root.name == "cases" else [f"core{i}" for i in range(173)]
        return _score(names)

    monkeypatch.setattr(CG, "grade", grade)
    F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert seen["cases"] == ["L0", "L1", "L2"], "scalability must not be raised to L3 by the floor"
    assert seen["core"] == ["L0", "L1", "L2", "L3"], "core capsules keep the operator floor"


def test_operator_child_document_carries_the_floor_exemption(tmp_path):
    from merlin_experiments.phase1.feedback import operator_child as OC

    directory = tmp_path / "case"
    directory.mkdir()
    capsule = {"name": "case", "required_oracle_tiers": ["L0", "L1", "L2"], "__dir__": str(directory)}
    assert tier_policy.FLOOR_EXEMPT_FIELD not in OC._capsule_document(capsule)
    with tier_policy.floor_exempt([directory]):
        document = OC._capsule_document(capsule)
    assert document[tier_policy.FLOOR_EXEMPT_FIELD] is True
    assert tier_policy.FLOOR_EXEMPT_FIELD not in OC._capsule_document(capsule)  # scoped to the block


# ---------------------------------------------------------------- drain checkpoint


def test_drain_checkpoint_only_at_the_joint_freeze_boundary(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_handoff as H

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(JC, "selected", lambda: None)
    assert H.drain_required(run) is False  # non-joint: ordinary post-turn grading
    state = _state()
    monkeypatch.setattr(JC, "selected", lambda: state)
    assert H.drain_required(run) is False  # before the freeze boundary
    state["started_at_epoch"] = time.time() - 7200 + 600
    assert H.drain_required(run) is True
    state["policy"]["drain_checkpoint"] = False  # the operator can keep the old behavior
    assert H.drain_required(run) is False
    state["policy"]["drain_checkpoint"] = True
    (run / "joint_epoch_stop.json").write_text(
        json.dumps({"schema": "merlin.joint_epoch_stop.v1", "reason": "library review"})
    )  # keeps its own handoff
    assert H.drain_required(run) is False
    with pytest.raises(ValueError, match="freeze boundary"):
        H.checkpoint(tmp_path / "ws", run, 0, drain=True)


def test_continuous_edits_and_slow_preparation_do_not_starve_a_snapshot(tmp_path, monkeypatch):
    # The author edits constantly and preparation outlasts min_age: a snapshot must still finish rows,
    # because it can only be preempted after the scheduler itself closed its admission.
    harness = _Harness(workers=2, names=4, prep_s=0.5, member_s=0.15, window=0.3)
    stop, counter = threading.Event(), [0]

    class Moving:
        def __getitem__(self, _):
            counter[0] += 1
            return f"{counter[0]:064d}"

    state = _state(snapshot_preemption={"mode": "abort", "min_age_s": 0, "abort_grace_s": 0})
    manager, run = _run(tmp_path, monkeypatch, harness, state, Moving(), stop=stop, interval=0.1)
    assert _wait(lambda: any(r["status"] == "pass" for rows in harness.verdicts.values() for r in rows), 8)
    for owner in _owners(run).values():
        if owner["preemption"] and owner["preemption"]["at_epoch"]:
            assert owner["scheduler_closed_at_epoch"] is not None, "preempted before the scheduler closed it"
    stop.set()
    harness.release.set()
    manager.join(5)


def test_group_share_cannot_turn_a_healthy_run_incomplete(tmp_path, monkeypatch):
    from test_joint_final_coverage import _native_fixture, _score

    from merlin.targetgen import capsule_grade as CG

    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    state = F.joint_control.selected()
    state["policy"].update(duration_s=7200, drain_s=200, final_model_s=3000, final_coverage_s=100)
    state["started_at_epoch"] = time.time() - 7200 - 45
    monkeypatch.setattr(F.joint_control, "remaining", _real_remaining)
    monkeypatch.setattr(F.joint_control, "require_full_model", lambda **kw: 100)
    seen = []

    def grade(candidate, **kw):
        tag = Path(kw["runs_root"]).name
        seen.append(tag)
        names = [f"scale{i}" for i in range(67)] if tag.startswith("scalability") else [f"core{i}" for i in range(173)]
        score = _score(names)
        if tag == "core":  # the 45% share cut the first core pass short
            score["per_capsule"][-1]["status"] = "gated"
            score["gradeable"] = False
        return score

    monkeypatch.setattr(CG, "grade", grade)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert "core-reclaim" in seen and report["groups"]["core"]["reclaim"]["closed_after"] == 173
    assert report["n_passed"] == 246 and report["passed"] is True
