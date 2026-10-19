"""Capsule grading is pipelined: no wave barrier, dependents wait only for their own sibling."""

import os
import subprocess
import sys
import threading
import time

import pytest

from merlin.targetgen.capsule_scheduling import operand_volume, run_pipelined


def _cap(name, extends=None, shape=(16, 16)):
    cap = {"name": name, "inputs": [{"name": "x", "shape": list(shape)}]}
    if extends:
        cap["extends"] = extends
    return cap


@pytest.mark.parametrize("workers", [1, 2])
def test_selected_grade_pool_is_inherited_by_children_and_restores_manager(monkeypatch, workers):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity required")
    original = os.sched_getaffinity(0)
    selected = min(original)
    monkeypatch.setenv("MERLIN_GRADE_CPUS", str(selected))

    def run(cap, parallel):
        assert os.sched_getaffinity(0) == {selected}
        child = subprocess.check_output(
            [sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"], text=True
        )
        assert child.strip() == str([selected])
        return {"capsule": cap["name"], "status": "pass"}

    assert len(run_pipelined([_cap("a"), _cap("b")], run, max_workers=workers)) == 2
    assert os.sched_getaffinity(0) == original

    def fail(cap, parallel):
        raise RuntimeError("grade failed")

    with pytest.raises(RuntimeError, match="grade failed"):
        run_pipelined([_cap("a")], fail, max_workers=workers)
    assert os.sched_getaffinity(0) == original


def test_a_slow_member_does_not_hold_independent_ones_and_dependents_wait_for_their_sibling():
    started, ended, lock = {}, {}, threading.Lock()
    slow_done = threading.Event()

    def run(cap, workers):
        with lock:
            started[cap["name"]] = time.monotonic()
        if cap["name"] == "slow":
            time.sleep(0.6)
            slow_done.set()
        else:
            time.sleep(0.05)
        if cap["name"] == "rests_on_slow":
            assert slow_done.is_set(), "a dependent started before the sibling it cites finished"
        with lock:
            ended[cap["name"]] = time.monotonic()
        return {"capsule": cap["name"], "status": "pass"}

    caps = [_cap("slow"), _cap("rests_on_slow", extends="slow"), *[_cap(f"small{i}") for i in range(6)]]
    out = run_pipelined(caps, run, max_workers=3)
    assert [row["capsule"] for row in out] == [c["name"] for c in caps], "results come back in input order"
    # every small member finished while the slow one was still running: no wave barrier
    assert all(ended[f"small{i}"] < ended["slow"] for i in range(6))
    assert started["rests_on_slow"] >= ended["slow"]


def test_calibration_runs_once_on_the_cheapest_member_and_stops_when_nothing_new_is_priced():
    order, priced = [], set()

    def run(cap, workers):
        order.append((cap["name"], workers))
        priced.add("functional")  # only the first run prices a new tier
        return {"capsule": cap["name"], "status": "fail"}

    caps = [_cap("huge", shape=(3136, 576)), _cap("tiny", shape=(4, 4)), _cap("mid", shape=(64, 64))]
    run_pipelined(caps, run, max_workers=2, calibrate=lambda: set(priced), calibration_cap=3)
    assert order[0] == ("tiny", 1), "the serial head is the cheapest member, not the first listed"
    assert order[1][1] == 1 and all(n == 2 for _, n in order[2:])
    assert operand_volume(caps[0]) == 3136 * 576


def test_cycles_and_duplicates_are_refused():
    with pytest.raises(ValueError, match="cyclic"):
        run_pipelined([_cap("a", "b"), _cap("b", "a")], lambda c, n: {}, max_workers=2)
    with pytest.raises(ValueError, match="unique"):
        run_pipelined([_cap("a"), _cap("a")], lambda c, n: {}, max_workers=2)


def test_a_slow_calibration_head_does_not_hold_the_pool():
    """A full-shape layer at the head used to run alone for hours: past its budget the pool fans out."""
    started, lock = {}, threading.Lock()
    head_done = threading.Event()

    def run(cap, workers):
        with lock:
            started[cap["name"]] = time.monotonic()
        if cap["name"] == "head":
            time.sleep(0.6)
            head_done.set()
        return {"capsule": cap["name"], "status": "fail"}

    caps = [_cap("head", shape=(1, 1)), *[_cap(f"layer{i}", shape=(64, 64)) for i in range(4)]]
    t0 = time.monotonic()
    run_pipelined(caps, run, max_workers=3, calibrate=lambda: {"new"}, calibration_cap=3, calibration_budget_s=0.1)
    assert all(started[f"layer{i}"] - t0 < 0.5 for i in range(4)), "the rest started while the head still ran"
    assert head_done.is_set()


def test_ready_members_start_longest_first():
    order = []

    def run(cap, workers):
        order.append(cap["name"])
        return {"capsule": cap["name"], "status": "pass"}

    caps = [_cap("small", shape=(4, 4)), _cap("large", shape=(512, 512)), _cap("mid", shape=(64, 64))]
    run_pipelined(caps, run, max_workers=1)
    assert order == ["large", "mid", "small"]


def test_shared_scheduler_does_not_hold_slots_while_children_acquire(monkeypatch, tmp_path):
    from merlin.common.cpu_pool import worker_pool_command

    cpu = min(os.sched_getaffinity(0))
    original = os.sched_getaffinity(0)
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "shared"))
    monkeypatch.setenv("MERLIN_GRADE_CPUS", "invalid-legacy-pool")
    parallelism = []
    script = (
        "import os; from merlin.common.cpu_pool import shared_compute_pool; "
        "scope=shared_compute_pool(timeout_s=.5); scope.__enter__(); "
        "print(sorted(os.sched_getaffinity(0))); scope.__exit__(None,None,None)"
    )

    def run(cap, parallel):
        parallelism.append(parallel)
        assert os.sched_getaffinity(0) == {cpu}
        command = worker_pool_command([sys.executable, "-c", script], dict(os.environ))
        child = subprocess.check_output(command, text=True, timeout=3)
        assert child.strip() == str([cpu])
        return {"capsule": cap["name"], "status": "pass"}

    rows = run_pipelined([_cap("first"), _cap("second")], run, max_workers=6)
    assert [row["capsule"] for row in rows] == ["first", "second"]
    assert parallelism == [1, 1]
    assert os.sched_getaffinity(0) == original


@pytest.mark.parametrize(
    "ordering,expected",
    [
        ("legacy", ["model", "large", "small", "dependent"]),
        ("operators-small-first", ["small", "large", "dependent", "model"]),
    ],
)
def test_optional_shortest_operator_order_preserves_dependencies_and_legacy_priority(monkeypatch, ordering, expected):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", ordering)
    caps = [
        _cap("large", shape=(64, 64)),
        dict(_cap("model", shape=(1,)), kind="model"),
        _cap("small", shape=(2,)),
        _cap("dependent", extends="large", shape=(1,)),
    ]
    order = []

    def run(cap, parallel):
        order.append(cap["name"])
        if cap["name"] == "dependent":
            assert "large" in order[:-1]
        return {"capsule": cap["name"], "status": "pass"}

    rows = run_pipelined(caps, run, max_workers=1, priority=lambda cap: 0 if cap.get("kind") == "model" else 1)
    assert order == expected
    assert [row["capsule"] for row in rows] == [cap["name"] for cap in caps]


def test_small_first_keeps_immediate_completion_and_explicit_model_prerequisites(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "operators-small-first")
    fast_done, model_started = threading.Event(), threading.Event()
    callbacks = []

    def run(cap, parallel):
        if cap["name"] == "fast":
            time.sleep(0.02)
        elif cap["name"] == "large":
            assert model_started.wait(timeout=2), "an admitted model waited for an unrelated active operator"
        else:
            assert fast_done.is_set(), "model bypassed its explicit prerequisite"
            model_started.set()
        return {"capsule": cap["name"], "status": "pass"}

    def complete(row):
        callbacks.append(row["capsule"])
        if row["capsule"] == "fast":
            fast_done.set()

    caps = [_cap("large", shape=(64, 64)), dict(_cap("model", shape=(1,)), kind="model"), _cap("fast", shape=(2,))]
    rows = run_pipelined(
        caps,
        run,
        max_workers=2,
        on_complete=complete,
        can_start=lambda cap: cap.get("kind") != "model" or fast_done.is_set(),
    )
    assert callbacks[0] == "fast"
    assert model_started.is_set()
    assert [row["capsule"] for row in rows] == [cap["name"] for cap in caps]


def test_invalid_capsule_order_refuses_before_execution(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "unknown")
    with pytest.raises(ValueError, match="MERLIN_CAPSULE_ORDER"):
        run_pipelined([_cap("unused")], lambda *a: pytest.fail("invalid selection started work"), max_workers=2)


def test_zero_operator_calibration_wait_starts_independent_workers_immediately(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_CALIBRATION_S", "0")
    barrier = threading.Barrier(2)

    def run(cap, workers):
        barrier.wait(timeout=2)
        return {"capsule": cap["name"], "status": "pass"}

    result = run_pipelined(
        [_cap("first"), _cap("second")], run, max_workers=2, calibrate=lambda: set(), calibration_cap=3
    )
    assert len(result) == 2 and all(row["status"] == "pass" for row in result)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_operator_calibration_wait_refuses(monkeypatch, value):
    monkeypatch.setenv("MERLIN_CAPSULE_CALIBRATION_S", value)
    with pytest.raises(ValueError, match="MERLIN_CAPSULE_CALIBRATION_S"):
        run_pipelined([_cap("first")], lambda c, n: {}, max_workers=2)


# --------------------------------------------------------------- measured longest-first


def _result(root, name, *, mtime, tiers):
    import json

    path = root / "runs" / "fixture-capsule-bench" / name / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"capsule": name, "status": "pass", "tiers": tiers}))
    os.utime(path, ns=(mtime, mtime))


def _timed(active, build=1.0, wall=None, cycles=None, measured=True):
    return {
        "status": "pass",
        "measured_now": measured,
        "cycles": cycles,
        "timing": {
            "build_s": build,
            "sim_active_s": active,
            "oracle_wait_s": 50.0,
            "adapter_wall_s": wall if wall is not None else active + build + 50.0,
        },
    }


@pytest.fixture()
def history(tmp_path):
    old, new = tmp_path / "runs_r0000_t000001", tmp_path / "selfchecks" / "selfcheck_x"
    _result(old, "boundary", mtime=10**18, tiers={"L2": _timed(2.0), "L3": _timed(466.0, cycles=369000)})
    _result(old, "beyond", mtime=10**18, tiers={"L3": _timed(137.0, cycles=62000)})
    _result(old, "kdepth", mtime=10**18, tiers={"L3": _timed(101.0, cycles=40000)})
    _result(old, "tiny", mtime=10**18, tiers={"L3": _timed(25.0, cycles=300)})
    # Newer evidence wins per tier; a carried certificate is not a cost measurement.
    _result(
        new,
        "tiny",
        mtime=2 * 10**18,
        tiers={"L2": _timed(1.0), "L3": {"status": "pass", "measured_now": False, "cycles": 300, "timing": None}},
    )
    _result(new, "kdepth", mtime=2 * 10**18, tiers={"L3": _timed(5.0, cycles=40000)})
    return (old, new)


def test_measured_costs_use_newest_measured_tier_and_ignore_carried(history):
    from merlin.targetgen.capsule_scheduling import measured_costs

    seconds, cycles = measured_costs(history)
    assert seconds == {"boundary": 3.0 + 467.0, "beyond": 138.0, "kdepth": 6.0, "tiny": 2.0 + 26.0}
    assert cycles["boundary"] == 369000 and cycles["tiny"] == 300


def test_measured_longest_first_starts_the_known_tail_first(monkeypatch, history):
    from merlin.targetgen.capsule_scheduling import timing_history

    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    caps = [_cap(name, shape=(1,)) for name in ("tiny", "kdepth", "fresh", "beyond", "boundary")]
    order = []

    def run(cap, parallel):
        order.append(cap["name"])
        return {"capsule": cap["name"], "status": "pass" if cap["name"] != "beyond" else "fail"}

    with timing_history(history):
        rows = run_pipelined(caps, run, max_workers=1)
    # Unmeasured "fresh" ranks as the median priced member of this batch (83 s): after the known
    # long members, ahead of the known short ones.
    assert order == ["boundary", "beyond", "fresh", "tiny", "kdepth"]
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "legacy")
    legacy = run_pipelined(caps, run, max_workers=1)
    assert rows == legacy, "start order never changes a result or its input-order position"


def test_cycles_predict_unmeasured_members_and_operand_volume_then_name_break_ties(monkeypatch, history):
    from merlin.targetgen.capsule_scheduling import cost_estimates, measured_costs, timing_history

    seconds, cycles = measured_costs(history)
    names = ("b_unknown", "kdepth", "long_unmeasured", "tiny", "beyond", "boundary", "a_unknown", "c_wide")
    caps = [_cap(name, shape=(64, 64) if name == "c_wide" else (4,)) for name in names]
    estimate = cost_estimates(caps, seconds, {**cycles, "long_unmeasured": 300000})
    assert estimate["boundary"] == 470.0
    assert estimate["boundary"] > estimate["long_unmeasured"] > estimate["beyond"] == estimate["a_unknown"]
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    _result(
        history[1],
        "long_unmeasured",
        mtime=3 * 10**18,
        tiers={"L3": {"status": "pass", "measured_now": False, "cycles": 300000, "timing": None}},
    )
    order = []
    with timing_history(history):
        run_pipelined(caps, lambda c, n: order.append(c["name"]) or {}, max_workers=1)
    assert order == ["boundary", "long_unmeasured", "c_wide", "a_unknown", "b_unknown", "beyond", "tiny", "kdepth"]


def test_measured_order_keeps_dependencies_caller_priority_and_parallel_results(monkeypatch, history):
    from merlin.targetgen.capsule_scheduling import timing_history

    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    caps = [_cap("tiny"), _cap("boundary", extends="tiny"), dict(_cap("model"), kind="model"), _cap("beyond")]
    order = []

    def run(cap, parallel):
        order.append(cap["name"])
        if cap["name"] == "boundary":
            assert "tiny" in order[:-1], "a dependent never starts before its sibling"
        return {"capsule": cap["name"], "status": "pass"}

    with timing_history(history):
        rows = run_pipelined(caps, run, max_workers=1, priority=lambda cap: 0 if cap.get("kind") == "model" else 1)
    assert order == ["model", "beyond", "tiny", "boundary"]
    with timing_history(history):
        parallel = run_pipelined(caps, lambda c, n: {"capsule": c["name"], "status": "pass"}, max_workers=3)
    assert [row["capsule"] for row in rows] == [row["capsule"] for row in parallel] == [c["name"] for c in caps]


def test_selfcheck_processes_read_history_from_the_selected_environment(monkeypatch, history, tmp_path):
    from merlin.targetgen.capsule_scheduling import selected_history

    monkeypatch.setenv("MERLIN_CAPSULE_TIMING_HISTORY", os.pathsep.join([str(tmp_path), str(tmp_path / "selfchecks")]))
    roots = selected_history()
    assert history[0] in roots and history[1] in roots
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    order = []
    run_pipelined([_cap("tiny"), _cap("boundary")], lambda c, n: order.append(c["name"]) or {}, max_workers=1)
    assert order == ["boundary", "tiny"]


def test_measured_order_refuses_a_certify_budget(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "600")
    with pytest.raises(ValueError, match="unlimited certify budget"):
        run_pipelined([_cap("a")], lambda *a: pytest.fail("started under a budget"), max_workers=1)


def test_without_history_measured_order_is_longest_declared_operands_then_name(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "measured-longest-first")
    monkeypatch.delenv("MERLIN_CAPSULE_TIMING_HISTORY", raising=False)
    order = []
    caps = [_cap("b_small", shape=(2,)), _cap("big", shape=(64, 64)), _cap("a_small", shape=(2,))]
    run_pipelined(caps, lambda c, n: order.append(c["name"]) or {}, max_workers=1)
    assert order == ["big", "a_small", "b_small"]


@pytest.mark.parametrize("defer_running", [False, True])
def test_superseded_snapshot_drains_started_workers_and_defers_full_pending_roster(defer_running):
    from merlin.targetgen.capsule_scheduling import active_defer_probe

    started = threading.Barrier(2)
    obsolete, observed = threading.Event(), threading.Event()
    calls = []

    def reason():
        if obsolete.is_set():
            observed.set()
            return "background source superseded"
        return None

    def run(cap, workers):
        calls.append(cap["name"])
        started.wait(timeout=2)
        if cap["name"] == "b":
            assert observed.wait(timeout=2)
            probe = active_defer_probe()
            assert (probe() if probe else None) == ("background source superseded" if defer_running else None)
        return {"capsule": cap["name"], "status": "pass"}

    def completed(row):
        if row["capsule"] == "a":
            obsolete.set()

    rows = run_pipelined(
        [_cap(name) for name in "abcd"],
        run,
        max_workers=2,
        on_complete=completed,
        defer_pending=reason,
        defer_running=defer_running,
    )
    assert sorted(calls) == ["a", "b"]
    assert [row["capsule"] for row in rows] == list("abcd")
    assert [row["status"] for row in rows] == ["pass", "pass", "gated", "gated"]
    assert all(row["background_deferred"] == "background source superseded" for row in rows[2:])
