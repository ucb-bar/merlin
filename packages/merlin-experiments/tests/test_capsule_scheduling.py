"""Capsule grading is pipelined: no wave barrier, dependents wait only for their own sibling."""

import threading
import time
import os
import subprocess
import sys

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


@pytest.mark.parametrize("ordering,expected", [
    ("legacy", ["model", "large", "small", "dependent"]),
    ("operators-small-first", ["small", "large", "dependent", "model"]),
])
def test_optional_shortest_operator_order_preserves_dependencies_and_legacy_priority(monkeypatch, ordering, expected):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", ordering)
    caps = [_cap("large", shape=(64, 64)), dict(_cap("model", shape=(1,)), kind="model"),
            _cap("small", shape=(2,)), _cap("dependent", extends="large", shape=(1,))]
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
            time.sleep(.02)
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
    rows = run_pipelined(caps, run, max_workers=2, on_complete=complete,
                        can_start=lambda cap: cap.get("kind") != "model" or fast_done.is_set())
    assert callbacks[0] == "fast"
    assert model_started.is_set()
    assert [row["capsule"] for row in rows] == [cap["name"] for cap in caps]


def test_invalid_capsule_order_refuses_before_execution(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "unknown")
    with pytest.raises(ValueError, match="MERLIN_CAPSULE_ORDER"):
        run_pipelined([_cap("unused")], lambda *a: pytest.fail("invalid selection started work"), max_workers=2)
