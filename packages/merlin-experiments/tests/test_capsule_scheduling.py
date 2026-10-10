"""Capsule grading is pipelined: no wave barrier, dependents wait only for their own sibling."""

import threading
import time

import pytest

from merlin.targetgen.capsule_scheduling import operand_volume, run_pipelined


def _cap(name, extends=None, shape=(16, 16)):
    cap = {"name": name, "inputs": [{"name": "x", "shape": list(shape)}]}
    if extends:
        cap["extends"] = extends
    return cap


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
