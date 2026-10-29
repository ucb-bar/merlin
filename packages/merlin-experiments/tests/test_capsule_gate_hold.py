"""A capstone waiting on its operator gate is not deferred by the admission window while that gate can open."""

import threading
import time

import pytest

from merlin.targetgen import capsule_scheduling as S


def _grade(monkeypatch, *, current, hold=True, hold_s="60"):
    ops = [{"name": f"op{i}", "kind": "layer"} for i in range(3)]
    model = {"name": "whole", "kind": "model"}
    done = set()
    lock = threading.Lock()

    def run_one(cap, _parallel):
        if cap["kind"] != "model":
            time.sleep(0.4)  # operators outlive the admission window
            with lock:
                done.add(cap["name"])
        return {"capsule": cap["name"], "status": "pass"}

    def can_start(cap):
        if cap["kind"] != "model":
            return True
        with lock:
            return len(done) >= 3  # the capstone's gate opens only after every operator passed

    monkeypatch.setenv("MERLIN_SNAPSHOT_GATE_HOLD_S", hold_s)
    closed = []
    with S.observe_admission(closed.append, current=lambda: current):
        with S.rolling_admission(admission_window_s=0.1,
                                 admission_hold=(lambda cap: cap["kind"] == "model") if hold else None):
            results = S.run_pipelined(ops + [model], run_one, max_workers=4, can_start=can_start,
                                      blocked_result=lambda cap: {"capsule": cap["name"], "status": "gated"},
                                      priority=lambda cap: 0 if cap["kind"] == "model" else 1)
    return {r["capsule"]: r for r in results}, closed


def test_the_capstone_starts_once_its_gate_opens_after_the_window(monkeypatch):
    results, closed = _grade(monkeypatch, current=True)
    assert results["whole"]["status"] == "pass"
    assert closed  # the owner still learned admission closed, so a newer snapshot may start


def test_a_superseded_snapshot_does_not_hold_admission_open(monkeypatch):
    results, _ = _grade(monkeypatch, current=False)
    assert results["whole"]["status"] == "gated" and results["whole"].get("admission_deferred")


def test_without_a_hold_callback_the_window_behaves_as_before(monkeypatch):
    results, _ = _grade(monkeypatch, current=True, hold=False)
    assert results["whole"]["status"] == "gated"


def test_the_hold_is_bounded(monkeypatch):
    results, _ = _grade(monkeypatch, current=True, hold_s="0")
    assert results["whole"]["status"] == "gated"
