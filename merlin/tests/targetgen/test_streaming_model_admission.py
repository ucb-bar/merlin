"""Model capstones share the operator pool and wait for a fixed full-roster gate."""
from threading import Event, Lock

import pytest

from merlin.targetgen import capsule_runner as R


def cap(name, *, kind="op", volume=1, threshold=None, extends=None):
    result = {"name": name, "kind": kind, "inputs": [{"shape": [volume]}]}
    if threshold is not None:
        result["gate"] = {"after_op_pass_fraction": threshold}
    if extends is not None:
        result["extends"] = extends
    return result


def run(ops, models, worker, monkeypatch, **kwargs):
    monkeypatch.setattr(R, "_gate_counts", lambda *args: True)
    return R._run_streaming_model_gate(ops, models, worker, target="unknown", oracle_adapters=None, **kwargs)


def test_partial_completed_denominator_cannot_open_gate(monkeypatch):
    started = []
    ops = [cap("pass", volume=2), cap("error")]
    model = cap("model", kind="model", threshold=.75)

    def worker(c, n):
        started.append(c["name"])
        return {"capsule": c["name"], "status": "pass" if c["name"] == "pass" else "error"}

    rows = run(ops, [model], worker, monkeypatch, max_workers=1)
    assert started == ["pass", "error"]
    assert rows[-1]["status"] == "gated"
    assert "0.50" in rows[-1]["failure"]["detail"]


def test_model_starts_while_unrelated_slow_operator_is_live(monkeypatch):
    slow_started, release, model_started = Event(), Event(), Event()
    completions = []
    ops = [cap("slow", volume=100), cap("fast", volume=50), cap("last")]
    model = cap("model", kind="model", threshold=1/3)

    def worker(c, n):
        if c["name"] == "slow":
            slow_started.set()
            assert release.wait(3), "model was blocked behind unrelated operator batch"
        elif c["name"] == "fast":
            assert slow_started.wait(3)
        elif c["name"] == "model":
            assert not release.is_set()
            model_started.set()
            release.set()
        return {"capsule": c["name"], "status": "pass"}

    rows = run(ops, [model], worker, monkeypatch, max_workers=2,
               on_complete=lambda row: completions.append(row["capsule"]))
    assert model_started.is_set()
    assert completions[0] == "fast"
    assert len(rows) == 4


def test_simultaneous_models_respect_same_pool_and_extends(monkeypatch):
    lock, sibling_started = Lock(), Event()
    state = {"active": 0, "peak": 0}
    completed = []
    models = [cap("parent", kind="model", volume=3), cap("sibling", kind="model", volume=2),
              cap("child", kind="model", extends="parent")]

    def worker(c, n):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            if c["name"] == "parent":
                assert sibling_started.wait(3)
            if c["name"] == "sibling":
                sibling_started.set()
            if c["name"] == "child":
                assert "parent" in completed
            assert n == 2
            return {"capsule": c["name"], "status": "pass"}
        finally:
            with lock:
                state["active"] -= 1

    rows = run([], models, worker, monkeypatch, max_workers=2,
               on_complete=lambda row: completed.append(row["capsule"]))
    assert len(rows) == 3
    assert state["peak"] == 2


def test_operator_cannot_depend_on_gated_model(monkeypatch):
    with pytest.raises(ValueError, match="dependency conflicts"):
        run([cap("op", extends="model")], [cap("model", kind="model", threshold=.8)],
            lambda *args: pytest.fail("no job may run"), monkeypatch, max_workers=2)


def test_calibration_selects_operators_only(monkeypatch):
    started = []
    def worker(c, n):
        started.append((c["name"], n))
        return {"capsule": c["name"], "status": "pass"}
    run([cap("operator", volume=10)], [cap("model", kind="model")], worker, monkeypatch,
        max_workers=2, calibrate=lambda: set(), calibration_cap=2)
    assert started[0] == ("operator", 1)


@pytest.mark.parametrize("budget,streamed", [(None, True), (100, False)])
def test_unbudgeted_suite_streams_and_historical_budget_path_remains(monkeypatch, tmp_path, budget, streamed):
    monkeypatch.setattr(R, "load_package", lambda *args, **kwargs: object())
    monkeypatch.setattr(R, "integrity_scan", lambda *args: None)
    monkeypatch.setattr(R, "build_package", lambda *args: None)
    monkeypatch.setattr(R, "_split_ineligible", lambda caps, target: (caps, []))
    monkeypatch.setattr(R, "_gate_counts", lambda *args: True)
    monkeypatch.setattr(R._tier_policy, "budget_seconds", lambda: budget)
    monkeypatch.setattr(R._tier_policy, "covering_set", lambda caps: [])
    monkeypatch.setattr(R._tier_policy, "priced_tiers", lambda target: {"L2"})
    monkeypatch.setattr(R, "run_capsule", lambda c, *args, **kwargs: {"capsule": c["name"], "status": "pass"})
    called = []
    original = R._run_streaming_model_gate
    def observed(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(R, "_run_streaming_model_gate", observed)
    rows = R._run_suite([cap("op"), cap("model", kind="model", threshold=.8)], tmp_path,
                        runs_root=tmp_path / "runs", target="unknown", max_workers=2)
    assert bool(called) is streamed
    assert [r["status"] for r in rows] == ["pass", "pass"]
