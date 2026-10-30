"""MERLIN_GRADE_PROCS: in-run operator grades through helper interpreters (protocol, deferral, fallback).

The helpers here are fake protocol peers; the real helper bootstrap (original owners, custody) is the
unchanged operator_child route and is exercised by test_operator_grade_process.py.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import grade_pool as GP
from merlin_experiments.phase1.feedback import operator_child as OC
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen import capsule_scheduling as CS

FAKE = r'''
import json, os, sys, threading, time
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
cancel_dir = Path(spec["cancel_dir"])
lock = threading.Lock()
def emit(m):
    with lock:
        sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
emit({"op": "ready", "pid": os.getpid(), "identity": spec["identity"]})
def job(m):
    rid, doc = m["id"], m["document"]
    name = doc["capsule"]["name"]
    if name.startswith("ok"):
        time.sleep(0.3)
        emit({"op": "done", "id": rid, "row": {"capsule": name, "status": "pass", "helper": os.getpid()},
              "costs": {"L2": 1.5}, "spend": 2.0, "no_spend": False})
    elif name == "defer":
        while not (cancel_dir / rid).exists():
            time.sleep(0.05)
        reason = (cancel_dir / rid).read_text()
        emit({"op": "done", "id": rid, "row": {"capsule": name, "status": "gated", "background_deferred": reason},
              "costs": {}, "spend": None, "no_spend": True})
    elif name == "gate":
        emit({"op": "error", "id": rid, "error": {"type": "StageGateError", "message": "child refused"}})
    elif name == "crash":
        os._exit(3)
for line in sys.stdin:
    threading.Thread(target=job, args=(json.loads(line),), daemon=True).start()
'''


def _document(name):
    return {"capsule": {"name": name, "kind": "isa"}, "target": "fake", "adapters": {"L2": {"engine": "spike"}}}


@pytest.fixture
def pool(tmp_path, monkeypatch):
    script = tmp_path / "fake_helper.py"
    script.write_text(FAKE)
    monkeypatch.setattr(GP, "_helper_command", lambda spec: [sys.executable, str(script), str(spec)])
    monkeypatch.setattr(GP, "_identity", lambda: {"source_sha256": "s", "snapshot_digest": "d"})
    inputs = SimpleNamespace(
        context=SimpleNamespace(descriptor="d", repo="r", harness=None, readback_policy=None, candidate_policy=None),
        arm="a", public_root=lambda: tmp_path, contract=None, additional_forbidden=(), compiler_profile_service=object(),
    )
    created = GP.GradePool(inputs=inputs, candidate=tmp_path / "cand", run_dir=tmp_path / "run", processes=3)
    token = GP._POOL.set(created)
    yield created
    GP._POOL.reset(token)
    created.close()


def test_knob_defaults_and_single_process_values():
    assert GP.configured({}) == GP.DEFAULT_PROCS == 8
    for raw in ("0", "1", "", "x", "-2"):
        assert GP.configured({GP.ENV: raw}) == 0
    assert GP.configured({GP.ENV: "12"}) == 12


def test_no_scope_keeps_the_in_process_route():
    assert GP.dispatch(_document("ok")) is None


def test_requests_complete_concurrently_across_helpers_and_return_costs(pool):
    started = time.monotonic()
    with ThreadPoolExecutor(6) as workers:  # the scheduler's submit copies the submitting thread's context
        futures = [workers.submit(copy_context().run, GP.dispatch, _document(f"ok{n}")) for n in range(6)]
        results = [future.result() for future in futures]
    assert time.monotonic() - started < 0.3 * 6  # overlapped, not serialized
    assert [r[0]["capsule"] for r in results] == [f"ok{n}" for n in range(6)]
    assert all(r[1] == {"L2": 1.5} and r[2] == 2.0 for r in results)
    assert len({r[0]["helper"] for r in results}) == 3
    assert pool.served == 6 and pool.broken is None


def test_scheduler_deferral_reaches_the_helper_child(pool):
    token = CS._DEFER_PROBE.set(lambda: "newer snapshot")
    try:
        row, costs, spend = GP.dispatch(_document("defer"))
    finally:
        CS._DEFER_PROBE.reset(token)
    assert row["background_deferred"] == "newer snapshot" and costs == {} and spend is OC._NO_SPEND


def test_capsule_errors_are_raised_not_regraded(pool):
    with pytest.raises(StageGateError, match="child refused"):
        GP.dispatch(_document("gate"))
    assert pool.broken is None


def test_helper_death_falls_back_and_stays_in_process(pool):
    assert GP.dispatch(_document("ok")) is not None
    assert GP.dispatch(_document("crash")) is None
    assert pool.broken is not None
    assert GP.dispatch(_document("ok")) is None


def test_start_failure_falls_back(pool, monkeypatch):
    monkeypatch.setattr(GP, "_helper_command", lambda spec: [sys.executable, "-c", "import sys; sys.exit(2)"])
    assert GP.dispatch(_document("ok")) is None
    assert pool.broken.startswith("start failed")


def test_operator_grade_applies_pooled_costs_in_the_controller(monkeypatch):
    recorded, spent = [], []
    monkeypatch.setattr(OC, "adapter_selection", lambda adapters, target: {"L2": {}})
    monkeypatch.setattr(OC.tier_policy, "observed_cost", lambda target, tier: None)
    monkeypatch.setattr(OC.tier_policy, "record_cost", lambda target, tier, value: recorded.append((tier, value)))
    monkeypatch.setattr(OC.tier_policy, "note_spend", lambda target, value: spent.append(value))
    monkeypatch.setattr(GP, "dispatch", lambda document: ({"capsule": document["capsule"]["name"]}, {"L2": 4.0}, 9.0))
    monkeypatch.setattr(OC, "run_document", lambda *a, **k: pytest.fail("pooled grade must not run in-process"))
    row = OC.grade({"name": "c", "kind": "isa"}, "/pkg", runs_root="/r", contract=None, target="t", suite=None,
                   dtype="i8", adapters={}, timeout=5, workers=1, no_oracle=False)
    assert row == {"capsule": "c"} and recorded == [("L2", 4.0)] and spent == [9.0]
    monkeypatch.setattr(GP, "dispatch", lambda document: ({"capsule": "c", "status": "gated"}, {}, OC._NO_SPEND))
    recorded.clear(), spent.clear()
    OC.grade({"name": "c", "kind": "isa"}, "/pkg", runs_root="/r", contract=None, target="t", suite=None,
             dtype="i8", adapters={}, timeout=5, workers=1, no_oracle=False)
    assert recorded == [] and spent == []  # a deferred child prices nothing (original early return)
