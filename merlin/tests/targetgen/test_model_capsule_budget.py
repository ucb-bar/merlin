"""A whole-model capsule must be bounded by a wall-clock BUDGET, not by a per-step timeout.

MEASURED (gemmini arm-4 calibration ``merlincirct_defcal1``, 2026-08-29): the agent round finished
in 40 min; the whole-model capstone was then scheduled (its op-pass gate cleared at 18/22 = 0.82),
ran a cycle-accurate Verilator simulation of the entire model for 5 h 30 m past the round's own 4 h
timeout, wrote not one byte into its run directory, and the round never graded. Its ``--qa-timeout``
was 900 s -- a PER-STEP subprocess cap, which a grade making many such calls can never be bounded by.
"""
from __future__ import annotations

import json
import time

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_grade as CGR
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES


def test_budget_is_unlimited_by_default(monkeypatch):
    """No ceiling unless one is asked for: an operator certification run legitimately takes hours."""
    monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    assert CR.model_budget_seconds() is None
    for bad in ("", "   ", "not-a-number", "0", "-5"):
        monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", bad)
        assert CR.model_budget_seconds() is None, bad
    monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", "90")
    assert CR.model_budget_seconds() == 90.0


def test_no_budget_runs_inline(monkeypatch):
    """With no budget the grade is the plain in-process call — no subprocess, no behaviour change."""
    seen = {}

    def _inline(capsule, *, target, timeout, package_dir):
        seen.update(capsule=capsule["name"], target=target, timeout=timeout)
        return {"capsule": capsule["name"], "status": "pass"}

    monkeypatch.setattr(CR, "_grade_model_capsule_inline", _inline)
    out = CR._grade_model_capsule({"name": "M0"}, target="t", timeout=7, budget_s=0)
    assert out["status"] == "pass"
    assert seen == {"capsule": "M0", "target": "t", "timeout": 7}


@pytest.mark.parametrize("budget", [3.0])
def test_budget_stops_a_grade_that_overruns(tmp_path, monkeypatch, budget):
    """The ceiling is enforced on the CAPSULE. A grade that overruns is stopped and reported."""
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(600)\n")

    import subprocess as sp
    orig = sp.Popen

    def _fake_popen(cmd, **kw):
        # Same contract as the real child (own session, inherited env) but it never finishes.
        return orig(["python3", str(slow)], **kw)

    monkeypatch.setattr(sp, "Popen", _fake_popen)
    t0 = time.monotonic()
    out = CR._grade_model_capsule({"name": "GX0", "label": "public"},
                                  target="gemmini", timeout=900, budget_s=budget)
    elapsed = time.monotonic() - t0
    assert budget <= elapsed < budget + 20, elapsed
    assert out["status"] == "budget_exhausted"
    assert out["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert out["model_budget_s"] == budget
    assert "MERLIN_MODEL_BUDGET_S" in out["failure"]["detail"]


def test_the_result_schema_knows_the_status():
    """A status a runner can WRITE must be one the contract's own schema accepts."""
    from merlin.targetgen.contract import schemas
    schemas.validate({"capsule": "GX0", "status": "budget_exhausted", "contract_version": "0.1",
                      "tiers": {}, "trace_check": {"status": "skipped"},
                      "numeric": {"status": "skipped"},
                      "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                  "detail": "exceeded its budget"}}, "capsule_result")


def test_budget_exhausted_is_neither_numerator_nor_denominator():
    """A capsule stopped by its own clock measured NOTHING about the submission.

    Counting it as a failure is what makes ``all_pass`` unreachable, which disables an agent loop's
    only early exit and turns every run into a fixed-price purchase of its whole round budget.
    """
    assert "budget_exhausted" in NOT_MEASURED_STATUSES
    results = [{"capsule": "A0", "status": "pass"}, {"capsule": "A1", "status": "pass"},
               {"capsule": "GX0", "status": "budget_exhausted", "kind": "model"}]
    graded = [r for r in results if r.get("status") not in NOT_MEASURED_STATUSES]
    assert [r["capsule"] for r in graded] == ["A0", "A1"]
    assert sum(1 for r in graded if r["status"] == "pass") == len(graded)   # all_pass stays reachable


def test_the_stopped_capsule_is_reported_by_name(tmp_path, monkeypatch):
    """Excluded is not hidden: the real scorer names it, counts it, and says why it is not graded."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir(); caps.mkdir()
    results = [{"capsule": "A0", "label": "public", "status": "pass", "tiers": {}},
               {"capsule": "A1", "label": "public", "status": "pass", "tiers": {}},
               {"capsule": "GX0", "label": "public", "kind": "model", "tiers": {},
                "status": "budget_exhausted",
                "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                            "detail": "exceeded its 3600s budget"}}]
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: [])
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: results)

    score = CGR.grade(str(pkg), capsules_root=str(caps), runs_root=str(tmp_path / "runs"),
                      target="gemmini", contract=None, oracle_adapters={})

    assert score["n_budget_exhausted"] == 1
    assert score["budget_exhausted"] == ["GX0"]
    assert "MERLIN_MODEL_BUDGET_S" in score["budget_exhausted_note"]
    # neither numerator nor denominator -- and all_pass therefore still reachable
    assert score["n_capsules"] == 2 and score["n_passed"] == 2
    assert score["functional_pass"] == 1


class _StubPkg:
    integrity_exempt = False


def test_child_entry_writes_a_result(tmp_path, monkeypatch):
    """``--model-grade`` is the child half: one spec file in, one result file out."""
    spec, out = tmp_path / "spec.json", tmp_path / "res.json"
    spec.write_text(json.dumps({"capsule": {"name": "M0"}, "target": "gemmini",
                                "timeout": 30, "package_dir": None}))
    monkeypatch.setattr(CR, "_grade_model_capsule_inline",
                        lambda c, **kw: {"capsule": c["name"], "status": "pass"})
    assert CR.main(["--model-grade", str(spec), "--model-grade-out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"capsule": "M0", "status": "pass"}


def test_the_kill_reaches_a_grandchild_that_left_the_process_group(tmp_path):
    """A process GROUP is not the tree.

    MEASURED (2026-08-30): the budgeted grade's Verilator child had called ``setsid`` of its own, so it
    sat in neither the grade child's group nor its session. ``killpg`` reaped the python and left the
    simulator running with PPID 1, holding a core on a shared host.
    """
    import subprocess
    import sys as _sys

    # parent -> child that leaves the group and outlives a killpg
    script = tmp_path / "tree.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],\n"
        "                       start_new_session=True)\n"
        "print(kid.pid, flush=True)\n"
        "time.sleep(600)\n")
    proc = subprocess.Popen([_sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    grandchild = int(proc.stdout.readline().strip())
    assert grandchild in CR._descendants(proc.pid)

    t0 = time.monotonic()
    CR._kill_tree(proc.pid, grace=10.0)
    assert time.monotonic() - t0 < 8, "a zombie is not alive; the wait must not sit out its grace"
    proc.wait(timeout=10)
    assert not CR._running(grandchild), "the grandchild outside the group must be dead too"
