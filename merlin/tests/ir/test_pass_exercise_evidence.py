"""Every catalogued production pass is reachable by a real model, and the gate can MEASURE it.

`check_pass_obligations.py --fail-on-dead` asks whether any capsule run actually invokes each
catalogued pass, and it answers from EVIDENCE: the entry points record their invocations to a JSONL log
and the gate reads it. With no log it exits 2 -- "a check that could not run has established nothing" --
which is the state it shipped in, so the dead-pass half of the gate had never actually run.

This produces the evidence the same way the grading path does (`passes.install_pass_recorder` plus
`pass_run_context` attributing to a capsule and its declared requirement classes) and over a REAL
captured model rather than a fixture, then asserts the gate reports all four exercised. Measured: 4/4,
with the dispatch program carrying 488 nodes for small_llama -- so this is reachability by real work,
not a formal touch.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest
import yaml

from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.xdsl_dialects.lowering import passes as PS

_CAPSTONE = merlin_dir() / "contract" / "capsules" / "model" / "M2_microvit_gemmini"
_CAPTURE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent" / "model.mlir"


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_pass_obligations.py"
    spec = importlib.util.spec_from_file_location("_pass_gate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pass_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _drive(log_path, monkeypatch) -> dict:
    """Run the four production passes over a real capture, recording to ``log_path``.

    ⚠️ CALLED THROUGH THEIR MODULES, never through a local binding. The recorder wraps the callable in
    its defining module and `_rebind_aliases` sweeps already-imported modules for other module-level
    references -- which is why `dispatch_program`'s own `from .outline import outline_dispatches` is
    recorded correctly. What no sweep can reach is a name bound in a FUNCTION's locals, so a driver
    written as `from ... import add_c_interface` then `add_c_interface(m)` calls the unwrapped original
    and the pass looks dead. That was this test's first shape, and the two passes it invoked that way
    were the two the gate reported missing.
    """
    from merlin.llvmlower import passes_xdsl as PX
    from merlin.targetgen.model_coverage import load_module
    from merlin.xdsl_dialects.lowering import dispatch_program as DP
    from merlin.xdsl_dialects.lowering import schedule_dispatch as SD

    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log_path))
    PS.install_pass_recorder()
    cap = yaml.safe_load((_CAPSTONE / "capsule.yaml").read_text(encoding="utf-8"))
    reqs = tuple(cap.get("pass_requirements") or ())
    assert reqs, (
        "the capstone must declare its requirement classes, or an exercised pass has no requiring "
        "capsule to be attributed to")
    with PS.pass_run_context(str(cap["name"]), reqs):
        module = load_module(_CAPTURE)
        _outlined, prog = DP.lower_model_to_dispatch_program(module)
        SD.partition_dispatches(prog, n_harts=4)
        PX.add_c_interface(module)
    return {"n_program_nodes": len(getattr(prog, "nodes", ()) or ())}


@pytest.mark.skipif(not _CAPTURE.is_file(), reason="no captured model to lower")
@pytest.mark.skipif(not (_CAPSTONE / "capsule.yaml").is_file(), reason="no capstone capsule")
def test_the_production_passes_are_exercised_by_a_real_model(tmp_path, monkeypatch):
    log = tmp_path / "passes.jsonl"
    info = _drive(log, monkeypatch)
    assert info["n_program_nodes"] > 0, (
        "the dispatch program is empty, so the passes were touched but did no work; that is a formal "
        "invocation, not evidence the pass is part of the compiler")

    records = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    invoked = {r.get("pass") or r.get("name") for r in records if r.get("kind") == "invoke"}
    catalogued = {p.name for p in PS.catalog()}
    missing = catalogued - invoked
    assert not missing, (
        f"catalogued pass(es) {sorted(missing)} were not invoked by lowering a real model; either the "
        f"pass is furniture or this driver no longer reaches it")

    # ⚠️ INVOKED IS NOT EXERCISED. Every invocation must be MEASURED to have done something, or this
    # test certifies a formal touch. Measured on this capture: 183 dispatches outlined, 1541 program
    # elements built, 9 partition stats, 1 func marked -- and note the outliner's subject signature is
    # IDENTICAL before and after, because it builds a new module rather than mutating the one it was
    # handed. A subject-only probe would call that a no-op, which is why the product is measured too.
    for r in (r for r in records if r.get("kind") == "invoke"):
        assert r.get("effect") == PS.EFFECT_CHANGED, (
            f"{r['pass']} ran to effect {r.get('effect')!r}: {r.get('evidence')}")
        assert (r.get("evidence") or {}).get("produced", 0) > 0, (
            f"{r['pass']} produced nothing measurable: {r.get('evidence')}")


def test_a_pass_that_runs_and_transforms_nothing_is_not_reported_exercised(tmp_path, monkeypatch):
    """The distinction the gate exists to keep, proven on the SAME pass in both directions.

    "A capsule reaches this pass" and "a capsule reaches this pass's work" are different claims, and
    the log recorded only the first: a driver over a one-func module invoked the outliner, outlined
    ZERO kernels, and the gate called it `exercised`. So a pass certified by such a run would be
    furniture the audit had blessed. The verdict must be `exercised_noop` -- neither `dead` (something
    calls it) nor `exercised` (nothing it does was reached).
    """
    pytest.importorskip("xdsl")
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    from merlin.xdsl_dialects.lowering import outline as OL

    log = tmp_path / "noop.jsonl"
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log))
    PS.install_pass_recorder()

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    module = Parser(ctx, "func.func @forward(%a: f32) -> f32 { func.return %a : f32 }").parse_module()
    with PS.pass_run_context("TINY_no_dispatch", ("model-boundary-capstone",)):
        result = OL.outline_dispatches(module)
    assert not result.dispatches, "the fixture must give the pass nothing to do, or it proves nothing"

    rep = PS.exercise_report(logs=[log])
    assert rep["per_pass"]["merlin-outline-dispatches"]["status"] == "exercised_noop"

    gate = _gate()
    assert gate.main(["--log", str(log), "--fail-on-noop"]) == 1, (
        "a pass that reached none of its work must fail the noop gate")
    # And the same evidence must NOT be spendable as liveness: dead and noop are distinct verdicts.
    f = gate.findings(gate.audit([log]), set())
    assert [it["pass"] for it in f["noop"]] == ["merlin-outline-dispatches"]
    assert "merlin-outline-dispatches" not in [it["pass"] for it in f["dead"]]


@pytest.mark.skipif(not _CAPTURE.is_file(), reason="no captured model to lower")
def test_the_gate_reports_no_dead_pass_when_given_that_evidence(tmp_path, monkeypatch):
    """End to end: the gate's dead-pass half goes from "could not run" to a verdict.

    Without a log it exits 2 by design. That is the honest answer and it is also why the check had
    never established anything, so the transition is the thing worth pinning.
    """
    gate = _gate()
    log = tmp_path / "passes.jsonl"

    monkeypatch.delenv(PS.PASS_LOG_ENV, raising=False)
    assert gate.main(["--fail-on-dead"]) == 2, (
        "with no log the gate must report UNMEASURED, never clean")

    _drive(log, monkeypatch)
    assert gate.main(["--log", str(log), "--fail-on-dead", "--fail-on-undischarged",
                      "--fail-on-unrequired", "--fail-on-noop"]) == 0, (
        "with real evidence every catalogued pass must be discharged, required and exercised -- and "
        "--fail-on-noop is part of that bar, or 'exercised' still means 'called'")

    monkeypatch.delenv(PS.PASS_LOG_ENV, raising=False)
    assert gate.main(["--fail-on-noop"]) == 2, (
        "with no log the noop half must report CANNOT DECIDE, never clean")
