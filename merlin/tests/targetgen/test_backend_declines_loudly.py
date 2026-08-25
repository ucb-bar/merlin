"""A backend that cannot lower a shape must be able to SAY so.

Before this, it had one way to refuse: emit a program that writes nothing. That arrives at the grader as
an output full of zeros and is scored as "your artifact does not compute the declared operation" -- the
same words a wrong multiply gets. Measured on a real submission: twelve shape-keyed builders chained with
``or``, falling through to a bare terminator, so twelve capsules were reported as arithmetic failures for
programs that had never been emitted. The agent's whole feedback loop pointed at the wrong defect.

A decline is NOT a pass -- the capsule stays in the denominator, uncertified -- but it is a different
result from a wrong answer, and the two need opposite repairs. These tests pin that difference at each
place a reader could lose it: the contract front half, the capsule result, the score, and the headline.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen.contract import schemas
from merlin.targetgen.oot_runner import BackendDeclined


def _cb(**extra):
    return {"abi_version": "1.0.0", "target": "t", "commands": [], **extra}


# --------------------------------------------------------------- the contract surface

def test_the_command_buffer_schema_accepts_a_decline():
    schemas.validate_command_buffer(
        _cb(declined={"reason": "no M-tiling: this backend lowers M=32 only", "shape": [64, 32, 32],
                      "op": "matmul"}))


def test_a_decline_must_say_what_it_could_not_lower():
    """A reason-less decline is a silent drop with extra steps; the schema refuses it."""
    with pytest.raises(schemas.ContractViolation):
        schemas.validate_command_buffer(_cb(declined={"shape": [64, 32, 32]}))


def test_the_exception_carries_shape_and_op_for_the_feedback_line():
    bd = BackendDeclined("lowers M=32 only", shape=(64, 32, 32), op="matmul")
    assert bd.to_dict() == {"reason": "lowers M=32 only", "shape": [64, 32, 32], "op": "matmul"}
    assert BackendDeclined("no").to_dict() == {"reason": "no"}, "absent stays absent"


# --------------------------------------------------------------- the result surface

def test_the_capsule_result_schema_accepts_the_declined_status():
    schemas.validate({"capsule": "A0", "status": "declined", "contract_version": "1.0.0",
                      "declined": {"reason": "lowers M=32 only", "shape": [64, 32, 32]},
                      "tiers": {}, "trace_check": {"status": "skipped"},
                      "numeric": {"status": "skipped"}, "failure": None}, "capsule_result")


# --------------------------------------------------------------- the score surface

def _score(monkeypatch, results):
    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG, "source_experiment_env", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]}
                                                                     for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    return CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="atlas", max_workers=1)


def _op(name, status="pass", **extra):
    return {"capsule": name, "kind": "op", "label": "public", "status": status,
            "tiers": {"L4": {"status": status, "derived_from_rtl": True}},
            "numeric": {"status": status}, "trace_check": {"status": status}, **extra}


def _declined_op(name, shape):
    return _op(name, status="declined",
               declined={"reason": "this backend lowers M=32 only", "shape": shape, "op": "matmul"})


def test_a_decline_stays_in_the_denominator_and_out_of_the_numerator(monkeypatch):
    """The backend was ASKED and said no. That is uncertified, so it counts against the suite.

    The alternative -- excluding it like `screened_only` -- would let a backend decline everything and
    score a perfect 0/0.
    """
    s = _score(monkeypatch, [_op("A0"), _declined_op("A1", [64, 32, 32])])
    assert (s["n_passed"], s["n_capsules"]) == (1, 2)
    assert s["functional_pass"] == 0


def test_the_declines_are_named_with_their_shapes(monkeypatch):
    s = _score(monkeypatch, [_declined_op("A1", [64, 32, 32]), _op("A0")])
    assert s["n_declined"] == 1
    assert s["declined"] == [{"capsule": "A1", "reason": "this backend lowers M=32 only",
                              "shape": [64, 32, 32], "op": "matmul"}]
    assert "coverage gaps" in s["declined_note"]


def test_the_headline_separates_a_decline_from_a_wrong_answer(monkeypatch):
    """'14/26' reads as 12 wrong answers. If they were declines that is a different repair entirely."""
    s = _score(monkeypatch, [_op("A0"), _declined_op("A1", [64, 32, 32]),
                             _declined_op("A2", [64, 64, 32])])
    assert "2 DECLINED by the backend" in s["headline"], s["headline"]
    assert "not wrong answers" in s["headline"]


def test_no_declines_leaves_the_headline_untouched(monkeypatch):
    """The qualifier appears only when it is true -- an unconditional one stops being read."""
    s = _score(monkeypatch, [_op("A0")])
    assert s["n_declined"] == 0
    assert "DECLINED" not in s["headline"]
    assert "declined" not in s


# --------------------------------------------------------------- the feedback surface

def test_the_round_verdict_carries_the_decline_to_the_agent(monkeypatch, tmp_path):
    """`qa/verdict.json` is the only channel between rounds; a decline invisible here is invisible."""
    import sys
    sys.path.insert(0, str(__import__("merlin.common.paths", fromlist=["x"]).repo_root()
                           / "merlin/experiments/capsule_bench/harness"))
    import qa_check

    rr = tmp_path / "runs" / "atlas-capsule-bench" / "A1"
    rr.mkdir(parents=True)
    (rr / "capsule_result.json").write_text(json.dumps({
        "capsule": "A1", "status": "declined",
        "declined": {"reason": "lowers M=32 only", "shape": [64, 32, 32]},
        "numeric": {"status": "skipped"}, "tiers": {}, "failure": None}))

    red = qa_check._per_capsule_from_results(tmp_path)
    assert red["A1"]["declined"] == {"reason": "lowers M=32 only", "shape": [64, 32, 32]}


def test_an_empty_program_still_needs_a_stated_reason():
    """The relaxation is scoped to declines ONLY.

    An empty command list is exactly the silent-drop this change exists to eliminate, so it stays a
    contract violation for any buffer that does not say why it is empty. Relaxing `minItems`
    unconditionally would have legalised the original bug.
    """
    with pytest.raises(schemas.ContractViolation):
        schemas.validate_command_buffer(_cb())
