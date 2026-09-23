"""Per-capsule promotion into the cycle-accurate cert, and the verdict an agent gets to READ.

Two defects, both measured on the live run ``merlincirct_arm4_func_20260901_codex1``:

1. THE CERT CHECKPOINT WAS ALL-OR-NOTHING. Its gate was ``verdict["all_pass"]``, which requires EVERY
   capsule to clear the loop tier. 19 of 32 capsules passed the loop tier and 13 did not, so
   ``all_pass`` was False and the whole checkpoint phase — its ``VERILATOR_ATTEMPTS`` fix rounds
   included — never ran. The 19 capsules that had earned a cycle-accurate cert got none of it because
   of the other 13.

2. THE AGENT RAN ITS WHOLE TURN WITH NO VERDICT. There was no ``qa/`` directory at all and the agent's
   own closing message said ``qa/verdict.json`` was not present: the round's verdict is produced after
   the turn, and the mandatory ladder that produces it includes the cert tier (one capsule, GM0, cost
   1818s of Verilator alone), so nothing landed inside the 6184-second turn.

What these tests pin is the honesty of the fixes, not just their presence: a capsule that was never
submitted is ``not_promoted`` and is NEVER counted as a cert pass, the whole-corpus completion
predicates still require the FULL set, and the fast first verdict reports a withheld mandatory tier as
UNKNOWN rather than as success (this repo has "a check that could not run reported SUCCESS" on record).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys

import pytest
import yaml
from merlin_experiments.phase1.feedback import lifecycle as FL
from merlin_experiments.phase1.feedback import loop_grading as LG
from phase1_feedback import feedback_source

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, feedback_source(name, HARNESS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(e).__name__}: {e}")
    return mod


def _round_verdict(n_pass: int, n_fail: int, loop_tier: str = "L2") -> dict:
    """The measured shape: `n_pass` capsules clear the loop tier, `n_fail` do not, all_pass is False."""
    rows = [{"capsule": f"P{i}", "status": "pass", "tiers": {loop_tier: "pass"}} for i in range(n_pass)]
    rows += [
        {"capsule": f"F{i}", "status": "fail", "tiers": {loop_tier: "fail"}, "failure_plane": "numeric"}
        for i in range(n_fail)
    ]
    return {"all_pass": False, "n_passed": n_pass, "n_capsules": n_pass + n_fail, "per_capsule": rows}


# --- DEFECT 1: a non-all-pass round must still reach the cert checkpoint --------------------------


def test_a_non_all_pass_round_reaches_the_checkpoint_for_its_passing_capsules():
    """19/32 is the measured case: the checkpoint MUST run, and exactly the 19 go to it."""
    from merlin_experiments.phase1 import authoring as loop

    v = _round_verdict(19, 13)
    assert v["all_pass"] is False, "the fixture must be the measured non-all-pass round"
    eligible, held = loop.CERT._l3_promotion(v, "L2")
    assert len(eligible) == 19 and len(held) == 13
    assert loop.CERT._l3_checkpoint_should_run(True, True, eligible) is True
    # ... and each held-back capsule carries the REASON it was not submitted.
    assert all(h.get("reason") for h in held)
    assert all(h["loop_tier_status"] == "fail" for h in held)


def test_one_passing_capsule_is_enough_and_zero_is_not():
    """The gate is 'at least one', not 'all' — and with nothing eligible there is nothing to certify."""
    from merlin_experiments.phase1 import authoring as loop

    one, _ = loop.CERT._l3_promotion(_round_verdict(1, 31), "L2")
    assert loop.CERT._l3_checkpoint_should_run(True, True, one) is True
    none, held = loop.CERT._l3_promotion(_round_verdict(0, 32), "L2")
    assert none == [] and len(held) == 32
    assert loop.CERT._l3_checkpoint_should_run(True, True, none) is False


def test_the_gate_keeps_its_other_preconditions():
    """Per-capsule promotion widens WHICH capsules are certified; it does not remove the guards."""
    from merlin_experiments.phase1 import authoring as loop

    eligible, _ = loop.CERT._l3_promotion(_round_verdict(19, 13), "L2")
    assert loop.CERT._l3_checkpoint_should_run(False, True, eligible) is False  # cert tier not mandatory
    assert loop.CERT._l3_checkpoint_should_run(True, False, eligible) is False  # workflow not conformant


def test_the_checkpoint_gate_no_longer_demands_all_pass():
    """Guard the call site itself, so nobody restores the all-or-nothing condition.

    Pin the PER-CAPSULE shape only -- the cert-tier flag and the eligible list -- and deliberately not
    which conformance view is passed as the middle argument. Spelling `workflow_conformant` here made
    this test contradict `test_run_level_conformance.py`, which requires the run-level view, and that
    contradiction is how the run-level fix was lost: `4e5ab2df` wired the barrier to
    `_conformant_over_run()`, `622ac429` rebuilt the barrier from a base without it, and this assertion
    then held the per-round spelling in place while the other test failed. Two tests pinning
    incompatible spellings of one line cannot both be satisfied; only one of them was about this
    behaviour.
    """
    from merlin_experiments.phase1 import authoring as loop

    src = inspect.getsource(loop.execute)
    assert "_l3_checkpoint_should_run(_run_l3, " in src, "the cert-tier flag stays the first guard"
    assert ", _l3_eligible)" in src, "and the gate is handed the per-capsule eligible list"
    assert 'verdict.get("all_pass") or (_EXPERIMENT == "realistic" and _ready_marker)' not in src
    assert "_verilator_grade(vatt, _l3_eligible, _l3_held)" in src


def test_an_unreadable_loop_tier_status_is_held_back_not_promoted():
    """No loop-tier verdict is UNKNOWN, and unknown is not a pass."""
    from merlin_experiments.phase1 import authoring as loop

    eligible, held = loop.CERT._l3_promotion(
        {"per_capsule": [{"capsule": "X", "tiers": {}}, {"capsule": "", "status": "pass"}]}, "L2"
    )
    assert eligible == []
    assert [h["capsule"] for h in held] == ["X", ""]


# --- DEFECT 1, the other half: a not-promoted capsule is NEVER a cert pass ------------------------


def test_a_not_promoted_capsule_is_never_counted_as_a_cert_pass():
    """It is neither dropped from the denominator nor credited as a pass."""
    from merlin_experiments.phase1 import authoring as loop

    v = _round_verdict(19, 13)
    eligible, held = loop.CERT._l3_promotion(v, "L2")
    # every promoted capsule certifies; the 13 held back never ran the cert tier at all
    red = {name: {"tiers": {"L3": "pass"}} for name in eligible}
    rows, npass, ncaps = loop.CERT._l3_attempt_tally(red, held, "L3")
    assert npass == 19, "only the capsules that actually certified count"
    assert ncaps == 32, "the held-back capsules stay in the denominator; they are not dropped"
    held_rows = [r for r in rows if r["capsule"].startswith("F")]
    assert len(held_rows) == 13
    assert {r["l3_status"] for r in held_rows} == {"not_promoted"}
    assert all(r.get("reason") for r in held_rows)
    assert "pass" not in {r["l3_status"] for r in held_rows}
    # and therefore the attempt cannot read as certified
    assert (npass == ncaps and ncaps > 0) is False


def test_a_cert_tier_that_did_not_report_is_not_a_pass():
    """A submitted capsule whose cert tier produced nothing (timeout, no oracle) is UNKNOWN, not pass."""
    from merlin_experiments.phase1 import authoring as loop

    rows, npass, ncaps = loop.CERT._l3_attempt_tally(
        {"A": {"tiers": {"L3": "pass"}}, "B": {"tiers": {}}, "C": {"tiers": {"L3": "fail"}}}, [], "L3"
    )
    assert (npass, ncaps) == (1, 3)
    assert [r["l3_status"] for r in rows] == ["pass", None, "fail"]


def test_the_full_corpus_cert_pass_is_still_the_only_success():
    """Promoting a subset must not make a partial run look certified — all_pass is over the WHOLE set."""
    from merlin_experiments.phase1 import authoring as loop

    eligible, held = loop.CERT._l3_promotion(_round_verdict(19, 13), "L2")
    _, npass, ncaps = loop.CERT._l3_attempt_tally({n: {"tiers": {"L3": "pass"}} for n in eligible}, held, "L3")
    assert loop.CERT._l3_barrier_decision(npass == ncaps and ncaps > 0, rnd=1, max_rounds=12) == "iterate"
    # only when the corpus is whole does the barrier terminate
    all_eligible, no_held = loop.CERT._l3_promotion(_round_verdict(32, 0), "L2")
    _, npass2, ncaps2 = loop.CERT._l3_attempt_tally({n: {"tiers": {"L3": "pass"}} for n in all_eligible}, no_held, "L3")
    assert (npass2, ncaps2) == (32, 32)
    assert loop.CERT._l3_barrier_decision(npass2 == ncaps2, rnd=1, max_rounds=12) == "done"


def test_the_fix_verdict_names_the_capsules_it_did_not_certify():
    """A fix round must be told WHY a capsule has no cert result, not left to read absence as success."""
    from merlin_experiments.phase1 import authoring as loop

    eligible, held = loop.CERT._l3_promotion(_round_verdict(19, 13), "L2")
    rows, npass, ncaps = loop.CERT._l3_attempt_tally({n: {"tiers": {"L3": "pass"}} for n in eligible}, held, "L3")
    fix = loop.CERT._l3_fix_verdict({"n_passed": npass, "n_capsules": ncaps, "per_capsule": rows}, 1)
    assert fix["all_pass"] is False
    assert fix["n_not_promoted"] == 13
    assert "not a pass" in fix["not_promoted_note"]


def test_completion_predicates_stay_whole_corpus():
    """`_authoring_completion`/`_formal_completion` are unchanged by per-capsule promotion."""
    from merlin_experiments.phase1 import authoring as loop

    assert loop.CERT._authoring_completion(True, True) is True
    assert loop.CERT._authoring_completion(False, True) is False
    assert loop.CERT._formal_completion(True, True, True, True) is True
    assert loop.CERT._formal_completion(True, True, True, False) is False
    assert loop.CERT._formal_completion(True, True, False, True) is False


def test_the_official_grade_still_refuses_a_partial_phase(tmp_path):
    """The post-freeze grade must reject a phase that did not pass every capsule."""
    from merlin_experiments.phase1 import authoring as loop

    phase = {
        "n_capsules": 32,
        "n_passed": 19,
        "formal_complete": True,
        "gradeable": True,
        "integrity_status": "clean",
        "numeric_all_exact": True,
        "trace_all_pass": True,
        "unmeasured_counts": {},
    }
    (tmp_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "completion": {"formal_grade_complete": True, "required_tier": "L3"},
                "public_dev": dict(phase),
                "hidden": dict(phase),
            }
        )
    )
    res = loop.CERT._official_grade_result(0, tmp_path)
    assert res.get("complete") is not True
    fails = " ".join(res.get("failures") or [])
    assert "public_dev:not_all_capsules_passed" in fails
    assert "hidden:not_all_capsules_passed" in fails


# --- DEFECT 2: a verdict must land inside the turn, and must not overclaim ------------------------


def test_a_withheld_mandatory_tier_reads_as_unknown_never_as_a_pass():
    """all_pass is null (UNKNOWN) while a mandatory cert tier has not run — never true."""
    __import__("merlin_experiments.phase1.authoring", fromlist=[""])
    red = {f"P{i}": {"status": "pass", "tiers": {"L2": "pass"}} for i in range(32)}
    doc = LG._fast_loop_verdict_doc(red, ["L2"], ["L3"])
    assert doc["all_pass"] is None, "a tier that did not run may not read as success"
    assert doc["tiers_not_run"] == ["L3"]
    assert doc["mandatory_tiers_complete"] is False
    assert doc["n_passed"] == 32 and doc["n_capsules"] == 32
    assert "UNKNOWN, not passed" in doc["note"]


def test_the_fast_verdict_may_conclude_only_when_nothing_was_withheld():
    __import__("merlin_experiments.phase1.authoring", fromlist=[""])
    red = {f"P{i}": {"status": "pass", "tiers": {"L2": "pass"}} for i in range(3)}
    assert LG._fast_loop_verdict_doc(red, ["L2"], [])["all_pass"] is True
    red["P0"]["status"] = "fail"
    assert LG._fast_loop_verdict_doc(red, ["L2"], [])["all_pass"] is False
    # an empty grade is not a pass either
    assert LG._fast_loop_verdict_doc({}, ["L2"], [])["all_pass"] is False


def test_the_fast_verdict_is_falsy_to_every_gate_while_incomplete():
    """`None` must not accidentally satisfy a `verdict.get("all_pass")` style gate."""
    from merlin_experiments.phase1 import authoring as loop

    doc = LG._fast_loop_verdict_doc({"P0": {"status": "pass"}}, ["L2"], ["L3"])
    assert not doc["all_pass"]
    assert loop.CERT._authoring_completion(bool(doc["all_pass"]), True) is False


def test_every_agent_turn_starts_a_grader_that_lands_a_verdict_under_it():
    """The measured failure was a 6184s turn with no qa/ at all. Pin the wiring at the call site."""
    from merlin_experiments.phase1 import authoring as loop

    src = inspect.getsource(loop.execute)
    import ast

    def calls(text, name):
        return [
            node
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == name
        ]

    def shape(node):
        return ast.dump(node, include_attributes=False)

    starts = calls(src, "start_background")
    assert starts
    assert any(
        [shape(arg) for arg in call.args[:2]]
        == [shape(ast.Name(id=name, ctx=ast.Load())) for name in ("ws", "run_dir")]
        and shape(call.args[2])
        == shape(ast.parse("FL.GradeCadence(a.qa_timeout, a.no_oracle, a.grade_interval)", mode="eval").body)
        and any(
            keyword.arg == "interval_grades"
            and shape(keyword.value) == shape(ast.parse('a.schedule == "continuous"', mode="eval").body)
            for keyword in call.keywords
        )
        for call in starts
    )
    assert any(
        [shape(arg) for arg in call.args] == [shape(ast.Name(id="_bg", ctx=ast.Load()))]
        for call in calls(src, "stop_background")
    )
    # the first grade is the CHEAP one, and the expensive full-ladder grade follows on the interval
    body = inspect.getsource(FL.start_background)
    assert calls(body, "fast_grade_callback")
    assert all(
        any(k.arg == "fast_grade_callback" and ast.unparse(k.value) == "_fast_grade" for k in call.keywords)
        for call in starts
    )
    assert all(
        any(k.arg == "grade_callback" and ast.unparse(k.value) == "_grade" for k in call.keywords) for call in starts
    )
    grades = calls(body, "grade_callback")
    assert any(
        [shape(arg) for arg in call.args[:3]]
        == [shape(ast.parse(value, mode="eval").body) for value in ("ws", "run_dir", "FIRST_BACKGROUND_TICK + t")]
        and any(
            keyword.arg == "label" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "inturn"
            for keyword in call.keywords
        )
        for call in grades
    ), "an in-turn grade is not a round and must not be filed as one"
    assert "cadence.grade_interval" in body


def test_the_first_grade_uses_the_loop_ladder_not_the_checkpoint_ladder():
    """Cheap by construction, and target-agnostic: the tier comes from qa_loop_adapters, not a sim name."""
    __import__("merlin_experiments.phase1.authoring", fromlist=[""])
    body = inspect.getsource(LG.fast_grade)
    assert "qa_loop_adapters(" in body
    assert "qa_checkpoint_adapters(" not in body
    for sim in ("spike", "verilator", "vcs"):
        assert sim not in body, f"the fast grade must not name a simulator ({sim})"


def test_background_grade_ids_cannot_collide_with_round_ids():
    """A background grade shares qa_grade's work tree; a colliding id would clobber a round's grade."""
    assert len({FL.grade_key(r, t) for r in range(3) for t in range(1, 151)}) == 450


def test_an_existing_verdict_is_not_replaced_by_the_fast_one():
    """Round 1+ already inherits the previous round's verdict; the fast grade is for a BLIND agent."""
    body = inspect.getsource(FL.start_background)
    assert 'if not (ws / "qa" / "verdict.json").exists():' in body
