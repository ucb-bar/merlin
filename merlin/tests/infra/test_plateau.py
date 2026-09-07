"""Progress across grades: it must be able to FIRE, and it must not fire on a productive run.

The detector this replaces could not fire at all. It counted consecutive ROUNDS with no progress, and
the default schedule is `continuous` -- one long session, so exactly one round -- putting its threshold
out of reach in the mode every real run uses, and it was opt-in besides. Measured on
merlincirct_g4p1_biasabi_20260906: 92/96 reached at 2.19 h, then 3.91 h more (64% of the run) with the
score never moving and nothing saying so. So the unit here is a GRADE, and the tests below pin both
directions: it fires on that history, and it stays quiet on a run that is still improving.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import plateau as PL


def _g(passed, statuses, n=4, mismatch=None):
    rows = [{"capsule": name, "status": st,
             **({"mismatch_count": mismatch[name]} if mismatch and name in mismatch else {})}
            for name, st in statuses.items()]
    return {"n_passed": passed, "n_capsules": n, "per_capsule": rows}


def _flat(n_grades, passed=92, n=96):
    st = {"A": "pass", "B": "fail"}
    return [_g(passed, st, n) for _ in range(n_grades)]


# --------------------------------------------------------------------------------------------
# it can fire
# --------------------------------------------------------------------------------------------

def test_a_flat_run_is_eventually_called_stuck():
    got = PL.assess(_flat(6), stall_threshold=4)
    assert got.stuck and got.stalled_grades >= 4 and got.reason


def test_it_fires_on_grades_not_on_rounds():
    """The whole defect: a round-counting threshold is unreachable under the continuous schedule,
    which produces one round and many grades."""
    got = PL.assess(_flat(8), stall_threshold=4)
    assert got.n_grades == 8 and got.stuck


# --------------------------------------------------------------------------------------------
# it stays quiet when it should
# --------------------------------------------------------------------------------------------

def test_an_improving_run_is_never_stuck():
    grades = [_g(p, {"A": "pass", "B": "fail"}) for p in (10, 40, 70, 82, 92)]
    assert PL.assess(grades, stall_threshold=2).stuck is False


def test_a_run_reducing_numeric_error_is_not_stuck_though_its_pass_count_is_flat():
    """The shape of real progress on a hard capsule. A pass-count-only detector cuts these runs, and
    an operator who is burned once switches the detector off for good."""
    grades = [_g(50, {"A": "pass", "B": "fail"}, mismatch={"B": m}) for m in (900, 700, 400, 100, 10)]
    got = PL.assess(grades, stall_threshold=2)
    assert got.stuck is False, got.reason


def test_a_structural_failure_does_not_read_as_solved():
    """No mismatch to count is not zero error -- treating it as zero makes a structural stall look
    like a run whose residual has reached the floor."""
    key = PL.progress_key(_g(1, {"A": "pass", "B": "fail"}))
    assert key[1] <= -PL._STRUCTURAL_RESIDUAL


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_little_history_is_never_a_plateau(n):
    got = PL.assess(_flat(n))
    assert got.stuck is False and str(PL.MIN_GRADES) in got.reason or n == 0


def test_no_capsule_count_is_unmeasurable_not_stuck():
    got = PL.assess([{"n_passed": 0, "per_capsule": []}] * 5)
    assert got.stuck is False and "cannot be measured" in got.reason


def test_garbage_is_survived():
    assert PL.assess(None).stuck is False
    assert PL.assess([None, 3, "x"]).stuck is False
    assert PL.assess([{}] * 5).stuck is False


# --------------------------------------------------------------------------------------------
# reachability, which is the stronger fact
# --------------------------------------------------------------------------------------------

def test_capsules_that_never_passed_are_named():
    grades = [_g(1, {"A": "pass", "B": "fail", "C": "fail"}) for _ in range(5)]
    got = PL.assess(grades)
    assert got.never_passed == ("B", "C")
    assert "NEVER passed" in got.sentence()


def test_a_capsule_that_passed_once_is_not_called_never_passed():
    """The paired direction: it distinguishes unreachable from merely-not-passing-now."""
    grades = [_g(2, {"A": "pass", "B": "pass"}), _g(1, {"A": "pass", "B": "fail"}),
              _g(1, {"A": "pass", "B": "fail"}), _g(1, {"A": "pass", "B": "fail"})]
    got = PL.assess(grades)
    assert got.never_passed == () and got.regressed == ("B",)
    assert "regression" in got.sentence()


def test_both_verdict_shapes_are_understood():
    """A mapping of name->status and a list of rows both occur in this repo's verdicts; a reader that
    knows only one sees an empty suite and reports nothing failing."""
    as_map = {"n_passed": 1, "n_capsules": 2, "per_capsule": {"A": "pass", "B": "fail"}}
    as_rows = _g(1, {"A": "pass", "B": "fail"}, n=2)
    assert PL._statuses(as_map) == PL._statuses(as_rows) == {"A": True, "B": False}


def test_stuck_always_carries_a_reason():
    for grades in (_flat(6), _flat(2), [], [{}] * 5):
        got = PL.assess(grades)
        assert got.reason, "every verdict must explain itself"
        if got.stuck:
            assert got.stalled_grades >= 1


# --------------------------------------------------------------------------------------------
# it must not become agent-visible feedback
# --------------------------------------------------------------------------------------------

def test_the_assessment_is_recorded_operator_side_only():
    """Naming the capsules that have never passed is FEEDBACK, and feedback defines an arm. Handing it
    to the agent would change the treatment and make the run incomparable with every earlier one."""
    from merlin.common.paths import merlin_dir
    loop = (merlin_dir() / "experiments" / "capsule_bench" / "harness"
            / "run_baseline_qa_loop.py").read_text()
    i = loop.index("def _record_plateau")
    body = loop[i:i + 2600]
    assert 'run_dir / "plateau.json"' in body, "the assessment must land in the run dir"
    assert 'ws / "qa"' not in body, "it must never be written into the agent's workspace"
    assert "feedback would change the arm" in body or "change the arm's treatment" in body
    # and it must actually be called where grades land
    assert loop.count("_record_plateau(run_dir)") >= 2, (
        "both operator-side grade paths must record it, or a continuous run reports nothing")
