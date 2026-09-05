"""CONTINUOUS mode: one long-lived agent session, graded on an interval instead of at a round barrier.

The round loop conflates two things: giving the agent a fresh context, and delivering it a graded
verdict. Only the first needs a relaunch. ``qa_grade`` copies the workspace submission to an
operator-only scratch and grades THAT, writing the redacted verdict to ``ws/qa/verdict.json`` — so it is
safe to run while the agent is still working, and the agent's feedback can refresh continuously.

These tests pin the properties that make continuous mode sound, structurally (via ``ast``) rather than by
launching an agent:

  * the snapshot property — ``qa_grade`` grades a COPY, never the live workspace. Without it, a grade
    racing a half-written submission would grade whatever was on disk mid-write;
  * the grader must not be able to kill the run — a failed grade is a skipped tick, not an exception out
    of the thread;
  * a FINAL authoritative grade after the session, because the interval grades are progress reports on a
    moving workspace and the run's verdict must describe the submission as the session left it;
  * per-capsule tier promotion is NOT part of this change and must stay immediate.
"""
from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir

_LOOP = merlin_dir() / "experiments/capsule_bench/harness/run_baseline_qa_loop.py"


def _tree() -> ast.Module:
    return ast.parse(_LOOP.read_text(encoding="utf-8"))


def _fn(name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(_tree())
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node: ast.AST) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_qa_grade_grades_a_snapshot_not_the_live_workspace():
    """The load-bearing property: background grading is only safe because the grade is of a COPY."""
    src = ast.get_source_segment(_LOOP.read_text(encoding="utf-8"), _fn("qa_grade")) or ""
    assert "copytree" in src, (
        "qa_grade must copy the submission before grading — continuous mode grades while the agent is "
        "still writing, and grading the live workspace would grade a half-written tree")
    assert "verdict.json" in src, "qa_grade must publish the redacted verdict the agent reads"


def test_continuous_flags_are_exposed():
    names = set()
    for call in (c for c in ast.walk(_fn("main")) if isinstance(c, ast.Call)):
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr == "add_argument" and call.args:
            a0 = call.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                names.add(a0.value)
    assert "--continuous" in names, "continuous mode must be selectable"
    assert "--grade-interval" in names, "the grading cadence must be settable"


def _continuous_branch() -> ast.If:
    """The `if a.continuous:` branch in main()."""
    for node in ast.walk(_fn("main")):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) \
                and node.test.attr == "continuous":
            return node
    raise AssertionError("no `if a.continuous:` branch in main()")


def _grader_fn() -> ast.FunctionDef:
    """The interval grader inside the continuous branch, found by what it DOES.

    ⚠️ THIS USED TO LOOK FOR A CLASS NAMED ``BackgroundGrader``, AND THAT CLASS HAS NEVER EXISTED.
    `git log -S` finds no commit that added it anywhere in the tree; the grader has always been a
    thread started inside ``main()``. So this gate -- one of the three properties the run shape is
    supposed to GUARANTEE, "no round barrier" -- could not fail for a reason that had nothing to do
    with the property. Ten tests across two files asserted the same absent name.

    Finding it structurally instead: it is the nested function, inside the continuous branch, whose
    body calls ``qa_grade`` in a loop. That is the grader by definition rather than by name, so a
    rename cannot blind this again.
    """
    br = _continuous_branch()
    for node in ast.walk(br):
        if isinstance(node, ast.FunctionDef) and "qa_grade" in _calls(node):
            if any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(node)):
                return node
    raise AssertionError(
        "no nested function in the continuous branch loops over qa_grade — nothing grades while the "
        "agent works, so the run has a round barrier in all but name")


def test_continuous_runs_one_session_and_grades_on_an_interval():
    """The interval grader runs ALONGSIDE the session, not after it.

    This is the first of the three gated run-shape properties: a background grader re-grades on an
    interval and refreshes the verdict, so feedback reaches the agent WHILE it works rather than at a
    round boundary.
    """
    br = _continuous_branch()
    calls = _calls(br)

    grader = _grader_fn()                      # raises if nothing grades in a loop
    # The grader must run on its own thread, or "alongside the session" is false — it would grade
    # before or after, which is the barrier this mode exists to remove.
    assert "Thread" in calls, (
        "the interval grader is not started on a thread, so it cannot run alongside the agent session")
    started_names = {n.func.attr for n in ast.walk(br)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "start" in started_names, "a grader thread is constructed but never started"

    # It must WAIT on an interval rather than spin, and the wait must be interruptible so the session
    # teardown below can stop it.
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "wait"
               for n in ast.walk(grader)), (
        "the grader does not wait on a stop event — it cannot be an INTERVAL grader that also stops "
        "when the session does")

    assert "qa_grade" in calls, "continuous mode must grade through the same path a round does"
    assert "launch_agent" in calls, "continuous mode still runs one real agent session"


def test_a_failed_grade_cannot_kill_the_run():
    """A grade racing a mid-write submission is expected; it must degrade to a skipped tick.

    The grader reads a workspace the agent is actively writing, so a torn read is normal traffic, not
    an incident. If one raises out of the thread the run keeps going with a grader that is silently
    dead — which looks exactly like a run whose agent stopped improving.
    """
    grader = _grader_fn()
    handlers = [h for h in ast.walk(grader) if isinstance(h, ast.ExceptHandler)]
    assert handlers, "the interval grader must guard qa_grade — a mid-write submission is normal"
    assert any(isinstance(n, ast.Continue) for h in handlers for n in ast.walk(h)), (
        "a failed grade must be a SKIPPED TICK: the grader has to keep grading after one raises")


def test_there_is_a_final_authoritative_grade_after_the_session():
    """Interval grades describe a moving workspace; the run's verdict must be of the final submission."""
    br = _continuous_branch()
    # the last qa_grade in the branch must sit OUTSIDE the grader function definition
    inner = {id(n) for fn in ast.walk(br) if isinstance(fn, ast.FunctionDef) for n in ast.walk(fn)}
    outer_grades = [n for n in ast.walk(br)
                    if isinstance(n, ast.Call) and id(n) not in inner
                    and isinstance(n.func, ast.Name) and n.func.id == "qa_grade"]
    assert outer_grades, (
        "continuous mode must re-grade after the session ends; without it the run's verdict is whatever "
        "the last interval tick happened to catch")


def test_per_capsule_tier_promotion_is_untouched_by_this_mode():
    """L2 -> L3 promotion is per capsule inside the runner's ladder and must not become a mode switch:
    a capsule that clears the screen goes on to the certifying tier in the SAME grade."""
    runner = (merlin_dir() / "python/merlin/targetgen/capsule_runner.py").read_text(encoding="utf-8")
    assert "for tier in _tier_seq:" in runner, (
        "the per-capsule tier ladder moved — promotion must stay immediate and per capsule, never "
        "deferred to a round or a mode")
