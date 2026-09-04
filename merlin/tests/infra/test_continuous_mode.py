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


def test_continuous_runs_one_session_and_grades_on_an_interval():
    """The interval grader runs ALONGSIDE the session, not after it.

    It used to be an inline ``threading.Thread`` here; it is now the shared ``BackgroundGrader`` (which
    still owns the thread) so that ``--schedule continuous`` -- the CERTIFIED path -- can install the very
    same mechanism instead of silently having none. See
    ``test_continuous_schedule_grades_under_the_agent.py`` for the gate on that path.
    """
    br = _continuous_branch()
    calls = _calls(br)
    assert "BackgroundGrader" in calls and "start" in calls, (
        "the interval grader must run alongside the session, not after it")
    assert "Thread" in _calls(_fn("start")), (
        "BackgroundGrader.start no longer starts a thread, so nothing grades while the agent works")
    assert "qa_grade" in calls, "continuous mode must grade through the same path a round does"
    assert "launch_agent" in calls, "continuous mode still runs one real agent session"


def test_a_failed_grade_cannot_kill_the_run():
    """A grade racing a mid-write submission is expected; it must degrade to a skipped tick."""
    loop = _fn("_loop")   # BackgroundGrader._loop -- the interval grader itself
    handlers = [h for h in ast.walk(loop) if isinstance(h, ast.ExceptHandler)]
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
