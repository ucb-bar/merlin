"""`--schedule continuous` must actually grade UNDER the running agent — the property nothing gated.

The run-shape convention promises three properties "gated, not documented-and-hoped-for". Property 1 is
the one that was never gated:

    "No round barrier — a background grader re-grades a *snapshot copy* every --grade-interval seconds
     and refreshes qa/verdict.json, so feedback reaches the agent while it works."

It was false. The background grader lived inside the LEGACY ``--continuous`` branch — the flag the same
convention forbids — while ``--schedule continuous``, the certified path every real run takes, ran the
ROUND loop with its cap lifted and installed no grader at all. The existing gate
(``test_continuous_is_the_default.py``) asserted only that ``--continuous`` stays opt-in and that the
round cap is lifted, so it passed the whole time: it checked the wrong thing.

Measured on the atlas run rb_atlasp1e (7 rounds / 8.58 h) the cost was visible in the score itself —
0, 35, 35, 21, 22, 39, 36 out of 57. It oscillated because every round handed the agent a fresh context
and its accumulated understanding of the corpus went with the old one.

So these tests check the MECHANISM, not the flag:

  * ``keep_launching`` really returns "one session" for the continuous schedule (behavioural, pure);
  * a ``BackgroundGrader`` really grades on its interval and really refreshes the verdict (behavioural,
    with a fake grade function — no agent, no oracle);
  * the driver really constructs one for the certified schedule and really starts it around the agent
    launch, OUTSIDE the legacy branch (structural, over the AST — a comment cannot detect silence);
  * and the post-freeze public+hidden L3 grade is still reachable afterwards, so the fix did not buy
    continuity by quietly adopting the legacy path's "can never report a formal success".
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import time

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
DRIVER = HARNESS / "run_baseline_qa_loop.py"


def _driver():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("run_baseline_qa_loop_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"driver not importable here: {type(exc).__name__}: {exc}")
    return mod


# --- the AST of main(), so "is the grader on the path the run takes?" is answered structurally -------
def _main_ast() -> ast.FunctionDef:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("run_baseline_qa_loop.main() is gone")


def _legacy_branch(main: ast.FunctionDef) -> ast.If:
    """The legacy ``if a.continuous:`` block — everything inside it is the forbidden path."""
    for node in ast.walk(main):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "a.continuous":
            return node
    raise AssertionError("the legacy `if a.continuous:` branch is gone; it must stay, opt-in")


def _agent_loop(main: ast.FunctionDef) -> ast.While:
    """The while-loop that invokes the agent — the certified path, whichever schedule is in force."""
    for node in ast.walk(main):
        if isinstance(node, ast.While) and any(
                isinstance(c, ast.Call) and getattr(c.func, "id", "") == "launch_agent"
                for c in ast.walk(node)):
            return node
    raise AssertionError("the agent-invocation loop is gone")


def _span(node) -> range:
    return range(node.lineno, (node.end_lineno or node.lineno) + 1)


def _grader_names(main: ast.FunctionDef, forbidden: range) -> list:
    """Names bound to a BackgroundGrader(...) OUTSIDE the legacy branch."""
    out = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if getattr(node.value.func, "id", "") != "BackgroundGrader":
            continue
        if node.lineno in forbidden:
            continue
        out += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return out


# --- 1. the policy: continuous is ONE session, not a sequence of rounds -----------------------------
def test_the_continuous_schedule_launches_one_session_not_rounds():
    """One long agent session per process, so the context is never thrown away at a round boundary."""
    M = _driver()
    kl = M.keep_launching
    base = dict(rnd=0, max_rounds=12, active_wall_s=0.0, max_wall_s=0, authoring_complete=False)

    go, why = kl(schedule="continuous", sessions_launched=0, **base)
    assert go and why == "", "the continuous schedule refuses to launch its one session"

    go, why = kl(schedule="continuous", sessions_launched=1, **{**base, "rnd": 1})
    assert not go and why == "session_complete", (
        "the continuous schedule relaunched the agent — that is a round barrier, and a fresh context "
        "each time is exactly the progress loss measured on rb_atlasp1e")


def test_the_continuous_schedule_honours_the_total_wall_budget():
    """`--max-wall-s` is the TOTAL active agent budget (the '12h' an operator declares)."""
    M = _driver()
    go, why = M.keep_launching(schedule="continuous", sessions_launched=0, rnd=0, max_rounds=12,
                               active_wall_s=43_201.0, max_wall_s=43_200, authoring_complete=False)
    assert not go and why == "max_wall_s"


def test_the_rounds_schedule_is_unchanged():
    """The default schedule keeps its historical --max-rounds bound, byte-for-byte in behaviour."""
    M = _driver()
    kl = M.keep_launching
    for rnd, expect in ((0, True), (11, True), (12, False)):
        go, _ = kl(schedule="rounds", sessions_launched=rnd, rnd=rnd, max_rounds=12,
                   active_wall_s=0.0, max_wall_s=0, authoring_complete=False)
        assert go is expect, f"rounds schedule changed at rnd={rnd}"
    assert kl(schedule="rounds", sessions_launched=0, rnd=0, max_rounds=12, active_wall_s=0.0,
              max_wall_s=0, authoring_complete=True)[0] is False


# --- 2. the mechanism: it really grades, on an interval, and really refreshes the verdict ------------
def test_the_background_grader_regrades_on_its_interval_while_the_agent_runs():
    """Behavioural, no agent and no oracle: a fake grade fn stands in for qa_grade.

    The grader must (a) fire repeatedly on its interval rather than once, (b) hand each verdict to the
    post-grade hook (that hook is what refreshes the checkpoint and sinks telemetry mid-run), and (c)
    keep running while the "session" is still going.
    """
    M = _driver()
    seen, ticks = [], []

    def grade(tick):
        ticks.append(tick)
        return {"all_pass": False, "n_passed": len(ticks), "n_capsules": 20}

    g = M.BackgroundGrader(grade, interval=0, min_interval=0.01, tick_base=M.BG_GRADE_TICK_BASE,
                           on_grade=lambda t, v: seen.append((t, v)))
    g.start()
    assert g.running(), "the grader thread did not start"
    deadline = time.time() + 5.0
    while len(ticks) < 3 and time.time() < deadline:
        time.sleep(0.02)
    g.stop()

    assert len(ticks) >= 3, f"the grader graded {len(ticks)} time(s); it must re-grade on its interval"
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks), (
        f"background grade ticks must be distinct and increasing so one grade cannot clobber "
        f"another's scratch dir: {ticks}")
    assert min(ticks) >= M.BG_GRADE_TICK_BASE, (
        "background grades must be numbered in their own band, or a snapshot grade reuses (and deletes) "
        "a round grade's _qa_work/cand_NN")
    assert [t for t, _ in seen] == ticks, "the post-grade hook did not see every background grade"
    assert not g.running(), "the grader outlived the session"


def test_a_grade_of_a_half_written_submission_never_kills_the_run():
    """The agent is editing while this runs, so a failed grade must be skipped and retried."""
    M = _driver()
    calls = []

    def grade(tick):
        calls.append(tick)
        if len(calls) == 1:
            raise RuntimeError("submission half-written")
        return {"all_pass": False, "n_passed": 0, "n_capsules": 20}

    g = M.BackgroundGrader(grade, interval=0, min_interval=0.01)
    g.start()
    deadline = time.time() + 5.0
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.02)
    g.stop()
    assert len(calls) >= 3, "a raising grade stopped the background grader"
    assert g.grades >= 1, "no grade completed after the raising one"


def test_the_grader_stops_when_the_session_does():
    """It must not survive the agent: the L3 barrier writes its own verdict.json and cannot be raced."""
    M = _driver()
    g = M.BackgroundGrader(lambda t: {"all_pass": False}, interval=0, min_interval=0.01)
    g.start()
    g.stop(timeout=5)
    assert not g.running()
    assert not any(t.name.endswith("-grader") and t.is_alive() for t in threading.enumerate())


# --- 3. the wiring: the CERTIFIED schedule is the one that gets it ----------------------------------
def test_the_certified_schedule_installs_the_grader_around_the_agent_session():
    """The defect this file exists for: the grader was constructed only inside `if a.continuous:`.

    Structural, because the alternative is a 12-hour paid run. The grader must be built OUTSIDE the
    legacy branch and started/stopped around the agent launch in the loop the certified path takes.
    """
    main = _main_ast()
    legacy = _span(_legacy_branch(main))
    loop = _agent_loop(main)

    assert loop.lineno not in legacy, "the agent loop is now inside the legacy branch"
    names = _grader_names(main, legacy)
    assert names, (
        "no BackgroundGrader is constructed outside the legacy `if a.continuous:` branch, so "
        "`--schedule continuous` still runs the round loop with no grader under the agent")

    started = {ast.unparse(n.func).rsplit(".", 1)[0]
               for n in ast.walk(loop)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "start"}
    stopped = {ast.unparse(n.func).rsplit(".", 1)[0]
               for n in ast.walk(loop)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "stop"}
    assert started & set(names), (
        f"the agent loop never starts a background grader (starts: {sorted(started)}); feedback would "
        f"still only reach the agent by ending its session")
    assert stopped & set(names), (
        "the grader is never stopped around the agent launch; it would race the authoritative grade and "
        "the L3 barrier's own verdict.json")


def test_the_single_session_is_bounded_by_the_total_budget_not_only_by_the_round_timeout():
    """`--max-wall-s` must reach the launch, or a '12h total' run is really a 12h-per-session run."""
    main = _main_ast()
    loop = _agent_loop(main)
    launch = next(n for n in ast.walk(loop)
                  if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "launch_agent")
    args = {ast.unparse(x) for x in launch.args} | {ast.unparse(k.value) for k in launch.keywords}
    assert "a.round_timeout" not in args, (
        "the agent session is bounded by --round-timeout alone, so --max-wall-s cannot cap the run")
    assert any("wall" in x for x in args), (
        f"no wall-budget-aware cap is passed to launch_agent: {sorted(args)}")


# --- 4. and the formal grade survives ---------------------------------------------------------------
def test_the_certified_schedule_still_reaches_the_post_freeze_public_hidden_grade():
    """Continuity must not be bought by adopting the legacy path's progress-only exit.

    The legacy branch returns 1 and hardcodes formal_complete=False. The certified schedule must fall
    THROUGH the loop into the L3 barrier, the finalize turn and grade_agent_run.py, or a formal success
    stops being reachable for every real run.
    """
    main = _main_ast()
    legacy = _legacy_branch(main)
    legacy_lines = _span(legacy)
    loop_end = _agent_loop(main).end_lineno

    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value == 1
               for n in ast.walk(legacy)), (
        "the legacy branch no longer owns the progress-only `return 1`; if that exit escaped into the "
        "certified path, no run could report a formal success")

    for needle in ("grade_agent_run.py", "_formal_completion", "_verilator_grade"):
        hits = [n.lineno for n in ast.walk(main)
                if isinstance(n, ast.Constant) and n.value == needle
                or isinstance(n, ast.Name) and n.id == needle
                or isinstance(n, ast.FunctionDef) and n.name == needle]
        assert hits, f"{needle} is gone from main(); the post-freeze grade is not reachable"
        assert any(h not in legacy_lines for h in hits), f"{needle} is only inside the legacy branch"
        if needle in ("grade_agent_run.py", "_formal_completion"):
            assert any(h > loop_end for h in hits), (
                f"{needle} no longer runs after the agent loop; the certified continuous path would "
                f"end without its public+hidden record")
