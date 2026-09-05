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

⚠️ THESE TESTS ONCE ASSERTED AGAINST A CLASS NAMED ``BackgroundGrader`` AND A FUNCTION
``keep_launching``. Both existed briefly (62343e5c) and were lost in a merge; eight of the nine tests
here then failed for a reason that had nothing to do with the property, and before that they had passed
by asserting on a name rather than on behaviour. So every gate below finds the grader and the stop
policy STRUCTURALLY — by what they DO — and drives the real functions:

  * ``_keep_going``, main()'s real stop policy, is COMPILED OUT OF main()'s OWN SOURCE and called, so
    the schedule policy is exercised as written rather than paraphrased;
  * ``_start_in_turn_grader`` / ``_stop_in_turn_grader`` are driven for real, with ``qa_grade`` and
    ``_fast_loop_verdict`` stubbed — no agent, no oracle;
  * the wiring (is the grader on the path the certified run takes?) stays structural over the AST,
    because the alternative is a 12-hour paid run;
  * and the post-freeze public+hidden L3 grade must still be reachable afterwards, so continuity is not
    bought by quietly adopting the legacy path's "can never report a formal success".
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
import threading
import time
import types

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


def _stop_policy():
    """Compile main()'s OWN ``_keep_going`` closure into a callable, and return a factory for it.

    The stop policy is a nested closure over ``a``, ``rnd``, ``active_wall_s`` and
    ``_authoring_complete``; it cannot be imported. Rather than restate it in the test (the mistake that
    let ten tests assert on a name that did not exist), lift the real source out of main() and bind those
    four names as parameters. What runs below is the policy as WRITTEN — a paraphrase cannot drift from
    it, because there is no paraphrase.
    """
    main = _main_ast()
    fn = next((n for n in ast.walk(main)
               if isinstance(n, ast.FunctionDef) and n.name == "_keep_going"), None)
    assert fn is not None, (
        "main() no longer has a `_keep_going` stop policy; the loop's terminator is unlocatable, so "
        "neither the round cap nor the wall budget can be gated")
    src = ("def _factory(a, rnd, active_wall_s, _authoring_complete):\n"
           + textwrap.indent(ast.unparse(fn), "    ")
           + "\n    return _keep_going\n")
    ns: dict = {}
    exec(compile(src, "<_keep_going lifted from main()>", "exec"), ns)  # noqa: S102 -- our own source

    def go(*, schedule, rnd=0, max_rounds=12, active_wall_s=0.0, max_wall_s=0,
           authoring_complete=False) -> bool:
        a = types.SimpleNamespace(schedule=schedule, max_rounds=max_rounds, max_wall_s=max_wall_s)
        return ns["_factory"](a, rnd, active_wall_s, lambda: authoring_complete)()

    return go


def _ws(tmp_path, *, with_verdict: bool):
    """A workspace + run dir shaped like the ones the grader reads."""
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    if with_verdict:
        (ws / "qa").mkdir()
        (ws / "qa" / "verdict.json").write_text("{}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return ws, run_dir


def _args(*, grade_interval: int, qa_timeout: int = 60):
    return types.SimpleNamespace(grade_interval=grade_interval, qa_timeout=qa_timeout,
                                 no_oracle=False)


def _shrink_waits(monkeypatch, cap: float = 0.01) -> None:
    """Make every ``Event.wait`` return promptly, so an interval grader can be observed in a unit test.

    It shortens the interval and nothing else: the grader still has to WAIT on a stop event, still has
    to loop, and still has to survive a raising grade. (That the wait is genuinely interruptible is the
    separate subject of ``test_the_grader_stops_when_the_session_does``, which does NOT shrink it.)
    """
    real = threading.Event.wait

    def fast(self, timeout=None):
        return real(self, cap if timeout is None else min(timeout, cap))

    monkeypatch.setattr(threading.Event, "wait", fast, raising=True)


def _spin(pred, seconds: float = 5.0) -> None:
    deadline = time.time() + seconds
    while not pred() and time.time() < deadline:
        time.sleep(0.01)


# --- 1. the policy: continuous is ONE continuing session, not a sequence of capped rounds ------------
def test_the_continuous_schedule_launches_one_session_not_rounds():
    """One long agent session per process, so the context is never thrown away at a round boundary."""
    go = _stop_policy()

    assert go(schedule="continuous", rnd=0, max_rounds=12), (
        "the continuous schedule refuses to launch its session")
    assert go(schedule="continuous", rnd=10_000, max_rounds=12), (
        "the continuous schedule stopped on the ROUND COUNT — that is the round barrier back again, "
        "and a fresh context each time is exactly the progress loss measured on rb_atlasp1e; "
        "continuous must stop on EVIDENCE (converged/plateaued) or a declared BUDGET, never on an "
        "arithmetic cap")
    assert not go(schedule="continuous", rnd=0, max_rounds=12, authoring_complete=True), (
        "continuous never stops on evidence either; nothing terminates the run")

    # ...and the invocation it does make is a CONTINUATION, not a fresh session: the schedule is handed
    # to launch_agent, which is what lets a turn-shaped driver (`codex exec`) resume the same thread
    # instead of opening a new context. Without it, "continuous" is a round loop with the cap lifted.
    loop = _agent_loop(_main_ast())
    launch = next(n for n in ast.walk(loop)
                  if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "launch_agent")
    kw = {k.arg: ast.unparse(k.value) for k in launch.keywords}
    assert "continuous" in kw and "schedule" in kw["continuous"], (
        f"launch_agent is not told the run is continuous ({kw}); each invocation would open a fresh "
        f"session and the accumulated context would be discarded every turn")


def test_the_continuous_schedule_honours_the_total_wall_budget():
    """`--max-wall-s` is the TOTAL active agent budget (the '12h' an operator declares)."""
    go = _stop_policy()
    assert not go(schedule="continuous", active_wall_s=43_201.0, max_wall_s=43_200)
    assert go(schedule="continuous", active_wall_s=43_199.0, max_wall_s=43_200), (
        "the wall budget stops the run BEFORE it is spent")
    assert go(schedule="continuous", active_wall_s=10 ** 9, max_wall_s=0), (
        "max_wall_s=0 must mean 'no wall cap', not 'stop immediately'")


def test_the_rounds_schedule_is_unchanged():
    """The default schedule keeps its historical --max-rounds bound, byte-for-byte in behaviour."""
    go = _stop_policy()
    for rnd, expect in ((0, True), (11, True), (12, False)):
        assert go(schedule="rounds", rnd=rnd, max_rounds=12) is expect, \
            f"rounds schedule changed at rnd={rnd}"
    assert go(schedule="rounds", rnd=0, max_rounds=12, authoring_complete=True) is False
    # the wall budget is a CONTINUOUS-only terminator; it must not silently start bounding rounds
    assert go(schedule="rounds", rnd=0, max_rounds=12, active_wall_s=10 ** 9, max_wall_s=1) is True


# --- 2. the mechanism: it really grades, on an interval, and really refreshes the verdict ------------
def test_the_background_grader_regrades_on_its_interval_while_the_agent_runs(tmp_path, monkeypatch):
    """Behavioural, no agent and no oracle: a fake grade fn stands in for qa_grade.

    The grader must (a) fire repeatedly on its interval rather than once, (b) hand each verdict to the
    path that refreshes the agent's ``qa/verdict.json`` (that is ``qa_grade`` itself — the snapshot
    grader whose contract is gated in test_continuous_mode), and (c) keep running while the "session" is
    still going.
    """
    M = _driver()
    ws, run_dir = _ws(tmp_path, with_verdict=True)   # phase 1 already satisfied; this is the interval
    ticks, labels = [], []

    def grade(_ws, _rd, tick, _no_oracle, _timeout, label="round"):
        # NB: recorded, never asserted in here -- the grader catches everything a grade raises (that is
        # the point of the next test), so an assertion in this stub would be swallowed into a skipped
        # tick and the gate would silently pass.
        ticks.append(tick)
        labels.append(label)
        return {"all_pass": False, "n_passed": len(ticks), "n_capsules": 20}

    monkeypatch.setattr(M, "qa_grade", grade)
    _shrink_waits(monkeypatch)

    h = M._start_in_turn_grader(ws, run_dir, _args(grade_interval=900), interval_grades=True)
    th, _stop = h
    assert th.is_alive(), "the grader thread did not start"
    _spin(lambda: len(ticks) >= 3)
    assert th.is_alive(), "the grader ended while the session was still running"
    M._stop_in_turn_grader(h)

    assert len(ticks) >= 3, f"the grader graded {len(ticks)} time(s); it must re-grade on its interval"
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks), (
        f"background grade ticks must be distinct and increasing so one grade cannot clobber "
        f"another's scratch dir: {ticks}")
    assert min(ticks) >= M._BG_TICK_BASE, (
        "background grades must be numbered in their own band, or a snapshot grade reuses (and deletes) "
        "a round grade's _qa_work/cand_NN")
    assert "round" not in labels, (
        f"an in-turn grade is filed under the ROUND archive namespace ({sorted(set(labels))}); every "
        f"per-round trajectory reader globs verdict_round_*.json and would read a mid-turn progress "
        f"report as the round's verdict")
    assert not th.is_alive(), "the grader outlived the session"


def test_the_rounds_schedule_gets_no_interval_grades(tmp_path, monkeypatch):
    """Interval re-grading is the continuous schedule's feature; `rounds` keeps its round barrier.

    The round loop's own post-turn ``qa_grade`` is the authoritative one there, and a second grader
    running underneath it would double the oracle cost of every historical run.
    """
    M = _driver()
    ws, run_dir = _ws(tmp_path, with_verdict=True)
    ticks = []
    monkeypatch.setattr(M, "qa_grade",
                        lambda *a, **k: ticks.append(a[2]) or {"all_pass": False})
    _shrink_waits(monkeypatch)

    h = M._start_in_turn_grader(ws, run_dir, _args(grade_interval=900), interval_grades=False)
    time.sleep(0.3)
    M._stop_in_turn_grader(h)
    assert ticks == [], f"the rounds schedule re-graded under the turn: {ticks}"


def test_a_grade_of_a_half_written_submission_never_kills_the_run(tmp_path, monkeypatch):
    """The agent is editing while this runs, so a failed grade must be skipped and retried."""
    M = _driver()
    ws, run_dir = _ws(tmp_path, with_verdict=True)
    calls, ok = [], []

    def grade(_ws, _rd, tick, _no_oracle, _timeout, label="round"):
        calls.append(tick)
        if len(calls) == 1:
            raise RuntimeError("submission half-written")
        ok.append(tick)
        return {"all_pass": False, "n_passed": 0, "n_capsules": 20}

    monkeypatch.setattr(M, "qa_grade", grade)
    _shrink_waits(monkeypatch)

    h = M._start_in_turn_grader(ws, run_dir, _args(grade_interval=900), interval_grades=True)
    _spin(lambda: len(calls) >= 3)
    alive = h[0].is_alive()
    M._stop_in_turn_grader(h)
    assert alive, "a raising grade killed the grader thread; the run would keep going with a silently "\
                  "dead grader, which looks exactly like an agent that stopped improving"
    assert len(calls) >= 3, "a raising grade stopped the background grader"
    assert ok, "no grade completed after the raising one"


def test_the_grader_stops_when_the_session_does(tmp_path, monkeypatch):
    """It must not survive the agent: the L3 barrier writes its own verdict.json and cannot be raced."""
    M = _driver()
    ws, run_dir = _ws(tmp_path, with_verdict=True)
    monkeypatch.setattr(M, "qa_grade", lambda *a, **k: {"all_pass": False})

    # NOTE: waits are NOT shrunk here. A grader that slept its interval instead of waiting on an
    # interruptible stop event would pass every other test in this file and hang the run for an hour at
    # teardown -- so the interval is a real 3600s and the stop has to cut through it.
    h = M._start_in_turn_grader(ws, run_dir, _args(grade_interval=3600), interval_grades=True)
    _spin(lambda: h[0].is_alive(), 2.0)
    t0 = time.time()
    M._stop_in_turn_grader(h)
    elapsed = time.time() - t0

    assert not h[0].is_alive(), (
        "the in-turn grader outlived the turn; it would race the authoritative post-turn grade and the "
        "L3 barrier's own verdict.json")
    assert elapsed < 10.0, (
        f"stopping the grader took {elapsed:.1f}s — its interval wait is not interruptible, so every "
        f"turn ends by blocking on it")
    assert not any(t.name.endswith("-grader") and t.is_alive() for t in threading.enumerate())


def test_a_turn_that_lands_no_first_verdict_is_reported_and_not_graded_on(tmp_path, monkeypatch,
                                                                          capfd):
    """A turn with no verdict at all means the agent worked BLIND — that must be said, not inferred.

    Round 0 has no previous round to inherit feedback from and, under `--schedule continuous`, one turn
    can be most of the run: measured, an agent spent 6184s with no `qa/` directory in its workspace. So
    the grader lands a cheap loop-tier verdict FIRST, retrying while the submission is mid-write. Two
    halves are gated here, and neither was gated before:

      * the happy path — the first verdict really is published to the workspace the agent reads, and the
        interval phase then takes over;
      * the unhappy path — when no first grade ever lands, the turn is ANNOUNCED as blind rather than
        passing quietly for a verdict-less verdict to be read later as "nothing to report".
    """
    M = _driver()
    monkeypatch.setattr(M, "_FIRST_GRADE_POLL_S", 0.01)

    # -- happy path: the first grade lands, and the interval phase follows it ------------------------
    ws, run_dir = _ws(tmp_path / "ok", with_verdict=False)
    first, interval = [], []

    def fast(w, _rd, tick, _timeout):
        first.append(tick)
        (w / "qa").mkdir(exist_ok=True)
        (w / "qa" / "verdict.json").write_text('{"n_passed": 1, "n_capsules": 20}')
        return {"n_passed": 1, "n_capsules": 20, "all_pass": None, "tiers_graded": ["loop"],
                "tiers_not_run": ["cert"]}

    monkeypatch.setattr(M, "_fast_loop_verdict", fast)
    monkeypatch.setattr(M, "qa_grade",
                        lambda *a, **k: interval.append(a[2]) or {"all_pass": False})
    with monkeypatch.context() as mp:
        _shrink_waits(mp)
        h = M._start_in_turn_grader(ws, run_dir, _args(grade_interval=900), interval_grades=True)
        _spin(lambda: first and interval)
        M._stop_in_turn_grader(h)
    assert first, "no first (loop-tier) grade was attempted; the agent opens the turn with no verdict"
    assert (ws / "qa" / "verdict.json").is_file(), (
        "the first grade never reached the workspace the agent reads")
    assert interval, "the interval phase never started after the first verdict landed"

    # -- unhappy path: it never lands, so the turn must be reported blind and must not grade on -------
    ws2, run_dir2 = _ws(tmp_path / "blind", with_verdict=False)
    interval.clear()

    def never(*_a, **_k):
        raise RuntimeError("no submission/manifest.yaml to grade yet")

    monkeypatch.setattr(M, "_fast_loop_verdict", never)
    with monkeypatch.context() as mp:
        _shrink_waits(mp)
        h = M._start_in_turn_grader(ws2, run_dir2, _args(grade_interval=900), interval_grades=True)
        time.sleep(0.5)
        M._stop_in_turn_grader(h)
    out = capfd.readouterr().out
    assert "[first-grade]" in out and "NO verdict landed" in out, (
        "a turn in which NO verdict ever landed said nothing about it; the agent ran blind for the "
        f"whole turn and the log gives no way to know. Saw:\n{out[-2000:]}")
    assert not (ws2 / "qa" / "verdict.json").exists(), "a verdict was invented without a grade"
    assert interval == [], (
        "the grader went on to the expensive interval phase for a turn it could not grade at all -- "
        "the full mandatory ladder costs tens of minutes per capsule, and a workspace with no "
        "gradeable submission has nothing for it to measure")


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

    starts = [n for n in ast.walk(loop)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_start_in_turn_grader"]
    assert starts, (
        "the agent loop never starts an in-turn grader; feedback would still only reach the agent by "
        "ending its session, which is the round barrier `--schedule continuous` exists to remove")
    assert all(s.lineno not in legacy for s in starts), (
        "the grader is only started inside the legacy `if a.continuous:` branch, so "
        "`--schedule continuous` still runs the round loop with no grader under the agent")

    # ...and it must re-grade on the interval for the CERTIFIED schedule, not merely land one verdict.
    iv = {ast.unparse(k.value) for s in starts for k in s.keywords if k.arg == "interval_grades"}
    assert iv and all("schedule" in x for x in iv), (
        f"the in-turn grader's interval phase is not keyed to the schedule ({iv or 'absent'}); "
        f"`--schedule continuous` would land ONE verdict per turn and then go quiet")

    stops = [n for n in ast.walk(loop)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_stop_in_turn_grader"]
    assert stops, (
        "the grader is never stopped around the agent launch; it would race the authoritative grade and "
        "the L3 barrier's own verdict.json")

    # The stop must be unconditional (a `finally`), or an agent timeout leaks a grader into the next
    # turn -- and it must precede the authoritative post-turn grade, which is a grade of the FINAL
    # submission and cannot be racing an interval tick.
    assert any(any(n is s for st in ast.walk(loop) if isinstance(st, ast.Try)
                   for n in ast.walk(ast.Module(body=st.finalbody, type_ignores=[])))
               for s in stops), (
        "the in-turn grader is not stopped in a `finally`; an agent TIMEOUT would leave it grading "
        "underneath the authoritative grade")
    authoritative = [n for n in ast.walk(loop)
                     if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "qa_grade"]
    assert authoritative, "the loop no longer takes an authoritative post-turn grade"
    assert min(s.lineno for s in stops) < max(g.lineno for g in authoritative), (
        "the authoritative grade runs while the in-turn grader is still going; the run's verdict would "
        "describe a moving workspace instead of the submission as the turn left it")


def test_the_single_session_is_bounded_by_the_total_budget_not_only_by_the_round_timeout():
    """`--max-wall-s` must reach the launch, or a '12h total' run is really a 12h-per-session run."""
    main = _main_ast()
    loop = _agent_loop(main)

    # --round-timeout bounds ONE invocation. The TOTAL budget binds only if the loop actually advances
    # `active_wall_s` -- a budget that is never advanced is a budget that never binds, and the stop
    # policy below would compare 0.0 against it forever. It has to advance on the path EVERY completed
    # turn takes (a statement of the loop body itself), not only inside the early-exit branches for a
    # rate limit or a dead turn: a run that never hits one of those would be unbounded.
    def _adds(stmts) -> bool:
        return any(isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
                   and isinstance(n.target, ast.Name) and n.target.id == "active_wall_s"
                   for n in stmts)

    assert _adds(loop.body), (
        "the agent loop does not add the turn's elapsed time to active_wall_s on the path every "
        "completed turn takes, so --max-wall-s can never be reached and a '12h total' run is really "
        "bounded only by --round-timeout")

    go = _stop_policy()
    assert go(schedule="continuous", active_wall_s=0.0, max_wall_s=43_200)
    assert not go(schedule="continuous", active_wall_s=43_200.0, max_wall_s=43_200), (
        "the stop policy does not consult the total wall budget; the run would be bounded by "
        "--round-timeout alone")


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
