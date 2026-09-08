"""The aet telemetry sink fires WHILE the agent works, not only when the run finishes.

A capsule-bench run under ``--schedule continuous`` is ONE round that can last twelve hours. For a
long time the only ``emit_to_aet`` call sat after that round returned, so a run that was killed, timed
out (rc=124) or hit its wall budget recorded nothing at all -- no ``logs/metrics.jsonl``, no
``metrics/trajectory.json``, invisible to ``aet spend`` and ``aet plot`` -- while its transcript sat on
disk the whole time. MEASURED: the gemmini run of 2026-09-07 graded 93/97 across two rc=124 rounds and
left no aet record whatsoever.

Three separate things have to hold, and each one is invisible when it breaks:

1. the in-turn grader -- the ONLY loop that runs under ``--schedule continuous`` -- offers a per-tick
   hook at all;
2. the real call site actually passes the sink into it. This is the half a comment cannot detect: an
   orphaned hook and a firing hook look identical from the function definition;
3. re-emitting on a cadence does not inflate the numbers. ``logs/metrics.jsonl`` is append-only, so a
   sink that ran 48 times over 12h would be a disaster if consumers summed it. They do not -- aet reads
   it as "the last occurrence of each name wins" -- and that contract is asserted here rather than
   assumed, because it lives in a different repo.

The sink is wrapped in ``try/except`` because telemetry must never gate a run. That is exactly why it
needs a test: five defects once hid in this file's other soft-failing path, and every one of them
presented as "nothing needed doing".
"""
from __future__ import annotations

import ast
import json
import os

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
LOOP = HARNESS / "run_baseline_qa_loop.py"


def _tree() -> ast.Module:
    # Parsed, not imported: the loop pulls in the whole harness at import time, and a test that
    # skipped on that would report success while checking nothing.
    return ast.parse(LOOP.read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {LOOP.name}")


def test_in_turn_grader_offers_a_tick_hook() -> None:
    """Property 1: the loop that runs under --schedule continuous can call back per grade."""
    fn = _func(_tree(), "_start_in_turn_grader")
    kwonly = [a.arg for a in fn.args.kwonlyargs] + [a.arg for a in fn.args.args]
    assert "on_tick" in kwonly, (
        "_start_in_turn_grader lost its on_tick hook; under --schedule continuous this is the only "
        "loop running while the agent works, so nothing can be sunk mid-run without it"
    )
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "on_tick"]
    assert calls, "_start_in_turn_grader accepts on_tick but never calls it"


def test_the_continuous_call_site_passes_the_sink() -> None:
    """Property 2: the hook is WIRED, not orphaned.

    Keyed on the call that carries ``interval_grades=(a.schedule == "continuous")`` -- the certified
    path -- so this cannot be satisfied by the legacy ``--continuous`` block's own grader.
    """
    tree = _tree()
    wired = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_start_in_turn_grader"):
            continue
        kwargs = {k.arg for k in node.keywords if k.arg}
        if "interval_grades" in kwargs:
            wired.append(kwargs)
    assert wired, "no _start_in_turn_grader(interval_grades=...) call site found"
    for kwargs in wired:
        assert "on_tick" in kwargs, (
            "the continuous call site does not pass on_tick, so the mid-run aet sink never fires and a "
            "killed run records no telemetry -- the 2026-09-07 failure exactly"
        )


def test_the_sink_helper_uses_the_sticky_check_and_never_raises() -> None:
    """``aet_sink_enabled`` must be asked WITH the run_dir, or its stickiness branch is dead.

    Branch 2 of ``aet_bridge.aet_sink_enabled`` ("this run already has a record") exists so a resume
    from a shell that never exported MERLIN_AET_SINK keeps recording. Called with no argument it can
    only ever consult the env var, and the run goes quiet.
    """
    tree = _tree()
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "aet_sink_enabled"]
    assert calls, "no aet_sink_enabled() call in the loop"
    for call in calls:
        assert call.args, (
            "aet_sink_enabled() called with no run_dir: the sticky branch is unreachable, so a "
            "--resume from another shell stops recording silently"
        )

    helper = _func(tree, "_sink_telemetry_now")
    handlers = [n for n in ast.walk(helper) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "_sink_telemetry_now must not be able to kill a run; wrap it in try/except"


def test_emit_writes_a_record_from_per_round_transcripts_alone(tmp_path, monkeypatch) -> None:
    """The mid-run condition: no combined transcript.jsonl exists yet, only rounds/.

    If the sink could only read the combined transcript (written at finalize) then calling it mid-run
    would be a no-op that logs nothing and warns nothing -- a sink that reports success while doing
    nothing.
    """
    pytest.importorskip("aet.tracking.run_logger")
    from merlin.targetgen import aet_bridge as AB

    rounds = tmp_path / "rounds"
    rounds.mkdir()
    (rounds / "round_00.transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "assistant", "message": {"model": "gpt-5.6-sol", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
        {"type": "result", "subtype": "success", "session_id": "s1", "num_turns": 1},
    ]))
    assert not (tmp_path / "transcript.jsonl").exists()

    monkeypatch.setenv("MERLIN_AET_SINK", "1")
    assert AB.emit_to_aet(run_dir=tmp_path, run_id="unit", method="raw_baseline",
                          model="gpt-5.6-sol", target="gemmini", suite="capsule-bench",
                          billing_mode="subscription_notional")
    assert (tmp_path / "logs" / "metrics.jsonl").is_file()
    assert (tmp_path / "metrics" / "trajectory.json").is_file()

    # ... and the record itself now switches the sink on, with no env var in sight.
    monkeypatch.delenv("MERLIN_AET_SINK")
    assert AB.aet_sink_enabled(tmp_path)


def test_re_emitting_on_a_cadence_supersedes_it_does_not_accumulate(tmp_path, monkeypatch) -> None:
    """Property 3: 48 ticks over 12h must not read as 48x the tokens.

    ``metrics.jsonl`` is append-only (aet's LocalBackend opens it "a"), so this is only safe because
    every consumer takes the LAST occurrence of each metric name. Asserted against aet's own reader,
    since that contract lives in another repo and could change under us.
    """
    pytest.importorskip("aet.tracking.run_logger")
    from aet.trajectory.rollup import _read_metrics

    from merlin.targetgen import aet_bridge as AB

    def _round(path, inp, out, cache):
        path.write_text("".join(json.dumps(r) + "\n" for r in [
            {"type": "assistant", "message": {"model": "gpt-5.6-sol", "content": [],
                "usage": {"input_tokens": inp, "output_tokens": out,
                          "cache_read_input_tokens": cache, "cache_creation_input_tokens": 0}}},
            {"type": "result", "subtype": "success", "session_id": "s1", "num_turns": 1},
        ]))

    rounds = tmp_path / "rounds"
    rounds.mkdir()
    _round(rounds / "round_00.transcript.jsonl", 100, 20, 900)

    monkeypatch.setenv("MERLIN_AET_SINK", "1")
    kw = dict(run_id="unit", method="raw_baseline", model="gpt-5.6-sol", target="gemmini",
              suite="capsule-bench", billing_mode="subscription_notional")
    assert AB.emit_to_aet(run_dir=tmp_path, **kw)
    _, tok, _, _ = _read_metrics(tmp_path)
    assert (tok.input, tok.output, tok.cache_read) == (100, 20, 900)

    # a second round lands; the sink fires again on the next grader tick
    _round(rounds / "round_01.transcript.jsonl", 50, 10, 100)
    assert AB.emit_to_aet(run_dir=tmp_path, **kw)
    _, tok, _, _ = _read_metrics(tmp_path)
    assert (tok.input, tok.output, tok.cache_read) == (150, 30, 1000), (
        "re-emitting double-counted: the periodic sink would inflate every figure it records "
        f"(got {tok.input}/{tok.output}/{tok.cache_read}, want the run totals 150/30/1000)"
    )
