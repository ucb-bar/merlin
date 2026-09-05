"""Where an agentic run's wall time went — and refusing to answer when the transcript cannot say.

Why this is pinned. The split was computed as ``think+gen = sum(result.duration_api_ms)``, a field only
the claude CLI emits. Every codex run therefore recorded ``think_generate_s: 0.0, tool_and_wait_s: 0.0,
think_pct: 0.0`` — not an error, not a gap, a confident zero meaning "this agent never thought". Each
rule below fails in the direction that manufactures a plausible number, so none is left to review.
"""
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(exc).__name__}: {exc}")
    return mod


TD = _mod("timing_decomposition")

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _at(sec: float) -> str:
    return (T0 + timedelta(seconds=sec)).isoformat()


def _init(round_=0):
    return {"type": "system", "subtype": "init", "driver": "codex", "round": round_,
            "started_at": _at(0)}


def _use(sec, call_id):
    return {"type": "assistant", "arrived_at": _at(sec), "message": {
        "content": [{"type": "tool_use", "id": call_id, "name": "Bash", "input": {}}]}}


def _res(sec, call_id):
    return {"type": "user", "arrived_at": _at(sec), "message": {
        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "ok"}]}}


def _text(sec):
    return {"type": "assistant", "arrived_at": _at(sec),
            "message": {"content": [{"type": "text", "text": "thinking out loud"}]}}


# --- the defect: a driver-specific field silently producing zero -----------------------------------

def test_a_codex_shaped_transcript_is_decomposed_from_its_arrival_stamps():
    """The regression. This transcript has no `duration_api_ms` anywhere — the old arithmetic returned
    0.0/0.0/0.0 for it, which a plot renders as "spent no time thinking"."""
    evts = [_init(), _use(10, "a"), _res(40, "a"), _text(50), _use(60, "b"), _res(70, "b")]
    rec = TD.decompose(evts)
    assert rec["method"] == "arrival_stamps"
    # span 0..70; tools occupy [10,40] and [60,70] = 40 s; the rest is think+generate.
    assert rec["measured_span_s"] == 70.0
    assert rec["tool_and_wait_s"] == 40.0
    assert rec["think_generate_s"] == 30.0
    assert rec["think_pct"] == pytest.approx(42.9, abs=0.1)
    assert rec["think_generate_s"] > 0 and rec["tool_and_wait_s"] > 0


def test_a_transcript_without_arrival_stamps_is_unknown_and_never_zero():
    """THE rule. No stamps and no duration fields ⇒ the split is not measurable. A 0.0 here is a
    measurement claim that was never made, and it averages into a study as if it were one."""
    evts = [{"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "a"}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "a"}]}},
            {"type": "result", "subtype": "success"}]
    rec = TD.decompose(evts)
    assert rec["method"] == "unknown"
    assert rec["think_generate_s"] is None
    assert rec["tool_and_wait_s"] is None
    assert rec["think_pct"] is None
    assert rec["unavailable_reason"]


def test_a_zero_duration_result_event_is_unknown_not_a_zero_split():
    """The exact codex shape: a `result` event exists but carries no duration fields. Summing them
    gives 0 ms of API time and 0 ms of wall — arithmetically fine, and a lie."""
    rec = TD.decompose([{"type": "result", "subtype": "success", "duration_ms": 0,
                         "duration_api_ms": 0}])
    assert rec["method"] == "unknown"
    assert rec["think_generate_s"] is None


def test_claude_duration_fields_are_still_honoured_when_they_are_real():
    """The fallback stays: a stamp-free transcript that genuinely measured itself is not thrown away,
    and `method` says which measurement the numbers came from."""
    rec = TD.decompose([{"type": "system", "subtype": "init"},
                        {"type": "result", "duration_ms": 100_000, "duration_api_ms": 40_000}])
    assert rec["method"] == "duration_api_ms"
    assert rec["think_generate_s"] == 40.0
    assert rec["tool_and_wait_s"] == 60.0
    assert rec["think_pct"] == 40.0


# --- properties the stamps make measurable ---------------------------------------------------------

def test_overlapping_tool_calls_occupy_one_clock_not_two():
    """Measured on a real codex run: a 2118 s backgrounded command ran while 90 further tool calls
    completed under it. Summing per-call durations invented 8327 s of wall time that never elapsed."""
    evts = [_init(), _use(0, "long"), _use(10, "short"), _res(20, "short"), _res(100, "long")]
    rec = TD.decompose(evts)
    assert rec["tool_call_seconds_sum"] == 110.0        # 100 + 10
    assert rec["tool_and_wait_s"] == 100.0              # the union — one clock
    assert rec["tool_concurrency_overlap_s"] == 10.0
    assert rec["think_generate_s"] == 0.0               # a tool was outstanding the whole time


def test_the_between_round_grading_gap_is_not_charged_to_the_agent():
    """A multi-round transcript is one file. The operator's grade between rounds is hours of wall time
    with no agent in it; folding it into `think` would make a slow grader look like a pensive model."""
    evts = [_init(0), _use(10, "a"), _res(20, "a")]
    late = 100_000
    evts += [{"type": "system", "subtype": "init", "round": 1, "started_at": _at(late)},
             _use(late + 10, "b"), _res(late + 20, "b")]
    rec = TD.decompose(evts)
    assert rec["sessions"] == 2
    assert rec["measured_span_s"] == 40.0               # 20 + 20, not 100 020
    assert rec["between_session_s"] == pytest.approx(late - 20, abs=1)
    assert rec["think_generate_s"] == 20.0              # 2 x the 10 s before each tool call


def test_a_tool_call_whose_result_never_arrived_is_counted_not_dropped():
    """A round cut off mid-command. Dropping the unmatched call would move its wall time into
    `think`, which is the one direction that flatters the model."""
    rec = TD.decompose([_init(), _use(10, "a"), _text(300)])
    assert rec["tool_calls_unterminated"] == 1
    assert rec["tool_and_wait_s"] == 290.0
    assert rec["think_generate_s"] == 10.0


def test_a_raw_codex_event_stream_decomposes_too():
    """`rounds/round_NN.codex_events.timestamped.jsonl` is the same timeline in the driver's own
    vocabulary. One algebra reads both, so neither shape is the shape that works."""
    def ev(sec, etype, item_id, status):
        return {"seq": 1, "arrived_at": _at(sec),
                "event": {"type": etype, "item": {"id": item_id, "type": "command_execution",
                                                  "status": status}}}
    rec = TD.decompose([ev(0, "item.started", "i1", "in_progress"),
                        ev(30, "item.completed", "i1", "completed"),
                        ev(50, "item.started", "i2", "in_progress"),
                        ev(60, "item.completed", "i2", "completed")])
    assert rec["method"] == "arrival_stamps"
    assert rec["tool_and_wait_s"] == 40.0
    assert rec["think_generate_s"] == 20.0


def test_the_split_partitions_the_measured_span():
    """think + tool must equal the span exactly. Any other identity means one of them is a guess."""
    evts = [_init(), _use(5, "a"), _use(7, "b"), _res(9, "b"), _res(40, "a"), _text(55),
            _use(60, "c"), _res(90, "c")]
    rec = TD.decompose(evts)
    assert rec["think_generate_s"] + rec["tool_and_wait_s"] == pytest.approx(rec["measured_span_s"])


# --- the run-directory product ---------------------------------------------------------------------

def test_a_run_with_no_transcript_says_so_instead_of_reporting_zeros(tmp_path):
    rec = TD.decompose_run(tmp_path)
    assert rec["method"] == "unknown"
    assert rec["think_generate_s"] is None
    assert "no transcript" in rec["unavailable_reason"]


def test_write_run_timing_round_trips_a_real_shaped_run(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    evts = [_init(), _use(10, "a"), _res(40, "a"), _text(60)]
    (rounds / "round_00.transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in evts), encoding="utf-8")
    (tmp_path / "circt_gate_log.jsonl").write_text(
        json.dumps({"sim_skipped": True}) + "\n" + json.dumps({"sim_skipped": False}), encoding="utf-8")
    out = TD.write_run_timing(tmp_path)
    rec = json.loads(out.read_text())
    assert rec["method"] == "arrival_stamps"
    assert rec["tool_and_wait_s"] == 30.0
    assert rec["think_generate_s"] == 30.0
    assert rec["circt_gate"] == {"sims_skipped": 1, "sims_run": 1}
    assert rec["transcripts"] == ["round_00.transcript.jsonl"]


def test_the_cli_reported_split_is_kept_beside_the_derived_one_not_merged_into_it():
    """The two readings cut the run differently — API latency vs everything else is not tools vs
    thinking — and on a measured claude run they disagree (98 s of stamped tool intervals against
    649 s of non-API time). Silently preferring either would erase a real disagreement."""
    evts = [_init(), _use(10, "a"), _res(40, "a"), _text(60),
            {"type": "result", "duration_ms": 60_000, "duration_api_ms": 50_000}]
    rec = TD.decompose(evts)
    assert rec["method"] == "arrival_stamps"
    assert rec["tool_and_wait_s"] == 30.0          # derived from the stamps, unchanged
    assert rec["cli_reported"]["api_time_s"] == 50.0
    assert rec["cli_reported"]["non_api_time_s"] == 10.0


# --- the WRITER, not just the library -----------------------------------------------------------
def test_the_run_writer_records_a_derived_split_not_zeros(tmp_path):
    """The library being right did not help while the writer still did the old arithmetic.

    `_emit_run_timing` computed `sum(result.duration_api_ms)` vs `duration_ms - api_ms` -- fields only
    the claude CLI's terminal result event carries -- so every codex run recorded 0.0/0.0/0.0: not an
    error, not a gap marker, a confident claim that the agent never thought. This pins that the writer
    uses the derived split and still emits the harness's own circt_gate counts.
    """
    import json
    import sys
    from merlin.common.paths import merlin_dir
    sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
    import run_baseline_qa_loop as L

    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    t = "2026-09-05T00:%02d:%02d+00:00"
    rows = [
        {"type": "assistant", "arrived_at": t % (0, 0), "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "arrived_at": t % (0, 30), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}},
        {"type": "assistant", "arrived_at": t % (1, 30), "message": {"content": [
            {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "arrived_at": t % (1, 40), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "b", "content": "ok"}]}},
    ]
    (run / "rounds" / "round_00.transcript.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    L._emit_run_timing(run, [])
    rec = json.loads((run / "timing_detailed.json").read_text())
    assert rec["think_generate_s"], "the writer still reports no thinking time"
    assert rec["tool_and_wait_s"], "the writer still reports no tool time"
    assert rec.get("method") == "arrival_stamps"
    assert "circt_gate" in rec, "the harness's own gate counts were dropped"


def test_the_writer_says_unknown_rather_than_zero_without_stamps(tmp_path):
    """No stamps must yield null + a reason, never a plausible 0.0."""
    import json
    import sys
    from merlin.common.paths import merlin_dir
    sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
    import run_baseline_qa_loop as L

    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    (run / "rounds" / "round_00.transcript.jsonl").write_text(json.dumps(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}]}}) + "\n")
    L._emit_run_timing(run, [])
    rec = json.loads((run / "timing_detailed.json").read_text())
    assert rec["think_generate_s"] is None and rec["think_pct"] is None
