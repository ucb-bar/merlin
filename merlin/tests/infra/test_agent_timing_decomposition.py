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


def _use(sec, call_id, name="Bash", input_=None):
    return {"type": "assistant", "arrived_at": _at(sec), "message": {
        "content": [{"type": "tool_use", "id": call_id, "name": name,
                     "input": input_ or {}}]}}


def _res(sec, call_id, content="ok", is_error=False):
    return {"type": "user", "arrived_at": _at(sec), "message": {
        "content": [{"type": "tool_result", "tool_use_id": call_id,
                     "content": content, "is_error": is_error}]}}


def _usage(sec, message_id, fresh, write, read, output, reasoning):
    return {"type": "assistant", "arrived_at": _at(sec), "message": {
        "id": message_id, "model": "gpt-test", "content": [], "usage": {
            "input_tokens": fresh, "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read, "output_tokens": output,
            "reasoning_output_tokens": reasoning}}}


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


def test_named_tools_latency_errors_tokens_rates_and_activity_share_are_recorded():
    evts = [_init(), _use(10, "a", "Bash", {"command": "one"}), _res(40, "a", "done"),
            _use(50, "b", "Edit", {"patch": "x"}), _use(55, "c", "Bash"),
            _res(60, "b", "bad", True), _usage(65, "m1", 100, 20, 300, 40, 25),
            _usage(66, "m1", 100, 20, 300, 40, 25),  # duplicate streaming envelope
            _res(75, "c"), _usage(80, "m2", 50, 10, 40, 20, 5)]
    rec = TD.decompose(evts)
    assert rec["activity_share"]["tool_and_wait"] == pytest.approx(55 / 80)
    assert rec["activity_share"]["think_generate"] == pytest.approx(25 / 80)
    assert sum(rec["activity_share"][k] for k in ("tool_and_wait", "think_generate")) == 1
    assert rec["tools"]["used"] == ["Bash", "Edit"]
    assert rec["tools"]["by_tool"]["Bash"]["calls_completed"] == 2
    assert rec["tools"]["by_tool"]["Bash"]["duration_sum_s"] == 50
    assert rec["tools"]["by_tool"]["Bash"]["duration_mean_s"] == 25
    assert rec["tools"]["by_tool"]["Bash"]["duration_max_s"] == 30
    assert rec["tools"]["by_tool"]["Edit"]["errors"] == 1
    assert rec["tool_call_seconds_sum"] == 60
    assert rec["tool_concurrency_overlap_s"] == 5
    tokens = rec["tokens"]
    assert tokens["tokens_fresh_input"] == 150
    assert tokens["tokens_cache_write"] == 30
    assert tokens["tokens_cache_read"] == 340
    assert tokens["tokens_output"] == 60
    assert tokens["tokens_reasoning"] == 30
    assert tokens["tokens_total"] == 580
    assert tokens["cache_read_share_of_input"] == pytest.approx(340 / 520)
    assert rec["rates"]["output_tokens_per_think_generate_s"] == pytest.approx(2.4)
    assert rec["rates"]["output_tokens_per_agent_span_s"] == pytest.approx(0.75)
    assert rec["rates"]["total_tokens_per_agent_span_s"] == pytest.approx(7.25)


def test_oracle_timing_keeps_every_l_tier_invocation_and_null_is_not_zero(tmp_path):
    base = tmp_path / "_qa_work" / "runs_r0" / "runs" / "target-capsule-bench"
    fixtures = {
        "A": {"L2": {"status": "pass", "engine": "spike", "timing": {
            "build_s": 1, "sim_active_s": 2, "oracle_wait_s": 3, "adapter_wall_s": 6}},
              "L3": {"status": "pass", "engine": "gsim", "timing": {
                  "build_s": 4, "sim_active_s": 5, "oracle_wait_s": 6, "adapter_wall_s": 15}}},
        "B": {"L3": {"status": "fail", "engine": "gsim", "timing": {
            "build_s": 7, "sim_active_s": 8, "oracle_wait_s": 9, "adapter_wall_s": 24}}},
        "C": {"L3": {"status": "unavailable", "engine": "gsim", "timing": None}},
    }
    for capsule, tiers in fixtures.items():
        path = base / capsule / "capsule_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"capsule": capsule, "tiers": tiers}))
    rec = TD.oracle_timing(tmp_path)
    assert rec["by_tier"]["L2"]["totals_s"]["adapter_wall_s"] == 6
    l3 = rec["by_tier"]["L3"]
    assert l3["records"] == 3 and l3["timed_records"] == 2
    assert l3["missing_timing_records"] == 1
    assert l3["totals_s"] == {"build_s": 11.0, "sim_active_s": 13.0,
                               "oracle_wait_s": 15.0, "adapter_wall_s": 39.0}
    assert l3["totals_are_lower_bounds"] is True
    assert len(rec["per_invocation"]) == 4


def _complete_codex_run(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    events = [_init(), _use(10, "a"), _res(20, "a"), _usage(30, "m1", 10, 0, 20, 5, 1),
              {"type": "codex_summary", "arrived_at": _at(31), "turns_started": 1,
               "turns_usage_reported": 1, "usage_complete": True, "unknown_types": []}]
    (rounds / "round_00.transcript.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n")
    raw = [{"type": "turn.started"}, {"type": "turn.completed", "usage": {
        "input_tokens": 30, "cached_input_tokens": 20, "output_tokens": 5}}]
    (rounds / "round_00.codex_events.raw.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw) + "\n")
    (rounds / "round_00.codex_events.timestamped.jsonl").write_text(
        "\n".join(json.dumps({"seq": i, "arrived_at": _at(i), "event": row})
                  for i, row in enumerate(raw, 1)) + "\n")
    for suffix, content in (("prompt.txt", "prompt"), ("final.txt", "answer")):
        (rounds / f"round_00.{suffix}").write_text(content)
    (rounds / "round_00.codex_summary.json").write_text(json.dumps({
        "turns_started": 1, "turns_usage_reported": 1, "usage_complete": True,
        "unknown_types": [], "wall_s": 31}))
    rollout = rounds / "round_00.codex_rollout_snapshot" / "rollout.jsonl"
    rollout.parent.mkdir()
    rollout.write_text("\n".join(json.dumps(row) for row in [
        {"timestamp": _at(0), "type": "event_msg", "payload": {"type": "task_started"}},
        {"timestamp": _at(5), "type": "token_usage_record", "payload": {
            "turn_id": "t", "response_id": "r", "usage": {
                "input_tokens": 30, "cached_input_tokens": 20, "cache_write_input_tokens": 0,
                "output_tokens": 5, "reasoning_output_tokens": 1}}}]) + "\n")
    evidence = tmp_path / "agent_evidence_snapshot" / ".qa_channel"
    evidence.mkdir(parents=True)
    (evidence / "progress.json").write_text("{}")
    (rounds / "round_00.resource_samples.jsonl").write_text(json.dumps({
        "sampled_at": _at(0), "rss_bytes": 1, "processes": 1, "threads": 1}) + "\n")
    return tmp_path


def test_complete_telemetry_is_sealed_reconciled_and_gateable(tmp_path):
    run = _complete_codex_run(tmp_path)
    rec = TD.decompose_run(run)
    assert rec["stream_reconciliation"]["complete"] is True
    assert rec["telemetry_integrity"] == {
        "complete": True, "failures": [], "rounds": [{
            "round": "round_00", "complete": True, "failures": [], "turns_started": 1,
            "turns_usage_reported": 1, "driver_wall_s": 31}],
        "policy": rec["telemetry_integrity"]["policy"]}
    assert rec["llm"]["tokens"]["tokens_cache_read"] == 20
    assert rec["resources"]["available"] is True
    roles = {row["role"] for row in rec["artifacts"]["files"]}
    assert "authoritative_provider_cli_stream" in roles
    assert "authoritative_codex_rollout_full_io_and_incremental_usage" in roles


def test_missing_raw_stream_blocks_telemetry_completion(tmp_path):
    run = _complete_codex_run(tmp_path)
    (run / "rounds" / "round_00.codex_events.raw.jsonl").unlink()
    rec = TD.decompose_run(run)
    assert rec["telemetry_integrity"]["complete"] is False
    assert "round_00:missing_raw" in rec["telemetry_integrity"]["failures"]


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
