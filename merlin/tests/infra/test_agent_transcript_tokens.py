"""Token accounting for the agent CLIs whose transcripts are NOT claude stream-json.

Why this is pinned: `parse_transcript` returns `available: False, "no usage metadata in transcript"`
for a codex or opencode file -- indistinguishable from a run that genuinely used no tokens. A study
whose headline axis is cumulative cost would have plotted zeros while every transcript on disk
carried complete usage. Each rule below fails in the direction that flatters or silently voids a cost
claim, so none is left to reviewer attention.
"""
import json

import pytest

from merlin.targetgen import experiment_tokens as ET


def _w(tmp_path, records, name="t.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def _codex_turn(inp, cached, out, reasoning=0, cache_write=0):
    return {"type": "turn.completed",
            "usage": {"input_tokens": inp, "cached_input_tokens": cached,
                      "cache_write_input_tokens": cache_write, "output_tokens": out,
                      "reasoning_output_tokens": reasoning}}


def _oc_step(inp, out, reasoning=0, read=0, write=0, cost=None):
    part = {"type": "step-finish",
            "tokens": {"input": inp, "output": out, "reasoning": reasoning,
                       "cache": {"read": read, "write": write}}}
    if cost is not None:
        part["cost"] = cost
    return {"type": "step_finish", "part": part}


# --- codex ------------------------------------------------------------------------------------

def test_codex_input_is_net_of_the_cached_prefix(tmp_path):
    """`input_tokens` ALREADY contains the cached prefix; adding it overstates fresh input."""
    t = _w(tmp_path, [_codex_turn(221310, 188928, 5060)])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol")
    assert r["tokens_input"] == 221310 - 188928
    assert r["tokens_cached"] == 188928


def test_codex_reasoning_is_not_added_to_the_total(tmp_path):
    """Codex reasoning is a SUBSET of output; adding it would double-count generated tokens."""
    t = _w(tmp_path, [_codex_turn(1000, 0, 500, reasoning=300)])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol")
    assert r["reasoning_is_subset_of_output"] is True
    assert r["tokens_total"] == 1000 + 500


def test_codex_rounds_accumulate(tmp_path):
    t = _w(tmp_path, [_codex_turn(100, 0, 10), _codex_turn(200, 0, 20)])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol")
    assert (r["tokens_input"], r["tokens_output"], r["turns_completed"]) == (300, 30, 2)


def test_a_turn_that_never_completed_is_unpriced_not_zero(tmp_path):
    """A timed-out round must not read as a free round."""
    t = _w(tmp_path, [{"type": "turn.started"}, {"type": "item.completed"}])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol")
    assert r["available"] is False
    assert r["usage_complete"] is False
    assert "not zero" in r["reason"]


def test_a_partially_completed_run_is_flagged_incomplete(tmp_path):
    t = _w(tmp_path, [{"type": "turn.started"}, {"type": "turn.started"},
                      _codex_turn(100, 0, 10)])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol")
    assert r["available"] is True and r["usage_complete"] is False


# --- opencode ---------------------------------------------------------------------------------

def test_opencode_sums_per_step_usage(tmp_path):
    t = _w(tmp_path, [_oc_step(100, 10), _oc_step(200, 20)])
    r = ET.parse_agent_transcript(t, driver="opencode", model="m")
    assert (r["tokens_input"], r["tokens_output"], r["steps"]) == (300, 30, 2)


def test_opencode_reasoning_is_additive_when_it_exceeds_output(tmp_path):
    """Measured: gemini reported reasoning 47377 against output 22608 -- not a subset.

    Treating it as codex's subset would drop those generated tokens from the cost axis entirely.
    """
    t = _w(tmp_path, [_oc_step(1000, 100, reasoning=400)])
    r = ET.parse_agent_transcript(t, driver="opencode", model="m")
    assert r["reasoning_is_subset_of_output"] is False
    assert r["tokens_total"] == 1000 + 100 + 400


def test_cached_reads_are_not_counted_as_fresh_input(tmp_path):
    """The cost curve is plotted against fresh tokens; a cached prefix is the cheap half."""
    t = _w(tmp_path, [_oc_step(1000, 10, read=5000)])
    r = ET.parse_agent_transcript(t, driver="opencode", model="m")
    assert r["tokens_input"] == 1000 and r["tokens_cached"] == 5000


def test_opencode_cached_inclusion_is_unknown_not_assumed(tmp_path):
    t = _w(tmp_path, [_oc_step(10, 1)])
    assert ET.parse_agent_transcript(t, driver="opencode", model="m")["input_included_cached"] is None


# --- billing separation -----------------------------------------------------------------------

def test_a_seat_run_reports_no_billed_dollars(tmp_path):
    """Notional dollars must never reach a field an aggregator could sum into a real budget."""
    t = _w(tmp_path, [_codex_turn(1000, 0, 100)])
    r = ET.parse_agent_transcript(t, driver="codex", model="gpt-5.6-sol",
                                  billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    assert r["estimated_cost_usd"] is None
    assert "not money spent" in r["cost_unavailable_reason"]


def test_an_unpriced_model_yields_no_dollar_figure(tmp_path):
    """Fail closed: a made-up rate is worse than a named gap."""
    t = _w(tmp_path, [_oc_step(1000, 10)])
    r = ET.parse_agent_transcript(t, driver="opencode", model="no-such-model-anywhere")
    assert r["estimated_cost_usd"] is None
    assert "no price entry" in r["cost_unavailable_reason"]


def test_the_cli_s_own_cost_is_preferred_when_present(tmp_path):
    t = _w(tmp_path, [_oc_step(1000, 10, cost=0.25)])
    r = ET.parse_agent_transcript(t, driver="opencode", model="m")
    assert r["estimated_cost_usd"] == 0.25 and r["cost_source"] == "opencode_reported"


# --- shape / robustness -----------------------------------------------------------------------

def test_an_unknown_driver_is_refused_by_name(tmp_path):
    r = ET.parse_agent_transcript(_w(tmp_path, [{"a": 1}]), driver="not-a-cli", model="m")
    assert r["available"] is False and "no transcript reader" in r["reason"]


def test_a_missing_transcript_is_reported_not_raised(tmp_path):
    r = ET.parse_agent_transcript(tmp_path / "nope.jsonl", driver="codex", model="m")
    assert r["available"] is False


def test_malformed_lines_do_not_abort_the_parse(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("garbage\n{\"broken\":\n" + json.dumps(_codex_turn(100, 0, 10)))
    r = ET.parse_agent_transcript(p, driver="codex", model="gpt-5.6-sol")
    assert r["available"] is True and r["tokens_input"] == 100


@pytest.mark.parametrize("driver,rec", [("codex", _codex_turn(100, 0, 10)),
                                        ("opencode", _oc_step(100, 10))])
def test_both_readers_return_the_shape_the_claude_parser_returns(tmp_path, driver, rec):
    """Interchangeable downstream, or the aggregator needs a branch per driver."""
    r = ET.parse_agent_transcript(_w(tmp_path, [rec]), driver=driver, model="m")
    assert {"available", "tokens_input", "tokens_cached", "tokens_output", "tokens_total",
            "billing_mode", "usage_source"} <= set(r)


# --- reasoning BLOCKS vs reasoning TOKENS -----------------------------------------------------------
# `thinking_blocks: 0` beside `tokens_reasoning: 106503` was recorded for a real 5.4-hour codex run.
# Both numbers came from the same file; one of them is impossible. The block counter recognised only
# the Claude `{"type": "thinking"}` content block, so every other driver's reasoning was reported as
# none rather than as unmeasured.

def _assistant(blocks, model="m", usage=None):
    msg = {"id": f"id_{id(blocks)}", "model": model, "content": blocks}
    if usage is not None:
        msg["usage"] = usage
    return {"type": "assistant", "message": msg}


def test_codex_shaped_run_records_unknown_reasoning_blocks(tmp_path):
    """THE rule. A driver that bills reasoning tokens but delimits no reasoning block cannot
    report a block COUNT. Zero is the one answer that is both wrong and quotable."""
    p = _w(tmp_path, [
        _assistant([{"type": "text", "text": "hi"}]),
        _assistant([{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
        _assistant([], usage={"input_tokens": 100, "output_tokens": 50,
                              "reasoning_output_tokens": 40, "cache_read_input_tokens": 0}),
    ])
    rec = ET.parse_transcript(p)
    assert rec["tokens_reasoning"] == 40
    assert rec["thinking_blocks"] is None, "0 reasoning blocks beside 40 reasoning tokens is impossible"
    assert rec["thinking_blocks_unavailable_reason"]


def test_a_run_that_reported_no_reasoning_at_all_is_zero_not_unknown(tmp_path):
    """Fail-closed must not become fail-noisy: no reasoning tokens AND no reasoning blocks is a
    genuine zero, and reporting it as unknown would hide a model that really did not think."""
    p = _w(tmp_path, [_assistant([{"type": "text", "text": "hi"}],
                                 usage={"input_tokens": 10, "output_tokens": 5})])
    rec = ET.parse_transcript(p)
    assert rec["thinking_blocks"] == 0
    assert "thinking_blocks_unavailable_reason" not in rec


def test_claude_thinking_blocks_are_still_counted(tmp_path):
    p = _w(tmp_path, [
        _assistant([{"type": "thinking", "thinking": "..."},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
                   usage={"input_tokens": 10, "output_tokens": 5}),
        _assistant([{"type": "redacted_thinking", "data": "..."}]),
    ])
    rec = ET.parse_transcript(p)
    assert rec["thinking_blocks"] == 2
    assert rec["tool_calls"] == 1


def test_a_raw_codex_stream_reasoning_item_is_a_block(tmp_path):
    """Codex publishes reasoning as its own ITEM type. Counting it means the number is real for that
    driver instead of permanently unknown."""
    def item(item_id, itype, extra=None):
        return {"type": "item.completed", "item": dict({"id": item_id, "type": itype}, **(extra or {}))}
    p = _w(tmp_path, [item("i0", "reasoning", {"text": "..."}),
                      item("i1", "command_execution", {"exit_code": 0}),
                      item("i2", "reasoning", {"text": "..."}),
                      _codex_turn(1000, 900, 100, reasoning=60)])
    rec = ET.parse_agent_transcript(p, driver="codex", model="gpt-5.6-sol")
    assert rec["thinking_blocks"] == 2
    assert rec["tool_calls"] == 1
    assert "thinking_blocks_unavailable_reason" not in rec


def test_the_raw_codex_path_reports_unknown_blocks_when_it_delimits_none(tmp_path):
    p = _w(tmp_path, [{"type": "item.completed",
                       "item": {"id": "i1", "type": "command_execution", "exit_code": 0}},
                      _codex_turn(1000, 900, 100, reasoning=60)])
    rec = ET.parse_agent_transcript(p, driver="codex", model="gpt-5.6-sol")
    assert rec["tool_calls"] == 1
    assert rec["thinking_blocks"] is None
    assert rec["thinking_blocks_unavailable_reason"]


def test_opencode_reasoning_parts_are_blocks(tmp_path):
    p = _w(tmp_path, [{"type": "part", "part": {"type": "reasoning", "text": "..."}},
                      {"type": "part", "part": {"type": "tool", "tool": "bash",
                                                "callID": "c1", "state": {}}},
                      _oc_step(100, 50, reasoning=20)])
    rec = ET.parse_agent_transcript(p, driver="opencode", model="glm")
    assert rec["thinking_blocks"] == 1
    assert rec["tool_calls"] == 1
