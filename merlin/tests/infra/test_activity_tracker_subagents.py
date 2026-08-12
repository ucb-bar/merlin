"""The activity/cost tracker must count SUBAGENT + background-tier activity, not just the top-level agent.

Claude Code emits per-turn `assistant` events whose streamed `usage.output_tokens` is a partial artifact
and which OMIT delegated sub-agent / background tiers entirely; the authoritative, subagent-inclusive
per-model totals live only in the terminal `result` event's `modelUsage`. `parse_transcript` must prefer
that (and fall back to the streamed aggregation only for a truncated run with no result event).
"""
from __future__ import annotations

import json

from merlin.targetgen.experiment_tokens import parse_transcript


def _write(tmp_path, events):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


def test_result_event_counts_subagent_and_true_output(tmp_path):
    # top-level assistant: streamed output_tokens is a tiny partial artifact (4), no haiku tier here
    assistant = {"type": "assistant", "message": {
        "id": "m1", "model": "opus",
        "usage": {"input_tokens": 200000, "output_tokens": 4, "cache_read_input_tokens": 50},
        "content": [{"type": "text", "text": "x" * 300}, {"type": "tool_use", "name": "Bash"}]}}
    # terminal result: AUTHORITATIVE per-model totals incl. the delegated haiku subagent + true cost
    result = {"type": "result", "total_cost_usd": 10.19, "modelUsage": {
        "opus": {"inputTokens": 200000, "outputTokens": 107207, "cacheReadInputTokens": 50,
                 "cacheCreationInputTokens": 0, "costUSD": 10.0},
        "haiku": {"inputTokens": 500, "outputTokens": 21, "cacheReadInputTokens": 0,
                  "cacheCreationInputTokens": 0, "costUSD": 0.19}}}
    s = parse_transcript(_write(tmp_path, [assistant, result]))
    assert s["available"] and s["usage_source"] == "result_event"
    # the subagent tier is tracked (was dropped entirely before) ...
    assert "haiku" in s["tokens_native_by_model"]
    # ... and output is the true subagent-inclusive total (107207+21), not the ~4 streamed artifact
    assert s["tokens_output"] == 107207 + 21
    assert s["estimated_cost_usd"] == 10.19          # the CLI's authoritative total
    # top-level tool call still counted, and we are honest that subagent tool counts are not recoverable
    assert s["tool_calls"] == 1 and s["subagent_tool_calls_tracked"] is False


def test_falls_back_to_stream_when_no_result_event(tmp_path):
    # a truncated/killed run: only assistant events, no terminal result -> use the streamed aggregation
    assistant = {"type": "assistant", "message": {
        "id": "m1", "model": "opus",
        "usage": {"input_tokens": 100, "output_tokens": 42, "cache_read_input_tokens": 0},
        "content": [{"type": "text", "text": "hi"}]}}
    s = parse_transcript(_write(tmp_path, [assistant]))
    assert s["available"] and s["usage_source"] == "assistant_stream"
    assert s["tokens_output"] == 42
    assert "opus" in s["tokens_native_by_model"] and "haiku" not in s["tokens_native_by_model"]
