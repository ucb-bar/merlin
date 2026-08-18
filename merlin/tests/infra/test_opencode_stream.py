"""The opencode driver reconstructs the transcript from the ``run --format json`` STDOUT stream (robust),
not a post-hoc ``opencode export`` (which is blind when the session data dir is sandbox-isolated — the
empty-transcript bug). Hermetic: a synthetic stream in opencode 1.18.10's exact shape, no opencode call.
"""
from __future__ import annotations

import json
import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "capsule_bench" / "harness"))  # sibling imports (model_tiers, ...)


def _load_driver():
    import opencode_agent
    return opencode_agent


_STREAM = "\n".join(json.dumps(x) for x in [
    {"type": "step_start", "part": {"type": "step-start"}},
    {"type": "tool_use", "part": {"type": "tool", "tool": "bash", "callID": "c1",
                                   "state": {"status": "completed", "input": {"command": "echo hi"}, "output": "hi\n"}}},
    {"type": "text", "part": {"type": "text", "id": "p1", "text": "done"}},
    {"type": "step_finish", "part": {"type": "step-finish", "tokens": {"input": 100, "output": 5, "cache": {"read": 3, "write": 1}}}},
])


def test_stream_parse_emits_correlated_tool_events():
    OA = _load_driver()
    events: list[dict] = []
    n = OA._parse_run_stream(_STREAM, "amazon-bedrock/zai.glm-5", 0, events.append)
    tool_uses = [b for e in events if e["type"] == "assistant"
                 for b in e["message"]["content"] if b.get("type") == "tool_use"]
    tool_results = [b for e in events if e["type"] == "user"
                    for b in e["message"]["content"] if b.get("type") == "tool_result"]
    assert n == 1
    assert [b["id"] for b in tool_uses] == ["c1"]
    assert [b["tool_use_id"] for b in tool_results] == ["c1"]          # G3 correlation holds
    assert tool_results[0]["content"] == "hi\n"                        # the tool RESULT is captured


def test_stream_parse_captures_text_and_usage():
    OA = _load_driver()
    events: list[dict] = []
    OA._parse_run_stream(_STREAM, "m", 0, events.append)
    texts = [b["text"] for e in events if e["type"] == "assistant"
             for b in e["message"]["content"] if b.get("type") == "text"]
    usage = [e["message"]["usage"] for e in events
             if e["type"] == "assistant" and e["message"].get("usage", {}).get("input_tokens")]
    assert texts == ["done"]
    assert usage and usage[0]["input_tokens"] == 100 and usage[0]["cache_read_input_tokens"] == 3


def test_empty_stream_is_safe():
    OA = _load_driver()
    events: list[dict] = []
    n = OA._parse_run_stream("", "m", 0, events.append)
    assert n == 0                                                      # falls back to `export` in run_round
