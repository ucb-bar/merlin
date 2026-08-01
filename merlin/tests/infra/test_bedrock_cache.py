"""Offline proof that the converse driver caches like Claude Code — BEFORE any Bedrock spend.

We drive the REAL ``bedrock_agent.run_round`` / ``_run_subagent`` with a fake ``boto3`` client that records
every ``converse`` request and returns scripted responses. That exercises the actual cachePoint placement
(no mocking of the code under test), so we can assert:

  * the STATIC breakpoints land on system + tools + the first user message every call;
  * a single ROLLING breakpoint moves to the end of the growing conversation (Claude-Code style), with the
    total never exceeding Bedrock's 4-breakpoint limit;
  * the delegate sub-agent caches its own system+tools+first-message prefix (no longer re-shipped);
  * a model that REJECTS cachePoint self-corrects to an uncached request instead of failing the round;
  * the ``cacheReadInputTokens`` the model returns propagates into the transcript the grader accounts.

None of this calls AWS. The live counterpart (a few cents) is a separate opt-in micro-probe.
"""
from __future__ import annotations

import copy
import json
import sys
import time
import types
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


@pytest.fixture()
def bedrock_agent():
    pytest.importorskip("boto3")
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import bedrock_agent  # noqa: PLC0415 — loaded off the import-isolated harness path
    return bedrock_agent


def _cp(blocks) -> int:
    """Number of cachePoint markers in a content/system/tools block list."""
    return sum(1 for b in blocks if isinstance(b, dict) and "cachePoint" in b)


class _FakeClient:
    """Records each converse request (deep-copied, since run_round mutates the messages list in place) and
    replays a scripted list of responses/exceptions."""

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.calls: list[dict] = []

    def converse(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        item = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


def _assistant(content, stop, *, cache_read=0, cache_write=0):
    return {"usage": {"inputTokens": 100, "outputTokens": 10,
                      "cacheReadInputTokens": cache_read, "cacheWriteInputTokens": cache_write},
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop}


def _tool_use(name, inp):
    return _assistant([{"toolUse": {"toolUseId": "t1", "name": name, "input": inp}}], "tool_use")


def _install_fake(monkeypatch, bedrock_agent, script):
    import boto3  # noqa: PLC0415
    fc = _FakeClient(script)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fc)
    return fc


def _run(bedrock_agent, tmp_path, monkeypatch, script, **kw):
    fc = _install_fake(monkeypatch, bedrock_agent, script)
    ws = tmp_path / "ws"; ws.mkdir()
    run_dir = tmp_path / "run"; run_dir.mkdir()
    te = types.SimpleNamespace(target="atlas")
    rc, tpath = bedrock_agent.run_round(
        ws, run_dir, "glm5", {}, te, "none", 1, timeout=100, max_iters=5, **kw)
    events = [json.loads(l) for l in Path(tpath).read_text().splitlines()]
    return fc, rc, events


def test_static_and_rolling_cachepoints(bedrock_agent, tmp_path, monkeypatch):
    # Round: iter0 writes a file (in-process, no sandbox), iter1 stops. Two converse calls → the rolling
    # breakpoint must appear on the SECOND call once the conversation has grown.
    fc, rc, events = _run(bedrock_agent, tmp_path, monkeypatch, [
        _tool_use("write_file", {"path": "note.txt", "content": "hi"}),
        _assistant([{"text": "done"}], "end_turn", cache_read=150),
    ])
    assert len(fc.calls) == 2

    c0 = fc.calls[0]
    assert _cp(c0["system"]) == 1                      # static: system
    assert _cp(c0["toolConfig"]["tools"]) == 1         # static: tools
    assert len(c0["messages"]) == 1                    # only the first user turn yet
    assert _cp(c0["messages"][0]["content"]) == 1      # static: first message
    assert sum(_cp(m["content"]) for m in c0["messages"]) == 1   # no rolling on iter0

    c1 = fc.calls[1]
    assert len(c1["messages"]) == 3                    # msg0 + assistant + tool_results
    assert _cp(c1["messages"][0]["content"]) == 1      # static msg0 preserved
    assert _cp(c1["messages"][-1]["content"]) == 1     # rolling moved to the newest turn
    msg_bps = sum(_cp(m["content"]) for m in c1["messages"])
    assert msg_bps == 2                                # exactly one static + one rolling
    total_bps = _cp(c1["system"]) + _cp(c1["toolConfig"]["tools"]) + msg_bps
    assert total_bps == 4                              # at Bedrock's ceiling, never above

    # the model's cacheReadInputTokens reached the transcript the grader accounts
    reads = [e["message"]["usage"]["cache_read_input_tokens"]
             for e in events if e.get("type") == "assistant"]
    assert 150 in reads


def test_cache_unsupported_self_corrects_to_uncached(bedrock_agent, tmp_path, monkeypatch):
    # First converse raises a cachePoint-rejection; run_round must strip cachePoints and retry uncached
    # within the same round rather than erroring out.
    fc, rc, events = _run(bedrock_agent, tmp_path, monkeypatch, [
        Exception("ValidationException: this model does not support cachePoint"),
        _assistant([{"text": "done"}], "end_turn"),
    ])
    assert rc == 0
    assert any(e.get("subtype") == "cache_disabled" for e in events)
    retry = fc.calls[-1]                               # the uncached retry
    assert _cp(retry["system"]) == 0
    assert _cp(retry["toolConfig"]["tools"]) == 0
    assert sum(_cp(m["content"]) for m in retry["messages"]) == 0


def test_delegate_subagent_caches_its_prefix(bedrock_agent, tmp_path, monkeypatch):
    fc = _FakeClient([_assistant([{"text": "sub done"}], "end_turn")])
    ws = tmp_path / "ws"; ws.mkdir()
    te = types.SimpleNamespace(target="atlas")
    out = bedrock_agent._run_subagent(
        fc, "amazon.nova-lite-v1:0", ws, te, {}, "none",
        "scaffold a file", "", lambda o: None, 1, "0_deleg", 30, time.time() + 100)
    assert "sub done" in out
    c0 = fc.calls[0]
    assert _cp(c0["system"]) == 1                      # sub-agent now caches its own header...
    assert _cp(c0["toolConfig"]["tools"]) == 1
    assert _cp(c0["messages"][0]["content"]) == 1      # ...and its first message (was uncached before)


def test_delegate_subagent_self_corrects_to_uncached(bedrock_agent, tmp_path, monkeypatch):
    fc = _FakeClient([
        Exception("ValidationException: cachePoint not supported"),
        _assistant([{"text": "sub done"}], "end_turn"),
    ])
    ws = tmp_path / "ws"; ws.mkdir()
    te = types.SimpleNamespace(target="atlas")
    out = bedrock_agent._run_subagent(
        fc, "amazon.nova-lite-v1:0", ws, te, {}, "none",
        "scaffold a file", "", lambda o: None, 1, "0_deleg", 30, time.time() + 100)
    assert "sub done" in out                           # a cache rejection is NOT a delegate failure
    retry = fc.calls[-1]
    assert _cp(retry["system"]) == 0
    assert sum(_cp(m["content"]) for m in retry["messages"]) == 0
