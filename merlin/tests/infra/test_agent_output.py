"""Structured agent-output helper — regex-free JSON/code extraction + constrained calls."""
from __future__ import annotations

import pytest

from merlin.common.agent_output import (StructuredOutputError, extract_code_block, locate_json,
                                        parse_json, strip_code_fence, structured_agent_call)


def test_strip_code_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence("```\nhello\n```") == "hello"
    assert strip_code_fence("no fence here") == "no fence here"
    # opener with no closer: keep the rest after the opener line
    assert strip_code_fence("```json\nbody-only") == "body-only"


def test_locate_json_first_balanced_not_greedy():
    # a greedy .*  would span both objects; the balanced scan stops at the first close
    assert locate_json('prefix {"a": 1} middle {"b": 2}') == '{"a": 1}'
    # brace inside a string literal must not unbalance the scan
    assert locate_json('{"k": "}"}') == '{"k": "}"}'
    assert locate_json("[1, 2, [3]]") == "[1, 2, [3]]"
    assert locate_json("no json") is None


def test_parse_json_variants():
    assert parse_json('```json\n{"is_exemplary": true}\n```') == {"is_exemplary": True}
    assert parse_json('Here it is: [1, 2, 3] done', default=None) == [1, 2, 3]
    assert parse_json("garbage", default={}) == {}
    assert parse_json(None, default={}) == {}
    with pytest.raises(StructuredOutputError):
        parse_json("no json at all")


def test_parse_json_schema_validation():
    schema = {"type": "object", "required": ["k"], "properties": {"k": {"type": "integer"}}}
    assert parse_json('{"k": 5}', schema=schema) == {"k": 5}
    # wrong type -> tolerant default when given
    assert parse_json('{"k": "x"}', schema=schema, default=None) is None


def test_extract_code_block_prefers_lang():
    text = "intro\n```text\nnope\n```\nthen\n```python\ncode_here\n```\n"
    assert extract_code_block(text, "python") == "code_here"
    # no matching lang -> first block
    assert extract_code_block("```\nfirst\n```", "python") == "first"
    assert extract_code_block("no block", default="") == ""
    with pytest.raises(StructuredOutputError):
        extract_code_block("no block")


def test_structured_agent_call_retry_then_succeed():
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    replies = iter(["not json", '{"ok": true}'])
    calls = []

    def runner(prompt):
        calls.append(prompt)
        return next(replies)

    out = structured_agent_call(runner, "do it", schema=schema, retries=1)
    assert out == {"ok": True}
    assert len(calls) == 2
    # the schema is injected into the turn (constrained output, not post-hoc scrape)
    assert "JSON" in calls[0]


def test_structured_agent_call_unavailable_and_exhausted():
    with pytest.raises(StructuredOutputError):
        structured_agent_call(lambda p: None, "x")
    with pytest.raises(StructuredOutputError):
        structured_agent_call(lambda p: "still not json", "x", retries=1)
