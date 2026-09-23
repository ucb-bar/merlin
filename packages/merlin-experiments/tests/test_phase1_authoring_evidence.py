"""Recovery selects evidence; it never decides treatment conformance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import recovery as R


def _events(path, *, work=False, terminal=None, real=True):
    events = [
        {
            "type": "assistant",
            "message": {
                "model": "model" if real else "<synthetic>",
                "content": [{"type": "tool_use", "name": "Bash", "id": "call", "input": {}}] if work else [],
            },
        }
    ]
    if terminal == "dead":
        events.append({"type": "result", "is_error": True, "result": "authentication failed"})
    elif terminal == "daily":
        events.append({"type": "result", "result": "429 daily quota limit"})
    elif terminal in {"five_hour", "weekly"}:
        events.append(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "status": "rejected",
                    "rateLimitType": "seven_day" if terminal == "weekly" else terminal,
                },
            }
        )
    path.write_text("\n".join(map(json.dumps, events)))
    return path


@pytest.mark.parametrize("terminal", ["dead", "daily", "five_hour", "weekly"])
@pytest.mark.parametrize("work", [False, True])
def test_late_terminal_error_only_disqualifies_zero_work(tmp_path, terminal, work):
    live = _events(tmp_path / "round_02.transcript.jsonl", work=True)
    failed = _events(tmp_path / "round_03.transcript.jsonl", work=work, terminal=terminal)
    before = {p: p.read_bytes() for p in (live, failed)}
    assert R.latest_live_authoring_transcript([failed, live]) == (failed if work else live)
    assert before == {p: p.read_bytes() for p in before}


def test_all_dead_absent_and_empty_have_no_authoring_evidence(tmp_path):
    empty = tmp_path / "empty"
    empty.touch()
    dead = _events(tmp_path / "dead", real=False)
    assert R.latest_live_authoring_transcript([empty, dead, tmp_path / "missing"]) is None
    assert R.latest_live_authoring_transcript([]) is None


def test_real_unproductive_turn_is_live_and_order_is_lexical(tmp_path):
    a = _events(tmp_path / "round_10.transcript.jsonl")
    b = _events(tmp_path / "round_2.transcript.jsonl")
    assert R.latest_live_authoring_transcript([a, b, a]) == b


@pytest.mark.parametrize("content", ["{broken", "null", "[]"])
def test_malformed_latest_evidence_does_not_fall_back_to_older_round(tmp_path, content):
    live = _events(tmp_path / "round_01.transcript.jsonl", work=True)
    bad = tmp_path / "round_02.transcript.jsonl"
    bad.write_text(content)
    with pytest.raises(ValueError):
        R.latest_live_authoring_transcript([live, bad])


def test_unreadable_latest_evidence_is_not_silently_replaced(tmp_path, monkeypatch):
    live = _events(tmp_path / "round_01.transcript.jsonl", work=True)
    bad = _events(tmp_path / "round_02.transcript.jsonl")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == bad:
            raise PermissionError("unreadable evidence")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(PermissionError, match="unreadable evidence"):
        R.latest_live_authoring_transcript([live, bad])


def test_historical_recovery_parser_still_ignores_partial_json(tmp_path):
    path = tmp_path / "partial"
    path.write_text('{"type":"result"}\n{unfinished')
    assert R.classify(path) == ""
    with pytest.raises(ValueError):
        R.latest_live_authoring_transcript([path])


@pytest.mark.parametrize("initially_live", [False, True])
def test_selector_classifies_the_single_validated_read_when_source_is_replaced(tmp_path, monkeypatch, initially_live):
    older = _events(tmp_path / "round_01.transcript.jsonl", work=True)
    latest = _events(tmp_path / "round_02.transcript.jsonl", work=initially_live, terminal="dead")
    replacement = _events(tmp_path / "replacement", work=not initially_live, terminal="dead").read_text()
    original = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        content = original(path, *args, **kwargs)
        reads.append(path)
        if path == latest:
            path.write_text(replacement)
        return content

    monkeypatch.setattr(Path, "read_text", read)
    assert R.latest_live_authoring_transcript([latest, older]) == (latest if initially_live else older)
    assert reads == ([latest] if initially_live else [latest, older])
    assert original(latest) == replacement


@pytest.mark.parametrize(
    "event,field",
    [
        ({"message": None}, "message"),
        ({"message": "not an object"}, "message"),
        ({"message": []}, "message"),
        ({"message": {"content": None}}, "message.content"),
        ({"message": {"content": "not a list"}}, "message.content"),
        ({"message": {"content": {}}}, "message.content"),
        ({"message": {"content": ["not an object"]}}, "message.content"),
        ({"rate_limit_info": None}, "rate_limit_info"),
        ({"rate_limit_info": "not an object"}, "rate_limit_info"),
    ],
)
def test_malformed_evidence_reports_the_field_without_changing_tolerant_parser(tmp_path, event, field):
    event = {"type": "rate_limit_event" if field == "rate_limit_info" else "assistant", **event}
    older = _events(tmp_path / "round_01.transcript.jsonl", work=True)
    latest = tmp_path / "round_02.transcript.jsonl"
    latest.write_text(json.dumps(event))
    assert list(R._iter_events(latest)) == [event]
    with pytest.raises(ValueError, match="authoring transcript") as error:
        R.latest_live_authoring_transcript([older, latest])
    assert field in str(error.value)


@pytest.mark.parametrize("event_type", ["user", "system"])
@pytest.mark.parametrize("message", ["plain prompt or telemetry", {"content": "plain prompt or telemetry"}])
def test_non_authoring_event_string_messages_remain_valid(tmp_path, event_type, message):
    latest = _events(tmp_path / "round_01.transcript.jsonl", work=True, terminal="weekly")
    events = [
        {"type": event_type, "message": message, "rate_limit_info": "not a rate-limit event"},
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "call", "content": "plain tool output"},
                ]
            },
        },
        {"type": "result", "result": "plain result text"},
    ]
    latest.write_text(latest.read_text() + "\n" + "\n".join(map(json.dumps, events)))
    assert R.latest_live_authoring_transcript([latest]) == latest


@pytest.mark.parametrize("terminal", ["dead", "daily", "weekly", "five_hour"])
def test_user_tool_use_shape_cannot_rescue_rejected_turn(tmp_path, terminal):
    path = _events(tmp_path / "round_01.transcript.jsonl", terminal=terminal)
    path.write_text(
        path.read_text()
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_use", "name": "Bash"}]},
            }
        )
    )
    assert R.latest_live_authoring_transcript([path]) is None
    # Preserve the historical public detector's broader tool-message behavior;
    # the selector explicitly opts into assistant-owned authoring work.
    assert R.agent_turn_dead(path) == (False, "")
