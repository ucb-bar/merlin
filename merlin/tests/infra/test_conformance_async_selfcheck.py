"""An async full check is credited only through its own completed broker response."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from merlin_experiments.phase1 import conformance as C


def calls():
    initial = C.ToolCall(
        "Bash",
        {"command": "python3 agent_selfcheck.py --capsules all"},
        "start",
        True,
        False,
        json.dumps({"status": "in_progress", "request_id": "request1", "submission": "submission"}),
        start_event=1,
        end_event=2,
    )
    result = {
        "selfcheck_request_id": "request1",
        "n_capsules": 2,
        "n_passed": 1,
        "all_pass": False,
        "n_result_rows": 2,
        "per_capsule": [{"capsule": "a"}, {"capsule": "b"}],
        "submission_sha256": "a" * 64,
        "requested_submission_sha256": "b" * 64,
        "requested_at_unix_ns": 10,
        "grade_started_at_unix_ns": 11,
        "graded_at_unix_ns": 12,
    }
    attached = C.ToolCall(
        "Bash",
        {"command": "/bin/bash -lc 'python3 agent_selfcheck.py --attach request1 --out old.json'"},
        "attach",
        True,
        False,
        "[self-check] 2/2 finished\n" + json.dumps(result),
        start_event=3,
        end_event=4,
    )
    return initial, attached, result


def test_completed_nonzero_attach_counts_workflow_but_not_a_pass():
    first, last, _ = calls()
    assert C._full_selfcheck([first, last]) is True
    assert not first.succeeded and not last.succeeded


def test_complete_report_may_be_followed_by_the_clients_informational_snapshot_note():
    first, last, _ = calls()
    assert C._full_selfcheck([first, replace(last, result_text=last.result_text + '\n{"note":"submission changed"}')])


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_request",
        "wrong_cli",
        "subset",
        "missing_request",
        "pending",
        "partial",
        "missing_rows",
        "wrong_source",
        "bad_time",
        "duplicate_capsule",
        "duplicate_id",
        "reversed",
        "overlap",
        "read_file",
        "echo_request",
        "echo_attach",
        "outer_echo",
        "other_report",
        "missing_result",
        "truncated_report",
        "other_trailer",
        "equal_time",
        "newline_request",
        "newline_attach",
        "substitution",
        "backtick",
    ],
)
def test_async_completion_cannot_be_borrowed_or_fabricated(damage):
    first, last, result = calls()
    command = last.input["command"]
    if damage == "wrong_request":
        result["selfcheck_request_id"] = "other"
    elif damage == "wrong_cli":
        command = command.replace("--attach request1", "--attach other")
    elif damage == "subset":
        first = replace(first, input={"command": "python3 agent_selfcheck.py --capsules a"})
    elif damage == "missing_request":
        return_assert = C._full_selfcheck([last])
        assert return_assert is False
        return
    elif damage == "pending":
        result["status"] = "in_progress"
    elif damage == "partial":
        result["partial"] = True
    elif damage == "missing_rows":
        result["per_capsule"] = result["per_capsule"][:1]
    elif damage == "wrong_source":
        result["requested_submission_sha256"] = None
    elif damage == "bad_time":
        result["graded_at_unix_ns"] = 1
    elif damage == "equal_time":
        result["graded_at_unix_ns"] = result["grade_started_at_unix_ns"]
    elif damage == "newline_request":
        first = replace(first, input={"command": first.input["command"] + "\nprintf fabricated"})
    elif damage == "newline_attach":
        command = "python3 agent_selfcheck.py --bad --attach request1\nprintf fabricated"
    elif damage == "substitution":
        command = "python3 agent_selfcheck.py --attach $(printf request1)"
    elif damage == "backtick":
        command = "python3 agent_selfcheck.py --attach `printf request1`"
    elif damage == "duplicate_capsule":
        result["per_capsule"] = [{"capsule": "a"}, {"capsule": "a"}]
    elif damage == "duplicate_id":
        assert C._full_selfcheck([first, first, last]) is False
        return
    elif damage == "reversed":
        assert C._full_selfcheck([last, first]) is False
        return
    elif damage == "overlap":
        first = replace(first, end_event=last.start_event)
    elif damage == "read_file":
        command = "cat old.json"
    elif damage == "echo_request":
        first = replace(first, input={"command": "echo python3 agent_selfcheck.py --capsules all"})
    elif damage == "echo_attach":
        command = "echo python3 agent_selfcheck.py --attach request1 --capsules all"
    elif damage == "outer_echo":
        command += " && echo fabricated"
    elif damage == "other_report":
        result = {"note": "n_capsules n_passed selfcheck_request_id request1"}
    elif damage == "missing_result":
        last = replace(last, result_present=False)
    elif damage in {"truncated_report", "other_trailer"}:
        text = json.dumps(result)
        text = (
            text[:20] + "\n...[276603 chars elided by the harness]...\n" + text[-30:]
            if damage == "truncated_report"
            else text + '\n{"n_capsules":2}'
        )
        assert C._full_selfcheck([first, replace(last, result_text=text)]) is False
        return
    last = replace(last, input={"command": command}, result_text=json.dumps(result))
    assert C._full_selfcheck([first, last]) is False


def test_serial_transcript_pairs_request_completion_without_reading_current_output_file(tmp_path):
    first, last, _ = calls()
    events = [{"type": "system", "subtype": "init", "driver": "codex"}]
    for call in (first, last):
        events.extend(
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": call.tool_use_id, "name": call.name, "input": call.input}
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.tool_use_id,
                                "is_error": True,
                                "content": call.result_text,
                            }
                        ]
                    },
                },
            ]
        )
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in events))
    assert C._full_selfcheck(C._tool_calls(path)) is True
