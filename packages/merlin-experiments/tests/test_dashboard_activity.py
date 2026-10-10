"""The agent-activity reader follows the provider's own stream, incrementally and without touching it.

A trailing line still being written is never parsed half-way; a second poll reads only the bytes added
since the first; a replaced or truncated stream is read again from the start; a run with no stream is
"not recorded".
"""

from __future__ import annotations

import json

from dashboard_fixtures import HOUR, T0, agent_round, codex_stream, command, iso
from merlin_experiments.tracking import activity, activity_html, records
from merlin_experiments.tracking.tail import JsonlTail, Tails


def test_the_stream_becomes_turns_calls_feed_and_tokens(tmp_path):
    codex_stream(tmp_path, 0, T0, agent_round(T0), step=60.0)
    act = activity.read(tmp_path, records.Inventory())
    assert len(act["turns"]) == 2 and all(t["status"] == "completed" for t in act["turns"])
    assert act["n_commands"] == 5 and act["failed_commands"] == 2
    assert act["command_counts"]["selfcheck"] == 2 and act["command_counts"]["build"] == 2
    assert act["command_counts"]["read"] == 1
    assert act["edits"] == {"submission/lib/Lower.cpp": 1, "submission/manifest.yaml": 1}
    round0 = act["rounds"][0]
    assert round0["output_tokens"] == 2330 and round0["usage_turns"] == 2
    assert act["now"]["last_message"]["text"] == "Working on the epilogue scale for cap_mm."
    assert act["now"]["todo"][1] == {"text": "fix the epilogue scale", "completed": False}
    assert act["running"] == []
    failed = [f for f in act["feed"] if f.get("error")]
    assert any("undefined reference to tile_mm" in (f.get("detail") or "") for f in failed)


def test_an_open_call_is_running_and_a_finished_turn_closes_what_it_left_open(tmp_path):
    codex_stream(tmp_path, 0, T0, agent_round(T0, running=True), step=60.0)
    act = activity.read(tmp_path, records.Inventory())
    assert [c["label"] for c in act["running"]] == ["selfcheck"]
    assert act["turns"][-1]["status"] == "running"
    page = activity_html.section(act, now=T0 + HOUR)
    assert "Running now" in page and "agent_selfcheck.py" in page and "fix the epilogue scale" in page
    events = [
        {"type": "turn.started"},
        *command("x", "sleep 100", exit_code=None),
        {"type": "turn.failed", "error": {"message": "upstream 400"}},
    ]
    codex_stream(tmp_path, 1, T0 + HOUR, events)
    act = activity.read(tmp_path, records.Inventory())
    orphan = next(c for c in act["calls"] if c.get("command") == "sleep 100")
    assert orphan["status"] == "no completion recorded" and orphan["end"] is not None
    assert any(f["kind"] == "turn failed" and "upstream 400" in f["text"] for f in act["feed"])


def test_a_partial_trailing_line_waits_for_its_newline(tmp_path):
    events = agent_round(T0)
    complete = events[:-1]
    last = json.dumps({"seq": len(events), "arrived_at": iso(T0 + 999), "event": events[-1]})
    path = codex_stream(tmp_path, 0, T0, complete, step=60.0, partial_tail=last[:25])
    tails = Tails()
    first = activity.read(tmp_path, records.Inventory(), tails)
    assert first["turns"][-1]["status"] == "running"  # the turn's completion is still being written
    tail = tails.files[str(path)]
    assert tail.bad == 0 and tail.offset == path.stat().st_size - 25
    with path.open("a", encoding="utf-8") as handle:
        handle.write(last[25:] + "\n")
    second = activity.read(tmp_path, records.Inventory(), tails)
    assert second["turns"][-1]["status"] == "completed" and tail.offset == path.stat().st_size


def test_polls_read_only_the_bytes_added_since_the_last_one(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps({"n": i, "pad": "x" * 200}) + "\n" for i in range(1000)), encoding="utf-8")
    tail, seen = JsonlTail(path), []
    assert tail.poll(seen.append) == 1000 and tail.bytes_read == path.stat().st_size
    size = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"n": 1000}) + "\n")
    assert tail.poll(seen.append) == 1
    assert tail.bytes_read == path.stat().st_size and tail.bytes_read - size == len(json.dumps({"n": 1000})) + 1
    assert tail.poll(seen.append) == 0 and seen[-1] == {"n": 1000}


def test_a_replaced_or_truncated_stream_is_read_again_and_a_vanished_one_is_absent(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"n": 1}\n{"n": 2}\n', encoding="utf-8")
    tail, seen, resets = JsonlTail(path), [], []
    tail.poll(seen.append, on_reset=lambda: resets.append(1))
    replacement = tmp_path / "new.jsonl"
    replacement.write_text('{"n": 9}\n', encoding="utf-8")
    replacement.replace(path)
    tail.poll(seen.append, on_reset=lambda: resets.append(1))
    assert resets == [1] and seen[-1] == {"n": 9}
    path.unlink()
    assert tail.poll(seen.append) == 0 and tail.present is False


def test_a_rewritten_stream_rebuilds_the_activity_state(tmp_path):
    codex_stream(tmp_path, 0, T0, agent_round(T0), step=60.0)
    tails = Tails()
    assert activity.read(tmp_path, records.Inventory(), tails)["n_commands"] == 5
    codex_stream(tmp_path, 0, T0, [{"type": "turn.started"}, *command("a", "pytest -q", exit_code=0)])
    again = activity.read(tmp_path, records.Inventory(), tails)
    assert again["n_commands"] == 1 and again["command_counts"]["test"] == 1


def test_no_stream_is_not_recorded(tmp_path):
    inventory = records.Inventory()
    assert activity.read(tmp_path, inventory) is None
    assert inventory.rows[0]["state"] == "absent"
    assert "not recorded" in activity_html.section(None, now=T0)


def test_command_labels_follow_the_words_a_command_runs():
    assert activity.unwrap("/bin/bash -lc 'cd ws && python3 agent_selfcheck.py --sim spike'") == (
        "cd ws && python3 agent_selfcheck.py --sim spike"
    )
    assert activity.label("python3 agent_selfcheck.py --sim spike") == activity.SELFCHECK
    assert activity.label("python3 await_verdict.py") == activity.WAIT
    assert activity.label("cd b && ninja -j8") == activity.BUILD
    assert activity.label("rg -n tile lib/") == activity.READ
    assert activity.label("python3 -m pytest -q") == activity.TEST
    assert activity.label("echo hi") == activity.OTHER
