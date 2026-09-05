"""Every driver's transcript must carry a REAL arrival time per event.

An activity-share plot over wall time ("how much of the run went to thinking, reading,
writing code, shell, waiting?") is only honest if each transcript event knows WHEN it
happened. Before this suite, exactly one driver knew: the codex driver stamps
``arrived_at`` as each line comes off the stream. The claude path redirected the child's
stdout STRAIGHT into the transcript file (``subprocess.run(..., stdout=tf)``), so no
process ever observed a line and nothing could stamp it -- which is why the trajectory
plot laid a round's messages out by *weighted* time inside the round instead of by the
clock. opencode reconstructed its transcript from a whole captured file after the child
exited, with the same result.

What is pinned here:

* the stamping primitive appends -- it may not change, drop or reorder anything the
  transcript already carried, because the transcript is the GRADED artifact and
  ``conformance`` / ``experiment_tokens`` / the plots read it;
* the claude launch sites go through the streaming launcher, not a raw stdout redirect;
* the opencode driver stamps the events it reconstructs from its captured stream;
* the aet bridge keeps recording across a ``--resume`` and reports a turn count that
  matches the turns actually in the transcript.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from merlin.common import arrival_stamp as AS
from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


# --------------------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------------------

def test_stamp_appends_without_disturbing_existing_fields():
    """A reader that never heard of arrival times must be unaffected: same keys, same
    values, same ORDER, with the new field last."""
    original = '{"type": "assistant", "message": {"id": "m1"}, "session_id": "s"}'
    out = AS.stamp_line(original, "2026-09-04T00:00:00+00:00")
    before, after = json.loads(original), json.loads(out)
    assert list(after)[:-1] == list(before)                 # order preserved
    assert list(after)[-1] == AS.ARRIVED_AT                 # appended last
    for k, v in before.items():
        assert after[k] == v                                # nothing rewritten
    datetime.fromisoformat(after[AS.ARRIVED_AT])            # parseable, tz-aware
    assert datetime.fromisoformat(after[AS.ARRIVED_AT]).tzinfo is not None


def test_stamp_passes_non_json_through_verbatim_and_is_idempotent():
    """An unparseable line is evidence, not noise -- and a driver that already stamped its
    own events (codex) must stream through unchanged."""
    assert AS.stamp_line("not json at all", "T") == "not json at all"
    assert AS.stamp_line("[1, 2, 3]", "T") == "[1, 2, 3]"    # JSON, but not an object
    assert AS.stamp_line("", "T") == ""
    already = '{"arrived_at": "EARLIER", "type": "x"}'
    assert json.loads(AS.stamp_line(already, "LATER"))[AS.ARRIVED_AT] == "EARLIER"


def test_stream_stamped_records_real_per_event_times(tmp_path):
    """End to end against a real child process: every event gets a stamp, the stamps are
    non-decreasing, and a child that emits over time is not collapsed onto one instant."""
    tpath, epath = tmp_path / "t.jsonl", tmp_path / "e.log"
    raw = tmp_path / "t.raw.jsonl"
    script = (
        'printf \'{"type":"system","subtype":"init"}\\n\'; sleep 0.4; '
        'printf \'{"type":"assistant","message":{"id":"m1"}}\\n\'; sleep 0.4; '
        'printf \'garbage not json\\n\'; '
        'printf \'{"type":"result","subtype":"success","num_turns":2}\\n\''
    )
    rc = AS.stream_stamped(["bash", "-c", script], cwd=tmp_path, transcript=tpath,
                           stderr_path=epath, timeout=60, raw_path=raw)
    assert rc == 0
    lines = tpath.read_text().splitlines()
    assert len(lines) == 4
    assert lines[2] == "garbage not json"                    # verbatim, not dropped
    objs = [json.loads(x) for x in lines if x.startswith("{")]
    assert len(objs) == 3
    stamps = [datetime.fromisoformat(o[AS.ARRIVED_AT]) for o in objs]
    assert stamps == sorted(stamps)
    # The child spread its output over ~0.8s; a synthetic axis would have made them equal.
    assert (stamps[-1] - stamps[0]).total_seconds() >= 0.5
    # existing content is untouched
    assert objs[0]["type"] == "system" and objs[2]["num_turns"] == 2
    # the untouched bytes are on disk too
    assert raw.read_text().count("\n") == 4
    assert AS.ARRIVED_AT not in raw.read_text()


def test_stream_stamped_kills_the_group_and_keeps_what_arrived(tmp_path):
    """A timed-out round still leaves its stamped evidence, and the whole process group dies."""
    tpath, epath = tmp_path / "t.jsonl", tmp_path / "e.log"
    script = 'printf \'{"type":"assistant"}\\n\'; sleep 300'
    with pytest.raises(subprocess.TimeoutExpired):
        AS.stream_stamped(["bash", "-c", script], cwd=tmp_path, transcript=tpath,
                          stderr_path=epath, timeout=2)
    objs = [json.loads(x) for x in tpath.read_text().splitlines() if x.strip()]
    assert len(objs) == 1 and AS.ARRIVED_AT in objs[0]


# --------------------------------------------------------------------------------------
# the claude launch sites
# --------------------------------------------------------------------------------------

def test_claude_launch_sites_stream_instead_of_redirecting():
    """The two claude launches (the round, and the FINALIZE turn) must route through the
    streaming launcher. A raw ``stdout=<file>`` redirect is what made the time axis
    synthetic, so its absence is the property worth gating."""
    import inspect

    import run_baseline_qa_loop as L

    for fn in (L.launch_agent, L.finalize_report):
        src = inspect.getsource(fn)
        assert "AS.stream_stamped(" in src, f"{fn.__name__} does not stream its transcript"
        assert "stdout=tf" not in src, f"{fn.__name__} still redirects stdout into the transcript"


def test_one_shot_experiment_and_muon_loop_stream_too():
    """The other two claude launchers in the repo share the one convention."""
    import inspect

    import run_agent_experiment as RX
    sys.path.insert(0, str(merlin_dir() / "experiments/muon_perf_bench_v0/scripts"))
    import run_muon_qa_loop as M

    assert "stdout=tf" not in inspect.getsource(RX.main)
    assert "AS.stream_stamped(" in inspect.getsource(RX.main)
    assert "AS.stream_stamped(" in inspect.getsource(M.launch_agent)


def test_the_raw_stdout_tee_is_not_double_counted_as_a_transcript(tmp_path):
    """The launcher tees the child's untouched bytes next to the transcript, so a
    re-serialisation bug or a kill still leaves the original stream on disk. That sibling
    must not be read a second time as if it were another round: any consumer globbing
    ``rounds/*.jsonl`` would count every tool call twice."""
    import abc_status as ABC

    rounds = tmp_path / "rounds"
    rounds.mkdir()
    event = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash",
         "input": {"command": f"{ABC.GENS[0]} --target x"}}]}}) + "\n"
    (rounds / "round_00.transcript.jsonl").write_text(event)
    (rounds / "round_00.stream.raw.jsonl").write_text(event)     # the untouched tee
    assert ABC._circt_gen_count(tmp_path) == 1


# --------------------------------------------------------------------------------------
# the opencode driver
# --------------------------------------------------------------------------------------

def test_opencode_stream_events_carry_their_line_arrival():
    """Each reconstructed event is stamped with the arrival of the STREAM LINE it came
    from -- not with the moment the post-hoc parse happened to run."""
    import opencode_agent as OA

    stream = "\n".join([
        '{"part": {"type": "text", "id": "t1", "text": "hello"}}',
        '{"part": {"type": "tool", "callID": "c1", "tool": "bash",'
        ' "state": {"input": {}, "output": "done"}}}',
        '{"part": {"type": "step-finish", "tokens": {"input": 11, "output": 3}}}',
    ])
    stamps = ["2026-09-04T00:00:01+00:00", "2026-09-04T00:00:02+00:00",
              "2026-09-04T00:00:03+00:00"]
    events: list[dict] = []
    n_tools = OA._parse_run_stream(stream, "prov/m", 0, events.append, stamps=stamps)
    assert n_tools == 1
    assert [e[AS.ARRIVED_AT] for e in events] == [stamps[0], stamps[1], stamps[1], stamps[2]]
    # the shape the readers already consume is unchanged
    assert events[0]["message"]["content"][0]["type"] == "text"
    assert events[-1]["message"]["usage"]["input_tokens"] == 11


def test_opencode_capture_fills_arrival_stamps_from_the_tailed_file(tmp_path):
    """opencode's stream goes to a FILE (it truncates a pipe at 64 KiB), so the arrival
    times come from tailing that file while the child runs. A stamp per line, in order."""
    import opencode_agent as OA

    stamps: list[str] = []
    script = ('printf \'{"part":{"type":"text","id":"a","text":"1"}}\\n\'; sleep 1.5; '
              'printf \'{"part":{"type":"text","id":"b","text":"2"}}\\n\'')
    code, stdout, _err = OA._capture(["bash", "-c", script], dict(os.environ), 60,
                                     str(tmp_path), stall_seconds=0, stamps=stamps)
    assert code == 0
    assert len(stamps) == len(stdout.split("\n")) - 1 == 2
    t0, t1 = (datetime.fromisoformat(s) for s in stamps)
    assert t1 > t0                       # the two lines did NOT arrive at the same instant


def test_opencode_export_salvage_uses_opencodes_own_message_times():
    """The export fallback runs when the live stream yielded nothing, so there is no
    observed arrival. It takes opencode's recorded message time rather than collapsing the
    salvaged round onto the instant the salvage ran -- and records nothing when opencode
    recorded nothing."""
    import opencode_agent as OA

    assert OA._msg_arrived_at({"time": {"created": 1756000000000}}).startswith("2025-")
    assert OA._msg_arrived_at({"time": {}}) is None
    assert OA._msg_arrived_at({}) is None

    events: list[dict] = []
    OA._export_to_transcript(
        {"messages": [{"info": {"role": "assistant", "id": "m1", "tokens": {"input": 1},
                                "time": {"completed": 1756000000000}},
                       "parts": [{"type": "text", "text": "x"}]}]},
        "prov/m", 0, events.append)
    assert events and events[0][AS.ARRIVED_AT].startswith("2025-")


# --------------------------------------------------------------------------------------
# the aet bridge: sticky across --resume, and a truthful turn count
# --------------------------------------------------------------------------------------

def test_aet_sink_stays_on_for_a_run_that_is_already_being_recorded(tmp_path, monkeypatch):
    """MEASURED failure this pins: the atlas run's first session recorded, and two later
    ``--resume`` launches -- started from a shell without the env var -- recorded nothing.
    The RUN, not one process's environment, remembers that it is being recorded."""
    from merlin.targetgen import aet_bridge as AB

    monkeypatch.delenv("MERLIN_AET_SINK", raising=False)
    assert AB.aet_sink_enabled() is False
    assert AB.aet_sink_enabled(tmp_path) is False            # nothing recorded yet
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "metrics.jsonl").write_text('{"name": "x", "value": 1}\n')
    assert AB.aet_sink_enabled(tmp_path) is True             # a resumed session keeps going
    monkeypatch.setenv("MERLIN_AET_SINK", "1")
    assert AB.aet_sink_enabled() is True                     # the original opt-in still works


def _transcript_without_a_cli_turn_count(path: Path, n_turns: int) -> None:
    """The exact shape the codex driver writes: assistant events, then a terminating
    ``result`` that carries NO ``num_turns``."""
    with open(path, "w") as f:
        for i in range(n_turns):
            f.write(json.dumps({
                "type": "assistant",
                "message": {"id": f"codex_0_{i}", "model": "gpt-5.6-sol",
                            "usage": {"input_tokens": 10, "output_tokens": 5},
                            "content": [{"type": "text", "text": "work"}]},
                "arrived_at": f"2026-09-04T00:00:{i:02d}+00:00"}) + "\n")
        f.write(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                            "result": "done"}) + "\n")


def test_num_turns_reflects_the_turns_actually_in_the_transcript(tmp_path):
    """MEASURED failure this pins: the atlas run's final record said
    ``aet.agent.num_turns: 0`` for a transcript holding 794 assistant turns, because the
    driver's terminating ``result`` event carries no ``num_turns`` and aet reads
    ``int(None or 0)`` while skipping its own no-result-event fallback."""
    from merlin.targetgen import aet_bridge as AB

    _transcript_without_a_cli_turn_count(tmp_path / "transcript.jsonl", 7)
    ok = AB.emit_to_aet(run_dir=tmp_path, run_id="r0", method="arm", model="gpt-5.6-sol",
                        target="t", save_trajectory=False)
    if not ok:
        pytest.skip("aet not installed in this environment")
    metrics = [json.loads(x) for x in
               (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines() if x.strip()]
    turns = [m["value"] for m in metrics if m["name"] == "aet.agent.num_turns"]
    observed = [m["value"] for m in metrics if m["name"] == "aet.agent.assistant_turns"]
    assert turns == [7], f"turn count did not reflect the transcript: {turns}"
    assert observed == [7]


def test_an_authoritative_cli_turn_count_is_never_discarded(tmp_path):
    """The claude CLI reports its own ``num_turns`` and it counts more than the assistant
    events (it includes the user turns). That larger, authoritative figure must survive."""
    from merlin.targetgen import aet_bridge as AB

    with open(tmp_path / "transcript.jsonl", "w") as f:
        f.write(json.dumps({"type": "assistant", "message": {
            "id": "m1", "model": "claude", "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": []}}) + "\n")
        f.write(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                            "num_turns": 12, "total_cost_usd": 0.5}) + "\n")
    ok = AB.emit_to_aet(run_dir=tmp_path, run_id="r1", method="arm", model="claude",
                        target="t", save_trajectory=False)
    if not ok:
        pytest.skip("aet not installed in this environment")
    metrics = [json.loads(x) for x in
               (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert [m["value"] for m in metrics if m["name"] == "aet.agent.num_turns"] == [12]
