"""The activity timeline must refuse to invent a time axis.

The point of this module is a plot of where an agentic run's time went. The dangerous failure is not
a missing chart -- it is a plausible one built on stamps that do not mean what they appear to.
Measured on a real atlas round: 169 min of tool time inside 43.6 min of wall, because eight calls
shared one end stamp. That would have rendered as a perfectly convincing figure.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.agent_trace import (ACTIVITIES, BASH, READING, THINKING, TOOL_WAIT, WRITING,
                                classify, timeline)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _use(tid: str, at: str, cmd: str = "ls", name: str = "Bash") -> dict:
    return {"type": "assistant", "arrived_at": at, "message": {"content": [
        {"type": "tool_use", "id": tid, "name": name, "input": {"command": cmd}}]}}


def _res(tid: str, at: str) -> dict:
    return {"type": "user", "arrived_at": at, "message": {"content": [
        {"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}}


_T = "2026-09-05T00:%02d:%02d+00:00"


def test_a_clean_transcript_yields_a_measured_axis(tmp_path):
    p = _write(tmp_path / "t.jsonl", [
        _use("a", _T % (0, 0)), _res("a", _T % (0, 2)),
        _use("b", _T % (0, 10)), _res("b", _T % (0, 11)),
    ])
    tl = timeline(p)
    assert tl.measured and tl.basis == "wall_clock"
    assert tl.totals()[THINKING] > 0, "the gap between a result and the next call is thinking"


def test_background_concurrency_is_accepted_not_mistaken_for_a_flush(tmp_path):
    """An agent may background a long build and keep working under it.

    Measured on a real round: a 986 s call with SIX further calls opening and closing strictly inside
    it, every end stamp distinct. An earlier version of this module summed span durations, called that
    impossible, and refused a perfectly good transcript. Occupancy is the UNION of intervals."""
    p = _write(tmp_path / "t.jsonl", [
        _use("long", _T % (0, 0)),
        _use("q1", _T % (1, 0)), _res("q1", _T % (1, 10)),
        _use("q2", _T % (2, 0)), _res("q2", _T % (2, 10)),
        _res("long", _T % (9, 0)),
    ])
    tl = timeline(p)
    assert tl.measured, f"background concurrency was refused: {tl.reason}"


def test_a_pile_of_identical_end_stamps_is_still_refused(tmp_path):
    """The other half, kept separate: identical end stamps are a buffered flush, not concurrency."""
    rows = [_use(f"t{i}", _T % (i, 0)) for i in range(4)]
    rows += [_res(f"t{i}", _T % (9, 0)) for i in range(4)]      # all completing at one instant
    tl = timeline(_write(tmp_path / "t.jsonl", rows))
    assert not tl.measured and "share one end stamp" in tl.reason


def test_no_guard_here_is_unfalsifiable(tmp_path):
    """An 'occupancy exceeds wall' guard was tried and removed: wall IS the last span's end, so the
    union is bounded by it by construction and the check could never fire. Keeping it would have been
    a safety net that tested nothing. This pins that it stays gone."""
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python/merlin/agent_trace.py").read_text()
    assert "exceeds wall time" not in src, "an unfalsifiable guard came back"


def test_an_unstamped_transcript_is_refused_not_guessed(tmp_path):
    """claude/opencode transcripts carry no arrival stamps today. Charting them would mean
    inventing an axis -- the existing figure does exactly that and says so in its docstring."""
    p = _write(tmp_path / "t.jsonl", [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}},
    ])
    tl = timeline(p)
    assert not tl.measured and tl.basis == "unstamped"
    assert "no measured time axis" in tl.reason


def test_share_bins_sum_to_one_where_anything_ran(tmp_path):
    p = _write(tmp_path / "t.jsonl", [
        _use("a", _T % (0, 0)), _res("a", _T % (0, 30)),
        _use("b", _T % (1, 0)), _res("b", _T % (1, 5)),
    ])
    tl = timeline(p)
    _, share = tl.share(bins=10)
    for b in range(10):
        total = sum(share[a][b] for a in ACTIVITIES)
        assert total in (0.0,) or abs(total - 1.0) < 1e-9, f"bin {b} sums to {total}"


def test_a_long_call_is_waiting_and_a_short_one_is_shell():
    """Split by MEASURED duration, never by naming simulators -- a new target's toolchain must land
    in the right band with no edit here."""
    assert classify("Bash", "./whatever", 60.0) == TOOL_WAIT
    assert classify("Bash", "make -j", 1.0) == BASH
    assert classify("Bash", "cat foo", 1.0) == READING
    assert classify("Bash", "/bin/bash -lc 'grep x y'", 1.0) == READING
    assert classify("Write", "", 0.0) == WRITING
    assert classify("Edit", "", 0.0) == WRITING


def test_classification_names_no_target_and_no_simulator():
    from merlin.common.paths import merlin_dir
    text = (merlin_dir() / "python/merlin/agent_trace.py").read_text().lower()
    for banned in ("gemmini", "atlas", "radiance", "saturn", "verilator", "spike", "gsim", "circt"):
        assert banned not in text, f"{banned!r} would make this need editing per target"
