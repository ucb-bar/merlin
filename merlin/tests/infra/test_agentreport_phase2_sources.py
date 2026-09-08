"""Where the perf lane's telemetry actually lives, versus where the reader looked for it.

A phase-1 stage writes ``control/``; a global phase-2 campaign writes ``global_control/`` with
four-digit round indices. The reader looked only in ``control/``, so every global campaign's entire
tool cost reported UNAVAILABLE while its receipts sat on disk -- and an availability report that
cannot find its own evidence is worse than no report, because "UNAVAILABLE" reads as "the lane made
no brokered calls".

Measured on the v15 run once the path was fixed: 15 brokered calls and 3,115 seconds of tool time,
including three `qualify-changed-region` calls that had each been refused.
"""
from __future__ import annotations

import json

from merlin.agentreport import phase2


def _receipts(stage, dirname, round_name, rows):
    d = stage / dirname / round_name
    d.mkdir(parents=True)
    (d / "receipts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


ROWS = [
    {"index": 0, "action": "analyze-whole-model", "elapsed_s": 240.5, "returncode": 0,
     "state": "complete"},
    {"index": 1, "action": "qualify-changed-region", "elapsed_s": 945.4, "returncode": 125,
     "state": "complete"},
]


class TestBrokerReceiptsAreFoundInBothLayouts:
    def test_a_phase1_stage_layout_is_read(self, tmp_path):
        _receipts(tmp_path, "control", "round_00", ROWS)
        calls = phase2.read_receipts(tmp_path)
        assert [c.action for c in calls] == ["analyze-whole-model", "qualify-changed-region"]

    def test_a_global_campaign_layout_is_read(self, tmp_path):
        """The one that was invisible: `global_control/round_0000/`, four-digit index."""
        _receipts(tmp_path, "global_control", "round_0000", ROWS)
        calls = phase2.read_receipts(tmp_path)
        assert [c.action for c in calls] == ["analyze-whole-model", "qualify-changed-region"]
        assert sum(c.elapsed_s for c in calls) == 1185.9

    def test_a_refused_call_keeps_its_return_code_so_its_cost_is_still_counted(self, tmp_path):
        """1,566 s of the v15 budget went into refused calls; a report must not drop them."""
        _receipts(tmp_path, "global_control", "round_0000", ROWS)
        refused = [c for c in phase2.read_receipts(tmp_path) if c.returncode != 0]
        assert len(refused) == 1 and refused[0].elapsed_s == 945.4

    def test_rounds_across_both_directories_are_read_together(self, tmp_path):
        _receipts(tmp_path, "control", "round_00", ROWS[:1])
        _receipts(tmp_path, "global_control", "round_0000", ROWS[1:])
        assert len(phase2.read_receipts(tmp_path)) == 2

    def test_a_stage_with_neither_directory_yields_no_calls_rather_than_raising(self, tmp_path):
        assert phase2.read_receipts(tmp_path) == []

    def test_the_declared_directories_are_the_two_layouts_that_exist(self):
        assert phase2.CONTROL_DIRS == ("control", "global_control")


class TestSpansStayUnavailableRatherThanBeingReconstructed:
    def test_the_reason_names_the_raw_stream_and_why_it_is_not_a_span_source(self, tmp_path):
        """Arrival stamps are when an event REACHED the reader, not when the tool ran.

        A plausible span built from the wrong clock is exactly the kind of number this package
        exists to refuse, so the raw stream is named as present-but-unusable rather than used.
        """
        agent = tmp_path / "agent"
        agent.mkdir()
        (agent / "events.00.raw.jsonl").write_text("{}\n", encoding="utf-8")
        (agent / "events.01.raw.jsonl").write_text("{}\n", encoding="utf-8")
        spanset, points = phase2.read_tool_spans(tmp_path)
        assert spanset.spans == [] and points == 0
        why = spanset.availability.get("spans").reason
        assert "2 raw driver event stream(s)" in why
        assert "arrival stamps" in why and "finalize_agent_telemetry" in why

    def test_with_no_raw_stream_either_the_reason_does_not_claim_one(self, tmp_path):
        spanset, _ = phase2.read_tool_spans(tmp_path)
        why = spanset.availability.get("spans").reason
        assert "raw driver event stream" not in why
