"""A round whose agent never RAN must not be graded, and must not report a conformance verdict.

Three failure classes, only two of which had guards. `_ratelimit.round_rejected` covers the five-hour
window (sleep and retry the same round) and `daily_limit_hit` the provider's daily token quota. Neither
covers a turn that never happened: a seat out of credits until a DATE, a model name the CLI cannot serve,
a failed auth.

Measured 2026-09-01, both on this bench:

  * the codex seat returned "You've hit your usage limit ... try again at Sep 6th, 2026 7:29 PM" with
    `content: []` and a 4.5 s turn, for three consecutive rounds;
  * a Bedrock inference-profile id handed to a subscription CLI returned "There's an issue with the
    selected model (us.anthropic.claude-opus-4-6-v1)" in 0 ms.

In BOTH cases the round went on to grade an unchanged submission and print `NOT CONFORMANT -- failing:
isa_tools_used, cca_used, rtl_facts_used, ...`, which reads as an agent that could not do the work. About
three and a half hours of wall-clock went to rounds where no agent ran. This is the third time in this
bench that a harness limitation has been reported as an agent defect, which is why it is now a test.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
RL = pytest.importorskip("_ratelimit")
LOOP = pytest.importorskip("run_baseline_qa_loop")


def _write(tmp_path, events):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


class TestTheTwoMeasuredFailures:
    def test_a_seat_out_of_credits_is_a_dead_turn(self, tmp_path):
        """The codex shape: an empty assistant turn, a failed-turn marker, a dated usage-limit error."""
        p = _write(tmp_path, [
            {"type": "system", "subtype": "init", "driver": "codex", "round": "4"},
            {"type": "assistant", "message": {"id": "codex_4_1", "model": "gpt-5.6-sol", "content": []},
             "codex_turn_failed": "True"},
            {"type": "result", "subtype": "error", "is_error": "True",
             "result": "You've hit your usage limit. Visit https://... to purchase more credits or "
                       "try again at Sep 6th, 2026 7:29 PM."},
        ])
        dead, why = RL.agent_turn_dead(p)
        assert dead, why
        assert "usage limit" in why

    def test_a_model_the_cli_cannot_serve_is_a_dead_turn(self, tmp_path):
        """The subscription/Bedrock-id shape: the CLI answers itself, with model '<synthetic>'."""
        p = _write(tmp_path, [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"model": "<synthetic>", "role": "assistant", "content": [
                {"type": "text", "text": "There's an issue with the selected model "
                                         "(us.anthropic.claude-opus-4-6-v1). It may not exist or you "
                                         "may not have access to it."}]}},
            {"type": "result", "is_error": True,
             "result": "There's an issue with the selected model (us.anthropic.claude-opus-4-6-v1)."},
        ])
        dead, why = RL.agent_turn_dead(p)
        assert dead, why
        assert "selected model" in why or "synthetic" in why


class TestItDoesNotStopARunThatIsMerelyUnproductive:
    def test_a_turn_that_used_tools_is_never_dead(self, tmp_path):
        """Even alongside an error: if the agent called tools, it ran, and the ledger judges the work."""
        p = _write(tmp_path, [
            {"type": "assistant", "message": {"model": "claude-opus-5", "content": [
                {"type": "tool_use", "name": "Bash", "id": "t1", "input": {}}]}},
            {"type": "result", "is_error": True, "result": "usage limit reached after the work"},
        ])
        assert RL.agent_turn_dead(p) == (False, "")

    def test_a_quiet_but_real_turn_is_unproductive_not_dead(self, tmp_path):
        """A real model replying without tool calls achieves nothing, which the stage ledger reports.
        Stopping the run for it would confuse 'did nothing' with 'could not run'."""
        p = _write(tmp_path, [
            {"type": "assistant", "message": {"model": "claude-opus-5", "content": [
                {"type": "text", "text": "I reviewed the ledger and have no change to make."}]}},
            {"type": "result", "is_error": False, "result": "ok"},
        ])
        dead, why = RL.agent_turn_dead(p)
        assert dead is False, why

    def test_an_empty_transcript_is_dead(self, tmp_path):
        """No assistant turn at all: the driver produced nothing, which cannot be graded either."""
        dead, why = RL.agent_turn_dead(_write(tmp_path, [{"type": "system", "subtype": "init"}]))
        assert dead and "no assistant turn" in why


class TestTheLoopActsOnIt:
    def test_the_loop_checks_before_grading_and_does_not_grade(self):
        src = (repo_root()
               / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py").read_text()
        i_check = src.index("RL.agent_turn_dead(tpath)")
        i_grade = src.index("verdict = qa_grade(ws, run_dir, rnd, a.no_oracle, a.qa_timeout)")
        assert i_check < i_grade, "the guard must precede grading, or the round grades a dead turn"
        assert "AGENT DID NOT RUN" in src, "the reason must be stated distinctly, not as a low score"
        # and it must checkpoint so --resume retries THIS round rather than skipping it
        seg = src[i_check:i_grade]
        assert "_checkpoint(rnd)" in seg and "break" in seg

    def test_dead_launch_does_not_replace_last_live_conformance_evidence(self, tmp_path):
        """A zero-turn round file is transport telemetry, not authoring evidence.

        This is the exact post-run failure observed in the Gemmini functional
        QA run: round 02 had tool evidence, then round 03 was a 0.19-second
        Codex launch failure.  Final conformance used the lexically latest
        file and incorrectly reported ``n_calls: 0``.
        """
        live = _write(tmp_path, [
            {"type": "assistant", "message": {"model": "gpt-5.6-sol", "content": [
                {"type": "tool_use", "name": "Bash", "id": "t1", "input": {}}]}},
            {"type": "result", "is_error": False, "result": "ok"},
        ])
        live = live.rename(tmp_path / "round_02.transcript.jsonl")
        dead = _write(tmp_path, [
            {"type": "system", "subtype": "init", "driver": "codex", "round": 3},
            {"type": "codex_summary", "turns_started": 0, "exit_code": 1},
            {"type": "result", "is_error": True, "result": "codex exited 1"},
        ]).rename(tmp_path / "round_03.transcript.jsonl")

        assert RL.agent_turn_dead(dead)[0] is True
        assert LOOP._latest_live_authoring_transcript([live, dead]) == live
