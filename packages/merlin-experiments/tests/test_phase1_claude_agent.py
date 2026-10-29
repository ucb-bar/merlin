"""The Claude CLI author gets the Codex author's kickoff/continuation bytes and a resume loop."""
import re
import subprocess
from pathlib import Path

from merlin_experiments.phase1.providers import claude_agent, codex_agent
from merlin_experiments.phase1.providers.authoring_prompt import authoring_prompt


def test_kickoff_and_continuation_are_the_codex_bytes():
    assert claude_agent.kickoff_prompt(None) == authoring_prompt()
    assert claude_agent.kickoff_prompt(6) == codex_agent.delegation_user_prompt(authoring_prompt(), 6)
    assert claude_agent.continuation_prompt(6) == codex_agent.delegation_user_prompt(
        codex_agent.DEFAULT_CONTINUATION_PROMPT, 6
    )
    assert "at most 6 helper sub-agent(s)" in claude_agent.kickoff_prompt(6)


def test_single_turn_script_sends_kickoff_as_argument_with_no_stdin():
    script = claude_agent.session_script(
        cli_model="m", effort="medium", ws=Path("/w"), limit=None, continuous=False, timeout=10, session_id="sid"
    )
    assert script.startswith("claude --print --model m --effort medium ")
    assert "--session-id sid" in script and script.endswith("< /dev/null") and "--resume" not in script


def test_continuous_script_resumes_same_session_until_wall_budget(tmp_path):
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 1; echo \"$@\" | cut -c1-400 >> %s/calls\nexit 0\n" % tmp_path)
    fake.chmod(0o755)
    script = claude_agent.session_script(
        cli_model="m", effort="high", ws=tmp_path, limit=6, continuous=True, timeout=123, session_id="sid",
        kickoff="KICK", continuation="CONT",
    )
    out = subprocess.run(
        ["bash", "-c", script], env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, capture_output=True, text=True
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    assert out.returncode == 0 and len(calls) >= 2
    assert "--session-id sid" in calls[0] and calls[0].endswith("KICK")
    assert all("--resume sid" in c and c.endswith("CONT") for c in calls[1:])
    assert re.search(r'"stopped_because":"wall budget spent"', out.stdout)


def test_failed_turn_stops_the_loop(tmp_path):
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    script = claude_agent.session_script(
        cli_model="m", effort="high", ws=tmp_path, limit=None, continuous=True, timeout=9999, session_id="sid",
        kickoff="K", continuation="C",
    )
    out = subprocess.run(["bash", "-c", script], env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, capture_output=True, text=True)
    assert out.returncode == 3 and "the turn did not complete" in out.stdout
