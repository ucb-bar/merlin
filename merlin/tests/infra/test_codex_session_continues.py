"""`codex exec` is one TURN, not one session — the continuous schedule must span the wall budget.

Measured on atlas arm-4 (2026-09-04): three runs in a row ended with most of the budget unspent, the
last after 83 minutes with 23890s of 28901s left, having emitted exactly ONE turn.started /
turn.completed pair. The harness read a completed turn as "the agent session ended", because
`keep_launching` allows one invocation per process — correct for a driver whose invocation IS a
session (claude --print), wrong for one whose invocation is a single turn.

These pin the fix and, more importantly, pin that it stays OFF for rounds and that every stop is
attributable.
"""
from __future__ import annotations
import shlex
import sys
from pathlib import Path

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))
import codex_agent as CA  # noqa: E402


def test_resume_keeps_every_flag_and_targets_the_thread():
    """A continuation that dropped --model or -o would silently be a different arm."""
    base = ["codex", "exec", "--json", "--model", "gpt-5.6-sol", "-C", "/ws", "-o", "/ws/f.txt"]
    out = CA._resume_cmd(base, "uuid-1", Path("/ws/p.txt"))
    assert out[:4] == ["codex", "exec", "resume", "uuid-1"]
    for flag in ("--json", "--model", "gpt-5.6-sol", "-C", "-o"):
        assert flag in out, f"{flag} lost on resume"


def test_resume_survives_the_real_nested_sandbox_argv():
    """The shape that actually ships, and the two bugs it exposed.

    `bash -c "bwrap ... bash -c 'export PATH=...; exec codex exec ...'"` nests the codex tokens inside
    a quoted string AND puts the shell's own `exec` BUILTIN before the binary. Anchoring on the first
    token spelled `exec` produced `exec resume ...`, which the live run reported as
    `bash: line 1: exec: resume: not found`. Splitting and re-quoting the inner string then swallowed
    the `;` separator, merging two commands into one."""
    import shlex
    inner = "export PATH=/x:/y; exec codex exec --json --model m -C /ws"
    out = CA._resume_cmd(["bash", "-c", f"bwrap --args 10 bash -c {shlex.quote(inner)}"],
                         "uuid-9", Path("/p"))
    assert "export PATH=/x:/y; exec codex exec resume uuid-9 --json" in out[2], out[2]
    assert out[2].count("resume") == 1


def test_a_command_with_no_codex_is_untouched():
    assert CA._resume_cmd(["bash", "-c", "echo hi"], "u", Path("/p")) == ["bash", "-c", "echo hi"]


def test_resume_stays_inside_the_sandbox():
    """Under bwrap the argv is `bash -c <inner>`; rewriting the OUTER command would run the
    continuation outside the sandbox, with the answer surfaces unmasked."""
    inner = " ".join(shlex.quote(c) for c in ["codex", "exec", "--json", "-C", "/ws"])
    out = CA._resume_cmd(["bash", "-c", f"bwrap --args 10 {inner}"], "uuid-2", Path("/ws/p.txt"))
    assert out[0] == "bash" and out[1] == "-c"
    assert out[2].index("bwrap") < out[2].index("resume"), "resume escaped the sandbox"
    assert "uuid-2" in out[2]


def test_a_command_without_exec_is_left_alone():
    """Fail safe: an argv this rewrite does not understand must not be mangled."""
    odd = ["somethingelse", "--flag"]
    assert CA._resume_cmd(odd, "uuid-3", Path("/p.txt")) == odd


def test_the_continuation_prompt_carries_no_hint():
    """The standing instruction only. An arm handed extra guidance mid-session is not comparable to
    one that was not, and this text is sent on every resumed turn."""
    src = (repo_root() / "merlin/experiments/capsule_bench/harness/codex_agent.py").read_text()
    start = src.index("_CONTINUE_MSG = (")
    msg = src[start:src.index("\n\n", start)].lower()
    for leak in ("golden", "expected", "answer", "hidden", "dma_config", "funct", "opcode"):
        assert leak not in msg, f"continuation prompt leaks {leak!r}"
    assert "qa/verdict.json" in msg and "agent_selfcheck" in msg


def test_continuation_is_off_unless_the_schedule_is_continuous():
    """In ROUNDS mode a round ending is how the harness re-grades and re-prompts. Continuing there
    would break that cadence, so the default must be off."""
    import inspect
    assert inspect.signature(CA.run_round).parameters["continue_session"].default is False
    loop = (repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py").read_text()
    assert 'continuous=(a.schedule == "continuous")' in loop


def test_every_stop_reason_is_named():
    """A continuation that stopped silently would be indistinguishable from the defect being fixed."""
    src = (repo_root() / "merlin/experiments/capsule_bench/harness/codex_agent.py").read_text()
    for reason in ("continuation not enabled", "no thread id to resume", "wall budget spent",
                   "the turn did not complete", "turn cap"):
        assert reason in src, f"stop reason {reason!r} is not attributable"
    assert '"type": "codex_session_turn"' in src, "continuation decisions must reach the transcript"
