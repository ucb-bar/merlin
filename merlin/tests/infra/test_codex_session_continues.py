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


def test_resume_passes_only_flags_the_subcommand_accepts():
    """`codex exec resume` takes a SMALLER option set than `codex exec`.

    Measured against 0.153.0 --help: it accepts --json/--model/-o/--skip-git-repo-check and the
    sandbox bypass, but NOT --color and NOT -C. Rewriting the exec argv therefore produced
    `Usage: codex exec resume ... <SESSION_ID> [PROMPT]` and the continuation died on its first
    attempt with 14772s of budget left. Verified against the live CLI: with these flags it gets past
    parsing to `no rollout found for thread id`, i.e. the arguments are accepted."""
    cmd = CA.build_resume_cmd(Path("/ws"), model="m", effort="high", final_path=Path("/ws/f.txt"),
                              sandbox="bwrap", thread_id="TID")
    assert cmd[:3] == ["codex", "exec", "resume"]
    assert "--color" not in cmd, "resume rejects --color"
    assert "-C" not in cmd, "resume rejects -C (cwd comes from the spawn instead)"
    for required in ("--json", "--model", "-o", "--skip-git-repo-check"):
        assert required in cmd, f"{required} lost on resume"


def test_the_session_id_precedes_the_stdin_prompt_marker():
    """The grammar is `[OPTIONS] [SESSION_ID] [PROMPT]`, and `-` is the prompt."""
    cmd = CA.build_resume_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"),
                              sandbox="bwrap", thread_id="TID")
    assert cmd[-2:] == ["TID", "-"]


def test_resume_keeps_the_sandbox_bypass_off_the_unsandboxed_path():
    """The outer bwrap is the boundary when sandboxed; without it codex needs its own policy."""
    boxed = CA.build_resume_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"),
                                sandbox="bwrap", thread_id="T")
    bare = CA.build_resume_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"),
                               sandbox="none", thread_id="T")
    assert "--dangerously-bypass-approvals-and-sandbox" in boxed
    assert "--sandbox" in bare and "workspace-write" in bare


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
