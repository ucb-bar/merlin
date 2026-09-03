"""The pass slot's proposer: one sandboxed agent turn, and nothing it SAYS is believed.

Every test here injects the runner, so none of them spends a token. What they pin is the bounding:
what goes into the prompt, what comes back out, that a failed turn is a record rather than a crash,
and that the sandbox is required and actually denies the answer surfaces.
"""
import json
import shutil
from pathlib import Path

import pytest

from merlin.mining import pass_agent as pa
from merlin.mining.pass_slot import PassProposal


class _Action:
    divergence_axis = "compute.activation_vectorization"
    intended_facet = {"compute.activation_vectorization": "vectorized_polynomial"}
    change = "extend the polynomial emitter's op coverage"
    target_seam = "pass:llvmlower/act_poly.py (extend the polynomial emitter's op coverage)"
    action_class = "CODEGEN"
    evidence = ["scalar exp is 16.48% of real model work"]


def _reply(source="X = 1\n", prose="I widened the op table."):
    return {"text": f"{prose}\n\n```python\n{source}```\n", "usage": {"input_tokens": 10}}


# --------------------------------------------------------------------------- the prompt

def test_the_prompt_carries_the_promise_the_gate_will_check():
    p = pa.build_prompt(_Action(), evidence=["measured: 16.48%"])
    assert "compute.activation_vectorization" in p
    assert "vectorized_polynomial" in p
    assert "measured: 16.48%" in p
    assert "{axis}" not in p and "{evidence}" not in p, "an unreplaced placeholder reached the agent"


def test_our_measured_value_comes_off_the_divergence_not_the_action():
    """The action carries the TARGET (intended_facet, derived from the expert) and has no field for
    where we are. Leaving it "(not recorded)" would hide the most useful single fact: that the value
    did not move when the cheap lever was applied."""
    from merlin.kernels.cca_compare import Divergence
    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    assert "scalar_libm_call" in pa.build_prompt(_Action(), divergence=d)
    assert "(not recorded)" in pa.build_prompt(_Action())


def test_the_prompt_never_names_the_workload():
    """The pass has to generalise. A prompt that names the model invites a pass that special-cases
    it -- which the cheat scan then rejects, wasting the turn."""
    from merlin.baselines import bundle as b
    p = pa.build_prompt(_Action(), evidence=["scalar exp dominates"]).lower()
    named = [m for m in b.known_models() if len(m) > 5 and m.lower() in p]
    assert not named, f"the task card names {named}"


def test_the_prompt_states_the_rules_the_gate_enforces():
    """The agent should fail for a reason it was told about, not be ambushed by the gate."""
    p = pa.build_prompt(_Action()).lower()
    for rule in ("bit-exact", "byte-identically", "complete new module", "never name a model"):
        assert rule in p, f"the card omits {rule!r}, which the gate rejects on"


# --------------------------------------------------------------------------- the turn

def test_a_successful_turn_yields_a_proposal_and_keeps_the_prose_as_rationale(tmp_path):
    att = pa.propose_pass(_Action(), module="merlin.llvmlower.act_poly",
                          current_source="OLD = 1\n", workspace=tmp_path / "ws",
                          require_sandbox=False, runner=lambda **kw: _reply())
    assert isinstance(att.proposal, PassProposal)
    assert att.proposal.source == "X = 1\n"
    assert "widened" in att.proposal.rationale
    assert att.error is None and att.usage == {"input_tokens": 10}


def test_the_workspace_holds_only_the_module_and_the_card(tmp_path):
    """The capsule property: the agent needs nothing else, so nothing else is there. Anything more
    would have to be re-audited every time a path is added elsewhere."""
    ws = tmp_path / "ws"
    pa.propose_pass(_Action(), module="m", current_source="OLD = 1\n", workspace=ws,
                    require_sandbox=False, runner=lambda **kw: _reply())
    assert sorted(p.name for p in ws.iterdir()) == ["TASK.md", "current_pass.py"]
    assert (ws / "current_pass.py").read_text() == "OLD = 1\n"


@pytest.mark.parametrize("reply,expect", [
    ({"text": "", "usage": {}}, "no text"),
    ({"text": "I could not do it.", "usage": {}}, "no usable python block"),
    ({"text": "```python\n```", "usage": {}}, "empty"),
])
def test_a_failed_turn_is_a_RECORD_not_an_exception(tmp_path, reply, expect):
    """A refused or empty turn is evidence about the seam and still cost tokens, so it is kept.
    run_pass_slot turns proposal=None into an honest no_proposal verdict."""
    att = pa.propose_pass(_Action(), module="m", current_source="x=1\n",
                          workspace=tmp_path / "ws", require_sandbox=False,
                          runner=lambda **kw: reply)
    assert att.proposal is None
    assert att.error and expect in att.error
    assert att.to_dict()["proposed"] is False


def test_the_attempt_record_digests_the_source_that_was_gated(tmp_path):
    att = pa.propose_pass(_Action(), module="m", current_source="x=1\n",
                          workspace=tmp_path / "ws", require_sandbox=False,
                          runner=lambda **kw: _reply("A = 2\n"))
    d = att.to_dict()
    assert d["proposed"] is True and d["source_digest"] and len(d["source_digest"]) == 16
    other = pa.propose_pass(_Action(), module="m", current_source="x=1\n",
                            workspace=tmp_path / "ws2", require_sandbox=False,
                            runner=lambda **kw: _reply("A = 3\n"))
    assert other.to_dict()["source_digest"] != d["source_digest"]


# --------------------------------------------------------------------------- the sandbox

def test_an_unsandboxed_proposer_is_REFUSED_by_default(tmp_path, monkeypatch):
    """An agentic run that can read the enclosing checkout has already been observed reading another
    arm's results and the study's own status file. A proposer that can reach a golden makes its own
    gate meaningless, so no-sandbox is a refusal rather than a downgrade."""
    monkeypatch.setattr(pa, "sandbox_argv", lambda ws: None)
    called = []
    att = pa.propose_pass(_Action(), module="m", current_source="x=1\n",
                          workspace=tmp_path / "ws",
                          runner=lambda **kw: called.append(1) or _reply())
    assert att.proposal is None and att.sandboxed is False
    assert "bwrap is unavailable" in att.error
    assert called == [], "the turn must not run at all without the sandbox"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_the_sandbox_masks_scratch_and_the_experimenters_session_history(tmp_path):
    """The two answer surfaces that matter here: /scratch holds every capture, golden and other run,
    and ~/.claude/projects holds the experimenter's own transcripts and persistent memory."""
    argv = pa.sandbox_argv(tmp_path / "ws")
    assert argv and argv[0] == "bwrap"
    flat = " ".join(argv)
    assert "--tmpfs /scratch " in flat + " ", "captures and goldens must be masked"
    assert "--tmpfs /scratch2" in flat
    assert ".claude/projects" in flat, "the experimenter's transcripts/memory must be masked"
    assert str(tmp_path / "ws") == argv[-1], "the workspace must be bound LAST so no mask clobbers it"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_the_sandbox_really_cannot_read_the_repo_or_scratch(tmp_path):
    """Executable proof, not an argv assertion: run a probe inside the box."""
    import subprocess
    from merlin.common.paths import repo_root
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    probe = ("import os;"
             "print('repo', os.path.exists(%r));"
             "print('scratch_proj', os.path.exists('/scratch/agustin/projects'))"
             % str(repo_root() / "merlin" / "python" / "merlin" / "kernels" / "cca.py"))
    argv = pa.sandbox_argv(ws) + ["python3", "-c", probe]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    body = out.stdout
    assert "repo False" in body, f"the checkout was readable inside the sandbox: {body} {out.stderr[-300:]}"
    assert "scratch_proj False" in body, f"/scratch was readable inside the sandbox: {body}"


def test_proposer_for_resolves_the_module_from_the_seam_and_collects_attempts(tmp_path):
    propose, attempts = pa.proposer_for(_Action(), current_source="x=1\n",
                                        workspace=tmp_path / "ws", require_sandbox=False,
                                        runner=lambda **kw: _reply())
    p = propose(_Action())
    assert p is not None and p.module == "merlin.llvmlower.act_poly"
    assert len(attempts) == 1 and attempts[0].proposal is p


def test_run_pass_slot_turns_a_refused_turn_into_an_honest_verdict(tmp_path):
    """End to end with the real gate: a turn that produced nothing must read as no_proposal, not as
    an accepted pass and not as a crash."""
    from merlin.mining.pass_slot import run_pass_slot
    propose, _ = pa.proposer_for(_Action(), current_source="x=1\n", workspace=tmp_path / "ws",
                                 require_sandbox=False,
                                 runner=lambda **kw: {"text": "no.", "usage": {}})
    proposal, verdict = run_pass_slot(_Action(), propose=propose)
    assert proposal is None and not verdict.accepted and verdict.stage == "no_proposal"


def test_the_agent_is_launched_with_stdin_closed(tmp_path, monkeypatch):
    """Headless `claude -p` waits on stdin for piped input, then exits 1 with "no stdin data received
    in 3s". Inherited stdin cost a full 561 s turn and presented as an agent failure rather than a
    launch bug, so the launch must close stdin explicitly."""
    seen = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"result": "ok\n```python\nX = 1\n```\n", "usage": {}})
        stderr = ""

    def _fake_run(argv, **kw):
        seen.update(kw)
        return _Proc()

    monkeypatch.setattr(pa.subprocess, "run", _fake_run)
    monkeypatch.setattr(pa, "sandbox_argv", lambda ws: ["bwrap", "--bind", str(ws), str(ws)])
    att = pa.propose_pass(_Action(), module="m", current_source="x=1\n",
                          workspace=tmp_path / "ws")
    assert att.proposal is not None, att.error
    assert seen.get("stdin") is pa.subprocess.DEVNULL, "stdin must be closed, not inherited"


def test_the_workspace_file_is_the_primary_proposal_channel(tmp_path):
    """The module is ~15 KB and a real rewrite came back at ~33 KB, so a fenced block in the reply is
    liable to be truncated -- and an agent with a writable workspace naturally writes a file anyway
    (the first real turn wrote exactly this name unprompted). Refusing it would throw away a completed
    turn over a formatting preference."""
    ws = tmp_path / "ws"

    def _runner(**kw):
        (kw["workspace"] / pa.PROPOSAL_FILENAME).write_text("FROM_FILE = 1\n")
        return {"text": "I wrote the new module to new_pass.py.", "usage": {}}

    att = pa.propose_pass(_Action(), module="m", current_source="old\n", workspace=ws,
                          require_sandbox=False, runner=_runner)
    assert att.proposal is not None and att.proposal.source == "FROM_FILE = 1\n"


def test_the_file_wins_over_a_fenced_block(tmp_path):
    """It cannot be truncated by a reply limit, so it is the more trustworthy of the two."""
    def _runner(**kw):
        (kw["workspace"] / pa.PROPOSAL_FILENAME).write_text("FROM_FILE = 1\n")
        return {"text": "```python\nFROM_BLOCK = 1\n```", "usage": {}}

    att = pa.propose_pass(_Action(), module="m", current_source="old\n",
                          workspace=tmp_path / "ws", require_sandbox=False, runner=_runner)
    assert att.proposal.source == "FROM_FILE = 1\n"


def test_a_turn_that_wrote_only_the_file_and_said_nothing_still_counts(tmp_path):
    def _runner(**kw):
        (kw["workspace"] / pa.PROPOSAL_FILENAME).write_text("FROM_FILE = 1\n")
        return {"text": "", "usage": {}}

    att = pa.propose_pass(_Action(), module="m", current_source="old\n",
                          workspace=tmp_path / "ws", require_sandbox=False, runner=_runner)
    assert att.proposal is not None and att.error is None


def test_an_empty_workspace_file_falls_back_to_the_block(tmp_path):
    def _runner(**kw):
        (kw["workspace"] / pa.PROPOSAL_FILENAME).write_text("   \n")
        return {"text": "```python\nFROM_BLOCK = 1\n```", "usage": {}}

    att = pa.propose_pass(_Action(), module="m", current_source="old\n",
                          workspace=tmp_path / "ws", require_sandbox=False, runner=_runner)
    assert att.proposal.source == "FROM_BLOCK = 1\n"
