"""The pass slot's PROPOSER: one sandboxed agent turn that rewrites a compiler pass.

:mod:`mining.pass_slot` is the gate and :mod:`mining.pass_slot_wiring` connects it to the toolchain.
This is the only non-deterministic part of the loop, and it is bounded on every side:

* it is reached only when the deterministic search has run out of levers -- the CCA lifts the loss,
  the router picks the cheapest action, the beam forks and measures it, and ``achieved_residual`` says
  the promise went unmet, at which point ``route_escalated`` returns a rung whose ``forkable_now`` is
  False. That rung names a module (``action_catalog.seam_module``), and this proposes a rewrite of it;
* it gets ONE turn with no tools beyond its workspace, and the workspace holds only the module source
  and the task card. Everything else is denied by ``targetgen.sandbox.bwrap``, which tmpfs-masks
  ``/scratch`` wholesale (so no capture, golden or other run is reachable) and masks the
  experimenter's own Claude session history and memory;
* nothing it SAYS is believed. The gate reads only executable consequences: a structural cheat scan,
  the frozen baseline still lowering byte-identically, the build still bit-exact against the golden,
  and the promised facet lifted from the emitted assembly.

So the agent's judgement is used for the one thing search cannot do -- writing code that does not
exist yet -- and none of its claims enter the record.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .pass_slot import PassProposal

#: The task card. Versioned and reviewed, like the other agent prompts, because the prompt IS the
#: experimental treatment: a run is only comparable to another run of the same version.
PROMPT_VERSION = 1


def prompt_path(version: int = PROMPT_VERSION) -> Path:
    from ..common.paths import prompts_dir
    return prompts_dir() / f"pass_slot_v{version}.md"


@dataclass
class ProposalAttempt:
    """One turn's full record. Kept whether or not it produced a usable proposal, because a refused
    or empty turn is evidence about the seam and still costs tokens."""
    proposal: PassProposal | None
    module: str
    prompt_version: int
    model: str
    sandboxed: bool
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    transcript_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "prompt_version": self.prompt_version, "model": self.model,
                "sandboxed": self.sandboxed, "usage": self.usage, "error": self.error,
                "transcript": self.transcript_path,
                "proposed": self.proposal is not None,
                "source_digest": (None if self.proposal is None
                                  else __import__("hashlib").sha256(
                                      self.proposal.source.encode()).hexdigest()[:16])}


def _fmt_evidence(evidence: "list[str] | tuple[str, ...] | None") -> str:
    if not evidence:
        return ("None recorded. Reason from the axis and the module alone; do not invent a "
                "measurement to justify a change.")
    return "\n".join(f"- {e}" for e in evidence)


def build_prompt(action, *, evidence=None, ours=None, divergence=None,
                 version: int = PROMPT_VERSION) -> str:
    """Render the task card for one escalated action.

    Only the action's own machine-readable fields and the supplied evidence go in. Notably NOT the
    model, the bundle, or any capture path: the pass has to generalise, and a prompt that names the
    workload invites a pass that special-cases it -- which the cheat scan would then reject, wasting
    the turn.
    """
    facet = getattr(action, "intended_facet", None) or {}
    axis = getattr(action, "divergence_axis", "?")
    # OUR measured value comes off the Divergence, not the action: the action carries the TARGET
    # (intended_facet, derived by the router from the expert) and has no field for where we are. Saying
    # "(not recorded)" here would hide the most useful single fact -- that the value did not move.
    if ours is None and divergence is not None:
        ours = getattr(divergence, "ours", None)
    tmpl = prompt_path(version).read_text(encoding="utf-8")
    return (tmpl.replace("{axis}", str(axis))
                .replace("{intended_facet}", json.dumps(facet, sort_keys=True))
                .replace("{ours}", "(not recorded)" if ours is None else str(ours))
                .replace("{expert}", str(facet.get(axis, "(see intended_facet)")))
                .replace("{change}", str(getattr(action, "change", "?")))
                .replace("{evidence}", _fmt_evidence(evidence)))


def sandbox_argv(ws: Path) -> list[str] | None:
    """The bwrap prefix isolating one proposer turn, or None when bwrap is unavailable.

    Deny-by-default with an EMPTY bundle, which is the strongest form: the agent needs nothing but
    its workspace, since the task card carries the promise and the workspace carries the module. That
    also means no path has to be re-audited when one is added elsewhere.
    """
    if shutil.which("bwrap") is None:
        return None
    from ..common.paths import repo_root
    from ..targetgen.sandbox import bwrap
    # `base_argv` binds the workspace LAST on purpose, so no mask can clobber it. Appending the CLI's
    # runtime binds after it breaks that ordering, so re-bind the workspace at the very end. Binding
    # the same path twice is harmless and keeps the invariant explicit rather than depending on the
    # runtime binds never happening to overlap the workspace.
    return (bwrap.base_argv(ws, {}, repo=repo_root())
            + bwrap.claude_runtime_binds()
            + ["--bind", str(ws), str(ws)])


#: Where the agent is asked to write the new module. A FILE, not a fenced block in the reply, is the
#: primary channel: the module here is ~15 KB and a rewrite came back at ~33 KB, which is large enough
#: that a reply is liable to be truncated -- and an agent with a writable workspace naturally writes a
#: file anyway (observed on the first real turn, which wrote exactly this name unprompted).
PROPOSAL_FILENAME = "new_pass.py"


def _extract_module_source(text: str, workspace: "Path | None" = None) -> str:
    """The proposed source: the workspace file if the agent wrote one, else a fenced block.

    Both are accepted because both are things the agent actually does, and refusing the file would
    throw away a completed turn over a formatting preference. The file wins when present: it cannot be
    truncated by a reply limit.
    """
    if workspace is not None:
        f = Path(workspace) / PROPOSAL_FILENAME
        if f.is_file():
            body = f.read_text(encoding="utf-8")
            if body.strip():
                return body
    from ..common import agent_output
    return agent_output.extract_code_block(text, "python")


def propose_pass(action, *, module: str, current_source: str, workspace: Path,
                 model: str = "opus", timeout: int = 1800, version: int = PROMPT_VERSION,
                 require_sandbox: bool = True, ours=None, divergence=None,
                 runner: Callable[..., dict] | None = None) -> ProposalAttempt:
    """Run ONE proposer turn for ``action`` and return the attempt record.

    ``require_sandbox`` defaults True and REFUSES to run unsandboxed: an agentic run that can read the
    enclosing checkout has already been observed reading another arm's results and the study's own
    status file, and a proposer that can reach a golden makes its own gate meaningless. Pass False
    only for a test with an injected ``runner``.
    """
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "current_pass.py").write_text(current_source, encoding="utf-8")
    prompt = build_prompt(action, evidence=getattr(action, "evidence", None), ours=ours,
                          divergence=divergence, version=version)
    (ws / "TASK.md").write_text(prompt, encoding="utf-8")

    sandbox = sandbox_argv(ws)
    if sandbox is None and require_sandbox:
        return ProposalAttempt(None, module, version, model, False,
                               error="bwrap is unavailable and require_sandbox is set; refusing to "
                                     "run a proposer that could read the enclosing checkout")

    if runner is not None:
        out = runner(prompt=prompt, workspace=ws, model=model, timeout=timeout, sandbox=sandbox)
    else:
        # cache-buster so repeated turns on the same seam do not serve one another's answer
        argv = list(sandbox or []) + [
            "claude", "-p", f"<!-- nonce: {uuid.uuid4().hex} -->\n{prompt}",
            "--model", model, "--output-format", "json"]
        env = dict(os.environ)
        try:
            # stdin=DEVNULL, not inherited. Headless `claude -p` waits on stdin for piped input and
            # then exits 1 with "no stdin data received in 3s" -- which cost a full 561 s turn and
            # looked like an agent failure rather than a launch bug. A detached/nohup parent has no
            # usable stdin, so the prompt must arrive by argv alone and stdin must be closed.
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  cwd=str(ws), env=env, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return ProposalAttempt(None, module, version, model, sandbox is not None,
                                   error=f"proposer timed out after {timeout}s")
        raw = proc.stdout or ""
        tp = ws / "agent_transcript.json"
        tp.write_text(raw, encoding="utf-8")
        if proc.returncode != 0:
            return ProposalAttempt(None, module, version, model, sandbox is not None,
                                   error=f"claude exited {proc.returncode}: "
                                         f"{(proc.stderr or '')[-800:]}",
                                   transcript_path=str(tp))
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
            try:
                obj = json.loads(lines[-1]) if lines else {}
            except json.JSONDecodeError:
                return ProposalAttempt(None, module, version, model, sandbox is not None,
                                       error="proposer output was not JSON",
                                       transcript_path=str(tp))
        out = {"text": obj.get("result") or obj.get("text") or "",
               "usage": obj.get("usage", {}), "transcript_path": str(tp)}

    text = out.get("text") or ""
    usage = out.get("usage") or {}
    tp = out.get("transcript_path")
    if not text.strip() and not (ws / PROPOSAL_FILENAME).is_file():
        return ProposalAttempt(None, module, version, model, sandbox is not None,
                               usage=usage, error="proposer returned no text and wrote no "
                                                  f"{PROPOSAL_FILENAME}",
                               transcript_path=tp)
    try:
        source = _extract_module_source(text, ws)
    except Exception as e:  # noqa: BLE001 - any extraction failure is an honest refusal, not a crash
        return ProposalAttempt(None, module, version, model, sandbox is not None, usage=usage,
                               error=f"no usable python block in the reply: {e}",
                               transcript_path=tp)
    if not source.strip():
        return ProposalAttempt(None, module, version, model, sandbox is not None, usage=usage,
                               error="the python block was empty", transcript_path=tp)
    rationale = text.split("```", 1)[0].strip()[:1200]
    if not source.endswith("\n"):
        source += "\n"        # the extractor strips; a module file ends with a newline
    return ProposalAttempt(PassProposal(module=module, source=source, rationale=rationale),
                           module, version, model, sandbox is not None, usage=usage,
                           transcript_path=tp)


def proposer_for(action, *, current_source: str, workspace: Path, **kw
                 ) -> tuple[Callable[[Any], PassProposal | None], list[ProposalAttempt]]:
    """A ``propose(action) -> PassProposal | None`` for :func:`pass_slot.run_pass_slot`, plus the list
    the attempt record lands in. ``run_pass_slot`` turns None into an honest ``no_proposal`` verdict,
    so a failed turn needs no special handling by the caller -- but its record is still kept."""
    from ..kernels import action_catalog as ac
    module = ac.seam_module(getattr(action, "target_seam", "") or "")
    attempts: list[ProposalAttempt] = []

    def _propose(a):
        att = propose_pass(a, module=module or "?", current_source=current_source,
                           workspace=Path(workspace), **kw)
        attempts.append(att)
        return att.proposal

    return _propose, attempts
