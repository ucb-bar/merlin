"""Pass slot: an agent extends a compiler PASS, a deterministic gate decides whether it counts.

This is the leaf the escalation ladder terminates in. Everything upstream of it is deterministic:
``kernels.cca`` lifts the loss from the emitted code, ``kernels.action_catalog.route`` maps it to the
cheapest lever, the beam forks that lever, and ``action_catalog.achieved_residual`` says whether the
emitted code delivered the promise. When the cheapest lever cannot deliver, ``route_escalated`` walks
up to a rung whose ``forkable_now`` is False -- meaning no knob or feature expresses it and new code
must be written. Before this module the beam recorded that as a work-item and stopped.

WHY A SLOT AND NOT A CAPSULE. A capsule grades a TARGET's numerical kernel against a hardware oracle at
a fidelity tier. What is needed here is different in kind: the artifact is OUR compiler's pass, and the
acceptance question is "did the emitted code acquire the facet the action promised, without changing any
number". That is answered by ``achieved_residual`` plus bit-exactness, not by a tier. It lives beside
``mining.tuning_agent`` because the beam owns the ladder, so the beam owns its leaf.

THE DISCIPLINE, inherited from targetgen's agent kernel slot: agent autonomy on visible data, a
deterministic oracle gate, held-out certification, and structural cheat detection. The agent's semantic
claims are never trusted -- only the executable consequences are checked. Two additions specific to a
compiler pass:

  * the FROZEN BASELINE must survive. An empty feature set has to lower byte-identically, or every
    measurement in the repo taken against the control silently moves. This is checked BEFORE anything
    else expensive, because it is the cheapest way for a proposal to be wrong.
  * the promise is machine-readable and already exists. ``CompilerAction.intended_facet`` is derived by
    the router from the expert's own value on the axis, so the gate does not restate a target -- it
    lifts the CCA from the emitted code and asks ``achieved_residual``. A gate that restated the target
    could disagree with the router about what was being asked for.

The agent is INJECTABLE and defaults to None, so the gate is fully testable -- and fully useful -- with
no agent and no budget. That is deliberate: the gate is the part that must be right.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---- pass_slot ---------------------------------------------------------------------------------

#: Tokens that mean the proposal is reaching for the answer instead of computing it. Narrower than
#: the kernel slot's list because a compiler pass legitimately imports numpy and legitimately contains the
#: word "reference" in prose -- so this keys on the specific escapes that would let a pass special-case
#: its way to a passing gate.
CHEAT_TOKENS: tuple[str, ...] = (
    "golden.npy",  # reading the answer
    "region_goldens",
    "achieved_residual",  # asserting its own gate verdict
    "intended_facet",
    "_REGISTRY[",  # mutating the feature registry to fake a facet
    "monkeypatch",
)


#: A pass that names a MODEL is overfit by construction: the whole point of the loop is that a lever
#: found on one model transfers. Checked against the source-owned workload roster, which remains
#: available in an installed core even when the capture corpus is absent.
def model_name_tokens() -> tuple[str, ...]:
    """Roster names the proposal must not mention; fail closed if the roster is unavailable."""
    from merlin.capture import bundle as _b

    return tuple(_b.known_models())


@dataclass
class PassProposal:
    """What the agent returns: replacement source for ONE pass module, and why."""

    module: str  # dotted module the source replaces, e.g. merlin.compiler.lower.options.nonlinear
    source: str  # the full new module source
    rationale: str = ""


@dataclass
class PassVerdict:
    """Why the proposal was accepted or refused. Ordered, so the first failure is the cheapest one."""

    accepted: bool
    stage: str  # cheat | frozen_baseline | bit_exact | facet | heldout | accepted
    reason: str = ""
    residual: tuple[str, ...] = ()  # axes promised but not achieved (from achieved_residual)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def checkable(self) -> bool:
        """False when the action carried no machine-readable promise, so 'accepted' would be unearned.
        The gate refuses in that case rather than passing something it cannot verify."""
        return self.stage != "unverifiable"


def _behavioural_words(source: str, _depth: int = 0) -> tuple[set[str], set[str], bool]:
    """Text that can affect BEHAVIOUR, as ``(exact_words, underscore_spans, parsed_ok)``.

    Parsed with ``ast`` for two reasons. Comments do not appear in an AST at all, which is exactly
    right: a comment cannot special-case a pass, and this repo WANTS model names in provenance prose
    (``llvmlower/act_poly.py`` records that a blanket rewrite "drove openvla whole-model cos to 0.541"
    -- the measurement that motivated the fix). Docstrings are excluded for the same reason. What
    remains -- a name or a string the code can compare or dispatch on -- is the only place a model name
    can change what the pass does.

    A string literal that is ITSELF parseable Python is scanned AS SOURCE, recursively. That is not an
    exotic case here: the compiler passes splice generated source as a string (``act_poly``,
    ``accum_microkernel``'s rewriter, ``perop_blocks``), so a pass proposal's real content usually
    lives inside one. Without the recursion the comments in that spliced source would be read as
    behavioural string content -- which is what rejected ``act_poly.py``'s own bytes.

    Structural, not regex: ``ast`` is a real parser, per the repo's no-regex rule.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set(), False
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    exact: set[str] = set()
    spans: set[str] = set()

    def _add_identifier(name: str) -> None:
        """A whole identifier, plus its contiguous underscore-joined spans.

        The spans are what catch ``small_llama_hack``: an identifier that EMBEDS a multi-word model
        name is the overfit pattern. They are kept in a separate set from exact words because a SHORT
        model token must not match a span -- the corpus list contains `small`, `rdt` and `pi05`, so
        span-matching those would flag ``small_m_fallback`` and ``rdtime`` (the K1 cycle counter). A
        gate that rejects every honest proposal is not strict, it is broken.
        """
        low = name.lower()
        exact.add(low)
        parts = [q for q in low.split("_") if q]
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                spans.add("_".join(parts[i:j]))

    def _add_text(text: str) -> None:
        cur: list[str] = []
        for ch in text:
            if ch.isalnum() or ch == "_":
                cur.append(ch)
            elif cur:
                _add_identifier("".join(cur))
                cur = []
        if cur:
            _add_identifier("".join(cur))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            _add_identifier(node.id)
        elif isinstance(node, ast.Attribute):
            _add_identifier(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _add_identifier(node.name)
        elif isinstance(node, ast.arg):
            _add_identifier(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            _add_identifier(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                _add_text(a.name)
            if isinstance(node, ast.ImportFrom) and node.module:
                _add_text(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            nested = None
            if _depth < 3 and ("\n" in node.value or "=" in node.value):
                e2, s2, ok = _behavioural_words(node.value, _depth + 1)
                nested = (e2, s2) if ok else None
            if nested is not None:
                exact |= nested[0]
                spans |= nested[1]
            else:
                _add_text(node.value)
    return exact, spans, True


def scan_cheats(source: str, *, models: tuple[str, ...] | None = None) -> list[str]:
    """Tokens in the proposal that would let it pass the gate without doing the work.

    Structural and cheap, run FIRST: a proposal that reads the golden or hardcodes a model name is
    rejected before any build. ``models`` defaults to the captures on disk, so this generalises as the
    corpus grows instead of being a literal list someone has to remember to extend.

    A model name is looked for only where it could change BEHAVIOUR -- an identifier or a
    string-literal word, never a comment or docstring -- and matched as a whole WORD. Both halves were
    needed: scanning raw text rejected ``act_poly.py``'s own bytes over the ``openvla`` in a comment
    recording the regression that motivated it, and substring matching flagged the word "small" in
    "small ranges" because ``small`` is in the corpus's model list. A scanner that rejects every honest
    proposal is not a strict gate, it is a broken one.

    Unparseable source is itself a finding: it cannot be gated, so it is reported rather than passed.
    """
    found = [t for t in CHEAT_TOKENS if t in source]
    exact, spans, parsed = _behavioural_words(source)
    if not parsed:
        found.append("unparseable:the proposal is not valid Python, so it cannot be gated")
        return found
    for m in models if models is not None else model_name_tokens():
        # a model NAME the pass can dispatch on is overfit by construction. A multi-word name is also
        # matched inside a longer identifier (small_llama_hack); a single short token is matched only
        # exactly, or `small` would flag `small_m_fallback` and `rdt` would flag `rdtime`.
        if not m:
            continue
        low = m.lower()
        if low in exact or ("_" in low and low in spans):
            found.append(f"model:{m}")
    return found


def verify_promise(action, achieved_cca) -> PassVerdict:
    """Did the emitted code acquire the facet the action promised?

    Delegates to ``action_catalog.achieved_residual`` rather than restating the target, so the gate and
    the router cannot disagree about what was asked for. An action with no promise is UNVERIFIABLE and
    is refused -- accepting it would credit a change nothing checked, which is the failure mode this
    whole loop exists to remove.
    """
    from merlin.analysis.kernels.cca import action_catalog as ac

    if not getattr(action, "intended_facet", None):
        return PassVerdict(
            False,
            "unverifiable",
            "the action carries no intended_facet, so there is no machine-checkable "
            "promise to hold the proposal to; refusing rather than crediting it",
        )
    residual = tuple(ac.achieved_residual(action, achieved_cca))
    if residual:
        return PassVerdict(
            False,
            "facet",
            f"promised {action.intended_facet} but the emitted code did not achieve {list(residual)}",
            residual=residual,
        )
    return PassVerdict(True, "accepted", "the emitted code achieved the promised facet")


def gate(
    proposal: PassProposal,
    action,
    *,
    frozen_baseline_ok: Callable[[PassProposal], bool],
    bit_exact_ok: Callable[[PassProposal], tuple[bool, str]],
    lift_cca: Callable[[PassProposal], Any],
    inert_ok: Callable[[PassProposal], tuple[bool, str]] | None = None,
    heldout_ok: Callable[[PassProposal], tuple[bool, str]] | None = None,
    models: tuple[str, ...] | None = None,
) -> PassVerdict:
    """Run the ordered gate. Cheapest disqualifier first; every stage fails CLOSED.

    The four checks are injected rather than imported so the gate is testable without a toolchain, a
    board, or an agent -- the same reason ``critic.py`` injects its runner. In production they are:
    ``frozen_baseline_ok``  -> empty features still lower byte-identically
    ``bit_exact_ok``        -> spike run matches the golden
    ``inert_ok``            -> the emitted code actually CHANGED vs the same package unpatched
    ``lift_cca``            -> cca.lift_asm over the emitted disassembly
    ``heldout_ok``          -> the same two checks on captures the agent never saw

    ``inert_ok`` runs before the facet check because the two failures need different answers. A
    proposal whose emitted code is byte-identical to the unpatched build did not RUN -- its op
    matching never fired -- and telling its author "the promised facet was not achieved" points them
    at the wrong thing entirely: they will improve a polynomial that was never reached. MEASURED on
    the first real agent turn: a 611-line rewrite passed the cheat scan, the frozen baseline and
    bit-exactness, and produced an object identical to the control down to the instruction count
    (47,988 insns / 10,782 vector, same undefined symbols) -- reported only as "facet not achieved".
    This is the same guard the beam has carried since two shipped levers measured inert while looking
    correctly wired.
    """
    cheats = scan_cheats(proposal.source, models=models)
    if cheats:
        return PassVerdict(False, "cheat", f"proposal references forbidden tokens {cheats}", detail={"tokens": cheats})
    if not frozen_baseline_ok(proposal):
        return PassVerdict(
            False,
            "frozen_baseline",
            "an empty feature set no longer lowers byte-identically, so every "
            "measurement taken against the control would silently move",
        )
    ok, why = bit_exact_ok(proposal)
    if not ok:
        # "bit-exactness not established", not "numerics changed": the check also fails when it could
        # not RUN (the proposed module was never imported by the build), and reporting that as a
        # numeric regression would send the next proposal chasing a change that never happened.
        return PassVerdict(False, "bit_exact", f"bit-exactness not established: {why}")
    if inert_ok is not None:
        ok, why = inert_ok(proposal)
        if not ok:
            return PassVerdict(False, "inert", f"the emitted code did not change: {why}")
    verdict = verify_promise(action, lift_cca(proposal))
    if not verdict.accepted:
        return verdict
    if heldout_ok is not None:
        ok, why = heldout_ok(proposal)
        if not ok:
            return PassVerdict(False, "heldout", f"held on the visible captures but not on held-out ones: {why}")
    return PassVerdict(
        True,
        "accepted",
        "gate passed: no cheats, baseline frozen, numerics "
        "bit-exact, promised facet achieved" + (", held out" if heldout_ok is not None else ""),
    )


def verdict_feedback(verdict: PassVerdict) -> str:
    """The refusal, rendered for the next turn. FACTS ONLY -- never a suggested fix.

    What separates a loop from a human doing the work through an agent is who supplies the insight. If
    this said "your op matcher probably misses linalg.generic", the resulting pass would be the
    author's idea typed by a model, and it would carry the author's blind spots into every future run.
    So it reports exactly what the gate MEASURED: the stage that refused, its reason, the axes still
    outstanding, and whatever diagnostic values the check recorded (digests, counts). The agent draws
    its own conclusion.

    That is also why the reasons are worth their length: "byte-identical to the unpatched build
    (digest X); the pass was imported but its matching never fired" is a measurement, and it is enough
    to redirect a turn without telling it where to go.
    """
    lines = [f"- **Stage that refused**: `{verdict.stage}`", f"- **What the gate measured**: {verdict.reason}"]
    if verdict.residual:
        lines.append(f"- **Axes still not achieved**: {list(verdict.residual)}")
    for k, v in sorted((verdict.detail or {}).items()):
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def _no_progress(a: PassVerdict, b: PassVerdict) -> bool:
    """True when two consecutive refusals are the same verdict, so another turn is spend without
    signal. Compared on stage + reason: a turn that moved the failure to a LATER stage made progress
    even though it also failed, and a turn that failed the same way twice did not."""
    return (a.stage, a.reason) == (b.stage, b.reason)


def iterate_pass_slot(
    action,
    *,
    propose: Callable[..., PassProposal | None] | None = None,
    max_turns: int = 3,
    on_turn: Callable[[int, Any, PassVerdict], None] | None = None,
    **gate_kwargs,
) -> list[tuple[PassProposal | None, PassVerdict]]:
    """Run the slot up to ``max_turns`` times, feeding each refusal back into the next proposal.

    One turn is not a loop. The gate already produces a precise, machine-checked refusal -- "inert:
    byte-identical to the unpatched build, the pass was imported but its matching never fired" -- and
    stopping there throws away the one piece of information the next attempt needs. MEASURED on the
    first real turn: an 18-minute, 611-line proposal was refused for a reason its author could have
    acted on immediately, and nothing carried it forward.

    ``propose(action, feedback=...)`` receives :func:`verdict_feedback` of the previous refusal, or
    None on the first turn. Every (proposal, verdict) pair is returned, refusals included -- a refused
    proposal is evidence about the seam.

    Stops early on acceptance, and on two consecutive IDENTICAL verdicts: the feedback is not landing,
    so further turns are spend without signal. Never stops on a verdict that merely moved to a later
    stage, which is progress even though it is still a refusal.
    """
    out: list[tuple[PassProposal | None, PassVerdict]] = []
    feedback: str | None = None
    for turn in range(1, max(1, max_turns) + 1):
        if propose is None:
            v = PassVerdict(False, "no_proposal", "no proposer supplied; the gate is a no-op without one")
            out.append((None, v))
            return out
        try:
            proposal = propose(action, feedback=feedback)
        except TypeError:
            # a proposer that predates the feedback argument still works, it just cannot learn
            proposal = propose(action)
        if proposal is None:
            v = PassVerdict(False, "no_proposal", "the proposer returned nothing")
            out.append((None, v))
            return out
        verdict = gate(proposal, action, **gate_kwargs)
        out.append((proposal, verdict))
        if on_turn is not None:
            on_turn(turn, proposal, verdict)
        if verdict.accepted:
            return out
        if len(out) >= 2 and out[-2][1] is not None and _no_progress(out[-2][1], verdict):
            return out
        feedback = verdict_feedback(verdict)
    return out


def run_pass_slot(
    action, *, propose: Callable[[Any], PassProposal | None] | None = None, **gate_kwargs
) -> tuple[PassProposal | None, PassVerdict]:
    """Ask for a proposal and gate it. ``propose=None`` -> no proposal, and an honest refusal.

    Returns ``(proposal, verdict)`` so a caller can record BOTH -- a refused proposal is evidence about
    the seam, not noise, and the beam already records non-actionable outcomes as work-items rather than
    dropping them.
    """
    if propose is None:
        return None, PassVerdict(
            False,
            "no_proposal",
            "no proposer supplied; the gate is a no-op without one (this is the "
            "default so the gate is testable and usable with no agent budget)",
        )
    proposal = propose(action)
    if proposal is None:
        return None, PassVerdict(False, "no_proposal", "the proposer returned nothing")
    return proposal, gate(proposal, action, **gate_kwargs)


# ---- pass_agent --------------------------------------------------------------------------------
# The pass slot's PROPOSER: one sandboxed agent turn that rewrites a compiler pass.
#
# :mod:`mining.pass_slot` is the gate and :mod:`mining.pass_slot_wiring` connects it to the toolchain.
# This is the only non-deterministic part of the loop, and it is bounded on every side:
#
# * it is reached only when the deterministic search has run out of levers -- the CCA lifts the loss,
#   the router picks the cheapest action, the beam forks and measures it, and ``achieved_residual`` says
#   the promise went unmet, at which point ``route_escalated`` returns a rung whose ``forkable_now`` is
#   False. That rung names a module (``action_catalog.seam_module``), and this proposes a rewrite of it;
# * it gets ONE turn with no tools beyond its workspace, and the workspace holds only the module source
#   and the task card. Everything else is denied by ``targetgen.sandbox.bwrap``, which tmpfs-masks
#   ``/scratch`` wholesale (so no capture, golden or other run is reachable) and masks the
#   experimenter's own Claude session history and memory;
# * nothing it SAYS is believed. The gate reads only executable consequences: a structural cheat scan,
#   the frozen baseline still lowering byte-identically, the build still bit-exact against the golden,
#   and the promised facet lifted from the emitted assembly.
#
# So the agent's judgement is used for the one thing search cannot do -- writing code that does not
# exist yet -- and none of its claims enter the record.

# ---- pass_agent --------------------------------------------------------------------------------

#: The task card. Versioned and reviewed, like the other agent prompts, because the prompt IS the
#: experimental treatment: a run is only comparable to another run of the same version.
PROMPT_VERSION = 1


def prompt_path(version: int = PROMPT_VERSION) -> Path:
    from merlin.common.paths import prompts_dir

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
        return {
            "module": self.module,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "sandboxed": self.sandboxed,
            "usage": self.usage,
            "error": self.error,
            "transcript": self.transcript_path,
            "proposed": self.proposal is not None,
            "source_digest": (
                None
                if self.proposal is None
                else __import__("hashlib").sha256(self.proposal.source.encode()).hexdigest()[:16]
            ),
        }


def _fmt_evidence(evidence: "list[str] | tuple[str, ...] | None") -> str:
    if not evidence:
        return (
            "None recorded. Reason from the axis and the module alone; do not invent a measurement to justify a change."
        )
    return "\n".join(f"- {e}" for e in evidence)


def _fmt_feedback(feedback: str | None) -> str:
    """The previous refusal, or a line saying this is the first attempt.

    Rendered rather than omitted so the card's shape is identical on every turn: a section that
    appears and disappears between turns is a change the agent has to interpret, on top of the change
    that matters.
    """
    if not feedback:
        return "This is the first attempt on this seam. Nothing has been refused yet."
    return (
        feedback + "\n\nThat verdict is machine-checked: it comes from running your predecessor's "
        "pass and reading the emitted assembly, not from anyone's opinion of the code. Nothing "
        "above suggests a fix -- work out for yourself what it implies."
    )


def build_prompt(
    action, *, evidence=None, ours=None, divergence=None, feedback: str | None = None, version: int = PROMPT_VERSION
) -> str:
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
    return (
        tmpl.replace("{axis}", str(axis))
        .replace("{intended_facet}", json.dumps(facet, sort_keys=True))
        .replace("{ours}", "(not recorded)" if ours is None else str(ours))
        .replace("{expert}", str(facet.get(axis, "(see intended_facet)")))
        .replace("{change}", str(getattr(action, "change", "?")))
        .replace("{feedback}", _fmt_feedback(feedback))
        .replace("{evidence}", _fmt_evidence(evidence))
    )


def sandbox_argv(ws: Path) -> list[str] | None:
    """The bwrap prefix isolating one proposer turn, or None when the sandbox is not USABLE.

    Deny-by-default with an EMPTY bundle, which is the strongest form: the agent needs nothing but
    its workspace, since the task card carries the promise and the workspace carries the module. That
    also means no path has to be re-audited when one is added elsewhere.

    "Usable" is an execution, not a file. Asking whether the binary is on PATH is a different question,
    and on a host whose policy forbids unprivileged user namespaces the two answers differ: the binary
    is installed, so the presence check passes and no refusal fires, while every invocation dies at
    "setting up uid map: Permission denied" before the inner command starts. The run then proceeds with
    a prefix that cannot run, every turn exits nonzero, and the outcome is reported as the agent having
    failed to produce anything. See :mod:`merlin.development.targetgen.sandbox.preflight`.
    """
    from merlin.development.targetgen.sandbox.preflight import probe_sandbox

    if not probe_sandbox().usable:
        return None
    from merlin.common.paths import repo_root
    from merlin.development.targetgen.sandbox import grants as bwrap

    # `base_argv` binds the workspace LAST on purpose, so no mask can clobber it. Appending the CLI's
    # runtime binds after it breaks that ordering, so re-bind the workspace at the very end. Binding
    # the same path twice is harmless and keeps the invariant explicit rather than depending on the
    # runtime binds never happening to overlap the workspace.
    return bwrap.base_argv(ws, {}, repo=repo_root()) + bwrap.claude_runtime_binds() + ["--bind", str(ws), str(ws)]


def _sandbox_condition() -> str:
    """The named sandbox condition, for a refusal that distinguishes "not installed" from "installed
    and cannot execute" from "could not be determined". Those are three different operator actions."""
    from merlin.development.targetgen.sandbox.preflight import probe_sandbox

    return probe_sandbox().describe()


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

    return extract_code_block(text, "python")


def propose_pass(
    action,
    *,
    module: str,
    current_source: str,
    workspace: Path,
    model: str = "opus",
    timeout: int = 1800,
    version: int = PROMPT_VERSION,
    require_sandbox: bool = True,
    ours=None,
    divergence=None,
    feedback: str | None = None,
    runner: Callable[..., dict] | None = None,
) -> ProposalAttempt:
    """Run ONE proposer turn for ``action`` and return the attempt record.

    ``require_sandbox`` defaults True and REFUSES to run unsandboxed: an agentic run that can read the
    enclosing checkout has already been observed reading another arm's results and the study's own
    status file, and a proposer that can reach a golden makes its own gate meaningless. Pass False
    only for a test with an injected ``runner``.
    """
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    # A retry must not inherit the previous turn's output: a turn that writes nothing would otherwise
    # be credited with its predecessor's proposal, and the gate would re-refuse the same source while
    # the record showed two independent attempts.
    stale = ws / PROPOSAL_FILENAME
    if stale.is_file():
        stale.unlink()
    (ws / "current_pass.py").write_text(current_source, encoding="utf-8")
    prompt = build_prompt(
        action,
        evidence=getattr(action, "evidence", None),
        ours=ours,
        divergence=divergence,
        feedback=feedback,
        version=version,
    )
    (ws / "TASK.md").write_text(prompt, encoding="utf-8")

    sandbox = sandbox_argv(ws)
    if sandbox is None and require_sandbox:
        return ProposalAttempt(
            None,
            module,
            version,
            model,
            False,
            error=(
                f"sandbox is not usable and require_sandbox is set; refusing to run a proposer "
                f"that could read the enclosing checkout — {_sandbox_condition()}"
            ),
        )

    if runner is not None:
        out = runner(prompt=prompt, workspace=ws, model=model, timeout=timeout, sandbox=sandbox)
    else:
        # cache-buster so repeated turns on the same seam do not serve one another's answer
        argv = list(sandbox or []) + [
            "claude",
            "-p",
            f"<!-- nonce: {uuid.uuid4().hex} -->\n{prompt}",
            "--model",
            model,
            "--output-format",
            "json",
        ]
        env = dict(os.environ)
        try:
            # stdin=DEVNULL, not inherited. Headless `claude -p` waits on stdin for piped input and
            # then exits 1 with "no stdin data received in 3s" -- which cost a full 561 s turn and
            # looked like an agent failure rather than a launch bug. A detached/nohup parent has no
            # usable stdin, so the prompt must arrive by argv alone and stdin must be closed.
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, cwd=str(ws), env=env, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            return ProposalAttempt(
                None, module, version, model, sandbox is not None, error=f"proposer timed out after {timeout}s"
            )
        raw = proc.stdout or ""
        tp = ws / "agent_transcript.json"
        tp.write_text(raw, encoding="utf-8")
        if proc.returncode != 0:
            return ProposalAttempt(
                None,
                module,
                version,
                model,
                sandbox is not None,
                error=f"claude exited {proc.returncode}: {(proc.stderr or '')[-800:]}",
                transcript_path=str(tp),
            )
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
            try:
                obj = json.loads(lines[-1]) if lines else {}
            except json.JSONDecodeError:
                return ProposalAttempt(
                    None,
                    module,
                    version,
                    model,
                    sandbox is not None,
                    error="proposer output was not JSON",
                    transcript_path=str(tp),
                )
        out = {
            "text": obj.get("result") or obj.get("text") or "",
            "usage": obj.get("usage", {}),
            "transcript_path": str(tp),
        }

    text = out.get("text") or ""
    usage = out.get("usage") or {}
    tp = out.get("transcript_path")
    if not text.strip() and not (ws / PROPOSAL_FILENAME).is_file():
        return ProposalAttempt(
            None,
            module,
            version,
            model,
            sandbox is not None,
            usage=usage,
            error=f"proposer returned no text and wrote no {PROPOSAL_FILENAME}",
            transcript_path=tp,
        )
    try:
        source = _extract_module_source(text, ws)
    except Exception as e:  # noqa: BLE001 - any extraction failure is an honest refusal, not a crash
        return ProposalAttempt(
            None,
            module,
            version,
            model,
            sandbox is not None,
            usage=usage,
            error=f"no usable python block in the reply: {e}",
            transcript_path=tp,
        )
    if not source.strip():
        return ProposalAttempt(
            None,
            module,
            version,
            model,
            sandbox is not None,
            usage=usage,
            error="the python block was empty",
            transcript_path=tp,
        )
    rationale = text.split("```", 1)[0].strip()[:1200]
    if not source.endswith("\n"):
        source += "\n"  # the extractor strips; a module file ends with a newline
    return ProposalAttempt(
        PassProposal(module=module, source=source, rationale=rationale),
        module,
        version,
        model,
        sandbox is not None,
        usage=usage,
        transcript_path=tp,
    )


def proposer_for(
    action, *, current_source: str, workspace: Path, **kw
) -> tuple[Callable[[Any], PassProposal | None], list[ProposalAttempt]]:
    """A ``propose(action) -> PassProposal | None`` for :func:`pass_slot.run_pass_slot`, plus the list
    the attempt record lands in. ``run_pass_slot`` turns None into an honest ``no_proposal`` verdict,
    so a failed turn needs no special handling by the caller -- but its record is still kept."""
    from merlin.analysis.kernels.cca import action_catalog as ac

    module = ac.seam_module(getattr(action, "target_seam", "") or "")
    attempts: list[ProposalAttempt] = []

    def _propose(a, *, feedback: str | None = None):
        att = propose_pass(
            a, module=module or "?", current_source=current_source, workspace=Path(workspace), feedback=feedback, **kw
        )
        attempts.append(att)
        return att.proposal

    return _propose, attempts


# ---- agent_output ------------------------------------------------------------------------------
# Structured output from an LLM/agent turn — constrain + parse, never regex-scrape prose.
#
# The old pattern was: run a free-text turn, then regex a JSON object/array out of the reply
# (``re.search(r"\{.*\}")`` and friends). That is brittle (greedy, fence-blind, prose-sensitive)
# and unprincipled. This module replaces it two ways:
#
#   * :func:`parse_json` / :func:`extract_code_block` — regex-free *extraction* from a reply that may
#     be fenced or wrapped in prose. A leading ```` ``` ```` fence is stripped structurally; a JSON
#     value embedded in prose is found by a string-aware balanced-delimiter scan (the first complete
#     ``{...}``/``[...]``), not a greedy ``.*``.
#   * :func:`structured_agent_call` — the *proper* path when we control the turn: instruct the model
#     to answer with ONLY a JSON value matching a JSON Schema, parse it, validate it, and retry once
#     with the error fed back. Callers keep a deterministic fallback: on exhaustion it raises
#     :class:`StructuredOutputError` rather than returning a silently-wrong value.
#
# No ``re`` here by construction — this is part of the de-regex sweep.

_RAISE = object()  # sentinel: no tolerant default -> raise on failure


class StructuredOutputError(RuntimeError):
    """The agent was unavailable, or never produced parseable/schema-valid output."""


def strip_code_fence(text: str) -> str:
    """Return the body of the first ```` ``` ```` fenced block, else the stripped text (regex-free).

    Handles ```` ```json ```` / ```` ``` ```` openers; if there is no closing fence, returns
    everything after the opener line."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    nl = s.find("\n")
    if nl == -1:
        return s
    body = s[nl + 1 :]
    end = body.rfind("```")
    return (body[:end] if end != -1 else body).strip()


def locate_json(s: str) -> str | None:
    """The first complete balanced ``{...}`` or ``[...]`` substring, or None (string-aware, regex-free).

    Skips braces inside JSON string literals (honoring ``\\`` escapes), so it stops at the matching
    close of the first structure rather than the last one a greedy ``.*`` would grab."""
    start = open_close = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


def parse_json(text: str | None, *, default: Any = _RAISE, schema: dict | None = None) -> Any:
    """Extract a JSON value from a (possibly fenced / prose-wrapped) agent reply, regex-free.

    Strategy: strip a leading code fence; try ``json.loads`` on the whole; else locate the first
    balanced JSON structure and parse that. With ``schema`` and ``jsonschema`` installed, the value
    is validated. On any failure return ``default`` when one was given (tolerant), otherwise raise
    :class:`StructuredOutputError`."""

    def _fail(msg: str):
        if default is _RAISE:
            raise StructuredOutputError(msg)
        return default

    if not text:
        return _fail("empty agent reply")
    blob = strip_code_fence(text)
    value = _RAISE
    try:
        value = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        located = locate_json(blob)
        if located is not None:
            try:
                value = json.loads(located)
            except (json.JSONDecodeError, TypeError):
                value = _RAISE
    if value is _RAISE:
        return _fail(f"reply was not valid JSON; head={blob[:200]!r}")
    if schema is not None:
        errs = _schema_errors(value, schema)
        if errs:
            return _fail("schema validation failed: " + "; ".join(errs))
    return value


def extract_code_block(text: str, lang: str | None = None, *, default: Any = _RAISE) -> str:
    """Body of the first ```` ``` ```` fenced code block, preferring one tagged ``lang`` (regex-free).

    Falls back to the first block of any language. On no block, returns ``default`` if given, else
    raises :class:`StructuredOutputError`."""
    blocks = _fenced_blocks(text or "")
    if not blocks:
        if default is _RAISE:
            raise StructuredOutputError("agent reply contained no fenced code block")
        return default
    if lang is not None:
        for info, body in blocks:
            if info.strip().lower() == lang.lower():
                return body
    return blocks[0][1]


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """All ```` ``` ```` blocks as (info-string, body). Regex-free line scan."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("```"):
            info = stripped[3:].strip()
            body_lines = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body_lines.append(lines[i])
                i += 1
            out.append((info, "\n".join(body_lines)))
            i += 1  # skip the closing fence
        else:
            i += 1
    return out


def _schema_errors(value: Any, schema: dict) -> list[str]:
    """JSON-Schema validation errors (empty if valid or jsonschema is unavailable)."""
    try:
        import jsonschema
    except Exception:
        return []
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in validator.iter_errors(value)]


def _instruction(schema: dict | None) -> str:
    base = "Respond with ONLY a single JSON value and nothing else — no prose, no explanation, no markdown code fences."
    if schema is not None:
        return base + " It MUST conform to this JSON Schema:\n" + json.dumps(schema, indent=2)
    return base


Runner = Callable[[str], "str | None"]  # prompt -> reply text (None when the agent is unavailable)


def structured_agent_call(
    runner: Runner, prompt: str, schema: dict | None = None, *, retries: int = 1, instruction: str | None = None
) -> Any:
    """Run ``runner(prompt)`` demanding a JSON reply matching ``schema``; parse, validate, retry.

    The turn is constrained (the schema/instruction is appended to ``prompt``) instead of the reply
    being regex-scraped afterward. Returns the parsed — and, when a schema is supplied and
    ``jsonschema`` is installed, validated — value. Retries up to ``retries`` times, feeding the
    parse/validation error back into the prompt. Raises :class:`StructuredOutputError` when the
    agent is unavailable or never returns valid output (callers keep a deterministic fallback)."""
    instr = instruction if instruction is not None else _instruction(schema)
    last_err = "no attempt made"
    for attempt in range(retries + 1):
        suffix = (
            instr
            if attempt == 0
            else (f"{instr}\n\nYour previous reply was rejected: {last_err}. Reply with JSON only.")
        )
        reply = runner(f"{prompt}\n\n{suffix}")
        if not reply:
            raise StructuredOutputError("agent returned no text (unavailable?)")
        try:
            return parse_json(reply, schema=schema)
        except StructuredOutputError as e:
            last_err = str(e)
    raise StructuredOutputError(f"no schema-valid reply after {retries + 1} attempt(s): {last_err}")
