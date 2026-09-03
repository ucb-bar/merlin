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

THE DISCIPLINE, inherited from ``targetgen.agent.kernel_slot``: agent autonomy on visible data, a
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Tokens that mean the proposal is reaching for the answer instead of computing it. Narrower than
#: kernel_slot's list because a compiler pass legitimately imports numpy and legitimately contains the
#: word "reference" in prose -- so this keys on the specific escapes that would let a pass special-case
#: its way to a passing gate.
CHEAT_TOKENS: tuple[str, ...] = (
    "golden.npy",            # reading the answer
    "region_goldens",
    "achieved_residual",     # asserting its own gate verdict
    "intended_facet",
    "_REGISTRY[",            # mutating the feature registry to fake a facet
    "monkeypatch",
)

#: A pass that names a MODEL is overfit by construction: the whole point of the loop is that a lever
#: found on one model transfers. Checked against the corpus's own model list rather than a literal set,
#: so a new capture is covered without editing this file.
def model_name_tokens() -> tuple[str, ...]:
    """Model names the proposal must not mention, derived from the captures actually on disk."""
    try:
        from ..baselines import bundle as _b
        return tuple(sorted(_b.known_models()))
    except Exception:  # noqa: BLE001 - a fresh checkout with no registry still gates on CHEAT_TOKENS
        return ()


@dataclass
class PassProposal:
    """What the agent returns: replacement source for ONE pass module, and why."""
    module: str                      # dotted module the source replaces, e.g. merlin.llvmlower.act_poly
    source: str                      # the full new module source
    rationale: str = ""


@dataclass
class PassVerdict:
    """Why the proposal was accepted or refused. Ordered, so the first failure is the cheapest one."""
    accepted: bool
    stage: str                       # cheat | frozen_baseline | bit_exact | facet | heldout | accepted
    reason: str = ""
    residual: tuple[str, ...] = ()   # axes promised but not achieved (from achieved_residual)
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
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
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
                _add_identifier("".join(cur)); cur = []
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
    for m in (models if models is not None else model_name_tokens()):
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
    from ..kernels import action_catalog as ac

    if not getattr(action, "intended_facet", None):
        return PassVerdict(False, "unverifiable",
                           "the action carries no intended_facet, so there is no machine-checkable "
                           "promise to hold the proposal to; refusing rather than crediting it")
    residual = tuple(ac.achieved_residual(action, achieved_cca))
    if residual:
        return PassVerdict(False, "facet",
                           f"promised {action.intended_facet} but the emitted code did not achieve "
                           f"{list(residual)}", residual=residual)
    return PassVerdict(True, "accepted", "the emitted code achieved the promised facet")


def gate(proposal: PassProposal, action, *,
         frozen_baseline_ok: Callable[[PassProposal], bool],
         bit_exact_ok: Callable[[PassProposal], tuple[bool, str]],
         lift_cca: Callable[[PassProposal], Any],
         inert_ok: Callable[[PassProposal], tuple[bool, str]] | None = None,
         heldout_ok: Callable[[PassProposal], tuple[bool, str]] | None = None,
         models: tuple[str, ...] | None = None) -> PassVerdict:
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
        return PassVerdict(False, "cheat", f"proposal references forbidden tokens {cheats}",
                           detail={"tokens": cheats})
    if not frozen_baseline_ok(proposal):
        return PassVerdict(False, "frozen_baseline",
                           "an empty feature set no longer lowers byte-identically, so every "
                           "measurement taken against the control would silently move")
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
            return PassVerdict(False, "heldout",
                               f"held on the visible captures but not on held-out ones: {why}")
    return PassVerdict(True, "accepted", "gate passed: no cheats, baseline frozen, numerics "
                                         "bit-exact, promised facet achieved"
                                         + (", held out" if heldout_ok is not None else ""))


def run_pass_slot(action, *, propose: Callable[[Any], PassProposal | None] | None = None,
                  **gate_kwargs) -> tuple[PassProposal | None, PassVerdict]:
    """Ask for a proposal and gate it. ``propose=None`` -> no proposal, and an honest refusal.

    Returns ``(proposal, verdict)`` so a caller can record BOTH -- a refused proposal is evidence about
    the seam, not noise, and the beam already records non-actionable outcomes as work-items rather than
    dropping them.
    """
    if propose is None:
        return None, PassVerdict(False, "no_proposal",
                                 "no proposer supplied; the gate is a no-op without one (this is the "
                                 "default so the gate is testable and usable with no agent budget)")
    proposal = propose(action)
    if proposal is None:
        return None, PassVerdict(False, "no_proposal", "the proposer returned nothing")
    return proposal, gate(proposal, action, **gate_kwargs)
