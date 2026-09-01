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


def scan_cheats(source: str, *, models: tuple[str, ...] | None = None) -> list[str]:
    """Tokens in the proposal that would let it pass the gate without doing the work.

    Structural and cheap, run FIRST: a proposal that reads the golden or hardcodes a model name is
    rejected before any build. ``models`` defaults to the captures on disk, so this generalises as the
    corpus grows instead of being a literal list someone has to remember to extend.
    """
    found = [t for t in CHEAT_TOKENS if t in source]
    lowered = source.lower()
    for m in (models if models is not None else model_name_tokens()):
        # a model NAME in a compiler pass is overfit by construction
        if m and m.lower() in lowered:
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
         heldout_ok: Callable[[PassProposal], tuple[bool, str]] | None = None,
         models: tuple[str, ...] | None = None) -> PassVerdict:
    """Run the ordered gate. Cheapest disqualifier first; every stage fails CLOSED.

    The four checks are injected rather than imported so the gate is testable without a toolchain, a
    board, or an agent -- the same reason ``critic.py`` injects its runner. In production they are:
    ``frozen_baseline_ok``  -> empty features still lower byte-identically
    ``bit_exact_ok``        -> spike run matches the golden
    ``lift_cca``            -> cca.lift_asm over the emitted disassembly
    ``heldout_ok``          -> the same two checks on captures the agent never saw
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
        return PassVerdict(False, "bit_exact", f"numerics changed: {why}")
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
