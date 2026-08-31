"""The fork protocol: which functional compiler a performance run forked from, and what it may not spend.

Two capsule systems, one hard boundary. **Phase F** proves a compiler functionally complete; that
exact compiler is then FORKED and **Phase P** optimizes it. The optimizer is allowed to change
anything about how fast the compiler's output runs and nothing about whether it is right, so the fork
has to be a RECORD rather than a convention -- a fork nobody wrote down is a fork nobody can check.

Three things have to hold, and each is a separate failure this module makes visible:

1. **The fork point is pinned by content, not by name.** A branch moves; bytes do not. The record
   carries the functional submission's whole digest AND its per-component digests
   (:mod:`merlin.targetgen.oracle_schedule`), so "which compiler did these numbers come from" has an
   answer that survives the working tree being edited under it.

2. **A Phase-P edit requeues the SMALL functional capsules that ride on what it touched.** With a
   single whole-submission digest, every edit kills every certificate, so the cycle-accurate tier --
   minutes per capsule -- is re-bought for the whole corpus per edit and the round never finishes.
   Comparing only the components a capsule DECLARED (``depends_on``) requeues the handful that
   actually moved, and the large performance workload is not a functional capsule at all: it is not
   in the fork's capsule set, so :func:`requeue` cannot emit it. Re-proving the compiler is cheap;
   re-running the workload to prove the compiler is the thing that made rounds unaffordable.

3. **"Not re-proven" is not "still holds".** :func:`check_invariants` is tri-state: an invariant is
   HELD only when the candidate's OWN bytes earned the verdict, WEAKENED when they earned a failure,
   and UNDETERMINABLE when nobody re-ran it. Folding the third into the first would let a certificate
   earned by the functional compiler stand for an optimizer that has since rewritten the component it
   depended on -- which is exactly the stale-certificate failure the content addressing exists to
   prevent, reintroduced at the phase boundary.

The tri-state comes out as :attr:`InvariantCheck.ok` (``True`` / ``False`` / ``None``) and feeds
:func:`merlin.perf.falsifier.ab_decision` unchanged, where ``None`` does not promote.

Nothing here names a target, a tier, a component or a capsule: the tier order comes from the target's
own adapter map and the component vocabulary from the submission's own manifest.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from merlin.targetgen.oracle_schedule import (
    CHANGED, NO_VERDICT, PASS, UNDETERMINABLE, UNKNOWN, WHOLE_SUBMISSION,
    CapsuleState, Staleness, Verdict, explain,
)

__all__ = [
    "ForkPoint", "HELD", "InvariantCheck", "UNDETERMINABLE", "WEAKENED", "candidate_states",
    "changed_components", "check_invariants", "fork_from", "fork_from_dict", "requeue",
]

#: The three states of "did the candidate keep what Phase F proved". ``UNDETERMINABLE`` is imported
#: rather than redefined so a caller cannot end up comparing two spellings of the same word.
HELD = "held"
WEAKENED = "weakened"


@dataclass(frozen=True)
class ForkPoint:
    """The functional submission a performance run forked from, and the verdicts it had earned.

    ``invariants`` holds ONLY verdicts that were about the forked bytes themselves -- built through
    :meth:`~merlin.targetgen.oracle_schedule.CapsuleState.known`, so a stale certificate sitting in
    the Phase-F state is not promoted to an invariant by the act of forking. A fork that inherited
    stale certificates would hand Phase P a correctness budget it never had.
    """

    #: Whole-submission digest of the functional compiler at the fork.
    digest: str
    #: ``{component: digest}`` at the fork. Empty when the submission had no computable
    #: decomposition -- which is undeterminable, not "no components", and makes every comparison
    #: fall back to the whole digest.
    components: dict[str, str] = field(default_factory=dict)
    #: ``{capsule: {tier: status}}`` -- the Phase-F verdicts that may not be weakened.
    invariants: dict[str, dict[str, str]] = field(default_factory=dict)
    #: ``{capsule: (component, ...)}`` as each capsule declared it. Absent == depends on everything.
    depends_on: dict[str, tuple[str, ...]] = field(default_factory=dict)
    recorded_at: str = ""
    #: Opaque hardware-provenance block (``merlin.common.provenance.record()``). The Phase-F verdicts
    #: were earned on a specific hardware revision, and a result attributed to the wrong device is
    #: worse than no result because it gets cited.
    provenance: dict | None = None
    detail: str = ""

    @property
    def capsules(self) -> tuple[str, ...]:
        """The Phase-F capsule set. :func:`requeue` never emits work outside it."""
        return tuple(sorted(self.invariants))

    def tiers_for(self, capsule: str) -> tuple[str, ...]:
        return tuple(sorted(self.invariants.get(capsule, {})))

    def to_dict(self) -> dict:
        return {"digest": self.digest, "components": dict(self.components),
                "invariants": {c: dict(t) for c, t in self.invariants.items()},
                "depends_on": {c: list(d) for c, d in self.depends_on.items()},
                "recorded_at": self.recorded_at, "provenance": self.provenance,
                "detail": self.detail}


def fork_from_dict(doc: Mapping) -> ForkPoint:
    """Rebuild a fork record read back off disk. The inverse of :meth:`ForkPoint.to_dict`."""
    return ForkPoint(
        digest=str(doc.get("digest") or ""),
        components={str(k): str(v) for k, v in (doc.get("components") or {}).items()},
        invariants={str(c): {str(t): str(s) for t, s in (tiers or {}).items()}
                    for c, tiers in (doc.get("invariants") or {}).items()},
        depends_on={str(c): tuple(str(x) for x in (d or ()))
                    for c, d in (doc.get("depends_on") or {}).items()},
        recorded_at=str(doc.get("recorded_at") or ""),
        provenance=doc.get("provenance"),
        detail=str(doc.get("detail") or ""))


def fork_from(states: Iterable[CapsuleState], *, tier_order: Sequence[str],
              digest: str | None = None,
              components: Mapping[str, str] | None = None,
              provenance: Mapping | None = None,
              recorded_at: str | None = None) -> ForkPoint:
    """Pin the functional submission a performance run is about to fork.

    ``states`` are the Phase-F capsules with their verdicts. ``digest`` / ``components`` default to
    the ones the states already carry; supplying them explicitly is the path for a caller that
    digested the submission itself (``tier_promote.submission_digests``).

    A state whose digest disagrees with the rest RAISES. A fork is a statement about ONE submission,
    and silently forking the majority digest would produce a record that pins bytes some of its own
    invariants were never measured against.
    """
    states = list(states)
    seen = {st.digest for st in states if st.digest}
    if digest is None:
        if len(seen) > 1:
            raise ValueError(f"the capsule states carry {len(seen)} different submission digests "
                             f"({sorted(seen)}); a fork pins one submission, not a majority")
        digest = next(iter(seen), "")
    disagree = sorted({st.name for st in states if st.digest and st.digest != digest})
    if disagree:
        raise ValueError(f"capsule(s) {disagree} were graded against a different submission than the "
                         f"one being forked ({digest!r}); their verdicts are not about these bytes")

    comps: dict[str, str] = dict(components or {})
    if components is None:
        for st in states:
            comps.update(st.components or {})

    invariants: dict[str, dict[str, str]] = {}
    deps: dict[str, tuple[str, ...]] = {}
    for st in states:
        # `known` and not `verdicts`: a verdict recorded against different bytes is not a verdict
        # about the submission being forked, so it is not an invariant Phase P inherits.
        earned = {t: st.known(t) for t in tier_order if st.known(t) == PASS}
        invariants[st.name] = earned
        if st.depends_on:
            deps[st.name] = tuple(st.depends_on)

    if recorded_at is None:
        from merlin.common.artifacts import utc_stamp
        recorded_at = utc_stamp()
    n_pass = sum(len(t) for t in invariants.values())
    return ForkPoint(
        digest=digest, components=comps, invariants=invariants, depends_on=deps,
        recorded_at=recorded_at, provenance=dict(provenance) if provenance is not None else None,
        detail=(f"{len(invariants)} functional capsule(s) carrying {n_pass} tier verdict(s) earned "
                f"by these exact bytes; {len(comps)} component digest(s) recorded"))


def changed_components(fork: ForkPoint, components: Mapping[str, str] | None) -> tuple[Staleness, ...]:
    """What moved between the fork and a candidate, as staleness reasons; empty == nothing moved.

    Fails closed in both directions the scheduler already fails closed in: a side with no computable
    decomposition falls back to the WHOLE submission (undeterminable, never "no components"), and a
    component present on one side only is UNDETERMINABLE rather than assumed unchanged.
    """
    now = dict(components or {})
    if not fork.components or not now:
        return (Staleness(WHOLE_SUBMISSION, UNDETERMINABLE),)
    out: list[Staleness] = []
    for name in sorted(set(fork.components) | set(now)):
        then, cur = fork.components.get(name), now.get(name)
        if then is None or cur is None:
            out.append(Staleness(name, UNDETERMINABLE))
        elif then != cur:
            out.append(Staleness(name, CHANGED))
    return tuple(out)


def candidate_states(fork: ForkPoint, *, digest: str,
                     components: Mapping[str, str] | None = None,
                     verdicts: Mapping[str, Mapping[str, str]] | None = None) -> list[CapsuleState]:
    """The Phase-F capsules as they stand against a Phase-P candidate's bytes.

    Each state carries the CANDIDATE's digest and component map as "what is on disk now", and the
    FORK's component digests inside every inherited verdict as "what earned this". The scheduler's
    content addressing then does the rest, and a capsule survives an edit to a component it never
    declared -- which is the entire saving.

    ``verdicts`` optionally overlays what the candidate has re-earned (``{capsule: {tier: status}}``,
    recorded against the candidate's own bytes). Anything not overlaid keeps the fork's verdict,
    which the digest comparison will invalidate if the edit touched it.
    """
    comps = dict(components or {})
    out: list[CapsuleState] = []
    for name in fork.capsules:
        vs: dict[str, Verdict] = {}
        for tier, status in fork.invariants[name].items():
            vs[tier] = Verdict(status=status, digest=fork.digest, components=dict(fork.components))
        for tier, status in ((verdicts or {}).get(name) or {}).items():
            # Re-earned against the candidate's OWN bytes, so it is stamped with them.
            vs[str(tier)] = Verdict(status=str(status), digest=digest, components=dict(comps))
        out.append(CapsuleState(name=name, digest=digest, verdicts=vs, components=comps,
                                depends_on=fork.depends_on.get(name)))
    return out


def requeue(fork: ForkPoint, *, digest: str, tier_order: Sequence[str],
            components: Mapping[str, str] | None = None,
            verdicts: Mapping[str, Mapping[str, str]] | None = None,
            cert_tiers: Sequence[str] = (), cert_cover=None,
            cost_s: Mapping[str, float] | None = None,
            budget_s: float | None = None) -> dict:
    """What a Phase-P candidate has to re-prove, and what it demonstrably does not.

    Returns :func:`~merlin.targetgen.oracle_schedule.explain`'s report with the fork's own framing
    attached: which components moved, and the assertion that the queue is drawn from the functional
    capsule set only. A scheduler that silently drops work is indistinguishable from one that has
    finished, so ``unchanged`` is reported beside ``queue`` rather than left implicit.
    """
    states = candidate_states(fork, digest=digest, components=components, verdicts=verdicts)
    report = explain(states, tier_order=list(tier_order), cert_tiers=tuple(cert_tiers),
                     cert_cover=cert_cover, cost_s=dict(cost_s or {}), budget_s=budget_s)
    moved = changed_components(fork, components)
    report["fork"] = {"digest": fork.digest, "recorded_at": fork.recorded_at}
    report["components_moved"] = [{"component": s.component, "reason": s.reason} for s in moved]
    report["scope"] = ("the Phase-F functional capsules only. The performance workload is not a "
                       "functional capsule, so an edit to the compiler re-proves the small capsules "
                       "that declared the component it touched, never the large workload")
    report["capsules_in_scope"] = list(fork.capsules)
    return report


@dataclass(frozen=True)
class InvariantCheck:
    """Whether a Phase-P candidate still holds everything Phase F proved. Tri-state throughout."""

    state: str
    #: ``(capsule, tier)`` that PASSED at the fork and FAILS for the candidate's bytes.
    weakened: tuple[tuple[str, str], ...] = ()
    #: ``(capsule, tier, why)`` -- passed at the fork, and no verdict about the candidate's bytes.
    unproven: tuple[tuple[str, str, str], ...] = ()
    #: ``(capsule, tier)`` re-earned by the candidate's own bytes.
    held: tuple[tuple[str, str], ...] = ()
    #: Phase-F capsules the candidate reported nothing at all about.
    missing: tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool | None:
        """``True`` held / ``False`` weakened / ``None`` undeterminable.

        Feeds :func:`merlin.perf.falsifier.ab_decision` as ``invariants_held``. The ``None`` is
        load-bearing: there it refuses to promote, rather than reading as a pass.
        """
        return {HELD: True, WEAKENED: False}.get(self.state)

    def to_dict(self) -> dict:
        return {"state": self.state, "ok": self.ok, "reason": self.reason,
                "weakened": [list(x) for x in self.weakened],
                "unproven": [list(x) for x in self.unproven],
                "held": [list(x) for x in self.held], "missing": list(self.missing)}


def check_invariants(fork: ForkPoint, states: Iterable[CapsuleState], *,
                     provenance: Mapping | None = None) -> InvariantCheck:
    """Has this candidate weakened anything Phase F proved?

    ``states`` are the candidate's capsules with whatever it has re-earned. Each Phase-F invariant is
    resolved through :meth:`~merlin.targetgen.oracle_schedule.CapsuleState.known`, so a verdict
    counts only if the candidate's OWN bytes earned it -- the inherited certificate is invalidated by
    the component digests the moment the optimizer touches something the capsule declared.

    Among the per-capsule outcomes ``WEAKENED`` dominates: one proven regression settles the question
    however many other capsules are unproven. A single unproven invariant is otherwise enough to
    refuse, because "we did not check" is not "it still holds". A hardware-revision mismatch refuses
    the WHOLE check ahead of either, since a failure observed on another device is not evidence about
    this one.
    """
    by_name = {st.name: st for st in states}
    weakened: list[tuple[str, str]] = []
    unproven: list[tuple[str, str, str]] = []
    held: list[tuple[str, str]] = []
    missing = tuple(c for c in fork.capsules if c not in by_name)

    for capsule in fork.capsules:
        st = by_name.get(capsule)
        for tier in fork.tiers_for(capsule):
            if st is None:
                unproven.append((capsule, tier, "the candidate reported nothing about this capsule"))
                continue
            status = st.known(tier)
            if status == PASS:
                held.append((capsule, tier))
            elif status == UNKNOWN:
                stale = st.invalidated_by(tier)
                why = (", ".join(str(s) for s in stale) if stale else NO_VERDICT)
                unproven.append((capsule, tier,
                                 f"no verdict for the candidate's bytes ({why})"))
            else:
                weakened.append((capsule, tier))

    # The Phase-F verdicts were earned on a hardware revision. Comparing a candidate graded on a
    # different one is comparing two devices, and a result attributed to the wrong device gets cited.
    # Only checked when the fork pinned one -- a fork that recorded no provenance cannot be made to
    # have recorded one after the fact, and saying so is better than inventing agreement.
    prov_note = ""
    if fork.provenance is not None:
        if provenance is None:
            return InvariantCheck(UNDETERMINABLE, tuple(weakened), tuple(unproven), tuple(held),
                                  missing,
                                  reason=("the fork pinned a hardware revision and the candidate "
                                          "stated none, so its verdicts cannot be shown to be about "
                                          "the same device"))
        if dict(provenance) != dict(fork.provenance):
            return InvariantCheck(UNDETERMINABLE, tuple(weakened), tuple(unproven), tuple(held),
                                  missing,
                                  reason=("the candidate was graded on a different hardware revision "
                                          "than the fork; these verdicts are about two devices"))
        prov_note = " on the hardware revision the fork pinned"

    if weakened:
        return InvariantCheck(WEAKENED, tuple(weakened), tuple(unproven), tuple(held), missing,
                              reason=(f"{len(weakened)} Phase-F invariant(s) that passed at the fork "
                                      f"now fail: {weakened}"))
    if unproven:
        return InvariantCheck(UNDETERMINABLE, tuple(weakened), tuple(unproven), tuple(held), missing,
                              reason=(f"{len(unproven)} Phase-F invariant(s) have no verdict for the "
                                      "candidate's bytes; not re-proven is not still-holds"))
    return InvariantCheck(HELD, tuple(weakened), tuple(unproven), tuple(held), missing,
                          reason=(f"all {len(held)} Phase-F invariant(s) were re-earned by the "
                                  f"candidate's own bytes{prov_note}"))
