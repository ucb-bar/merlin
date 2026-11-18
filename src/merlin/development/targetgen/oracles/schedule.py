"""What oracle work to run next, and what not to run at all.

The round-based loop spends its time badly in three separate ways, all measured on this repo's SIMT
target during a live A/B:

  * **It re-certifies work that did not change.** Every self-check copies the submission to a fresh
    temp dir, rebuilds it, and re-grades every capsule from scratch. An agent that fixes one capsule pays
    to re-prove the other thirty-four.
  * **It pays the expensive tier for capsules that cannot use it.** The ladder ran the cycle-accurate
    tier even for capsules whose numerics had already failed; a sweep scoring 18/35 bought RTL time for
    all 35.
  * **It batches.** The verdict for a fix made early in a round arrives hours later, at the round
    boundary, by which time the agent has moved on.

Together those were 80% of one agent round's wall clock, and 34% of that round returned no verdict at
all (a broker timeout and a sweep the agent abandoned).

This module is the decision core that replaces the barrier: given what is known, what changed, and what
each tier costs, it returns the work queue in the order that buys the most information per second. It is
deliberately pure -- no I/O, no oracle, no broker -- so the policy can be tested exhaustively and the
plumbing can be wired around it.

Three rules do the work:

1. **Content-address every verdict.** A verdict is keyed by ``(capsule, digest, tier)`` where ``digest``
   is the submission bytes that produced it. Unchanged bytes therefore need no re-run *ever*, and changed
   bytes invalidate exactly the capsules they affect -- not the corpus. ``depends_on`` narrows WHICH bytes
   that is: only the named components' digests are compared (see :meth:`CapsuleState.invalidated_by`), and
   naming nothing means the whole submission. The caller supplies it from the submission's own
   decomposition -- every component, so that the bytes the decomposition PROVED no command can read stop
   invalidating anything. It is NOT a per-capsule claim: a capsule cannot honestly say it does not ride on
   a command, because the grader runs every command for every capsule. (``capsule.yaml`` briefly offered a
   ``depends_on`` block; it was retired for exactly that reason -- see ``tier_promote``.) Without any of
   this, an optimization pass that edits the compiler on every iteration re-buys the cycle-accurate tier
   for the whole corpus per edit, and a round that re-certifies 35 capsules at minutes each to learn about
   one never finishes.
2. **The cheap tier gates the expensive one.** A deeper tier is scheduled only for a capsule that has
   PASSED the shallower one, so RTL time is only ever spent on work that could plausibly certify.
3. **The expensive tier runs on a representative subset.** Full coverage at the functional tier,
   representative coverage at the cert tier (see
   :func:`~merlin.development.targetgen.corpus.materialization.cert_capsule_cover`) -- the hardware cannot tell two
   capsules in the same (family, dtype) cell apart, so certifying both buys nothing.

Ordering is by information density -- expected information divided by expected seconds -- which puts a
cheap unknown ahead of an expensive unknown without ever starving the expensive one, because a capsule
that has no verdict at a tier keeps its claim in the queue until it gets one.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.development.targetgen.oracles import tier_policy

# ---- oracle_schedule ---------------------------------------------------------------------------

# A tier's verdict can be one of these; only PASS promotes.
PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"

# Pseudo-components. These are NOT names from a target's component vocabulary -- they are the two buckets
# that exist whether or not anything was declared, and they are spelled with delimiters no path-derived
# component name can produce so they can never collide with a real one.
WHOLE_SUBMISSION = "<whole-submission>"  # "this capsule rides on every byte" -- the undeclared default
UNATTRIBUTED = "<unattributed>"  # submission bytes no declared component claims
EXECUTION_ARTIFACT = "<execution-artifact>"  # the exact per-capsule executable + hardware identity

# The reserved key under which a capsule's per-identity certificate LEDGER sits inside a run's
# ``qa/tier_state.json`` (``{capsule: {CERT_LEDGER: {tier: {identity: entry}}}}``). Spelled with the same
# delimiters no tier label can produce, so it can never be mistaken for a tier.
#
# It lives HERE, beside the vocabulary the scheduler already owns, because the file has two independent
# parties: the harness's ``tier_promote`` writes it, and ``merlin.development.targetgen.oracles.tier_cache`` reads it to
# decide whether a tier may be carried instead of re-executed. Two spellings of this key would not fail
# loudly -- the reader would simply find nothing and re-buy every certificate, which is indistinguishable
# from a run with nothing to reuse.
CERT_LEDGER = "<certs>"

# Why a verdict is not a verdict about the current bytes. Kept as three distinct reasons because they are
# three distinct states: NO_VERDICT is not-yet-known (nobody has run it), CHANGED is known-to-be-stale,
# and UNDETERMINABLE is "the decomposition could not be computed, so staleness cannot be decided". The
# last one must never be folded into either of the others: read as NO_VERDICT it loses the fact that the
# tooling failed, and read as "fresh" it would let a stale certificate stand.
CHANGED, UNDETERMINABLE, NO_VERDICT = "changed", "undeterminable", "no_verdict"

# A recorded STATUS (not a verdict) for a cert-tier record whose own job finished but whose result could
# not be attributed to it. A ``pending`` record is rightly treated as known -- its job is in flight and
# must not be bought twice -- but once that job has come back unattributable nothing else will resolve
# it, and counting it as known forever is how a run stranded 101 capsules at "pending" and stopped asking
# for their cert tier. This status reads as NO verdict, so the capsule re-queues, and the reason names it.
UNATTRIBUTED_RESULT = "unattributed"


@dataclass(frozen=True)
class Staleness:
    """One reason a recorded verdict does not describe the submission currently on disk."""

    component: str  # a component name, or one of the pseudo-components above
    reason: str  # CHANGED | UNDETERMINABLE | NO_VERDICT

    def __str__(self) -> str:
        return f"{self.component} ({self.reason})"


@dataclass(frozen=True)
class Verdict:
    """What is known about one (capsule, tier), and for WHICH bytes it was known.

    ``execution_digest`` identifies the exact executable and hardware revision this verdict was earned on,
    when the run that earned it produced one. It is the NARROWEST identity available and it is per
    capsule, which the submission digests are not.

    ``components`` is the per-component digest map AS OF THE RUN THAT EARNED THIS VERDICT. A verdict
    recorded before per-component digests existed carries an empty map, which makes every declared
    dependency UNDETERMINABLE and re-runs the capsule -- the fail-closed direction.
    """

    status: str
    digest: str
    components: dict[str, str] = field(default_factory=dict)
    execution_digest: str | None = None


@dataclass
class CapsuleState:
    """Everything the scheduler knows about one capsule.

    ``execution_digest`` identifies this capsule's CURRENT executable and hardware revision, when the
    verdict reader could compute one. ``digest`` is the whole submission; ``components`` decomposes those
    same bytes by component, and ``depends_on`` is which of those components the certificate rides on --
    supplied by the caller from the decomposition, never claimed by the capsule.

    The three form a ladder, narrowest first: the executable identity distinguishes capsule from capsule;
    the component decomposition distinguishes live bytes from inert ones; the whole submission
    distinguishes nothing and is the fail-closed floor.

    The reason the decomposition exists: Phase P forks a functionally-complete compiler and then edits it
    continuously, and a whole-submission digest makes EVERY edit invalidate EVERY certificate. The cert
    tier is minutes per capsule, so a corpus-wide invalidation per edit means the round never finishes.
    Comparing only the components a capsule declared requeues the handful that actually moved.
    """

    name: str
    digest: str  # the submission bytes currently on disk for it
    verdicts: dict[str, Verdict] = field(default_factory=dict)  # tier -> Verdict
    components: dict[str, str] = field(default_factory=dict)  # component -> digest of the CURRENT bytes
    depends_on: tuple[str, ...] | None = None  # None/() == unset == depends on the whole submission
    execution_digest: str | None = None  # the exact CURRENT executable + hardware identity

    def _dep_components(self) -> tuple[str, ...] | None:
        """The components this capsule's certificate rides on, or ``None`` for "the whole submission".

        FAIL CLOSED, in both directions that matter:

        * an unset (or empty) ``depends_on`` means "depends on everything", never "depends on
          nothing". Getting this backwards makes a stale certificate look valid, which is far worse than
          re-running a capsule that did not need it;
        * a submission with no computable decomposition (``components`` empty) also falls back to the
          whole digest, so a broken/absent component map degrades to today's behaviour rather than
          silently certifying against a decomposition nobody produced.

        ``UNATTRIBUTED`` is appended to every declared set. A submission byte that no component claims
        could be anything -- the entrypoint script, a shared helper, the manifest itself -- so it is a
        dependency of every capsule. Without that, an agent could shrink the attributed set and keep
        certificates alive across edits nobody accounted for.
        """
        if not self.depends_on or not self.components:
            return None
        deps = list(dict.fromkeys(self.depends_on))
        if UNATTRIBUTED in self.components and UNATTRIBUTED not in deps:
            deps.append(UNATTRIBUTED)
        return tuple(deps)

    def invalidated_by(self, tier: str) -> tuple[Staleness, ...]:
        """Why the recorded verdict for ``tier`` is not about the current bytes; empty == it is.

        This is the "which component requeued me" report: a reader of the queue sees the component name,
        not just that something moved.
        """
        v = self.verdicts.get(tier)
        if v is None:
            return (Staleness(WHOLE_SUBMISSION, NO_VERDICT),)
        if v.status == UNATTRIBUTED_RESULT:
            return (
                Staleness(
                    EXECUTION_ARTIFACT if valid_execution_digest(v.execution_digest) else WHOLE_SUBMISSION,
                    UNATTRIBUTED_RESULT,
                ),
            )
        # The executable is the narrowest safe identity for a HARDWARE verdict, and the only one that is
        # per capsule. Source bytes may change without changing the program the cert tier executes;
        # conversely each capsule emits its own executable, so one capsule's changed artifact must not
        # invalidate an unrelated capsule's certificate. BOTH sides must be well-formed SHA-256 values: a
        # missing, legacy or malformed value falls through to the component or whole-submission
        # comparison below, because absence is never evidence of freshness.
        if valid_execution_digest(self.execution_digest) and valid_execution_digest(v.execution_digest):
            return () if self.execution_digest == v.execution_digest else (Staleness(EXECUTION_ARTIFACT, CHANGED),)
        deps = self._dep_components()
        if deps is None:
            return () if v.digest == self.digest else (Staleness(WHOLE_SUBMISSION, CHANGED),)
        out = []
        for c in deps:
            now, then = self.components.get(c), v.components.get(c)
            if now is None or then is None:
                # Either this run could not digest the component (an unknown name, an undeclared
                # component) or the verdict predates the decomposition. Neither is evidence of freshness.
                out.append(Staleness(c, UNDETERMINABLE))
            elif now != then:
                out.append(Staleness(c, CHANGED))
        return tuple(out)

    def known(self, tier: str) -> str:
        """The verdict for the CURRENT bytes, or UNKNOWN. A verdict earned by different bytes is not a
        verdict about this submission -- that is the whole point of content-addressing it."""
        v = self.verdicts.get(tier)
        if v is None or self.invalidated_by(tier):
            return UNKNOWN
        return v.status


def valid_execution_digest(value: object) -> bool:
    """Whether *value* is the lowercase, full-width SHA-256 the verdict readers emit.

    ONE definition, imported by every caller. Two copies of this predicate is how the recorder and the
    scheduler come to disagree about what counts as an identity, and that disagreement always resolves as
    a silent fallback to the whole submission -- which looks exactly like correct conservative behaviour.
    """
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _why(stale: tuple[Staleness, ...]) -> str | None:
    """The human-readable "what requeued this", or ``None`` when nothing was ever known."""
    named = [s for s in stale if s.reason != NO_VERDICT]
    if not named:
        return None
    return "invalidated by " + ", ".join(str(s) for s in named)


@dataclass(frozen=True)
class WorkItem:
    capsule: str
    tier: str
    cost_s: float
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.capsule, self.tier)


def schedule(states, *, tier_order, cert_tiers=(), cert_cover=None, cost_s=None, budget_s=None):
    """The work queue, most-informative-per-second first.

    ``tier_order``  shallow -> deep, e.g. ``["L2", "L3"]``. Order is the caller's (it comes from the
                    target's own adapter map), never assumed here.
    ``cert_tiers``  the subset of ``tier_order`` that is expensive/cycle-accurate. A cert tier is gated
                    on BOTH the shallower tier passing and membership of ``cert_cover``.
    ``cert_cover``  capsule names worth certifying; ``None`` means "no cover computed, certify anything
                    eligible" -- deliberately permissive, because a missing cover is a caller bug and
                    silently certifying nothing would look identical to everything already being done.
    ``cost_s``      measured seconds per tier (``TierResult.timing.sim_active_s`` is the real source).
                    Missing costs fall back to 1.0, which makes the order arbitrary but never wrong.
    ``budget_s``    optional cutoff; items beyond it are dropped from the queue and reported by
                    :func:`explain`, never silently truncated.
    """
    cost_s = cost_s or {}
    items: list[WorkItem] = []

    for st in states:
        prev_tier = None
        for tier in tier_order:
            status = st.known(tier)
            if status != UNKNOWN:
                prev_tier = tier
                continue  # already known FOR THESE BYTES -> never re-run

            if tier in cert_tiers:
                # Gate 1: the shallower tier must have PASSED. An unknown or failed shallow tier means
                # RTL time cannot certify anything -- and a failed one means the capsule is failed however
                # the deeper tier votes.
                if prev_tier is None or st.known(prev_tier) != PASS:
                    break
                # Gate 2: representative coverage. The hardware cannot distinguish two capsules in the
                # same cell, so certifying both spends minutes to learn nothing.
                if cert_cover is not None and st.name not in cert_cover:
                    break
                reason = f"{prev_tier} passed and {st.name} is in the cert cover"
                why = _why(st.invalidated_by(tier))
                if why:  # a certificate that EXISTED and was invalidated
                    reason = f"{reason}; {why}"
            else:
                reason = _why(st.invalidated_by(tier)) or "no verdict for the current submission bytes"

            items.append(WorkItem(st.name, tier, float(cost_s.get(tier, 1.0)), reason))
            break  # one item per capsule at a time: its next open tier

    # Information per second. Every queued item is worth exactly one unknown verdict, so density is 1/cost
    # -- cheap-and-gating first. Ties break on tier order then name, so the queue is deterministic and a
    # replay of the same state produces the same schedule.
    items.sort(key=lambda w: (-1.0 / max(w.cost_s, 1e-9), tier_order.index(w.tier), w.capsule))

    if budget_s is None:
        return items
    out, spent = [], 0.0
    for w in items:
        if spent + w.cost_s > budget_s:
            continue
        out.append(w)
        spent += w.cost_s
    return out


def explain(states, *, tier_order, cert_tiers=(), cert_cover=None, cost_s=None, budget_s=None) -> dict:
    """The schedule plus what was deliberately NOT scheduled, and why.

    A scheduler that silently drops work is indistinguishable from one that has finished. Every exclusion
    is named and counted so the operator log can say "34 capsules need no work because their bytes are
    unchanged" rather than going quiet.
    """
    queued = schedule(
        states, tier_order=tier_order, cert_tiers=cert_tiers, cert_cover=cert_cover, cost_s=cost_s, budget_s=budget_s
    )
    qkeys = {w.key for w in queued}
    unchanged, blocked, not_covered, over_budget = [], [], [], []

    full = schedule(
        states, tier_order=tier_order, cert_tiers=cert_tiers, cert_cover=cert_cover, cost_s=cost_s, budget_s=None
    )
    for w in full:
        if w.key not in qkeys:
            over_budget.append(w.key)

    invalidated = []
    for st in states:
        opens = [t for t in tier_order if st.known(t) == UNKNOWN]
        # WHICH component requeued this capsule. Only tiers that HAD a verdict are reported: a tier
        # nobody ever ran was not invalidated by anything, and listing it as such would drown the signal
        # (every capsule is "unknown" at the cert tier on the first round).
        for t in tier_order:
            if t in st.verdicts:
                for s in st.invalidated_by(t):
                    invalidated.append({"capsule": st.name, "tier": t, "component": s.component, "reason": s.reason})
        if not opens:
            unchanged.append(st.name)
            continue
        for t in opens:
            if t in cert_tiers and (st.name, t) not in qkeys and (st.name, t) not in set(over_budget):
                shallower = [x for x in tier_order if x != t and tier_order.index(x) < tier_order.index(t)]
                if shallower and st.known(shallower[-1]) != PASS:
                    blocked.append((st.name, t))
                elif cert_cover is not None and st.name not in cert_cover:
                    not_covered.append((st.name, t))

    return {
        "queue": [{"capsule": w.capsule, "tier": w.tier, "cost_s": w.cost_s, "reason": w.reason} for w in queued],
        "queued_cost_s": round(sum(w.cost_s for w in queued), 3),
        "unchanged": sorted(unchanged),
        "invalidated_by": sorted(invalidated, key=lambda r: (r["capsule"], r["tier"], r["component"])),
        "blocked_on_shallower_tier": sorted(blocked),
        "outside_cert_cover": sorted(not_covered),
        "deferred_over_budget": sorted(over_budget),
    }


# ---- oracle_blind_spots ------------------------------------------------------------------------
# What each oracle tier of a target cannot see, as data, and the tier a program therefore needs.
#
# A tier that cannot observe a behaviour certifies every program in that behaviour's class, right or
# wrong, and nothing in the result says so. The measured case: one executable passed the functional
# tier with no mismatching output and failed on the FPGA with every output wrong, because the
# functional model pins an accumulator base the RTL rotates. That was known, and it lived in notes.
#
# This module makes it a declaration the grader can act on. A target's contract lists its blind
# spots under ``oracle_blind_spots``; each names the tiers that are blind, and the CLASS of program
# it concerns in the vocabulary the corpus already states about itself (``tier_policy.capsule_axes``:
# an instruction class, a mode, an epilogue stage, an op). From that:
#
# * :func:`blind_tiers` -- which tiers cannot judge this capsule, and why;
# * :func:`required_tier` -- the shallowest declared tier that can; ``None`` when no declared tier can,
#   which is a finding about the ladder and never a pass;
# * :func:`audit` -- capsules whose deepest reachable tier sits inside a blind spot for their own
#   axes. Such a capsule can be screened and can never be certified, whatever its result says.
#
# Nothing here names a target, a tier or an instruction: the entries are the target's data, and a
# target with no entries declares no blind spot and changes nothing.

CONTRACT_KEY = "oracle_blind_spots"
SCHEMA = "oracle_blind_spot_audit_v1"
#: A capsule whose every reachable tier is blind to one of its own axes.
UNCERTIFIABLE = "certifiable_only_inside_blind_spot"
#: No declared tier can see the class at all: the ladder itself is short.
NO_SEEING_TIER = "no_declared_tier_can_see"


class BlindSpotError(ValueError):
    """A registry entry is malformed; says which entry and which field."""


@dataclass(frozen=True)
class BlindSpot:
    """One class of behaviour a set of tiers cannot observe."""

    id: str
    tiers: tuple[str, ...]
    #: ``(axis, value)`` pairs in the capsule-axis vocabulary. A program exercising ANY of them is
    #: in the class: an entry lists the several ways one behaviour is reached.
    axes: tuple[tuple[str, str], ...]
    what: str
    evidence: str
    reproducer: str | None = None
    engine: str | None = None

    def concerns(self, axes: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        held = set(axes)
        return tuple(pair for pair in self.axes if pair in held)


def parse(entries: Any) -> tuple[BlindSpot, ...]:
    """Registry entries as declared in a contract. Fails closed on a malformed one: an entry that
    silently matched nothing would read as "no blind spot", which is the defect this exists for."""
    if entries is None:
        return ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise BlindSpotError(f"`{CONTRACT_KEY}` is a list of entries")
    out: list[BlindSpot] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise BlindSpotError(f"entry {index} is not a mapping")
        ident = str(entry.get("id") or "")
        if not ident or ident in seen:
            raise BlindSpotError(f"entry {index} needs a unique `id`")
        seen.add(ident)
        tiers = tuple(str(t) for t in entry.get("tiers") or ())
        axes = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in entry.get("axes") or ()
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
        )
        if not tiers:
            raise BlindSpotError(f"{ident}: names no blind tier")
        if not axes or len(axes) != len(entry.get("axes") or ()):
            raise BlindSpotError(f"{ident}: `axes` is a non-empty list of [axis, value] pairs")
        for key in ("what", "evidence"):
            if not str(entry.get(key) or "").strip():
                raise BlindSpotError(f"{ident}: an entry without `{key}` is a rumour, not a record")
        out.append(
            BlindSpot(
                id=ident,
                tiers=tiers,
                axes=axes,
                what=str(entry["what"]).strip(),
                evidence=str(entry["evidence"]).strip(),
                reproducer=(str(entry["reproducer"]).strip() if entry.get("reproducer") else None),
                engine=(str(entry["engine"]) if entry.get("engine") else None),
            )
        )
    return tuple(out)


def for_target(target: str, *, contract: Mapping[str, Any] | None = None) -> tuple[BlindSpot, ...]:
    """The blind spots ``target``'s own contract declares."""
    if contract is None:
        return _declared(str(target))
    return parse((contract or {}).get(CONTRACT_KEY))


@functools.lru_cache(maxsize=None)
def _declared(target: str) -> tuple[BlindSpot, ...]:
    # Asked once per capsule tier by the grader; a contract does not change under a running grade.
    from merlin.target.registry.identity import load_contract

    return parse((load_contract(target) or {}).get(CONTRACT_KEY))


def program_axes(trace: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    """The axis values an EMITTED program exercises, read off its decoded instruction trace.

    A capsule states what it asks for; which instructions answer it is the backend's choice, and a
    blind spot is about the instructions. A backend that reaches for a loop command on a capsule
    whose declaration never mentions one has put that program in the loop command's class.
    """
    found: set[tuple[str, str]] = set()
    for instruction in (trace or {}).get("instructions") or ():
        if isinstance(instruction, Mapping) and instruction.get("class"):
            found.add(("instruction_class", str(instruction["class"])))
    return found


def blind_tiers(
    spots: Iterable[BlindSpot], capsule: Mapping[str, Any], *, program: Iterable[tuple[str, str]] = ()
) -> dict[str, list[dict[str, Any]]]:
    """``{tier: [{id, via}]}`` for every tier that cannot judge ``capsule``.

    ``program`` adds what the emitted program was observed to use (:func:`program_axes`).
    """
    axes = tier_policy.capsule_axes(capsule) | set(program)
    out: dict[str, list[dict[str, Any]]] = {}
    for spot in spots:
        via = spot.concerns(axes)
        if not via:
            continue
        for tier in spot.tiers:
            out.setdefault(tier, []).append({"id": spot.id, "via": [list(pair) for pair in via]})
    return out


def required_tier(spots: Iterable[BlindSpot], capsule: Mapping[str, Any], declared_tiers: Iterable[str]) -> str | None:
    """The shallowest of ``declared_tiers`` that is blind to none of the capsule's axes.

    Every tier shallower than the deepest blind one is refused too, even if no entry names it: a
    ladder is ordered by what a tier observes, and a blind spot is a statement about a depth.
    """
    ladder = tier_policy.tier_depth_order(declared_tiers)
    blind = blind_tiers(spots, capsule)
    if not blind:
        return ladder[0] if ladder else None
    universe = tier_policy.tier_depth_order([*ladder, *blind])
    deepest_blind = max(universe.index(tier) for tier in blind)
    for tier in ladder:
        if universe.index(tier) > deepest_blind:
            return tier
    return None


@dataclass
class Audit:
    target: str
    capsules: int = 0
    concerned: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    unexercised: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "target": self.target,
            "capsules": self.capsules,
            "capsules_in_a_blind_class": self.concerned,
            "findings": self.findings,
            "blind_spots_no_capsule_exercises": self.unexercised,
        }


def audit(target: str, capsules: Iterable[Mapping[str, Any]], *, spots: Sequence[BlindSpot] | None = None) -> Audit:
    """Capsules that can only ever be judged by a tier blind to what they exercise.

    A capsule's reach is the deepest of its required tiers, lowered by a declared correctness
    ceiling. When that reach does not get past the blind tiers for its own axes, a pass is a screen
    and the corpus is claiming a certification no tier it runs can give.
    """
    spots = tuple(spots) if spots is not None else for_target(target)
    report = Audit(target=target)
    exercised: set[str] = set()
    for capsule in capsules:
        report.capsules += 1
        blind = blind_tiers(spots, capsule)
        if not blind:
            continue
        report.concerned += 1
        exercised.update(row["id"] for rows in blind.values() for row in rows)
        declared = [str(t) for t in capsule.get("required_oracle_tiers") or ()]
        cap, _source = tier_policy.declared_ceiling(capsule)
        ladder = tier_policy.tier_depth_order([*declared, *([cap] if cap else [])])
        reach = [t for t in ladder if not cap or ladder.index(t) <= ladder.index(cap)]
        needed = required_tier(spots, capsule, reach)
        if needed is not None:
            continue
        seeing = required_tier(spots, capsule, declared)
        report.findings.append(
            {
                "capsule": str(capsule.get("name") or ""),
                "finding": UNCERTIFIABLE if seeing is not None else NO_SEEING_TIER,
                "reaches": reach[-1] if reach else None,
                "declared_ceiling": cap,
                "needs": seeing,
                "blind": blind,
            }
        )
    report.unexercised = sorted(spot.id for spot in spots if spot.id not in exercised)
    return report


def judge_result(
    spots: Iterable[BlindSpot],
    capsule: Mapping[str, Any],
    tiers: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Whether the tiers a graded program PASSED can see what that program did.

    ``tiers`` is a result's per-tier record. The deepest passing tier has to lie past every tier
    blind to the capsule's axes or to the instructions the program was observed to use; otherwise
    the pass is a screen. Reported beside a grade, and never a substitute for one.
    """
    passed = [str(t) for t, row in tiers.items() if isinstance(row, Mapping) and row.get("status") == "pass"]
    blind = blind_tiers(spots, capsule, program=program_axes(trace))
    record: dict[str, Any] = {"blind": blind, "passed_tiers": tier_policy.tier_depth_order(passed)}
    if not blind or not passed:
        record["seen"] = not blind
        return record
    universe = tier_policy.tier_depth_order([*passed, *blind])
    deepest_blind = max(universe.index(tier) for tier in blind)
    record["seen"] = any(universe.index(tier) > deepest_blind for tier in passed)
    if not record["seen"]:
        record["finding"] = "passed_only_where_the_oracle_cannot_see"
    return record


# ---- capsule_budget ----------------------------------------------------------------------------
# The per-capsule wall budget, raised to cover what this run already measured a capsule needs.
#
# One number, the grade's per-capsule ``timeout`` (``--qa-timeout`` for in-turn and formal grades, the
# self-check's ``--timeout``), bounds every step of a capsule, including its elaborated-RTL run. When it
# is below what a capsule is known to cost, that capsule's cert tier can never finish: in one run a
# capsule whose gSIM run took 559 s in a self-check was killed after 287 s by a 300 s budget in every
# later grade, and stayed ``cert_not_measured`` to the end with nothing naming the budget as the cause.
#
# The budget is derived only from this run's own completed results (:func:`capsule_scheduling.
# selected_history`), never from a constant about a target. It can only RAISE the requested value, by
# at most :data:`MAX_MULTIPLE`, so it still bounds a hang. A tier that completed prices it at
# :data:`SAFETY` times its wall; a tier that was stopped on its wall clock proves only a lower bound, so
# the next grade gets :data:`SAFETY` times that wall, and a capsule that keeps timing out escalates to the
# cap instead of failing the same way forever. Carried certificates were not run and price nothing.

#: Margin over a measured wall, the same 2x the measured-T_obs certificate budget applies.
SAFETY = 2.0
#: The derived budget never exceeds this multiple of the requested one.
MAX_MULTIPLE = 4
#: Absolute ceiling, the CLI's own bound on a per-capsule budget.
CEILING_S = 86400
_TIMEOUT_CATEGORIES = {"testbench_timeout", "TESTBENCH_TIMEOUT"}


def _seconds(value) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _observations(document: Mapping):
    """``(tier, wall_s, completed)`` for each tier this result actually ran and timed."""
    failure = document.get("failure") if isinstance(document.get("failure"), Mapping) else {}
    stopped_tier = failure.get("tier") if failure.get("category") in _TIMEOUT_CATEGORIES else None
    for tier, record in (document.get("tiers") or {}).items():
        if not isinstance(record, Mapping) or record.get("measured_now") is not True:
            continue
        timing = record.get("timing")
        if not isinstance(timing, Mapping):
            continue
        wall = _seconds(timing.get("adapter_wall_s"))
        if timing.get("partial") is True:
            # Stopped before its split: only a wall-clock stop says the work needs more than this.
            if tier == stopped_tier and wall is not None:
                yield str(tier), wall, False
            continue
        if wall is None:
            active, build = _seconds(timing.get("sim_active_s")), _seconds(timing.get("build_s"))
            wall = None if active is None else active + (build or 0.0)
        if wall is not None:
            yield str(tier), wall, True


def measured_capsule_budget(
    capsules: Sequence[Mapping], runs_roots: Iterable[str | Path], *, timeout: int | float
) -> dict:
    """The budget for ``capsules`` and the measurement it rests on.

    ``runs_roots`` are earlier runs roots of this run (``<root>/runs/<suite>/<capsule>/``). The record
    always states the requested value, the value used and, when raised, which capsule and tier set it.
    """
    requested = max(1, int(timeout))
    names = {str(capsule.get("name")) for capsule in capsules if capsule.get("name")}
    record = {"requested_s": requested, "budget_s": requested, "raised": False, "basis": None}
    if not names:
        return record
    worst = None
    for root in runs_roots:
        base = Path(root)
        if not base.is_absolute() or not base.is_dir():
            continue
        for path in (base / "runs").glob("*/*/capsule_result.json"):
            if path.parent.name not in names:
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # a result still being written is not history yet
            if not isinstance(document, dict):
                continue
            for tier, wall, completed in _observations(document):
                need = math.ceil(SAFETY * wall)
                if worst is None or need > worst[0]:
                    worst = (need, path.parent.name, tier, wall, completed, base.name)
    if worst is None or worst[0] <= requested:
        return record
    need, capsule, tier, wall, completed, source = worst
    cap = min(CEILING_S, MAX_MULTIPLE * requested)
    budget = max(requested, min(need, cap))
    record.update(
        budget_s=budget,
        raised=budget > requested,
        capped=need > cap,
        capsule=capsule,
        tier=tier,
        measured_wall_s=round(wall, 3),
        measured=("completed" if completed else "stopped on its wall clock"),
        history=source,
        basis=(
            f"{capsule} {tier} {'completed in' if completed else 'was stopped on its wall clock after'} "
            f"{wall:.0f} s in {source}; the requested {requested} s per-capsule budget is raised to "
            f"{budget} s ({SAFETY:g}x that wall{', capped at ' + str(MAX_MULTIPLE) + 'x the request' if need > cap else ''})"
        ),
    )
    return record


def timed_out(results: Sequence[Mapping], budget: Mapping) -> dict:
    """``{capsule: {tier, budget_s, ...}}`` for each result stopped by this grade's wall budget."""
    stopped = {}
    for result in results:
        failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
        if failure.get("category") not in _TIMEOUT_CATEGORIES:
            continue
        stopped[str(result.get("capsule"))] = {
            "tier": failure.get("tier"),
            "status": result.get("status"),
            "budget_s": budget.get("budget_s"),
            "requested_s": budget.get("requested_s"),
            "remedy": (
                "the per-capsule wall budget ran out before this tier finished, so it is unmeasured, "
                "not failed; raise the grade's per-capsule timeout (--qa-timeout for in-turn and formal "
                "grades, --timeout for a self-check) above this capsule's measured cost"
            ),
        }
    return stopped
