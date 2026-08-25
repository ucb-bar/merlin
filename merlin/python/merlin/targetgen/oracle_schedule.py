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
   bytes invalidate exactly the capsules they affect -- not the corpus.
2. **The cheap tier gates the expensive one.** A deeper tier is scheduled only for a capsule that has
   PASSED the shallower one, so RTL time is only ever spent on work that could plausibly certify.
3. **The expensive tier runs on a representative subset.** Full coverage at the functional tier,
   representative coverage at the cert tier (see
   :func:`~merlin.targetgen.contract.materialize.cert_capsule_cover`) -- the hardware cannot tell two
   capsules in the same (family, dtype) cell apart, so certifying both buys nothing.

Ordering is by information density -- expected information divided by expected seconds -- which puts a
cheap unknown ahead of an expensive unknown without ever starving the expensive one, because a capsule
that has no verdict at a tier keeps its claim in the queue until it gets one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A tier's verdict can be one of these; only PASS promotes.
PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


@dataclass(frozen=True)
class Verdict:
    """What is known about one (capsule, tier), and for WHICH bytes it was known."""
    status: str
    digest: str


@dataclass
class CapsuleState:
    """Everything the scheduler knows about one capsule."""
    name: str
    digest: str                                    # the submission bytes currently on disk for it
    verdicts: dict[str, Verdict] = field(default_factory=dict)   # tier -> Verdict

    def known(self, tier: str) -> str:
        """The verdict for the CURRENT bytes, or UNKNOWN. A verdict earned by different bytes is not a
        verdict about this submission -- that is the whole point of content-addressing it."""
        v = self.verdicts.get(tier)
        if v is None or v.digest != self.digest:
            return UNKNOWN
        return v.status


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
                continue                            # already known FOR THESE BYTES -> never re-run

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
            else:
                reason = "no verdict for the current submission bytes"

            items.append(WorkItem(st.name, tier, float(cost_s.get(tier, 1.0)), reason))
            break                                   # one item per capsule at a time: its next open tier

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
    queued = schedule(states, tier_order=tier_order, cert_tiers=cert_tiers, cert_cover=cert_cover,
                      cost_s=cost_s, budget_s=budget_s)
    qkeys = {w.key for w in queued}
    unchanged, blocked, not_covered, over_budget = [], [], [], []

    full = schedule(states, tier_order=tier_order, cert_tiers=cert_tiers, cert_cover=cert_cover,
                    cost_s=cost_s, budget_s=None)
    for w in full:
        if w.key not in qkeys:
            over_budget.append(w.key)

    for st in states:
        opens = [t for t in tier_order if st.known(t) == UNKNOWN]
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
        "queue": [{"capsule": w.capsule, "tier": w.tier, "cost_s": w.cost_s, "reason": w.reason}
                  for w in queued],
        "queued_cost_s": round(sum(w.cost_s for w in queued), 3),
        "unchanged": sorted(unchanged),
        "blocked_on_shallower_tier": sorted(blocked),
        "outside_cert_cover": sorted(not_covered),
        "deferred_over_budget": sorted(over_budget),
    }
