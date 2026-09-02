"""Choose WHICH proposals a generation spends its width on.

The beam used to take ``[p for p in props if p.forkable][:width]`` — a truncation in whatever order
the proposer emitted, which is the order of the divergence list. Three things follow from that, and
all three waste the expensive part of the search (a build plus a board run per candidate):

* a proposal the corpus already refuted is built ahead of an untried one, purely by list position;
* ``width`` can be spent on several variants of ONE idea, because nothing groups them;
* a proposal that conflicts with what the parent already applied, or whose requirement the parent
  does not satisfy, is built anyway — and an action applied without its requirement usually builds
  and does nothing, so the intended-facet audit blames the action and the loop escalates for a
  reason that is not the real one.

This module is the ranking step between the proposer and the truncation. It never invents evidence:
an action with no prior is ranked as UNMEASURED, which is its own band, above a measured refutation
and below a measured success. ``evidence_prior=None`` means nobody measured it, not 0.5.

There is a fourth, found by running the search: a proposal deferred ``over_width`` STARVES. Band and
arrival order are both deterministic functions of the proposal, so a proposal that lost the width cut
at one generation loses identically at every later one -- it never gets built, so it never becomes
measured, so it can never out-rank anything. The beam's reachable lever set is therefore bounded by
``width``, not by ``width x depth``: depth only refines the chosen prefix. MEASURED on small_llama
fp32 at width=3: ``perop_register_block`` (which measures 25.56x on the int8 whole model),
``vectorized_transcendental_activation`` (the model's scalar `exp` is 16.48% of real work),
``fuse_transpose_b``, ``perop_nr_fill_register`` and ``accumulator_resident_wholemodel_vf_mrpad`` were
each proposed at ALL THREE generations and built at none. ``starved_fn`` is the fix: deferring a
proposal AGES it, and age breaks the tie inside its band, so depth widens coverage instead of
re-exploring one prefix. It is history, not randomness, so the search stays reproducible.

Nothing is dropped silently. Everything not chosen comes back in the rejection list with the reason,
so a generation's unspent proposals stay visible in the run record — a refutation is training signal
against a 13.45% base rate, and a search that records only what it built over-proposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..kernels.action_catalog import lineage_problems

# Prior bands, worst last. The middle band is the point: an unmeasured action is not a coin flip and
# must not be ranked as one, but it also outranks something the corpus measured and refuted.
BAND_MEASURED_HELPS = 0
BAND_UNMEASURED = 1
BAND_MEASURED_REFUTED = 2
BAND_NAMES = {BAND_MEASURED_HELPS: "measured_helps",
              BAND_UNMEASURED: "unmeasured",
              BAND_MEASURED_REFUTED: "measured_refuted"}


@dataclass(frozen=True)
class Rejection:
    """A proposal this generation did not build, and why. Recorded, never discarded."""
    targets: str
    reason: str                 # illegal_on_parent | over_width
    detail: str
    lever: str = ""
    family: str = ""
    band: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"targets": self.targets, "reason": self.reason, "detail": self.detail,
                "lever": self.lever, "family": self.family, "band": self.band}


def _prior_of(prop, prior_fn: Callable[[Any], float | None] | None) -> float | None:
    """The action's own measured prior, else whatever the injected corpus prior knows. None = nobody
    measured it, which is a distinct answer from 'measured, and it is a coin flip'."""
    action = getattr(prop, "action", None)
    if action is not None and getattr(action, "evidence_prior", None) is not None:
        return float(action.evidence_prior)
    if prior_fn is not None:
        p = prior_fn(prop)
        if p is not None:
            return float(p)
    return None


def _band(prior: float | None) -> int:
    if prior is None:
        return BAND_UNMEASURED
    return BAND_MEASURED_HELPS if prior > 0.5 else BAND_MEASURED_REFUTED


def _family_of(prop) -> str:
    """The KIND of idea this proposal is, so width buys distinct ideas rather than six spellings of
    one. Falls back through the seam to the divergence it addresses; a proposal that identifies no
    kind gets its own bucket rather than being merged with unrelated ones."""
    action = getattr(prop, "action", None)
    if action is not None:
        fam = getattr(action, "action_family", "") or ""
        if fam:
            return fam
        seam = getattr(action, "target_seam", "") or ""
        if seam:
            return seam
    return str(getattr(prop, "targets", "") or id(prop))


def proposal_key(p: Any) -> tuple[str, str]:
    """The identity under which a proposal is recognised as "the same one" across generations.

    ``(family, targets)`` -- what a Rejection records -- so a deferral at one generation can be
    charged against the same proposal at the next. Not the object id: the proposer rebuilds its
    proposals every generation, so identity has to be by content."""
    return (_family_of(p), str(getattr(p, "targets", "")))


def select_proposals(props: Sequence[Any], *, width: int,
                     applied_actions: Iterable[Any] = (),
                     prior_fn: Callable[[Any], float | None] | None = None,
                     starved_fn: Callable[[Any], int] | None = None,
                     ) -> tuple[list[Any], list[Rejection]]:
    """Return ``(chosen, rejected)`` — at most ``width`` forkable proposals, best evidence first and
    diversified by action family, plus every proposal not chosen with its reason.

    ``applied_actions`` are the actions already applied on the parent package, used for the legality
    check (see :func:`action_catalog.lineage_problems`, which deliberately does NOT apply the bundle
    same-seam rule to a lineage: overwriting a parent's seam is refinement, and the parent-to-child
    delta assigns credit unambiguously).

    ``prior_fn`` supplies a prior for actions that carry none — this is the seam through which mined
    corpus evidence (``kernels.space.corpus_prior``) reaches the search. Injected, never imported, so
    the selector stays independent of any one corpus.

    ``starved_fn(proposal) -> int`` reports how many previous generations deferred this proposal for
    width. Age breaks ties WITHIN a band and never across one: a lever the corpus refuted does not
    climb over a promising one by being passed over repeatedly, but two equally-unmeasured proposals
    are no longer separated forever by their arrival order. Without it the tail of the proposal list
    is unreachable at any depth (see the module docstring for the measured instance).

    A proposal carrying no typed action cannot be legality-checked. It is KEPT rather than dropped —
    the legacy motif router predates the composition declarations entirely, and dropping its
    proposals would silently disable that path — but it is banded as unmeasured like anything else.
    """
    forkable = [p for p in props if getattr(p, "forkable", False)]
    applied = list(applied_actions)
    rejected: list[Rejection] = []
    legal: list[tuple[int, int, str, Any]] = []      # (band, order, family, proposal)

    for order, p in enumerate(forkable):
        action = getattr(p, "action", None)
        if action is not None and applied:
            problems = lineage_problems(applied, action)
            if problems:
                rejected.append(Rejection(
                    targets=str(getattr(p, "targets", "")), reason="illegal_on_parent",
                    detail="; ".join(problems), lever=str(getattr(p, "lever", "")),
                    family=_family_of(p)))
                continue
        band = _band(_prior_of(p, prior_fn))
        # negated so MORE generations of starvation sorts EARLIER, inside the band.
        starved = -int(starved_fn(p)) if starved_fn is not None else 0
        legal.append((band, starved, order, _family_of(p), p))

    # Round-robin across families, taking each family's best band first. One pass per family gives
    # width distinct ideas when they exist; later passes backfill from families that have depth.
    by_family: dict[str, list[tuple[int, int, int, Any]]] = {}
    for band, starved, order, fam, p in legal:
        by_family.setdefault(fam, []).append((band, starved, order, p))
    for fam in by_family:
        by_family[fam].sort(key=lambda t: (t[0], t[1], t[2]))

    # Families are visited best-first by their own best candidate, so a family holding a measured
    # winner is not made to wait behind one holding only unmeasured guesses.
    fam_order = sorted(by_family, key=lambda f: (by_family[f][0][0], by_family[f][0][1],
                                                 by_family[f][0][2]))
    chosen: list[Any] = []
    chosen_ids: set[int] = set()
    depth = max((len(v) for v in by_family.values()), default=0)
    for i in range(depth):
        for fam in fam_order:
            if len(chosen) >= width:
                break
            bucket = by_family[fam]
            if i < len(bucket):
                p = bucket[i][3]
                chosen.append(p)
                chosen_ids.add(id(p))
        if len(chosen) >= width:
            break

    for band, starved, order, fam, p in sorted(legal, key=lambda t: (t[0], t[1], t[2])):
        if id(p) not in chosen_ids:
            rejected.append(Rejection(
                targets=str(getattr(p, "targets", "")), reason="over_width",
                detail=f"generation width {width} spent on better-evidenced or more diverse proposals",
                lever=str(getattr(p, "lever", "")), family=fam, band=BAND_NAMES[band]))
    return chosen, rejected
