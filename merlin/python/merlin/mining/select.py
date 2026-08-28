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


def select_proposals(props: Sequence[Any], *, width: int,
                     applied_actions: Iterable[Any] = (),
                     prior_fn: Callable[[Any], float | None] | None = None,
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
        legal.append((band, order, _family_of(p), p))

    # Round-robin across families, taking each family's best band first. One pass per family gives
    # width distinct ideas when they exist; later passes backfill from families that have depth.
    by_family: dict[str, list[tuple[int, int, Any]]] = {}
    for band, order, fam, p in legal:
        by_family.setdefault(fam, []).append((band, order, p))
    for fam in by_family:
        by_family[fam].sort(key=lambda t: (t[0], t[1]))

    # Families are visited best-first by their own best candidate, so a family holding a measured
    # winner is not made to wait behind one holding only unmeasured guesses.
    fam_order = sorted(by_family, key=lambda f: (by_family[f][0][0], by_family[f][0][1]))
    chosen: list[Any] = []
    chosen_ids: set[int] = set()
    depth = max((len(v) for v in by_family.values()), default=0)
    for i in range(depth):
        for fam in fam_order:
            if len(chosen) >= width:
                break
            bucket = by_family[fam]
            if i < len(bucket):
                p = bucket[i][2]
                chosen.append(p)
                chosen_ids.add(id(p))
        if len(chosen) >= width:
            break

    for band, order, fam, p in sorted(legal, key=lambda t: (t[0], t[1])):
        if id(p) not in chosen_ids:
            rejected.append(Rejection(
                targets=str(getattr(p, "targets", "")), reason="over_width",
                detail=f"generation width {width} spent on better-evidenced or more diverse proposals",
                lever=str(getattr(p, "lever", "")), family=fam, band=BAND_NAMES[band]))
    return chosen, rejected
