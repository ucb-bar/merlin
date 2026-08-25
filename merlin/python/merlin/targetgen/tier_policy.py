"""Which oracle tier to spend on, in what order, and on which capsules.

Two decisions live here, both DERIVED rather than declared, because both were being made badly by
accident:

**Order.** The ladder ran its tiers in lexicographic order, so a target whose expensive oracle happens
to be named ``L3`` and whose cheap one is named ``L4`` paid the expensive one first and then aborted --
on every failing capsule. Measured across two targets, the cost spread between a target's own tiers is
two to three orders of magnitude, and it points in OPPOSITE directions:

===========  ==================  ====================
target       cheapest tier       most expensive tier
===========  ==================  ====================
one          spike      0.44 s   verilator   132.5 s
another      verilator  0.29 s   arc cosim    24.5 s
===========  ==================  ====================

So "run the cheap functional tier first" cannot be written as a tier NAME. It is whichever of the
target's applicable tiers measures cheapest, learned at run time. The first capsule of a target pays
full price to calibrate; everything after it runs cheapest-first, and the ladder's existing fail-fast
then stops at the cheapest tier that can refute the capsule.

Screening is sound in one direction ONLY, and both directions were measured:

* a capsule the cheap tier REFUTES will not certify -- confirmed 12/12, where replaying the 0.29 s tier
  on twelve capsules the 24.5 s tier had failed reproduced all twelve failures exactly;
* a capsule the cheap tier PASSES is NOT certified -- one graded submission passed the cheap functional
  tier on 20 of 20 capsules while the RTL tier passed 1. A screen may eliminate; it may never certify.

**Coverage.** When the expensive tier cannot be afforded on everything, the subset that DOES run must be
chosen to cover the axes the corpus itself declares (semantic family, generalization axis, epilogue and
mode flags, dtype, tile counts, instruction classes) -- not the first N, and not a hand-listed set that
goes stale as the corpus grows. :func:`covering_set` is a greedy set cover over exactly those declared
axes. It is ordered FIRST even when the whole suite is affordable, so that a run which is interrupted,
times out, or exhausts its budget still has every axis represented instead of a lexicographic prefix.
"""
from __future__ import annotations

import statistics
import threading
from collections.abc import Iterable, Mapping, Sequence

# (target, tier) -> observed adapter wall-clock samples. Process-local: the grader runs capsules on
# threads within one process, which is exactly the scope over which the ordering must be consistent.
_COST: dict[tuple[str, str], list[float]] = {}
_LOCK = threading.Lock()


def record_cost(target: str, tier: str, seconds: float | None) -> None:
    """Record one observation of what ``tier`` cost on ``target``. Ignores missing/absurd values so a
    skipped or unavailable tier never teaches the order anything."""
    if not target or not tier or seconds is None:
        return
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if s < 0:
        return
    with _LOCK:
        _COST.setdefault((target, tier), []).append(s)


def observed_cost(target: str, tier: str) -> float | None:
    """Median observed cost of ``tier`` on ``target``, or None when never observed."""
    with _LOCK:
        v = list(_COST.get((target, tier)) or ())
    return statistics.median(v) if v else None


def reset_costs() -> None:
    """Forget every observation (tests; and a caller that deliberately wants to recalibrate)."""
    with _LOCK:
        _COST.clear()


def tier_order(target: str, tiers: Iterable[str]) -> list[str]:
    """``tiers`` cheapest-measured-first; tiers never yet measured go LAST, ties lexicographic.

    Unmeasured last, not first: an unknown tier might be the expensive one, and putting it ahead of a
    tier already known to be cheap would reintroduce exactly the accident this replaces. The cost of
    that choice is bounded -- a target's first passing capsule runs every tier and calibrates them all.
    """
    names = sorted(set(tiers))
    return sorted(names, key=lambda t: (observed_cost(target, t) is None,
                                        observed_cost(target, t) or 0.0, t))


def is_calibrated(target: str, tiers: Iterable[str]) -> bool:
    """True once every tier in ``tiers`` has at least one observation for ``target``."""
    return all(observed_cost(target, t) is not None for t in set(tiers))


# --- coverage ------------------------------------------------------------------------------------

def capsule_axes(capsule: Mapping) -> set[tuple[str, str]]:
    """The declared axis VALUES one capsule exercises, as ``(axis, value)`` pairs.

    Read off what the corpus already states about itself. Nothing here is target-specific: a target that
    declares different families, modes or instruction classes is covered by the same code, and a capsule
    that declares nothing contributes nothing (and so is never chosen as a representative).
    """
    out: set[tuple[str, str]] = set()
    sem = capsule.get("semantic") or {}
    for key in ("semantic_family", "generalization_axis"):
        if sem.get(key):
            out.add((key, str(sem[key])))
    if capsule.get("kind"):
        out.add(("kind", str(capsule["kind"])))
    op = capsule.get("operation") or {}
    if op.get("op"):
        out.add(("op", str(op["op"])))
    attrs = op.get("attributes") or {}
    for e in attrs.get("epilogue") or ():
        out.add(("epilogue", str(e)))
    for key in ("output_dtype", "dtype", "compile_dtype"):
        if attrs.get(key):
            out.add((key, str(attrs[key])))
    # tile counts are a shape AXIS, and the value matters: a corpus whose every capsule is 1x1x1 has no
    # tiling coverage at all, which is how a backend bounded to one M tile passed every public capsule.
    for key in ("M_tiles", "K_tiles", "N_tiles"):
        if capsule.get(key) is not None:
            out.add((key, str(capsule[key])))
    exp = capsule.get("expected") or {}
    for cls in exp.get("instruction_classes") or ():
        out.add(("instruction_class", str(cls)))
    for mode, on in (exp.get("modes") or {}).items():
        if on:
            out.add(("mode", str(mode)))
    for dt in (capsule.get("inputs") or ()):
        if isinstance(dt, Mapping) and dt.get("dtype"):
            out.add(("input_dtype", str(dt["dtype"])))
    return out


def covering_set(capsules: Sequence[Mapping]) -> list[str]:
    """Greedy minimum set cover: the fewest capsule NAMES whose declared axes cover every axis value the
    whole collection declares. Deterministic (ties break on name), so two runs of the same corpus
    certify the same representatives and their results are comparable.
    """
    axes = {c["name"]: capsule_axes(c) for c in capsules if c.get("name")}
    uncovered = set().union(*axes.values()) if axes else set()
    chosen: list[str] = []
    while uncovered:
        best = max(sorted(axes), key=lambda n: len(axes[n] & uncovered))
        gain = axes[best] & uncovered
        if not gain:                      # nothing left can cover the remainder
            break
        chosen.append(best)
        uncovered -= gain
    return sorted(chosen)


def certify_order(capsules: Sequence[Mapping]) -> list[str]:
    """Every capsule name, with the covering set FIRST. What an interrupted or budget-capped run keeps."""
    cover = covering_set(capsules)
    rest = sorted(c["name"] for c in capsules if c.get("name") and c["name"] not in set(cover))
    return cover + rest


# --- budget --------------------------------------------------------------------------------------

_SPEND: dict[str, float] = {}


def note_spend(target: str, seconds: float | None) -> None:
    """Charge ``seconds`` of CERTIFY-tier time (anything above the screen) to ``target``'s budget."""
    if not target or seconds is None:
        return
    with _LOCK:
        _SPEND[target] = _SPEND.get(target, 0.0) + max(0.0, float(seconds))


def spent(target: str) -> float:
    with _LOCK:
        return _SPEND.get(target, 0.0)


def reset_spend() -> None:
    with _LOCK:
        _SPEND.clear()


def budget_seconds() -> float | None:
    """The certify-tier budget in seconds, or None for unlimited (the default).

    Unlimited by DEFAULT on purpose: a budget silently narrowing what was certified is precisely the
    kind of quiet coverage loss this file exists to make visible. A caller that wants the saving opts
    in, and every capsule the budget then skips is recorded by name with the reason.
    """
    import os
    raw = os.environ.get("MERLIN_CERTIFY_BUDGET_S", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def may_certify(target: str, capsule: Mapping) -> tuple[bool, str | None]:
    """May this capsule spend a CERTIFY tier? ``(True, None)``, or ``(False, reason)``.

    A capsule in the derived covering set always may -- the point of the cover is that the axes it
    represents are certified even under a budget. Everything else is affordable only while budget
    remains.
    """
    budget = budget_seconds()
    if budget is None or capsule.get("_covering", True):
        return True, None
    used = spent(target)
    if used < budget:
        return True, None
    return False, (f"certify-tier budget exhausted ({used:.0f}s of {budget:.0f}s) and this capsule is "
                   f"not in the derived covering set, so the axes it exercises are already certified "
                   f"by a capsule that is. NOT a verdict on this capsule -- it did not run.")
