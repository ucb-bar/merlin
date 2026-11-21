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

Relationship to ``.oracle_timing.json``: the harness already measures an oracle's per-capsule cost, but
for a DIFFERENT consumer -- ``readiness_check`` writes it and the launchers read it to size driver
TIMEOUTS (and refuse to launch when it is missing or stale). It records one tier for one config; this
module needs the RELATIVE cost of a target's tiers against each other, learned per run. The two agree
where they overlap (136.5 s recorded there, 132.5 s median measured here for the same oracle), which is
a useful cross-check -- they are not competing sources and neither should be derived from the other.
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Iterable, Mapping, Sequence

# ---- tier_policy -------------------------------------------------------------------------------

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


def priced_tiers(target: str) -> set[str]:
    """Which tiers of ``target`` have a price yet. Used to decide when calibration is done."""
    with _LOCK:
        return {t for (tgt, t) in _COST if tgt == target}


def reset_costs() -> None:
    """Forget every observation (tests; and a caller that deliberately wants to recalibrate)."""
    with _LOCK:
        _COST.clear()


def tier_order(target: str, tiers: Iterable[str]) -> list[str]:
    """``tiers`` cheapest-measured-first, with tiers never yet measured FIRST. Ties lexicographic.

    Unmeasured first is the counter-intuitive half, and it is load-bearing. The ladder stops at the
    first tier that refutes a capsule, so a tier only gets measured on capsules that reach it. Sorting
    unknowns LAST -- the obvious choice, and the one tried first -- deadlocks on any target whose early
    capsules fail: the expensive tier is measured, aborts the capsule, and the cheap tier below it is
    never reached, so it stays unknown and keeps sorting behind the expensive one forever. Measured
    exactly that way on a live suite: nine capsules in, every one had run the 24.5 s tier and only the
    single passing capsule had ever reached the 0.29 s one.

    Unknown-first inverts that: the moment one tier has a price, any tier without one is tried ahead of
    it, so the ladder always learns. The exposure is bounded to roughly one capsule per worker paying an
    unmeasured tier that turns out to be expensive -- once, per target, per process.
    """
    names = sorted(set(tiers))
    return sorted(names, key=lambda t: (observed_cost(target, t) is not None, observed_cost(target, t) or 0.0, t))


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
    for dt in capsule.get("inputs") or ():
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
        if not gain:  # nothing left can cover the remainder
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
    return False, (
        f"certify-tier budget exhausted ({used:.0f}s of {budget:.0f}s) and this capsule is "
        f"not in the derived covering set, so the axes it exercises are already certified "
        f"by a capsule that is. NOT a verdict on this capsule -- it did not run."
    )


# --- per-capsule oracle-tier ceiling -------------------------------------------------------------
# The budget above is a SUITE-level saving: it stops buying certify tiers once a RUN has spent its
# allowance, and which capsules lose out depends on the order they happen to reach. That is the right
# shape for "the run is out of time" and the wrong shape for "THIS capsule cannot be certified at all",
# which is a property of the capsule and is knowable before the run starts.
#
# Measured, which is why this exists. Fitted from the runs already on disk
# (:mod:`merlin.development.targetgen.oracles.certification_cost`, 644 samples), the two elaborated-RTL engines on this target's cert
# rung cost ~14.0s + 0.0047s/cycle and ~134.2s + 0.0694s/cycle. At the second rate the corpus's deepest
# capsule -- 28,118 measured cycles -- is ~35 minutes, and a residency sweep reaching 16,384 in the
# reduction depth is ~1.4 hours for ONE capsule. At the first it is ~8 minutes. So the ceiling is
# corpus-scaling insurance: it is what keeps a corpus that grows deep capsules from having one grade
# quietly become unaffordable, and an unaffordable grade does not announce itself -- it presents as a
# grade that is still running.
#
# TWO AXES, AND CONFUSING THEM IS THE EASY MISTAKE. A tier is bought for one of two different reasons
# and a ceiling on one must never silence the other:
#
# * CORRECTNESS -- "is the emitted kernel right". A capsule derived from a functional sibling that was
#   already certified deeper does not have to re-earn that: it is SCREENED at the cheap tier and its
#   correctness claim rests on the sibling's certification. This is what `max_oracle_tier` caps, and the
#   perf family already declares the same thing in its own words as
#   `performance.acceptance.evidence.correctness_tier`.
# * TIMING -- "how many cycles did it take". A performance claim NEEDS the cycle-accurate count; that is
#   the entire point of the family, and the perf family declares which rung owes it
#   (`...evidence.timing_tier`). A correctness ceiling must therefore NOT decline the timing tier, or a
#   cell that was measured reads back as though it never was. Excluding a member from the measurement
#   matrix is a separate declaration (`max_timing_tier`) and is recorded as a separate fact.
#
# A CEILING IS NOT A SILENCE. The capped tier is recorded as `skipped` with a reason naming the ceiling
# and the budget that set it, because this codebase's rule is that a tier with no record is not evidence
# (`not_run_is_not_pass`, and MERLIN_FULL_LADDER exists so a failing capsule still reports every declared
# tier). Omitting the key is the exact bug that made two model capsules look ungraded.
#
# AND `extends` IS VERIFIED, NOT TRUSTED. A capsule claiming to rest on a sibling's certification is
# entitled to that claim only if the named sibling actually earned the deeper tier in the run being
# cited. An unverifiable `extends` is WEAKER than no `extends`, because it reads as certified -- so it
# gets its own claim strength and can never be recorded as the certified one.

#: A capsule field capping the deepest oracle tier its CORRECTNESS will be graded at. Declared by the
#: capsule, so a reader knows what it is asking to be graded at without consulting a run.
CEILING_FIELD = "max_oracle_tier"

#: A capsule field capping the deepest tier bought for its TIMING measurement -- an exclusion from the
#: measurement matrix, deliberately NOT the same field as the correctness ceiling. See the two-axis note.
TIMING_CEILING_FIELD = "max_timing_tier"

#: A capsule field naming the SIBLING capsule whose deeper certification this capsule rests on. The
#: honesty mechanism for a ceiling: a capped capsule is not "uncertified", it is "screened at <cap> and
#: resting on <sibling>'s certification". Stated on the capsule rather than inferred from a naming
#: convention, so what the claim rests on is readable from the capsule and auditable afterwards.
EXTENDS_FIELD = "extends"

#: Where the perf family declares the same two axes in its own vocabulary. Read rather than duplicated:
#: the acceptance block is frozen before any observation, so it is the authority on which rung owes
#: which kind of evidence, and a second copy of that fact here would drift from it.
_ACCEPTANCE_PATH = ("performance", "acceptance", "evidence")
_CORRECTNESS_TIER_KEY = "correctness_tier"
_TIMING_TIER_KEY = "timing_tier"

#: The two axes a tier can be bought for. Recorded on every ceiling decision so a reader can tell
#: "correctness rests on a sibling" from "this cell has no measurement".
AXIS_CORRECTNESS = "correctness"
AXIS_TIMING = "timing"

#: Claim strengths a capped tier can carry, recorded verbatim in the tier record. Three, not two: a
#: named-but-unverifiable sibling is its own state and must never collapse into the certified one.
CLAIM_EXTENDS = "screened_at_cap_resting_on_verified_sibling"
CLAIM_EXTENDS_UNVERIFIED = "screened_at_cap_resting_on_UNVERIFIED_sibling"
CLAIM_SCREENED_ONLY = "screened_at_cap_resting_on_nothing"

#: Where a ceiling came from. Distinguishable in the record on purpose: a declaration is an author's
#: choice and a derived cap is a measurement, and the two are fixed differently.
SOURCE_DECLARED = "declared"
SOURCE_DECLARED_ACCEPTANCE = "declared_in_acceptance_block"
SOURCE_DERIVED_BUDGET = "derived_from_measured_cost"
SOURCE_UNPRICED = "unpriced"


def tier_depth_order(tiers: Iterable[str]) -> list[str]:
    """``tiers`` in LADDER order (shallowest first), derived from the tier NAMES alone.

    Deliberately not read off any tier->simulator map. Which simulator serves a rung is an availability
    and cost decision that changes -- one target's cert rung is served by three different elaborated-RTL
    engines, and a rung was retired outright once the engines on it were recognised as equal fidelity --
    while the DEPTH a rung denotes is what the corpus declares in ``required_oracle_tiers`` and is
    stable. Sorting the names is that order for an ``L<n>`` ladder, so a ceiling expressed this way
    survives a re-mapping of which engine answers where.
    """
    return sorted({str(t) for t in tiers if t is not None})


def _rank(tier: str, universe: Sequence[str]) -> int:
    """Depth rank of ``tier`` within ``universe``, or -1 when it is not part of the ladder."""
    order = tier_depth_order(list(universe) + [tier])
    return order.index(str(tier)) if str(tier) in order else -1


def _acceptance_evidence(capsule: Mapping) -> Mapping:
    """The perf family's frozen ``acceptance.evidence`` block, or an empty mapping."""
    node: object = capsule
    for key in _ACCEPTANCE_PATH:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
    return node if isinstance(node, Mapping) else {}


def declared_axes(capsule: Mapping) -> tuple[str | None, str | None]:
    """``(correctness_tier, timing_tier)`` as the capsule itself declares them, or ``(None, None)``.

    DERIVED from the capsule's own acceptance block rather than assumed: a family that owes its timing
    evidence on a different rung, or owes none at all, is covered by the same code.
    """
    ev = _acceptance_evidence(capsule)
    c, t = ev.get(_CORRECTNESS_TIER_KEY), ev.get(_TIMING_TIER_KEY)
    return (str(c) if c else None, str(t) if t else None)


def axis_of(capsule: Mapping, tier: str) -> str:
    """Which axis ``tier`` is bought for on this capsule: :data:`AXIS_TIMING` or :data:`AXIS_CORRECTNESS`.

    The timing rung is whatever the capsule declares owes its timing evidence. Everything else -- and
    every capsule that declares no timing rung at all -- is a correctness purchase.
    """
    _correctness, timing = declared_axes(capsule)
    return AXIS_TIMING if timing and str(tier) == timing else AXIS_CORRECTNESS


def declared_ceiling(capsule: Mapping, axis: str = AXIS_CORRECTNESS) -> tuple[str | None, str | None]:
    """``(cap_tier, source)`` for ``axis`` on this capsule, read and never inferred.

    On the correctness axis the explicit ``max_oracle_tier`` wins; absent that, the perf family's own
    ``correctness_tier`` IS a declared correctness ceiling and is honoured as one -- which is the whole
    point of reading the acceptance block instead of duplicating it. On the timing axis only the explicit
    ``max_timing_tier`` counts: a member is dropped from the measurement matrix deliberately or not at
    all. A cap guessed from a naming convention is a cap nobody can audit.
    """
    if axis == AXIS_TIMING:
        cap = capsule.get(TIMING_CEILING_FIELD)
        return (str(cap), SOURCE_DECLARED) if cap else (None, None)
    cap = capsule.get(CEILING_FIELD)
    if cap:
        return str(cap), SOURCE_DECLARED
    correctness, _timing = declared_axes(capsule)
    return (correctness, SOURCE_DECLARED_ACCEPTANCE) if correctness else (None, None)


def ceiling_budget_seconds() -> float | None:
    """The per-capsule certification budget in seconds, or None (the default: no derived ceiling).

    OPT-IN, and there is no default number, for the same reason :func:`budget_seconds` has none: a
    budget nobody declared, silently deciding which capsules get certified, is exactly the quiet
    coverage loss this file exists to make visible. The value is a DECLARED parameter -- it is not
    derived from the target, and no literal in this repo stands in for it. The per-cycle rate it is
    compared against IS derived, from measurement (:mod:`merlin.development.targetgen.oracles.certification_cost`).
    """
    import os

    raw = os.environ.get("MERLIN_ORACLE_CEILING_BUDGET_S", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


class ExtendsVerdict:
    """Whether a capsule's ``extends`` claim is actually backed by the sibling it names.

    ``verified`` is only ever True when a sibling result was READ and shows the named capsule passing a
    tier DEEPER than the cap. Absent, unreadable, present-but-failed and present-but-not-deeper all come
    back False with the reason, because each of them is a claim that would otherwise read as certified.
    """

    __slots__ = ("sibling", "verified", "tier", "reason", "source")

    def __init__(
        self,
        sibling: str | None,
        verified: bool,
        *,
        tier: str | None = None,
        reason: str = "",
        source: str | None = None,
    ):
        self.sibling = sibling
        self.verified = verified
        self.tier = tier
        self.reason = reason
        self.source = source

    @property
    def claim(self) -> str:
        if self.verified:
            return CLAIM_EXTENDS
        if self.sibling:
            return CLAIM_EXTENDS_UNVERIFIED
        return CLAIM_SCREENED_ONLY

    def to_dict(self) -> dict:
        return {
            "extends": self.sibling,
            "verified": self.verified,
            "certified_at_tier": self.tier,
            "reason": self.reason,
            "source": self.source,
        }


def verify_extends(
    target: str, capsule: Mapping, cap_tier: str | None, *, declared_tiers: Iterable[str] = (), roots=None
) -> ExtendsVerdict:
    """Did the sibling named by ``extends`` actually earn a tier deeper than ``cap_tier``?

    FAIL CLOSED. A perf capsule claiming to rest on a functional sibling is entitled to that claim only
    if the sibling passed a deeper cycle-accurate tier in the run being cited; anything else is recorded as
    UNVERIFIED, which is a weaker claim than naming nobody. The sibling's verdict is read from the same
    per-capsule results the cost model reads, so no new record has to be written for this to work.
    """
    sibling = capsule.get(EXTENDS_FIELD)
    if not sibling:
        return ExtendsVerdict(None, False, reason=f"capsule declares no `{EXTENDS_FIELD}` sibling")
    sibling = str(sibling)
    # ⚠️ A MISSING CAP IS REFUSED, NOT TREATED AS "NO FLOOR". `cap_rank` below is -1 when `cap_tier` is
    # falsy, after which the depth test is never taken and ANY passing tier verifies -- an L0 functional
    # pass would certify a member nobody ran cycle-accurately. That is the inverse of this function's
    # whole contract, so an unstated cap fails closed here rather than verifying everything.
    if not str(cap_tier or "").strip():
        return ExtendsVerdict(
            sibling,
            False,
            reason=(
                f"the tier this capsule is screened at was not stated, so "
                f'"deeper than the cap" has no meaning and sibling {sibling!r} '
                f"cannot corroborate anything -- recorded as UNVERIFIED rather "
                f"than accepting any passing tier, which would let an L0 pass "
                f"read as a certification"
            ),
        )
    from merlin.development.targetgen.oracles import tier_affordability as CC

    universe = list(declared_tiers) or list(capsule.get("required_oracle_tiers") or ())
    cap_rank = _rank(str(cap_tier), universe + [str(cap_tier)]) if cap_tier else -1
    found_any = False
    for base in CC._result_roots(str(target), roots):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("capsule_result.json")):
            try:
                doc = _json_loads(path)
            except (OSError, ValueError):
                continue
            if not isinstance(doc, Mapping) or str(doc.get("capsule") or "") != sibling:
                continue
            found_any = True
            if doc.get("status") != "pass":
                continue
            for name, rec in (doc.get("tiers") or {}).items():
                if not isinstance(rec, Mapping) or rec.get("status") != "pass" or not CC._is_cycle_accurate(dict(rec)):
                    continue
                if cap_tier and _rank(str(name), universe + [str(name), str(cap_tier)]) <= cap_rank:
                    continue  # not DEEPER than the cap: it corroborates nothing
                return ExtendsVerdict(
                    sibling,
                    True,
                    tier=str(name),
                    reason=(
                        f"sibling {sibling!r} passed cycle-accurate {name}, deeper than the "
                        f"{cap_tier} ceiling this capsule is screened at"
                    ),
                    source=str(path),
                )
    if found_any:
        return ExtendsVerdict(
            sibling,
            False,
            reason=(
                f"sibling {sibling!r} has a result on disk but no passing cycle-accurate tier "
                f"deeper than {cap_tier} in an overall passing grade, so it carries no certification for "
                f"this capsule to rest on"
            ),
        )
    return ExtendsVerdict(
        sibling,
        False,
        reason=(
            f"sibling {sibling!r} has no result under this target's run roots, so "
            f"the claim that its certification covers this capsule cannot be "
            f"verified -- recorded as UNVERIFIED, which is weaker than naming "
            f"nobody, because an unchecked `{EXTENDS_FIELD}` reads as certified"
        ),
    )


def _json_loads(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def certified_on_disk(target: str, *, roots=None) -> dict[str, tuple[str, str]]:
    """``capsule -> (tier, source)`` for every capsule this target has ACTUALLY certified.

    THE EVIDENCE SIDE OF A QUESTION THAT WAS ONLY EVER ASKED PROSPECTIVELY. ``phase_policy.certifiable``
    predicts, from a cost fit, whether a member COULD be certified inside a budget; it never asks whether
    one WAS. Measured on this repo's largest corpus, 12 of the 29 members the anchor gate demanded an
    ``extends`` from already held a passing cycle-accurate tier on disk -- the gate was asking capsules
    that are themselves anchors to name an anchor, and a prediction was overruling a measurement.

    Cycle-accuracy is read from the record's own ``cycle_accurate`` / ``derived_from_rtl`` declaration
    (:func:`tier_affordability._is_cycle_accurate`), never from the tier's NAME: one target's L3 is
    elaborated RTL and another's is a model, so a name-based test reads a functional pass as a
    certification. A pass that does not declare itself cycle-accurate is not counted -- there is exactly
    one such L3 record on disk today, and counting it would be the same substitution in miniature.

    Deterministic: the DEEPEST declared tier wins, ties broken on the sorted source path, so two runs of
    the same tree agree on which record is cited.
    """
    from merlin.development.targetgen.oracles import tier_affordability as CC

    out: dict[str, tuple[str, str]] = {}
    for base in CC._result_roots(str(target), roots):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("capsule_result.json")):
            try:
                doc = _json_loads(path)
            except (OSError, ValueError):
                continue  # unreadable is not evidence
            if not isinstance(doc, Mapping):
                continue
            name = str(doc.get("capsule") or "")
            tiers = doc.get("tiers")
            if not name or not isinstance(tiers, Mapping):
                continue
            universe = [str(t) for t in tiers]
            for tier_name, rec in tiers.items():
                if not isinstance(rec, Mapping) or rec.get("status") != "pass":
                    continue
                if not CC._is_cycle_accurate(dict(rec)):
                    continue
                prev = out.get(name)
                if prev is None or _rank(str(tier_name), universe + [prev[0]]) > _rank(prev[0], universe + [prev[0]]):
                    out[name] = (str(tier_name), str(path))
    return out


class Ceiling:
    """Whether one capsule may spend one oracle tier, and the record that says why not.

    ``allowed`` is the only thing the ladder acts on. ``record`` is what the tier result carries, so the
    capsule's own result states the axis, the ceiling, the budget behind it, and what the claim rests on.
    """

    __slots__ = ("allowed", "source", "axis", "reason", "record")

    def __init__(
        self,
        allowed: bool,
        *,
        source: str | None = None,
        axis: str = AXIS_CORRECTNESS,
        reason: str | None = None,
        record: dict | None = None,
    ):
        self.allowed = allowed
        self.source = source
        self.axis = axis
        self.reason = reason
        self.record = record

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"Ceiling(allowed={self.allowed!r}, axis={self.axis!r}, source={self.source!r})"


def _rests_on_clause(verdict: ExtendsVerdict, cap: str) -> str:
    if verdict.verified:
        return (
            f"This capsule is SCREENED at {cap} and rests on sibling capsule {verdict.sibling!r}, "
            f"VERIFIED to have passed {verdict.tier}. It is not a verdict on this capsule."
        )
    if verdict.sibling:
        return (
            f"This capsule is SCREENED at {cap} and names sibling {verdict.sibling!r}, but that "
            f"claim is UNVERIFIED: {verdict.reason}. Treat it as resting on nothing until the "
            f"sibling's deeper pass is on disk. It is not a verdict on this capsule."
        )
    return (
        f"This capsule is SCREENED at {cap} and declares no `{EXTENDS_FIELD}` sibling, so nothing "
        f"carries a deeper certification for it -- a strictly WEAKER claim than a capped capsule "
        f"that names one. It is not a verdict on this capsule."
    )


def _blind_at(target: str, capsule: Mapping, tier: str) -> list[dict]:
    """The declared blind spots that make ``tier`` unable to judge ``capsule``; empty when none.

    A target with no resolvable contract declares nothing. A MALFORMED registry raises: the gate that
    owns it (``check_oracle_blind_spots``) fails first, and a grade must not read it as "no blind spot".
    """
    from merlin.development.targetgen.oracles import schedule as blind

    try:
        spots = blind.for_target(str(target))
    except blind.BlindSpotError:
        raise
    except Exception:  # noqa: BLE001 -- no contract for this name: nothing is declared
        return []
    return blind.blind_tiers(spots, capsule).get(str(tier), [])


def oracle_ceiling(
    target: str,
    capsule: Mapping,
    tier: str,
    *,
    declared_tiers: Iterable[str] = (),
    engine: str | None = None,
    cycles: int | None = None,
    functional_cycles: int | None = None,
    budget_s: float | None = None,
    cost_roots=None,
    extends_roots=None,
) -> Ceiling:
    """May ``capsule`` spend ``tier`` on ``target``? Three outcomes, all recorded.

    First the AXIS is derived -- is this tier bought for correctness or for a timing measurement -- and
    only that axis's ceiling can decline it. A correctness ceiling declining the timing rung would make
    a measured cell read as unmeasured; a timing exclusion declining the correctness rung would drop a
    numeric gate. Then, in order:

    1. **A declared ceiling for that axis wins outright.** It is the author's statement about what the
       capsule is asking for, and nothing measured overrides it.
    2. **Otherwise, if a budget is declared, the measured cost decides** -- and only then, because
       without a declared budget "affordable" is not a question with an answer.
    3. **An unpriced capsule is UNKNOWN**, and UNKNOWN fails closed: the tier is not bought and the
       record says the cost could not be established. It is NOT recorded as a budget cap (nobody
       measured it exceeding anything) and NOT as affordable (nobody measured it fitting).

    The screen tier is the caller's to exclude -- a ceiling must never suppress the cheap tier, since
    "screened at the cheap tier" is the whole claim a capped capsule makes.
    """
    from merlin.development.targetgen.oracles import tier_affordability as CC

    universe = list(declared_tiers) or list(capsule.get("required_oracle_tiers") or ())
    axis = axis_of(capsule, tier)
    cap, cap_source = declared_ceiling(capsule, axis)
    ladder = tier_depth_order(list(universe) + ([cap] if cap else []))
    base = {"axis": axis, "capped_tier": str(tier), "ladder": ladder, "measurement_excluded": axis == AXIS_TIMING}

    if cap:
        cap_rank = _rank(cap, universe + [cap])
        tier_rank = _rank(tier, universe + [cap])
        if tier_rank > cap_rank >= 0:
            ev = verify_extends(
                target,
                capsule,
                cap,
                declared_tiers=universe,
                roots=extends_roots if extends_roots is not None else cost_roots,
            )
            field = TIMING_CEILING_FIELD if axis == AXIS_TIMING else CEILING_FIELD
            if axis == AXIS_TIMING:
                why = (
                    f"capsule declares `{field}: {cap}`, so {tier} is outside its MEASUREMENT "
                    f"matrix. This is an exclusion from the timing measurement, NOT a correctness "
                    f"ceiling: no cycle count is claimed for this member at {tier}."
                )
            else:
                why = (
                    f"capsule declares its correctness ceiling as `{field}: {cap}` and {tier} is "
                    f"deeper than that. " + _rests_on_clause(ev, cap)
                )
            # A ceiling at a tier the target DECLARES blind to what this capsule exercises is not a
            # weaker certification, it is none: the screen cannot refute the class at all.
            unseen = _blind_at(target, capsule, cap) if axis == AXIS_CORRECTNESS else []
            if unseen:
                why += (
                    f" The target declares {cap} blind to what this capsule exercises "
                    f"({', '.join(row['id'] for row in unseen)}), so a pass at {cap} does not "
                    f"screen it either."
                )
            return Ceiling(
                False,
                source=cap_source,
                axis=axis,
                reason=why,
                record={
                    **base,
                    "blind_spots_at_ceiling": unseen,
                    "max_oracle_tier": cap,
                    "source": cap_source,
                    "claim": ev.claim,
                    "extends": ev.to_dict(),
                    "budget_s": None,
                },
            )
        return Ceiling(True, source=cap_source, axis=axis)

    budget = budget_s if budget_s is not None else ceiling_budget_seconds()
    if budget is None:
        return Ceiling(True, axis=axis)

    aff = CC.affordability(
        str(target),
        str(tier),
        budget_s=budget,
        capsule=str(capsule.get("name") or "") or None,
        cycles=cycles,
        functional_cycles=functional_cycles,
        engine=engine,
        roots=cost_roots,
    )
    if aff.verdict == CC.AFFORDABLE:
        return Ceiling(True, source=SOURCE_DERIVED_BUDGET, axis=axis)

    ev = verify_extends(
        target,
        capsule,
        str(tier),
        declared_tiers=universe,
        roots=extends_roots if extends_roots is not None else cost_roots,
    )
    shared = {
        **base,
        "budget_s": budget,
        "affordability": aff.to_dict(),
        "max_oracle_tier": None,
        "extends": ev.to_dict(),
    }
    # A DERIVED cap on the timing rung is still a measurement exclusion, and says so: a reader must not
    # read "too expensive to time" as "correctness was never certified", nor as a measured cell.
    axis_note = (
        "No cycle count is claimed for this member at this tier -- it is an exclusion from the "
        "MEASUREMENT matrix, not a correctness verdict."
        if axis == AXIS_TIMING
        else _rests_on_clause(ev, "the cheapest tier that ran")
    )
    if aff.verdict == CC.TOO_EXPENSIVE:
        return Ceiling(
            False,
            source=SOURCE_DERIVED_BUDGET,
            axis=axis,
            reason=(f"{tier} is above this capsule's DERIVED {axis} ceiling: {aff.reason}. " + axis_note),
            record={**shared, "source": SOURCE_DERIVED_BUDGET, "claim": ev.claim},
        )
    # UNKNOWN. Fail closed and say which of the two things it is NOT.
    return Ceiling(
        False,
        source=SOURCE_UNPRICED,
        axis=axis,
        reason=(
            f"{tier} was NOT bought and NOT priced: {aff.reason}. A declared {budget:.0f}s "
            f"per-capsule budget is in force and this capsule has no measured cost basis at {tier}, "
            f"so its cost is UNKNOWN -- neither shown to fit the budget nor shown to exceed it. "
            f"Recorded as UNKNOWN rather than defaulted either way; grade this capsule once at "
            f"{tier} with no budget in force to give it a basis. " + axis_note
        ),
        record={**shared, "source": SOURCE_UNPRICED, "claim": ev.claim, "cost_unknown": True},
    )


# ---- tier_integrity ----------------------------------------------------------------------------
# Why a declared tier did not certify, and what a capsule may claim once one did not.
#
# Two silences are closed here, and both were measured on the same graded run.
#
# **A tier that did not run recorded no reason where anyone reads it.** The ladder writes a `reason`
# onto every tier record it produces, and that text is genuinely good -- on one SIMT run it said the
# elaborated-RTL model *"ran and neither failed nor completed OBSERVABLY: the console carries no
# completion witness"*, which names the defect exactly. But the score projects a tier record down to its
# bare status STRING, so the sentence stopped at the durable per-capsule file and the artifact that gets
# cited carried `L3: "unavailable"` with `failure_plane: null` and `failure_detail: null`. 24 of 29 rows
# looked like that. Uninterpretable, and indistinguishable from a tier nobody ever tried.
#
# **A cert tier that ran and did not certify left the capsule reading `pass`.** Whether a tier blocks is
# decided by ``required_oracle_tiers``; on that target no capsule declares its elaborated-RTL tier, so the
# not-run-is-not-pass gate in :mod:`.capsule_runner` -- which walks the REQUIRED set only -- never looked at
# it. Four rows carried ``L3: fail`` next to ``status: pass``, and 19 more carried ``L3: unavailable`` next
# to ``status: pass``. An executed instrument that refused the program, and an instrument that produced no
# answer at all, both read as a clean certification.
#
# This module is the rule for both, and it is deliberately a pure function over a capsule result so it can
# be tested by mutation rather than by running a simulator.
#
# **Derived, not named.** Which tier certifies is NOT a name (``"L3"`` is one target's spelling and another
# target's screen). A tier is treated as elaborated-RTL here when its OWN record says so --
# ``derived_from_rtl`` -- which is the same fact :mod:`.capsule_runner` stamps from ``cfg.rtl_tiers``, so no
# target string, tier name, or engine name appears in this file.
#
# **Fail closed.** Every branch that cannot establish a reason records an explicit ``UNKNOWN`` sentence
# naming the tier and the status it came back with. Nothing here may return ``None`` for a tier that did
# not pass, because ``None`` is exactly the state that made 24 rows unreadable.

#: Prefix for a reason this module had to synthesise because the record carried none. It is a SENTENCE,
#: not an empty string or ``None``: a reader (and a gate) must be able to tell "nobody wrote a reason"
#: apart from "the tier was never tried", and both apart from a real diagnosis.
UNKNOWN_PREFIX = "UNKNOWN"

#: The status a capsule takes when its elaborated-RTL tier was attempted and produced NO verdict. It
#: joins :data:`.capsule_common.NOT_MEASURED_STATUSES` for the same reason ``screened_only`` is there:
#: nothing about the submission was measured at that fidelity, so it belongs in neither the numerator
#: nor the denominator -- and must never be reported as a pass.
CERT_NOT_MEASURED = "cert_not_measured"

#: Tier statuses that are not a verdict on the submission at that fidelity.
NO_VERDICT_STATUSES = ("unavailable", "skipped")

#: The two KINDS of not-certified :func:`cert_tier_outcome` reports. They are not tier statuses: a tier
#: abandoned on the clock and a tier that refused the program are both recorded ``fail``, and only the
#: kind separates them.
REFUTED = "refuted"
NOT_MEASURED = "not_measured"

#: The one tier status that IS a verdict and IS negative -- subject to :func:`abandoned_on_budget`,
#: which is the difference between "the instrument refused this program" and "the clock ran out".
REFUTED_STATUS = "fail"

#: The vocabulary the two producers of an abandonment actually write: a subprocess timeout
#: (``Command '[...]' timed out after 1200 seconds``) and the broker's own budget wording. Substring
#: membership over a small closed set, NOT a pattern -- a too-narrow pattern silently reclassifies a
#: real abandonment as a defect, which is the failure this distinction exists to prevent, and this repo
#: gates ``import re`` in library code for exactly that reason. Kept in one place because the
#: consequence is asymmetric: measured on a 97-capsule batch, six capsules were abandoned at exactly
#: 900 s and counted against the arm's cert-clean headline as though their lowering were wrong.
BUDGET_MARKERS = ("timed out", "time budget", "budget exhausted", "out of budget")


def abandoned_on_budget(reason: object) -> bool:
    """Did this tier run out of TIME rather than answer wrongly?

    A cert the engine was killed on is not evidence of incorrectness. Treating it as one is how a
    harness limit gets written into the record as an agent defect -- a failure mode this repo has
    tracked eleven separate times.
    """
    if not isinstance(reason, str):
        return False
    low = reason.lower()
    return any(marker in low for marker in BUDGET_MARKERS)


def tier_field(record: object, key: str) -> object:
    """One field of a tier record, tolerating the bare-string form the score projects tiers into.

    A tier can reach a reader as ``{"status": "fail", "reason": ...}`` or as the string ``"fail"``.
    The string form reports no fields at all, which is a fact worth having rather than a crash.
    """
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def tier_status(record: object) -> str:
    """The status of a tier record in either carrier shape; ``"absent"`` when there is no record."""
    if record is None:
        return "absent"
    if isinstance(record, str):
        return record
    value = tier_field(record, "status")
    return str(value) if value is not None else "absent"


def is_rtl_tier(record: object) -> bool:
    """Does this record claim to come from elaborated RTL? Read off the record, never off a name."""
    return bool(tier_field(record, "derived_from_rtl"))


def reason_for(tier: str, record: object) -> str:
    """WHY this tier did not pass, as a sentence that is never empty and never ``None``.

    Order matters: the tier's own text leads whenever it has any, because that is the only half a
    submitter can act on. Everything below it is a synthesised statement of the state the record is
    actually in, and each one says which state that was.
    """
    status = tier_status(record)
    stated = tier_field(record, "reason")
    if isinstance(stated, str) and stated.strip():
        return stated.strip()
    if record is None:
        return (
            f"{UNKNOWN_PREFIX}: tier {tier} has no record in this result -- it was neither run nor "
            f"reported, and nothing states why"
        )
    if isinstance(record, str):
        return (
            f"{UNKNOWN_PREFIX}: tier {tier} reached this reader as the bare status {record!r}, "
            f"which carries no reason field"
        )
    if tier_field(record, "budget_deferred"):
        return (
            f"tier {tier} was not purchased: the certify budget stopped before it, so it is a "
            f"deliberate non-measurement rather than a verdict"
        )
    if tier_field(record, "not_applicable"):
        return f"tier {tier} is declared not applicable to this capsule and states no reason for it"
    if tier_field(record, "measured_now") is False:
        return f"tier {tier} carries a verdict earned by an earlier run and states no reason of its own"
    return (
        f"{UNKNOWN_PREFIX}: tier {tier} came back {status!r} and its record states no reason. An "
        f"instrument that produced no answer must say which one it was."
    )


def not_certified(tiers: Mapping | None, *, ladder: Iterable[str] = ()) -> dict[str, str]:
    """``{tier: why}`` for every tier that did not pass -- one entry per tier, never a ``None`` value.

    This is what travels with a verdict so ``unavailable`` stops being a bare word. ``ladder`` orders
    the result (tiers outside it follow in encounter order), so the cheap-to-expensive reading a target
    declares is preserved instead of re-sorted lexicographically.
    """
    found = {str(t): reason_for(str(t), r) for t, r in (tiers or {}).items() if tier_status(r) != "pass"}
    ordered = {t: found[t] for t in (str(x) for x in ladder) if t in found}
    ordered.update({t: why for t, why in found.items() if t not in ordered})
    return ordered


def cert_tier_outcome(tiers: Mapping | None) -> tuple[str, str, str, str] | None:
    """The worst elaborated-RTL tier outcome that is not a pass: ``(kind, tier, status, why)``.

    ``kind`` is :data:`REFUTED` or :data:`NOT_MEASURED` and is the load-bearing half. It is NOT
    recoverable from ``status``: a tier abandoned on the clock is recorded ``fail`` by its adapter and a
    tier that refused the program is recorded ``fail`` too, so a caller that re-derives the kind from the
    status string collapses the one distinction this function exists to make.

    ``None`` means every elaborated-RTL tier in this result passed, or there is no such tier at all --
    the two cases the caller must NOT act on, and which it distinguishes by looking for one itself.

    A record flagged ``not_applicable`` is exempt on purpose. That flag is how the ladder records both a
    tier a capsule's datapath cannot use and the advisory legality smoke that is documented never to
    block; turning either into a failure would break a deliberate escape hatch rather than close a
    silence. ``budget_deferred`` is exempt for the same reason: the ``screened_only`` verdict already
    covers a tier that was deliberately not bought, and reporting it twice would double-count it.
    """
    refuted: tuple[str, str, str, str] | None = None
    silent: tuple[str, str, str, str] | None = None
    for tier, record in sorted((tiers or {}).items(), key=lambda kv: str(kv[0])):
        if not is_rtl_tier(record):
            continue
        if tier_field(record, "not_applicable") or tier_field(record, "budget_deferred"):
            continue
        status = tier_status(record)
        why = reason_for(str(tier), record)
        if status == REFUTED_STATUS and abandoned_on_budget(tier_field(record, "reason")):
            # ABANDONED, not refuted. The tier is recorded `fail` by the adapter because its child
            # process did not return, but the clock is the thing that ended it, so it says nothing
            # about the program. It is a NON-measurement, and it must not become the agent's defect.
            if silent is None:
                silent = (NOT_MEASURED, str(tier), status, why)
        elif status == REFUTED_STATUS and refuted is None:
            refuted = (REFUTED, str(tier), status, why)
        elif status in NO_VERDICT_STATUSES and silent is None:
            silent = (NOT_MEASURED, str(tier), status, why)
    # A REFUTATION outranks a non-measurement: one of them is evidence about the submission.
    return refuted or silent


def qualify(result: Mapping) -> dict | None:
    """``{"status": ..., "failure": {...}}`` when ``result`` claims more than its cert tier certified.

    ``None`` when there is nothing to correct. Only a ``pass`` is ever rewritten -- a result that
    already carries a stronger, more specific failure keeps it, because the danger this closes is the
    flattering verdict, not the honest one.
    """
    if str(result.get("status")) != "pass":
        return None
    outcome = cert_tier_outcome(result.get("tiers"))
    if outcome is None:
        return None
    kind, tier, status, why = outcome
    if kind == REFUTED:
        return {
            "status": "fail",
            "failure": {
                "plane": "cert_tier",
                "category": "EXECUTED_CERT_TIER_REFUTED",
                "tier": tier,
                "tier_status": status,
                "tier_reason": why,
                "detail": (
                    f"the elaborated-RTL tier {tier} RAN and did not certify this "
                    f"capsule: {why}. A cheaper tier passing does not overturn it -- "
                    f"a screen may eliminate, it may never certify."
                ),
            },
        }
    return {
        "status": CERT_NOT_MEASURED,
        "failure": {
            "plane": "cert_not_measured",
            "category": "NOT_RUN_IS_NOT_PASS",
            "tier": tier,
            "tier_status": status,
            "tier_reason": why,
            "detail": (
                f"the elaborated-RTL tier {tier} produced NO verdict ({status}): "
                f"{why}. This is NOT a verdict on this capsule -- the cheaper tiers "
                f"that did pass are a screen, and a screen may never certify."
            ),
        },
    }


def apply(result: dict) -> dict:
    """Rewrite ``result`` in place when :func:`qualify` says its verdict outruns its evidence."""
    verdict = qualify(result)
    if verdict is None:
        return result
    result["status"] = verdict["status"]
    if not result.get("failure"):
        result["failure"] = verdict["failure"]
    return result
