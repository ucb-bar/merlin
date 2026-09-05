"""Which phase a capsule can serve, DERIVED from the target's own facts -- never declared.

A corpus that grades a compiler has two jobs and they want different capsules. Phase 1 asks *is this
compiler correct*, which needs an independent golden and a cycle-accurate oracle, and that bounds how
big a member may be. Phase 2 asks *is this compiler fast*, which needs the member to carry work worth
optimising and a lever that reaches it, and that bounds how SMALL a member may usefully be. The two
bounds point in opposite directions, which is why one corpus cannot serve both by accident.

Today the split is not derived anywhere: the functional corpus is synthesized from a conformance
requirement while the performance corpus is a hand-authored sweep template, and nothing relates them.
This module states the two admission predicates so a capsule's phase is a computed property of the
target and the capsule, in the same way its tier already is.

THE TWO PREDICATES

``certifiable`` -- can this member's answer be checked at full fidelity?
  A golden engine exists for its datapath regime; the corpus declares a cycle-accurate tier for it;
  and its size is inside what a certification budget affords, measured on the target's OWN certified
  runs (:mod:`merlin.targetgen.cert_cost`) rather than assumed.

``priceable`` -- can a performance claim about this member be falsified?
  Its declared work can be counted, so utilization and both ceilings exist. A member whose work
  cannot be priced is worse than absent: a ``None`` price nulls every derived rate AND disables the
  corpus-wide attainment stop condition for every OTHER member, so one unpriced capsule costs more
  than itself.

WHY ``both`` IS THE DEFAULT AND EXCLUSION NEEDS A REASON. Certification cost on the targets measured
here is dominated by a per-member FLOOR, not by member size: the fitted law is a large constant plus a
small per-element rate, and the corpus sits far below the crossover. So a member that serves both
phases costs one floor and yields two verdicts, while two disjoint corpora pay two floors for verdicts
that never meet. The arithmetic, not a preference, is why this module reports ``both`` as the healthy
state and makes single-phase membership the thing that must justify itself.

FAIL CLOSED, AND ``UNKNOWN`` IS INHABITED. Every predicate returns a tri-state carrying its reason.
A target with no measured certification history yields ``UNKNOWN`` for size, never a default budget:
sizing a capsule from a number nobody measured is how a corpus acquires a claim it cannot support.
``phase_of`` therefore has four outcomes and ``neither`` is a real one -- it names a capsule that can
be neither certified nor priced, which is a finding about the corpus and must not be silently
indistinguishable from a capsule nobody asked for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "YES", "NO", "UNKNOWN", "Verdict", "PhaseVerdict",
    "PHASE1", "PHASE2", "BOTH", "NEITHER", "UNDETERMINED",
    "certifiable", "priceable", "phase_of", "declared_macs", "split_report", "anchors",
    "cycle_accurate_seen",
]

#: Tri-state. ``UNKNOWN`` is not a soft ``NO``: it says the question could not be answered here, which
#: is a different fact and must reach the caller as one.
YES = "yes"
NO = "no"
UNKNOWN = "unknown"

#: Phases. ``both`` is the anchor state -- one certification, two verdicts.
PHASE1 = "phase1"
PHASE2 = "phase2"
BOTH = "both"
NEITHER = "neither"
#: Neither phase could be decided, because a predicate returned UNKNOWN. This is NOT ``neither``:
#: ``neither`` says the capsule serves no phase, which is a finding about the corpus, while this says
#: the question could not be answered here, which is a finding about the EVIDENCE. Collapsing the two
#: is how a target with no measured certification history comes to look like a target whose capsules
#: are unusable -- the first is fixable by certifying something, the second by rewriting the corpus.
UNDETERMINED = "undetermined"

#: Tier indices that denote a cycle-accurate oracle rather than a functional model. Read from the
#: capsule's own declared tiers; the mapping from index to simulator is the TARGET's business and is
#: resolved elsewhere (``rtl_engine_policy``), so this set is about fidelity, not about a binary.
_CYCLE_ACCURATE_TIERS = frozenset({"L3", "L4", "L5"})


@dataclass(frozen=True)
class Verdict:
    """A tri-state answer that always carries why it came out that way."""

    value: str
    reason: str

    def __bool__(self) -> bool:  # pragma: no cover - guard against truthiness bugs
        raise TypeError(
            "a Verdict is tri-state; compare .value to YES/NO/UNKNOWN rather than using truthiness, "
            "because UNKNOWN would otherwise read as NO and a question nobody could answer would "
            "become an answer"
        )


@dataclass(frozen=True)
class PhaseVerdict:
    """Which phase a capsule serves, with both predicates kept so the reason survives."""

    phase: str
    cert: Verdict
    price: Verdict
    name: str = ""

    @property
    def reason(self) -> str:
        return f"certifiable={self.cert.value} ({self.cert.reason}); priceable={self.price.value} ({self.price.reason})"


def _operand_shapes(capsule: Mapping[str, Any]) -> dict[str, list[int]]:
    """Declared operand extents by name. Shapes come from the capsule's own ``inputs``, so this stays
    a statement about the WORKLOAD rather than about whatever a compiler emitted for it."""
    out: dict[str, list[int]] = {}
    for row in capsule.get("inputs") or ():
        if not isinstance(row, Mapping):
            continue
        shape = row.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            continue
        if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape):
            out[str(row.get("name"))] = [int(v) for v in shape]
    return out


def largest_operand_elements(capsule: Mapping[str, Any]) -> int:
    """Elements in the biggest declared operand. The materialization ceiling is about the tensor a
    golden has to SYNTHESIZE, which a skewed shape makes far larger than the written output."""
    best = 0
    for shape in _operand_shapes(capsule).values():
        n = 1
        for v in shape:
            n *= v
        best = max(best, n)
    return best


def declared_macs(capsule: Mapping[str, Any]) -> tuple[int | None, str]:
    """The multiply-accumulates this capsule's DECLARATION requires, or ``None`` with a reason.

    Derived from the declared operands and the semantic family, not from an operation-name allowlist:
    an allowlist has to be edited every time a family becomes priceable, and the edit is silently
    forgotten -- which is exactly how a corpus ends up with an unpriced member that disables the
    attainment stop condition for every other member.

    A contraction's work is the product of its parallel extents times its reduction extent, which is
    recoverable from any two rank-2 operands sharing one axis. Where the shared axis cannot be
    identified the answer is ``None`` WITH the reason, never a guess.
    """
    op = (capsule.get("operation") or {})
    if not isinstance(op, Mapping):
        return None, "the capsule declares no operation block"
    shapes = _operand_shapes(capsule)
    if not shapes:
        return None, "no declared operand carries a concrete positive shape"

    attrs = op.get("attributes") if isinstance(op.get("attributes"), Mapping) else {}
    lhs = shapes.get(str(attrs.get("lhs")))
    weight = shapes.get(str(attrs.get("weight")))

    # A capsule that names its operands positionally declares `arg_order` instead of `lhs`/`weight`.
    # That is still a declaration, so read it rather than treating the member as unpriceable.
    order = attrs.get("arg_order")
    if lhs is None and isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
        named = [shapes.get(str(a)) for a in order]
        present = [sh for sh in named if sh is not None]
        if len(present) >= 2:
            lhs = lhs if lhs is not None else present[0]
            weight = weight if weight is not None else present[1]

    # A convolution declares its window and its geometry, so its work is derivable without inventing
    # anything: one output position costs kh*kw*ci multiply-accumulates and the position count comes
    # from the declared padding, stride and dilation. Pricing it matters more than the count suggests
    # -- conv is the family measured furthest from the achievable rate, so an unpriced conv is the
    # headroom the performance corpus cannot see.
    kh, kw = attrs.get("kh"), attrs.get("kw")
    if isinstance(kh, int) and isinstance(kw, int) and weight is not None:
        ifm = shapes.get(str(attrs.get("ifm")))
        if ifm is None:
            ifm = next((sh for nm, sh in shapes.items() if len(sh) == 4), None)
        if ifm is not None and len(ifm) == 4 and len(weight) >= 2:
            # NHWC is the declared layout on every capsule that carries this attribute set; the
            # spatial extents are the two interior axes either way.
            h, w = ifm[1], ifm[2]
            pad = attrs.get("padding") or [0, 0, 0, 0]
            stride = attrs.get("stride") or [1, 1]
            dil = attrs.get("dilation") or [1, 1]
            try:
                oh = (h + int(pad[0]) + int(pad[1]) - int(dil[0]) * (kh - 1) - 1) // int(stride[0]) + 1
                ow = (w + int(pad[2]) + int(pad[3]) - int(dil[1]) * (kw - 1) - 1) // int(stride[1]) + 1
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                return None, "the declared convolution geometry is not a usable padding/stride/dilation triple"
            if oh <= 0 or ow <= 0:
                return None, f"the declared convolution geometry leaves no output position ({oh}x{ow})"
            per_position = _trailing(weight) * _leading(weight)  # (kh*kw*ci) x out_c for an im2col weight
            return oh * ow * per_position, f"convolution: {oh}x{ow} output positions x a {_leading(weight)}-tap window"

    if lhs is not None and weight is not None and len(lhs) >= 2 and len(weight) >= 2:
        # The reduction axis is the one the two operands share. Checking it rather than assuming
        # position keeps this correct for a transposed weight, where assuming lhs[-1]==weight[0]
        # prices a shape the capsule does not declare.
        if lhs[-1] == weight[0]:
            m, k, n = _leading(lhs), lhs[-1], _trailing(weight)
            return m * k * n, "contraction: declared lhs x weight"
        if lhs[-1] == weight[-1]:
            m, k, n = _leading(lhs), lhs[-1], _leading(weight)
            return m * k * n, "contraction: declared lhs x weight (transposed)"
        return None, (f"the declared operands share no reduction axis: lhs {lhs} and weight {weight} "
                      "cannot both be extents of one contraction")

    family = ((capsule.get("semantic") or {}) or {}).get("semantic_family")

    # No `lhs` attribute: fall back to the operands' declared ROLES. A weight-stationary member names
    # one weight and several activations rather than one lhs -- the reuse IS the point, so the work is
    # the sum over the activations that share the pushed weight, not one contraction. Reading the role
    # keeps this derived from what the capsule declares instead of from an operation-name allowlist.
    if weight is None or lhs is None:
        roles: dict[str, list[list[int]]] = {}
        for row in capsule.get("inputs") or ():
            if not isinstance(row, Mapping):
                continue
            shape = shapes.get(str(row.get("name")))
            if shape is not None:
                roles.setdefault(str(row.get("role")), []).append(shape)
        weights = roles.get("weight") or ([weight] if weight is not None else [])
        acts = roles.get("input") or []
        if len(weights) == 1 and acts and len(weights[0]) >= 2:
            w = weights[0]
            total, counted = 0, 0
            for a in acts:
                if len(a) >= 2 and a[-1] == w[0]:
                    total += _leading(a) * a[-1] * _trailing(w)
                    counted += 1
                elif len(a) >= 2 and a[-1] == w[-1]:
                    total += _leading(a) * a[-1] * _leading(w)
                    counted += 1
            if counted == len(acts):
                return total, (f"contraction: {counted} declared activation(s) sharing one weight"
                               if counted > 1 else "contraction: declared roles")
        # No weight at all. A contraction between two ACTIVATIONS is still a contraction -- a scores
        # block contracts Q against K and neither is a parameter -- so the reduction axis is found the
        # same way it is everywhere else here: as the extent the two operands share.
        if not weights and len(acts) >= 2:
            a, b = acts[0], acts[1]
            if len(a) >= 2 and len(b) >= 2 and a[-1] == b[-1]:
                return _leading(a) * a[-1] * _leading(b), "contraction: two declared activations sharing a reduction axis"

    # An attention block contracts twice and carries no weight at all -- Q, K and V are all
    # activations -- so neither the attribute pair nor the weight/activation roles reach it. Its work
    # is derived from the family rather than from the operation name, because the family is the
    # vocabulary that is common across targets while the op spelling is not.
    if family == "attention":
        acts = [sh for sh in shapes.values() if len(sh) >= 2]
        if len(acts) >= 3:
            q, k, v = acts[0], acts[1], acts[2]
            if q[-1] == k[-1] and k[0] == v[0]:
                m, dk, n, dv = _leading(q), q[-1], _leading(k), _trailing(v)
                return m * dk * n + m * n * dv, "attention: scores (Q.K^T) plus context (P.V)"
        if len(acts) == 2:
            q, k = acts
            if q[-1] == k[-1]:
                return _leading(q) * q[-1] * _leading(k), "attention: scores only (Q.K^T), no context operand declared"
        return None, (f"attention declares {len(acts)} rank>=2 operands; a scores-and-context pair needs "
                      "three whose head and key extents agree")

    # The family decides whether multiply-accumulate work is a meaningful quantity at all here.
    if family in ("elementwise_map", "movement", "synchronization", "reduction", "normalization", "softmax"):
        return 0, (f"{family} contracts nothing, so its multiply-accumulate work is zero -- a true "
                   "quantity, not a missing price")
    return None, f"no contraction operands are recoverable from the declaration and family {family!r} does not fix the work"


def _leading(shape: Sequence[int]) -> int:
    n = 1
    for v in shape[:-1]:
        n *= v
    return n


def _trailing(shape: Sequence[int]) -> int:
    return int(shape[-1])


def certifiable(capsule: Mapping[str, Any], *, target: str, fit: Any = None,
                budget_s: float | None = None, max_operand_elements: int | None = None,
                cycle_accurate_available: "bool | None" = None) -> Verdict:
    """Can this member's answer be checked at full fidelity, on THIS target, inside a budget?

    ``cycle_accurate_available`` is a property of the TARGET and the caller must establish it. ``None``
    means nobody asked, reported as UNKNOWN rather than assumed: the capsule's own required-tier list
    cannot answer it.
    """
    from merlin.targetgen import cert_cost as CC

    # ``required_oracle_tiers`` is the MANDATORY set, NOT a ceiling. A target may run a cycle-accurate
    # tier that gates nothing -- one in this repo does exactly that, and its runner records "L3 is an
    # RTL-cert tier that NEVER gates a capsule" -- so reading the required list as the set of tiers that
    # CAN run declares such a target unable to certify anything while it is certifying 32 members. The
    # capsule-level ceiling is the explicit cap; target capability is supplied by the caller.
    cap = capsule.get("max_oracle_tier")
    if cap is not None and str(cap) not in _CYCLE_ACCURATE_TIERS:
        return Verdict(NO, f"caps its oracle tier at {str(cap)!r}, which is not cycle-accurate, so it is "
                           "screened rather than certified and must name the sibling it rests on")
    if cycle_accurate_available is False:
        return Verdict(NO, f"{target!r} has no cycle-accurate tier available, so nothing can certify it")
    if cycle_accurate_available is None:
        return Verdict(UNKNOWN, f"whether {target!r} can run a cycle-accurate tier was not established; "
                                "a required-tier list does not answer it, because such a tier may run "
                                "without gating any capsule")

    ceiling = CC.MEASURED_MAX_OPERAND_ELEMENTS if max_operand_elements is None else int(max_operand_elements)
    operand = largest_operand_elements(capsule)
    if operand > ceiling:
        return Verdict(NO, f"largest operand {operand} elements exceeds the measured range {ceiling}; "
                           "the cost of moving it is unknown, not merely large")

    if budget_s is None:
        return Verdict(YES, "a cycle-accurate tier is available and its operands are inside the measured "
                            "range; no budget was supplied, so size was not tested against one")

    affordable = CC.max_elements_within(fit, float(budget_s))
    if affordable is None:
        return Verdict(UNKNOWN, f"{target!r} has no measured certification history, so no size can be shown "
                                "affordable; certify this target's existing corpus first")
    size = CC.capsule_elements(dict(capsule))
    if size > affordable:
        return Verdict(NO, f"{size} elements exceeds the {affordable} a {budget_s:g}s budget affords here")
    return Verdict(YES, f"{size} elements is inside the {affordable} affordable at {budget_s:g}s")


def priceable(capsule: Mapping[str, Any], *, achievable_macs_per_cycle: float | None = None) -> Verdict:
    """Can a performance claim about this member be falsified?

    Work must be countable, and the work must be non-zero: a member that performs no
    multiply-accumulate has no utilization to improve, so admitting it to a performance corpus adds a
    member that cannot move the objective while still costing a full certification floor.
    """
    macs, why = declared_macs(capsule)
    if macs is None:
        return Verdict(NO, f"work cannot be priced -- {why}; an unpriced member also disables the "
                           "corpus-wide attainment stop condition for every other member")
    if macs == 0:
        return Verdict(NO, f"declares zero multiply-accumulates ({why}), so it carries no utilization to improve")
    if achievable_macs_per_cycle is None:
        return Verdict(YES, f"{macs} MACs are derivable ({why}); no achievable ceiling was supplied, so "
                            "headroom was not tested")
    if achievable_macs_per_cycle <= 0:
        return Verdict(UNKNOWN, "the achievable ceiling is not positive, so share-of-achievable cannot be formed")
    return Verdict(YES, f"{macs} MACs against an achievable {achievable_macs_per_cycle:g} MAC/cycle")


def phase_of(capsule: Mapping[str, Any], *, target: str, fit: Any = None, budget_s: float | None = None,
             achievable_macs_per_cycle: float | None = None,
             cycle_accurate_available: "bool | None" = None) -> PhaseVerdict:
    """Which phase this capsule can serve. ``both`` is the healthy state; ``neither`` is a finding."""
    cert = certifiable(capsule, target=target, fit=fit, budget_s=budget_s,
                       cycle_accurate_available=cycle_accurate_available)
    price = priceable(capsule, achievable_macs_per_cycle=achievable_macs_per_cycle)
    if cert.value == UNKNOWN or price.value == UNKNOWN:
        # Fail closed on the EVIDENCE, not on the capsule. An UNKNOWN folded into NO would report a
        # missing measurement as a property of the corpus.
        phase = UNDETERMINED
    elif cert.value == YES and price.value == YES:
        phase = BOTH
    elif cert.value == YES:
        phase = PHASE1
    elif price.value == YES:
        phase = PHASE2
    else:
        phase = NEITHER
    return PhaseVerdict(phase=phase, cert=cert, price=price, name=str(capsule.get("name") or ""))


def split_report(capsules: Sequence[Mapping[str, Any]], *, target: str, fit: Any = None,
                 budget_s: float | None = None,
                 achievable_macs_per_cycle: float | None = None,
                 cycle_accurate_available: "bool | None" = None) -> dict[str, Any]:
    """The phase split for one target's corpus, with every single-phase member's reason kept.

    A count on its own cannot be acted on: the useful output is WHY a member is single-phase, because
    that names the thing to fix -- a missing golden engine, an unpriced family, or a size nobody can
    afford to certify.
    """
    verdicts = [phase_of(c, target=target, fit=fit, budget_s=budget_s,
                         achievable_macs_per_cycle=achievable_macs_per_cycle,
                         cycle_accurate_available=cycle_accurate_available) for c in capsules]
    counts: dict[str, int] = {BOTH: 0, PHASE1: 0, PHASE2: 0, NEITHER: 0, UNDETERMINED: 0}
    for v in verdicts:
        counts[v.phase] += 1
    reasons: dict[str, list[str]] = {}
    for v in verdicts:
        if v.phase == BOTH:
            continue
        if v.phase == UNDETERMINED:
            key = v.cert.reason if v.cert.value == UNKNOWN else v.price.reason
        else:
            key = v.price.reason if v.phase == PHASE1 else v.cert.reason
        reasons.setdefault(f"{v.phase}: {key}", []).append(v.name)
    return {
        "target": target,
        "n_capsules": len(verdicts),
        "counts": counts,
        "single_phase_reasons": reasons,
        "verdicts": verdicts,
    }


# --------------------------------------------------------------------------------- the anchor relation

def _obligation_key(capsule: Mapping[str, Any]) -> tuple:
    """What makes two capsules witnesses of the SAME obligation, independent of scale.

    Family, datapath and epilogue decide what is being computed; extents decide how much of it. A
    phase-2 member is the same claim as its anchor at a larger scale, so the key deliberately excludes
    every extent -- that is the whole content of "bigger sibling".
    """
    sem = capsule.get("semantic") or {}
    op = capsule.get("operation") or {}
    attrs = op.get("attributes") or {}
    epilogue = tuple(sorted(str(e) for e in (attrs.get("epilogue") or ())))
    dtypes = tuple(sorted({str(r.get("dtype")) for r in (capsule.get("inputs") or ())
                           if isinstance(r, Mapping) and r.get("dtype")}))
    return (str(sem.get("semantic_family") or op.get("op") or "?"), dtypes, epilogue,
            str(attrs.get("output_dtype") or ""))


def anchors(capsules: Sequence[Mapping[str, Any]], *, target: str, fit: Any = None,
            budget_s: float | None = None,
            cycle_accurate_available: "bool | None" = None,
            verify: bool = False, roots: Any = None) -> dict[str, Any]:
    """Pair every phase-2 member with the certified sibling it can rest on.

    The relation this computes is the one ``extends`` already declares and nothing verifies: a member
    too large to certify is admissible only as an EXTENSION of a sibling that was certified, so the
    anchor must exist, must be certifiable, and must witness the same obligation. Where no such sibling
    exists the member is reported as ORPHANED rather than silently accepted -- an L2 pass on a shape
    nothing ever certified cycle-accurately is exactly the "read tier_reached, never a bare score"
    failure this corpus already has scar tissue for.

    The anchor chosen is the LARGEST certifiable witness of the obligation, not the smallest. A bigger
    anchor is a stronger guarantee for the same certification floor, and the floor is what dominates
    the cost.
    """
    by_ob: dict[tuple, list[tuple[Mapping[str, Any], PhaseVerdict]]] = {}
    for c in capsules:
        v = phase_of(c, target=target, fit=fit, budget_s=budget_s,
                     cycle_accurate_available=cycle_accurate_available)
        by_ob.setdefault(_obligation_key(c), []).append((c, v))

    paired: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []
    for key, members in sorted(by_ob.items(), key=lambda kv: str(kv[0])):
        certified = [(c, v) for c, v in members if v.cert.value == YES]
        extensions = [(c, v) for c, v in members if v.cert.value != YES and v.price.value == YES]
        if not extensions:
            continue
        if certified:
            # DETERMINISTIC, ties broken on name. `max` alone picks whichever equal-sized witness the
            # caller happened to enumerate first, so the same corpus yielded a different anchor -- and a
            # different verification verdict -- depending on how its capsules were walked. Measured: the
            # same corpus reported 6 verified from one enumeration and 0 from another.
            anchor = max(certified, key=lambda cv: (largest_operand_elements(cv[0]),
                                                    str(cv[0].get("name") or "")))
            for c, v in extensions:
                row = {"member": str(c.get("name") or ""), "anchor": str(anchor[0].get("name") or ""),
                       "obligation": key, "why": v.cert.reason}
                if verify:
                    # PAIRED IS NOT VERIFIED. That a certifiable sibling EXISTS is structural; that it
                    # was CERTIFIED is evidential, and only the second entitles a member to rest on it.
                    # tier_policy fails closed: a sibling with no deeper passing tier on disk records as
                    # UNVERIFIED, which is a weaker claim than naming nobody, because an unchecked
                    # `extends` reads as certified.
                    from merlin.targetgen import tier_policy as TP

                    probe = dict(c)
                    probe["extends"] = row["anchor"]
                    verdict = TP.verify_extends(target, probe, _deepest_declared(c), roots=roots)
                    row["verified"] = bool(getattr(verdict, "verified", False))
                    row["verification"] = str(getattr(verdict, "reason", ""))
                paired.append(row)
        else:
            for c, v in extensions:
                orphaned.append({"member": str(c.get("name") or ""), "obligation": key,
                                 "why": "no certifiable witness of this obligation exists on this target"})
    out = {"target": target, "n_obligations": len(by_ob),
           "paired": paired, "orphaned": orphaned,
           "n_paired": len(paired), "n_orphaned": len(orphaned)}
    if verify:
        out["n_verified"] = sum(1 for r in paired if r.get("verified"))
        out["n_unverified"] = len(paired) - out["n_verified"]
    return out


def _deepest_declared(capsule: Mapping[str, Any]) -> "str | None":
    """The deepest tier this capsule declares -- the ceiling it is screened at."""
    tiers = [str(t) for t in (capsule.get("required_oracle_tiers") or ())]
    return max(tiers) if tiers else None


def cycle_accurate_seen(target: str, *, roots: Any = None) -> "bool | None":
    """Has a cycle-accurate tier ACTUALLY RUN on this target? ``True`` or ``None``, never ``False``.

    Measured, not declared: read from the per-capsule results the cost model already reads, so a target
    that has certified something says so with evidence. The absence of a record is ``None`` -- unknown --
    and never ``False``, because "nobody has run one here" and "this target cannot run one" are
    different facts with different remedies, and only the second would justify excluding a target's
    whole corpus from phase 1.

    The obvious substitute is wrong, which is why this exists. A capsule's ``required_oracle_tiers`` is
    the MANDATORY set, not the runnable one: a target in this repo runs a cycle-accurate tier that gates
    nothing, and reading its required list as its capability declared it unable to certify anything
    while it had in fact certified 32 members.
    """
    from merlin.targetgen import cert_cost as CC

    try:
        records = CC._timing_records(str(target), roots)
    except Exception:  # noqa: BLE001 - an unreadable run tree is "unknown", not "cannot"
        return None
    return True if records else None


# ----------------------------------------------------- necessity, and whether a member is worth timing

def covers_cell(capsule: Mapping[str, Any]) -> tuple:
    """The conformance cell this capsule witnesses: ``(family, dtype, alignment-unknown)``.

    Alignment is deliberately left out. It is a property the requirement computes from extents against
    the target's tile, and re-deriving it here would be a second implementation that can disagree with
    the first -- so this answers the two axes a capsule states about itself and lets the caller match on
    those. A capsule stating neither is not a witness of anything and says so by returning ``()``.
    """
    sem = capsule.get("semantic") or {}
    family = sem.get("semantic_family") or (capsule.get("operation") or {}).get("op")
    dtypes = sorted({str(r.get("dtype")) for r in (capsule.get("inputs") or ())
                     if isinstance(r, Mapping) and r.get("dtype")})
    if not family or not dtypes:
        return ()
    return (str(family), tuple(dtypes))


def necessary(capsule: Mapping[str, Any], required: "set[tuple] | None") -> Verdict:
    """Does this capsule witness something the REQUIREMENT asks for?

    Necessity is not the same question as either admission predicate, and conflating them is how a
    corpus grows: a capsule can be perfectly certifiable and perfectly priceable and still witness no
    obligation, in which case it costs a certification floor to tell us something nobody asked. It is
    also not a licence to delete -- an unrequired capsule may be a deliberate edge case the requirement
    cannot express -- so this REPORTS rather than condemns.
    """
    if required is None:
        return Verdict(UNKNOWN, "the requirement for this target could not be derived, so necessity "
                                "cannot be decided -- and an underived requirement is not an empty one")
    cell = covers_cell(capsule)
    if not cell:
        return Verdict(UNKNOWN, "the capsule states neither a family nor an operand dtype, so what it "
                                "witnesses cannot be read off it")
    if cell in required:
        return Verdict(YES, f"witnesses required cell {cell[0]}/{'+'.join(cell[1])}")
    return Verdict(NO, f"witnesses {cell[0]}/{'+'.join(cell[1])}, which the requirement does not ask for")


def worth_timing(capsule: Mapping[str, Any], *, share_of_achievable: float | None = None,
                 band: float = 0.10) -> Verdict:
    """Is there enough headroom in this member for a performance claim to be about anything?

    THE MEASURED REASON THIS EXISTS. A campaign converged at roughly 0.2% while improving members with
    zero regressions, because the members carrying almost all of its cycles were already at 0.59-0.94 of
    achievable. Priceable and worth optimising are different properties, and a corpus that admits on the
    first spends itself on members that cannot move.

    ``band`` is the cost model's own uncertainty: a member within it of the achievable rate has nothing
    a measurement could distinguish from noise. ``None`` is UNKNOWN, never "assume there is room".
    """
    if share_of_achievable is None:
        return Verdict(UNKNOWN, "no measured share of the achievable rate for this member, so headroom "
                                "is undecided -- assuming room is how a corpus admits members that "
                                "cannot move its objective")
    if share_of_achievable >= 1.0 - band:
        return Verdict(NO, f"at {share_of_achievable:.0%} of the achievable rate, inside the {band:.0%} "
                           "band, so any improvement is indistinguishable from noise")
    return Verdict(YES, f"at {share_of_achievable:.0%} of achievable, {1 - share_of_achievable:.0%} of "
                        "the rate is still on the table")


def lever_is_reachable(capsule: Mapping[str, Any]) -> Verdict:
    """Does the lever this member's family declares have an analyzer that can decide it?

    The measured case: a family owning 92.4% of a corpus's cycles declared ``operand_residency`` while
    every action the search could take moved tiling, hoisting or barriers. It was never improved once,
    across three trials -- not because the search was weak but because nothing it could do reached the
    thing the family was about.
    """
    perf = capsule.get("performance")
    if not isinstance(perf, Mapping):
        return Verdict(UNKNOWN, "declares no performance block, so it names no lever to reach")
    try:
        from merlin.perf import claim_reach as CR

        reach = CR.family_reach(perf)
    except Exception as exc:  # noqa: BLE001
        return Verdict(UNKNOWN, f"the family's reach could not be derived ({type(exc).__name__})")
    ok = getattr(reach, "reachable", None)
    why = str(getattr(reach, "reason", "") or "")
    if ok is True:
        return Verdict(YES, why or f"the analyzer for lever {perf.get('lever')!r} resolves")
    if ok is False:
        return Verdict(NO, why or f"nothing reaches lever {perf.get('lever')!r}")
    return Verdict(UNKNOWN, why or "the family declares a lever whose reach is not established")
