"""Derive a small set of tasks that covers most of a model's accelerable cost.

WHY A BASIS AND NOT A LIST. A study cannot generate a kernel for every contraction in a model, and
picking a handful by hand makes the result a statement about the picker. So the task set is DERIVED:
group the model's regions by a structural signature, weight each group by what it actually costs,
and take groups greedily until a stated fraction of the eligible cost is covered. The same census
gives the same basis, every time, and the certificate says what was left out and why.

THE SIGNATURE IS SHAPE-FREE, ON PURPOSE. Two matmuls of different sizes are the same *task* at
different *configurations*. Folding shape into task identity would make every shape its own task,
which destroys two things at once: the reuse ladder (a library keyed by exact shape is a lookup
table, not a library) and the specialization audit (a policy keyed bijectively to shapes cannot be
distinguished from a policy that generalizes). Shape lives in the config ladder instead.

THREE WAYS THIS COULD LIE, EACH GUARDED HERE:

*Undetermined must leave BOTH sides.* When the evidence cannot decide whether the hardware supports a
family, counting it ineligible shrinks the denominator and flatters coverage; counting it eligible
inflates the work demanded. It is excluded from numerator and denominator alike and reported as
unmeasured -- the same discipline ``EligibilityVerdict.undetermined`` already uses.

*Measured shares must not be summed.* Several contractions can join one provenance bucket -- an
attention layer's two contractions carry one fqn -- so adding their percentages counts that bucket
twice. On one model that turns 96.01% into 106.23% of a model that is by definition 100%. Cost comes
from ``Census.measured_share``, which deduplicates by tick bucket.

*A lower bound must not read as a measurement.* ``CensusRow.work`` is a lower bound when the row's
arithmetic could not be fully attributed. Such a group is still included -- excluding it would be
worse -- but carries ``weight_is_lower_bound`` so a cover fraction computed from it is not quoted as
exact.

Target-agnostic: every hardware fact arrives as a parameter (the capability map, the regime
boundaries). Nothing here knows what target it is describing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from merlin.targetgen import eligibility as EL
from merlin.targetgen import semantic_families as sf

#: Cost fraction the cover must reach before it stops. The plan's figure; a caller may raise it.
DEFAULT_COVER_TARGET = 0.95


def shape_regime(extents: Sequence[int], *, boundaries: Sequence[int] = ()) -> str:
    """Coarse size class for an iteration space, as a stable string.

    Two shapes share a regime when a kernel written for one plausibly transfers to the other, and
    what makes that true is whether the working set fits in registers, in shared memory, or neither.
    Those are HARDWARE boundaries, so they arrive as a parameter derived from the target's facts
    rather than being invented here.

    With no boundaries supplied the fallback is a power-of-two bucketing of the total volume, which
    is target-independent and monotone -- weaker than real boundaries, but never wrong in a way that
    silently depends on a target this module cannot see.
    """
    vol = 1
    for e in extents:
        vol *= max(int(e), 1)
    if boundaries:
        for i, b in enumerate(sorted(int(x) for x in boundaries)):
            if vol <= b:
                return f"b{i}"
        return f"b{len(tuple(boundaries))}"
    # log2 decade, floored to a multiple of 4 so neighbouring sizes group together.
    bits = max(vol.bit_length() - 1, 0)
    return f"v2^{(bits // 4) * 4}"


@dataclass(frozen=True)
class Signature:
    """Task identity: what a kernel is FOR, deliberately excluding how big it is."""

    op_class: str
    family: str
    role: str
    dtypes: tuple[str, ...]
    rank: int
    regime: str

    def key(self) -> str:
        """A stable string, used for deterministic ordering and as a dict key."""
        return "|".join((self.op_class, self.family, self.role,
                         ",".join(self.dtypes), str(self.rank), self.regime))


@dataclass
class Group:
    """Every region sharing one signature, with what it costs and whether the target can run it."""

    signature: Signature
    row_indices: tuple[int, ...]
    cost: float                      # share of the model, in [0, 1]
    cost_source: str                 # 'measured_ticks' | 'work_share' | 'none'
    weight_is_lower_bound: bool
    eligible: bool
    undetermined: bool
    reason: str
    shapes: tuple[tuple[int, ...], ...] = ()   # the config ladder is built from these

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signature_key"] = self.signature.key()
        return d


@dataclass
class TaskBasis:
    entries: tuple[Group, ...]
    certificate: dict[str, Any] = field(default_factory=dict)

    def signature_keys(self) -> tuple[str, ...]:
        return tuple(g.signature.key() for g in self.entries)

    def write_certificate(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps(self.certificate, indent=2, sort_keys=True))


def _semantic_family(row: Any) -> str:
    """The row's SEMANTIC family, resolving an op-class name when that is what it carries.

    ``CensusRow.family`` is documented as the semantic family, but it falls back to ``prov.op`` when
    a capture carries no ``prov.family`` -- and then it holds an op class. Measured on the seed
    model, every row said "matmul"/"batch_matmul" where the capability map is keyed "contraction",
    so every group came back ineligible and the basis was empty. An empty basis is not obviously
    wrong from the outside; it reads as "the target supports nothing", which is the same shape a
    real answer would have. Resolved through the op registry instead of trusted.
    """
    fam = row.family or ""
    if fam and sf.is_family(fam):
        return fam
    return sf.from_op(fam or row.op_class or "") or fam


def _row_signature(row: Any, *, boundaries: Sequence[int]) -> Signature:
    extents = tuple(row.parallel) + tuple(row.reduction)
    return Signature(
        op_class=row.op_class or "",
        family=_semantic_family(row),
        role=row.role or "",
        dtypes=tuple(row.dtypes or ()),
        rank=len(extents),
        regime=shape_regime(extents, boundaries=boundaries),
    )


def _descriptor(row: Any) -> EL.RegionDescriptor:
    """A census row as an eligibility question. Shape axes only where the row observed them."""
    par = tuple(row.parallel or ())
    red = tuple(row.reduction or ())
    dt = tuple(row.dtypes or ())
    return EL.RegionDescriptor(
        source="census",
        op=row.op_class or None,
        family=_semantic_family(row) or None,
        in_dtype=dt[0] if dt else None,
        weight_dtype=dt[1] if len(dt) > 1 else None,
        m=par[0] if par else None,
        n=par[1] if len(par) > 1 else None,
        k=red[0] if red else None,
        rank=len(par) + len(red),
    )


def _group_cost(census: Any, rows: Sequence[Any]) -> tuple[float, str, bool]:
    """(cost share, source, is_lower_bound) for one group.

    Measured ticks are preferred because they are what the model actually spends. `measured_share`
    deduplicates by tick bucket -- per-row percentages must never be summed.
    """
    share = None
    try:
        share = census.measured_share(rows)
    except Exception:
        share = None
    if share is not None:
        # An upper bound in the other direction: a bucket covering a contraction also covers whatever
        # else shares its key. Recorded as measured, with the caveat carried by the certificate.
        return float(share), "measured_ticks", False

    total = getattr(census, "total_work", 0) or 0
    if total <= 0:
        return 0.0, "none", True
    work = sum(int(getattr(r, "work", 0) or 0) for r in rows)
    incomplete = any(not getattr(r, "work_complete", True) for r in rows)
    return work / float(total), "work_share", incomplete


def derive_basis(census: Any, cap_map: Mapping[str, Any], *,
                 cover_target: float = DEFAULT_COVER_TARGET,
                 regime_boundaries: Sequence[int] = (),
                 census_enumerates: Sequence[str] = (),
                 family_floor: bool = True) -> TaskBasis:
    """Group, weigh, filter and cover -- deterministically.

    ``cap_map`` is the target's capability map (``eligibility.capability_map_for_target``). It decides
    only what is ELIGIBLE; it never decides what is measured, so a target cannot shrink its own
    denominator by declaring less.

    ``census_enumerates`` names the families the census can SEE. It matters because a census with a
    narrow scope makes a model look narrow: a contraction census over the seed model reports seven
    declared families as unevidenced, which reads as "the model never normalizes" when the model
    plainly does -- the census simply does not enumerate normalization. Supplying the scope splits
    that list into families the model really lacks and families nothing looked for. Left empty, the
    distinction cannot be drawn and the certificate says so rather than implying the stronger claim.
    """
    if not 0 < cover_target <= 1:
        raise ValueError(f"cover_target must be in (0, 1]; got {cover_target}")

    buckets: dict[str, list[Any]] = {}
    sigs: dict[str, Signature] = {}
    for row in census.rows:
        sig = _row_signature(row, boundaries=regime_boundaries)
        buckets.setdefault(sig.key(), []).append(row)
        sigs.setdefault(sig.key(), sig)

    caps = dict(cap_map)          # converted once; is_eligible takes a plain dict
    groups: list[Group] = []
    for key, rows in buckets.items():
        cost, source, lower = _group_cost(census, rows)
        # One verdict per group: the signature already fixes family and dtypes, which is what
        # eligibility keys on, so the first row speaks for the group.
        verdict = EL.is_eligible(_descriptor(rows[0]), caps)
        groups.append(Group(
            signature=sigs[key],
            row_indices=tuple(int(getattr(r, "index", i)) for i, r in enumerate(rows)),
            cost=cost, cost_source=source, weight_is_lower_bound=lower,
            eligible=bool(verdict.eligible) and not verdict.undetermined,
            undetermined=bool(verdict.undetermined),
            reason=verdict.reason,
            shapes=tuple(sorted({tuple(r.parallel or ()) + tuple(r.reduction or ())
                                 for r in rows})),
        ))

    # Undetermined leaves BOTH sides of the ratio.
    eligible = [g for g in groups if g.eligible]
    undetermined = [g for g in groups if g.undetermined]
    ineligible = [g for g in groups if not g.eligible and not g.undetermined]

    denominator = sum(g.cost for g in eligible)
    # Deterministic: cost descending, signature key breaking every tie.
    ordered = sorted(eligible, key=lambda g: (-g.cost, g.signature.key()))

    chosen: list[Group] = []
    covered = 0.0
    for g in ordered:
        if denominator > 0 and covered / denominator >= cover_target:
            break
        chosen.append(g)
        covered += g.cost

    # Family floor: one task for every family that is eligible AND evidenced in this model. A family
    # the target DECLARES but the model does not exercise gets no manufactured task -- inventing one
    # would put work in the basis that the model never does, and then report covering it.
    floor_added: list[str] = []
    if family_floor:
        have = {g.signature.family for g in chosen}
        for g in ordered:
            if g.signature.family not in have:
                chosen.append(g)
                covered += g.cost
                have.add(g.signature.family)
                floor_added.append(g.signature.family)

    evidenced = {g.signature.family for g in groups if g.signature.family}
    declared = {f for f in cap_map}
    scope = {f for f in census_enumerates}
    missing = declared - evidenced
    # A family the census cannot see is UNSEARCHED, not absent. Conflating the two would let a
    # narrow census be reported as a narrow model.
    outside_scope = sorted(missing - scope) if scope else []
    not_evidenced = sorted(missing & scope) if scope else sorted(missing)

    chosen_sorted = tuple(sorted(chosen, key=lambda g: (-g.cost, g.signature.key())))
    cert = {
        "census_model": getattr(census, "model", ""),
        "census_stage": getattr(census, "stage", ""),
        "census_source": getattr(census, "source", ""),
        "cover_target": cover_target,
        "cover_fraction": (covered / denominator) if denominator > 0 else None,
        "denominator": denominator,
        "denominator_source": ("measured_ticks"
                               if any(g.cost_source == "measured_ticks" for g in eligible)
                               else "work_share"),
        # True when ANY chosen group's weight is a lower bound: the cover fraction is then itself a
        # bound, and must not be quoted as an exact percentage.
        "cover_fraction_is_bounded": any(g.weight_is_lower_bound for g in chosen_sorted),
        "groups_total": len(groups),
        "groups_chosen": len(chosen_sorted),
        "families_covered": sorted({g.signature.family for g in chosen_sorted}),
        "families_evidenced": sorted(evidenced),
        # Declared by the target, and the census looked for them and did not find them.
        "families_declared_not_evidenced": not_evidenced,
        # Declared by the target, but this census never enumerates them -- says nothing about
        # whether the model uses them.
        "families_outside_census_scope": outside_scope,
        "census_enumerates": sorted(scope),
        "census_scope_known": bool(scope),
        "family_floor_added": floor_added,
        "excluded_undetermined": [
            {"signature": g.signature.key(), "cost": g.cost, "reason": g.reason}
            for g in sorted(undetermined, key=lambda g: g.signature.key())],
        "excluded_ineligible": [
            {"signature": g.signature.key(), "cost": g.cost, "reason": g.reason}
            for g in sorted(ineligible, key=lambda g: g.signature.key())],
        "eligible_not_chosen": [
            {"signature": g.signature.key(), "cost": g.cost}
            for g in ordered if g not in chosen_sorted],
        "regime_boundaries": list(regime_boundaries),
        "regime_source": "target_facts" if regime_boundaries else "log2_volume_fallback",
    }
    return TaskBasis(entries=chosen_sorted, certificate=cert)
