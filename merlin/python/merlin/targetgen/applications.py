"""The shapes a user's own applications contain, grouped by what the compiler must DO with them.

The corpus derives its capsules from a target's RTL facts and a three-key workload spec, and every
synthesized capsule gets one of two shapes: ``corpus_synth.extents_for`` returns ``{M: tile, K:
2*tile, N: tile}`` or the partial variant, for the whole corpus. Measured against the real captures,
that is not what models contain -- a model holds 21-25 distinct contraction shapes and they look
like ``(256,196)x2304``, ``(128,9216)x2304``, batched ``(32,8,345)x72``, while the corpus tests
16x16x16. The top five shapes carry 53-68% of all multiply-accumulate work and none of them is
exercised anywhere.

WHY NOT SIMPLY EMIT THE TOP-N SHAPES. Because N is a threshold nobody can defend, and because a
shape is not the thing a compiler gets wrong -- a BEHAVIOUR is. Two shapes that tile the same way,
spill the same way and carry the same arithmetic exercise one code path between them; a capsule for
each buys nothing. So regions are grouped by the axes that actually decide what the compiler must
do:

    (family, dtype, alignment vs the tile edge, memory regime, rank, geometry class)

Every one of those already exists and is reused rather than restated: the first three are the
conformance cell's own vocabulary, ``memory_regime.classify`` gives the fourth from the target's
operand store, rank comes off the iteration space, and ``shape_taxonomy.classify_geometry`` gives
the last (gemv_like / tall_skinny / squareish_gemm / odd_tail_heavy / small_dispatch_fragment ...).

For each OCCUPIED class the representative is the highest-work real shape in it. That is what makes
the coverage claim sayable: every region falls in some class, every occupied class has a capsule
drawn from that class at a shape the application actually contains, and the capsule count is bounded
by the lattice rather than by how large the model is.

SHAPES COME FROM ``kernels.shapes``, NOT FROM THE REGION WALK. ``model_coverage.regions_from_module``
resolves family and dtype and leaves ``m``/``k``/``n``/``rank`` unset -- measured at 100% None across
198, 1557 and 496 regions of three real captures. ``observe_contractions`` is the reader that
recovers the iteration space, and it classifies contraction GENERICS structurally, which matters
because the int8 rewrite leaves generics behind and a name-only reader reports zero contractions on
every int8 capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["RegionClass", "ClassEvidence", "SizedCapsule", "classify_capture", "classify_captures",
           "size_class"]


@dataclass(frozen=True)
class RegionClass:
    """What the compiler must do with a region — the grouping key, not the shape."""

    family: str
    dtype: str
    alignment: str                                 # aligned | partial (vs the target's tile edge)
    regime: str                                    # memory_regime.classify verdict
    rank: int
    geometry: str                                  # shape_taxonomy.classify_geometry verdict

    def key(self) -> str:
        return f"{self.family}/{self.dtype}/{self.alignment}/{self.regime}/rank{self.rank}/{self.geometry}"


@dataclass(frozen=True)
class ClassEvidence:
    """One occupied class: its representative real shape and the mass behind it."""

    region_class: RegionClass
    m: int
    k: int
    n: int
    batch: int
    multiplicity: int                              # regions in this class
    work: int                                      # summed multiply-accumulates
    work_complete: bool                            # False when any member's extents were partial
    source: str                                    # the capture the representative came from

    def to_dict(self) -> dict:
        return {
            "class": self.region_class.key(),
            "family": self.region_class.family, "dtype": self.region_class.dtype,
            "alignment": self.region_class.alignment, "regime": self.region_class.regime,
            "rank": self.region_class.rank, "geometry": self.region_class.geometry,
            "M": self.m, "K": self.k, "N": self.n, "batch": self.batch,
            "multiplicity": self.multiplicity, "work": self.work,
            "work_complete": self.work_complete, "source": self.source,
        }


def _mkn(shape) -> "tuple[int, int, int, int] | None":
    """``(M, K, N, batch)`` for a contraction, or ``None`` when its iteration space is unreadable.

    ``parallel`` is ``(batch..., M, N)`` and ``reduction[0]`` is K, which is the convention
    ``kernels.shapes`` records and the one ``system.offload``'s selector already keys on. A shape
    with fewer than two parallel dims or no reduction dim is not a contraction this can size.
    """
    parallel = [int(d) for d in (getattr(shape, "parallel", ()) or ())]
    reduction = [int(d) for d in (getattr(shape, "reduction", ()) or ())]
    if len(parallel) < 2 or not reduction:
        return None
    batch = 1
    for d in parallel[:-2]:
        batch *= d
    return parallel[-2], reduction[0], parallel[-1], batch


def _alignment(m: int, k: int, n: int, tile: int | None) -> str:
    """Whether every extent is a whole multiple of the tile edge.

    ``unknown`` when the target declares no edge -- not ``aligned``, because "there is no edge" and
    "it lines up with the edge" are different facts and only one of them says anything about tails.
    """
    if not tile:
        return "unknown"
    return "aligned" if all(x % tile == 0 for x in (m, k, n)) else "partial"


def _regime(target: str, m: int, k: int, n: int, dtype: str | None, cache: dict) -> str:
    """The operand-store residency this shape demands, via the same classifier the corpus uses."""
    from merlin.targetgen import memory_regime as MR

    if "store" not in cache:
        try:
            cache["store"], cache["capacity"] = MR.operand_store(target)
        except Exception:                          # noqa: BLE001 -- an underivable store is not a regime
            cache["store"], cache["capacity"] = None, None
    store, capacity = cache["store"], cache["capacity"]
    if store is None or not capacity:
        return "unknown"
    try:
        rows = MR.deep_k_rows(store, int(k), m_extent=int(m), n_extent=int(n), dtype=dtype)
    except Exception:                              # noqa: BLE001
        return "unknown"
    return MR.classify(rows, rows, capacity)


def _dtype_of(shape, fallback: str | None) -> str:
    """The operand dtype this contraction carries, or the capture's dominant one.

    ``ContractionShape.dtypes`` is positionally ``(lhs, rhs, out)`` in MLIR spelling and is empty
    when the observer could not read it; the capture-level fallback keeps a region in its class
    rather than dropping it into an ``unknown`` bucket that would then look like its own behaviour.
    """
    dtypes = tuple(getattr(shape, "dtypes", ()) or ())
    token = str(dtypes[0]) if dtypes else (fallback or "unknown")
    try:
        from merlin.targetgen.conformance import capsule_dtype
        return capsule_dtype(token)
    except Exception:                              # noqa: BLE001 -- keep an unmappable token visible
        return token


def classify_capture(capture: str | Path, target: str, *,
                     dtype_hint: str | None = None) -> "list[ClassEvidence]":
    """Group one capture's contractions into behavioural classes, heaviest representative each.

    Returns ``[]`` for a capture with no readable contraction -- an unreadable model is evidence
    neither for nor against a requirement, the same rule ``conformance.observed`` follows.
    """
    from merlin.dse_guidance.shape_taxonomy import classify_geometry
    from merlin.kernels.shapes import observe_contractions
    from merlin.targetgen.corpus_spec import _tile_dim  # noqa: PLC2701 -- one tile-edge derivation

    try:
        observed = observe_contractions(Path(capture))
    except Exception:                              # noqa: BLE001 -- an unreadable capture yields none
        return []
    if not observed:
        return []

    try:
        from merlin.targetgen.target_registry import load_contract
        tile = int(_tile_dim(target, load_contract(target)) or 0)
    except Exception:                              # noqa: BLE001 -- no edge is a real answer
        tile = 0

    cache: dict = {}
    grouped: dict[RegionClass, list] = {}
    for _op, shape in observed:
        sized = _mkn(shape)
        if sized is None:
            continue
        m, k, n, batch = sized
        dtype = _dtype_of(shape, dtype_hint)
        rank = len(tuple(getattr(shape, "parallel", ()) or ()))
        cls = RegionClass(
            family="contraction", dtype=dtype,
            alignment=_alignment(m, k, n, tile),
            regime=_regime(target, m, k, n, dtype, cache),
            rank=rank,
            geometry=classify_geometry(m, n, k),
        )
        grouped.setdefault(cls, []).append((m, k, n, batch))

    out: list[ClassEvidence] = []
    for cls, members in grouped.items():
        # THE HEAVIEST MEMBER REPRESENTS THE CLASS. Every member exercises the same compiler
        # behaviour by construction -- that is what the class means -- so the one carrying the most
        # work is the one whose cost and numerics are worth reproducing.
        work_of = {mem: mem[0] * mem[1] * mem[2] * mem[3] for mem in members}
        rep = max(members, key=lambda mem: work_of[mem])
        out.append(ClassEvidence(
            region_class=cls, m=rep[0], k=rep[1], n=rep[2], batch=rep[3],
            multiplicity=len(members), work=sum(work_of.values()), work_complete=True,
            source=Path(capture).parent.name,
        ))
    out.sort(key=lambda e: (-e.work, e.region_class.key()))
    return out


def classify_captures(captures: dict, target: str) -> dict:
    """Every application's classes, merged, with the work coverage the representatives account for.

    ``captures`` is ``{label: path}`` -- the same shape the conformance axes already take, so an
    application store and the roster store are interchangeable here.

    The returned ``work_coverage`` is the point of the whole exercise: it is the fraction of
    multiply-accumulate work that lives in classes a capsule was emitted for. It is REPORTED rather
    than assumed, because a corpus that covers every class of a model it has mostly not looked at is
    a different claim from one that covers the work.
    """
    merged: dict[RegionClass, ClassEvidence] = {}
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            evidence = classify_capture(path, target)
        except Exception as exc:                   # noqa: BLE001 -- reported, never skipped silently
            unreadable[str(label)] = f"{type(exc).__name__}: {str(exc)[-160:]}"
            continue
        for ev in evidence:
            prior = merged.get(ev.region_class)
            if prior is None:
                merged[ev.region_class] = ev
                continue
            # Merge: sum the mass, keep the heavier representative.
            heavier = ev if (ev.m * ev.k * ev.n * ev.batch) > (prior.m * prior.k * prior.n * prior.batch) else prior
            merged[ev.region_class] = ClassEvidence(
                region_class=ev.region_class, m=heavier.m, k=heavier.k, n=heavier.n,
                batch=heavier.batch, multiplicity=prior.multiplicity + ev.multiplicity,
                work=prior.work + ev.work,
                work_complete=prior.work_complete and ev.work_complete,
                source=heavier.source,
            )

    classes = sorted(merged.values(), key=lambda e: (-e.work, e.region_class.key()))
    total = sum(e.work for e in classes)
    return {
        "classes": [e.to_dict() for e in classes],
        "n_classes": len(classes),
        "n_regions": sum(e.multiplicity for e in classes),
        "total_work": total,
        "work_coverage": 1.0 if total else None,
        "captures_unreadable": unreadable,
        "axis_basis": (
            "the contraction shapes the declared applications actually contain, grouped by what the "
            "compiler must DO with them -- (family, dtype, alignment, memory regime, rank, geometry) "
            "-- with the heaviest real shape in each class as its representative. A capsule per "
            "distinct shape would weight a one-off the same as a shape appearing 52 times; a top-N "
            "cut would rest on a threshold nobody can defend. Grouping by behaviour bounds the "
            "capsule count by the lattice instead of by the size of the model"),
    }


# ---------------------------------------------------------------------------------------------
# Sizing: a class becomes one capsule that can be certified and, when the application is bigger
# than that, a second that extends it.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SizedCapsule:
    """One capsule derived from a class, with the tier it can afford and why."""

    region_class: RegionClass
    m: int
    k: int
    n: int
    batch: int
    tier: str                                      # the deepest tier this size can afford
    extends: "str | None"                          # the sibling this one rests on, when L2-only
    basis: dict

    def to_dict(self) -> dict:
        return {"class": self.region_class.key(), "M": self.m, "K": self.k, "N": self.n,
                "batch": self.batch, "tier": self.tier, "extends": self.extends,
                "basis": dict(self.basis)}


def _round_down_to_tile(value: int, tile: int) -> int:
    """The largest whole number of tiles at or below ``value``, never zero."""
    if tile <= 1:
        return max(1, int(value))
    return max(tile, (int(value) // tile) * tile)


def size_class(evidence: "ClassEvidence", *, target: str, budget_s: float,
               tile: int | None = None, fit=None) -> "tuple[list[SizedCapsule], str | None]":
    """``([capsules], refusal)`` for one behavioural class.

    THE CONSTRAINT THAT DECIDES WHETHER ANY OF THIS IS USABLE. A capsule at an application's real
    shape is worthless if nobody can afford to certify it: the heaviest class measured here carries a
    65.5M-element operand against a cost model whose measured range tops out at 4096. Emitting it and
    calling it certified would be a claim nothing could pay for.

    So each class yields up to two capsules:

    * a **cycle-accurate** one, at the largest size the cost model puts inside ``budget_s``. K is
      preserved -- it is where accumulation, residency and spill behaviour live, and it is what makes
      the capsule a member of this class at all -- and only the parallel extents are clamped, then
      rounded to whole tiles.
    * an **L2-only** one at the application's true shape, emitted ONLY when the true shape is bigger
      than the clamp, and always naming the cycle-accurate capsule it ``extends``. That is what keeps
      the large, representative, perf-facing capsule resting on a functional guarantee instead of
      replacing it. An L2 pass on a shape nothing ever certified cycle-accurately is exactly the
      "read ``tier_reached``, never a bare score" failure this corpus already has scar tissue for.

    When no cost model exists the affordable size falls back to the corpus's own tile convention --
    the size this target is already certifying today, so a sibling provably exists -- and the basis
    records that the number came from convention rather than measurement. The refusal is returned,
    never raised, when even one tile of this class's K cannot be afforded: that is a real statement
    about the class, and dropping it silently would make an unaffordable behaviour look absent.
    """
    from merlin.targetgen import cert_cost as CC

    tile = int(tile or 0) or 1
    k = int(evidence.k)
    budget_elements = CC.max_elements_within(fit, budget_s) if fit is not None else None
    if budget_elements is None:
        # NO MEASUREMENT, NO CAPSULE. A tile-convention fallback suggests itself and is wrong: the
        # convention is M=tile, K=2*tile, N=tile, and this class's K is whatever the application
        # carries -- 2048 in the measured LM-head case, 32x the convention. Clamping the parallel
        # extents to a tile while leaving K there is NOT "the size the corpus already certifies", so
        # calling it known-payable would be an invention. Clamping K instead would put the capsule in
        # a different behavioural class, which is the one thing this axis may not do.
        #
        # So the class is refused with the reason. The consequence is deliberate: a target must have
        # certified something before application-derived capsules are admitted for it, because
        # otherwise the large L2 capsule would rest on a sibling nobody could size.
        return [], (f"{evidence.region_class.key()}: no measured certification history for "
                    f"{target!r}, so no size of this class can be shown affordable; certify this "
                    f"target's existing corpus first, then the fit gives a size")
    else:
        # max operand elements is max(M*K, K*N), so both parallel extents are bounded by
        # budget_elements / K. K is never clamped: a shallower K is a different behavioural class.
        per_extent = budget_elements // max(1, k)
        if per_extent < tile:
            return [], (f"{evidence.region_class.key()}: K={k} alone needs {k * tile} elements for a "
                        f"single tile of parallel extent, over the {budget_elements}-element budget "
                        f"at {budget_s:.0f}s; no size of this class is certifiable here")
        affordable_m = _round_down_to_tile(min(per_extent, evidence.m), tile)
        affordable_n = _round_down_to_tile(min(per_extent, evidence.n), tile)
        size_basis = {"sized_by": "measured_cost_model", "budget_s": budget_s,
                      "budget_elements": budget_elements,
                      "fitted_seconds": CC.predict_seconds(fit, max(affordable_m, affordable_n) * k),
                      "cost_fit": fit.to_dict()}

    cert = SizedCapsule(
        region_class=evidence.region_class, m=affordable_m, k=k, n=affordable_n,
        batch=int(evidence.batch), tier="L3", extends=None,
        basis={**size_basis, "clamped_from": [evidence.m, evidence.k, evidence.n],
               "clamped": [affordable_m, k, affordable_n] != [evidence.m, k, evidence.n],
               "representative_of": evidence.multiplicity, "work": evidence.work,
               "source": evidence.source},
    )
    out = [cert]

    if (evidence.m, evidence.n) != (affordable_m, affordable_n):
        # The application's own shape, kept as an L2 EXTENSION of the certified one. Perf wants this
        # size; correctness rests on its smaller sibling having been certified cycle-accurately.
        out.append(SizedCapsule(
            region_class=evidence.region_class, m=int(evidence.m), k=k, n=int(evidence.n),
            batch=int(evidence.batch), tier="L2", extends=cert.region_class.key(),
            basis={"sized_by": "application_shape",
                   "why": "the shape the application actually contains, too large to certify "
                          "cycle-accurately; admissible only as an extension of the certified "
                          "sibling of the same behavioural class",
                   "representative_of": evidence.multiplicity, "work": evidence.work,
                   "source": evidence.source},
        ))
    return out, None
