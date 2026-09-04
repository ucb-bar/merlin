"""Which memory-mapping regime a program puts a target in — derived, and measured on both sides.

The coverage cells say WHAT a corpus computes and the composition shapes say HOW it is assembled. Both
are silent about WHERE the operands were put, and on an accelerator with a small explicitly-managed
on-chip store that is where the correctness failures and the performance failures both live.

Measured, and the reason this module exists. Against the operand store derived for the interlocked
target here — 16384 rows of 16 bytes, four banks:

  * of 1829 contraction regions across 20 real captured models, **90.1% exceed the store however they
    are allocated**, 69.8% exceed the accumulator too, and only 9.2% fit twice over;
  * of the 37 public capsules in that target's corpus, **100% fit twice over**, and the largest of them
    uses **2.06% of capacity**.

So the corpus exercises the 9% regime exclusively and the 90% regime never. A schedule that loads every
weight tile up front is not punished by any capsule, and the one time it mattered a graded backend
addressed all kt*nt tiles as simultaneously resident, asked for 16384 rows against 16384, and the
simulator aborted three layers away in a range check — a failure indistinguishable from an unreachable
oracle.

Re-derived later against the capture store as it then stood, the imbalance had not moved: 1600
contraction regions across the 17 requirement-deriving captures put **97.2% in spills**, 1.5% in
``fits_double`` and 1.3% in ``fits_single`` (2106 regions and 96.1% with the held-out claim models
folded back in). The FUNCTIONAL corpus had by then acquired two capsules outside ``fits_double``, but
the generated PERFORMANCE corpus was still 16 of 16 ``fits_double`` with its largest member at 1.6% of
capacity — so every performance coefficient was fitted where essentially none of the work lands. That
is what :func:`reduction_depth_regimes` exists to fix.

The regimes are not adjectives. Each one changes what the compiler is even allowed to do:

``fits_double``
    the working set fits TWICE, so movement for the next tile can overlap compute on the current one.
    The only regime in which failing to double-buffer is a defect rather than an impossibility.
``fits_single``
    fits once. Staging cannot happen; serialising is correct. Keeping this apart from ``fits_double`` is
    what stops us charging a compiler for an overlap the hardware could not have provided.
``fits_on_reuse``
    the sum of LIVE RANGES fits but the sum of all tensors does not, so only an allocator that reuses
    rows freed by a dead tensor works. A bump allocator fails, and on a store that wraps it fails
    silently.
``spills``
    exceeds capacity however it is allocated. The question stops being "does it fit" and becomes "how
    much re-load traffic did the loop order cost".
``unknown``
    the target declares no capacity we could derive, or the program declares no shape. Never folded into
    a fitting regime: an unmeasurable capacity reported as satisfied is exactly how the abort above
    reached the simulator with nothing recorded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

FITS_DOUBLE = "fits_double"
FITS_SINGLE = "fits_single"
FITS_ON_REUSE = "fits_on_reuse"
SPILLS = "spills"
UNKNOWN = "unknown"

#: Weakest-demand first. A corpus that covers a later entry has exercised a strictly harder mapping
#: problem than one that covers only earlier entries.
ORDER = (FITS_DOUBLE, FITS_SINGLE, FITS_ON_REUSE, SPILLS)


def classify(rows_live: int | None, rows_total: int | None, capacity_rows: int | None) -> str:
    """The regime a working set of ``rows_live`` (peak concurrently live) and ``rows_total`` occupies.

    ``rows_total`` separates ``fits_on_reuse`` from the fitting regimes: when everything the program
    touches would not fit but its peak live set does, only reuse makes the program legal, and that is a
    real obligation on the allocator rather than a property of the arithmetic.
    """
    if not capacity_rows or rows_live is None:
        return UNKNOWN
    if rows_live > capacity_rows:
        return SPILLS
    if rows_total is not None and rows_total > capacity_rows:
        return FITS_ON_REUSE
    return FITS_DOUBLE if rows_live * 2 <= capacity_rows else FITS_SINGLE


def operand_store(target: str, *, dtype: str | None = None):
    """``(Store, capacity_rows)`` for the target's operand store, or ``(None, None)``.

    Picks the NARROWEST-row store, because that is the one operands live in: a separate accumulator
    space exists precisely because its row is wider (it holds the accumulate type). Choosing by name
    would bake a target's spelling into shared code.

    ``dtype`` IS PART OF THE CAPACITY on a lane-granular store. A device with no compute array reaches
    its store one warp at a time: the access width is a lane COUNT, so a row's size -- and therefore
    how many rows the store holds -- depends on the element in it. Requiring a fixed byte row here
    discarded such a store entirely and reported the target as declaring no operand store at all, so
    its whole memory-mapping axis read `0 / 0 required regimes` over a 128 KiB shared memory.
    """
    try:
        from merlin.targetgen import address_space as AS
        space = AS.derive_address_space(target)
    except Exception:                                          # noqa: BLE001 — unresolvable target
        return None, None
    stores = [s for s in (getattr(space, "stores", ()) or ())
              if getattr(s, "row_bytes", None) or getattr(s, "row_elems", None)]
    if not stores:
        return None, None

    def _width(s):
        """Row width in bits at this dtype -- the one scale on which a fixed-byte row and a
        lane-granular one are comparable."""
        if getattr(s, "row_bytes", None):
            return int(s.row_bytes) * 8
        per_row = s.elems_per_row(dtype)
        from merlin.targetgen.address_space import element_bits
        bits = element_bits(dtype) if dtype else s.element_bits
        return per_row * bits if (per_row and bits) else float("inf")

    store = min(stores, key=_width)
    return store, store.capacity_rows(dtype)


def _dominant_dtype(capsule_doc: dict) -> str | None:
    """The element type a capsule's operands are in, or ``None`` when it declares none.

    The first declared input's dtype: a capsule's inputs are one datapath's operands and share a type
    in every capsule this corpus emits. When they do NOT, ``capsule_regime`` reports the disagreement
    rather than averaging over it -- summing rows measured against two different row sizes is adding
    two different units.
    """
    for t in (capsule_doc.get("inputs") or []):
        if t.get("dtype"):
            return str(t["dtype"])
    return None


@lru_cache(maxsize=None)
def _store_for(target: str, dtype: str | None):
    """``operand_store`` memoised per (target, dtype) -- the region walk asks per region."""
    return operand_store(target, dtype=dtype)


def _rows(store, shape, dtype) -> int | None:
    try:
        return store.working_set_rows(shape, dtype)
    except Exception:                                          # noqa: BLE001
        return None


def capsule_regime(capsule_dir: str | Path, target: str, *, store=None, capacity=None) -> dict:
    """The regime one capsule's declared tensors put the target in.

    Uses the capsule's OWN declared inputs, which is what the grader materialises, rather than anything
    inferred from its MLIR — so the number is the residency the harness actually asks for.
    """
    import yaml

    cy = Path(capsule_dir) / "capsule.yaml"
    if not cy.is_file():
        return {"regime": UNKNOWN, "why": "no capsule.yaml"}
    try:
        doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return {"regime": UNKNOWN, "why": f"unparseable capsule.yaml: {type(e).__name__}"}
    # THE CAPSULE'S OWN DTYPE DECIDES HOW BIG A ROW IS on a lane-granular store (see `operand_store`),
    # so the store has to be resolved AFTER the capsule is read, not before. Caller-supplied values are
    # still honoured -- a caller that resolved for this dtype should not pay for it per capsule.
    if store is None:
        store, capacity = operand_store(target, dtype=_dominant_dtype(doc))
    if store is None or not capacity:
        return {"regime": UNKNOWN, "why": f"{target!r} declares no operand-store capacity we can derive"}

    rows, unsized = 0, []
    for t in (doc.get("inputs") or []):
        shape = [int(x) for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()]
        if not shape:
            unsized.append(t.get("name") or "?")
            continue
        r = _rows(store, shape, t.get("dtype"))
        if r is None:
            unsized.append(t.get("name") or "?")
            continue
        rows += int(r)
    if unsized and not rows:
        return {"regime": UNKNOWN, "why": f"no input shape could be sized ({', '.join(unsized)})"}
    # Every declared input is live at once for a capsule (the harness materialises them all before the
    # program runs), so peak-live and total coincide here and `fits_on_reuse` cannot arise from a
    # capsule's inputs alone. Said out loud rather than left as an accident of the arithmetic.
    out = {"regime": classify(rows, rows, capacity), "rows": rows, "capacity_rows": int(capacity),
           "fraction_of_capacity": round(rows / float(capacity), 6)}
    if unsized:
        out["unsized_inputs"] = sorted(unsized)
    return out


def corpus_regimes(corpus_roots, target: str, *, labels=None, exclude=None) -> dict:
    """``regime -> [capsule names]`` for a corpus, plus the largest working set observed."""
    import yaml

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    # NOT resolved once here: each capsule's row size depends on its own element type on a
    # lane-granular store, so `capsule_regime` resolves it per capsule (memoised by dtype).
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    capacity = None
    by: dict[str, list[str]] = {}
    largest = {"name": None, "rows": 0, "fraction_of_capacity": 0.0}
    for root in roots:
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = cap.get("name") or cy.parent.name
            if name in exclude:
                continue
            got = capsule_regime(cy.parent, target)
            by.setdefault(got["regime"], []).append(name)
            if got.get("capacity_rows") and capacity is None:
                capacity = got["capacity_rows"]
            if got.get("rows", 0) > largest["rows"]:
                largest = {"name": name, "rows": got["rows"],
                           "fraction_of_capacity": got.get("fraction_of_capacity", 0.0)}
    return {"by_regime": {k: sorted(v) for k, v in sorted(by.items())},
            "capacity_rows": int(capacity) if capacity else None,
            "largest_working_set": largest}


def _operand_shapes(module):
    """``[(shape, dtype)]`` per computation-carrying region, for the operands it reads.

    Read off the parsed IR's own operand types rather than from any annotation: a region's residency is
    decided by the tensors it actually touches, and a capture's provenance tags are a hint that disagrees
    with the IR often enough to be untrustworthy for arithmetic.
    """
    from merlin.targetgen import model_coverage as mc

    out = []
    for op in module.walk():
        if not mc._is_region_op(op):
            continue
        shapes = []
        for operand in op.operands:
            ty = operand.type
            if not hasattr(ty, "get_shape"):
                continue
            try:
                shapes.append(list(ty.get_shape()))
            except Exception:                                  # noqa: BLE001
                continue
        if shapes:
            out.append((mc._short_op(op.name), shapes, mc._elem_dtype(op)))
    return out


def required_regimes(captures: dict, target: str) -> dict:
    """``regime -> [capture labels]`` — the regimes REAL models put this target in.

    Same evidential rule as the other two axes: a regime no real model occupies is not required of the
    corpus. The measured answer for the interlocked target here is that 90.1% of contraction regions
    across 20 captures land in ``spills`` and only 9.2% fit twice, against a corpus that is 100%
    ``fits_double`` -- so the corpus exercises the rare regime exclusively and the common one never.
    """
    from merlin.targetgen import model_coverage as mc

    # Resolved PER REGION below (`_store_for`), because a lane-granular store's row size follows the
    # region's own element type. This probe only answers "is there a store at all".
    # Resolved PER REGION below (`_store_for`), because a lane-granular store's row size follows the
    # region's own element type -- so the CAPACITY from a dtype-less probe is legitimately None there
    # and the guard asks only whether a store exists at all.
    store, _probe_capacity = operand_store(target)
    capacity = None
    by: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    unreadable: dict[str, str] = {}
    if store is None:
        return {"by_regime": {}, "region_counts": {}, "captures_unreadable": {},
                "why": f"{target!r} declares no operand-store capacity we can derive, so no regime is "
                       f"required of the corpus -- that is 'we do not know', never 'nothing is required'"}
    for label, path in sorted((captures or {}).items()):
        try:
            module = mc.load_module(path)
        except Exception as e:                                 # noqa: BLE001
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        for _op, shapes, dtype in _operand_shapes(module):
            store, capacity = _store_for(target, dtype)
            if store is None or not capacity:
                continue
            rows = 0
            sized = False
            for shape in shapes:
                r = _rows(store, shape, dtype)
                if r is not None:
                    rows += int(r)
                    sized = True
            if not sized:
                continue
            reg = classify(rows, rows, capacity)
            if reg == UNKNOWN:
                continue
            counts[reg] = counts.get(reg, 0) + 1
            if label not in by.setdefault(reg, []):
                by[reg].append(label)
    return {"by_regime": {k: sorted(v) for k, v in sorted(by.items())},
            "region_counts": dict(sorted(counts.items())),
            "capacity_rows": int(capacity) if capacity else None,
            "captures_unreadable": unreadable}


def uncovered_regimes(required: dict, corpus: dict) -> dict:
    """Regimes real models present that no capsule in the corpus reaches."""
    want = [r for r in (required or {}).get("by_regime") or {} if r != UNKNOWN]
    have = {r for r in (corpus or {}).get("by_regime") or {} if r != UNKNOWN}
    missing = [r for r in ORDER if r in set(want) - have]
    return {
        "n_required": len(want),
        "n_covered": len(set(want) & have),
        "uncovered": missing,
        "corpus_regimes": sorted(have),
        "capacity_rows": corpus.get("capacity_rows"),
        "largest_working_set": corpus.get("largest_working_set"),
        "note": ("a regime real models occupy that no capsule reaches means the corpus cannot detect a "
                 "memory-mapping failure of that kind; on an interlocked target the schedule is correct "
                 "whatever it chooses, so nothing else will report it either"),
    }


#: Tile multiples the regime search walks, smallest first. Powers of two with the intermediate multiple
#: between each pair, so a band narrower than a doubling (``fits_single`` is exactly one such band) still
#: has a candidate inside it rather than being stepped over.
_REGIME_MULTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def extents_for_regime(target: str, regime: str, *, tile_dim: int, dtype: str | None = None,
                       store=None, capacity=None) -> dict | None:
    """Tile-relative ``{M, K, N}`` that put a matmul capsule's declared inputs in ``regime``.

    Found by SEARCHING with the very functions the coverage gate measures with -- :func:`classify` over
    :func:`merlin.targetgen.address_space.working_set_rows` -- never with a second model of the store. A
    synthesizer that predicted residency its own way could emit a capsule the gate then classifies into
    a different regime, and the requirement would read as covered while the regime stayed untouched.

    The operands are sized exactly as :func:`capsule_regime` sizes a capsule's declared inputs: ``A0`` is
    ``[M, K]`` and ``W`` is ``[K, N]``, both live at once, so peak-live and total coincide and
    ``fits_on_reuse`` is unreachable from a capsule's inputs alone -- asking for it returns ``None``
    rather than a capsule that would be classified as something else.

    ``None`` when the target declares no operand store we can size, or when no multiple in the search
    reaches the regime. That is a reportable gap; it is never a quietly smaller capsule.
    """
    if store is None and capacity is None:
        store, capacity = operand_store(target, dtype=dtype)
    if store is None or not capacity or not tile_dim:
        return None
    tile = int(tile_dim)
    per_row_probe = _rows(store, [tile, tile], dtype)
    if per_row_probe is None:
        return None
    # TWO axes, not one. Scaling M, K and N together multiplies the working set by the square of the
    # multiple, which steps straight over `fits_single`: on a 16384-row store that band is
    # (8192, 16384], and square scaling jumps 8192 -> 18432. Varying the square edge and the M extent
    # independently reaches every band. Candidates are ordered by the residency they produce, so the
    # answer is the SMALLEST capsule in the band -- a spills capsule is paid for on every grade.
    cands = []
    for m_mult in _REGIME_MULTS:
        for e_mult in _REGIME_MULTS:
            m, edge = tile * m_mult, tile * e_mult
            a_rows = _rows(store, [m, edge], dtype)
            w_rows = _rows(store, [edge, edge], dtype)
            if a_rows is None or w_rows is None:
                return None
            cands.append((int(a_rows) + int(w_rows), m_mult, e_mult))
    for total, m_mult, e_mult in sorted(cands):
        if classify(total, total, capacity) == regime:
            tok = lambda k: f"{k}*tile" if k != 1 else "tile"      # noqa: E731 -- local spelling helper
            return {"M": tok(m_mult), "K": tok(e_mult), "N": tok(e_mult),
                    "rows": total, "capacity_rows": int(capacity),
                    "fraction_of_capacity": round(total / float(capacity), 6)}
    return None


def required_regime_extents(target: str, regimes, *, tile_dim: int, dtype: str | None = None) -> dict:
    """``{regime: extents-or-None}`` for every regime in ``regimes``, resolved once against one store.

    Emitted into the conformance spec so :mod:`merlin.targetgen.corpus_synth` stays pure: the search
    needs the target's address space, which is exactly the I/O the synthesizer is not allowed to do.
    """
    store, capacity = operand_store(target, dtype=dtype)
    out: dict = {}
    for regime in regimes:
        out[str(regime)] = extents_for_regime(target, str(regime), tile_dim=tile_dim, dtype=dtype,
                                              store=store, capacity=capacity)
    return out


#: The regimes a set of ALL-LIVE operands can occupy, ordered by the residency that produces them.
#: ``fits_on_reuse`` is deliberately absent: it is defined by peak-live being smaller than the sum of
#: all tensors, and a capsule's declared inputs are every one of them live at once, so the two numbers
#: coincide and the band cannot exist. Said here once, so the search below can rely on the sequence
#: being monotone in rows and the caller gets a REASON rather than an empty list.
_ROWS_MONOTONE_ORDER = (FITS_DOUBLE, FITS_SINGLE, SPILLS)

#: Why ``fits_on_reuse`` is never a point. Returned verbatim so a corpus gate can print the reason it
#: found no capsule rather than reporting a silent absence.
UNREACHABLE_ON_ALL_LIVE_INPUTS = (
    "unreachable from a capsule's declared inputs: the harness materialises every declared input before "
    "the program runs, so peak-live equals the sum of all tensors and the band that separates them "
    "(rows_total > capacity >= rows_live) is empty by construction. Reaching it needs a program with a "
    "DEAD tensor -- an allocator obligation, not an operand-size one -- which the capsule input "
    "declaration cannot express"
)


def deep_k_rows(store, k: int, *, m_extent: int, n_extent: int, dtype: str | None = None) -> int | None:
    """Operand rows a ``[m_extent, k] x [k, n_extent]`` contraction occupies, or ``None``.

    The DEEP-K shape, and the reason the regime sweep uses it rather than growing the parallel extents:
    the output is ``m_extent x n_extent`` whatever ``k`` is, so every point in the sweep drains the same
    number of result bytes. That keeps the L3 timing cert affordable (its cost tracks output size) and,
    more importantly, keeps the fill/drain intercept the fit is trying to separate from the rate ACTUALLY
    fixed across the points instead of growing with them.
    """
    a_rows = _rows(store, [int(m_extent), int(k)], dtype)
    w_rows = _rows(store, [int(k), int(n_extent)], dtype)
    if a_rows is None or w_rows is None:
        return None
    return int(a_rows) + int(w_rows)


def reduction_depth_regimes(target: str, regimes=None, *, tile_dim: int, dtype: str | None = None,
                            m_tiles: int = 1, n_tiles: int = 1, points_per_regime: int = 2,
                            spills_max_fraction: float = 2.0, store=None, capacity=None) -> dict:
    """Reduction depths that put a deep-K contraction in each requested regime — several per band.

    :func:`extents_for_regime` answers "give me ONE capsule in this regime", which is what a conformance
    cell needs: the cell is covered or it is not. A performance fit needs something strictly stronger.
    A rate and a fixed fill/drain intercept are two parameters, so separating them needs at least two
    points inside the SAME regime, and two points that sit next to each other at the band edge separate
    nothing -- the whole spread has to be inside the band. Measured on the interlocked target here, the
    corpus reached ``spills`` with exactly one capsule at 1.002 of capacity, so any rate fitted there was
    a one-point extrapolation dressed as a fit.

    ``K`` is the only varied axis: see :func:`deep_k_rows` for why. Points are spread evenly across each
    band's reachable tile multiples rather than taken from its edge.

    Returns ``{"by_regime": {regime: {"points": [...], "unreachable": reason-or-None}}, ...}``. A regime
    with no reachable shape returns its REASON, never an empty list with no explanation: an unreachable
    regime is an answer, a silently missing one is not.
    """
    if store is None and capacity is None:
        store, capacity = operand_store(target, dtype=dtype)
    wanted = [str(r) for r in (regimes if regimes is not None else ORDER)]
    tile = int(tile_dim or 0)
    out: dict = {"capacity_rows": int(capacity) if capacity else None, "tile_dim": tile or None,
                 "m_tiles": int(m_tiles), "n_tiles": int(n_tiles), "by_regime": {}}
    if store is None or not capacity or tile < 1:
        why = (f"{target!r} declares no operand-store capacity we can derive"
               if store is None or not capacity else "no tile edge was supplied")
        out["by_regime"] = {r: {"points": [], "unreachable": why} for r in wanted}
        return out

    m_extent, n_extent = tile * int(m_tiles), tile * int(n_tiles)
    out["m_extent"], out["n_extent"] = m_extent, n_extent

    def rows_at(mult: int):
        return deep_k_rows(store, tile * int(mult), m_extent=m_extent, n_extent=n_extent, dtype=dtype)

    def rank_at(mult: int):
        rows = rows_at(mult)
        if rows is None:
            return None
        reg = classify(rows, rows, capacity)
        return _ROWS_MONOTONE_ORDER.index(reg) if reg in _ROWS_MONOTONE_ORDER else None

    # The top of the search. `spills` has no upper edge, so the ceiling is a declared COST decision --
    # every point in that band is paid for on every grade -- and it is expressed as a multiple of the
    # target's own capacity so the same declaration means the same thing on a different store.
    ceiling_rows = float(spills_max_fraction) * float(capacity)
    mult_max, guard = 1, 0
    while True:
        rows = rows_at(mult_max)
        if rows is None:
            out["by_regime"] = {r: {"points": [], "unreachable":
                                    "the operand store cannot size this contraction's operands"}
                                for r in wanted}
            return out
        if rows >= ceiling_rows:
            break
        guard += 1
        if guard > 64:                                  # 2**64 tiles is not a shape; fail closed
            out["by_regime"] = {r: {"points": [], "unreachable":
                                    f"no tile multiple within the search reached "
                                    f"{spills_max_fraction}x capacity"} for r in wanted}
            return out
        mult_max *= 2

    # Rows must not DECREASE with K for the band search to mean anything. Checked rather than assumed:
    # a store whose row packing made residency non-monotone would silently hand back points from the
    # wrong band, and a wrong band is exactly the failure this module exists to make visible.
    lo_rows, hi_rows = rows_at(1), rows_at(mult_max)
    if lo_rows is None or hi_rows is None or hi_rows < lo_rows:
        out["by_regime"] = {r: {"points": [], "unreachable":
                                "operand residency is not monotone in the reduction depth on this "
                                "store, so a band cannot be bracketed"} for r in wanted}
        return out

    def first_mult_at_least(rank: int) -> "int | None":
        """Smallest tile multiple in ``[1, mult_max]`` whose regime is ``rank`` or harder."""
        top = rank_at(mult_max)
        if top is None or top < rank:
            return None
        lo, hi = 1, mult_max
        if (bottom := rank_at(1)) is not None and bottom >= rank:
            return 1
        while lo < hi:
            mid = (lo + hi) // 2
            got = rank_at(mid)
            if got is not None and got >= rank:
                hi = mid
            else:
                lo = mid + 1
        return lo

    for regime in wanted:
        if regime not in _ROWS_MONOTONE_ORDER:
            reason = (UNREACHABLE_ON_ALL_LIVE_INPUTS if regime == FITS_ON_REUSE
                      else f"{regime!r} is not a residency band a deep-K contraction can occupy")
            out["by_regime"][regime] = {"points": [], "unreachable": reason}
            continue
        rank = _ROWS_MONOTONE_ORDER.index(regime)
        band_lo = first_mult_at_least(rank)
        nxt = first_mult_at_least(rank + 1) if rank + 1 < len(_ROWS_MONOTONE_ORDER) else None
        band_hi = mult_max if nxt is None else nxt - 1
        if band_lo is None or band_hi < band_lo or (rank_at(band_lo) != rank):
            out["by_regime"][regime] = {"points": [], "unreachable":
                                        f"no tile multiple in [1, {mult_max}] (up to "
                                        f"{spills_max_fraction}x capacity) lands in {regime!r}"}
            continue
        span = band_hi - band_lo
        want = max(1, int(points_per_regime))
        if span == 0:
            mults = [band_lo]
        else:
            # Evenly spread across the band, endpoints included: clustering at an edge is the defect
            # this function exists to fix, and the widest separation is what a two-parameter fit wants.
            mults = sorted({band_lo + (span * i) // max(1, want - 1) for i in range(want)}) \
                if want > 1 else [band_lo + span // 2]
        points, dropped = [], []
        for mult in mults:
            rows = rows_at(mult)
            got = classify(rows, rows, capacity) if rows is not None else UNKNOWN
            if got != regime:                            # verified, not trusted
                dropped.append({"K": tile * mult, "classified_as": got})
                continue
            points.append({"K": tile * mult, "K_tiles": mult, "M": m_extent, "N": n_extent,
                           "rows": int(rows), "capacity_rows": int(capacity),
                           "fraction_of_capacity": round(rows / float(capacity), 6)})
        record = {"points": points, "unreachable": None,
                  "band_tile_multiples": [band_lo, band_hi]}
        if dropped:
            record["rejected_points"] = dropped
        if not points:
            record["unreachable"] = (f"every candidate in the {regime!r} band re-classified elsewhere "
                                     f"when checked: {dropped}")
        elif len(points) < want:
            record["short_of_requested"] = (
                f"{len(points)} of {want} requested points: the band spans tile multiples "
                f"[{band_lo}, {band_hi}], which offers no more distinct depths")
        out["by_regime"][regime] = record
    return out
