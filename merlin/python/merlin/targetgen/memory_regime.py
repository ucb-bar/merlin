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


def operand_store(target: str):
    """``(Store, capacity_rows)`` for the target's operand store, or ``(None, None)``.

    Picks the NARROWEST-row store, because that is the one operands live in: a separate accumulator
    space exists precisely because its row is wider (it holds the accumulate type). Choosing by name
    would bake a target's spelling into shared code.
    """
    try:
        from merlin.targetgen import address_space as AS
        space = AS.derive_address_space(target)
    except Exception:                                          # noqa: BLE001 — unresolvable target
        return None, None
    stores = [s for s in (getattr(space, "stores", ()) or ()) if getattr(s, "row_bytes", None)]
    if not stores:
        return None, None
    store = min(stores, key=lambda s: int(s.row_bytes))
    return store, getattr(store, "total_rows", None)


def store_gap(target: str) -> str:
    """WHY ``target`` yields no operand store -- in the terms of what the extractor actually found.

    "declares no operand-store capacity we can derive" is true but flattens three different situations
    into one sentence, and they call for different work:

      * no facts at all             -> run the extractor;
      * facts with no memories      -> the memory extraction does not reach this target;
      * memories found, no row width -> the banks ARE discovered and only their CLASSIFICATION failed.

    Measured 2026-09-01: one target discovers 39 SRAM banks (a 32-bank, 64-deep, 32-byte-row register
    file among them) and still yields no operand store, because nothing identifies WHICH bank the compute
    unit reads its operands from. Reporting that as "declares no capacity" sends the reader to the
    extractor, which is not where the gap is.
    """
    store, capacity = operand_store(target)
    if store is not None and capacity:
        return ""                                              # there is no gap to explain
    try:
        from merlin.targetgen import address_space as AS
        space = AS.derive_address_space(target)
    except Exception as e:                                     # noqa: BLE001 -- unresolvable target
        return f"the address space could not be derived at all ({type(e).__name__}: {e})"
    stores = tuple(getattr(space, "stores", ()) or ())
    reasons = [u.reason for u in (getattr(space, "unknowns", ()) or ())
               if getattr(u, "quantity", None) in ("stores", "row_bytes", "row_elems", "element_bits")]
    if not stores:
        return ("; ".join(reasons) if reasons else
                "the facts carry no on-chip memory this reader could turn into a store")
    named = ", ".join(f"{st.name} ({st.nbytes} bytes)" for st in stores)
    if not any(getattr(st, "row_bytes", None) for st in stores):
        return (f"{len(stores)} on-chip store(s) ARE derived ({named}) but none has a ROW width, so a "
                f"byte capacity says nothing about how many rows are addressable"
                + (f" -- {'; '.join(reasons)}" if reasons else ""))
    return f"stores derived ({named}) but none was selectable as the operand store"


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

    if store is None:
        store, capacity = operand_store(target)
    cy = Path(capsule_dir) / "capsule.yaml"
    if not cy.is_file():
        return {"regime": UNKNOWN, "why": "no capsule.yaml"}
    try:
        doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return {"regime": UNKNOWN, "why": f"unparseable capsule.yaml: {type(e).__name__}"}
    if store is None or not capacity:
        return {"regime": UNKNOWN,
                "why": f"{target!r} yields no operand store: {store_gap(target)}"}

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
    store, capacity = operand_store(target)
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
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
            got = capsule_regime(cy.parent, target, store=store, capacity=capacity)
            by.setdefault(got["regime"], []).append(name)
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

    store, capacity = operand_store(target)
    by: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    unreadable: dict[str, str] = {}
    if store is None or not capacity:
        return {"by_regime": {}, "region_counts": {}, "captures_unreadable": {},
                "why": f"{target!r} yields no operand store, so no regime is required of the corpus -- "
                       f"that is 'we do not know', never 'nothing is required'. "
                       f"{store_gap(target)}"}
    for label, path in sorted((captures or {}).items()):
        try:
            module = mc.load_module(path)
        except Exception as e:                                 # noqa: BLE001
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        for _op, shapes, dtype in _operand_shapes(module):
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
            "capacity_rows": int(capacity),
            "captures_unreadable": unreadable}


#: The axis could not be derived at all -- distinct from "derived, and nothing is required".
UNDETERMINABLE_AXIS = "undeterminable"


def uncovered_regimes(required: dict, corpus: dict) -> dict:
    """Regimes real models present that no capsule in the corpus reaches."""
    want = [r for r in (required or {}).get("by_regime") or {} if r != UNKNOWN]
    have = {r for r in (corpus or {}).get("by_regime") or {} if r != UNKNOWN}
    missing = [r for r in ORDER if r in set(want) - have]
    # AN AXIS THAT REQUIRES NOTHING BECAUSE WE COULD NOT DERIVE IT IS NOT A SATISFIED AXIS.
    # `required_regimes` already returns the honest `why` when it cannot resolve the operand store --
    # "that is 'we do not know', never 'nothing is required'" -- but that string was dropped here, so a
    # target with no derivable capacity reported `0 / 0 required regime(s)`, which reads exactly like a
    # closed axis. Measured 2026-09-01: mx_gemmini and radiance both printed 0/0 while their capacity
    # was absent, and a capsule authored against that axis would have proved nothing.
    why = (required or {}).get("why")
    if not want and why:
        return {
            "status": UNDETERMINABLE_AXIS,
            "n_required": None, "n_covered": None, "uncovered": [],
            "corpus_regimes": sorted(have),
            "capacity_rows": corpus.get("capacity_rows"),
            "largest_working_set": corpus.get("largest_working_set"),
            "why": why,
            "note": "the memory-regime axis could not be derived, so it requires nothing YET -- this is "
                    "UNDETERMINABLE, not covered. Deriving the operand store is a prerequisite for "
                    "authoring any capsule against this axis.",
        }
    return {
        "status": "ok",
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
