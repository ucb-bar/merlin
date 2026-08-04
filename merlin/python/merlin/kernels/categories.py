"""Improvement CATEGORIES — the "what should we improve?" question the beam search asks.

The beam does not reason over raw per-axis CCA divergences; it asks a higher-level question — "what
kind of thing should we improve next?" — and picks a CATEGORY to push. This module groups the CCA
facet axes into the semantic optimization buckets a search reasons over (the ones the design names:
tiling/dataflow, fusion/layout, register-residency, instruction-selection, runtime/sync), so a set of
divergences becomes a ranked "these categories are open" view.

Distinct from ``regions`` (WHERE in the compiler an axis lives) and ``action_catalog`` (the concrete
lever/seam for an axis): a category is WHAT KIND of optimization it is. Deterministic; no LLM.
"""
from __future__ import annotations

# The improvement categories the search chooses among. runtime-sync has no lever axis yet (runtime
# hooks aren't captured as a CCA facet) — kept for completeness so the search can still ask about it.
CATEGORIES = ("tiling-dataflow", "fusion-layout", "register-residency", "instruction-selection",
              "runtime-sync")

# CCA facet axis -> improvement category (every RVV lever axis is categorized; see check_categories).
_AXIS_CATEGORY = {
    "compute.register_block": "tiling-dataflow",
    "compute.nr_is_vsetvlmax": "tiling-dataflow",
    "compute.reduction_form": "tiling-dataflow",
    "compute.epilogue": "fusion-layout",
    "memory.access_pattern": "fusion-layout",   # packed unit-stride layout — the data-movement lever
    # The envelope axes are data-movement too: a redundant tile-epilogue copy IS layout traffic.
    "envelope.runtime_calls": "fusion-layout",
    "envelope.calls_in_loop": "fusion-layout",
    "compute.accumulator_resident": "register-residency",
    "compute.accumulator_dtype": "register-residency",
    # coverage (whole-model): "is this work even ON the vector path" is a tiling/dataflow question for
    # the contraction classes (the block decides whether a class is claimed at all) and an
    # instruction-selection question for the non-contraction tail (scalar loop vs vector instructions).
    "coverage.claimed_mac_fraction": "tiling-dataflow",
    "coverage.unclaimed_op_classes": "tiling-dataflow",
    "coverage.non_contraction_op_fraction": "instruction-selection",
    "compute.contraction_form": "instruction-selection",
    "compute.widening": "instruction-selection",
    "compute.activation_vectorization": "instruction-selection",
    "vector.sew": "instruction-selection",
    "vector.lmul": "instruction-selection",
    "vector.vl_strategy": "instruction-selection",
    "vector.tail": "instruction-selection",
}


def category_for_axis(axis: str) -> str | None:
    """The improvement category an axis belongs to (e.g. 'compute.accumulator_resident' ->
    'register-residency')."""
    return _AXIS_CATEGORY.get(axis)


def categorize(divergences) -> dict[str, list]:
    """Group divergences (anything with a ``.axis``) by improvement category. The result is the
    beam's 'what should we improve?' view: category -> the divergences in it. Uncategorized axes go
    under ``None`` so nothing is silently dropped."""
    out: dict[str, list] = {}
    for d in divergences:
        cat = category_for_axis(getattr(d, "axis", None))
        out.setdefault(cat, []).append(d)
    return out


def check_categories() -> list[str]:
    """Invariant (empty = OK): every RVV CCA LEVER axis has an improvement category, and every category
    in the map is a declared CATEGORY."""
    from . import cca_contract

    problems: list[str] = []
    for ax in sorted(cca_contract.leverable_axes("rvv")):
        if ax not in _AXIS_CATEGORY:
            problems.append(f"lever axis {ax}: no improvement category")
    for ax, cat in _AXIS_CATEGORY.items():
        if cat not in CATEGORIES:
            problems.append(f"axis {ax}: unknown category {cat!r}")
    return problems
