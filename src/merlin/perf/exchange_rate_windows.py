"""A window plan that can be checked for isolation before anything is measured.

WHY THIS EXISTS. An exchange rate is a slope, and a slope is only a rate if exactly one thing moved
between the points. That property is easy to state, easy to believe, and invisible in a plan written
as prose -- the sketch this replaces read as three isolated arms and one of them was not isolated,
one was aimed at a model that prices the intended change at exactly zero, and the third varied two
terms at once. None of that is visible until the numbers come back wrong, by which point a campaign
has been spent.

So the plan is DATA and the isolation is DERIVED FROM IT. :func:`check_plan` recomputes each window's
model terms from its declared tiling and asserts, per arm, that the arm's own term moves and every
term it claims to pin is constant. A later edit that breaks the isolation fails here rather than
silently measuring two things at once.

WHAT IS CHECKED, AND IN WHAT UNITS. Everything is computed in EDGE MULTIPLES of the array edge ``E``,
which is derived from the target's own facts and never appears here. Every quantity below is either
exactly E-independent or carries a single factor of E that is common to all windows in an arm, so an
isolation proved here holds at every array size. The model is
:func:`merlin.perf.mesh_occupancy.tile_issue_cycles` summed over the chunk grid:

    issue = (sum_i m_i) * ceil(K/E) * (sum_j ceil(n_j/E))

which is where the asymmetry lives: M chunking cannot change the issue count at all, while an N chunk
narrower than the array edge costs a whole array width regardless.

WHAT THIS MODULE REFUSES. A K split, because it reassociates the accumulation and the per-group
checksums that establish bit-identity stop being comparable. A chunk count that is not exact, because
a ragged edge adds partial-block waste the arm did not intend and cannot separate. An arm that leaves
a term neither varied, nor pinned, nor already fit by an earlier arm -- a free term is an unidentified
model, and a fit against one is a number rather than a rate.

Target-neutral: no target, device, array size or rate is named here. The plan supplies all of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import ceil
from typing import Any

__all__ = [
    "WindowPlanError",
    "ALL_TERMS",
    "load_plan",
    "window_terms",
    "arm_terms",
    "check_plan",
]

_PLAN = ("contract", "phase2_exchange_rate_windows.yaml")

#: Every term the plan's arms are allowed to name. ``useful_macs`` is not a cost term -- it is the
#: bit-identity guard, and it must be constant in EVERY arm including the one that varies nothing.
ALL_TERMS = ("host_dynamic_operations_total", "dispatches", "mesh_issue_cycles", "useful_macs")

#: Terms an arm may name as varied or pinned. ``host_config_emissions`` is the structural proxy the
#: host-operation arm actually moves; the measured term it stands for is priced from it.
_STRUCTURAL = ("dispatches", "mesh_issue_cycles", "useful_macs", "host_config_emissions")


class WindowPlanError(ValueError):
    """The plan is malformed, or an arm does not isolate the term it claims to."""


def load_plan(*, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The declared plan. ``body`` injects one instead of reading the tracked file."""
    if body is None:
        import yaml

        from ..common.paths import merlin_dir

        path = merlin_dir().joinpath(*_PLAN)
        if not path.is_file():
            raise WindowPlanError(f"no window plan at {path}; a sweep design is declared, never assumed")
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(body, Mapping):
        raise WindowPlanError("a window plan must be a mapping")
    if body.get("schema") != "exchange_rate_window_plan_v1":
        raise WindowPlanError(f"plan schema is {body.get('schema')!r}, not exchange_rate_window_plan_v1")
    if not isinstance(body.get("arms"), Sequence) or not body["arms"]:
        raise WindowPlanError("a window plan must declare at least one arm")
    return dict(body)


def _exact_chunks(extent_edges: int, chunk_edges: Fraction, *, what: str, window: str) -> int:
    """How many chunks of ``chunk_edges`` tile ``extent_edges``, refusing a ragged edge.

    A ragged last chunk is not merely untidy: it adds partial-block waste that belongs to no arm, so
    the window would move a term it does not declare and the isolation proof would be false while
    still passing an equality test on the intended term.
    """
    if chunk_edges <= 0:
        raise WindowPlanError(f"window {window}: {what} chunk size must be positive, got {chunk_edges}")
    count = Fraction(extent_edges) / chunk_edges
    if count.denominator != 1:
        raise WindowPlanError(
            f"window {window}: {what} extent {extent_edges}E does not divide into {chunk_edges}E chunks "
            f"({count}); a ragged edge adds partial-block waste this arm does not declare"
        )
    return int(count)


def window_terms(window: Mapping[str, Any], shape: Mapping[str, Any]) -> dict[str, Fraction]:
    """The model terms one window produces, in edge multiples.

    ``shape`` gives ``a``, ``b``, ``c`` with ``M = a*E``, ``N = b*E``, ``K = c*E``.
    """
    name = str(window.get("id", "<unnamed>"))
    for key in ("m_edges", "n_edges", "k_chunks"):
        if key not in window:
            raise WindowPlanError(f"window {name} declares no {key}")
    if int(window["k_chunks"]) != 1:
        raise WindowPlanError(
            f"window {name} splits K into {window['k_chunks']} chunks. K IS NEVER CUT: splitting the "
            "reduction reassociates the accumulation, so the per-group checksums that establish "
            "bit-identity between windows stop being comparable."
        )
    a, b, c = (int(shape[k]) for k in ("a", "b", "c"))
    m_edges = Fraction(str(window["m_edges"]))
    n_edges = Fraction(str(window["n_edges"]))
    if m_edges < 1 or m_edges.denominator != 1:
        # An M chunk narrower than the array edge is not wrong, but it is not priced by this model --
        # rows are charged linearly -- so a plan that used one would be measuring nothing here.
        raise WindowPlanError(
            f"window {name}: m_edges must be a whole number of array edges (got {m_edges}); the issue "
            "model charges rows linearly, so a sub-edge M chunk moves no modelled term"
        )
    a_chunks = _exact_chunks(a, m_edges, what="M", window=name)
    b_chunks = _exact_chunks(b, n_edges, what="N", window=name)
    # Each N chunk occupies ceil(n/E) whole array widths, whatever fraction of one it fills.
    widths_per_chunk = ceil(n_edges)
    return {
        # A*B groups.
        "dispatches": Fraction(a_chunks * b_chunks),
        # M * ceil(K/E) * sum_j ceil(n_j/E), in units of E.
        "mesh_issue_cycles": Fraction(a * c * b_chunks * widths_per_chunk),
        # a*b*c hardware tiles, invariant under regrouping: the emitter always steps by the edge.
        "useful_macs": Fraction(a * b * c),
        # The host-operation proxy the loop-body arm moves.
        "host_config_emissions": Fraction(a * b * c) * Fraction(str(window.get("config_per_tile", 0))),
    }


def arm_terms(arm: Mapping[str, Any], default_shape: Mapping[str, Any]) -> dict[str, dict[str, Fraction]]:
    """Every window in one arm, priced. An arm may override the program shape (the closure arm does)."""
    shape = arm.get("shape") or default_shape
    windows = arm.get("windows")
    if not isinstance(windows, Sequence) or not windows:
        raise WindowPlanError(f"arm {arm.get('id')!r} declares no windows")
    out: dict[str, dict[str, Fraction]] = {}
    for window in windows:
        name = str(window.get("id", "<unnamed>"))
        if name in out:
            raise WindowPlanError(f"arm {arm.get('id')!r} repeats window id {name!r}")
        out[name] = window_terms(window, shape)
    return out


def check_plan(*, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Prove every arm isolates the term it claims, or say exactly which one does not.

    Returns a report rather than a bare assertion so a caller can show the computed terms beside the
    verdict -- the numbers are the argument, and an isolation nobody can see is one nobody checks.
    """
    plan = load_plan(body=body)
    default_shape = plan.get("base_shape") or {}
    order = list(plan.get("fit_order") or [])
    arms = list(plan["arms"])
    ids = [str(arm.get("id")) for arm in arms]
    if sorted(order) != sorted(ids):
        raise WindowPlanError(f"fit_order {order} does not name exactly the declared arms {ids}")

    by_id = {str(arm.get("id")): arm for arm in arms}
    rows: list[dict[str, Any]] = []
    fitted: set[str] = set()
    saw_closure = False

    for arm_id in order:
        arm = by_id[arm_id]
        terms = arm_terms(arm, default_shape)
        varies = str(arm.get("varies", "none"))
        pinned = list(arm.get("pinned") or [])
        previously = list(arm.get("previously_fit") or [])
        is_closure = str(arm.get("role", "")) == "closure"
        saw_closure = saw_closure or is_closure

        for term in [*pinned, *([varies] if varies != "none" else [])]:
            if term not in _STRUCTURAL:
                raise WindowPlanError(f"arm {arm_id}: {term!r} is not a structural term this plan can compute")

        # A rate claimed from a term an earlier arm has not fitted is a rate claimed from nothing.
        for term in previously:
            if term not in fitted and term != "host_dynamic_operations_total":
                raise WindowPlanError(
                    f"arm {arm_id} carries {term!r} as previously fit, but no earlier arm in "
                    f"fit_order {order} fitted it"
                )

        values = {term: {name: cols[term] for name, cols in terms.items()} for term in _STRUCTURAL}
        moved = {term for term, per in values.items() if len(set(per.values())) > 1}

        # The bit-identity guard, in every arm including the closure one.
        if len(set(values["useful_macs"].values())) > 1:
            raise WindowPlanError(
                f"arm {arm_id} changes useful_macs across its windows "
                f"({sorted(set(values['useful_macs'].values()))}); the compared output cannot be "
                "bit-identical, so the windows are not comparable at all"
            )

        if is_closure:
            if len(terms) != 1:
                raise WindowPlanError(f"closure arm {arm_id} must declare exactly one window, got {len(terms)}")
        else:
            if varies == "none":
                raise WindowPlanError(f"arm {arm_id} is not the closure arm and varies nothing")
            if varies not in moved:
                raise WindowPlanError(
                    f"arm {arm_id} claims to vary {varies!r} but it is CONSTANT at "
                    f"{next(iter(set(values[varies].values())))} across every window. The arm would "
                    "measure a slope through points that differ in nothing."
                )
            if len(terms) < 3:
                raise WindowPlanError(
                    f"arm {arm_id} has {len(terms)} windows. A slope through two points is not a "
                    "rate: it cannot separate a per-unit cost from a fixed overhead."
                )
            # A term that moves and is not the arm's own is fatal whether or not the arm PINNED it.
            # Naming it in `pinned` is a claim about it, and an unnamed stray is worse, not better:
            # the arm did not even know it was there.
            stray = sorted(moved - {varies})
            if stray:
                broken = [term for term in stray if term in pinned]
                unnamed = [term for term in stray if term not in pinned]
                detail = ", ".join(
                    part
                    for part in (
                        f"pinned but moving: {broken}" if broken else "",
                        f"moving and not even named: {unnamed}" if unnamed else "",
                    )
                    if part
                )
                raise WindowPlanError(
                    f"arm {arm_id} claims to isolate {varies!r} but other terms move across its "
                    f"windows ({detail}). Two terms moving at once is a fitted number, not an "
                    "exchange rate."
                )
            fitted.add(str(arm.get("term", varies)))

        rows.append(
            {
                "arm": arm_id,
                "role": "closure" if is_closure else "fit",
                "varies": varies,
                "windows": {name: {term: str(value) for term, value in cols.items()} for name, cols in terms.items()},
                "moved": sorted(moved),
            }
        )

    if not saw_closure:
        raise WindowPlanError(
            "the plan declares no closure arm. Three slopes without a program in which all three "
            "terms are known are three unrelated fits, and `exchange_rates` refuses them under the "
            "ledger's declared residual bound -- correctly."
        )
    batch = plan.get("batch") or {}
    if not str(batch.get("order_control") or "").strip():
        raise WindowPlanError(
            "the plan declares no order control. A batched number that is not reproducible within "
            "its own batch is not comparable to a solo one."
        )
    return {
        "schema": "exchange_rate_window_plan_check_v1",
        "status": "consistent",
        "design": plan.get("design"),
        "fit_order": order,
        "arms": rows,
        "reading": (
            "each fit arm moves exactly one structural term and holds the rest constant; the closure "
            "arm moves none and exists to price a program where all three are known. useful_macs is "
            "constant in every arm, which is the bit-identity precondition for comparing them."
        ),
    }
