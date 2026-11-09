"""Resolve measured throughput inputs for the existing placement policy.

Missing measurements stay unpriced. This legacy cost accounts for the existing
router's kernel terms; it does not price transfers, packing, synchronization or
shared CPU/resource contention and must not stand in for a complete system cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def measured_cost_for(system) -> Callable[[Any, Any], float | None] | None:
    """Read contract-recorded throughput and target tile edges, or return ``None``.

    This preserves the existing factory: rates come from per-unit ``requant``
    metadata; missing rates contribute no price and absent rates leave placement
    in declaration order. A registry entry supplies no measured throughput by
    itself. This adapter does not independently qualify measurement provenance.
    """
    from merlin.targetgen import compute_units as _cu
    from merlin.targetgen import routing as _r
    from merlin.targetgen import target_registry as _tr

    rates: dict[str, float] = {}
    edges: dict[str, int] = {}
    overhead: dict[str, float] = {}
    for dev in getattr(system, "devices", ()) or ():
        try:
            units = _cu.compute_units(_tr.load_contract(dev.name))
        except Exception:  # noqa: BLE001
            continue
        for u in units:
            rate = _declared_rate(u)
            if rate:
                rates[u.name] = rate
            edge = _declared_tile_edge(dev.name)
            if edge:
                edges[u.name] = edge
    if not rates:
        return None  # nothing measured: say so rather than price with a default
    return _r.MeasuredCost(macs_per_cycle=rates, tile_edge=edges, tile_overhead_cycles=overhead)


def _declared_rate(unit) -> float | None:
    """A unit's measured MACs/cycle where its contract records one; never inferred from geometry.

    Deliberately not ``rows * cols``: peak lane count is what a unit could do at full occupancy, and
    costing with it credits every shape with work it will not do — which is precisely the error
    ``MeasuredCost`` documents itself as avoiding.
    """
    raw = getattr(unit, "requant", None)  # opaque per-unit dict; never interpreted structurally
    if isinstance(raw, dict):
        got = raw.get("macs_per_cycle")
        if isinstance(got, (int, float)) and got > 0:
            return float(got)
    return None


def _declared_tile_edge(device: str) -> int | None:
    """The device's tile edge from its own RTL facts, or None."""
    try:
        from merlin.llvmlower.device_shim import tile_edge_for

        return tile_edge_for(device)
    except Exception:  # noqa: BLE001
        return None
