"""Target-neutral, explicitly selected resource composition shared by screens."""

from __future__ import annotations

from collections.abc import Mapping

from merlin.analysis.perf.decompose import ResourceKind
from merlin.analysis.perf.envelope import Basis, ResourceTime, compose
from merlin.analysis.perf.headroom import Composition
from merlin.compiler.dialects.lowering.global_plan import CycleInterval


def compose_resource_times(
    resources: Mapping[str, tuple[ResourceKind, float]],
    *,
    operator: Composition,
    eta: float,
    provenance: str,
) -> float:
    times = tuple(
        ResourceTime(name, kind, value, "cycles", Basis.MOVED, evidence_kind="calibration_fit", provenance=provenance)
        for name, (kind, value) in sorted(resources.items())
    )
    result = compose(times, operator=operator, eta=eta)
    if not result.known:
        raise ValueError("resource composition remained unresolved")
    return float(result.cycles)


def compose_resource_intervals(
    resources: Mapping[str, tuple[ResourceKind, CycleInterval]],
    *,
    operator: Composition,
    eta: float,
    provenance: str,
) -> CycleInterval:
    missing = tuple(reason for _, interval in resources.values() for reason in interval.missing)
    if missing:
        return CycleInterval.unknown(*missing)
    endpoints = []
    for endpoint in ("lo", "hi"):
        endpoints.append(
            compose_resource_times(
                {name: (kind, float(getattr(interval, endpoint))) for name, (kind, interval) in resources.items()},
                operator=operator,
                eta=eta,
                provenance=provenance,
            )
        )
    return CycleInterval(*endpoints, provenance=(provenance,))
