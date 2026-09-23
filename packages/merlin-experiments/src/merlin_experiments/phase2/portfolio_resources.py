"""Portfolio analysis budgets and memory-admitted deterministic scheduling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from . import contracts as P2_CONTRACTS
from .broker_evidence import _is_sha256

_PORTFOLIO_WORKER_HEADROOM_BYTES = 16 * 1024**3


class PortfolioMember(Protocol):
    """Frozen member fields consumed by allocation; no native controller dependency."""

    @property
    def frozen_source_path(self) -> str | Path: ...

    @property
    def capsule_sha256(self) -> str: ...

    @property
    def capsule(self) -> str: ...


def portfolio_member_analysis_allocation(
    remaining_seconds: float,
    remaining_sentinels: Sequence[PortfolioMember],
    *,
    emission_seconds_by_capsule_sha256: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Allocate one member's bounded analysis time from generic frozen-source complexity.

    Every remaining graph receives an equal chance floor capped at 60 seconds. Remaining time is
    weighted by exact prior baseline-emission measurements when available. Missing measurements are
    projected from frozen interface bytes and the median observed seconds/byte; an entirely cold
    cache uses interface bytes directly. Recomputing after actual elapsed time rolls surplus forward
    while preserving declared portfolio order and the outer iteration deadline.
    """
    if not remaining_sentinels:
        raise ValueError("portfolio allocation requires at least one remaining member")
    remaining_seconds = max(0.0, remaining_seconds)

    interface_sizes: list[int] = []
    for sentinel in remaining_sentinels:
        source = Path(sentinel.frozen_source_path)
        descriptor = P2_CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
        interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if interface.is_symlink() or not interface.is_file():
            raise ValueError(f"frozen portfolio member has no real interface MLIR: {sentinel.capsule}")
        interface_sizes.append(max(1, interface.stat().st_size))

    members = len(remaining_sentinels)
    measurements = dict(emission_seconds_by_capsule_sha256 or {})
    for digest, seconds in measurements.items():
        if (
            not _is_sha256(digest)
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("portfolio emission-cost measurement is malformed")
    known_rates = sorted(
        measurements[sentinel.capsule_sha256] / size
        for sentinel, size in zip(remaining_sentinels, interface_sizes, strict=True)
        if sentinel.capsule_sha256 in measurements
    )
    median_rate = (
        None
        if not known_rates
        else known_rates[len(known_rates) // 2]
        if len(known_rates) % 2
        else 0.5 * (known_rates[len(known_rates) // 2 - 1] + known_rates[len(known_rates) // 2])
    )
    estimates = [
        measurements.get(sentinel.capsule_sha256, size * median_rate if median_rate is not None else float(size))
        for sentinel, size in zip(remaining_sentinels, interface_sizes, strict=True)
    ]
    weights = [max(float(value), 1e-9) for value in estimates]
    chance_floor = remaining_seconds if members == 1 else min(60.0, remaining_seconds / (2 * members))
    weighted_budget = max(0.0, remaining_seconds - chance_floor * members)
    allocated = remaining_seconds if members == 1 else chance_floor + weighted_budget * weights[0] / sum(weights)
    first_digest = remaining_sentinels[0].capsule_sha256
    return {
        "schema": "portfolio_analysis_allocation_v1",
        "policy": "bounded_equal_chance_floor_plus_measured_emission_cost_with_rolling_surplus",
        "allocated_seconds": allocated,
        "remaining_seconds": remaining_seconds,
        "remaining_members": members,
        "interface_bytes": interface_sizes[0],
        "remaining_interface_bytes": sum(interface_sizes),
        "chance_floor_seconds": chance_floor,
        "estimated_emission_seconds": estimates[0],
        "emission_cost_basis": (
            "exact_cached_baseline_emission"
            if first_digest in measurements
            else "interface_bytes_scaled_by_measured_median"
            if median_rate is not None
            else "frozen_interface_bytes_proxy"
        ),
        "known_emission_measurements": len(known_rates),
    }


def portfolio_analysis_concurrency(
    *, requested_workers: int, members: int, memory_available_bytes: int, minimum_memory_available_bytes: int
) -> dict[str, Any]:
    """Admit bounded member parallelism from current host headroom above the launch guard."""
    if min(requested_workers, members) < 1 or min(memory_available_bytes, minimum_memory_available_bytes) < 0:
        raise ValueError("portfolio concurrency inputs are invalid")
    headroom = memory_available_bytes - minimum_memory_available_bytes
    if headroom < 0:
        raise TimeoutError("host memory is below the portfolio analysis admission floor")
    memory_workers = max(1, headroom // _PORTFOLIO_WORKER_HEADROOM_BYTES)
    admitted = min(requested_workers, members, memory_workers)
    return {
        "schema": "portfolio_analysis_concurrency_v1",
        "requested_workers": requested_workers,
        "admitted_workers": admitted,
        "members": members,
        "memory_available_bytes": memory_available_bytes,
        "minimum_memory_available_bytes": minimum_memory_available_bytes,
        "per_worker_headroom_bytes": _PORTFOLIO_WORKER_HEADROOM_BYTES,
        "policy": "host_memory_headroom_bounded_concurrent_member_analysis",
    }


def portfolio_concurrent_schedule(cost_seconds: Sequence[float], workers: int) -> dict[str, Any]:
    """Longest-estimated members first; retain declared order as the tie breaker and output order."""
    if workers < 1 or not cost_seconds:
        raise ValueError("portfolio concurrent schedule requires workers and member costs")
    costs = [float(value) for value in cost_seconds]
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("portfolio concurrent schedule cost is malformed")
    admitted = min(workers, len(costs))
    order = sorted(range(len(costs)), key=lambda index: (-costs[index], index))
    loads = [0.0] * admitted
    assignments = []
    for index in order:
        worker = min(range(admitted), key=lambda item: (loads[item], item))
        assignments.append({"member_index": index, "worker": worker, "estimated_seconds": costs[index]})
        loads[worker] += costs[index]
    return {
        "schema": "portfolio_concurrent_schedule_v1",
        "submission_order": order,
        "workers": admitted,
        "worker_estimated_seconds": loads,
        "projected_wall_seconds": max(loads),
        "policy": "longest_estimated_member_first",
    }
