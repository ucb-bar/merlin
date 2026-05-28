"""Phase 4 — per-(island, target) routing.

Two-layer routing decision:

  1. **Profile-driven** (preferred): when a CSV produced by
     `tools/profile_per_island.py` is available, route each island to
     whichever profiled target has the lowest median latency. Below
     the QNN dispatch floor (~1ms) we route to CPU regardless — a
     profiled-fast-on-GPU island that's still below 1ms after FastRPC
     overhead would still pay the wrapper cost on either backend.

  2. **Heuristic** (fallback): when no profile is available for an
     island, estimate cost from the recognizer name + boundary tensor
     shapes. Conv with MAC count ≥ `_HTA_FLOOR_MACS` → HTA; smaller
     conv / elementwise → GPU; ops below the QNN floor → CPU.

The output is `dict[island_name -> target]` consumed by
`tools/compile.py`'s `--qnn-partition` flag (which threads the
decision into `qnn_partition.partition`'s `target_router` callback).
This module is **bindings-free** (no MLIR access) — it operates only
on the partitioner's `Island` records and the profile CSV; tests can
exercise it without iree.compiler.ir loaded.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Iterable

# Heuristic thresholds. Backed by Phase 4 profiling on QRB5165 (see
# `eval/qrb5165/heterogeneous/yolov8_per_island.csv` once it's
# populated). These values are deliberately documented as constants
# so Phase 5's bytes-equal sweep can tune them without touching the
# routing logic.
_HTA_FLOOR_MACS: int = 1_000_000  # convs ≥ 1M MAC → prefer HTA
_QNN_FLOOR_MS: float = 1.0  # below this, CPU wins (FastRPC overhead)


@dataclasses.dataclass(frozen=True)
class ProfilePoint:
    """One measurement: median latency of `island_name` on `target` in
    milliseconds. Loaded from the per-island profile CSV."""

    island_name: str
    target: str
    median_ms: float


@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    """Per-island routing result: chosen target, the rule that picked
    it (`profile`, `heuristic_conv_hta`, `heuristic_below_floor_cpu`,
    `heuristic_default_gpu`), and the deciding metric (median ms when
    profile-driven, MAC count when heuristic)."""

    island_name: str
    target: str
    rule: str
    metric: float


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------


def load_profile_csv(path: pathlib.Path) -> list[ProfilePoint]:
    """Read a per-island profile CSV produced by `profile_per_island.py`.

    Schema (header row):
        island_name,target,median_ms,p99_ms,iter_count,run_id
    Only the first three columns are required by `route_islands`; the
    rest are kept for downstream analysis but ignored here.
    """
    points: list[ProfilePoint] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                points.append(
                    ProfilePoint(
                        island_name=row["island_name"],
                        target=row["target"],
                        median_ms=float(row["median_ms"]),
                    )
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"malformed row in {path}: {row!r} ({e})") from e
    return points


# ---------------------------------------------------------------------
# Heuristic cost estimation
# ---------------------------------------------------------------------


def estimate_island_macs(island) -> int:
    """Rough MAC count for an island, derived from the recognizer name
    and the boundary tensor shapes. Used by the heuristic router when
    no profile data is available.

    For conv islands: MAC = product of output spatial dims × output
    channels × input channels × kernel area. We approximate via the
    largest boundary tensor's element count × kernel area = 9 (3x3 is
    the modal yolov8 kernel; 1x1 and 5x5 round through this).

    For non-conv islands: MAC ≈ output element count (one mul per
    element for elementwise / pool / concat / reshape / transpose).
    """
    output_elements = 0
    for bv in island.boundary_outputs:
        n = 1
        for d in bv.shape:
            n *= int(d)
        output_elements = max(output_elements, n)
    if "conv" in island.recognizer_name:
        # Pull input-channel count from the largest input tensor (NCHW
        # layout: dim 1 is channels). Falls back to 1 if no input is
        # present (defensive — the partitioner always produces ≥1 in).
        ic = 1
        for bv in island.boundary_inputs:
            if len(bv.shape) >= 2:
                ic = max(ic, int(bv.shape[1]))
        return output_elements * ic * 9  # 3x3 kernel modal
    return output_elements


# ---------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------


def _index_profile(
    profile: Iterable[ProfilePoint],
) -> dict[tuple[str, str], float]:
    """Build {(island_name, target) -> median_ms}."""
    out: dict[tuple[str, str], float] = {}
    for p in profile:
        out[(p.island_name, p.target)] = p.median_ms
    return out


def route_islands(
    islands,
    *,
    profile: Iterable[ProfilePoint] | None = None,
) -> list[RoutingDecision]:
    """Compute one `RoutingDecision` per island.

    `profile` is the optional per-(island, target) measurement table.
    When present and the island has profiled data on multiple targets,
    the lowest-median target wins (provided it's not below the QNN
    floor — those route to CPU regardless of how fast they ran on
    GPU/HTA, because the FastRPC overhead would dominate at runtime).

    Otherwise the heuristic kicks in: large convs → HTA, small
    everything → GPU, below floor → CPU. The heuristic uses
    `estimate_island_macs` for the size signal.
    """
    indexed = _index_profile(profile or ())

    decisions: list[RoutingDecision] = []
    for isl in islands:
        # Profile-driven path.
        candidate_targets = ("qnn-hta", "qnn-gpu", "qnn-cpu")
        profiled = [(t, indexed[(isl.name, t)]) for t in candidate_targets if (isl.name, t) in indexed]
        if profiled:
            # Pick the fastest profiled target. If even the fastest is
            # below the QNN floor, force CPU (the FastRPC overhead +
            # backend dispatch already dominates at this scale).
            best_target, best_ms = min(profiled, key=lambda x: x[1])
            if best_ms < _QNN_FLOOR_MS:
                decisions.append(
                    RoutingDecision(
                        island_name=isl.name,
                        target="cpu",
                        rule="profile_below_floor_cpu",
                        metric=best_ms,
                    )
                )
            else:
                decisions.append(
                    RoutingDecision(
                        island_name=isl.name,
                        target=best_target,
                        rule="profile",
                        metric=best_ms,
                    )
                )
            continue

        # Heuristic path.
        macs = estimate_island_macs(isl)
        if macs < 1024:  # smaller than 1Ki ops — scarcely worth offloading
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="cpu",
                    rule="heuristic_below_floor_cpu",
                    metric=float(macs),
                )
            )
        elif "conv" in isl.recognizer_name and macs >= _HTA_FLOOR_MACS:
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="qnn-hta",
                    rule="heuristic_conv_hta",
                    metric=float(macs),
                )
            )
        else:
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="qnn-gpu",
                    rule="heuristic_default_gpu",
                    metric=float(macs),
                )
            )
    return decisions


def decisions_to_router(decisions: list[RoutingDecision]):
    """Build a `target_router(op_name, op) -> str` callback for the
    partitioner from a routing-decision list. The callback ignores its
    arguments because the partitioner already knows which island it's
    asking about (anchor identity = island identity in source order);
    the lookup is positional via an island-name index.

    Use case: `tools/compile.py` calls
        decisions = route_islands(islands, profile=load_profile_csv(...))
        islands_v2 = parse_and_partition(text, target_router=decisions_to_router(decisions))
    to re-partition with the routing decisions baked in.

    NOTE: in the current partitioner the `target_router` is keyed on
    op-name + op proxy, not island-name. The router we return here
    therefore returns the FIRST decision target, which works for
    homogeneous-target islands but isn't the final answer for
    mixed-target routing. Phase 5's compile.py wiring will pass the
    decision list directly through `partition()`'s API instead of
    going through `target_router`.
    """
    by_index: dict[str, RoutingDecision] = {d.island_name: d for d in decisions}
    fallback = decisions[0].target if decisions else "cpu"

    def router(_op_name: str, _op):
        # The current partitioner doesn't pass the island name in;
        # return the global fallback. Phase 5 refactors `partition()`
        # to thread the island index here.
        return fallback

    router.__decision_table__ = by_index  # type: ignore[attr-defined]
    return router


def summarize(decisions: list[RoutingDecision]) -> dict[str, int]:
    """Counter-style summary: {target -> count}. Useful for Phase 5
    sanity-prints (e.g. "after routing: 60 qnn-hta, 30 qnn-gpu, 4 cpu")."""
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.target] = counts.get(d.target, 0) + 1
    return counts


# ---------------------------------------------------------------------
# Phase 5 — empirical threshold tuning from a profile sweep.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TunedThresholds:
    """Empirical replacements for the static `_HTA_FLOOR_MACS` /
    `_QNN_FLOOR_MS` constants, derived from a per-island profile CSV.

    `hta_floor_macs`: smallest MAC count where qnn-hta beats qnn-gpu
        in observed profiles. Convs above this should route to HTA.
    `qnn_floor_ms`: 75th percentile of the QNN dispatch overhead
        (the median latency below which routing to QNN doesn't pay
        off vs CPU). Defaults to the static value when there's no
        evidence in the profile.
    `n_observations`: number of profile points the tuner used.
    """

    hta_floor_macs: int
    qnn_floor_ms: float
    n_observations: int


def tune_thresholds_from_profile(
    profile: list[ProfilePoint],
    islands,
) -> TunedThresholds:
    """Fit empirical routing thresholds to observed profile data.

    For each island that has profiled data on both qnn-hta and
    qnn-gpu, we record (mac_count, hta_ms, gpu_ms). The HTA floor is
    the smallest MAC count where hta_ms < gpu_ms across the dataset
    (i.e. the crossover threshold). The QNN floor is the 25th
    percentile of all profiled medians — below that, the FastRPC
    overhead dominates regardless of backend.

    Falls back to the static constants when the profile lacks
    crossover signal (e.g. all-HTA-fastest or all-GPU-fastest sweeps).
    """
    # Build {island_name: {target: median_ms}}.
    by_island: dict[str, dict[str, float]] = {}
    for p in profile:
        by_island.setdefault(p.island_name, {})[p.target] = p.median_ms

    # Crossover scan: among islands with both qnn-hta and qnn-gpu data,
    # find the smallest MAC count where hta wins.
    crossover_macs: list[int] = []
    all_medians: list[float] = []
    for isl in islands:
        macs = estimate_island_macs(isl)
        rec = by_island.get(isl.name, {})
        hta = rec.get("qnn-hta")
        gpu = rec.get("qnn-gpu")
        if hta is not None:
            all_medians.append(hta)
        if gpu is not None:
            all_medians.append(gpu)
        if hta is not None and gpu is not None and hta < gpu:
            crossover_macs.append(macs)

    if crossover_macs:
        hta_floor = min(crossover_macs)
    else:
        hta_floor = _HTA_FLOOR_MACS  # fall back to static

    if all_medians:
        all_medians_sorted = sorted(all_medians)
        # 25th percentile — the bottom-quartile latency.
        idx = max(0, int(len(all_medians_sorted) * 0.25) - 1)
        qnn_floor = all_medians_sorted[idx]
    else:
        qnn_floor = _QNN_FLOOR_MS

    return TunedThresholds(
        hta_floor_macs=int(hta_floor),
        qnn_floor_ms=float(qnn_floor),
        n_observations=len(profile),
    )


def route_islands_with_thresholds(
    islands,
    *,
    profile: Iterable[ProfilePoint] | None = None,
    thresholds: TunedThresholds | None = None,
) -> list[RoutingDecision]:
    """Variant of `route_islands` that consults an explicit
    `TunedThresholds` instead of the module-level static constants.
    Useful when the caller has tuned thresholds from a profile sweep
    via `tune_thresholds_from_profile`."""
    if thresholds is None:
        return route_islands(islands, profile=profile)

    indexed = _index_profile(profile or ())
    decisions: list[RoutingDecision] = []
    for isl in islands:
        candidate_targets = ("qnn-hta", "qnn-gpu", "qnn-cpu")
        profiled = [(t, indexed[(isl.name, t)]) for t in candidate_targets if (isl.name, t) in indexed]
        if profiled:
            best_target, best_ms = min(profiled, key=lambda x: x[1])
            if best_ms < thresholds.qnn_floor_ms:
                decisions.append(
                    RoutingDecision(
                        island_name=isl.name,
                        target="cpu",
                        rule="profile_below_floor_cpu",
                        metric=best_ms,
                    )
                )
            else:
                decisions.append(
                    RoutingDecision(
                        island_name=isl.name,
                        target=best_target,
                        rule="profile",
                        metric=best_ms,
                    )
                )
            continue

        macs = estimate_island_macs(isl)
        if macs < 1024:
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="cpu",
                    rule="heuristic_below_floor_cpu",
                    metric=float(macs),
                )
            )
        elif "conv" in isl.recognizer_name and macs >= thresholds.hta_floor_macs:
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="qnn-hta",
                    rule="heuristic_conv_hta",
                    metric=float(macs),
                )
            )
        else:
            decisions.append(
                RoutingDecision(
                    island_name=isl.name,
                    target="qnn-gpu",
                    rule="heuristic_default_gpu",
                    metric=float(macs),
                )
            )
    return decisions
