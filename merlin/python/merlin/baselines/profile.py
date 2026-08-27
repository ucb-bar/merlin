"""Two-level profiling: whole-model E2E + per-region "kernel-style" breakdown.

Every baseline runner must emit BOTH levels. To keep parsing uniform across five very different
frameworks, each runner prints marker lines to stdout that this module parses:

    MERLIN_E2E ticks=<rdtime> wall_ns=<n>
    MERLIN_REGION name=<gemm|attention|norm|elementwise|other> ticks=<rdtime> [wall_ns=<n>] [calls=<n>]

``ticks`` are raw K1 ``rdtime`` counts (24 MHz platform timer). We convert to an *estimated* core
cycle count with the K1 CPU/timebase ratio — reported ``cycle_accurate=False`` (spike/FireSim remain
the cycle authorities), exactly as ``merlin.mining.k1`` does for our own runs.

A framework that also exposes an isolated kernel driver (EXO's natural granularity, or a per-op
micro-benchmark) can reuse the SAME markers so region numbers are comparable to whole-model brackets
and to the existing ``kernels/ceiling_drivers`` measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.baselines.contract import RegionProfile

# Reuse the exact K1 constants so cycle estimates match our own runs.
try:
    from merlin.mining.k1 import K1_CPU_HZ, K1_TIMEBASE_HZ
except Exception:  # pragma: no cover - defensive if k1 import chain changes
    K1_CPU_HZ, K1_TIMEBASE_HZ = 1_600_000_000, 24_000_000

from merlin.common.driver_output import kv_pairs as _kv


def ticks_to_cycles(ticks: int | None) -> int | None:
    """Estimate core cycles from rdtime ticks (NOT cycle-accurate; K1 rdtime is a 24 MHz timer)."""
    if ticks is None:
        return None
    return int(round(ticks * (K1_CPU_HZ / K1_TIMEBASE_HZ)))


@dataclass
class WholeModelProfile:
    rdtime_ticks: int | None = None
    cycles: int | None = None
    wall_ns: int | None = None


def parse_profile(stdout: str) -> tuple[WholeModelProfile, list[RegionProfile]]:
    """Parse MERLIN_E2E + MERLIN_REGION markers from a run's stdout.

    Returns (whole_model, regions). Missing markers yield None fields / an empty region list —
    the runner then records that as a gap rather than inventing numbers.
    """
    e2e = WholeModelProfile()
    regions: list[RegionProfile] = []
    for line in stdout.splitlines():
        if "MERLIN_E2E" in line:
            kv = _kv(line[line.index("MERLIN_E2E") + len("MERLIN_E2E"):])
            e2e.rdtime_ticks = int(kv["ticks"]) if "ticks" in kv else None
            e2e.wall_ns = int(kv["wall_ns"]) if "wall_ns" in kv else None
            e2e.cycles = ticks_to_cycles(e2e.rdtime_ticks)
            continue
        if "MERLIN_REGION" in line:
            kv = _kv(line[line.index("MERLIN_REGION") + len("MERLIN_REGION"):])
            name = kv.get("name", "other")
            ticks = int(kv["ticks"]) if "ticks" in kv else None
            regions.append(RegionProfile(
                name=name,
                rdtime_ticks=ticks,
                cycles=ticks_to_cycles(ticks),
                wall_ns=int(kv["wall_ns"]) if "wall_ns" in kv else None,
                calls=int(kv["calls"]) if "calls" in kv else None,
            ))
    return e2e, regions
