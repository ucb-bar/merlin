"""Two-level profiling: whole-model E2E + per-region "kernel-style" breakdown.

Every baseline runner must emit BOTH levels. To keep parsing uniform across five very different
frameworks, each runner prints marker lines to stdout that this module parses:

    MERLIN_E2E ticks=<rdtime> wall_ns=<n>
    MERLIN_REGION name=<gemm|attention|norm|elementwise|other> ticks=<rdtime> [wall_ns=<n>] [calls=<n>]

``ticks`` are raw timer counts. A runner may supply its measurement clock to obtain an
*estimated* core cycle count; without one, cycles remain unknown. Neither estimate is
cycle-accurate, and this parser does not assume a particular board's clock.

A framework that also exposes an isolated kernel driver (EXO's natural granularity, or a per-op
micro-benchmark) can reuse the SAME markers so region numbers are comparable to whole-model brackets
and to the existing ``kernels/ceiling_drivers`` measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.baselines.contract import RegionProfile
from merlin.common.driver_output import kv_pairs as _kv


@dataclass(frozen=True)
class MeasurementClock:
    """Runner-supplied timer and estimated core rates, in Hz (not cycle-accurate)."""

    timebase_hz: int
    estimated_core_hz: int

    def __post_init__(self) -> None:
        if type(self.timebase_hz) is not int or self.timebase_hz <= 0:
            raise ValueError("timebase_hz must be a positive integer")
        if type(self.estimated_core_hz) is not int or self.estimated_core_hz <= 0:
            raise ValueError("estimated_core_hz must be a positive integer")


def ticks_to_cycles(ticks: int | None, *, clock: MeasurementClock | None) -> int | None:
    """Estimate core cycles from timer ticks only when the runner supplies its clock."""
    if ticks is None or clock is None:
        return None
    return int(round(ticks * (clock.estimated_core_hz / clock.timebase_hz)))


@dataclass
class WholeModelProfile:
    rdtime_ticks: int | None = None
    cycles: int | None = None
    wall_ns: int | None = None


def parse_profile(stdout: str, *, clock: MeasurementClock | None) -> tuple[WholeModelProfile, list[RegionProfile]]:
    """Parse MERLIN_E2E + MERLIN_REGION markers from a run's stdout.

    Returns (whole_model, regions). Missing markers yield None fields / an empty region list —
    the runner then records that as a gap rather than inventing numbers. ``clock=None``
    preserves raw ticks and wall time while leaving estimated cycles unknown.
    """
    e2e = WholeModelProfile()
    regions: list[RegionProfile] = []
    for line in stdout.splitlines():
        if "MERLIN_E2E" in line:
            kv = _kv(line[line.index("MERLIN_E2E") + len("MERLIN_E2E") :])
            e2e.rdtime_ticks = int(kv["ticks"]) if "ticks" in kv else None
            e2e.wall_ns = int(kv["wall_ns"]) if "wall_ns" in kv else None
            e2e.cycles = ticks_to_cycles(e2e.rdtime_ticks, clock=clock)
            continue
        if "MERLIN_REGION" in line:
            kv = _kv(line[line.index("MERLIN_REGION") + len("MERLIN_REGION") :])
            name = kv.get("name", "other")
            ticks = int(kv["ticks"]) if "ticks" in kv else None
            regions.append(
                RegionProfile(
                    name=name,
                    rdtime_ticks=ticks,
                    cycles=ticks_to_cycles(ticks, clock=clock),
                    wall_ns=int(kv["wall_ns"]) if "wall_ns" in kv else None,
                    calls=int(kv["calls"]) if "calls" in kv else None,
                )
            )
    return e2e, regions
