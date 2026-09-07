"""Target-neutral host memory admission for bounded compiler experiments.

This module prices no target and knows nothing about a workload.  It only turns
Linux's measured host-memory state into an explicit, caller-selected admission
decision.  Experiment drivers choose the limits; keeping that policy outside
the module avoids baking one machine's capacity into reusable compiler code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Mapping


_REQUIRED_FIELDS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")


@dataclass(frozen=True)
class HostMemorySample:
    observed_at_unix_s: float
    memory_total_bytes: int
    memory_available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    @property
    def swap_used_bytes(self) -> int:
        return self.swap_total_bytes - self.swap_free_bytes

    def record(self) -> dict[str, int | float]:
        return {
            "observed_at_unix_s": self.observed_at_unix_s,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "swap_used_bytes": self.swap_used_bytes,
        }


def parse_proc_meminfo(text: str, *, observed_at_unix_s: float | None = None) -> HostMemorySample:
    """Parse the four kernel counters needed by the admission policy.

    ``/proc/meminfo`` reports these values in KiB.  Unknown units and missing or
    internally inconsistent counters are refused rather than guessed.
    """
    fields: dict[str, int] = {}
    for raw in text.splitlines():
        name, separator, remainder = raw.partition(":")
        if not separator or name not in _REQUIRED_FIELDS:
            continue
        parts = remainder.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ValueError(f"unsupported /proc/meminfo field: {name}")
        try:
            kib = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"non-integer /proc/meminfo field: {name}") from exc
        if kib < 0:
            raise ValueError(f"negative /proc/meminfo field: {name}")
        fields[name] = kib * 1024
    missing = tuple(name for name in _REQUIRED_FIELDS if name not in fields)
    if missing:
        raise ValueError(f"missing /proc/meminfo fields: {', '.join(missing)}")
    if (fields["MemAvailable"] > fields["MemTotal"]
            or fields["SwapFree"] > fields["SwapTotal"]):
        raise ValueError("inconsistent /proc/meminfo counters")
    return HostMemorySample(
        observed_at_unix_s=time() if observed_at_unix_s is None else observed_at_unix_s,
        memory_total_bytes=fields["MemTotal"],
        memory_available_bytes=fields["MemAvailable"],
        swap_total_bytes=fields["SwapTotal"],
        swap_free_bytes=fields["SwapFree"],
    )


def sample_host_memory(path: Path = Path("/proc/meminfo")) -> HostMemorySample:
    return parse_proc_meminfo(path.read_text())


@dataclass(frozen=True)
class HostResourcePolicy:
    minimum_memory_available_bytes: int
    maximum_swap_used_bytes: int
    consecutive_violations_to_stop: int = 2

    def __post_init__(self) -> None:
        values = (self.minimum_memory_available_bytes, self.maximum_swap_used_bytes,
                  self.consecutive_violations_to_stop)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("host resource limits must be integers")
        if min(self.minimum_memory_available_bytes, self.maximum_swap_used_bytes) < 0:
            raise ValueError("host resource byte limits must be nonnegative")
        if self.consecutive_violations_to_stop < 1:
            raise ValueError("consecutive violation limit must be positive")

    def record(self) -> dict[str, int]:
        return {
            "minimum_memory_available_bytes": self.minimum_memory_available_bytes,
            "maximum_swap_used_bytes": self.maximum_swap_used_bytes,
            "consecutive_violations_to_stop": self.consecutive_violations_to_stop,
        }


def violations(sample: HostMemorySample, policy: HostResourcePolicy) -> tuple[str, ...]:
    reasons = []
    if sample.memory_available_bytes < policy.minimum_memory_available_bytes:
        reasons.append("memory_available_below_limit")
    if sample.swap_used_bytes > policy.maximum_swap_used_bytes:
        reasons.append("swap_used_above_limit")
    return tuple(reasons)


class HostResourceTripwire:
    """Require consecutive bad samples before stopping a supervised workload."""

    def __init__(self, policy: HostResourcePolicy):
        self.policy = policy
        self.consecutive_violations = 0

    def observe(self, sample: HostMemorySample) -> Mapping[str, object]:
        reasons = violations(sample, self.policy)
        self.consecutive_violations = self.consecutive_violations + 1 if reasons else 0
        return {
            "status": ("stop" if reasons and self.consecutive_violations
                       >= self.policy.consecutive_violations_to_stop else "continue"),
            "reasons": list(reasons),
            "consecutive_violations": self.consecutive_violations,
            "sample": sample.record(),
            "policy": self.policy.record(),
        }


def summarize_samples(samples: list[HostMemorySample]) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0}
    return {
        "sample_count": len(samples),
        "first": samples[0].record(),
        "last": samples[-1].record(),
        "minimum_memory_available_bytes": min(row.memory_available_bytes for row in samples),
        "maximum_swap_used_bytes": max(row.swap_used_bytes for row in samples),
    }
