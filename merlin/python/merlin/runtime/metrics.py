"""Common runtime metrics (matches merlin/schemas/metrics.schema.yaml).

A plain accumulator the simulator updates as it executes a command buffer. ``as_dict``
emits the common metric vocabulary so results are comparable across targets/backends.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

COMMON_METRIC_NAMES = (
    "cycles", "bytes_moved", "bytes_read", "bytes_written", "command_count",
    "dispatch_count", "pack_count", "resident_hits", "resident_misses",
    "evictions", "accumulator_commits", "intermediate_write_bytes",
)


@dataclass
class Metrics:
    cycles: int = 0
    bytes_moved: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    command_count: int = 0
    dispatch_count: int = 0
    pack_count: int = 0
    resident_hits: int = 0
    resident_misses: int = 0
    evictions: int = 0
    accumulator_commits: int = 0
    intermediate_write_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
