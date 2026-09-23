"""The machine a schedule is written against.

:mod:`merlin.sched.ir` says what a kernel does; this says what the hardware running it is made of —
units and the queues they consume, per-(instruction, unit) latency, banked on-chip stores, the parallel
iteration hierarchy, and whether hazards are resolved by the hardware or by the compiler.

Everything here is derived from a target's own data. Nothing here names a target.
"""

from .model import (
    COMPLETION_KINDS,
    DISCHARGES_DEPENDENCE,
    HAZARD_RESOLUTIONS,
    MOVER_KINDS,
    ROLE_UNKNOWN,
    SYNC_ORDERS,
    UNIT_KINDS,
    Hierarchy,
    Latency,
    Level,
    MachError,
    Machine,
    Memory,
    Sync,
    Unit,
    Unknown,
)

__all__ = [
    "COMPLETION_KINDS",
    "DISCHARGES_DEPENDENCE",
    "SYNC_ORDERS",
    "HAZARD_RESOLUTIONS",
    "MOVER_KINDS",
    "ROLE_UNKNOWN",
    "UNIT_KINDS",
    "Hierarchy",
    "Latency",
    "Level",
    "MachError",
    "Machine",
    "Memory",
    "Sync",
    "Unit",
    "Unknown",
]
