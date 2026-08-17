"""Failing closed, with the boundary named.

A kernel frontend fails constantly — that is normal, because Triton's surface is much larger than
what any compiler covers on day one. What matters is that a failure says *which* boundary was hit,
so a user can tell "Merlin cannot lower this op yet" from "your access pattern is not affine" from
"the target has no unit for this". A bare ``compile failed`` sends people to read the wrong code.

The other half is the capability report: the ops the bridge SAW versus the ops it LOWERED. A kernel
that compiles is only trustworthy if those two sets account for each other, and the report is what
makes that checkable after the fact rather than a claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BridgeError(RuntimeError):
    """A Triton construct the bridge will not translate (INV-8: never a silent approximation).

    ``op`` and ``hint`` are carried separately from the message so a caller can group failures by
    boundary instead of by wording.
    """

    def __init__(self, message: str, *, op: str | None = None, hint: str | None = None) -> None:
        parts = [message]
        if op:
            parts.append(f"(at {op})")
        if hint:
            parts.append(f"\n  hint: {hint}")
        super().__init__(" ".join(parts))
        self.op = op
        self.hint = hint


@dataclass
class CapabilityReport:
    """What the bridge saw, what it lowered, and how each pointer argument was accessed.

    ``unaccounted`` is the load-bearing field: an op that was neither lowered nor deliberately
    discarded means the translation is incomplete, and the bridge refuses rather than emitting a
    module that is quietly missing part of the computation.
    """

    kernel_name: str
    ttir_ops_seen: dict[str, int] = field(default_factory=dict)
    ttir_ops_lowered: dict[str, int] = field(default_factory=dict)
    ttir_ops_discarded: dict[str, int] = field(default_factory=dict)
    pointer_patterns: dict[str, str] = field(default_factory=dict)
    output_dialects: list[str] = field(default_factory=list)
    grid: tuple[int, int, int] = (1, 1, 1)
    notes: list[str] = field(default_factory=list)

    def saw(self, op_name: str) -> None:
        self.ttir_ops_seen[op_name] = self.ttir_ops_seen.get(op_name, 0) + 1

    def lowered(self, op_name: str) -> None:
        self.ttir_ops_lowered[op_name] = self.ttir_ops_lowered.get(op_name, 0) + 1

    def discarded(self, op_name: str) -> None:
        self.ttir_ops_discarded[op_name] = self.ttir_ops_discarded.get(op_name, 0) + 1

    @property
    def unaccounted(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, n in self.ttir_ops_seen.items():
            rest = n - self.ttir_ops_lowered.get(name, 0) - self.ttir_ops_discarded.get(name, 0)
            if rest:
                out[name] = rest
        return out

    def as_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "grid": list(self.grid),
            "ttir_ops_seen": dict(sorted(self.ttir_ops_seen.items())),
            "ttir_ops_lowered": dict(sorted(self.ttir_ops_lowered.items())),
            "ttir_ops_discarded": dict(sorted(self.ttir_ops_discarded.items())),
            "unaccounted": dict(sorted(self.unaccounted.items())),
            "pointer_patterns": dict(sorted(self.pointer_patterns.items())),
            "output_dialects": sorted(self.output_dialects),
            "notes": list(self.notes),
        }
