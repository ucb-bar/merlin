"""Pure instruction declarations; no backend discovery or target implementation.

Targets supply these typed call interfaces and their independently qualified
rendering/checking capabilities. The declarations contain no target schedule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

OPERAND_KINDS = ("int", "flag", "ptr", "float")


class IsaError(ValueError):
    pass


@dataclass(frozen=True)
class Operand:
    name: str
    kind: str

    def __post_init__(self):
        if self.kind not in OPERAND_KINDS:
            raise IsaError(f"operand {self.name!r}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class InstrDef:
    name: str
    operands: tuple[Operand, ...]
    #: C operand texts in operand order -> one C statement (no trailing newline).
    render_c: Callable[[Sequence[str]], str]
    check: Callable[[Mapping[str, Any], dict], list[str]] | None = None
    footprint: Callable[[Mapping[str, Any]], list[tuple[str, int, str]]] | None = None
    doc: str = ""

    def operand_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.operands)


@dataclass(frozen=True)
class InstructionSet:
    target: str
    instrs: Mapping[str, InstrDef]
    #: IR dtype -> the target's C element type for a tensor argument of that dtype.
    c_types: Mapping[str, str]
    #: One C statement that waits until every issued instruction has completed and its writes are
    #: visible to the host; the emitter ends every kernel with it.
    drain_c: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    #: Rules on the state left after a kernel's LAST instance (e.g. an unfinished accumulation).
    finish: Callable[[dict], list[str]] | None = None

    def instr(self, name: str) -> InstrDef:
        if name not in self.instrs:
            raise IsaError(f"{self.target}: no schedule instruction {name!r} (known: {sorted(self.instrs)})")
        return self.instrs[name]

    def digest(self) -> str:
        """Identity of everything that decides what a call means: operand lists, facts, provenance."""
        doc = {
            "target": self.target,
            "drain": self.drain_c,
            "c_types": dict(self.c_types),
            "instrs": {n: [(o.name, o.kind) for o in d.operands] for n, d in sorted(self.instrs.items())},
            "facts": self.facts,
            "provenance": dict(self.provenance),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()
