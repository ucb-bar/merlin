"""A target's schedule instruction set: the calls a kernel schedule may make, and how each is checked.

A target provides one ``InstructionSet`` through its backend hook ``sched_instruction_set()``. Each
``InstrDef`` gives the instruction's operand list (names and kinds), how one call renders as a C
statement, and optionally two checks over one dynamic instance's concrete operand values:

- ``check(values, state)`` returns violated legality rules. ``state`` is one dict shared by every
  instance of one kernel, in program order, so a rule that spans instructions (a loop instruction whose
  row strides must agree with the most recent load configuration) can be stated;
- ``footprint(values)`` returns the DRAM byte extents the instance touches, as
  ``(pointer operand, bytes, "read" | "write")``, so the generic checker can refuse an out-of-bounds or
  read-only-write access without knowing the instruction.

The instruction set is derived by the target from its own sources; this module only defines the shape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

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


def instruction_set(target: str) -> InstructionSet:
    from merlin.runtime.backends import base

    backend = base.get_backend(target)
    hook = getattr(backend, "sched_instruction_set", None)
    if hook is None:
        raise IsaError(f"target {target!r} declares no schedule instruction set")
    iset = hook()
    if not isinstance(iset, InstructionSet):
        raise IsaError(f"target {target!r}: sched_instruction_set() returned {type(iset).__name__}")
    return iset
