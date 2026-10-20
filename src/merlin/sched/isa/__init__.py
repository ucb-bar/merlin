"""Instruction declarations and the legacy explicit backend lookup bridge."""

from __future__ import annotations

from .types import OPERAND_KINDS, InstrDef, InstructionSet, IsaError, Operand

__all__ = ["OPERAND_KINDS", "InstrDef", "InstructionSet", "IsaError", "Operand", "instruction_set"]


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
