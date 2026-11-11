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

from .model import OPERAND_KINDS, InstrDef, InstructionSet, IsaError, Operand


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
