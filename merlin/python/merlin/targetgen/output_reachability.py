"""Structural check: every DECLARED output must be written by some command.

A command buffer can be schema-valid, non-empty, and still compute into nowhere. Measured cost of not
checking this: a run emitted single-command buffers (``['VECTOR_MAP']``, ``['VREDUCE']``, ``['RMSNORM']``)
for 13 capsules with no store command at all. Each validated cleanly, ran on the RTL oracle, read the
output buffer's untouched fill back, and was reported as a numeric mismatch. The mismatch count was a
function of the reference's own value distribution rather than of the kernel, so it could not move -- and
seven rounds of agent effort went into arithmetic that was never the problem. The defect was visible in
the emitted artifact before any simulator ran.

This is deliberately a **reachability** question, not an opcode question. It never asks "is there a
COMMIT": that would be an assumed ISA constant, and it is also WRONG -- in the same measured run two
capsules with no COMMIT passed, writing their outputs by another route. It asks the target-agnostic
question instead: *does any command in this buffer name this output as a destination?* An output that no
command names cannot have been written by any hardware, on any target, under any encoding.

Findings are advisory strings in the house style (same shape as ``verify.structural_checks``), so a
caller can gate on them or merely report them.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: Operand keys that denote a DESTINATION. Kept as a named, extensible set rather than inferred, because
#: guessing direction from a name is how a check becomes silently too narrow. A command that writes
#: through a key not listed here is reported as indeterminate rather than as a failure -- fail closed on
#: the CHECK, not on the submission.
DEST_KEYS = frozenset({"dst", "out", "output", "destination", "result", "acc", "accumulator"})

#: Roles that make a tensor an output the kernel is obliged to produce.
OUTPUT_ROLES = frozenset({"output"})


def _dests(command: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """``(named destinations, operand values whose direction is unknown)`` for one command."""
    ops = command.get("operands")
    if not isinstance(ops, Mapping):
        return set(), set()
    named, unknown = set(), set()
    for key, val in ops.items():
        if not isinstance(val, str):
            continue
        (named if key.lower() in DEST_KEYS else unknown).add(val)
    return named, unknown


def declared_outputs(cb: Mapping[str, Any]) -> list[str]:
    """Tensor names the buffer itself declares as outputs."""
    tensors = cb.get("tensors")
    if not isinstance(tensors, Mapping):
        return []
    return sorted(name for name, spec in tensors.items()
                  if isinstance(spec, Mapping) and spec.get("role") in OUTPUT_ROLES)


def unwritten_outputs(cb: Mapping[str, Any]) -> list[str]:
    """Declared outputs that NO command names as a destination."""
    commands = cb.get("commands")
    if not isinstance(commands, Iterable):
        return declared_outputs(cb)
    written: set[str] = set()
    for c in commands:
        if isinstance(c, Mapping):
            named, _ = _dests(c)
            written |= named
    return [o for o in declared_outputs(cb) if o not in written]


def output_reachability_findings(cb: Mapping[str, Any]) -> list[str]:
    """Advisory findings for a command buffer, empty when every declared output is written.

    Run this BEFORE a simulator. A finding here means the artifact cannot produce a result, so the
    numeric tiers would only measure the output buffer's initial fill.
    """
    outs = declared_outputs(cb)
    if not outs:
        return []                       # nothing declared -- a different check's business
    missing = unwritten_outputs(cb)
    if not missing:
        return []

    # Mention the indeterminate operands: if the buffer writes through a key this module does not
    # recognize as a destination, say so rather than asserting the output is unwritten.
    unknown: set[str] = set()
    for c in (cb.get("commands") or []):
        if isinstance(c, Mapping):
            _, u = _dests(c)
            unknown |= u
    hint = ""
    if unknown & set(missing):
        hint = (f" NOTE: {sorted(unknown & set(missing))} appear as operands under key(s) this check "
                f"does not classify as a destination, so the write may exist through a route it cannot "
                f"see -- treat this as indeterminate, not proven missing.")
    return [f"declared output(s) {', '.join(missing)} are never named as a destination by any command, "
            f"so nothing in this buffer writes them; the numeric tiers would compare the output buffer's "
            f"untouched fill and their mismatch counts would not respond to the kernel." + hint]
