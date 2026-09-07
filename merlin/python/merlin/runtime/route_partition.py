"""The admission partition, checked on a produced command buffer.

WHAT THIS IS FOR
----------------
A compiler handed a program must end up in exactly one of three routes, and it must SAY which:

* ``A`` — accepted: target work was emitted;
* ``H`` — an explicit host or composed route, whose seam is checked elsewhere;
* ``D`` — an explicit decline: no target command and no target-visible effect.

The routes must be disjoint and they must cover every input. A checker's own
"outside my model" disposition is not a fourth route: it has to land in ``H`` or ``D``, never in
``A``. Written as the partition obligation this module enforces::

    A, H, D pairwise disjoint          every input lands in one of them
    a decline is EXPLICIT              exit code and an empty list are not a decline
    a decline has NO target effect     it may not also emit commands

**An empty command list plus a zero exit code does not identify accepted work.** That sentence is the
whole module. It is also the single most productive defect this repo has: measured 2026-09-07, 244
command buffers under ``out/runs`` are empty AND carry no decline, so nothing in them distinguishes
"this compiler declined, correctly" from "this compiler dropped the program on the floor and reported
success". Three of those cases were reproduced from a spec in one command each — an
``rmsnorm``, a forbidden direct-scratchpad op, and a stateful random op — every one exiting 0 with an
empty stderr, no decline, and the program's declared OUTPUT TENSOR silently absent from the buffer.

The vocabulary already existed and was only ever read as an EXCUSE:
``commandbuffer.validate_command_buffer`` lets a ``declined`` buffer skip its
operands-reference-a-tensor check. Nothing ever required the key. So a backend that emits nothing and
says nothing has been contract-conformant this whole time, which is why no grader caught it.

WHY THIS IS SEPARATE FROM ``validate_command_buffer``
-----------------------------------------------------
Because turning this on CHANGES WHAT THE GRADER ACCEPTS, and the grader is an instrument: comparing two
runs whose validators disagree compares the instruments as well as the submissions. ``preflight``,
``oot_runner`` and ``capsule_common`` all call ``validate_command_buffer``, so wiring the partition into
it retro-fails live and archived submissions at once.

This module is therefore a PURE classifier with no callers in the grading path. The gate
``build_tools/scripts/check_route_partition.py`` reports against a ratchet. Wiring it into
``validate_command_buffer`` is a deliberate, scheduled instrument change — not a side effect of
landing the check.

TARGET-AGNOSTIC
---------------
Nothing here knows a target. The partition is a property of the command-buffer ABI, which every
endpoint speaks — the backend contract requires ``emit_command_buffer`` of every package regardless of
whether the endpoint is RoCC instructions, a SIMT kernel or a self-hosted ISA. A target that grew a
fourth route would declare it in the ABI, and this module would then read it from there.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ROUTE_ACCEPT", "ROUTE_DECLINE", "ROUTE_VIOLATION", "ROUTES",
           "SILENT_NO_WORK", "DECLINES_AND_EMITS", "DECLINE_WITHOUT_REASON", "VIOLATION_KINDS",
           "RouteVerdict", "route_of"]

#: Target work was emitted. Route ``A`` of the partition (``H`` is not distinguishable from ``A`` by
#: the buffer alone — the seam that separates them is declared outside it — so both land here and the
#: host-lane axis is what tells them apart).
ROUTE_ACCEPT = "A"

#: An explicit decline carrying a reason, with no emitted command. Route ``D``.
ROUTE_DECLINE = "D"

#: Neither route is identified. This is NOT a route — it is the partition failing to close, and the
#: reason it gets its own name is that it used to be silently counted as ``D``.
ROUTE_VIOLATION = "!"

ROUTES = (ROUTE_ACCEPT, ROUTE_DECLINE, ROUTE_VIOLATION)

#: No commands and no decline: indistinguishable from a dropped program reported as success.
SILENT_NO_WORK = "silent_no_work"

#: A decline AND emitted commands. The two routes are meant to be disjoint, and a consumer reading
#: ``declined`` excuses the case while a consumer reading ``commands`` executes it — so the same
#: artifact grades as "declined, not a failure" and still runs. Measured 2026-09-07: 30 such buffers
#: under ``out/runs``, one of them declining ``RES_PACK`` while emitting
#: ``RES_PACK, MATMUL_RESIDENT, COMMIT, EVICT`` and declaring an output.
DECLINES_AND_EMITS = "declines_and_emits"

#: A decline with no reason, or one that is not a mapping. A decline has to be readable by whoever has
#: to act on it; "declined: true" tells a caller nothing about what to fix.
DECLINE_WITHOUT_REASON = "decline_without_reason"

VIOLATION_KINDS = (SILENT_NO_WORK, DECLINES_AND_EMITS, DECLINE_WITHOUT_REASON)


class RouteVerdict:
    """One buffer's route, plus every way it fails the partition.

    ``route`` is :data:`ROUTE_VIOLATION` exactly when ``violations`` is non-empty, so a caller cannot
    read a route of ``D`` off a buffer that only looked declined because it was empty.
    """

    __slots__ = ("route", "violations", "reason", "declined_op", "n_commands", "outputs")

    def __init__(self, route: str, violations: list[tuple[str, str]], *, reason: str | None,
                 declined_op: str | None, n_commands: int, outputs: list[str]):
        self.route = route
        self.violations = violations
        self.reason = reason
        self.declined_op = declined_op
        self.n_commands = n_commands
        self.outputs = outputs

    @property
    def ok(self) -> bool:
        """True when this buffer lands in exactly one route."""
        return not self.violations

    @property
    def kinds(self) -> list[str]:
        """The violation kinds only, for counting."""
        return [k for k, _ in self.violations]

    def __repr__(self) -> str:
        return (f"RouteVerdict(route={self.route!r}, n_commands={self.n_commands}, "
                f"violations={self.kinds!r})")


def _output_names(cb: dict[str, Any]) -> list[str]:
    """Names the buffer declares as outputs.

    ``tensors`` is a mapping in the ABI, but a buffer that declares none has been seen to spell it as
    an empty LIST, so this tolerates both rather than raising on the shape — the point of the module is
    to classify malformed output, not to reject it before classifying.
    """
    tensors = cb.get("tensors")
    if not isinstance(tensors, dict):
        return []
    out = []
    for name, spec in tensors.items():
        if isinstance(spec, dict) and str(spec.get("role")) == "output":
            out.append(str(name))
    return sorted(out)


def route_of(cb: dict[str, Any]) -> RouteVerdict:
    """Classify one command buffer into the admission partition.

    Pure: reads the buffer and nothing else — no target, no manifest, no filesystem. A buffer that
    satisfies the partition returns :data:`ROUTE_ACCEPT` or :data:`ROUTE_DECLINE` with an empty
    ``violations``; anything else returns :data:`ROUTE_VIOLATION` and says why, with enough context in
    the message to act on without reopening the file.
    """
    commands = cb.get("commands")
    commands = commands if isinstance(commands, list) else []
    declined = cb.get("declined")
    has_decline = declined is not None
    outputs = _output_names(cb)

    reason = None
    declined_op = None
    if isinstance(declined, dict):
        raw_reason = declined.get("reason")
        reason = str(raw_reason) if raw_reason else None
        raw_op = declined.get("op")
        declined_op = str(raw_op) if raw_op else None

    violations: list[tuple[str, str]] = []

    if has_decline and commands:
        violations.append((DECLINES_AND_EMITS, (
            f"the buffer declines ({'op=' + repr(declined_op) + ', ' if declined_op else ''}"
            f"reason={reason!r}) and ALSO emits {len(commands)} command(s) "
            f"{[str(c.get('opcode')) for c in commands if isinstance(c, dict)][:6]}"
            f"{', declaring output(s) ' + repr(outputs) if outputs else ''}. Routes A and D are "
            f"disjoint: a consumer reading 'declined' excuses this case while a consumer reading "
            f"'commands' executes it, so the same artifact both grades as declined and runs")))

    if has_decline and not reason:
        violations.append((DECLINE_WITHOUT_REASON, (
            f"the buffer declines but gives no readable reason (declined={declined!r}). A decline is "
            f"an instruction to whoever has to act on it; it needs to say what was not lowered")))

    if not commands and not has_decline:
        violations.append((SILENT_NO_WORK, (
            f"the buffer emits no command and declares no 'declined' block"
            f"{', and declares output(s) ' + repr(outputs) + ' that nothing computes' if outputs else ''}"
            f". An empty command list plus a zero exit code does not identify accepted work, so this "
            f"is indistinguishable from a program dropped on the floor and reported as success")))

    if violations:
        route = ROUTE_VIOLATION
    elif commands:
        route = ROUTE_ACCEPT
    else:
        route = ROUTE_DECLINE
    return RouteVerdict(route, violations, reason=reason, declined_op=declined_op,
                        n_commands=len(commands), outputs=outputs)
