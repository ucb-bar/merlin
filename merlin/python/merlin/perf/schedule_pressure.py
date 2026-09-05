"""How many on-chip tiles a schedule keeps live at once, read from the emitted program.

Two candidate schedules for the same workload issue the same commands and do the same arithmetic;
what separates them is the ORDER, and ordering is what the cheap signals available here are worst at.
Measured on this corpus over 774 within-workload ordered pairs: the correctness simulator's cycles
agree with the timing oracle 46.1% of the time and a per-command cost model 39.3% -- both at or below
chance, the second because within one workload the term tracking the work never varies.

What does vary, and what the fastest and slowest schedules of the same program differ in, is how much
is in flight. Sorting one workload's schedules by measured time and printing their command order
showed the fastest tightly interleaving loads with computes and the slowest batching every load
first: same instruction count, same arithmetic, 28% apart. This module measures that directly --
the peak number of operand tiles simultaneously live between the command that loads one and the
command that consumes it.

It is a HEURISTIC over an emitted program, not a cycle model. It reports a count; the caller decides
whether that count has earned the right to order anything, using measured agreement rather than
plausibility. Nothing here names a target: the field vocabulary is the emitted ABI's own, declared
below the way a completion opcode vocabulary is declared, and a program using none of it yields
UNKNOWN rather than a pressure of zero.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Decoded fields that DEFINE an on-chip operand tile, and those that CONSUME one. Both are ABI-level
#: names carried by the emitted program itself, not target facts: a backend speaking this ABI emits
#: them whatever its ISA. A program declaring none of them is unreadable here, and says so.
DEFINES = ("spad_addr",)
CONSUMES = ("a_spad", "weight_spad")

UNKNOWN = "UNKNOWN"


def _decoded(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("decoded")
    return payload if isinstance(payload, Mapping) else {}


def peak_live_tiles(instructions: object) -> dict[str, Any]:
    """Peak simultaneously-live operand tiles across one emitted program.

    A tile becomes live where a command defines its address and dies after the LAST command that
    consumes it. The peak over the program is the count reported. A tile that is defined and never
    consumed stays live to the end -- that is real pressure, not an error, and is counted.
    """
    if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
        return {"status": UNKNOWN, "reason": "the program is not a sequence of instructions"}
    rows = [r for r in instructions if isinstance(r, Mapping)]
    if not rows:
        return {"status": UNKNOWN, "reason": "the program declares no instructions"}

    defs: dict[int, int] = {}          # address -> index where it became live
    last_use: dict[int, int] = {}      # address -> index of its final consumer
    seen_any = False
    for index, row in enumerate(rows):
        payload = _decoded(row)
        for name in DEFINES:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                seen_any = True
                defs.setdefault(value, index)
        for name in CONSUMES:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                seen_any = True
                last_use[value] = index
    if not seen_any:
        return {"status": UNKNOWN,
                "reason": ("no instruction declares an operand-tile address in the ABI vocabulary, "
                           "so this program's tile pressure cannot be read"),
                "vocabulary": {"defines": list(DEFINES), "consumes": list(CONSUMES)}}

    end = len(rows) - 1
    intervals = [(start, last_use.get(address, end)) for address, start in defs.items()]
    peak = 0
    for index in range(len(rows)):
        live = sum(1 for start, stop in intervals if start <= index <= stop)
        peak = max(peak, live)
    return {"status": "counted", "peak_live_tiles": peak, "tiles": len(intervals),
            "instructions": len(rows),
            "basis": "peak operand tiles simultaneously live between definition and final use"}


def pressure_of(trace: object) -> dict[str, Any]:
    """Read a decoded instruction trace and report its tile pressure, or why it cannot be read.

    Refuses on a program carrying an undecodable command or a loop construct: a loop means the
    static command list is not the dynamic one, and pressure read off the static list would describe
    a program that never ran.
    """
    if not isinstance(trace, Mapping):
        return {"status": UNKNOWN, "reason": "the trace is not a mapping"}
    rows = trace.get("instructions")
    if not isinstance(rows, Sequence):
        return {"status": UNKNOWN, "reason": "the trace declares no instruction list"}
    classes = {str(r.get("class") or "") for r in rows if isinstance(r, Mapping)}
    for blocking, why in (("UNKNOWN", "the decoder could not read a command"),
                          ("LOOP_WS", "a loop makes the static command list differ from what ran")):
        if blocking in classes:
            return {"status": UNKNOWN, "reason": why, "blocking_class": blocking}
    return peak_live_tiles(rows)
