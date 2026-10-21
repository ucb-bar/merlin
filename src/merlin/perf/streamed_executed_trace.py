"""Incremental exact register replay for complete committed command streams.

Every input line is consumed. Retired CPU instructions use constant state;
commands and fences are yielded in their original order without a trace list.
"""
from __future__ import annotations

from merlin.perf.executed_trace import _hex


def iter_source_commands(view, *, isa, max_retired=200_000_000, max_commands=4_000_000):
    from merlin.targetgen.rocc import semantics

    opcode = isa.get("CUSTOM_OPCODE")
    if type(opcode) is not int or not 0 < opcode < 128 or any(type(v) is not int or v < 1 for v in (max_retired, max_commands)):
        raise ValueError("streamed trace needs exact custom opcode and finite command/retirement bounds")
    registers = [0] * 32
    retired = commands = events = 0
    for raw in view.splitlines():
        parts = raw.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("complete source trace requires the original single hart")
        word = _hex(parts[4])
        if word is None:
            continue
        retired += 1
        if retired > max_retired:
            raise ValueError("complete source trace exceeds retired instruction limit")
        if word & 3 == 3 and word & 127 == opcode:
            rs1 = {"raw": registers[(word >> 15) & 31], "kind": "const"}
            rs2 = {"raw": registers[(word >> 20) & 31], "kind": "const"}
            funct = word >> 25
            cls, decoded = semantics.decode_instruction(funct, rs1, rs2, dict(isa))
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield dict(index=commands, **{"class": cls}, funct=funct, rs1=rs1, rs2=rs2, decoded=decoded)
            commands += 1
        elif word & 127 == 15 and word >> 12 & 7 == 0:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield {"class": "FENCE"}
        writes = parts[5:]
        for at in range(len(writes) - 1):
            name = writes[at]
            if len(name) > 1 and name[0] == "x" and name[1:].isdigit():
                value, index = _hex(writes[at + 1]), int(name[1:])
                if value is not None and 0 < index < 32:
                    registers[index] = value
    if not commands:
        raise ValueError("complete source trace contains no accelerator commands")
