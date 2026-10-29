"""Incremental exact register replay for complete committed command streams.

Every input line is consumed. Retired CPU instructions use constant state;
commands and fences are yielded in their original order without a trace list.
"""

from __future__ import annotations

from merlin.perf.executed_trace import _hex


#: Retired-instruction bound of one complete source trace.
DEFAULT_MAX_RETIRED = 200_000_000
#: Issued accelerator commands plus fences of one complete source trace.
DEFAULT_MAX_COMMANDS = 4_000_000


def iter_source_commands(view, *, isa, max_retired=DEFAULT_MAX_RETIRED, max_commands=DEFAULT_MAX_COMMANDS, raw=False):
    """Yield each executed command (decoded) and each fence in order.

    ``raw=True`` yields ``(funct, rs1, rs2)`` integer tuples and ``None`` per fence instead, for a
    consumer that reads operand slots from derived facts and needs no per-command decode.
    """
    from merlin.targetgen.rocc import semantics

    opcode = isa.get("CUSTOM_OPCODE")
    if (
        type(opcode) is not int
        or not 0 < opcode < 128
        or any(type(v) is not int or v < 1 for v in (max_retired, max_commands))
    ):
        raise ValueError("streamed trace needs exact custom opcode and finite command/retirement bounds")
    registers = [0] * 32
    retired = commands = events = 0
    for line in view.splitlines():
        parts = line.split()
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
            if raw:
                events += 1
                if events > max_commands:
                    raise ValueError("complete source trace exceeds issued event limit")
                yield (word >> 25, registers[(word >> 15) & 31], registers[(word >> 20) & 31])
                commands += 1
                _write_registers(parts, registers)
                continue
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
            yield None if raw else {"class": "FENCE"}
        if len(parts) > 6:
            _write_registers(parts, registers)
    if not commands:
        raise ValueError("complete source trace contains no accelerator commands")


def _write_registers(parts, registers):
    writes = parts[5:]
    for at in range(len(writes) - 1):
        name = writes[at]
        if len(name) > 1 and name[0] == "x" and name[1:].isdigit():
            value, index = _hex(writes[at + 1]), int(name[1:])
            if value is not None and 0 < index < 32:
                registers[index] = value


def iter_recorded_commands(records, *, isa, max_commands=DEFAULT_MAX_COMMANDS):
    """:func:`iter_source_commands` with ``raw=True`` over recorded ``(bits, rs1, rs2)`` retirements.

    ``records`` holds, in program order, every retired instruction of the kinds this reader yields (the
    command recorder's stream, :mod:`merlin.perf.command_recorder`) with the values its source
    registers held when it issued. The commit log's register replay gives exactly those values, so the
    yielded items, their order and every refusal are the commit-log reader's. The retired-instruction
    bound is the recorder's to check (it records the count); every other line of the commit log yields
    nothing.
    """
    opcode = isa.get("CUSTOM_OPCODE")
    if type(opcode) is not int or not 0 < opcode < 128 or type(max_commands) is not int or max_commands < 1:
        raise ValueError("streamed trace needs exact custom opcode and finite command/retirement bounds")
    commands = events = 0
    for word, rs1, rs2 in records:
        if word & 3 == 3 and word & 127 == opcode:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield (word >> 25, rs1, rs2)
            commands += 1
        elif word & 127 == 15 and word >> 12 & 7 == 0:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield None
    if not commands:
        raise ValueError("complete source trace contains no accelerator commands")
