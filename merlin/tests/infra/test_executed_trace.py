"""Executed accelerator-command profile from a functional commit log: register values are replayed from
the log's own integer writes, every command is decoded through the generic RoCC semantics, and a run
that cannot be read completely is UNKNOWN rather than partial."""

from __future__ import annotations

import pytest
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import feedback_metrics as FM
from merlin_experiments.phase2.contracts import StageGateError

from merlin.perf import executed_trace as ET

OPCODE = 0x0B
ISA = {
    "CUSTOM_OPCODE": OPCODE,
    "FUNCT_CLASS": {1: "MOVE", 2: "WORK"},
    "CONFIG_CLASS": None,
    "OPERAND_FIELDS": {
        "MOVE": {
            "unresolved": "omit",
            "fields": [{"name": "rows", "as": "int", "operand": "rs2", "offset": 0, "width": 8}],
        }
    },
}


def _insn(funct: int, rs1: int, rs2: int, rd: int = 0) -> int:
    return (funct << 25) | (rs2 << 20) | (rs1 << 15) | (0b011 << 12) | (rd << 7) | OPCODE


def _log(*rows: tuple[int, str]) -> list[str]:
    lines = []
    for pc, (insn, writes) in enumerate(rows):
        lines.append(f"core   0: 0x{0x80000000 + 4 * pc:016x} (0x{insn:08x}) something")
        lines.append(f"core   0: 3 0x{0x80000000 + 4 * pc:016x} (0x{insn:08x}) {writes}".rstrip())
    return lines


def test_operand_values_are_replayed_from_the_log_and_each_command_decoded():
    lines = _log(
        (0x00500513, "x10 0x0000000000000005"),  # li a0, 5
        (_insn(1, 0, 10), ""),  # MOVE rows=a0
        (0x00700513, "x10 0x0000000000000007"),  # li a0, 7
        (_insn(1, 0, 10), ""),
        (_insn(2, 0, 0), ""),
        (0x4501, "x10 0x0000000000000000"),  # a compressed instruction is never a command
    )
    trace = ET.executed_trace(lines, isa=ISA)
    assert trace["status"] == "measured" and trace["retired_instructions"] == 6
    assert [(row["class"], row["decoded"].get("rows")) for row in trace["instructions"]] == [
        ("MOVE", 5),
        ("MOVE", 7),
        ("WORK", None),
    ]


def test_a_log_beyond_its_bound_is_unknown_never_partial():
    lines = _log(*((_insn(2, 0, 0), "") for _ in range(5)))
    assert ET.executed_trace(lines, isa=ISA, max_retired=3)["status"] == "unknown"
    assert ET.executed_trace(lines, isa={**ISA, "CUSTOM_OPCODE": None})["status"] == "unknown"


def test_the_agent_projection_is_closed_and_consistent():
    measured = {
        "status": "measured",
        "accelerator_commands": 3,
        "retired_instructions": 6,
        "by_class": {"MOVE": 2, "WORK": 1},
        "local_high_water": {"status": "unknown", "why": "no protocol"},
    }
    cell = FM.executed_commands_cell(measured, None)
    assert cell["baseline"]["by_class"] == {"MOVE": 2, "WORK": 1} and cell["baseline"]["local_memory"] is None
    assert cell["candidate"]["status"] == "unknown" and cell["candidate"]["why"]
    CF._validate_executed(cell, index=0, measured=True)
    broken = {**cell, "baseline": {**cell["baseline"], "accelerator_commands": 4}}
    with pytest.raises(StageGateError):
        CF._validate_executed(broken, index=0, measured=True)
    with pytest.raises(StageGateError):
        CF._validate_executed(cell, index=0, measured=False)
    CF._validate_executed(FM.executed_commands_cell(None, None), index=0, measured=False)
