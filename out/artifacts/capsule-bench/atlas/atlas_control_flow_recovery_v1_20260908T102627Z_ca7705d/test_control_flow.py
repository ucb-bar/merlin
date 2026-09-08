"""Regression for the Atlas backend's word-indexed scalar control flow.

This test imports the packaged backend itself.  It deliberately checks the decoded
immediate rather than source text, so changing ``* 2`` back to ``* 4`` fails even
though both versions produce well-formed RISC-V-shaped instruction words.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from submission.mlir_oot import codegen
from submission.mlir_oot import encoder


ROOT = Path(__file__).parent


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _branch_immediate(word: int) -> int:
    value = (
        ((word >> 31) & 1) << 12
        | ((word >> 7) & 1) << 11
        | ((word >> 25) & 0x3F) << 5
        | ((word >> 8) & 0xF) << 1
    )
    return _signed(value, 13)


def _jump_immediate(word: int) -> int:
    value = (
        ((word >> 31) & 1) << 20
        | ((word >> 12) & 0xFF) << 12
        | ((word >> 20) & 1) << 11
        | ((word >> 21) & 0x3FF) << 1
    )
    return _signed(value, 21)


def _declared_units(mnemonic: str) -> int:
    contract = yaml.safe_load((ROOT / "evidence" / "schedule_contract.yaml").read_text())
    rules = contract["control_flow"]["relative_branches"]
    matches = [
        rule["decoded_immediate_units_per_instruction"]
        for rule in rules
        if mnemonic in rule["mnemonics"]
    ]
    assert len(matches) == 1, f"expected one control-flow rule for {mnemonic}, got {len(matches)}"
    return int(matches[0])


def _resolved_back_edge(kind: str) -> tuple[int, int]:
    program = codegen.Program()
    program.label("loop")
    program.emit(encoder.addi(1, 1, 1))
    branch_index = len(program.words)
    if kind == "BNE":
        program.bne(1, 2, "loop")
    elif kind == "BEQ":
        program.beq(1, 2, "loop")
    else:
        program.jump("loop")
    program.resolve()
    decode = _jump_immediate if kind == "JAL" else _branch_immediate
    return decode(program.words[branch_index]), branch_index


def test_all_relative_control_flow_uses_the_target_declared_pc_units() -> None:
    for mnemonic in ("BNE", "BEQ", "JAL"):
        actual, branch_index = _resolved_back_edge(mnemonic)
        expected = (0 - branch_index) * _declared_units(mnemonic)
        assert actual == expected, f"{mnemonic}: decoded {actual}, expected {expected}"


def test_the_observed_squareish_jump_encodes_the_intended_back_edge() -> None:
    """The diagnosed program intended word 3470 -> 2797; ``* 4`` reached word 2124 instead."""
    program = codegen.Program()
    program.words = [encoder.addi(0, 0, 0)] * 3471
    program.labels["loop"] = 2797
    program.patches.append(("jal", 3470, 0, 0, "loop"))
    program.resolve()

    expected = (2797 - 3470) * _declared_units("JAL")
    assert expected == -1346
    assert _jump_immediate(program.words[3470]) == expected


if __name__ == "__main__":
    rows = []
    for mnemonic in ("BNE", "BEQ", "JAL"):
        actual, branch_index = _resolved_back_edge(mnemonic)
        rows.append({
            "mnemonic": mnemonic,
            "branch_index": branch_index,
            "decoded_immediate": actual,
            "declared_units_per_instruction": _declared_units(mnemonic),
        })
    test_all_relative_control_flow_uses_the_target_declared_pc_units()
    test_the_observed_squareish_jump_encodes_the_intended_back_edge()
    print(json.dumps({"ok": True, "checks": rows}, indent=2, sort_keys=True))
