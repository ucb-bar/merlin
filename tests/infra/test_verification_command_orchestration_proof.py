from __future__ import annotations

from verification_witnesses.command_orchestration_proof import (
    _COMMAND,
    _ISA,
    _MVIN,
    HA,
    Context,
    Parser,
    SimpleNamespace,
    _asm,
    _kernel,
    builtin,
    decode,
    judge,
    llvm,
    pytest,
)


def test_plain_tile_walk_is_orchestration(judge) -> None:
    row = judge(_kernel())
    assert row["verdict"] == HA.CLEAN, row
    assert row["arithmetic_basis"] == "pointer_only_target_commands_v1"
    assert row["host_command_orchestration_arithmetic"] > 4 and "command_only_refusal" not in row
