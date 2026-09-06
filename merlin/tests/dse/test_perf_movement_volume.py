"""Operand traffic recovered from the command buffer -- the roofline's denominator.

A work total without a traffic total is not a roofline: it says how much arithmetic a program does and
nothing about whether the machine could feed it. These tests pin the counting rules that decide
whether a movement lever is visible at all.
"""
from __future__ import annotations

import pytest

from merlin.perf.movement_volume import (ProgramMovement, movement_evidence,
                                         movement_from_command_buffer,
                                         NO_COMMAND_BUFFER_REFUSAL)
from merlin.perf.work_volume import work_from_command_buffer


def _resident_matmul(k: int = 2048, uses: int = 1) -> dict:
    """A resident-weight matmul: pack W once, read A0 per use, commit Y0."""
    commands = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}}]
    for i in range(uses):
        commands.append({"opcode": "MATMUL_RESIDENT",
                         "operands": {"lhs": "A0", "rhs": "W_res", "dst": f"acc{i}"}})
    commands += [{"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}},
                 {"opcode": "EVICT", "operands": {"handle": "W_res"}}]
    return {"abi_version": 1, "target": "t", "backend": "b",
            "tensors": {"W": {"shape": [k, 16], "dtype": "i8", "role": "weight"},
                        "A0": {"shape": [16, k], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [16, 16], "dtype": "i32", "role": "output"}},
            "commands": commands}


def test_traffic_is_the_declared_tensor_bytes_that_crossed_the_boundary():
    mv = movement_from_command_buffer(_resident_matmul())
    assert mv.known_bytes_in == 2048 * 16 + 16 * 2048          # W packed once, A0 read once
    assert mv.known_bytes_out == 16 * 16 * 4                   # Y0 is i32
    assert mv.exact_bytes == 66560
    assert not mv.is_lower_bound and not mv.refusals


def test_a_resident_operand_is_charged_once_at_its_pack_not_at_every_use():
    """THE RULE THE RESIDENCY LEVER DEPENDS ON.

    Charging a resident handle per use would report a weight re-fetched on every tile -- exactly the
    traffic residency removes -- so the lever would read as inert, or as making things worse.
    """
    one, four = (movement_from_command_buffer(_resident_matmul(uses=u)) for u in (1, 4))
    assert four.known_bytes_in - one.known_bytes_in == 3 * (16 * 2048), "only the extra A0 reads"
    assert four.known_bytes_in == 2048 * 16 + 4 * (16 * 2048)


def test_an_unknown_opcode_makes_the_total_unknown_rather_than_smaller():
    cb = _resident_matmul()
    cb["commands"].append({"opcode": "FUSE_SOMETHING", "operands": {}})
    mv = movement_from_command_buffer(cb)
    assert mv.exact_bytes is None and mv.is_lower_bound
    assert mv.known_bytes > 0, "what WAS counted survives as a lower bound"
    assert any("no traffic-counting rule" in r for r in mv.refusals)


def test_an_operand_naming_no_declared_tensor_is_refused_not_skipped():
    cb = _resident_matmul()
    cb["commands"][1]["operands"]["lhs"] = "GHOST"
    mv = movement_from_command_buffer(cb)
    assert mv.exact_bytes is None
    assert any("do not resolve" in r for r in mv.refusals)


def test_a_sub_byte_format_occupies_what_it_actually_occupies():
    cb = _resident_matmul(k=16)
    cb["tensors"]["W"]["dtype"] = "i8"
    plain = movement_from_command_buffer(cb).exact_bytes
    assert plain is not None and plain > 0


def test_no_command_buffer_is_unknown_never_zero():
    block = movement_evidence(None, compiler_provenance="p")
    assert block["exact_bytes"] is None
    assert block["known_bytes"] == 0
    assert block["refusals"] == [NO_COMMAND_BUFFER_REFUSAL]
    assert block["compiler_provenance"] == "p"


def test_both_axes_agree_on_the_program_they_counted():
    """An intensity taken across two different programs is not an intensity."""
    cb = _resident_matmul()
    mv, wk = movement_from_command_buffer(cb), work_from_command_buffer(cb)
    assert mv.artifact_sha256 == wk.artifact_sha256 != ""
    assert wk.exact_macs / mv.exact_bytes == pytest.approx(524288 / 66560)


def test_the_block_declares_what_it_cannot_measure() -> None:
    """A resident operand is charged once at its pack, so this module counts the traffic the command
    buffer DECLARES, not what the emitted program issues. A lowering that re-loads a resident tile
    per output tile yields an identical number here -- measured on gemmini, whose command buffer
    emits a correct RES_PACK / MATMUL_RESIDENT / EVICT sequence while the emitted stream reloads.

    The limitation therefore has to travel with the number: a reader holding only this block must be
    able to tell that residency is NOT evidenced by it. Without these fields the module reports
    intent and reads as measurement, which is the failure this repository keeps paying for.
    """
    block = ProgramMovement(commands=(), known_bytes_in=0, known_bytes_out=0,
                            is_lower_bound=False, refusals=()).to_dict()
    assert block["counts"] == "declared_by_command_buffer"
    assert block["resident_operand_charged"] == "once_at_pack"
    assert "re-loads a resident operand" in block["cannot_detect"]
    assert "RES_PACK" in block["cannot_detect"], "name the check that WOULD detect it"
