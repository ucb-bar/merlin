"""A package's offload is read from what it emitted, and an emission that says nothing is named."""

from __future__ import annotations

from merlin.targetgen import offload_census as OC


def _buffer(commands=(), placement=None, declined=None) -> dict:
    buffer: dict = {
        "abi_version": "1",
        "target": "synthetic",
        "commands": list(commands),
        "tensors": {
            "A": {"shape": [4, 8], "dtype": "i8"},
            "B": {"shape": [8, 2], "dtype": "i8"},
            "C": {"shape": [4, 2], "dtype": "i32"},
        },
    }
    if placement is not None:
        buffer["params"] = {"lane_placement": placement}
    if declined is not None:
        buffer["declined"] = declined
    return buffer


def test_a_program_that_emits_nothing_and_says_nothing_is_silent() -> None:
    assert OC.program_row("p", _buffer())["outcome"] == "silent"


def test_a_decline_and_a_declared_host_program_are_not_silent() -> None:
    declined = OC.program_row("p", _buffer(declined={"reason": "rank 5 is not lowered"}))
    assert declined["outcome"] == "declined" and "rank 5" in declined["reason"]
    host = OC.program_row("p", _buffer(placement=[{"lane": "host", "family": "elementwise_map"}]))
    assert host["outcome"] == "host_only" and host["placement_declared"]


def test_work_on_a_unit_counts_and_its_placement_must_be_declared() -> None:
    matmul = {"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "B", "dst": "C"}}
    placed = OC.program_row(
        "p",
        _buffer(
            [matmul],
            placement=[{"lane": "mesh", "family": "contraction"}, {"lane": "host", "family": "elementwise_map"}],
        ),
    )
    assert placed["outcome"] == "offloaded" and placed["commands"] == 1
    assert placed["contraction_offload_fraction"] == 1.0
    bare = OC.program_row("p", _buffer([matmul]))
    assert bare["outcome"] == "offloaded" and not bare["placement_declared"]


def test_instruction_use_is_measured_against_the_derived_table_or_says_why_not(monkeypatch) -> None:
    from merlin.kernels.decode import rocc

    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {})
    assert OC.instruction_use(["anything"], target="synthetic")["status"] == "unavailable"

    monkeypatch.setattr(
        rocc,
        "funct_table_for",
        lambda target: {"custom_opcode": "0x7b", "names": {"2": "LOAD", "3": "COMPUTE", "9": "SEQUENCER"}},
    )
    artifact = 'llvm.inline_asm asm_string = ".insn r 0x7b, 0x3, 2, x0, x10, x11"'
    use = OC.instruction_use([artifact], target="synthetic")
    assert use["status"] == "measured"
    assert [row["name"] for row in use["unused"]] == ["COMPUTE", "SEQUENCER"]
