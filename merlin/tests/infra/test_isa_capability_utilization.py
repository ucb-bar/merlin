"""What a program emits, against what its target declares it can do.

Measured on a whole-model ResNet-50 emission: 8 of the 25 declared functs were ever emitted, and
the 17 that were not included the entire device-side convolution sequencer. Because it was never
emitted, im2col patch generation ran as host scalar code -- 37% of all host dynamic operations --
and the accelerator idled behind the CPU. Nothing in the compiler's output said so: the command
buffer was well formed and the correctness gate passed. The absence is only visible by diffing
emitted instructions against declared ones.

The funct table is a parameter here, exactly as it is in the module: these tests use invented
names so that nothing pins a real target's ISA into the test suite.
"""

from __future__ import annotations

from merlin.perf.isa_utilization import capability_utilization, emitted_functs

OPCODE = 0x5B
OTHER_OPCODE = 0x2B
DECLARED = {0: "CONFIG", 1: "LOAD", 2: "STORE", 8: "SEQ", 9: "SEQ_BOUNDS", 15: "SEQ_CONV"}


def _asm(template: str) -> str:
    return f'"llvm.inline_asm"() <{{asm_string = "{template}", constraints = "r,r"}}>'


def _artifact(*templates: str) -> str:
    return "\n".join(_asm(t) for t in templates)


def test_only_the_targets_own_opcode_is_counted() -> None:
    """A fence or an unrelated custom instruction is not an accelerator instruction."""
    art = _artifact(".insn r 0x5b, 0x3, 0x8, x0, $0, $1", ".insn r 0x2b, 0x3, 0x8, x0, $0, $1", "fence")
    assert emitted_functs(art, custom_opcode=OPCODE) == {8: 1}
    assert emitted_functs(art, custom_opcode=OTHER_OPCODE) == {8: 1}


def test_occurrences_are_counted_not_just_presence() -> None:
    art = _artifact(*[".insn r 0x5b, 0x3, 0x8, x0, $0, $1"] * 3, ".insn r 0x5b, 0x3, 0x0, x0, $0, $1")
    assert emitted_functs(art, custom_opcode=OPCODE) == {8: 3, 0: 1}


def test_unused_declared_instructions_are_reported_by_name() -> None:
    """The finding: a capability the hardware offers and the compiler never reaches for."""
    art = _artifact(".insn r 0x5b, 0x3, 0x8, x0, $0, $1", ".insn r 0x5b, 0x3, 0x9, x0, $0, $1")
    out = capability_utilization(art, declared_functs=DECLARED, custom_opcode=OPCODE)
    assert out["declared_count"] == 6
    assert out["used_count"] == 2
    assert out["used"] == {"SEQ": 1, "SEQ_BOUNDS": 1}
    assert [row["name"] for row in out["unused"]] == ["CONFIG", "LOAD", "STORE", "SEQ_CONV"]
    assert [row["funct"] for row in out["unused"]] == [0, 1, 2, 15]


def test_a_program_using_everything_reports_no_opportunity() -> None:
    art = _artifact(*[f".insn r 0x5b, 0x3, {code:#x}, x0, $0, $1" for code in DECLARED])
    out = capability_utilization(art, declared_functs=DECLARED, custom_opcode=OPCODE)
    assert out["unused"] == []
    assert out["used_count"] == out["declared_count"] == 6


def test_an_emitted_funct_the_facts_do_not_declare_is_flagged_separately() -> None:
    """Using an instruction nobody vouched for is a provenance problem, not an opportunity."""
    art = _artifact(".insn r 0x5b, 0x3, 0x1f, x0, $0, $1")
    out = capability_utilization(art, declared_functs=DECLARED, custom_opcode=OPCODE)
    assert out["undeclared_emitted"] == [0x1F]
    assert out["used_count"] == 0


def test_an_empty_program_uses_nothing_and_says_so() -> None:
    out = capability_utilization("", declared_functs=DECLARED, custom_opcode=OPCODE)
    assert out["used_count"] == 0
    assert len(out["unused"]) == 6
    assert out["undeclared_emitted"] == []


def test_a_malformed_template_is_skipped_not_guessed() -> None:
    art = _artifact(".insn r 0x5b", ".insn r 0x5b, 0x3, notanumber, x0", "fence", ".insn r 0x5b, 0x3, 0x8, x0, $0, $1")
    assert emitted_functs(art, custom_opcode=OPCODE) == {8: 1}
