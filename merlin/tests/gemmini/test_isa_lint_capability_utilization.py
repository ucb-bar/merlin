"""The ISA lint the assisted arms call now says which declared instructions the program never emits.

Target-specific on purpose: the measurement's two inputs -- the instruction table and the custom
major opcode -- are DERIVED from one target's own RTL facts, so the only way to show that the
derivation reaches the lint is to run it against a target that has facts on disk. The instrument
itself is target-neutral and is unit-tested with an invented ISA in
``merlin/tests/infra/test_isa_capability_utilization.py``.

Measured on a whole-model emission of this target: the program used 8 of the 25 functs the RTL
declares, and the 17 it never emitted included the entire device-side convolution sequencer -- so
every convolution's patch generation ran as host scalar code while the command buffer stayed well
formed, the lint stayed clean and the counters looked busy.
"""

from __future__ import annotations

import pytest

TARGET = "gemmini"


def _broker():
    from merlin_experiments.phase1.brokers import isa_tools

    return isa_tools


@pytest.fixture(scope="module")
def linted():
    broker = _broker()
    ctx = broker.BrokerCtx(endpoint=broker.ROCC_ENDPOINT, target=TARGET)
    assembled = broker._handle({"cmd": "asm", "text": "CONFIG_EX 0 0\nFENCE\n"}, ctx)
    if "error" in assembled:
        pytest.skip(f"the derived assembler is unavailable here: {assembled['error']}")
    return broker._handle({"cmd": "lint", "mlir": assembled["mlir"]}, ctx)


def test_the_lint_measures_declared_against_emitted(linted) -> None:
    out = linted["capability_utilization"]
    if out.get("status") != "measured":
        pytest.skip(f"this checkout carries no derived funct table: {out.get('reason')}")
    assert out["declared_count"] > 1
    assert 0 < out["used_count"] < out["declared_count"]


def test_the_lint_names_the_instructions_the_program_never_reached_for(linted) -> None:
    """The finding, in words the agent reads -- not a count it has to interpret."""
    out = linted["capability_utilization"]
    if out.get("status") != "measured":
        pytest.skip(f"this checkout carries no derived funct table: {out.get('reason')}")
    assert out["unused"], "a two-instruction program used every declared instruction"
    assert any("Never emitted" in f for f in linted["findings"])


def test_the_pretty_printed_operation_form_is_counted(linted) -> None:
    """The assembler emits ``llvm.inline_asm has_side_effects ".insn ..."`` -- the operation form,
    not the ``asm_string`` attribute. Reading only the attribute reported 0 of 26 declared
    instructions on a program that had just emitted one, which is a clean number produced by not
    looking."""
    out = linted["capability_utilization"]
    if out.get("status") != "measured":
        pytest.skip(f"this checkout carries no derived funct table: {out.get('reason')}")
    assert out["used"], "no emitted instruction was recognised in the assembler's own output"


def test_an_underivable_funct_table_is_an_explicit_unknown() -> None:
    """A target with no derived table must not report full utilization.

    An empty declared table makes every emitted instruction undeclared and nothing unused, so the
    lint would read as though the program used everything the machine offers. That is the shape of
    a check that cannot fail, so the absence is recorded instead.
    """
    broker = _broker()
    out = broker._capability_utilization(".insn r 0x7b, 0x3, 0x0, x0, $0, $1", "a_target_with_no_rtl_facts_here")
    assert out["status"] == "UNKNOWN"
    assert out["declared_count"] is None
    assert "NOT measured" in out["reason"]
