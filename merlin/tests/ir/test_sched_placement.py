"""On-chip placement: the two failures it exists to catch, and the guesses it refuses.

Both failures are measured and both are silent. A destination written over a live operand produced zero
correct elements out of 16,384 on a shipped kernel, with no error anywhere -- the program ran and the
harness completed. And a bank shared between two engines starves the one the arbiter deprioritises,
which decides whether a stage overlaps at all and is invisible from an address.

Neither is a question about the archetype, so unlike the synchronisation check this asks one question of
every machine. What it needs from the machine is the memory's geometry, not its hazard model.
"""

from __future__ import annotations

import pytest

from merlin.sched.check.placement import check_placement
from merlin.sched.ir import Kernel, Stage, TensorArg, call
from merlin.sched.mach import Machine, Memory, Unit
from merlin.sched.primitives import Cursor, NotApplicable, PrimitiveError, bind_bank, proof_of, stage_mem

pytestmark = pytest.mark.target("gemmini", "muon")

ARG = (TensorArg("a", (8,), "i8", "read"),)
UNITS = (Unit(name="mesh", kind="systolic", queue="q"), Unit(name="simt", kind="simt", queue="q"))
SHARED = Memory(name="smem", rows=1024, row_bytes=64, banks=4, arbiter="lowest_index_first", shared_by=("mesh", "simt"))
#: A store with no declared bank count -- the state a target is in before anything derived one.
UNBANKED = Memory(name="spad", rows=256, row_bytes=16, banks=None)


def _machine(memories=(SHARED, UNBANKED)) -> Machine:
    return Machine(target="t", hazard_resolution="interlocked", units=UNITS, memories=memories)


# -- the 0 / 16,384 failure ------------------------------------------------------------------------


def test_a_write_over_a_live_operand_is_caught():
    k = Kernel(
        "k",
        ARG,
        (
            call(
                "mm",
                unit="mesh",
                stages=(
                    Stage(memory="smem", row=0, rows=64),
                    Stage(memory="smem", row=32, rows=64, writes=True),
                ),
            ),
        ),
    )
    report = check_placement(k, _machine())
    assert not report.ok
    assert any("overlap and one of them writes" in p for p in report.problems)
    assert any("16,384" in p for p in report.problems), "the finding does not say what it cost"


def test_two_readers_may_share_rows():
    """The control. Without it the test above passes for a check that flags every overlap."""
    k = Kernel(
        "k",
        ARG,
        (
            call(
                "mm",
                unit="mesh",
                stages=(
                    Stage(memory="smem", row=0, rows=64),
                    Stage(memory="smem", row=32, rows=64),
                ),
            ),
        ),
    )
    assert check_placement(k, _machine()).ok, "two reads of the same rows were reported as a conflict"


def test_adjacent_stages_do_not_overlap():
    """Off-by-one in the other direction: rows [0:64) and [64:128) touch and do not collide."""
    k = Kernel(
        "k",
        ARG,
        (
            call(
                "mm",
                unit="mesh",
                stages=(
                    Stage(memory="smem", row=0, rows=64),
                    Stage(memory="smem", row=64, rows=64, writes=True),
                ),
            ),
        ),
    )
    assert check_placement(k, _machine()).ok


# -- the starved bank ------------------------------------------------------------------------------


def test_two_contending_engines_in_one_bank_are_reported():
    k = Kernel(
        "k",
        ARG,
        (
            call("mm", unit="mesh", stages=(Stage(memory="smem", row=0, rows=8, bank=2),)),
            call("vv", unit="simt", stages=(Stage(memory="smem", row=8, rows=8, bank=2),)),
        ),
    )
    problems = check_placement(k, _machine()).problems
    assert any("contend for it" in p and "bank 2" in p for p in problems), problems


def test_the_same_two_engines_in_different_banks_are_fine():
    k = Kernel(
        "k",
        ARG,
        (
            call("mm", unit="mesh", stages=(Stage(memory="smem", row=0, rows=8, bank=0),)),
            call("vv", unit="simt", stages=(Stage(memory="smem", row=8, rows=8, bank=1),)),
        ),
    )
    assert check_placement(k, _machine()).ok, "separate banks were reported as contention"


def test_starvation_is_reported_only_where_the_machine_declares_both_facts():
    """Without a declared arbiter there is no priority to lose; without a contender list nothing says
    two units are the two. Inferring either turns a fact about the device into a guess about it."""
    k = Kernel(
        "k",
        ARG,
        (
            call("mm", unit="mesh", stages=(Stage(memory="smem", row=0, rows=8, bank=2),)),
            call("vv", unit="simt", stages=(Stage(memory="smem", row=8, rows=8, bank=2),)),
        ),
    )
    no_arbiter = Memory(name="smem", rows=1024, row_bytes=64, banks=4, shared_by=("mesh", "simt"))
    no_contenders = Memory(name="smem", rows=1024, row_bytes=64, banks=4, arbiter="lowest_index_first")
    for memory in (no_arbiter, no_contenders):
        assert check_placement(k, _machine((memory,))).ok, f"claimed starvation from {memory}"


# -- capacity and honest unknowns ------------------------------------------------------------------


def test_a_stage_past_the_declared_capacity_is_caught():
    k = Kernel("k", ARG, (call("mm", unit="mesh", stages=(Stage(memory="smem", row=1000, rows=64),)),))
    assert any("runs past" in p for p in check_placement(k, _machine()).problems)


def test_a_bank_named_against_an_underived_bank_count_is_refused():
    """A bank chosen out of an unknown count is a guess wearing a placement's clothes."""
    k = Kernel("k", ARG, (call("mm", unit="mesh", stages=(Stage(memory="spad", row=0, rows=8, bank=1),)),))
    assert any("never derived" in p for p in check_placement(k, _machine()).problems)


def test_a_stage_into_a_memory_the_machine_lacks_is_caught():
    k = Kernel("k", ARG, (call("mm", unit="mesh", stages=(Stage(memory="nowhere", row=0, rows=8),)),))
    assert any("does not have" in p for p in check_placement(k, _machine()).problems)


def test_the_report_says_how_many_stages_it_looked_at():
    """No problems over zero stages is an UNPLACED kernel, not a well-placed one."""
    assert check_placement(Kernel("k", ARG, (call("mm", unit="mesh"),)), _machine()).checked == 0
    k = Kernel("k", ARG, (call("mm", unit="mesh", stages=(Stage(memory="smem", row=0, rows=8),)),))
    assert check_placement(k, _machine()).checked == 1


# -- the primitive ---------------------------------------------------------------------------------


def test_bind_bank_declares_a_bounded_check_and_stages_an_operand():
    machine = _machine()
    assert proof_of(bind_bank) == "bounded_check"
    k = bind_bank(
        Kernel("k", ARG, (call("mm", unit="mesh"),)),
        Cursor(index=0),
        memory="smem",
        row=0,
        rows=64,
        bank=1,
        machine=machine,
    )
    assert "in smem[0:64]@1" in k.text()
    assert check_placement(k, machine).ok


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"memory": "smem", "row": 1000, "rows": 64}, "runs past"),
        ({"memory": "smem", "row": 0, "rows": 8, "bank": 9}, "is not one of them"),
        ({"memory": "spad", "row": 0, "rows": 8, "bank": 1}, "a guess rather than a placement"),
        ({"memory": "smem", "row": 0, "rows": 0}, "occupies nothing"),
    ],
)
def test_bind_bank_refuses_what_it_cannot_justify(kwargs, expect):
    with pytest.raises(NotApplicable, match=expect):
        bind_bank(Kernel("k", ARG, (call("mm", unit="mesh"),)), Cursor(index=0), machine=_machine(), **kwargs)


def test_bind_bank_refuses_a_memory_the_machine_does_not_have():
    """Definite, not a miss: the caller was reasoning about a machine that is not this one."""
    with pytest.raises(PrimitiveError, match="no memory named"):
        bind_bank(
            Kernel("k", ARG, (call("mm", unit="mesh"),)),
            Cursor(index=0),
            memory="nowhere",
            row=0,
            rows=8,
            machine=_machine(),
        )


# -- staging into a memory space -------------------------------------------------------------------


def _staging_machine() -> Machine:
    return Machine(
        target="t",
        hazard_resolution="interlocked",
        units=(Unit(name="dma", kind="dma", queue="q"), Unit(name="mesh", kind="systolic", queue="q")),
        memories=(Memory(name="stage", rows=1024, row_bytes=16, banks=4), UNBANKED),
    )


def test_stage_mem_inserts_the_move_and_records_both_ends():
    """The move WRITES the rows and the consumer READS them, which is what makes every placement check
    above possible. Annotating only one end would leave a consumer reading rows nothing wrote."""
    machine = _staging_machine()
    k = stage_mem(
        Kernel("k", ARG, (call("mm", unit="mesh"),)),
        Cursor(index=0),
        move=call("mvin", unit="dma"),
        memory="stage",
        row=0,
        rows=64,
        bank=0,
        machine=machine,
    )
    lines = [line.strip() for line in k.text().splitlines() if "stage" in line]
    assert lines[0].endswith("stage[0:64]@0w"), f"the move does not write the rows: {lines}"
    assert lines[1].endswith("stage[0:64]@0"), f"the consumer does not read them: {lines}"
    assert check_placement(k, machine).ok and check_placement(k, machine).checked == 2


def test_stage_mem_puts_the_move_before_its_consumer():
    """A move after the call that reads it is not a staging, whatever the annotations say."""
    machine = _staging_machine()
    k = stage_mem(
        Kernel("k", ARG, (call("mm", unit="mesh"),)),
        Cursor(index=0),
        move=call("mvin", unit="dma"),
        memory="stage",
        row=0,
        rows=8,
        machine=machine,
    )
    order = [line.strip().split("(")[0] for line in k.text().splitlines()[1:]]
    assert order == ["mvin", "mm"], order


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"memory": "stage", "row": 1000, "rows": 64}, "runs past"),
        ({"memory": "stage", "row": 0, "rows": 0}, "brings nothing on chip"),
        ({"memory": "stage", "row": 0, "rows": 8, "bank": 9}, "is not one of them"),
        ({"memory": "nowhere", "row": 0, "rows": 8}, "no memory named"),
    ],
)
def test_stage_mem_refuses_what_it_cannot_justify(kwargs, expect):
    machine = _staging_machine()
    with pytest.raises((NotApplicable, PrimitiveError), match=expect):
        stage_mem(
            Kernel("k", ARG, (call("mm", unit="mesh"),)),
            Cursor(index=0),
            move=call("mvin"),
            machine=machine,
            **kwargs,
        )


def test_stage_mem_refuses_to_overlap_rows_the_consumer_already_reads():
    machine = _staging_machine()
    k = stage_mem(
        Kernel("k", ARG, (call("mm", unit="mesh"),)),
        Cursor(index=0),
        move=call("mvin", unit="dma"),
        memory="stage",
        row=0,
        rows=64,
        machine=machine,
    )
    with pytest.raises(NotApplicable, match="overlaps"):
        stage_mem(k, Cursor(index=1), move=call("mvin"), memory="stage", row=32, rows=8, machine=machine)


def test_two_operands_may_be_staged_into_disjoint_rows():
    """The control: without it the refusal above would pass for a primitive that stages only once."""
    machine = _staging_machine()
    k = Kernel("k", ARG, (call("mm", unit="mesh"),))
    k = stage_mem(k, Cursor(index=0), move=call("mvin", unit="dma"), memory="stage", row=0, rows=64, machine=machine)
    k = stage_mem(k, Cursor(index=1), move=call("mvin", unit="dma"), memory="stage", row=64, rows=64, machine=machine)
    assert check_placement(k, machine).ok
    assert check_placement(k, machine).checked == 4


def test_stage_mem_needs_the_move_from_the_caller():
    """Choosing the instruction would mean a per-target table inside a generic vocabulary."""
    with pytest.raises(PrimitiveError, match="from its own ISA"):
        stage_mem(
            Kernel("k", ARG, (call("mm", unit="mesh"),)),
            Cursor(index=0),
            move="mvin",
            memory="stage",
            row=0,
            rows=8,
            machine=_staging_machine(),
        )
