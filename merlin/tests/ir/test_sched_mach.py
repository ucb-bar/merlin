"""The machine model: what it must refuse, and the distinctions it must not lose.

Every test here is anchored to a fact measured on real hardware, named in the docstring, because a
model of a machine that is only self-consistent is a model of nothing.
"""

from __future__ import annotations

import pytest

from merlin.sched.mach import (
    UNIT_KINDS,
    Hierarchy,
    Latency,
    Level,
    MachError,
    Machine,
    Memory,
    Unit,
    Unknown,
)


def _explicit_two_matrix_units() -> Machine:
    """One frontend, two matrix units of different microarchitecture, one bulk-move channel."""
    return Machine(
        target="t_explicit",
        hazard_resolution="explicit",
        units=(
            Unit("mxu0", "systolic", queue="issue"),
            Unit("mxu1", "systolic", queue="issue"),
            Unit("vpu", "vector", queue="issue", lanes=16),
            Unit("ch0", "dma", queue="issue", in_flight=1),
        ),
        memories=(Memory("trf", rows=64, row_bytes=1024, banks=64, read_ports=1, write_ports=1),),
        hierarchy=Hierarchy((Level("stream", 1),)),
        delay_instruction="DELAY",
        latencies=(
            Latency("mma", "mxu0", issue=1, result=95, completion="counted", contended=True),
            Latency("mma", "mxu1", issue=1, result=34, completion="counted", contended=True),
            Latency("load", "ch0", issue=1, result=None, completion="polled"),
        ),
    )


def test_issue_is_serial_but_units_still_overlap():
    """Measured: 'operations execute concurrently ... but only one new instruction issues per cycle'.

    Conflating the two says two matrix units behind one frontend cannot overlap, which would delete the
    dominant performance lever on that machine.
    """
    m = _explicit_two_matrix_units()
    assert m.shares_issue("mxu0", "mxu1") is True
    assert m.can_overlap("mxu0", "mxu1") is True


def test_two_units_may_declare_one_shared_execution_resource():
    m = Machine(
        target="t_shared",
        hazard_resolution="explicit",
        units=(
            Unit("a", "vector", queue="q", executes_on="alu"),
            Unit("b", "vector", queue="q", executes_on="alu"),
        ),
        hierarchy=Hierarchy((Level("stream", 1),)),
    )
    assert m.can_overlap("a", "b") is False


def test_latency_is_keyed_by_instruction_and_unit():
    """Measured: the same matmul is 95 cycles on a systolic array and 34 on an inner-product tree."""
    m = _explicit_two_matrix_units()
    assert m.latency("mma", "mxu0").result == 95
    assert m.latency("mma", "mxu1").result == 34


def test_an_underived_latency_is_none_not_zero():
    """An unmeasured cost must read as UNKNOWN. A zero reads as free and silently ranks first."""
    m = _explicit_two_matrix_units()
    assert m.latency("mma", "vpu") is None


def test_bulk_movement_completes_polled_and_needs_no_cycle_count():
    """Measured guidance: 'for DMA completion always use the wait, never a fixed delay'."""
    m = _explicit_two_matrix_units()
    lat = m.latency("load", "ch0")
    assert lat.completion == "polled"
    assert lat.result is None


def test_a_counted_completion_without_a_count_is_refused():
    with pytest.raises(MachError, match="counted completion needs a result latency"):
        Latency("op", "u", issue=1, result=None, completion="counted")


def test_a_counted_completion_without_a_delay_instruction_is_refused():
    """If the compiler must emit the wait, the machine has to say what it emits."""
    with pytest.raises(MachError, match="declares no delay instruction"):
        Machine(
            target="t",
            hazard_resolution="explicit",
            units=(Unit("u", "vector", queue="q"),),
            latencies=(Latency("op", "u", issue=1, result=9, completion="counted"),),
        )


def test_an_underived_hazard_model_must_carry_its_reason():
    """Defaulting this makes an illegal schedule look legal, or a dead gate look alive."""
    with pytest.raises(MachError, match="no Unknown explaining why"):
        Machine(target="t", hazard_resolution=None)
    ok = Machine(
        target="t",
        hazard_resolution=None,
        unknowns=(Unknown("hazard_resolution", "decoder RTL not introspected"),),
    )
    assert ok.hazard_resolution is None


def test_hazard_resolution_is_a_closed_vocabulary():
    with pytest.raises(MachError, match="hazard_resolution"):
        Machine(target="t", hazard_resolution="mostly")


def test_banked_shared_memory_names_its_contenders_and_its_arbiter():
    """Measured: the shared scratchpad arbitrates lowest-index-first, so a bank shared between the
    SIMT lanes and the mesh starves the mesh. A flat byte range cannot express that."""
    m = Machine(
        target="t_simt",
        hazard_resolution="explicit",
        units=(Unit("lanes", "simt", queue="warp", lanes=16), Unit("mesh", "systolic", queue="mmio")),
        memories=(
            Memory(
                "smem",
                rows=2048,
                row_bytes=64,
                banks=4,
                arbiter="lowest_index_first",
                shared_by=("lanes", "mesh"),
            ),
        ),
        hierarchy=Hierarchy((Level("core", 2), Level("warp", 8), Level("lane", 16))),
    )
    smem = m.memory("smem")
    assert smem.banks == 4
    assert smem.rows_per_bank == 512
    assert smem.nbytes == 131072
    assert smem.shared_by == ("lanes", "mesh")
    assert m.can_overlap("lanes", "mesh") is True
    assert m.shares_issue("lanes", "mesh") is False


def test_a_memory_cannot_be_contended_by_a_unit_the_machine_does_not_have():
    with pytest.raises(MachError, match="not a unit of this machine"):
        Machine(
            target="t",
            hazard_resolution="explicit",
            units=(Unit("a", "vector", queue="q"),),
            memories=(Memory("m", rows=1, row_bytes=1, banks=1, shared_by=("ghost",)),),
        )


def test_one_instruction_stream_is_a_degenerate_hierarchy_not_an_absent_one():
    """The corpus-anatomy finding: generalising the single-stream shape bakes 'one instruction stream'
    into the abstraction, and an obligation to map work across threads can then never be discharged."""
    single = Machine(
        target="t1",
        hazard_resolution="interlocked",
        units=(Unit("mesh", "systolic", queue="cmd"),),
        hierarchy=Hierarchy((Level("stream", 1),)),
    )
    assert single.hierarchy.degenerate is True
    assert single.hierarchy.width == 1

    simt = Hierarchy((Level("core", 2), Level("warp", 8), Level("lane", 16)))
    assert simt.degenerate is False
    assert simt.width == 256


def test_an_unknown_hierarchy_extent_makes_the_width_unknown_not_one():
    h = Hierarchy((Level("warp", None), Level("lane", 16)))
    assert h.width is None


def test_movers_are_units_but_not_compute_units():
    """A bulk-move engine is schedulable and computes nothing. Declaring it a compute unit would
    corrupt every capability query that reads that vocabulary."""
    from merlin.targetgen.compute_units import KINDS as COMPUTE_KINDS

    assert "dma" in UNIT_KINDS
    assert "dma" not in COMPUTE_KINDS
    assert COMPUTE_KINDS <= UNIT_KINDS


def test_unknown_unit_kind_is_refused():
    with pytest.raises(MachError, match="not in"):
        Unit("u", "quantum", queue="q")


def test_duplicate_names_are_refused():
    with pytest.raises(MachError, match="duplicate unit names"):
        Machine(
            target="t",
            hazard_resolution="explicit",
            units=(Unit("u", "vector", queue="q"), Unit("u", "simt", queue="q")),
        )
    with pytest.raises(MachError, match="duplicate latency"):
        Machine(
            target="t",
            hazard_resolution="explicit",
            units=(Unit("u", "vector", queue="q"),),
            latencies=(
                Latency("op", "u", issue=1, result=2),
                Latency("op", "u", issue=1, result=3),
            ),
        )


def test_a_latency_cannot_name_a_unit_the_machine_does_not_have():
    with pytest.raises(MachError, match="not a unit of this machine"):
        Machine(
            target="t",
            hazard_resolution="explicit",
            units=(Unit("u", "vector", queue="q"),),
            latencies=(Latency("op", "ghost", issue=1, result=2),),
        )


def test_digest_covers_provenance_and_every_cost():
    """Two machines derived from different RTL are different machines even when the numbers match: a
    measurement keyed on one must not be reused for the other."""
    a = _explicit_two_matrix_units()
    assert a.digest() == _explicit_two_matrix_units().digest()

    other_prov = Machine(
        target=a.target,
        hazard_resolution=a.hazard_resolution,
        units=a.units,
        memories=a.memories,
        hierarchy=a.hierarchy,
        latencies=a.latencies,
        delay_instruction=a.delay_instruction,
        provenance={"facts": "different_rtl"},
    )
    assert other_prov.digest() != a.digest()

    slower = Machine(
        target=a.target,
        hazard_resolution=a.hazard_resolution,
        units=a.units,
        memories=a.memories,
        hierarchy=a.hierarchy,
        delay_instruction=a.delay_instruction,
        latencies=(
            Latency("mma", "mxu0", issue=1, result=96, completion="counted", contended=True),
            Latency("mma", "mxu1", issue=1, result=34, completion="counted", contended=True),
            Latency("load", "ch0", issue=1, result=None, completion="polled"),
        ),
    )
    assert slower.digest() != a.digest()


def test_text_reports_unknowns_and_the_contention_caveat():
    """A latency floor measured with every other unit idle is a lower bound, not a value; a schedule
    that deliberately overlaps traffic must be able to see that."""
    m = Machine(
        target="t",
        hazard_resolution=None,
        units=(Unit("u", "vector", queue="q"),),
        delay_instruction="DELAY",
        latencies=(Latency("op", "u", issue=1, result=64, completion="counted", contended=True),),
        unknowns=(Unknown("hazard_resolution", "decoder RTL not introspected"),),
    )
    text = m.text()
    assert "hazards=UNKNOWN" in text
    assert "UNKNOWN hazard_resolution: decoder RTL not introspected" in text
    assert "assumes no contention" in text


# -- the model has to be able to travel ---------------------------------------------------------------


def test_the_compute_kinds_still_agree_with_the_contract_vocabulary():
    """Replaces what the import used to guarantee, and says louder what it used to say silently.

    `mach.model` declared its compute kinds by importing `targetgen.compute_units.KINDS`, so the two
    could not drift. That worked, and it was this module's ONLY reach outside `merlin.sched` -- which is
    what kept the machine model out of a minted package, since such a package may not import this tree
    at all. So the kinds are declared locally and held equal here instead.

    Failing this test is not a bug to silence: a kind added upstream used to arrive here silently and
    widen what a `Machine` accepts. Now it asks for a decision, which is what widening a model's
    admissible values should be.
    """
    from merlin.sched.mach.model import COMPUTE_KINDS
    from merlin.targetgen.compute_units import KINDS

    assert COMPUTE_KINDS == KINDS, (
        "the scheduling machine model and the capability contract disagree about what kinds of compute "
        f"unit exist: {sorted(COMPUTE_KINDS ^ KINDS)} is in one and not the other. Declare it in both, "
        "or decide deliberately that a Machine does not admit it."
    )


def test_the_machine_model_imports_nothing_outside_the_scheduling_package():
    """The property that lets the model be carried into a package beside the vocabulary that reads it.

    Checked with the integrity scan's OWN predicate rather than a substring search, because that scan
    is what will reject the package: a name is not an import, and an import is not always spelled the
    same way. `merlin.sched.*` is internal cohesion, which the namespace-based scan cannot tell from a
    reach into the harness -- so this asserts the stricter thing the scan cannot, that nothing outside
    the scheduling package is reached at all.
    """
    import pathlib

    from merlin.sched.mach import model
    from merlin.targetgen.oot_runner import _py_imports_merlin

    hit = _py_imports_merlin(pathlib.Path(model.__file__).read_text(encoding="utf-8"))
    assert hit is None or hit.startswith("merlin.sched"), (
        f"merlin.sched.mach.model reaches {hit!r}, outside the scheduling package. A minted package may "
        "not import this tree, so a model that reaches out cannot travel with the primitives."
    )
