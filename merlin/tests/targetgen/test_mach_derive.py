"""What a machine derivation may and may not say about a real target.

Every test here is about a state the derivation must REACH, not about the model refusing a bad one --
``merlin.sched.mach.model`` already refuses those, and its own tests pin that. The failure mode this
file exists for is the opposite one: a derivation that reports a plausible number where it should
report a gap, which reads as knowledge and is worse than silence.
"""

from __future__ import annotations

import dataclasses

import pytest

from merlin.sched.mach import HAZARD_RESOLUTIONS, ROLE_UNKNOWN, MachError
from merlin.sched.mach.derive import derive, machine_from_address_space, unknown_census
from merlin.targetgen.address_space import derive_address_space

pytestmark = pytest.mark.target("gemmini", "atlas", "muon", "toy_npu")

#: Targets whose machine this suite asserts over. Named here rather than discovered because each one is
#: present for a DIFFERENT reason -- a fully derived one, one whose only store must be refused, one with
#: a real thread hierarchy, one that models no silicon -- and a discovered roster would silently drop
#: whichever of those four stopped resolving.
TARGETS = ("gemmini", "atlas", "muon", "toy_npu")


def _machines() -> dict[str, object]:
    out = {}
    for name in TARGETS:
        try:
            out[name] = derive(name)
        except Exception as exc:  # noqa: BLE001 - a target that cannot derive is reported, not skipped
            out[name] = exc
    return out


def test_every_target_either_derives_a_hazard_model_or_names_what_is_missing():
    """Three-state, per target. A Machine coming back is not the assertion -- one with a null hazard
    model and a matching Unknown satisfies that trivially, which is why the next test exists."""
    for name, machine in _machines().items():
        assert not isinstance(machine, Exception), f"{name}: {machine!r}"
        if machine.hazard_resolution is None:
            reasons = [u.reason for u in machine.unknowns if u.quantity == "hazard_resolution"]
            assert reasons, f"{name}: hazard model is None with no Unknown naming why"
            assert len(reasons[0]) > 40, f"{name}: the reason does not say what is missing: {reasons[0]!r}"
        else:
            assert machine.hazard_resolution in HAZARD_RESOLUTIONS


def test_at_least_one_target_derives_a_hazard_model_and_at_least_one_reports_unknown():
    """The existence proof that stops the test above being vacuous.

    A derivation hardwired to answer UNKNOWN for everything passes every per-target assertion in this
    file. Requiring BOTH states to be reachable is what makes the suite able to fail in that direction.
    """
    machines = [m for m in _machines().values() if not isinstance(m, Exception)]
    resolved = [m for m in machines if m.hazard_resolution is not None]
    unknown = [m for m in machines if m.hazard_resolution is None]
    assert resolved, "no target derived a hazard model; the derivation may be refusing unconditionally"
    assert unknown, "every target derived one; a target with no declaration must still report UNKNOWN"


def test_an_underived_quantity_is_none_and_never_zero():
    """An unread quantity is not a zero.

    A zero row count, a zero latency or a zero lane width all read as facts about the device, and each
    one has a cost: a zero capacity admits every schedule, a zero latency ranks first.
    """
    for name, machine in _machines().items():
        if isinstance(machine, Exception):
            continue
        for memory in machine.memories:
            for field in ("rows", "row_bytes", "banks", "read_ports", "write_ports"):
                assert getattr(memory, field) != 0, f"{name}/{memory.name}.{field}: an unread quantity is not a zero"
        for unit in machine.units:
            assert unit.lanes != 0 and unit.in_flight != 0, f"{name}/{unit.name}: an unread quantity is not a zero"
        for latency in machine.latencies:
            assert latency.issue != 0 and latency.result != 0, (
                f"{name}/{latency.instr}: an underived latency is not free"
            )
        assert machine.array_edge != 0


def test_a_store_described_only_by_its_write_port_geometry_is_refused():
    """A port declares a write granularity, not a capacity.

    One target's only extracted store is a byte-enable geometry that over-reports its real on-chip
    capacity by more than an order of magnitude, and the facts say so in the store's own provenance. A
    derivation that passed it through would hand every capacity and bank decision a plausible wrong
    number. Verified against the real artifact, so this fires if the extractor's provenance changes.
    """
    space = derive_address_space("atlas")
    ported = [s for s in space.stores if (s.sources or {}).get("bytes_depth") == "firrtl_port_geometry"]
    if not ported:
        pytest.skip("this checkout's artifact describes no port-geometry store; nothing to refuse")
    machine = derive("atlas")
    assert [m.name for m in machine.memories if m.name in {s.name for s in ported}] == [], (
        "a write-port byte-enable geometry was reported as a memory"
    )
    refusals = [u for u in machine.unknowns if u.quantity == "memory"]
    assert refusals, "the store was dropped without saying so; a silent drop is indistinguishable from no store"
    assert "write granularity" in refusals[0].reason


def test_a_degenerate_hierarchy_is_one_level_not_an_absent_one():
    """A target with one instruction stream still has a hierarchy, written the same way as every other.

    ``Hierarchy(())`` would satisfy ``degenerate`` and ``width == 1`` vacuously -- ``all()`` over an
    empty tuple is True -- so a derivation that skipped the hierarchy entirely would read as degenerate.
    Requiring at least one level is what distinguishes the two.
    """
    machine = derive("gemmini")
    assert len(machine.hierarchy.levels) >= 1, "a skipped hierarchy reads as degenerate; it is not"
    assert machine.hierarchy.degenerate and machine.hierarchy.width == 1
    simt = derive("muon")
    assert len(simt.hierarchy.levels) >= 2 and not simt.hierarchy.degenerate
    assert simt.hierarchy.width > 1


def test_a_unit_the_evidence_did_not_classify_keeps_its_unknown_role():
    """Whatever else it does, the derivation never invents a unit's kind."""
    for name, machine in _machines().items():
        if isinstance(machine, Exception):
            continue
        for unit in machine.units:
            if unit.kind == ROLE_UNKNOWN:
                assert any(u.quantity == "unit.kind" and u.where == unit.name for u in machine.unknowns)
                with pytest.raises(MachError):
                    machine.units_of(ROLE_UNKNOWN)


def test_the_census_is_a_named_set_not_a_count():
    """The exit criterion is which quantities are unanswered, not how many.

    A new quantity appearing is a regression even when the total falls, and one disappearing is progress
    that has to be re-asserted here. Pinning the total instead would accept a swap in silence.
    """
    census = unknown_census(derive("gemmini"))
    assert set(census) == {"arbiter", "read_ports", "write_ports", "unit.in_flight", "latency"}, (
        f"gemmini's unanswered quantities changed: {census}. If this is progress, narrow the set here; "
        "if it is new, the derivation started reporting a gap it used to answer."
    )


def test_an_address_space_alone_cannot_answer_the_hazard_question():
    """The half-machine a caller gets from an address space says so rather than guessing."""
    machine = machine_from_address_space(derive_address_space("gemmini"))
    assert machine.hazard_resolution is None
    assert any(u.quantity == "hazard_resolution" for u in machine.unknowns)
    assert machine.array_edge == 16


def test_the_declared_hazard_model_is_cross_checked_against_the_isa():
    """A contract claiming hardware interlocks while the target declares a delay instruction is a
    contradiction, and the derivation refuses to pick a winner.

    The mutation this guards: quietly preferring one declaration makes the other unfalsifiable, and the
    one likelier to be stale is the one nobody reads.
    """
    contract = {"memory_model": {"hazard_resolution": "interlocked"}, "endpoint_kind": "inline_asm_insn"}
    with pytest.raises(MachError, match="interlocked"):
        derive(
            "gemmini",
            contract=contract,
            schedule_contract={
                "delay_instruction": {"mnemonic": "DELAY", "cycles_operand": "imm"},
                "minimum_issue_gap": [{"name": "r", "unit": "systolic_mesh", "producers": ["OP"], "cycles": 34}],
            },
        )


def test_an_issue_gap_is_a_distance_in_slots_not_a_latency():
    """A declared issue gap counts the producer's own slot; the occupancy is one less.

    The target's own rationale records it: a 34-cycle resource occupancy yields a 35-cycle issue
    distance "because the producer and delay instruction each consume an issue cycle". Writing the
    conversion down is what stops the two numbers being reconciled by deleting one of them.
    """
    machine = derive(
        "gemmini",
        contract={
            "memory_model": {"hazard_resolution": "explicit"},
            "endpoint_kind": "inline_asm_insn",
            "compute_units": [{"name": "u", "kind": "systolic"}],
        },
        schedule_contract={
            "delay_instruction": {"mnemonic": "DELAY", "cycles_operand": "imm"},
            "minimum_issue_gap": [{"name": "occ", "unit": "u", "producers": ["OP"], "cycles": 35}],
            "register_dependency_gap": [{"name": "vis", "unit": "u", "producers": ["OP"], "cycles": 66}],
        },
    )
    cost = machine.latency("OP", "u")
    assert cost is not None and cost.issue == 34, "the producer's own slot was not removed from the gap"
    assert cost.result == 66, "result visibility is a separate number from issue occupancy"
    assert cost.contended is True, "the declared floors assume an idle machine and must say so"


def test_a_rule_that_names_no_unit_becomes_an_unknown_not_a_guess():
    """Recovering a unit from a rule's name or its mnemonics' spelling is a guess in the middle of the
    one quantity that must not be guessed."""
    machine = derive(
        "gemmini",
        contract={
            "memory_model": {"hazard_resolution": "explicit"},
            "endpoint_kind": "inline_asm_insn",
            "compute_units": [{"name": "u", "kind": "systolic"}],
        },
        schedule_contract={
            "delay_instruction": {"mnemonic": "DELAY", "cycles_operand": "imm"},
            "minimum_issue_gap": [{"name": "mxu0_matmul_resource", "producers": ["OP"], "cycles": 96}],
        },
    )
    assert machine.latency("OP", "u") is None
    assert any(u.quantity == "latency.unit" and u.where == "mxu0_matmul_resource" for u in machine.unknowns)


def test_a_machine_with_no_schedule_contract_says_it_has_no_costs():
    """Zero latencies is not the same claim as "this machine's instructions are free".

    Measured on every target in the tree: `derive(target)` yields no latencies at all, because nothing
    loads a schedule contract -- the parameter is injectable and has no loader, and the only contract
    in the tree lives under `merlin/experiments/`, which library code may not read. That is a defensible
    state; being silent about it is not. Without this the census lists four answered-looking gaps and a
    reader concludes the costs are known, when in fact the most important quantity for scheduling is
    absent on every target.
    """
    machine = derive("gemmini")
    assert machine.latencies == ()
    gap = [u for u in machine.unknowns if u.quantity == "latency"]
    assert len(gap) == 1, f"a machine with no costs must name that gap once; got {gap}"
    assert "no schedule contract was supplied" in gap[0].reason


def test_a_declared_cost_for_a_unit_the_machine_lacks_is_reported_not_dropped():
    """The likeliest cause is the one that matters most: if the contract and the unit list disagree
    about what the units are called, EVERY cost is filtered out and the machine prices a fully serial
    schedule at zero -- silently, since a machine with no costs is otherwise a legal machine."""
    machine = derive(
        "gemmini",
        contract={
            "memory_model": {"hazard_resolution": "explicit"},
            "endpoint_kind": "inline_asm_insn",
            "compute_units": [{"name": "u", "kind": "systolic"}],
        },
        schedule_contract={
            "delay_instruction": {"mnemonic": "DELAY", "cycles_operand": "imm"},
            "minimum_issue_gap": [
                {"name": "occ", "unit": "a_unit_that_is_not_here", "producers": ["OP"], "cycles": 35}
            ],
        },
    )
    assert machine.latencies == ()
    orphan = [u for u in machine.unknowns if u.quantity == "latency.unit" and u.where == "OP"]
    assert orphan, f"the dropped cost was not reported: {machine.unknowns}"
    assert "a_unit_that_is_not_here" in orphan[0].reason


def test_a_cost_naming_a_unit_the_machine_has_is_kept():
    """The control for the test above: without it, reporting every cost as an orphan would pass."""
    machine = derive(
        "gemmini",
        contract={
            "memory_model": {"hazard_resolution": "explicit"},
            "endpoint_kind": "inline_asm_insn",
            "compute_units": [{"name": "u", "kind": "systolic"}],
        },
        schedule_contract={
            "delay_instruction": {"mnemonic": "DELAY", "cycles_operand": "imm"},
            "minimum_issue_gap": [{"name": "occ", "unit": "u", "producers": ["OP"], "cycles": 35}],
        },
    )
    assert machine.latency("OP", "u") is not None
    assert not [u for u in machine.unknowns if u.quantity == "latency"]


def test_every_machine_field_reaches_the_digest():
    """Reflected over the dataclass rather than listed, so a field added later is covered by this test
    the day it is added -- which a hand-written list can never be."""
    machine = derive("gemmini")
    base = machine.digest()
    for field in dataclasses.fields(machine):
        if field.name in {"target", "unknowns", "provenance"}:
            continue
        current = getattr(machine, field.name)
        moved = 1 if current in (None, 0) else None
        if field.name == "hazard_resolution":
            moved = "explicit" if current != "explicit" else "interlocked"
        elif field.name in {"units", "memories", "latencies"}:
            moved = ()
        elif field.name == "hierarchy":
            continue
        elif field.name == "delay_instruction":
            moved = "X" if current is None else None
        if moved is None or moved == current:
            continue
        assert dataclasses.replace(machine, **{field.name: moved}).digest() != base, (
            f"{field.name} does not reach Machine.digest(); a measurement keyed on it would be reused "
            "across two machines that differ in it"
        )
