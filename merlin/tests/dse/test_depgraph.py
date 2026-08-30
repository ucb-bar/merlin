"""Dependence, liveness and the ranking of schedules, on a program small enough to check by hand.

The fixture is a producer/consumer chain with a declared wait in it and an independent transfer that
could sit inside that wait. That is the shape the real question has -- three quarters of the measured
cycles had no unit busy, and this is the smallest program in which "the wait is covered by nothing"
is true and fixable.
"""
from __future__ import annotations

import pytest

from merlin.perf import depgraph as DG
from merlin.perf import differential
from merlin.perf.decompose import UNKNOWN
from merlin.perf.deps import liveness as LV
from merlin.targetgen import isa_direction as ID


def _dir(mnemonic: str, **spec) -> dict:
    return {name: ID.OperandDirection(mnemonic, name, direction, file, written_slots=slots,
                                      reason="fixture")
            for name, (direction, file, slots) in spec.items()}


DIRECTIONS = ID.DirectionModel(target="fixture", by_mnemonic={
    "LOAD": _dir("LOAD", vd=(ID.DEF, "mrf", (0,)), rs1=(ID.USE, "scalar", ())),
    "MUL": _dir("MUL", vd=(ID.DEF, "acc", (0,)), vs1=(ID.USE, "mrf", ()), vs2=(ID.USE, "mrf", ())),
    "POP": _dir("POP", vd=(ID.DEF, "mrf", (0, 1)), vs2=(ID.USE, "acc", ())),
    "MOVE": _dir("MOVE", rd=(ID.DEF, "scalar", (0,)), rs1=(ID.USE, "scalar", ())),
    "WAIT": {},
})

ISSUE = DG.IssueModel(issue_cycles=1.0, stall_unit=1.0, tier="fixture",
                      provenance="fixture: one cycle per instruction, one cycle per stall unit")
ROLES = {"LOAD": "memory", "MUL": "matmul", "POP": "readout", "MOVE": "scalar", "WAIT": "scalar"}


def _program() -> DG.Program:
    ins = [
        LV.Instruction(0, "LOAD", {"vd": 0, "rs1": 1}),
        LV.Instruction(1, "WAIT", {"imm": 10}),
        LV.Instruction(2, "MUL", {"vd": 0, "vs1": 0, "vs2": 0}),
        LV.Instruction(3, "WAIT", {"imm": 10}),
        LV.Instruction(4, "POP", {"vd": 4, "vs2": 0}),
        LV.Instruction(5, "MOVE", {"rd": 2, "rs1": 3}),      # independent of everything above
    ]
    effects = tuple(LV.effects_of(i, DIRECTIONS) for i in ins)
    return DG.Program(instructions=tuple(ins), effects=effects,
                      regions=(DG.Region("all", 0, len(ins)),), roles=ROLES)


@pytest.fixture()
def program() -> DG.Program:
    return _program()


@pytest.fixture()
def dag(program) -> DG.Dag:
    return DG.build_dag(program.instructions, program.effects, issue=ISSUE,
                        stall_mnemonic="WAIT", roles=ROLES)


# ---------------------------------------------------------------------------------------------------
# def-use and liveness
# ---------------------------------------------------------------------------------------------------
def test_a_wide_definition_covers_every_slot_it_was_measured_to_write(program):
    defs = program.effects[4].defs
    assert defs == (LV.Access("mrf", 4), LV.Access("mrf", 5))


def test_an_instruction_never_probed_is_incomplete_not_empty():
    effect = LV.effects_of(LV.Instruction(0, "NEVER_PROBED", {"rd": 1}), DIRECTIONS)
    assert effect.defs == () and effect.uses == ()
    assert effect.unresolved and not effect.complete


def test_liveness_reaches_a_fixed_point_across_a_backward_branch():
    ins = [
        LV.Instruction(0, "MOVE", {"rd": 1, "rs1": 0}),
        LV.Instruction(1, "MOVE", {"rd": 2, "rs1": 1}),
        LV.Instruction(2, "WAIT", {"imm": 1}, branch_target=1),
    ]
    effects = tuple(LV.effects_of(i, DIRECTIONS) for i in ins)
    live_in, _ = LV.liveness(ins, effects)
    # x1 is read at the top of the body, so it must be live at the bottom of it -- which one backward
    # sweep would miss, and which is exactly why a loop cannot simply reuse the register.
    assert LV.Access("scalar", 1) in live_in[2]


def test_pressure_reports_unknown_capacity_as_unchecked_not_as_a_pass(program):
    live_in, _ = LV.liveness(program.instructions, program.effects)
    by_file = {p.file: p for p in LV.pressure(live_in)}
    assert by_file["mrf"].capacity is None
    assert by_file["mrf"].fits is None
    assert "NOT checked" in by_file["mrf"].claim()


def test_pressure_against_a_known_capacity_decides(program):
    live_in, _ = LV.liveness(program.instructions, program.effects)
    by_file = {p.file: p for p in LV.pressure(live_in, {"mrf": 1})}
    assert by_file["mrf"].fits is True
    by_file = {p.file: p for p in LV.pressure(live_in, {"mrf": 0})}
    assert by_file["mrf"].fits is False


def test_constant_propagation_kills_a_value_it_cannot_evaluate(program):
    state = LV.constant_state(program.instructions, program.effects,
                              immediate_forms={"MOVE": "imm"}, zero_slot={"scalar": 0})
    # MOVE at index 5 writes scalar[2] but carries no immediate here, so the value is UNKNOWN rather
    # than whatever happened to be there.
    assert state[-1][LV.Access("scalar", 2)] is None
    assert state[-1][LV.Access("scalar", 0)] == 0


def test_a_backward_branch_invalidates_every_propagated_constant():
    ins = [
        LV.Instruction(0, "MOVE", {"rd": 1, "rs1": 0, "imm": 7}),
        LV.Instruction(1, "WAIT", {"imm": 1}, branch_target=0),
        LV.Instruction(2, "MOVE", {"rd": 2, "rs1": 1, "imm": 9}),
    ]
    effects = tuple(LV.effects_of(i, DIRECTIONS) for i in ins)
    state = LV.constant_state(ins, effects, immediate_forms={"MOVE": "imm"})
    assert state[0][LV.Access("scalar", 1)] == 7
    assert LV.Access("scalar", 1) not in state[2]        # invalidated by the backward branch


# ---------------------------------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------------------------------
def test_read_after_write_edges_follow_the_measured_direction(dag):
    raw = {(e.src, e.dst) for e in dag.edges if e.kind == DG.RAW}
    assert (0, 2) in raw, "the multiply reads the tile the load wrote"
    assert (2, 4) in raw, "the readout reads the accumulator the multiply wrote"


def test_a_declared_wait_is_a_KNOWN_separation_and_an_undeclared_one_is_not(dag):
    by_src = {e.src: e for e in dag.edges if e.kind == DG.RAW}
    assert by_src[0].known and by_src[0].cycles == 10
    assert "declared" in by_src[0].edge_class


def test_an_unpriced_separation_is_counted_by_class_never_given_a_latency():
    ins = [LV.Instruction(0, "LOAD", {"vd": 0, "rs1": 1}),
           LV.Instruction(1, "MUL", {"vd": 0, "vs1": 0, "vs2": 0})]
    effects = tuple(LV.effects_of(i, DIRECTIONS) for i in ins)
    dag = DG.build_dag(ins, effects, issue=ISSUE, stall_mnemonic="WAIT", roles=ROLES)
    exposed = dag.exposed_classes()
    # two operands reading the same value is ONE dependence, so the class is demanded once
    assert exposed.get("separation.memory") == 1
    assert all(e.cycles is UNKNOWN or e.known for e in dag.edges)
    assert DG.critical_path(dag).complete is False


def test_the_critical_path_is_a_lower_bound_on_the_ordering_the_machine_runs(dag):
    cp = DG.critical_path(dag)
    emitted = DG.makespan(dag, list(range(len(dag.instructions))))
    assert cp.cycles <= emitted
    # the independent move is off the dependence chain, so the bound is strictly below the schedule
    assert cp.cycles < emitted


def test_an_order_that_violates_a_dependence_is_refused_not_priced(dag):
    order = [2, 0, 1, 3, 4, 5]
    with pytest.raises(ValueError):
        DG.makespan(dag, order)


def test_moving_work_into_a_wait_shadow_is_credited_against_the_wait(dag):
    emitted = DG.makespan(dag, [0, 1, 2, 3, 4, 5])
    hoisted = DG.makespan(dag, [0, 5, 1, 2, 3, 4])
    assert hoisted < emitted, "the independent instruction should be absorbed by the declared wait"


# ---------------------------------------------------------------------------------------------------
# the composed bound and the ranking
# ---------------------------------------------------------------------------------------------------
def test_a_bound_with_an_exposed_unknown_never_reports_a_total():
    ins = [LV.Instruction(0, "LOAD", {"vd": 0, "rs1": 1}),
           LV.Instruction(1, "MUL", {"vd": 0, "vs1": 0, "vs2": 0})]
    effects = tuple(LV.effects_of(i, DIRECTIONS) for i in ins)
    dag = DG.build_dag(ins, effects, issue=ISSUE, stall_mnemonic="WAIT", roles=ROLES)
    composed = DG.to_composed(2.0, dag)
    assert composed.cycles is UNKNOWN
    assert composed.partial_cycles == 2.0
    assert composed.unresolved


def test_the_three_candidates_are_reorderings_of_the_same_work_and_rank_exactly(program, dag):
    indices = list(range(len(program.instructions)))
    schedules = DG.candidates_for(dag, indices, stall_mnemonic="WAIT", hoist_role="memory",
                                  roles=ROLES)
    assert set(schedules) >= {"as_emitted", "stalls_tightened"}
    for name, order in schedules.items():
        assert sorted(order) == indices, f"{name} is not a reordering of the same instructions"
    composed = {n: DG.to_composed(DG.makespan(dag, o), dag) for n, o in schedules.items()}
    demands = {n: DG.demands_of(dag) for n in composed}
    order, refusals = differential.rank_schedules(composed, demands=demands)
    assert not refusals, [r.reason for r in refusals]
    names = sorted(composed)
    c = differential.compare(composed[names[0]], composed[names[1]], demands_a=demands[names[0]],
                             demands_b=demands[names[1]], label_a=names[0], label_b=names[1])
    assert c.basis == differential.EXACT
    assert order[0] in schedules


def test_the_report_states_the_gap_between_the_bound_and_a_measurement(program):
    report = DG.analyse_program(program, DIRECTIONS, issue=ISSUE, stall_mnemonic="WAIT",
                                hoist_role="memory", measured_cycles=1000.0)
    assert report["bound_vs_measured"]["verdict"].startswith("consistent")
    assert report["reorder_slack_cycles"] > 0
    assert report["critical_path"]["cycles"] <= report["as_emitted_cycles"]


def test_a_bound_above_its_measurement_is_reported_as_a_falsification(program):
    report = DG.analyse_program(program, DIRECTIONS, issue=ISSUE, stall_mnemonic="WAIT",
                                measured_cycles=1.0)
    assert report["bound_vs_measured"]["verdict"].startswith("FALSIFIED")


def test_the_issue_model_is_measured_from_two_points_and_refuses_a_run_that_did_not_halt():
    def build(n, stall):
        return f"{n}:{stall}"

    costs = {"4:0": 40, "64:0": 100, "4:256": 296}

    model = DG.probe_issue_model(lambda src: costs[src], build, tier="fixture")
    assert model.issue_cycles == pytest.approx(1.0)
    assert model.stall_unit == pytest.approx(1.0)

    with pytest.raises(ValueError):
        DG.probe_issue_model(lambda src: None, build, tier="fixture")
