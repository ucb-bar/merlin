"""Confronting the dependence graph with a measured trace: what one run can and cannot settle."""
from __future__ import annotations

from dataclasses import dataclass

from merlin.perf.decompose import UNKNOWN
from merlin.perf.deps.measured import (
    issue_times,
    measured_separations,
    price_unknown_classes,
    time_attribution,
)


@dataclass(frozen=True)
class _E:
    src: int
    dst: int
    kind: str = "raw"
    cycles: object = UNKNOWN
    edge_class: str = "sep.x"

    @property
    def known(self):
        return self.cycles is not UNKNOWN


class _Dag:
    def __init__(self, edges, n=8):
        self.edges = tuple(edges)
        self.instructions = tuple(range(n))


class TestIssueTimes:
    def test_an_instruction_issues_when_the_counter_arrives(self):
        # Cycles where the counter does not move are a stall, not a second issue.
        assert issue_times([0, 0, 0, 1, 1, 2]) == {0: (0,), 1: (3,), 2: (5,)}

    def test_a_revisited_address_issues_again(self):
        assert issue_times([0, 1, 0, 1]) == {0: (0, 2), 1: (1, 3)}


class TestFalsification:
    def test_a_weight_larger_than_a_correct_run_used_is_falsified(self):
        # The one direction a single run can refute the model outright.
        dag = _Dag([_E(0, 1, cycles=32.0)])
        obs, skipped = measured_separations(dag, issue_times([0, 1]))
        assert skipped == []
        assert obs[0].measured == 1 and obs[0].falsified is True

    def test_a_weight_the_run_respected_is_not_falsified(self):
        dag = _Dag([_E(0, 1, cycles=2.0)])
        obs, _ = measured_separations(dag, issue_times([0, 0, 0, 1]))
        assert obs[0].measured == 3 and obs[0].falsified is False
        assert obs[0].slack == 1.0

    def test_an_unpriced_edge_is_never_falsified(self):
        # UNKNOWN makes no claim, so a measurement cannot contradict it.
        dag = _Dag([_E(0, 1)])
        obs, _ = measured_separations(dag, issue_times([0, 1]))
        assert obs[0].predicted is UNKNOWN and obs[0].falsified is False
        assert obs[0].slack is None

    def test_an_edge_whose_endpoint_never_ran_is_skipped_not_passed(self):
        # An unexercised edge is a hole in the validation, not a passing edge.
        dag = _Dag([_E(0, 5, cycles=99.0)])
        obs, skipped = measured_separations(dag, issue_times([0, 1]))
        assert obs == [] and len(skipped) == 1 and "never issued" in skipped[0]

    def test_the_tightest_separation_is_the_one_reported(self):
        # Across repeats the minimum is the tightest the machine was seen to accept.
        dag = _Dag([_E(0, 1)])
        obs, _ = measured_separations(dag, {0: (0, 100), 1: (5, 102)})
        assert obs[0].measured == 2
        assert obs[0].all_measured == (2, 5, 102)


class TestPricing:
    def test_an_unknown_class_is_narrowed_from_above(self):
        dag = _Dag([_E(0, 1, edge_class="sep.a"), _E(1, 2, edge_class="sep.a")])
        obs, _ = measured_separations(dag, issue_times([0, 0, 1, 2, 2, 2]))
        pr = price_unknown_classes(obs)
        assert pr["sep.a"].n_edges == 2
        assert pr["sep.a"].tightest == 1 and pr["sep.a"].loosest == 2
        assert "UPPER bound" in pr["sep.a"].basis

    def test_a_priced_edge_contributes_nothing_to_the_pricing(self):
        dag = _Dag([_E(0, 1, cycles=1.0, edge_class="sep.a")])
        obs, _ = measured_separations(dag, issue_times([0, 1]))
        assert price_unknown_classes(obs) == {}


class TestTimeAttribution:
    def test_every_cycle_is_charged_to_the_instruction_it_waited_on(self):
        t = time_attribution([0, 0, 0, 1, 2, 2])
        assert t.by_instruction == {0: 3, 1: 1, 2: 2}
        assert sum(t.by_instruction.values()) == t.total_cycles

    def test_the_costliest_instructions_come_first(self):
        t = time_attribution([0, 0, 0, 1, 2, 2])
        assert t.top[0] == (0, 3)
        assert t.concentration(1) == 0.5

    def test_mnemonics_aggregate_when_the_program_is_supplied(self):
        class I:
            def __init__(self, m): self.mnemonic = m
        t = time_attribution([0, 0, 1, 2], [I("stall"), I("add"), I("stall")])
        assert t.by_mnemonic == {"stall": 3, "add": 1}

    def test_an_empty_run_reports_no_concentration_rather_than_dividing_by_zero(self):
        assert time_attribution([]).concentration() == 0.0


class TestCounterOffset:
    """A traced counter is a pipeline register; reading it as the executing index mis-attributes."""

    def test_the_lead_is_derived_from_which_dwells_moved(self):
        from merlin.perf.deps.measured import derive_counter_offset
        # Instructions 4 and 8 got cheaper; the counter values that moved are 6 and 10.
        off = derive_counter_offset({6: 33, 10: 33}, {6: 129, 10: 129}, [4, 8])
        assert off.slots == 2 and off.established is True

    def test_a_partial_match_is_not_established(self):
        from merlin.perf.deps.measured import derive_counter_offset
        off = derive_counter_offset({6: 33, 99: 5}, {6: 129, 99: 9}, [4, 8])
        assert off.matched < off.total and off.established is False

    def test_no_change_pins_nothing(self):
        from merlin.perf.deps.measured import derive_counter_offset
        off = derive_counter_offset({1: 2}, {1: 2}, [])
        assert off.established is False and "nothing pins" in off.detail

    def test_the_offset_shifts_issue_times_onto_the_executing_instruction(self):
        assert issue_times([5, 5, 6], offset=2) == {3: (0,), 4: (2,)}

    def test_an_unapplied_offset_manufactures_a_falsification(self):
        # The regression this exists for: with the counter read raw, a correctly separated pair
        # looks like it ran 2 cycles apart and refutes a weight it actually respected.
        dag = _Dag([_E(2, 4, cycles=32.0)], n=8)
        pcs = [2, 3] + [4] * 40 + [5, 6]                 # the counter leads the stall by 2 slots
        raw, _ = measured_separations(dag, issue_times(pcs))
        assert raw[0].measured == 2 and raw[0].falsified is True

        aligned, _ = measured_separations(dag, issue_times(pcs, offset=2))
        assert aligned[0].measured == 41 and aligned[0].falsified is False
