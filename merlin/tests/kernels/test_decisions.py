"""The belief ladder must REFUSE, or it is documentation rather than a check.

Measured base rate this exists for: of 1509 recorded transform attempts, 203 improved, 119 regressed,
365 failed to compile, 719 were incorrect -- 13.45% improvement. A loop that promotes a corpus
reading straight to a compiler rule is wrong most of the time, and keeps no record of it.
"""
from __future__ import annotations

import pytest

from merlin.kernels.decisions import STATUSES, DecisionRecord, promote


def _rec(**kw) -> DecisionRecord:
    base = dict(scope="kernel", family="contraction", decision="weight_stationary")
    base.update(kw)
    return DecisionRecord(**base)


class TestARungIsRefusedWithoutItsEvidence:
    def test_an_observation_needs_nothing_beyond_itself(self):
        assert _rec(evidence=("k1",)).is_honest()

    def test_a_motif_citing_one_artifact_is_refused(self):
        """NEGATIVE CASE: one artifact is an observation with ambitions, not a generalization."""
        with pytest.raises(ValueError, match="at least two"):
            promote(_rec(evidence=("k1",)), "motif")

    def test_a_motif_citing_the_same_artifact_twice_is_still_one_artifact(self):
        with pytest.raises(ValueError, match="at least two"):
            promote(_rec(evidence=("k1", "k1")), "motif")

    def test_a_hypothesis_with_no_cca_axis_cannot_be_tested_so_is_refused(self):
        with pytest.raises(ValueError, match="CCA axis"):
            promote(_rec(evidence=("k1", "k2")), "hypothesis")

    def test_a_validated_policy_needs_a_control(self):
        """A delta against nothing is not a delta."""
        r = _rec(evidence=("k1", "k2"), cca_axes=("memory.onchip_resident",),
                 measured_cycles=100, delta_vs_control=0.2,
                 measurement_authority="spike", correctness_ok=True)
        with pytest.raises(ValueError, match="matched control"):
            promote(r, "validated_policy")

    def test_a_validated_policy_needs_a_measurement_not_an_expectation(self):
        r = _rec(evidence=("k1", "k2"), cca_axes=("memory.onchip_resident",),
                 control="baseline", measurement_authority="spike", correctness_ok=True)
        with pytest.raises(ValueError, match="MEASURED"):
            promote(r, "validated_policy")

    def test_a_measured_number_must_name_its_substrate(self):
        r = _rec(evidence=("k1", "k2"), cca_axes=("memory.onchip_resident",),
                 control="baseline", measured_cycles=100, delta_vs_control=0.2,
                 correctness_ok=True)
        with pytest.raises(ValueError, match="substrate"):
            promote(r, "validated_policy")

    @pytest.mark.parametrize("gate", [None, False])
    def test_no_speedup_is_credited_without_a_passing_correctness_gate(self, gate):
        """NEGATIVE CASE, both ways: an unrun gate is not a passing one."""
        r = _rec(evidence=("k1", "k2"), cca_axes=("memory.onchip_resident",),
                 control="baseline", measured_cycles=100, delta_vs_control=0.9,
                 measurement_authority="spike", correctness_ok=gate)
        with pytest.raises(ValueError, match="correctness"):
            promote(r, "validated_policy")

    def test_a_fully_evidenced_policy_is_allowed(self):
        r = _rec(evidence=("k1", "k2"), cca_axes=("memory.onchip_resident",),
                 control="baseline", measured_cycles=100, delta_vs_control=0.2,
                 measurement_authority="spike", correctness_ok=True)
        assert promote(r, "validated_policy").status == "validated_policy"


class TestLadderMechanics:
    def test_promote_does_not_demote(self):
        r = promote(_rec(evidence=("k1", "k2")), "motif")
        with pytest.raises(ValueError, match="does not demote"):
            promote(r, "observation")

    def test_an_unknown_status_is_rejected_rather_than_ranked(self):
        assert _rec(status="probably").problems()
        with pytest.raises(ValueError, match="unknown status"):
            promote(_rec(), "very_sure")

    def test_the_ladder_is_ordered_weakest_first(self):
        assert STATUSES == ("observation", "motif", "hypothesis", "validated_policy")

    def test_problems_are_carried_into_the_serialized_form(self):
        """A record written to an artifact must carry its own doubts with it."""
        d = _rec(status="validated_policy").to_dict()
        assert d["problems"], "a dishonest record must not serialize as if it were clean"
