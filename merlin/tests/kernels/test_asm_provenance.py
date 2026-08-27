"""From an emitted instruction to the compiler seam that can change it.

"Where did this assembly come from" has two readings and only one is useful. The literal one — which
pass emitted this byte — names a stage nobody can edit even when you can compute it. The useful one is
that the instruction is here because of a decision, and the decision has an owner:

    instruction -> ROLE -> CCA AXIS -> COMPILER REGION -> the seam that can change it

The last link is what makes it provenance rather than trivia: a divergence traced to a role whose
region has no forkable seam is a finding, not a task, and saying so beats routing it somewhere that
does not exist.
"""
from __future__ import annotations

import pytest

from merlin.kernels import asm_provenance as P
from merlin.kernels import roles as R


class TestTheChainIsComplete:
    def test_every_role_in_the_vocabulary_has_a_chain(self):
        for role in R.ROLES:
            P.provenance_of_role(role)          # raises if the role is unknown

    def test_no_role_is_left_unowned(self):
        """An instruction we can read but cannot attribute to an owner is a dead end: the loop can see
        the divergence and has nowhere to send it."""
        assert P.unowned_roles() == (), f"roles no region governs: {P.unowned_roles()}"

    def test_every_axis_named_is_a_real_classified_field(self):
        """A dead row reads as coverage: the chain would report an axis for a role and the axis would
        resolve to nothing downstream, which looks like provenance and is not."""
        assert P.check_axes_exist() == []

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(KeyError, match="unknown instruction role"):
            P.provenance_of_role("teleport")


class TestActionabilityIsHonest:
    def test_a_gap_seam_is_not_reported_as_actionable(self):
        """Found by running this: the EditPoint attribute is `forkable_now`, and reading a
        non-existent `forkable` with a True default made every declared GAP look like a task — the
        exact inversion this chain exists to prevent."""
        got = P.provenance_of_role("sync")
        assert got.regions == ("hw-sync",)
        assert got.actionable is False
        assert any("stated GAP" in n for n in got.notes)

    def test_a_real_seam_is_reported_as_actionable(self):
        got = P.provenance_of_role("accumulate")
        assert got.actionable and got.regions

    def test_the_reader_has_no_default_for_a_missing_attribute(self):
        import inspect
        src = inspect.getsource(P.provenance_of_role)
        assert 'getattr(ep, "forkable"' not in src, (
            "a default here turns a declared gap into a fork-ready seam")


class TestOpportunitiesComeFromTheAssembly:
    def test_an_unfused_multiply_add_is_detected(self):
        """A real finding on a real kernel: the 'matmul' used vfmul + vfadd, so there genuinely is no
        accumulate — which is the contraction_form divergence, visible only because the roles
        distinguish a multiply-accumulate from a bare multiply."""
        opps = P.opportunities({"elementwise": 9, "operand_load": 11})
        axes = [o.axis for o in opps]
        assert "compute.contraction_form" in axes
        top = next(o for o in opps if o.axis == "compute.contraction_form")
        assert top.confidence == "high" and top.forkable_now and top.seam

    def test_an_accumulate_with_no_readout_is_flagged(self):
        opps = P.opportunities({"accumulate": 6})
        assert any(o.axis == "compute.accumulator_resident" and o.confidence == "high" for o in opps)

    def test_a_shuffle_dominated_stream_is_flagged(self):
        opps = P.opportunities({"accumulate": 2, "broadcast": 200, "move": 50})
        assert any(o.axis == "compute.register_block" for o in opps)

    def test_config_churn_is_flagged(self):
        opps = P.opportunities({"config": 20, "accumulate": 4, "readout": 1})
        assert any(o.axis == "dispatch.descriptor_reuse" for o in opps)

    def test_a_healthy_stream_proposes_nothing_spurious(self):
        # A balanced contraction on a lane engine should not trigger the shape rules.
        opps = P.opportunities({"accumulate": 40, "readout": 2, "operand_load": 12, "config": 1},
                               engine="vector", total=100)
        assert [o.axis for o in opps] == [], [o.axis for o in opps]

    def test_every_opportunity_names_a_lever_not_a_metric(self):
        """An opportunity pointing at a METRIC is pointing at a thermometer and calling it a dial.
        Found by running this: the broadcast rule named memory.a_broadcast_vf, which is classified
        METRIC and correctly governed by no region."""
        from merlin.kernels.cca_contract import FIELD_REGISTRY
        seen = []
        for hist in ({"elementwise": 9}, {"accumulate": 6}, {"accumulate": 2, "broadcast": 200},
                     {"config": 20, "accumulate": 4, "readout": 1},
                     {"accumulate": 1, "sync": 9}, {"control": 60}):
            seen += P.opportunities(hist, engine="simt", total=100)
        for o in seen:
            spec = FIELD_REGISTRY.get(o.axis)
            assert spec is not None, f"{o.axis} is not a classified CCA field"
            assert spec.classification != "METRIC", (
                f"{o.axis} is a METRIC: it diagnoses, it is not a dial")

    def test_the_status_separates_three_different_failures(self):
        """'no region governs this', 'the seam is a declared gap' and 'this is a metric' are different
        problems with different fixes; one boolean cannot tell them apart."""
        assert P._seam_for("compute.contraction_form")[2] == "forkable"
        assert P._seam_for("simt.barriers_in_loop")[2] == "seam_is_a_gap"
        assert P._seam_for("memory.a_broadcast_vf")[2] == "metric_not_a_lever"
