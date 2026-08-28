"""Which substrate may produce which number, per target — declared, never picked by field name.

The beam chose its cycle count by looking for a measurement stamped with one substrate label and its
wall time by looking for another, with a comment saying the choice was "INHERENT to this path, not a
derivable per-target fact". That is fine for one target and wrong for five: a target with no such
substrate silently yields no number, and a number read off the wrong substrate is worse than none
because it gets cited.
"""
from __future__ import annotations

import pytest

from merlin.kernels import measurement as M

_DECL = {"measurement": {"cycles_from": "spike", "wall_from": "k1",
                         "cycles_tier": "functional", "wall_tier": "silicon",
                         "speed_of_light": "sol", "citable_tier": "rtl"}}


class TestUndeclaredIsUnknownNotDefault:
    def test_no_declaration_yields_an_authority_that_answers_nothing(self):
        a = M.authority_for("nobody", {})
        assert a.declared is False and a.cycles_from is None and a.wall_from is None

    def test_it_says_so_rather_than_reporting_zero(self):
        """An undeclared authority must not silently become somebody else's. UNKNOWN and zero are
        different, and only one of them is safe to publish."""
        gaps = M.authority_for("nobody", {}).gaps()
        assert gaps and "UNKNOWN" in gaps[0] and "NOT the same as zero" in gaps[0]

    def test_picking_from_an_undeclared_authority_returns_nothing(self):
        a = M.authority_for("nobody", {})
        assert M.pick([{"target": "spike", "cycles": 5}], a, "cycles") == (None, None)


class TestTheAuthoritativeSubstrateWins:
    def test_cycles_come_from_the_declared_substrate(self):
        a = M.authority_for("t", _DECL)
        ms = [{"target": "k1", "cycles": 999}, {"target": "spike", "cycles": 1234}]
        assert M.pick(ms, a, "cycles") == (1234, "spike")

    def test_it_does_not_fall_back_to_another_substrate_carrying_the_field(self):
        """The measured trap: more than one substrate emits `cycles` while only one is authoritative,
        the other being a timer-derived ESTIMATE. Picking by field name gets the estimate."""
        a = M.authority_for("t", _DECL)
        ms = [{"target": "k1", "cycles": 999}]          # authoritative substrate absent
        assert M.pick(ms, a, "cycles") == (None, None)

    def test_wall_time_has_its_own_authority(self):
        a = M.authority_for("t", _DECL)
        ms = [{"target": "k1", "wall_ns": 500}, {"target": "spike", "cycles": 1234}]
        assert M.pick(ms, a, "wall") == (500, "k1")

    def test_a_target_with_no_wall_authority_reports_the_gap(self):
        a = M.authority_for("t", {"measurement": {"cycles_from": "cyclotron", "wall_from": None,
                                                  "speed_of_light": "sol"}})
        assert any("no wall-time authority" in g for g in a.gaps())
        assert M.pick([{"target": "k1", "wall_ns": 1}], a, "wall") == (None, None)


class TestATierGatesWhatMayBeQuoted:
    def test_a_number_below_the_citable_tier_is_not_a_hardware_result(self):
        """The recorded failure: a completion certificate quoted as an output-equality result, and a
        headline quoted without the tier it was reached at."""
        a = M.authority_for("t", _DECL)
        assert M.citable(a, "functional") is False
        assert M.citable(a, "rtl") is True and M.citable(a, "silicon") is True

    def test_an_unknown_tier_fails_closed(self):
        a = M.authority_for("t", _DECL)
        assert M.citable(a, "vibes") is False

    def test_an_undeclared_authority_can_cite_nothing(self):
        assert M.citable(M.authority_for("nobody", {}), "silicon") is False

    def test_the_tier_order_is_ascending_in_what_it_can_claim(self):
        assert M.TIER_ORDER.index("functional") < M.TIER_ORDER.index("rtl") < M.TIER_ORDER.index("silicon")


class TestTheWholeModelObjective:
    def test_it_is_claimed_fraction_times_attainment(self):
        """Never a bare pass count. A kernel-scoped score can look excellent while the model runs at a
        few percent of peak, because most of the arithmetic was never claimed by the schedule."""
        assert M.whole_model_objective(0.5, 0.5, numerics_ok=True) == 0.25

    def test_it_is_fail_closed_on_numerics(self):
        # A fast wrong answer scores None, not a number with a caveat attached.
        assert M.whole_model_objective(1.0, 1.0, numerics_ok=False) is None

    def test_an_unknown_factor_is_not_one(self):
        assert M.whole_model_objective(None, 0.9, numerics_ok=True) is None
        assert M.whole_model_objective(0.9, None, numerics_ok=True) is None


class TestTheRealTargets:
    @pytest.mark.parametrize("target", ["gemmini", "saturn", "muon"])
    def test_the_declared_targets_have_a_cycle_authority(self, target):
        a = M.authority_for(target)
        if not a.declared:
            pytest.skip(f"{target} contract not resolvable in this checkout")
        assert a.cycles_from, target

    def test_a_target_with_no_silicon_declares_no_wall_authority(self):
        """Rather than pointing wall time at a simulator, which would publish a simulated duration as a
        measured one."""
        a = M.authority_for("muon")
        if not a.declared:
            pytest.skip("muon contract not resolvable")
        assert a.wall_from is None
        assert any("no wall-time authority" in g for g in a.gaps())

    def test_a_cycle_model_target_cannot_cite_an_rtl_claim(self):
        a = M.authority_for("muon")
        if not a.declared:
            pytest.skip("muon contract not resolvable")
        assert M.citable(a, "functional") is False


class TestEveryRegisteredTargetDeclaresAnAuthority:
    """An undeclared authority is not a small gap: `pick` returns nothing, so cycles, wall time and
    attainment are all UNKNOWN for that target. radiance, atlas and rvv were all undeclared — and not
    by oversight. All three resolve a GENERATED contract, and the contract deriver never emitted a
    `measurement` block at all, so no generated target COULD declare one. The block now lives in each
    target's residual (the per-target declaration side-input) and is carried through by the deriver.
    """

    TARGETS = ("gemmini", "muon", "saturn", "radiance", "atlas", "rvv")

    def test_all_six_declare(self):
        from merlin.kernels import measurement as M
        undeclared = [t for t in self.TARGETS if not M.authority_for(t).declared]
        assert not undeclared, f"no measurement authority declared for {undeclared}"

    def test_a_cycle_tier_is_never_stronger_than_its_substrate(self):
        """atlas's cycles come from the arc PROGRAM oracle, which reports `derived_from_rtl: false`.
        Calling that `rtl` because it is the most expensive tier the target has would claim a
        fidelity the substrate does not provide."""
        from merlin.kernels import measurement as M
        a = M.authority_for("atlas")
        assert a.cycles_tier == "cycle_model" and a.citable_tier == "cycle_model"

    def test_an_unimplemented_denominator_is_null_not_a_name(self):
        """A `speed_of_light` NAME that nothing computes makes attainment look declared while it is
        UNKNOWN. radiance and atlas declare null and say so in `gaps()`."""
        from merlin.kernels import measurement as M
        for t in ("radiance", "atlas"):
            a = M.authority_for(t)
            assert a.speed_of_light is None
            assert any("speed-of-light" in g for g in a.gaps()), a.gaps()


class TestAttainmentPicksByAuthorityNotByFieldName:
    def test_the_authoritative_substrate_wins_over_the_first_entry(self):
        from merlin.kernels import measurement as M
        auth = M.MeasurementAuthority(target="t", cycles_from="spike", declared=True)
        meas = [{"target": "k1", "cycles": 999}, {"target": "spike", "cycles": 42}]
        assert M.pick(meas, auth, "cycles") == (42, "spike")

    def test_a_non_authoritative_substrate_is_not_a_fallback(self):
        """NEGATIVE CASE: the authoritative substrate did not report, and another one did. The answer
        is UNKNOWN, not the other one's number."""
        from merlin.kernels import measurement as M
        auth = M.MeasurementAuthority(target="t", cycles_from="spike", declared=True)
        assert M.pick([{"target": "k1", "cycles": 999}], auth, "cycles") == (None, None)
