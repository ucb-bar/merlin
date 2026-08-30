"""The zero-cost preflight: what a workload costs, and whether it may be scheduled at all.

Every case here is hermetic -- synthetic machine facts, no target lookup, no oracle, no simulator --
because that is the property the module exists to have. The three things under test are the two hazards
that return a WRONG NUMBER rather than an error (a footprint that aliases inside the runner's finite DRAM
window, and a program longer than instruction memory), and the honesty of the projection built on top of
them (a rate derived from one datapoint is an extrapolation, and a check that could not run is not a pass).
"""
from __future__ import annotations

import pytest

from merlin.perf import preflight as PF
from merlin.perf.oracle_cost import CostLaw, Provenance, Term


# The numbers below are a plausible machine, NOT any target's: a 32x32 array, one-byte operands, a
# two-byte accumulator, a 1 MiB program-runner window and a 32,768-word instruction memory. They are
# fixtures precisely because the module must never know a real one.
def budget(*, window=1 << 20, imem=32768, rows=32, cols=32, ob=1, ab=2) -> PF.MachineBudget:
    return PF.MachineBudget(tile_rows=rows, tile_cols=cols, operand_bytes=ob, accum_bytes=ab,
                            dram_window=window, imem_words=imem, dram_base=0,
                            provenance={"tile": "synthetic fixture"})


def emitter(*, looped=96, prologue=12, k_step=24, epilogue=18) -> PF.EmitterShape:
    return PF.EmitterShape(prologue_words=prologue, k_step_words=k_step, tile_epilogue_words=epilogue,
                           looped_words=looped, provenance="synthetic fixture")


SINGLE_POINT = PF.rate_from_observations([(4394, 3300328)])


class TestRefusesTheOversizedShape:
    """The headline behaviour: a shape too big for the runner's window is refused BEFORE anything runs."""

    def test_a_footprint_that_wraps_the_window_is_refused(self):
        # 512x1024x512 on one-byte operands and a two-byte accumulator is 1,572,864 B against a
        # 1,048,576 B window. The runner would reduce every address modulo the window, alias the
        # tensors onto one another, complete, and report cycles for a computation nobody asked for.
        pf = PF.preflight_matmul("oversized", m=512, k=1024, n=512, budget=budget(),
                                 rate=SINGLE_POINT, emitter=emitter())
        assert not pf.ok
        codes = {r.code for r in pf.refusals}
        assert PF.DRAM_ALIAS in codes
        assert pf.proven_hazards, "a wrapped span is a PROVEN hazard, not an unchecked one"
        assert "wrap" in " ".join(r.detail for r in pf.refusals)

    def test_the_largest_fitting_shape_is_not_refused(self):
        # One tile-step smaller in every direction fits disjointly; the point of the pair is that the
        # boundary is decided by arithmetic on declared shapes, not by running anything.
        pf = PF.preflight_matmul("at_the_ceiling", m=416, k=832, n=416, budget=budget(),
                                 rate=SINGLE_POINT, emitter=emitter())
        assert pf.ok, PF.render(pf)
        assert pf.footprint_bytes == 416 * 832 + 832 * 416 + 416 * 416 * 2
        assert pf.alias.ok and not pf.alias.wrapped and not pf.alias.collisions

    def test_no_oracle_and_no_target_are_touched(self):
        # The whole value of the preflight is that it is free. If it ever needed a simulator or a
        # target lookup it could not be used to decide whether to pay for one.
        pf = PF.preflight_matmul("free", m=64, k=64, n=64, budget=budget(), rate=SINGLE_POINT,
                                 emitter=emitter())
        assert pf.tile_passes == 2 * 2 * 2
        assert pf.projected_cycles == SINGLE_POINT.cycles(8)


class TestImemRefusal:
    def test_an_unrolled_program_over_capacity_is_refused(self):
        # The failure this catches does not raise: the program's tail is simply never loaded, the
        # device runs the prefix and halts, and the cycle count describes the prefix.
        pf = PF.preflight_matmul("unrolled_layer", m=416, k=832, n=416, budget=budget(),
                                 rate=SINGLE_POINT, emitter=emitter(), loops=False)
        assert not pf.ok
        assert PF.IMEM_OVERFLOW in {r.code for r in pf.refusals}
        assert pf.unrolled_words > 32768

    def test_the_same_shape_is_fine_when_the_emitter_loops(self):
        looped = PF.preflight_matmul("looped_layer", m=416, k=832, n=416, budget=budget(),
                                     rate=SINGLE_POINT, emitter=emitter(), loops=True)
        assert looped.ok, PF.render(looped)
        # Both lengths are reported: the gap between them is the argument for the loops.
        assert looped.program_words == 96 and looped.unrolled_words > 100_000

    def test_an_unknown_capacity_refuses_as_UNCHECKED_not_as_a_pass(self):
        pf = PF.preflight_matmul("no_imem_fact", m=64, k=64, n=64, budget=budget(imem=None),
                                 rate=SINGLE_POINT, emitter=emitter())
        assert not pf.ok
        assert PF.IMEM_UNCHECKED in {r.code for r in pf.refusals}
        assert not pf.proven_hazards, "an unchecked capacity is not a demonstrated overflow"
        assert pf.unchecked

    def test_no_program_length_refuses_rather_than_assuming_it_fits(self):
        pf = PF.preflight_matmul("no_emitter", m=64, k=64, n=64, budget=budget(), rate=SINGLE_POINT)
        assert not pf.ok and PF.IMEM_UNCHECKED in {r.code for r in pf.refusals}


class TestUncheckedIsNeverAPass:
    def test_an_unknown_dram_window_refuses(self):
        pf = PF.preflight_matmul("no_window_fact", m=64, k=64, n=64, budget=budget(window=None),
                                 rate=SINGLE_POINT, emitter=emitter())
        assert not pf.ok
        assert PF.DRAM_WINDOW_UNKNOWN in {r.code for r in pf.refusals}
        assert not pf.proven_hazards

    def test_a_partial_tile_is_refused_before_a_footprint_is_even_computed(self):
        pf = PF.preflight_matmul("ragged", m=33, k=64, n=64, budget=budget(), rate=SINGLE_POINT,
                                 emitter=emitter())
        assert not pf.ok
        assert [r.code for r in pf.refusals] == [PF.PARTIAL_TILE]
        assert pf.tile_passes is None and pf.projected_cycles is None

    def test_a_missing_rate_refuses_instead_of_projecting_zero(self):
        pf = PF.preflight_matmul("unpriced", m=64, k=64, n=64, budget=budget(), emitter=emitter())
        assert PF.RATE_UNKNOWN in {r.code for r in pf.refusals}
        assert pf.projected_cycles is None, "UNKNOWN must never be readable as 0"


class TestRateHonesty:
    def test_one_datapoint_is_an_extrapolation_and_says_so(self):
        r = SINGLE_POINT
        assert r.basis is PF.RateBasis.SINGLE_POINT_EXTRAPOLATION
        assert r.is_extrapolation
        assert r.fixed is None, "one point cannot separate a fill intercept from a rate"
        assert 751.0 < r.per_tile_pass < 751.2

    def test_two_distinct_points_fit_a_rate_AND_an_intercept(self):
        # cycles = 500 + 700*passes, exactly.
        r = PF.rate_from_observations([(10, 7500), (100, 70500)])
        assert r.basis is PF.RateBasis.FITTED and not r.is_extrapolation
        assert r.per_tile_pass == pytest.approx(700.0)
        assert r.fixed == pytest.approx(500.0)
        assert r.cycles(50) == 35500

    def test_several_observations_at_ONE_tile_count_still_cannot_fit_an_intercept(self):
        r = PF.rate_from_observations([(8, 8000), (8, 8200), (8, 7800)])
        assert r.basis is PF.RateBasis.SINGLE_POINT_EXTRAPOLATION and r.fixed is None
        assert r.n_points == 3

    def test_no_observation_is_UNKNOWN_never_zero(self):
        r = PF.rate_from_observations([])
        assert r.basis is PF.RateBasis.UNKNOWN
        assert r.per_tile_pass is None and r.cycles(100) is None

    def test_a_projection_reports_how_far_past_the_evidence_it_reaches(self):
        assert SINGLE_POINT.reach(4394) == pytest.approx(1.0)
        assert SINGLE_POINT.reach(43940) == pytest.approx(10.0)
        rendered = PF.render(PF.preflight_matmul("big", m=416, k=832, n=416, budget=budget(),
                                                 rate=SINGLE_POINT, emitter=emitter()))
        assert "EXTRAPOLATION" in rendered


class TestProjectedWall:
    """The wall projection must go through the two-term law: a cycles-only fit charges the program-load
    cost to the cycles and overstated one tier's marginal rate by 1.77x at r2=0.97."""

    def law(self, *, per_cycle: float, per_word: float, fixed: float = 1.0) -> CostLaw:
        return CostLaw(
            substrate="fixture", concurrency=1,
            fixed=Term("fixed", fixed, "s", Provenance.MEASURED, "floor probe", n=3,
                       domain=(1.0, 1.0)),
            per_cycle=Term("per_cycle", per_cycle, "s/cycle", Provenance.MEASURED, "cycle probe", n=4,
                           domain=(0.0, 4_000_000.0)),
            per_word=Term("per_word", per_word, "s/word", Provenance.MEASURED, "halt-first probe", n=4,
                          domain=(0.0, 200_000.0)))

    def test_both_terms_reach_the_projection(self):
        law = self.law(per_cycle=1e-4, per_word=1e-3)
        pf = PF.preflight_matmul("priced", m=416, k=832, n=416, budget=budget(), rate=SINGLE_POINT,
                                 emitter=emitter(), laws={"L3": law})
        est = pf.wall["L3"]
        assert set(est.by_term) == {"fixed", "per_cycle", "per_word"}
        assert est.seconds == pytest.approx(1.0 + 1e-4 * pf.projected_cycles + 1e-3 * 96)
        assert not est.is_lower_bound

    def test_a_law_missing_the_word_term_projects_a_LOWER_BOUND(self):
        law = CostLaw(
            substrate="fixture", concurrency=1,
            fixed=Term("fixed", 1.0, "s", Provenance.MEASURED, "floor", n=1, domain=(1.0, 1.0)),
            per_cycle=Term("per_cycle", 1e-4, "s/cycle", Provenance.MEASURED, "cycle probe", n=4,
                           domain=(0.0, 4_000_000.0)),
            per_word=Term("per_word", None, "s/word", Provenance.UNKNOWN, "no load probe was run"))
        pf = PF.preflight_matmul("half_priced", m=64, k=64, n=64, budget=budget(), rate=SINGLE_POINT,
                                 emitter=emitter(), laws={"L3": law})
        assert pf.wall["L3"].is_lower_bound
        assert "per_word" in pf.wall["L3"].excluded

    def test_the_cheapest_tier_is_the_one_a_budget_should_buy_first(self):
        laws = {"cheap": self.law(per_cycle=1e-6, per_word=1e-5),
                "dear": self.law(per_cycle=1e-3, per_word=1e-3)}
        pf = PF.preflight_matmul("two_tiers", m=64, k=64, n=64, budget=budget(), rate=SINGLE_POINT,
                                 emitter=emitter(), laws=laws)
        assert pf.cheapest_tier == "cheap"

    def test_no_wall_is_projected_without_a_cycle_projection(self):
        pf = PF.preflight_matmul("unpriced", m=64, k=64, n=64, budget=budget(), emitter=emitter(),
                                 laws={"L3": self.law(per_cycle=1e-4, per_word=1e-3)})
        assert pf.wall == {}


class TestUsefulBytesFromDeclaredShapes:
    """A capsule DECLARES inputs[].shape and inputs[].dtype, which is what makes useful_bytes derivable
    instead of hand-entered -- the open half of the amplification ratio."""

    def test_declared_tensors_become_sized_operands(self):
        specs = PF.operands_from_declaration(
            [{"name": "X", "role": "input", "shape": [16, 16], "dtype": "bf16"},
             {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "bf16"},
             {"name": "B", "role": "bias", "shape": [16], "dtype": "bf16"}],
            element_bytes_of=lambda d: {"bf16": 2}[d])
        assert [s.nbytes for s in specs] == [512, 512, 32]

    def test_a_declaration_with_no_shape_raises_rather_than_sizing_it_as_nothing(self):
        with pytest.raises(ValueError):
            PF.operands_from_declaration([{"name": "X", "role": "input", "dtype": "bf16"}],
                                         element_bytes_of=lambda d: 2)

    def test_useful_bytes_come_from_the_declaration_and_the_footprint_from_the_placement(self):
        specs = PF.operands_from_declaration(
            [{"name": "X", "role": "input", "shape": [64, 64], "dtype": "i8"},
             {"name": "Y", "role": "output", "shape": [64, 64], "dtype": "i32"}],
            element_bytes_of=lambda d: {"i8": 1, "i32": 4}[d])
        pf = PF.preflight_operands("declared", specs, budget=budget(), program_words=40)
        assert pf.useful_bytes == 64 * 64 * 1 + 64 * 64 * 4
        assert pf.footprint_bytes == pf.useful_bytes
        assert pf.alias.ok

    def test_placements_are_aligned_and_disjoint(self):
        specs = (PF.TensorSpec("a", (10,), 1.0), PF.TensorSpec("b", (10,), 1.0))
        placed = PF.place_operands(specs, origin=0, align=64)
        assert [p.base for p in placed] == [0, 64]


class TestSerialization:
    def test_the_record_says_out_loud_that_its_cycles_are_an_extrapolation(self):
        d = PF.preflight_matmul("rec", m=64, k=64, n=64, budget=budget(), rate=SINGLE_POINT,
                                emitter=emitter()).as_dict()
        assert d["cycles_are_an_extrapolation"] is True
        assert d["rate"]["basis"] == "SINGLE_POINT_EXTRAPOLATION"
        assert d["ok"] is True

    def test_a_refusal_survives_serialization_with_its_proven_flag(self):
        d = PF.preflight_matmul("rec", m=512, k=1024, n=512, budget=budget(), rate=SINGLE_POINT,
                                emitter=emitter()).as_dict()
        assert d["ok"] is False
        assert any(r["code"] == PF.DRAM_ALIAS and r["proven"] for r in d["refusals"])
