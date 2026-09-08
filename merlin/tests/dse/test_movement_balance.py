"""Separating a transfer rate from a fixed per-transfer cost, and what the result may claim.

The measured fit this pins is a real one: six pure-movement capsules emitted by one compiler package
and run on GSIM, 40..1280 B, giving 16.119 B/cycle marginal and a 128.5-cycle fixed cost at
r-squared 0.99294. What every test here guards is the gap between that number and the conclusion it
invites -- because the same measurement supports "resnet50 is provably compute-bound" and does NOT
support "tiny_llama is memory-bound", and only the one-sided licence separates them.
"""
from __future__ import annotations

import pytest

from merlin.perf import movement_balance as MB


def _series(pairs, *, engine="gsim", package="pkg"):
    return [MB.MovementSample(f"p{i}", b, c, engine, package)
            for i, (b, c) in enumerate(pairs)]


#: The real measurement, from out/runs/gemmini/movement_balance_probe_20260908.
MEASURED = ((40, 130), (512, 161), (1125, 201), (1200, 206), (1275, 206), (1280, 204))


class TestTheMeasuredSeries:
    def test_it_separates_the_rate_from_the_fixed_cost(self):
        got = MB.fit(_series(MEASURED))
        assert got.derived
        assert got.peak_bytes_per_cycle == pytest.approx(16.119, abs=1e-3)
        assert got.base_latency_cycles == pytest.approx(128.5, abs=0.1)
        assert got.r_squared == pytest.approx(0.99294, abs=1e-4)
        assert got.domain_bytes == (40, 1280) and got.n_distinct_sizes == 6

    def test_the_residuals_travel_with_the_fit(self):
        """r-squared alone hides a systematic curve; the residuals are what show one."""
        got = MB.fit(_series(MEASURED))
        assert len(got.residual_cycles) == len(MEASURED)
        assert max(abs(r) for r in got.residual_cycles) < 5.0

    def test_it_says_the_intercept_dominates_this_domain(self):
        """Which is exactly why the slope is a marginal rate and not an achievable bandwidth."""
        got = MB.fit(_series(MEASURED))
        assert any("intercept dominates" in n for n in got.notes)
        assert any("MARGINAL rate" in n for n in got.notes)


class TestTheRidgeIsOneSided:
    """The whole result. Reading it two-sidedly is the inference this module's history shows failing."""

    def test_the_bound_is_named_an_upper_bound_and_needs_the_compute_peak(self):
        got = MB.fit(_series(MEASURED))
        assert got.macs_per_byte_upper_bound(256.0) == pytest.approx(15.882, abs=1e-3)
        # The compute peak belongs to the target, not to this measurement.
        assert got.macs_per_byte_upper_bound(0.0) is None
        assert got.macs_per_byte_upper_bound(None) is None

    def test_the_licence_states_both_directions_including_the_one_that_proves_nothing(self):
        got = MB.fit(_series(MEASURED))
        assert "provably compute-bound" in got.ridge_licence
        assert "proves NOTHING" in got.ridge_licence

    def test_a_refused_fit_yields_no_bound_at_all(self):
        got = MB.fit(_series(((100, 10),)))
        assert not got.derived and got.macs_per_byte_upper_bound(256.0) is None


class TestWhatTheFitRefuses:
    def test_a_size_that_disagrees_with_itself_is_refused_never_averaged(self):
        """THE CORPUS FAILURE. A1_mvin_mvout at 512 B measured 148 and 1899 cycles.

        Those are different emitted programs at one declared volume, so what varies between them is
        not the transfer size. Averaging would turn a 12.8x disagreement into a plausible number.
        """
        got = MB.fit(_series(((40, 130), (512, 148), (512, 1899), (1280, 204))))
        assert not got.derived
        assert "disagreeing by up to 12.8x" in got.reason
        assert "not the transfer size" in got.reason

    def test_too_few_distinct_sizes_is_refused(self):
        got = MB.fit(_series(((40, 130), (1280, 204))))
        assert not got.derived and "distinct transfer size" in got.reason

    def test_a_narrow_domain_is_refused_even_though_r_squared_would_be_high(self):
        """Over a narrow domain the intercept is measured precisely and the rate is not."""
        got = MB.fit(_series(((1000, 190), (1050, 193), (1100, 196), (1150, 199))))
        assert not got.derived and "under the" in got.reason and "narrow domain" in got.reason

    def test_a_non_positive_slope_is_refused(self):
        """A larger transfer that did not cost more is not measuring a transfer rate."""
        got = MB.fit(_series(((40, 300), (512, 200), (1280, 100))))
        assert not got.derived and "not positive" in got.reason

    def test_a_series_mixing_two_engines_is_refused(self):
        rows = _series(MEASURED)
        rows[2] = MB.MovementSample("p2", 1125, 201, "spike_gemmini_functional", "pkg")
        got = MB.fit(rows, engine="gsim")
        assert not got.derived
        assert "not a cycle count" in got.reason

    def test_a_series_mixing_two_compiler_packages_is_refused(self):
        rows = _series(MEASURED)
        rows[3] = MB.MovementSample("p3", 1200, 206, "gsim", "another_package")
        got = MB.fit(rows, package="pkg")
        assert not got.derived and "two schedules" in got.reason

    def test_a_non_positive_extent_is_refused(self):
        for bad in (((0, 130), (512, 161), (1280, 204)), ((40, 0), (512, 161), (1280, 204))):
            got = MB.fit(_series(bad))
            assert not got.derived and "non-positive extent" in got.reason

    def test_an_empty_series_is_refused_rather_than_yielding_a_zero_rate(self):
        got = MB.fit([])
        assert not got.derived and "no movement samples" in got.reason

    def test_a_refusal_still_reports_what_it_saw(self):
        """So a reader can tell "nobody measured this" from "the series was unusable"."""
        got = MB.fit(_series(((40, 130), (1280, 204))))
        assert got.n_samples == 2 and got.n_distinct_sizes == 2 and len(got.samples) == 2


class TestHarvestingFromCapsuleRuns:
    def _run(self, tmp_path, name, *, cycles, engine="gsim", rtl=True, status="pass",
             opcode="MOVEMENT", shape=(16, 16), dtype="i8"):
        import json
        d = tmp_path / name
        (d / "generated").mkdir(parents=True)
        (d / "capsule_result.json").write_text(json.dumps({
            "tiers": {"L2": {"cycles": 9, "engine": "spike_gemmini_functional",
                             "derived_from_rtl": False, "status": "pass"},
                      "L3": {"cycles": cycles, "engine": engine, "derived_from_rtl": rtl,
                             "status": status}}}), encoding="utf-8")
        (d / "generated" / "command_buffer.json").write_text(json.dumps({
            "tensors": {"X": {"shape": list(shape), "dtype": dtype},
                        "Y": {"shape": list(shape), "dtype": dtype}},
            "commands": [{"opcode": opcode, "operands": {"src": "X", "dst": "Y"}}]}),
            encoding="utf-8")
        return d

    def test_a_passing_cycle_accurate_movement_run_contributes_its_bytes(self, tmp_path):
        self._run(tmp_path, "a", cycles=161, shape=(16, 16))
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert refusals == []
        assert len(samples) == 1
        # src + dst, 256 i8 elements each.
        assert samples[0].bytes_moved == 512 and samples[0].cycles == 161

    def test_a_functional_tier_never_contributes_an_instruction_count_as_cycles(self, tmp_path):
        self._run(tmp_path, "a", cycles=27, engine="spike_gemmini_functional", rtl=False)
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert samples == []
        assert "not a cycle count" in refusals[0]["reason"]

    def test_a_failing_run_is_refused_with_its_status(self, tmp_path):
        self._run(tmp_path, "a", cycles=161, status="fail")
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert samples == [] and "did not compute its declared operation" in refusals[0]["reason"]

    def test_a_program_that_is_not_pure_movement_is_refused(self, tmp_path):
        """Otherwise its compute cycles would be fitted as transfer cost."""
        self._run(tmp_path, "a", cycles=300, opcode="MATMUL")
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert samples == [] and "not all transfer cost" in refusals[0]["reason"]

    def test_an_unpriceable_dtype_is_refused_rather_than_silently_skipped(self, tmp_path):
        self._run(tmp_path, "a", cycles=161, dtype="mystery")
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert samples == [] and "no priceable shape/dtype" in refusals[0]["reason"]

    def test_a_run_with_no_emitted_buffer_is_refused(self, tmp_path):
        import json
        d = tmp_path / "a"
        d.mkdir(parents=True)
        (d / "capsule_result.json").write_text(json.dumps({"tiers": {}}), encoding="utf-8")
        samples, refusals = MB.samples_from_capsule_runs(tmp_path)
        assert samples == [] and "no emitted command buffer" in refusals[0]["reason"]


class TestSaturatedBandwidthIsTwoSidedOnlyWhenItSaturates:
    """The stronger estimator, and the guard that keeps it honest.

    :func:`fit` divides bytes by the whole window, so the fixed per-transfer cost sits in the
    denominator and its slope is a LOWER bound on achievable bandwidth -- licensing only an upper
    bound on the ridge, which can prove a workload compute-bound and can never prove one
    memory-bound. Charging the bytes against the movement engines' own BUSY cycles, from the
    hardware occupancy partition, removes that term. But the result is a PEAK only if the curve has
    flattened, so ``saturated`` is decided from the data.
    """

    #: Measured on GSIM with the full MAIN_* partition: (bytes, DMA-busy cycles).
    MEASURED = ((40, 28), (512, 56), (1125, 75), (1200, 80), (1275, 85), (1280, 80))

    def test_the_measured_curve_saturates_at_the_derived_row_width(self):
        got = MB.saturated_bandwidth(self.MEASURED, structural_width_bytes=16)
        assert got.saturated and got.two_sided
        assert got.bytes_per_busy_cycle == pytest.approx(16.0)
        assert got.plateau_points == 4
        assert got.macs_per_byte(256.0) == pytest.approx(16.0)
        assert any("1.000x the target's derived 16-byte row width" in n for n in got.notes)

    def test_it_agrees_with_the_independent_marginal_fit(self):
        """Two estimators, different denominators, 0.7% apart -- that is the corroboration."""
        marginal = MB.fit(_series(MEASURED)).peak_bytes_per_cycle
        saturating = MB.saturated_bandwidth(self.MEASURED).bytes_per_busy_cycle
        assert abs(marginal - saturating) / saturating < 0.01

    def test_a_still_rising_curve_refuses_the_two_sided_ridge(self):
        """THE GUARD. Otherwise "the largest transfer we tried" becomes "the machine's peak"."""
        rising = MB.saturated_bandwidth(((40, 28), (512, 56), (1024, 80)),
                                        structural_width_bytes=16)
        assert not rising.saturated and not rising.two_sided
        assert rising.macs_per_byte(256.0) is None
        assert any("NOT saturated" in n and "can never prove one memory-bound" in n
                   for n in rising.notes)

    def test_too_few_observations_cannot_show_a_plateau(self):
        got = MB.saturated_bandwidth(((512, 56), (1280, 80)))
        assert not got.saturated and "distinguishable from a rising curve" in got.notes[0]

    def test_a_matching_rate_at_a_SMALL_size_is_not_saturation(self):
        """A ceiling is reached at the large end; a fast small point is a coincidence of that size."""
        got = MB.saturated_bandwidth(((40, 3), (512, 56), (1024, 100), (1280, 128)))
        assert not got.saturated, "the trailing run must be the plateau, not the best point anywhere"

    def test_non_positive_observations_are_dropped_not_counted(self):
        got = MB.saturated_bandwidth(((0, 10), (512, 0), (1125, 75), (1200, 80), (1275, 85)))
        assert got.domain_bytes == (1125, 1275)

    def test_the_structural_note_says_corroboration_not_derivation(self):
        got = MB.saturated_bandwidth(self.MEASURED, structural_width_bytes=16)
        note = next(n for n in got.notes if "row width" in n)
        assert "CORROBORATION" in note
        assert "wider burst path" in note, "the caveat must travel with the claim"

    def test_it_serialises_with_its_licence_fields(self):
        d = MB.saturated_bandwidth(self.MEASURED, structural_width_bytes=16).to_dict()
        assert d["schema"] == "merlin_saturated_bandwidth_v1"
        assert d["two_sided"] is True and d["plateau_points"] == 4
        assert len(d["observations"]) == 6
