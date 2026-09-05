"""Size a capsule by the work it does, not by the shape it declares -- and price it from a free run.

An RTL simulator's time is spent advancing cycles, not contemplating a tensor's declared extent, and
the measurements say so. Over 72 real gemmini certifications, seconds against the metric `cert_cost`
shipped with (`max_operand_elements`) fits at r2 0.20, and the best of five shape candidates
(`output_elements`) only reaches 0.33. Against CYCLES the same data fits at r2 0.91.

That matters beyond accuracy, because cycles are almost free to obtain: the FUNCTIONAL tier costs
0.006-0.008s and reports a cycle count, and across 40 capsules that ran both tiers the cycle-accurate
count tracks it at a stable ratio (6.29 on gemmini). So a capsule can be priced for certification
WITHOUT being certified -- which is what makes "how big may this capsule be" answerable before
committing hours of simulation to find out.

Both fits are kept. A capsule that has never run has no cycles either, and only the shape fit can
speak for it; neither is allowed to pretend to the other's authority.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import cert_cost as CC


def _result(name: str, *, accurate_s=None, accurate_cycles=None, func_cycles=None) -> dict:
    tiers: dict = {}
    if func_cycles is not None:
        tiers["L2"] = {"timing": {"sim_active_s": 0.006}, "cycles": func_cycles,
                       "cycle_accurate": False, "derived_from_rtl": False}
    if accurate_s is not None:
        tiers["L3"] = {"timing": {"sim_active_s": accurate_s}, "cycles": accurate_cycles,
                       "cycle_accurate": True, "derived_from_rtl": True}
    return {"capsule": name, "tiers": tiers}


def _write(tmp_path, results) -> None:
    for i, doc in enumerate(results):
        d = tmp_path / "runs" / f"r{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "capsule_result.json").write_text(json.dumps(doc), encoding="utf-8")


def _linear_corpus(tmp_path, *, intercept=100.0, per_cycle=0.145, ratio=6.0, n=6):
    docs = []
    for i in range(n):
        # Chosen divisible by `ratio` so the functional count is EXACT: an integer truncation here
        # would perturb the measured median ratio and make the assertions below test the fixture's
        # rounding rather than the estimator.
        func = 100 * (i + 1)
        cyc = int(func * ratio)
        docs.append(_result(f"C{i}", accurate_s=intercept + per_cycle * cyc,
                            accurate_cycles=cyc, func_cycles=func))
    _write(tmp_path, docs)


def test_the_fit_recovers_the_line_it_was_given(tmp_path):
    _linear_corpus(tmp_path)
    fit = CC.fit_cycles_for("t", timing_root=tmp_path)
    assert fit is not None
    assert abs(fit.intercept_s - 100.0) < 1e-6
    assert abs(fit.per_cycle_s - 0.145) < 1e-9
    assert fit.r2 > 0.999
    assert abs(fit.functional_ratio - 6.0) < 0.05, fit.functional_ratio
    assert fit.n_ratio_samples == 6


def test_a_functional_run_prices_a_certification_that_never_happened(tmp_path):
    """The cheap path: milliseconds of functional simulation, and no certification at all."""
    _linear_corpus(tmp_path)
    fit = CC.fit_cycles_for("t", timing_root=tmp_path)
    # A capsule the functional tier says runs 1000 cycles -> ~6000 cycle-accurate cycles.
    est, basis = CC.predict_seconds_from_functional_cycles(fit, 1000)
    assert est is not None
    assert abs(est - (100.0 + 0.145 * 6000)) < 1.0, est
    assert "measured ratio" in basis and "6.00" in basis


def test_it_refuses_to_scale_when_no_capsule_ran_both_tiers(tmp_path):
    """An unmeasured ratio must be refused, not assumed -- the whole estimate hangs off it."""
    docs = [_result(f"C{i}", accurate_s=100.0 + 0.145 * (500 * (i + 1)),
                    accurate_cycles=500 * (i + 1)) for i in range(6)]
    _write(tmp_path, docs)
    fit = CC.fit_cycles_for("t", timing_root=tmp_path)
    assert fit is not None and fit.functional_ratio is None
    est, basis = CC.predict_seconds_from_functional_cycles(fit, 1000)
    assert est is None
    assert "unmeasured and cannot be assumed" in basis


def test_a_functional_only_corpus_yields_no_cycle_cost_fit(tmp_path):
    """0.006s per run must never become a seconds-per-cycle slope."""
    _write(tmp_path, [_result(f"C{i}", func_cycles=100 * (i + 1)) for i in range(6)])
    assert CC.fit_cycles_for("t", timing_root=tmp_path) is None


def test_too_few_points_is_no_fit(tmp_path):
    _linear_corpus(tmp_path, n=2)
    assert CC.fit_cycles_for("t", timing_root=tmp_path) is None, (
        "two points define a line through anything")


def test_the_budget_answers_in_cycles_and_clamps_to_the_evidence(tmp_path):
    _linear_corpus(tmp_path)
    fit = CC.fit_cycles_for("t", timing_root=tmp_path)
    # (300 - 100) / 0.145 = 1379 cycles, inside the measured 500..3000 range.
    assert CC.max_cycles_within(fit, 300.0) == int((300.0 - 100.0) / 0.145)  # 1379 cycles
    assert CC.max_cycles_within(fit, 50.0) is None, "under the floor, no capsule of any size fits"
    huge = CC.max_cycles_within(fit, 10_000_000.0)
    assert huge == int(fit.cycles_max * 2), "past the evidence the line is an opinion; clamp it"


def test_the_deepest_accurate_tier_is_the_cost_when_several_ran(tmp_path):
    """A target may certify on more than one rung; the binding cost is the longest."""
    doc = {"capsule": "multi", "tiers": {
        "L3": {"timing": {"sim_active_s": 100.0}, "cycles": 1000,
               "cycle_accurate": True, "derived_from_rtl": True},
        "L4": {"timing": {"sim_active_s": 900.0}, "cycles": 1000,
               "cycle_accurate": True, "derived_from_rtl": True}}}
    _write(tmp_path, [doc])
    recs = CC._cycle_records("t", root=tmp_path)
    assert recs[("multi", CC.UNKNOWN_ENGINE)]["seconds"] == 900.0


# --- the engine is part of what identifies a cost sample ------------------------------------------
# Two elaborated-RTL engines answer the SAME capsule at the SAME fidelity and are not interchangeable
# as cost samples. Measured on this repo's largest corpus: 0.229 s/cycle against 0.0035 s/cycle, 65x
# apart with no overlap, so a line through both describes neither machine -- 88 pooled samples fit at
# r2 0.111 where the same records split by engine fit at 0.82 and 0.61.

def _engine_corpus(tmp_path):
    """One corpus, two engines, each a clean line -- and the two lines wildly apart."""
    docs = []
    for i in range(6):
        cyc = 500 * (i + 1)
        docs.append({"capsule": f"S{i}", "tiers": {"L3": {
            "timing": {"sim_active_s": 50.0 + 0.23 * cyc}, "cycles": cyc,
            "cycle_accurate": True, "derived_from_rtl": True,
            "evidence": "rtl_verilator_console.log"}}})
        docs.append({"capsule": f"F{i}", "tiers": {"L3": {
            "timing": {"sim_active_s": 15.0 + 0.0035 * cyc}, "cycles": cyc,
            "cycle_accurate": True, "derived_from_rtl": True, "engine": "gsim"}}})
    _write(tmp_path, docs)


def test_two_engines_are_never_pooled_into_one_law(tmp_path):
    """THE REGRESSION. ``_cycle_records`` keyed on the capsule NAME and never read the engine, so both
    engines' seconds were fitted as one line -- and the key also silently discarded a sample whenever
    one capsule had run on both."""
    _engine_corpus(tmp_path)
    fits = CC.fits_cycles_for("t", timing_root=tmp_path)
    assert set(fits) == {"verilator", "gsim"}, fits
    assert fits["verilator"].per_cycle_s == pytest.approx(0.23, rel=1e-6)
    assert fits["gsim"].per_cycle_s == pytest.approx(0.0035, rel=1e-6)
    assert fits["verilator"].r2 > 0.999 and fits["gsim"].r2 > 0.999

    # THE FALSIFIER: the same seconds fitted with no engine axis describe neither engine, and the
    # pooled r2 collapses. This is the number that was measured on the real corpus (0.111).
    pooled = _pooled_fit(tmp_path)
    assert pooled["r2"] < 0.5, "a pooled fit through two 65x-apart engines cannot explain the variance"
    assert pooled["per_cycle_s"] != pytest.approx(0.23, rel=1e-2)
    assert pooled["per_cycle_s"] != pytest.approx(0.0035, rel=1e-2)


def test_a_capsule_certified_on_BOTH_engines_keeps_both_samples(tmp_path):
    """Keyed on the name alone, whichever file sorted last silently won -- so the cheap engine's
    evidence disappeared the moment the slow one ran. On the real corpus that cost 22 of 110 records."""
    _write(tmp_path, [
        {"capsule": "shared", "tiers": {"L3": {"timing": {"sim_active_s": 800.0}, "cycles": 1000,
                                               "cycle_accurate": True, "derived_from_rtl": True,
                                               "evidence": "verilator_console.log"}}},
        {"capsule": "shared", "tiers": {"L3": {"timing": {"sim_active_s": 4.0}, "cycles": 1000,
                                               "cycle_accurate": True, "derived_from_rtl": True,
                                               "engine": "gsim"}}}])
    recs = CC._cycle_records("t", root=tmp_path)
    assert {k[1] for k in recs} == {"verilator", "gsim"}
    assert recs[("shared", "verilator")]["seconds"] == 800.0
    assert recs[("shared", "gsim")]["seconds"] == 4.0


def test_the_engine_bucket_says_whether_it_was_STATED_or_INFERRED(tmp_path):
    """An evidence filename is an inference -- the console name comes from a static map a run-time
    engine substitution does not update -- so a bucket resting on one must say so rather than read like
    a statement by the runner."""
    _engine_corpus(tmp_path)
    fits = CC.fits_cycles_for("t", timing_root=tmp_path)
    assert fits["gsim"].engine_basis == "engine", "a stated engine is not an inference"
    assert fits["verilator"].engine_basis == "evidence"
    assert fits["verilator"].to_dict()["engine"] == "verilator"


def test_with_no_engine_named_the_BINDING_bucket_is_returned(tmp_path):
    """Not a line through all of them. This answers "may this capsule be allowed to run here", the
    binding cost is what decides, and under-predicting is how a run gets committed to that never
    finishes."""
    _engine_corpus(tmp_path)
    binding = CC.fit_cycles_for("t", timing_root=tmp_path)
    assert binding is not None and binding.engine == "verilator"
    assert CC.fit_cycles_for("t", engine="gsim", timing_root=tmp_path).engine == "gsim"
    assert CC.fit_cycles_for("t", engine="no_such_engine", timing_root=tmp_path) is None, (
        "an engine with no measured history does not inherit another's law")


def _pooled_fit(tmp_path) -> dict:
    """The fit this module used to produce: every sample, keyed on the capsule name alone."""
    import statistics

    pooled: dict[str, tuple[float, int]] = {}
    for (name, _engine), rec in sorted(CC._cycle_records("t", root=tmp_path).items()):
        pooled[name] = (rec["seconds"], rec["cycles"])       # last one wins, as the old key did
    xs = [c for _s, c in pooled.values()]
    ys = [s for s, _c in pooled.values()]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    icept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys)) / sst
    return {"r2": r2, "per_cycle_s": slope, "n": len(xs)}
