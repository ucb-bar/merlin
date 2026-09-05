"""A per-capsule oracle-tier ceiling: it fires, it refuses when unpriced, and it is always RECORDED.

Certification cost grows with a capsule's depth and it is measured, not guessed: fitted from the runs
already on disk, one elaborated-RTL engine costs ~14s + 0.0047s/cycle and another ~134s + 0.069s/cycle
(``merlin.targetgen.cert_cost``). At the second rate a 28k-cycle capsule is ~35 minutes and one twice
as deep is over an hour, so a corpus that grows deep capsules needs a ceiling or a single grade becomes
unaffordable -- and an unaffordable grade does not announce itself, it presents as a grade that is
still running.

Five properties, because each is a distinct way the mechanism could be wrong:

1. **It fires.** A declared ceiling declines the deeper tier, and a derived one declines a capsule the
   measured cost prices above the declared budget.
2. **It refuses rather than guessing.** A capsule with no measured cost basis is UNKNOWN -- not
   "affordable" (which schedules an unaffordable grade) and not "capped" (which drops a certification
   nobody showed to be expensive).
3. **A capped tier is RECORDED as skipped-with-reason, never absent.** This is the load-bearing one:
   an omitted tier key is what made two model capsules read as though they had never been graded, and
   ``not_run_is_not_pass`` means a tier with no record is not evidence.
4. **A ceiling on CORRECTNESS never silences a TIMING measurement.** A perf capsule iterating on an
   already-certified compiler screens correctness cheaply and rests on its functional sibling -- but
   its cycle-accurate count is the entire point of the family, and no other rung can supply it. The two
   axes come from the capsule's own frozen acceptance block, and excluding a member from the
   measurement matrix is a separate declaration recorded as a separate fact.
5. **`extends` is verified, not trusted.** A capsule is entitled to rest on a sibling only if that
   sibling actually earned the deeper tier in the run being cited. An unverifiable claim gets its own
   strength, because it is weaker than naming nobody -- it READS as certified.

The ceiling is expressed over the tier NAMES the corpus declares, never over any tier->simulator map,
so it survives a re-mapping of which engine answers on which rung (one rung was retired outright while
this was being written, and nothing here moved).
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import cert_cost as CC
from merlin.targetgen import tier_policy as TP

TARGET = "ceiling_fixture_target"


# --- fixtures: a synthetic measured history, so nothing here depends on a real target -------------

def _write_result(root, capsule, *, cycles, seconds, engine, functional_cycles=None, tier="L3"):
    """One capsule_result.json shaped exactly like the grader writes: per-tier, engine-attributed."""
    tiers = {}
    if functional_cycles:
        tiers["L2"] = {"status": "pass", "cycles": functional_cycles, "cycle_accurate": False,
                       "derived_from_rtl": False, "timing": {"sim_active_s": 0.03}}
    tiers[tier] = {"status": "pass", "cycles": cycles, "cycle_accurate": True,
                   "derived_from_rtl": True, "engine": engine,
                   "timing": {"sim_active_s": seconds}}
    d = root / capsule
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule_result.json").write_text(
        json.dumps({"capsule": capsule, "status": "pass", "tiers": tiers}), encoding="utf-8")


@pytest.fixture
def costly_history(tmp_path):
    """A measured history for one expensive engine: ~130s fixed + ~0.07s/cycle, the rate measured here.

    Six capsules at six distinct cycle counts, which is what a fit is allowed to rest on (two points
    define a line through anything, so :data:`cert_cost.MIN_SAMPLES` is 5).
    """
    root = tmp_path / "costly"
    for i, cycles in enumerate((200, 500, 1000, 2000, 4000, 8000)):
        _write_result(root, f"H{i}_depth_{cycles}", cycles=cycles,
                      seconds=130.0 + 0.07 * cycles, engine="slow_rtl",
                      functional_cycles=cycles // 6)
    CC.reset_cache()
    yield [root]
    CC.reset_cache()


@pytest.fixture
def no_history(tmp_path):
    """A target with nothing on disk. The UNKNOWN case, and the default state of a new target."""
    root = tmp_path / "empty"
    root.mkdir()
    CC.reset_cache()
    yield [root]
    CC.reset_cache()


def _capsule(name="C0_probe", **extra):
    cap = {"name": name, "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    cap.update(extra)
    return cap


def _perf_capsule(name="PK00_k16", *, correctness="L2", timing="L3", **extra):
    """A perf-family capsule, declaring its two axes the way the frozen acceptance block does."""
    cap = _capsule(name, **extra)
    cap["performance"] = {"acceptance": {"evidence": {
        "correctness_tier": correctness, "timing_tier": timing,
        "timing_simulator": "fast_rtl", "correctness_simulator": "functional"}}}
    return cap


# --- 1. the cost model reads the measurement, per engine -----------------------------------------

def test_the_fit_recovers_the_rate_it_was_given(costly_history):
    fit = CC.fit_for(TARGET, "L3", roots=costly_history)
    assert fit is not None, "six samples at six cycle counts must produce a fit"
    assert fit.intercept_s == pytest.approx(130.0, abs=1.0)
    assert fit.per_cycle_s == pytest.approx(0.07, rel=0.02)
    assert fit.engine == "slow_rtl", "the engine bucket key is read off the record, verbatim"
    # The cheap predictor, measured rather than assumed to be 1.0.
    assert fit.functional_ratio == pytest.approx(6.0, rel=0.05)


def test_engine_buckets_are_never_pooled(tmp_path):
    """Two engines on the same rung differ by ~15x in rate; a pooled law describes neither."""
    root = tmp_path / "two_engines"
    for i, cycles in enumerate((200, 500, 1000, 2000, 4000, 8000)):
        _write_result(root, f"S{i}", cycles=cycles, seconds=130.0 + 0.07 * cycles, engine="slow_rtl")
        _write_result(root, f"F{i}", cycles=cycles, seconds=14.0 + 0.005 * cycles, engine="fast_rtl")
    CC.reset_cache()
    try:
        fits = CC.fits_for(TARGET, roots=[root], tier="L3")
        assert {e for _t, e in fits} == {"slow_rtl", "fast_rtl"}
        slow = next(f for f in fits.values() if f.engine == "slow_rtl")
        fast = next(f for f in fits.values() if f.engine == "fast_rtl")
        assert slow.per_cycle_s > 10 * fast.per_cycle_s
        # fit_for takes the BINDING (most expensive) bucket when the engine is not known, because the
        # question it answers is "may this be allowed to run", and the slow engine is what decides.
        assert CC.fit_for(TARGET, "L3", roots=[root]).engine == "slow_rtl"
        assert CC.fit_for(TARGET, "L3", engine="fast", roots=[root]).engine == "fast_rtl"
    finally:
        CC.reset_cache()


# --- 2. it fires ----------------------------------------------------------------------------------

def test_a_declared_ceiling_declines_the_deeper_tier(costly_history):
    # The sibling must actually have earned the deeper tier for the claim to verify -- see
    # test_extends_is_verified_not_trusted. H0_depth_200 passed L3 in the fixture history.
    cap = _capsule(max_oracle_tier="L2", extends="H0_depth_200")
    at_cap = TP.oracle_ceiling(TARGET, cap, "L2", declared_tiers=["L2", "L3"],
                               cost_roots=costly_history)
    assert at_cap.allowed, "the tier AT the ceiling is exactly what the capsule asks for"
    deeper = TP.oracle_ceiling(TARGET, cap, "L3", declared_tiers=["L2", "L3"],
                               cost_roots=costly_history)
    assert not deeper.allowed
    assert deeper.source == TP.SOURCE_DECLARED
    assert deeper.record["max_oracle_tier"] == "L2"
    assert deeper.record["extends"]["extends"] == "H0_depth_200"
    assert deeper.record["extends"]["verified"] is True
    assert deeper.record["extends"]["certified_at_tier"] == "L3"
    assert deeper.record["claim"] == TP.CLAIM_EXTENDS
    assert deeper.record["axis"] == TP.AXIS_CORRECTNESS


def test_a_ceiling_without_extends_is_recorded_as_the_weaker_claim(costly_history):
    """A cap resting on a named sibling and a cap resting on nothing are different claims."""
    with_sibling = TP.oracle_ceiling(TARGET, _capsule(max_oracle_tier="L2", extends="H0_depth_200"),
                                     "L3", declared_tiers=["L2", "L3"], cost_roots=costly_history)
    alone = TP.oracle_ceiling(TARGET, _capsule(max_oracle_tier="L2"), "L3",
                              declared_tiers=["L2", "L3"], cost_roots=costly_history)
    assert with_sibling.record["claim"] == TP.CLAIM_EXTENDS
    assert alone.record["claim"] == TP.CLAIM_SCREENED_ONLY
    assert alone.record["extends"]["extends"] is None
    assert "WEAKER" in alone.reason, "the weaker claim must say so where a reader will see it"


def test_a_derived_ceiling_fires_on_the_measured_rate(costly_history):
    """No declaration at all: the measured cost against a declared budget is what caps it.

    8000 cycles at the fitted ~130s + 0.07s/cycle is ~690s, so a 300s budget declines it while a
    1200s budget buys it. Both the budget and the prediction are named in the reason.
    """
    cap = _capsule("H5_depth_8000")
    declined = TP.oracle_ceiling(TARGET, cap, "L3", declared_tiers=["L2", "L3"],
                                 budget_s=300.0, cost_roots=costly_history)
    assert not declined.allowed
    assert declined.source == TP.SOURCE_DERIVED_BUDGET
    assert declined.record["budget_s"] == 300.0
    assert declined.record["affordability"]["verdict"] == CC.TOO_EXPENSIVE
    assert "300s budget" in declined.reason and "690" in declined.reason.replace(",", "")

    afforded = TP.oracle_ceiling(TARGET, cap, "L3", declared_tiers=["L2", "L3"],
                                 budget_s=1200.0, cost_roots=costly_history)
    assert afforded.allowed, "the same capsule under a budget that covers it is not capped"


def test_a_capsule_never_certified_is_priced_from_its_screen_tier(costly_history):
    """The cheap path: the screen tier costs milliseconds and reports cycles, so a capsule with no
    cycle-accurate history of its own is still priceable -- via the MEASURED functional ratio."""
    aff = CC.affordability(TARGET, "L3", budget_s=300.0, capsule="never_run_here",
                           functional_cycles=1200, roots=costly_history)
    assert aff.verdict == CC.TOO_EXPENSIVE, "1200 screen cycles x ~6 is ~7200 cycles, ~635s"
    assert "measured ratio" in aff.reason


# --- 3. it refuses rather than guessing -----------------------------------------------------------

def test_no_measured_history_is_unknown_not_affordable(no_history):
    aff = CC.affordability(TARGET, "L3", budget_s=300.0, capsule="C0_probe",
                           functional_cycles=100, roots=no_history)
    assert aff.verdict == CC.UNKNOWN
    assert aff.verdict not in (CC.AFFORDABLE, CC.TOO_EXPENSIVE)
    assert "no measured certification cost" in aff.reason


def test_an_unpriced_capsule_fails_closed_and_is_labelled_unknown(no_history):
    """The refusal requirement: neither affordable nor capped. Both mislabels are checked for."""
    ceiling = TP.oracle_ceiling(TARGET, _capsule(), "L3", declared_tiers=["L2", "L3"],
                                budget_s=300.0, cost_roots=no_history)
    assert not ceiling.allowed, "an unpriced capsule must not be bought as though affordable"
    assert ceiling.source == TP.SOURCE_UNPRICED
    assert ceiling.source != TP.SOURCE_DERIVED_BUDGET, "not a budget cap: nothing was measured"
    assert ceiling.record["cost_unknown"] is True
    assert ceiling.record["max_oracle_tier"] is None, "no ceiling was declared; none is invented"
    assert "UNKNOWN" in ceiling.reason
    assert "neither shown to fit the budget nor shown to exceed it" in ceiling.reason


def test_a_capsule_beyond_the_measured_range_is_unknown_not_extrapolated(costly_history):
    """A fit over 200..8000 cycles says nothing about a million. Past the margin: UNKNOWN."""
    aff = CC.affordability(TARGET, "L3", budget_s=300.0, cycles=1_000_000, roots=costly_history)
    assert aff.verdict == CC.UNKNOWN
    assert "beyond the measured range" in aff.reason
    fit = CC.fit_for(TARGET, "L3", roots=costly_history)
    assert fit.predict(1_000_000) is None
    assert fit.predict(8000) == pytest.approx(690.0, rel=0.02)


def test_no_declared_budget_means_no_derived_ceiling(costly_history, monkeypatch):
    """Opt-in, with no default number: a budget nobody declared must not quietly cap anything."""
    monkeypatch.delenv("MERLIN_ORACLE_CEILING_BUDGET_S", raising=False)
    assert TP.ceiling_budget_seconds() is None
    ceiling = TP.oracle_ceiling(TARGET, _capsule("H5_depth_8000"), "L3",
                                declared_tiers=["L2", "L3"], cost_roots=costly_history)
    assert ceiling.allowed, "a capsule priced at ~690s is still bought when no budget is declared"


def test_the_budget_is_a_declared_parameter(monkeypatch):
    monkeypatch.setenv("MERLIN_ORACLE_CEILING_BUDGET_S", "450")
    assert TP.ceiling_budget_seconds() == 450.0
    monkeypatch.setenv("MERLIN_ORACLE_CEILING_BUDGET_S", "0")
    assert TP.ceiling_budget_seconds() is None, "a non-positive budget is no budget, not a zero one"
    monkeypatch.setenv("MERLIN_ORACLE_CEILING_BUDGET_S", "not-a-number")
    assert TP.ceiling_budget_seconds() is None


# --- 4. a capped tier is RECORDED, never absent ---------------------------------------------------

def _tier_result_for(ceiling, tier="L3", mandatory=True):
    """Build the tier record the grader's loop builds, so this asserts on the emitted shape."""
    from merlin.targetgen.capsule_runner import TierResult
    return TierResult(tier, "skipped", mandatory, reason=ceiling.reason,
                      budget_deferred=True, oracle_ceiling=ceiling.record,
                      derived_from_rtl=True)


def test_a_capped_tier_is_skipped_with_a_reason_not_omitted(costly_history):
    """The bug this exists to prevent: the tier key MISSING, so the capsule reads as never graded."""
    ceiling = TP.oracle_ceiling(TARGET, _capsule(max_oracle_tier="L2", extends="H0_depth_200"), "L3",
                                declared_tiers=["L2", "L3"], cost_roots=costly_history)
    row = _tier_result_for(ceiling).to_dict()

    assert row["status"] == "skipped", "recorded as skipped -- never pass, never fail"
    assert row["reason"], "a skip with no reason is indistinguishable from a tier nobody ran"
    assert row["not_run_is_not_pass"] is True
    assert "max_oracle_tier: L2" in row["reason"], "the reason must NAME the ceiling"
    assert "H0_depth_200" in row["reason"], "and what the claim rests on"
    assert row["oracle_ceiling"]["capped_tier"] == "L3"
    assert row["oracle_ceiling"]["source"] == TP.SOURCE_DECLARED


def test_a_derived_cap_records_the_budget_that_set_it(costly_history):
    ceiling = TP.oracle_ceiling(TARGET, _capsule("H5_depth_8000"), "L3",
                                declared_tiers=["L2", "L3"], budget_s=300.0,
                                cost_roots=costly_history)
    row = _tier_result_for(ceiling).to_dict()
    assert row["status"] == "skipped"
    assert row["oracle_ceiling"]["budget_s"] == 300.0, "the budget that set the cap is in the record"
    fit = row["oracle_ceiling"]["affordability"]["fit"]
    assert fit["n_samples"] >= CC.MIN_SAMPLES and fit["measured_range_cycles"] == [200, 8000], (
        "the record carries the evidence the cap rests on, not just a verdict")
    assert "300s budget" in row["reason"]


def test_a_capped_mandatory_tier_can_never_read_as_a_pass(costly_history):
    """A ceiling must not become a way to switch off a failing mandatory oracle.

    ``budget_deferred`` is what the finalizer keys on to downgrade a pass to ``screened_only``, so a
    capped mandatory tier is carried on that same flag deliberately.
    """
    ceiling = TP.oracle_ceiling(TARGET, _capsule(max_oracle_tier="L2"), "L3",
                                declared_tiers=["L2", "L3"], cost_roots=costly_history)
    row = _tier_result_for(ceiling, mandatory=True).to_dict()
    assert row["mandatory"] is True
    assert row["status"] != "pass"
    assert row["budget_deferred"] is True
    assert row.get("not_applicable") is not True, (
        "a cap is an affordability decision, NOT a claim that the tier is inapplicable -- "
        "not_applicable is the one flag that exempts a tier from not_run_is_not_pass")


def test_an_uncapped_tier_carries_no_ceiling_block(costly_history):
    """A capsule nothing declined must be byte-identical to before this mechanism existed."""
    from merlin.targetgen.capsule_runner import TierResult
    ceiling = TP.oracle_ceiling(TARGET, _capsule(), "L3", declared_tiers=["L2", "L3"],
                                cost_roots=costly_history)
    assert ceiling.allowed and ceiling.record is None
    assert "oracle_ceiling" not in TierResult("L3", "pass", True).to_dict()


# --- 5. the ceiling is expressed over tier names, not over a simulator map ------------------------

def test_depth_order_comes_from_the_tier_names_alone():
    """Deliberately independent of which engine serves which rung: that mapping is being changed, and
    a ceiling written against it would silently move with it."""
    assert TP.tier_depth_order(["L3", "L0", "L2", "L1"]) == ["L0", "L1", "L2", "L3"]
    assert TP.tier_depth_order(["L4", "L3"]) == ["L3", "L4"]
    # A cap at L3 declines L4 and permits L3 and L2, whatever simulator any of them resolves to.
    cap = _capsule(max_oracle_tier="L3", extends="H0_depth_200")
    tiers = ["L2", "L3", "L4"]
    assert TP.oracle_ceiling(TARGET, cap, "L2", declared_tiers=tiers).allowed
    assert TP.oracle_ceiling(TARGET, cap, "L3", declared_tiers=tiers).allowed
    assert not TP.oracle_ceiling(TARGET, cap, "L4", declared_tiers=tiers).allowed


def test_the_ceiling_fields_are_schema_valid():
    """``max_oracle_tier`` and ``extends`` must be real declared capsule fields, not invented here."""
    from merlin.common.paths import merlin_dir
    schema = json.loads((merlin_dir() / "contract" / "schemas" / "capsule.schema.json")
                        .read_text(encoding="utf-8"))
    props = schema["properties"]
    assert TP.CEILING_FIELD in props and TP.EXTENDS_FIELD in props
    assert props[TP.CEILING_FIELD]["enum"] == props["required_oracle_tiers"]["items"]["enum"], (
        "a ceiling onto a tier the ladder does not have would leave the capsule demanding everything")


# --- 6. the two axes: a correctness ceiling must not silence a timing measurement ----------------
# Phase 1 established correctness at the cert tier; phase 2 optimises PERFORMANCE on an
# already-certified compiler. So a perf capsule derived from a working functional capsule does not
# re-earn correctness -- it iterates at the cheap tier resting on the sibling's cert. But its whole
# reason for existing is the cycle-accurate COUNT, and the perf family declares the split itself:
# `correctness_tier` on the cheap rung, `timing_tier` on the cert rung. Capping the first must not
# cap the second, or a cell that was measured reads back as though it never was.

def test_the_axis_is_derived_from_the_capsules_own_acceptance_block():
    perf = _perf_capsule(correctness="L2", timing="L3")
    assert TP.declared_axes(perf) == ("L2", "L3")
    assert TP.axis_of(perf, "L3") == TP.AXIS_TIMING
    assert TP.axis_of(perf, "L2") == TP.AXIS_CORRECTNESS
    # A capsule that declares no timing rung has no timing axis to protect: everything is correctness.
    assert TP.axis_of(_capsule(), "L3") == TP.AXIS_CORRECTNESS
    assert TP.declared_axes(_capsule()) == (None, None)


def test_a_correctness_ceiling_never_declines_the_timing_tier(costly_history):
    """The load-bearing separation. `correctness_tier: L2` caps correctness; L3 is still BOUGHT,
    because the perf claim needs the cycle count and nothing else can supply it."""
    perf = _perf_capsule("PK03_k128", correctness="L2", timing="L3", extends="H0_depth_200")
    timing = TP.oracle_ceiling(TARGET, perf, "L3", declared_tiers=["L2", "L3"],
                               cost_roots=costly_history)
    assert timing.allowed, (
        "L3 is this capsule's declared TIMING rung; a correctness ceiling must not decline it, or a "
        "measured cycle count reads back as a cell that was never measured")
    assert timing.axis == TP.AXIS_TIMING


def test_the_acceptance_blocks_correctness_tier_is_honoured_as_a_ceiling(costly_history):
    """The perf family already declares its correctness ceiling; it is read, not duplicated.

    With no `max_oracle_tier` at all, a tier that is DEEPER than the declared `correctness_tier` and
    is NOT the timing rung is declined -- and the record says the declaration came from the
    acceptance block rather than from the general field.
    """
    perf = _perf_capsule("PK03_k128", correctness="L2", timing="L3", extends="H0_depth_200")
    deeper_non_timing = TP.oracle_ceiling(TARGET, perf, "L4", declared_tiers=["L2", "L3", "L4"],
                                          cost_roots=costly_history)
    assert not deeper_non_timing.allowed
    assert deeper_non_timing.source == TP.SOURCE_DECLARED_ACCEPTANCE
    assert deeper_non_timing.record["axis"] == TP.AXIS_CORRECTNESS
    assert deeper_non_timing.record["max_oracle_tier"] == "L2"


def test_a_timing_exclusion_is_its_own_declaration_and_says_so(costly_history):
    """Dropping a member from the measurement matrix is a separate field and a separate record.

    `max_timing_tier` is the only thing that declines the timing rung, and when it does the record
    carries `measurement_excluded` -- so a reader concludes "no cycle count is claimed here", never
    "correctness was not certified".
    """
    perf = _perf_capsule("PR08_spills_k16384", correctness="L2", timing="L3",
                         max_timing_tier="L2", extends="H0_depth_200")
    excluded = TP.oracle_ceiling(TARGET, perf, "L3", declared_tiers=["L2", "L3"],
                                 cost_roots=costly_history)
    assert not excluded.allowed
    assert excluded.axis == TP.AXIS_TIMING
    assert excluded.record["measurement_excluded"] is True
    assert "MEASUREMENT matrix" in excluded.reason
    assert "NOT a correctness ceiling" in excluded.reason
    # And the correctness ceiling on the same capsule is a DIFFERENT field, so declaring one of them
    # can never be mistaken for declaring the other.
    assert TP.declared_ceiling(perf, TP.AXIS_TIMING)[0] == "L2"
    assert TP.declared_ceiling(perf, TP.AXIS_CORRECTNESS)[0] == "L2"
    assert TP.TIMING_CEILING_FIELD != TP.CEILING_FIELD


def test_a_derived_cap_on_the_timing_rung_is_still_labelled_a_measurement_exclusion(costly_history):
    """Cost can decline a timing rung too -- 8000 cycles is ~690s against a 300s budget -- but the
    record must not let that read as an uncertified correctness claim OR as a measured cell."""
    perf = _perf_capsule("H5_depth_8000", correctness="L2", timing="L3")
    declined = TP.oracle_ceiling(TARGET, perf, "L3", declared_tiers=["L2", "L3"],
                                 budget_s=300.0, cost_roots=costly_history)
    assert not declined.allowed and declined.axis == TP.AXIS_TIMING
    assert declined.record["measurement_excluded"] is True
    assert "MEASUREMENT matrix" in declined.reason
    assert "not a correctness verdict" in declined.reason


# --- 7. `extends` is verified, not trusted -------------------------------------------------------
# A capsule claiming to rest on a sibling's certification is entitled to that claim only if the named
# sibling actually earned the deeper tier in the run being cited. An unverifiable `extends` is WEAKER
# than no `extends`, because it READS as certified -- so it can never be recorded as the certified one.

def test_extends_is_verified_against_the_siblings_own_result(costly_history):
    ok = TP.verify_extends(TARGET, _capsule(extends="H0_depth_200"), "L2",
                           declared_tiers=["L2", "L3"], roots=costly_history)
    assert ok.verified is True
    assert ok.tier == "L3" and ok.claim == TP.CLAIM_EXTENDS
    assert ok.source and "capsule_result.json" in ok.source, "verified against a record, not a name"


def test_an_absent_sibling_fails_closed_as_unverified(costly_history):
    missing = TP.verify_extends(TARGET, _capsule(extends="A2_single_tile_matmul"), "L2",
                                declared_tiers=["L2", "L3"], roots=costly_history)
    assert missing.verified is False
    assert missing.claim == TP.CLAIM_EXTENDS_UNVERIFIED
    assert missing.claim != TP.CLAIM_EXTENDS, "an unchecked claim must never read as the certified one"
    assert missing.claim != TP.CLAIM_SCREENED_ONLY, "and it is a DISTINCT state from naming nobody"
    assert "cannot be verified" in missing.reason and "reads as certified" in missing.reason


def test_a_sibling_that_did_not_pass_deeper_carries_nothing(tmp_path):
    """Present but failed, and present but not DEEPER than the cap, are both refused."""
    root = tmp_path / "mixed"
    # A sibling whose cert tier FAILED.
    (root / "S_failed").mkdir(parents=True)
    (root / "S_failed" / "capsule_result.json").write_text(json.dumps(
        {"capsule": "S_failed", "status": "fail",
         "tiers": {"L2": {"status": "pass"}, "L3": {"status": "fail"}}}), encoding="utf-8")
    # A sibling that only ever passed the cap tier itself -- it corroborates nothing deeper.
    (root / "S_shallow").mkdir(parents=True)
    (root / "S_shallow" / "capsule_result.json").write_text(json.dumps(
        {"capsule": "S_shallow", "status": "pass", "tiers": {"L2": {"status": "pass"}}}),
        encoding="utf-8")
    CC.reset_cache()
    try:
        failed = TP.verify_extends(TARGET, _capsule(extends="S_failed"), "L2",
                                   declared_tiers=["L2", "L3"], roots=[root])
        assert failed.verified is False and failed.claim == TP.CLAIM_EXTENDS_UNVERIFIED
        assert "no PASSING tier deeper" in failed.reason

        shallow = TP.verify_extends(TARGET, _capsule(extends="S_shallow"), "L2",
                                    declared_tiers=["L2", "L3"], roots=[root])
        assert shallow.verified is False, "passing the cap tier is not certifying anything deeper"
    finally:
        CC.reset_cache()


def test_an_unverified_extends_is_recorded_as_unverified_on_the_tier(costly_history):
    """End to end: the emitted tier record must not let an unchecked claim read as a certification."""
    cap = _capsule("P0_rests_on_a_ghost", max_oracle_tier="L2", extends="A2_single_tile_matmul")
    ceiling = TP.oracle_ceiling(TARGET, cap, "L3", declared_tiers=["L2", "L3"],
                                cost_roots=costly_history)
    row = _tier_result_for(ceiling).to_dict()
    assert row["status"] == "skipped"
    assert row["oracle_ceiling"]["claim"] == TP.CLAIM_EXTENDS_UNVERIFIED
    assert row["oracle_ceiling"]["extends"]["verified"] is False
    assert "UNVERIFIED" in row["reason"]
    assert "resting on nothing until the sibling's deeper pass is on disk" in row["reason"]


def test_the_timing_ceiling_field_is_schema_valid():
    from merlin.common.paths import merlin_dir
    schema = json.loads((merlin_dir() / "contract" / "schemas" / "capsule.schema.json")
                        .read_text(encoding="utf-8"))
    props = schema["properties"]
    assert TP.TIMING_CEILING_FIELD in props
    assert props[TP.TIMING_CEILING_FIELD]["enum"] == props["required_oracle_tiers"]["items"]["enum"]


def test_the_ceiling_fields_are_carried_by_the_generator():
    """Declarable ONCE in a profile and carried onto every member it derives -- so the link between a
    derived sweep member and the functional capsule it extends is machine-readable, not prose."""
    import importlib.util
    from merlin.common.paths import merlin_dir

    path = merlin_dir() / "contract" / "capsules" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("_gc_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {TP.CEILING_FIELD, TP.TIMING_CEILING_FIELD, TP.EXTENDS_FIELD} <= set(mod._DECLARED_BLOCKS)
    cap: dict = {}
    assert mod._carry_declared_blocks(
        {TP.CEILING_FIELD: "L2", TP.EXTENDS_FIELD: "A2_single_tile_matmul"}, cap)
    assert cap[TP.CEILING_FIELD] == "L2" and cap[TP.EXTENDS_FIELD] == "A2_single_tile_matmul"
    # A hand-authored capsule stays the source of record.
    already = {TP.CEILING_FIELD: "L3"}
    mod._carry_declared_blocks({TP.CEILING_FIELD: "L2"}, already)
    assert already[TP.CEILING_FIELD] == "L3"
