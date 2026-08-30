"""The instruction-timing harvest: what it recovers, and everything it refuses to claim.

The harvest exists because functional runs already generate calibration evidence and it was being
thrown away. Recovering it is the easy half; the tests below are mostly about the hard half, which is
that a number read out of a running program is **contended** and must never be spelled as anything
stronger than that. Each honesty rule has a test that fails if the rule is relaxed.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.kernels.measurement import MeasurementAuthority
from merlin.perf import harvest as H
from merlin.perf.term import UNKNOWN, UnknownValueError


# ---------------------------------------------------------------------------------------------
# fixtures: a declared authority and a couple of hand-built capsule results
# ---------------------------------------------------------------------------------------------


def _authority(**kw):
    base = {"target": "unit-under-test", "cycles_from": "sim", "cycles_tier": "rtl",
            "citable_tier": "rtl", "declared": True}
    base.update(kw)
    return MeasurementAuthority(**base)


def _obs(value, *, workload="w", submission="s", status="pass", **kw):
    return H.Observation(submission=submission, workload=workload, stage="L3", substrate="sim",
                         tier="rtl", quantity="total_cycles", value=float(value), unit="cycles",
                         status=status, concurrent=("a movement engine",), evidence=("run.json",),
                         **kw)


def _write_capsule_result(tmp_path, submission, capsule, tiers):
    d = tmp_path / submission / "runs" / "suite" / capsule
    d.mkdir(parents=True, exist_ok=True)
    p = d / "capsule_result.json"
    p.write_text(json.dumps({"capsule": capsule, "status": "pass", "tiers": tiers,
                             "toolchain_shas": {"tool": "abc123"}}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------------------
# Rule 1 -- a harvested latency is a CONTENDED upper bound and can never become a constant
# ---------------------------------------------------------------------------------------------


def test_harvested_term_is_trace_derived_and_never_stronger():
    term = H.harvested_term("busy", [_obs(100), _obs(140)], unit="cycles", regime="a corpus run")
    assert term.provenance.kind == H.HARVEST_KIND == "trace_derived"
    # weaker than a dedicated measurement, so anything composed from it inherits the weaker kind
    from merlin.perf.term import combine_kinds
    assert combine_kinds([term.provenance.kind, "measured"]) == H.HARVEST_KIND


def test_harvested_term_names_what_else_was_active():
    term = H.harvested_term("busy", [_obs(100)], unit="cycles", regime="a corpus run")
    assert "movement engine" in term.validity.weak_regime
    assert "UPPER BOUND" in term.validity.weak_regime


def test_harvested_term_records_the_spread_across_occurrences():
    term = H.harvested_term("busy", [_obs(1090), _obs(3078), _obs(8889)], unit="cycles",
                            regime="three submissions of one capsule")
    assert term.bounds.lower == 1090 and term.bounds.upper == 8889
    assert term.value == 3078                    # the median is a PRIOR, not the constant
    assert "spread" in term.validity.expected_error
    assert H.spread([1090, 3078, 8889])["ratio"] == pytest.approx(8889 / 1090)


def test_spread_of_nothing_is_not_a_point():
    assert H.spread([]) == {"n": 0}


def test_promotion_to_calibrated_always_raises():
    term = H.harvested_term("busy", [_obs(100)], unit="cycles", regime="r")
    with pytest.raises(H.ContendedTermError) as exc:
        H.promote(term, experiment="a big corpus")
    assert "dedicated experiment" in str(exc.value)


def test_a_term_spelled_stronger_than_trace_derived_is_rejected():
    from merlin.perf.term import Bounds, PerformanceTerm, Provenance, Validity
    strong = PerformanceTerm(name="busy", value=100, unit="cycles",
                             provenance=Provenance("measured", ("run.json",)),
                             validity=Validity(validated_regime="r"), bounds=Bounds(0, 200))
    with pytest.raises(H.ContendedTermError):
        H.assert_contended(strong)


# ---------------------------------------------------------------------------------------------
# Rule 2 -- only substrates the target declares citable contribute
# ---------------------------------------------------------------------------------------------


def test_an_undeclared_authority_contributes_nothing(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {
        "L3": {"status": "pass", "cycles": 500, "derived_from_rtl": True, "cycle_accurate": True}})
    obs, refusals = H.harvest_capsule_result(p, authority=MeasurementAuthority(target="t"))
    assert obs == []
    assert refusals and "nothing is declared at all" in refusals[0].reason


def test_a_tier_below_the_declared_citable_tier_is_refused_with_its_reason(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {
        "L2": {"status": "pass", "cycles": 543, "derived_from_rtl": False},
        "L3": {"status": "pass", "cycles": 178, "derived_from_rtl": True, "cycle_accurate": True}})
    obs, refusals = H.harvest_capsule_result(p, authority=_authority())
    assert [o.value for o in obs] == [178.0]
    assert any("functional" in r.reason for r in refusals)


def test_the_oracles_own_fidelity_outranks_the_tier_name(tmp_path):
    # A tier NAMED L3 whose oracle calls itself a functional model is not an RTL result.
    p = _write_capsule_result(tmp_path, "sub", "cap", {
        "L3": {"status": "pass", "cycles": 7, "derived_from_rtl": True, "cycle_accurate": True,
               "fidelity": "functional_model"}})
    obs, _ = H.harvest_capsule_result(p, authority=_authority())
    assert obs == []
    obs, _ = H.harvest_capsule_result(p, authority=_authority(citable_tier="functional"))
    assert [o.tier for o in obs] == ["functional"]


def test_a_tier_record_stating_no_provenance_reaches_no_tier(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {"L3": "pass"})
    obs, refusals = H.harvest_capsule_result(p, authority=_authority())
    assert obs == []
    assert refusals == []                        # a bare-string record reports no cycles at all


# ---------------------------------------------------------------------------------------------
# Rule 3 -- nothing is filled in silently
# ---------------------------------------------------------------------------------------------


def test_an_empty_series_is_unknown_with_a_reason_and_refuses_to_read_as_zero():
    term = H.harvested_term("busy", [], unit="cycles", regime="nothing ran")
    assert term.is_unknown and term.unknown_reason
    with pytest.raises(UnknownValueError):
        float(term.value)
    with pytest.raises(UnknownValueError):
        bool(term.value)                         # `x or 0` must not silently publish a zero
    assert term.value is UNKNOWN


def test_an_adapter_with_no_timing_capability_emits_nothing_never_zeros(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {
        "L3": {"status": "pass", "cycles": None, "derived_from_rtl": True, "cycle_accurate": True},
        "L4": {"status": "skipped", "cycles": None, "not_applicable": True}})
    obs, refusals = H.harvest_capsule_result(p, authority=_authority())
    assert obs == [] and refusals == []


def test_a_score_summary_with_no_tier_is_refused_rather_than_attributed(tmp_path):
    p = tmp_path / "score_capsule.json"
    p.write_text(json.dumps({"package": "pkg", "cycles_diagnostic": {"A1": 159, "A2": 316}}),
                 encoding="utf-8")
    obs, refusals = H.harvest_score_file(p, authority=_authority())
    assert obs == []
    assert len(refusals) == 2 and all("no tier" in r.reason for r in refusals)


# ---------------------------------------------------------------------------------------------
# Cycles are a property of the SUBMISSION
# ---------------------------------------------------------------------------------------------


def test_observations_are_keyed_by_submission_not_by_capsule_name(tmp_path):
    tier = {"status": "pass", "derived_from_rtl": True, "cycle_accurate": True}
    a = _write_capsule_result(tmp_path, "subA", "AT2", {"L3": {**tier, "cycles": 1090}})
    b = _write_capsule_result(tmp_path, "subB", "AT2", {"L3": {**tier, "cycles": 3078}})
    oa, _ = H.harvest_capsule_result(a, authority=_authority())
    ob, _ = H.harvest_capsule_result(b, authority=_authority())
    assert oa[0].workload == ob[0].workload == "AT2"
    assert oa[0].submission != ob[0].submission
    assert oa[0].submission.endswith("subA") and ob[0].submission.endswith("subB")


def test_a_term_refuses_to_pool_observations_with_different_verdicts():
    with pytest.raises(ValueError) as exc:
        H.harvested_term("busy", [_obs(2, status="fail"), _obs(6349, status="pass")],
                         unit="cycles", regime="r")
    assert "different verdicts" in str(exc.value)


def test_the_failing_stages_are_kept_and_reachable(tmp_path):
    tier = {"derived_from_rtl": True, "cycle_accurate": True}
    _write_capsule_result(tmp_path, "sub", "cap", {"L3": {**tier, "status": "fail", "cycles": 2}})
    h = H.retro_mine([tmp_path], target="t", authority=_authority())
    assert [o.value for o in h.series("total_cycles")] == [2.0]
    assert h.series("total_cycles", status="pass") == ()


# ---------------------------------------------------------------------------------------------
# >=2 points per fitted parameter
# ---------------------------------------------------------------------------------------------


def _axis(points, **kw):
    kw.setdefault("x_name", "x")
    return H.AxisEvidence(axis="a", y_name="y", y_unit="cycles",
                          points=tuple(H.Point(x, y, f"p{i}") for i, (x, y) in enumerate(points)),
                          **kw)


def test_an_affine_fit_refuses_below_four_points():
    fit = H.fit_points(_axis([(1, 10), (2, 20), (3, 30)]), "affine")
    assert not fit.ok and ">=4 points" in fit.refusal


def test_an_affine_fit_refuses_when_every_point_shares_one_x():
    fit = H.fit_points(_axis([(1, 10), (1, 11), (1, 12), (1, 13)]), "affine")
    assert not fit.ok and "distinct" in fit.refusal


def test_a_proportional_fit_refuses_on_one_point():
    assert not H.fit_points(_axis([(2, 64)]), "proportional").ok


def test_a_well_supported_affine_fit_separates_the_rate_from_the_overhead():
    fit = H.fit_points(_axis([(10, 24), (20, 44), (30, 64), (40, 84)]), "affine")
    assert fit.ok
    assert fit.parameters["slope"] == pytest.approx(2.0)
    assert fit.parameters["intercept"] == pytest.approx(4.0)


def test_an_unimplemented_fit_form_is_refused_not_guessed():
    assert "no fit form" in H.fit_points(_axis([(1, 1), (2, 2)]), "quadratic").refusal


def test_a_fill_law_inverts_through_the_law_itself_and_refuses_what_it_cannot_produce():
    from merlin.perf.record import fill_cycles
    dim, why = H.invert_fill_law("systolic_2d", fill_cycles("systolic_2d", 32))
    assert dim == 32 and "systolic_2d(32)" in why
    none, reason = H.invert_fill_law("systolic_2d", 63)     # an odd fill: 2*d-2 is always even
    assert none is None and "does not describe this unit" in reason


# ---------------------------------------------------------------------------------------------
# Deriving the axes: pairings are DERIVED, and refuse when they are not unique
# ---------------------------------------------------------------------------------------------


def _suite(kernels, meta=None):
    return {"_meta": dict(meta or {}), "kernels": kernels}


def _kernel(*, ops, truth, buckets, reads, writes, footprint):
    act = {"truth": truth, "none": 0, "reads": reads, "writes": writes, "halt_reason": 1}
    act.update(buckets)
    return {"op_stream": ops, "arc": act, "footprint_bytes": footprint, "npu_cycles": truth}


def _linear_suite():
    """Two engines: one whose cycles track the beat counters, one that does not.

    The compute unit's busy is ``groups * 16`` and the program schedules 10 cycles behind each
    compute op, so what the unit itself contributes per drained result is 6 -- a ``systolic_2d``
    fill of 6, i.e. a structural dimension of 4. One kernel issues no compute ops at all, which is
    what lets the bucket's support single out its op family.
    """
    kernels = {}
    for i, (groups, beats) in enumerate([(1, 128), (1, 256), (2, 512), (2, 1024), (0, 64)]):
        ops = []
        for _ in range(groups):
            ops += [["Grid", "push", 0], ["Sched", "delay", 4],
                    ["Grid", "mul", 0], ["Sched", "delay", 10],
                    ["Grid", "pop", 0], ["Sched", "delay", 4]]
        ops += [["Move", "xfer", 0]]
        kernels[f"k{i}"] = _kernel(ops=ops, truth=beats + 7 + groups * 16,
                                   buckets={"engine": beats + 7, "grid": groups * 16},
                                   reads=beats // 2, writes=beats - beats // 2,
                                   footprint=beats * 8)
    return _suite(kernels, {"beat_bytes": 8})


def test_the_movement_bucket_is_identified_by_behaviour_not_by_name():
    axes, refusals, deriv = H.axes_from_suite(_linear_suite())
    assert deriv["movement"]["bucket"] == "engine"
    assert deriv["compute_pairings"] == {"grid": "Grid"}
    assert len(axes["movement_beat_count"].points) == 5


def test_the_beat_width_axis_recovers_the_instruments_own_granule():
    axes, _, _ = H.axes_from_suite(_linear_suite())
    fit = H.fit_points(axes["moved_byte_footprint"], "proportional")
    assert fit.ok and fit.parameters["ratio"] == pytest.approx(8.0)


def test_the_movement_pairing_is_refused_when_two_buckets_look_linear():
    suite = _linear_suite()
    for entry in suite["kernels"].values():
        beats = entry["arc"]["reads"] + entry["arc"]["writes"]
        entry["arc"]["grid"] = beats + 3          # a second bucket that also tracks the beats
    axes, refusals, deriv = H.axes_from_suite(suite)
    assert deriv["movement"]["bucket"] is None
    assert any("ambiguous" in r.reason for r in refusals)
    assert "movement_beat_count" not in axes


def test_a_compute_bucket_whose_support_does_not_match_any_family_is_not_paired():
    suite = _linear_suite()
    # the instrument reports the unit idle in a program that plainly issues its ops
    first = next(iter(suite["kernels"]))
    suite["kernels"][first]["arc"]["grid"] = 0
    _axes, refusals, deriv = H.axes_from_suite(suite)
    assert "grid" not in deriv["compute_pairings"]
    assert any("not attributable" in r.reason for r in refusals)


def test_a_program_outside_the_one_compute_per_drain_regime_is_excluded_not_fitted():
    suite = _linear_suite()
    k = suite["kernels"]["k0"]
    k["op_stream"] = [["Grid", "push", 0], ["Sched", "delay", 4],
                      ["Grid", "mul", 0], ["Sched", "delay", 10],
                      ["Grid", "mul", 0], ["Sched", "delay", 10],
                      ["Grid", "pop", 0], ["Sched", "delay", 4], ["Move", "xfer", 0]]
    axes, _, _ = H.axes_from_suite(suite)
    excluded = axes["compute_group_count"].excluded
    assert any("accumulates" in r.reason for r in excluded)
    assert all(p.label != "k0:grid" for p in axes["compute_group_count"].points)


def test_the_overlap_axis_is_empty_because_partitioned_buckets_cannot_carry_it():
    axes, _, _ = H.axes_from_suite(_linear_suite())
    assert axes["concurrent_unit_busy"].points == ()
    assert "PARTITION" in axes["concurrent_unit_busy"].note


def test_traits_the_evidence_establishes_never_overturn_a_refutation():
    axes, _, deriv = H.axes_from_suite(_linear_suite())
    detected = H.detected_traits(axes, deriv)
    assert detected["explicit_dma"][0] is True
    assert detected["explicit_completion"][0] is False


# ---------------------------------------------------------------------------------------------
# The op-stream adapter
# ---------------------------------------------------------------------------------------------


def test_the_delay_marker_is_derived_from_the_corpus_not_assumed():
    h = H.harvest_op_stream(_linear_suite(), target="t", authority=_authority(cycles_tier="rtl"))
    assert h.observations
    assert {o.quantity for o in h.observations} == {"scheduled_delay.push", "scheduled_delay.mul",
                                                    "scheduled_delay.pop"}
    assert all(o.concurrent for o in h.observations)


def test_the_op_stream_adapter_refuses_when_the_corpus_tier_is_not_citable():
    h = H.harvest_op_stream(_linear_suite(), target="t",
                            authority=_authority(cycles_tier="functional", citable_tier="rtl"))
    assert h.observations == ()
    assert any("not citable" in r.reason for r in h.refusals)


def test_the_op_stream_adapter_refuses_an_ambiguous_delay_marker():
    suite = _linear_suite()
    suite["kernels"]["k0"]["op_stream"].append(["Sched", "pause", 9])
    h = H.harvest_op_stream(suite, target="t", authority=_authority())
    assert h.observations == () and any("delay marker" in r.what for r in h.refusals)


# ---------------------------------------------------------------------------------------------
# Retro-mining runs that are actually on disk
# ---------------------------------------------------------------------------------------------


def test_retro_mine_reports_its_roots_and_its_refusals_even_when_nothing_is_there(tmp_path):
    h = H.retro_mine([tmp_path / "nope"], target="t", authority=_authority())
    assert h.observations == ()
    assert any("not a directory" in r.reason for r in h.refusals)
    assert h.to_dict()["n_observations"] == 0


def test_retro_mine_walks_a_nested_grade_layout(tmp_path):
    tier = {"status": "pass", "derived_from_rtl": True, "cycle_accurate": True}
    for sub in ("subA", "subB"):
        for cap in ("c0", "c1"):
            _write_capsule_result(tmp_path, sub, cap, {"L3": {**tier, "cycles": 100}})
    h = H.retro_mine([tmp_path], target="t", authority=_authority())
    assert len(h.observations) == 4
    assert len(h.submissions()) == 2
    assert h.quantities() == ("total_cycles",)


@pytest.mark.parametrize("target", ["atlas", "gemmini"])
def test_retro_mine_recovers_real_observations_from_runs_already_on_disk(target):
    """The point of the module, on the corpus this repo actually has."""
    roots = H.discover_roots(target)
    if not roots:
        pytest.skip(f"no graded-run root for {target} on this host")
    h = H.retro_mine(target=target)
    if not h.observations:
        pytest.skip(f"{target} has run roots but no citable timing on this host")
    assert len(h.submissions()) > 1, "cycles are a property of the submission; expect several"
    assert all(o.tier for o in h.observations)
    assert all(o.evidence for o in h.observations)
    # every recovered number is citable at the tier the target declares
    from merlin.kernels.measurement import citable
    assert all(citable(h.authority, o.tier) for o in h.observations)
    # and the refusals are named rather than dropped
    assert all(r.reason for r in h.refusals)


def test_the_module_parses_structurally_and_names_no_target():
    src = (repo_root() / "merlin" / "python" / "merlin" / "perf" / "harvest.py").read_text()
    assert "import re" not in src.replace("import record", "")


# ---------------------------------------------------------------------------------------------
# The optional per-unit block an adapter MAY carry (R5.1's consumer side)
# ---------------------------------------------------------------------------------------------


def test_a_per_unit_timing_block_is_harvested_under_the_same_citability_gate(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {"L3": {
        "status": "pass", "cycles": 1090, "derived_from_rtl": True, "cycle_accurate": True,
        H.TIMING_OBSERVATIONS_KEY: [
            {"quantity": "busy_cycles.grid", "value": 158, "unit": "cycles",
             "concurrent": ["the movement engine"], "note": "per-unit activity"},
            {"quantity": "busy_cycles.mover", "value": 2054, "unit": "cycles"}]}})
    obs, _ = H.harvest_capsule_result(p, authority=_authority())
    assert {o.quantity for o in obs} == {"total_cycles", "busy_cycles.grid", "busy_cycles.mover"}
    fine = [o for o in obs if o.quantity == "busy_cycles.grid"][0]
    assert fine.tier == "rtl" and fine.concurrent == ("the movement engine",)
    # and the whole block is dropped when the tier is not citable, exactly like the cycle count
    obs, _ = H.harvest_capsule_result(p, authority=_authority(citable_tier="silicon"))
    assert obs == []


def test_an_unreported_per_unit_entry_is_skipped_never_recorded_as_zero(tmp_path):
    p = _write_capsule_result(tmp_path, "sub", "cap", {"L3": {
        "status": "pass", "cycles": 10, "derived_from_rtl": True, "cycle_accurate": True,
        H.TIMING_OBSERVATIONS_KEY: [{"quantity": "busy_cycles.grid", "value": None},
                                    {"value": 5}]}})
    obs, _ = H.harvest_capsule_result(p, authority=_authority())
    assert [o.quantity for o in obs] == ["total_cycles"]


def test_an_adapter_that_carries_no_block_is_byte_identical_to_before(tmp_path):
    tier = {"status": "pass", "cycles": 10, "derived_from_rtl": True, "cycle_accurate": True}
    p = _write_capsule_result(tmp_path, "sub", "cap", {"L3": tier})
    q = _write_capsule_result(tmp_path, "sub2", "cap", {"L3": {**tier,
                                                              H.TIMING_OBSERVATIONS_KEY: []}})
    a, _ = H.harvest_capsule_result(p, authority=_authority())
    b, _ = H.harvest_capsule_result(q, authority=_authority())
    assert len(a) == len(b) == 1
