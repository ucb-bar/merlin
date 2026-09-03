"""The PR operand-residency DIFFERENTIAL is decidable, band by band, and refuses what it must.

Two properties this file exists to hold, both learned the hard way in this tree:

* the POSITIVE path actually runs. A negative test that passes because an import failed inside a
  broad ``except`` proves nothing, so every refusal test below is paired with a case that reaches a
  real verdict on real overlap readings, and the positive case asserts the intermediate facts (bands
  usable, controls fired, boundaries compared) rather than only the final word;
* the rule is not vacuous. The last two tests MUTATE the analyzer -- one removes the fill-transient
  guard, one replaces exact rational rates with rounded floats -- and assert that the verdicts this
  file pins would flip. If a later edit relaxes either, those tests fail.
"""
from __future__ import annotations

import sys
from fractions import Fraction

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf import residency_claim as RC
from merlin.perf.hw_counters import OccupancyCounters


SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
import perf_pr_claim as PR  # noqa: E402

REPLICATES = ("r000", "r001")

#: A two-engine combination-counter set of exactly the shape ``hw_counters`` derives from a target's
#: own header. Named A/B because nothing in this decision depends on which engines a target has.
COUNTERS = OccupancyCounters(
    prefix="X", engines=("A", "B"),
    by_combination={frozenset({"A"}): "A_CYC", frozenset({"B"}): "B_CYC",
                    frozenset({"A", "B"}): "AB_CYC"})

#: Overlap readings whose eta is CONSTANT across a band: realised = c, available = 2c, so eta = 1/2 at
#: every depth. Constant is what SATURATED means -- the trend never falls and its last step is zero.
def _settled(step: int) -> dict:
    c = 100 * step
    return {"A_CYC": c, "B_CYC": c, "AB_CYC": c}


#: Overlap readings whose eta RISES at every depth, which is the fill transient by definition.
def _filling(step: int) -> dict:
    return {"A_CYC": 100, "B_CYC": 100, "AB_CYC": 10 * step}


def _descriptor(index: int, band: str, k: int, by_regime: dict) -> dict:
    return {
        "name": f"PR{index:02d}_{band}_k{k}",
        "kind": "model_slice",
        "label": "dev",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [16, 16], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [],
                           "output_dtype": "i32"},
        },
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "performance": {
            "level": "L2_intra_layer",
            "family": "PR",
            "lever": "operand_residency",
            "claim": "DIFFERENTIAL",
            "comparand": {
                "kind": "group_arithmetic",
                "against": "the_same_affine_fit_in_a_different_residency_regime",
            },
            "falsifier": {
                "observation": "per_regime_fitted_rate_and_intercept",
                "fires_when":
                    "the_rates_fitted_in_different_residency_regimes_agree_within_the_noise_band",
                "negative_control": "two_disjoint_depth_ranges_inside_one_regime",
            },
            "gate": {"instrument": "cycle_count",
                     "capacity": "at_least_two_reduction_depths_inside_each_reachable_residency_band"},
            "emitter": {
                "status": "existing",
                "derived_axes": {"K": {
                    "derive": "memory_regime_reduction_depth",
                    "value": k,
                    "label": band,
                    "derivation": {"capacity_rows": 16384, "by_regime": by_regime},
                }},
            },
        },
    }


#: Three bands of three depths each, exactly the reachable shape the real corpus materialises.
BANDS = {
    "fits_double": [16, 2048, 4096],
    "fits_single": [4112, 6144, 8192],
    "spills": [8208, 12288, 16384],
}
BY_REGIME = {band: {"points": [{"K": k} for k in depths]} for band, depths in BANDS.items()}


def _cohort() -> list[dict]:
    out, index = [], 0
    for band, depths in BANDS.items():
        for k in depths:
            out.append(_descriptor(index, band, k, BY_REGIME))
            index += 1
    return out


def _rows(descriptors: list[dict], *, cycles_of, overlap_of) -> list[dict]:
    """Result rows in the exact shape the run seals: L2 correctness plus citable L3 timing."""
    rows = []
    for descriptor in descriptors:
        axis = descriptor["performance"]["emitter"]["derived_axes"]["K"]
        band, k = str(axis["label"]), int(axis["value"])
        step = BANDS[band].index(k) + 1
        for replicate in REPLICATES:
            common = {"approach": "arm4", "correct": True, "tier_status": "pass",
                      "grade_status": "pass", "numeric_status": "pass",
                      "error": None, "failure": None}
            rows.append({**common,
                         "identity": {"family": "PR", "capsule": descriptor["name"],
                                      "simulator": "spike", "replicate": replicate},
                         "tier": "L2", "purpose": "correctness_screen", "citable": False,
                         "cycles": None})
            rows.append({**common,
                         "identity": {"family": "PR", "capsule": descriptor["name"],
                                      "simulator": "verilator", "replicate": replicate},
                         "tier": "L3", "purpose": "performance_certification", "citable": True,
                         "cycles": cycles_of(band, k, replicate),
                         "counter_values": overlap_of(band, step)})
    return rows


#: Cycles that are EXACTLY affine inside each band, with a different rate per band. The intercept is
#: shared on purpose: only the rate is the comparand, so an equal intercept cannot carry the verdict.
RATES = {"fits_double": 2, "fits_single": 3, "spills": 5}


def _affine(rates: dict):
    def cycles_of(band: str, k: int, _replicate: str) -> int:
        return rates[band] * k + 100
    return cycles_of


def _settled_everywhere(band: str, step: int) -> dict:
    return _settled(step)


@pytest.fixture
def descriptors() -> list[dict]:
    return _cohort()


# ---------------------------------------------------------------------------------------------------
# the declaration side
# ---------------------------------------------------------------------------------------------------

def test_the_real_corpus_preflights_ready_with_three_derived_bands():
    """The POSITIVE declaration path, on the capsules actually on disk -- not a hand-built cohort."""
    root = repo_root() / "merlin/contract/capsules/_perf"
    frozen = [yaml.safe_load((path / "capsule.yaml").read_text())
              for path in sorted(root.iterdir())
              if path.is_dir() and (path / "capsule.yaml").is_file()
              and yaml.safe_load((path / "capsule.yaml").read_text())
              .get("performance", {}).get("family") == "PR"]
    assert len(frozen) == 9
    result = PR.preflight_pr_claim(frozen, replicates=REPLICATES)
    assert result["status"] == "READY", result["refusal_reasons"]
    assert list(result["cohort"]["bands"]) == ["fits_double", "fits_single", "spills"]
    assert result["cohort"]["bands"]["spills"]["K_values"] == [8208, 12288, 16384]
    assert len(result["expected_identities"]) == 9 * len(REPLICATES) * 2
    # The gap this analyzer exists to close: the profile does not yet freeze it.
    assert result["contract_frozen"] is False


def test_a_declared_acceptance_block_must_be_the_one_this_analyzer_implements(descriptors):
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PR")
    declared = sweep["base"]["performance"].get("acceptance")
    if declared is not None:
        assert declared == PR.supported_acceptance()

    for descriptor in descriptors:
        descriptor["performance"]["acceptance"] = PR.supported_acceptance()
    assert PR.preflight_pr_claim(descriptors, replicates=REPLICATES)["contract_frozen"] is True
    descriptors[0]["performance"]["acceptance"]["noise_band"]["declared_constant"] = 8
    refused = PR.preflight_pr_claim(descriptors, replicates=REPLICATES)
    assert refused["status"] == "REFUSED"
    assert "acceptance contract this analyzer does not implement" in refused["refusal_reasons"][0]


def test_one_replicate_is_refused_because_the_noise_band_is_the_measured_dispersion(descriptors):
    result = PR.preflight_pr_claim(descriptors, replicates=("r000",))
    assert result["status"] == "REFUSED"
    assert "UNDETERMINABLE rather than zero" in result["refusal_reasons"][0]


def test_a_band_of_two_depths_cannot_carry_its_own_negative_control():
    bands = {"fits_double": [16, 2048], "spills": [8208, 12288]}
    by_regime = {band: {"points": [{"K": k} for k in depths]} for band, depths in bands.items()}
    descriptors = [_descriptor(i, band, k, by_regime)
                   for i, (band, k) in enumerate(
                       (band, k) for band, depths in bands.items() for k in depths)]
    result = PR.preflight_pr_claim(descriptors, replicates=REPLICATES)
    assert result["status"] == "REFUSED"
    assert "two disjoint depth ranges inside one regime" in result["refusal_reasons"][0]


def test_a_band_label_its_own_derivation_contradicts_is_refused(descriptors):
    descriptors[0]["performance"]["emitter"]["derived_axes"]["K"]["label"] = "spills"
    result = PR.preflight_pr_claim(descriptors, replicates=REPLICATES)
    assert result["status"] == "REFUSED"
    assert "the label and the derivation disagree" in result["refusal_reasons"][0]


# ---------------------------------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------------------------------

def test_established_when_every_boundary_changes_the_rate(descriptors):
    """THE POSITIVE PATH. Real overlap readings, real fits, a real ESTABLISHED verdict."""
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert decision["status"] == RC.ESTABLISHED, decision

    verdict = decision["verdict"]
    assert verdict["usable_bands"] == ["fits_double", "fits_single", "spills"]
    assert verdict["refused_bands"] == []
    assert verdict["bands_refused_for_transient_reasons"] == []
    # The overlap instrument really ran: every band saturated on a resolved reading, not on a None.
    for band in verdict["bands"]:
        assert band["transient"]["state"] == "saturated"
        assert all(member["realised_overlap_cycles"] is not None for member in band["members"])
        assert band["negative_control"]["fired"] is True
    assert [b["rate"]["value"] for b in verdict["bands"]] == [2.0, 3.0, 5.0]
    assert len(verdict["boundaries"]) == 3
    assert all(row["falsifier_fired"] is False for row in verdict["boundaries"])
    assert verdict["noise_band"]["cycles"] == 0
    assert verdict["noise_band"]["declared_constant"] is None
    assert PR.promotion_status(decision) == "PROMOTED"
    assert PR.decision_boundary(decision)["promotion_integration"] == "integrated"


def test_refuted_when_one_boundary_leaves_the_rate_unchanged(descriptors):
    agreeing = {"fits_double": 2, "fits_single": 2, "spills": 5}
    rows = _rows(descriptors, cycles_of=_affine(agreeing), overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert decision["status"] == RC.REFUTED
    assert decision["refutation_reasons"] == ["fits_double|fits_single"]
    fired = [row for row in decision["verdict"]["boundaries"] if row["falsifier_fired"]]
    assert [(row["lower_band"], row["upper_band"]) for row in fired] == [
        ("fits_double", "fits_single")]
    assert PR.promotion_status(decision) == "BLOCKED"


def test_a_band_inside_the_fill_transient_is_refused_and_quotes_no_rate(descriptors):
    """The trap that refuted the sibling family, one level down: the cheapest band is still filling."""
    def overlap_of(band: str, step: int) -> dict:
        return _filling(step) if band == "fits_double" else _settled(step)

    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=overlap_of)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    verdict = decision["verdict"]
    assert verdict["bands_refused_for_transient_reasons"] == ["fits_double"]
    transient = next(band for band in verdict["bands"] if band["band"] == "fits_double")
    assert transient["status"] == RC.BAND_TRANSIENT
    assert transient["rate"] is None and transient["negative_control"] is None
    assert "no rate is quoted for this band" in transient["reason"]
    # The remaining two bands still decide -- a refused band narrows the claim, it does not void it.
    assert verdict["usable_bands"] == ["fits_single", "spills"]
    assert decision["status"] == RC.ESTABLISHED


def test_every_band_in_the_transient_leaves_no_boundary_to_compare(descriptors):
    rows = _rows(descriptors, cycles_of=_affine(RATES),
                 overlap_of=lambda band, step: _filling(step))
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert decision["status"] == RC.REFUSED
    assert decision["verdict"]["bands_refused_for_transient_reasons"] == [
        "fits_double", "fits_single", "spills"]
    assert "a residency differential needs two" in decision["refusal_reasons"][0]


def test_absent_counters_refuse_rather_than_skip_the_transient_guard(descriptors):
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=None)
    assert decision["status"] == RC.REFUSED
    assert "fill-transient guard cannot run" in decision["refusal_reasons"][0]


def test_a_member_with_no_counter_reading_is_unknown_not_zero_overlap(descriptors):
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    for row in rows:
        if row["identity"]["capsule"].startswith("PR00") and row["tier"] == "L3":
            row["counter_values"] = None
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    band = next(row for row in decision["verdict"]["bands"] if row["band"] == "fits_double")
    assert band["status"] == RC.BAND_OVERLAP_UNDETERMINABLE
    assert band["rate"] is None
    assert "never zero" in band["members"][0]["overlap_detail"]


def test_replicates_that_disagree_refuse_the_band_rather_than_averaging(descriptors):
    def cycles_of(band: str, k: int, replicate: str) -> int:
        base = RATES[band] * k + 100
        return base + (1 if (band == "spills" and replicate == "r001") else 0)

    rows = _rows(descriptors, cycles_of=cycles_of, overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    band = next(row for row in decision["verdict"]["bands"] if row["band"] == "spills")
    assert band["status"] == RC.BAND_REPLICATES_DISAGREE
    assert "would invent a point" in band["reason"]
    assert band["rate"] is None


def test_inert_when_no_band_can_show_its_own_two_ranges_agreeing(descriptors):
    """Curved-but-saturated bands: each band's own sub-ranges differ, so agreement is undetectable."""
    def cycles_of(band: str, k: int, _replicate: str) -> int:
        # Convex inside the band (marginals RISE, so the affine form is not contradicted by a fall)
        # but not one rate, so the declared negative control cannot fire anywhere.
        step = BANDS[band].index(k)
        return RATES[band] * k + 100 + (0, 0, 5000)[step]

    rows = _rows(descriptors, cycles_of=cycles_of, overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert decision["status"] == RC.INERT, decision["verdict"]["refused_bands"]
    assert all(band["status"] == RC.BAND_CONTROL_DID_NOT_FIRE
               for band in decision["verdict"]["bands"])
    assert "has not shown it can report agreement" in decision["verdict"]["reason"]


def test_evidence_that_is_not_a_correct_measurement_is_refused(descriptors):
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    next(row for row in rows if row["tier"] == "L3")["citable"] = False
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert decision["status"] == RC.REFUSED
    assert "L2/L3 evidence semantics" in decision["refusal_reasons"][0]


def test_the_decision_does_not_depend_on_the_order_rows_or_descriptors_arrive_in(descriptors):
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    forward = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    reverse = PR.analyze_pr_claim(list(reversed(descriptors)), list(reversed(rows)),
                                  replicates=REPLICATES, counters=COUNTERS)
    assert forward == reverse


def test_the_negative_control_evidence_reaches_the_campaign_judge(descriptors):
    from merlin.perf.campaign import FalsifierEvidence, ReplicaIdentity

    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    evidence = PR.falsifier_evidence(decision, member_of=lambda band: ReplicaIdentity(
        family="PR", member=band, tier="L3", replica="r000"))
    assert len(evidence) == 3
    assert all(isinstance(row, FalsifierEvidence) for row in evidence)
    assert all(row.negative_control is True and row.fired is True for row in evidence)
    with pytest.raises(ValueError):
        PR.falsifier_evidence(decision, member_of=None)


def test_the_derived_counter_mapping_is_accepted_and_a_useless_one_is_refused(descriptors):
    """The dict form ``counters_for_target`` actually hands a caller must reach the same verdict."""
    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=_settled_everywhere)
    live = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    from_dict = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES,
                                    counters=COUNTERS.to_dict())
    assert from_dict == live
    assert from_dict["status"] == RC.ESTABLISHED

    one_engine = {"prefix": "X", "engines": ["A"], "by_combination": {"A": "A_CYC"}}
    refused = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=one_engine)
    assert refused["status"] == RC.REFUSED
    assert "one engine cannot overlap with itself" in refused["refusal_reasons"][0]


def test_a_band_whose_SHALLOW_end_is_still_filling_is_caught_by_its_own_control(descriptors):
    """The transient guard reads the DEEPEST step, so the shallow end needs the control to catch it.

    ``transient_verdict`` answers "had the machine finished filling by the deepest point measured".
    A band whose two deep points are settled but whose shallowest point is not therefore reports
    SATURATED -- and this is exactly PR's ``fits_double``, whose first depth is the very K=16 the
    sibling family measured at 301 cycles inside the transient. What catches it is the band's OWN
    negative control: a contaminated shallow point makes the lower depth range fit a different rate
    from the upper one, so no rate is quoted for the band. This test pins that second line of defence.
    """
    def overlap_of(band: str, step: int) -> dict:
        if band != "fits_double":
            return _settled(step)
        # eta 3/10 at the shallowest depth, 1/2 at both deep ones: rising, then flat at the bottom.
        return ({"A_CYC": 700, "B_CYC": 700, "AB_CYC": 300} if step == 1
                else {"A_CYC": 500, "B_CYC": 500, "AB_CYC": 500})

    def cycles_of(band: str, k: int, _replicate: str) -> int:
        if band != "fits_double":
            return RATES[band] * k + 100
        # The real citable Verilator measurement at M=N=16, K=16; the deep points sit on the band's
        # settled affine law.
        return 301 if k == 16 else 2 * k + 100

    rows = _rows(descriptors, cycles_of=cycles_of, overlap_of=overlap_of)
    decision = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    band = next(row for row in decision["verdict"]["bands"] if row["band"] == "fits_double")
    assert band["transient"]["state"] == "saturated"          # the guard alone would have passed it
    assert band["status"] == RC.BAND_CONTROL_DID_NOT_FIRE     # the control does not
    assert band["negative_control"]["lower_range"]["rate"]["value"] != pytest.approx(2.0)
    assert band["negative_control"]["upper_range"]["rate"]["value"] == pytest.approx(2.0)
    assert decision["verdict"]["usable_bands"] == ["fits_single", "spills"]


# ---------------------------------------------------------------------------------------------------
# vacuity: the suite must FAIL if either rule is relaxed
# ---------------------------------------------------------------------------------------------------

def test_removing_the_transient_guard_would_flip_the_transient_refusal(descriptors, monkeypatch):
    """Mutation: if the guard stopped refusing, a band inside the transient would quote a rate."""
    def overlap_of(band: str, step: int) -> dict:
        return _filling(step) if band == "fits_double" else _settled(step)

    rows = _rows(descriptors, cycles_of=_affine(RATES), overlap_of=overlap_of)
    guarded = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert guarded["verdict"]["bands_refused_for_transient_reasons"] == ["fits_double"]

    real = RC._ft.transient_verdict
    monkeypatch.setattr(RC._ft, "transient_verdict",
                        lambda points: {**real(points), "state": RC._ft.SATURATED})
    relaxed = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert relaxed["verdict"]["bands_refused_for_transient_reasons"] == []
    assert relaxed["verdict"]["usable_bands"] == ["fits_double", "fits_single", "spills"]


def test_introducing_a_tolerance_would_flip_established_into_refuted(descriptors, monkeypatch):
    """Mutation: the one repair this contract forbids, applied, and the verdict it would buy.

    ``fits_single`` here really does cost more per depth than ``fits_double`` -- 2.0625 cycles against
    2 -- and the exact rational predicate reports that as a change across the residency boundary. Give
    the predicate a tolerance of a tenth of a cycle and the same evidence says the rate is unchanged.
    So the ESTABLISHED verdict this file pins is load-bearing on the predicate being exact, and a
    later edit that widens it fails here rather than quietly re-deciding the family.
    """
    def cycles_of(band: str, k: int, _replicate: str) -> int:
        # An extra sixteenth of a cycle per depth in fits_single, measured from that band's own base.
        # Exactly affine inside the band (both its spans, 2032 and 2048, are multiples of 16), so the
        # band's own negative control still fires and the difference is a rate difference, not noise.
        if band == "spills":
            return 5 * k + 100
        if band != "fits_single":
            return 2 * k + 100
        return 2 * k + 100 + (k - BANDS["fits_single"][0]) // 16

    rows = _rows(descriptors, cycles_of=cycles_of, overlap_of=_settled_everywhere)
    exact = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert exact["status"] == RC.ESTABLISHED
    boundary = next(row for row in exact["verdict"]["boundaries"]
                    if (row["lower_band"], row["upper_band"]) == ("fits_double", "fits_single"))
    assert boundary["falsifier_fired"] is False
    assert boundary["rate_difference"]["value"] == pytest.approx(0.0625)

    monkeypatch.setattr(RC, "_agree", lambda left, right: abs(left - right) <= Fraction(1, 10))
    relaxed = PR.analyze_pr_claim(descriptors, rows, replicates=REPLICATES, counters=COUNTERS)
    assert relaxed["status"] == RC.REFUTED
    assert "fits_double|fits_single" in relaxed["refutation_reasons"]
