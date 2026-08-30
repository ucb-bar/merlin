"""``merlin.perf.oracle_cost`` — the two-term oracle cost law, checked against measured fixtures.

Every number asserted here was MEASURED on a real substrate by a prior session and published in
``out/artifacts/capsule-bench/*/tier_policy/``. The raw ladder rows are inlined so the core
reproduction runs anywhere; the large held-out corpus is read from the artifact and the test SKIPS
loudly (never silently passes) when that untracked output is absent.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pytest

from merlin.common.paths import repo_root
from merlin.perf.oracle_cost import (
    CostLaw, CostSample, MixedConcurrency, NotEnoughEvidence, Probe, ProbeKind,
    Provenance, RunOutcome, average_replicates, concurrency_inflation, cycles_only_slope,
    fit_cost_law, halt_first_probes, measure, render_law,
)

ART = repo_root() / "out/artifacts/capsule-bench/atlas/tier_policy"
LAYERSCALE = ART / "oracle_cost_layerscale_atlas.json"
GSIM = ART / "oracle_gsim_evaluation_atlas.json"

# --- measured fixture rows --------------------------------------------------------------------------
# oracle_cost_layerscale_atlas.json, phase D_imem_load, kind="load_only": the halt-first program padded
# to W words. Cycles pinned at 2 on every rung; wall_s is the load term, alone. (words, seconds)
LOAD_LADDER = {
    "cycle_model": [(1, 0.0504), (1001, 3.2169), (4001, 12.4834), (16001, 50.4158), (32001, 99.2764)],
    "elaborated_rtl": [(1, 0.0124), (1001, 0.2576), (4001, 1.0459), (16001, 4.1346), (32001, 8.4792)],
}
# phase C_ladder: ONE program, swept trip count. Words pinned. (cycles, words, sim_seconds)
CYCLE_LADDER = {
    "cycle_model": [(1090, 414, 3.1408), (4367, 418, 8.2515), (17459, 418, 29.4032),
                    (69827, 418, 113.548), (139651, 418, 219.4591)],
    "elaborated_rtl": [(1090, 414, 0.2501), (4367, 418, 0.685), (17459, 418, 2.3623),
                       (69827, 418, 9.1678), (279299, 418, 35.4069), (1117187, 418, 144.8314),
                       (2233280, 418, 292.3265)],
}
# The published law these ladders must reproduce (measured_law.per_tier).
PUBLISHED = {
    "cycle_model": {"ms_per_cycle": 1.553303, "cycles_per_s": 643.8,
                    "ms_per_word": 3.105214, "words_per_s": 322.0},
    "elaborated_rtl": {"ms_per_cycle": 0.130877, "cycles_per_s": 7640.7,
                       "ms_per_word": 0.264247, "words_per_s": 3784.3},
}
LARGE_ONLY_MIN_CYCLES = 17459

# oracle_gsim_evaluation_atlas.json, phase B_inert_pad_word_ladder: the same construction on a second
# pair of substrates. (words, fast_seconds, slow_seconds); cycles pinned at 1478 on every rung.
GSIM_PAD_LADDER = [
    (138, 0.009309, 0.339672), (1138, 0.016377, 0.637589), (4138, 0.025873, 1.641244),
    (16138, 0.041146, 5.465903), (32138, 0.094347, 10.616321),
]
# phase A_corpus_head_to_head: real capsule programs. Both axes vary and are CORRELATED.
# (cycles, words, fast_seconds, slow_seconds)
GSIM_CORPUS = [
    (1478, 138, 0.008332, 0.279821), (444, 118, 0.005571, 0.123622),
    (1279, 65, 0.010976, 0.237835), (3262, 853, 0.019027, 0.770808),
    (1970, 685, 0.011095, 0.511682), (7817, 2793, 0.032498, 1.967491),
    (1252, 42, 0.008495, 0.213063), (734, 30, 0.006942, 0.139301),
    (273, 25, 0.00648, 0.066356), (453, 76, 0.006126, 0.111157),
    (2887, 1591, 0.01444, 0.914395), (1042, 129, 0.01011, 0.203765),
    (1090, 414, 0.009478, 0.317881), (178, 19, 0.004881, 0.046975),
    (2305, 1183, 0.014517, 0.717112), (1252, 418, 0.011118, 0.318804),
    (2005, 815, 0.012744, 0.551996),
]


def _law(kind: str, *, with_floor: bool = False, min_cycles: int | None = LARGE_ONLY_MIN_CYCLES):
    samples = [CostSample(seconds=s, cycles=2, words=w, concurrency=1, kind=ProbeKind.LOAD)
               for w, s in LOAD_LADDER[kind]]
    if with_floor:
        w, s = min(LOAD_LADDER[kind])
        samples.append(CostSample(seconds=s, cycles=2, words=w, concurrency=1, kind=ProbeKind.FLOOR))
    samples += [CostSample(seconds=s, cycles=c, words=w, concurrency=1, kind=ProbeKind.CYCLE)
                for c, w, s in CYCLE_LADDER[kind]]
    return fit_cost_law(samples, substrate=kind, cycle_fit_min_cycles=min_cycles)


# --- fixture reproduction ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(PUBLISHED))
def test_reproduces_the_published_two_term_law(kind):
    law = _law(kind)
    want = PUBLISHED[kind]
    assert law.per_cycle.value * 1e3 == pytest.approx(want["ms_per_cycle"], rel=1e-5)
    assert law.cycles_per_second == pytest.approx(want["cycles_per_s"], rel=1e-3)
    assert law.per_word.value * 1e3 == pytest.approx(want["ms_per_word"], rel=1e-5)
    assert law.words_per_second == pytest.approx(want["words_per_s"], rel=1e-3)
    assert law.concurrency == 1


def test_both_terms_are_isolated_by_construction_not_by_regression():
    law = _law("elaborated_rtl")
    assert law.per_word.provenance is Provenance.MEASURED
    assert "halt-first" in law.per_word.construction
    assert law.per_cycle.provenance is Provenance.MEASURED
    assert "pinned program size" in law.per_cycle.construction


def test_the_slow_substrate_reproduces_its_published_headline_rates():
    law = _law("cycle_model")
    assert round(law.cycles_per_second, 0) == 644.0
    assert round(law.per_cycle.value * 1e3, 3) == 1.553
    assert round(law.per_word.value * 1e3, 3) == 3.105


def test_the_word_term_is_about_twice_the_cycle_term_on_both_substrates():
    # the structural fact that makes a cycles-only fit wrong rather than merely imprecise
    for kind in PUBLISHED:
        law = _law(kind)
        assert law.per_word.value / law.per_cycle.value == pytest.approx(2.0, abs=0.05)


def test_the_large_cycle_slope_matches_the_whole_range_so_the_law_has_not_bent():
    whole = _law("elaborated_rtl", min_cycles=None)
    large = _law("elaborated_rtl")
    assert whole.per_cycle.r2 > 0.9999
    assert large.per_cycle.value == pytest.approx(whole.per_cycle.value, rel=0.005), (
        "a slope that changes on the large-cycle subset means the law has bent")


@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_linear_across_a_12500x_cycle_range_at_r2_0_99997():
    """The claim that makes a projection defensible: no bend anywhere in four orders of magnitude."""
    corpus = _corpus_samples()["elaborated_rtl"]
    cycle_samples = [CostSample(seconds=s, cycles=c, words=w, concurrency=1, kind=ProbeKind.CYCLE)
                     for c, w, s in CYCLE_LADDER["elaborated_rtl"]]
    # the corpus points sit BELOW the synthetic ladder's floor, extending the range down to 178
    corpus_as_cycle = [CostSample(seconds=s.seconds, cycles=s.cycles, words=s.words,
                                  concurrency=1, kind=ProbeKind.CYCLE, label=s.label)
                       for s in corpus]
    law = fit_cost_law(
        [CostSample(seconds=s, cycles=2, words=w, concurrency=1, kind=ProbeKind.LOAD)
         for w, s in LOAD_LADDER["elaborated_rtl"]] + cycle_samples + corpus_as_cycle,
        substrate="elaborated_rtl")
    lo, hi = law.per_cycle.domain
    assert hi / lo == pytest.approx(12_547, rel=0.01)
    assert law.per_cycle.r2 == pytest.approx(0.99997, abs=2e-5)
    assert law.per_cycle.value * 1e3 == pytest.approx(0.130531, rel=1e-3), (
        "the published whole-range slope, within 0.3% of the large-cycle-only slope")


# --- the mistake the tool exists to prevent ---------------------------------------------------------

@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_a_cycles_only_fit_overstates_the_per_cycle_rate_by_1_77x():
    corpus = _corpus_samples()["elaborated_rtl"]
    naive = cycles_only_slope(corpus)
    true_rate = _law("elaborated_rtl").per_cycle.value
    assert naive * 1e3 == pytest.approx(0.23121, rel=1e-3), "the published cycles-only slope"
    assert naive / true_rate == pytest.approx(1.77, abs=0.01)


@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_the_law_reports_the_naive_fit_alongside_its_own():
    corpus = _corpus_samples()["elaborated_rtl"]
    law = _law("elaborated_rtl")
    combined = fit_cost_law(
        [*_ladder_samples("elaborated_rtl"), *corpus],
        substrate="elaborated_rtl", cycle_fit_min_cycles=LARGE_ONLY_MIN_CYCLES)
    assert combined.per_cycle.value == pytest.approx(law.per_cycle.value, rel=1e-9), (
        "adding correlated corpus samples must not move a term isolated by construction")
    assert combined.cycles_only_overstatement == pytest.approx(1.77, abs=0.01)
    assert "1.77x" in render_law(combined)


def test_without_a_load_probe_the_word_term_is_unknown_and_the_cycle_term_is_flagged():
    corpus_like = [CostSample(seconds=s, cycles=c, words=w, concurrency=1)
                   for c, w, s, _ in GSIM_CORPUS]
    law = fit_cost_law(corpus_like, substrate="no-load-probe")
    assert law.per_word.provenance is Provenance.UNKNOWN
    assert law.per_word.value is None, "an unmeasured term must never be readable as 0.0"
    assert law.per_cycle.provenance is Provenance.DERIVED
    assert any("NO LOAD PROBE" in n for n in law.notes)
    est = law.estimate(cycles=10_000, words=5_000)
    assert "per_word" in est.excluded and est.is_lower_bound
    assert "LOWER BOUND" in str(est)


# --- held-out validation ----------------------------------------------------------------------------

def _corpus_samples() -> dict[str, list[CostSample]]:
    """The 52 held-out corpus queries (26 capsules x 2 substrates), replicates averaged."""
    tier_kind = {"L3": "cycle_model", "L4": "elaborated_rtl"}
    raw = json.loads(LAYERSCALE.read_text())
    out: dict[str, list[CostSample]] = defaultdict(list)
    for s in raw["phases"]["A_corpus_sweep"]["samples"]:
        if s.get("status") != "ok":
            continue
        kind = tier_kind[s["tier"]]
        out[kind].append(CostSample(seconds=s["sim_s"], cycles=s["cycles"],
                                    words=s["kernel_words"], concurrency=raw["concurrency"],
                                    kind=ProbeKind.CORPUS, label=s["capsule"]))
    return {k: average_replicates(v) for k, v in out.items()}


def _ladder_samples(kind: str) -> list[CostSample]:
    return ([CostSample(seconds=s, cycles=2, words=w, concurrency=1, kind=ProbeKind.LOAD)
             for w, s in LOAD_LADDER[kind]]
            + [CostSample(seconds=s, cycles=c, words=w, concurrency=1, kind=ProbeKind.CYCLE)
               for c, w, s in CYCLE_LADDER[kind]])


@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_predicts_52_held_out_corpus_queries_at_2_8_percent_median_error():
    corpus = _corpus_samples()
    errs = []
    for kind, samples in corpus.items():
        law = _law(kind)
        # neither ladder reaches the origin credibly, so the fixed term is honestly UNKNOWN and the
        # prediction is the two rate terms -- exactly what the published artifact predicted with.
        assert law.fixed.provenance is Provenance.UNKNOWN
        v = law.validate(samples)
        errs += [abs(r["rel_err"]) for r in v.rows]
    assert len(errs) == 52
    errs.sort()
    median = (errs[25] + errs[26]) / 2
    assert median == pytest.approx(0.0278, abs=0.0005)
    assert max(errs) == pytest.approx(0.2721, abs=0.001)


@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_a_floor_probe_measures_the_fixed_term_and_halves_the_worst_case_error():
    corpus = _corpus_samples()
    worst = 0.0
    for kind, samples in corpus.items():
        law = _law(kind, with_floor=True)
        assert law.fixed.provenance is Provenance.MEASURED
        assert law.fixed.value > 0
        worst = max(worst, law.validate(samples).max_abs_rel_err)
    assert worst < 0.15, "a measured fixed term should pull in the small-query tail"


# --- a second, independent pair of substrates -------------------------------------------------------

def _gsim_law(col: int, substrate: str) -> CostLaw:
    samples = [CostSample(seconds=row[col], cycles=1478, words=row[0], concurrency=1,
                          kind=ProbeKind.LOAD) for row in GSIM_PAD_LADDER]
    samples += [CostSample(seconds=row[col + 1], cycles=row[0], words=row[1], concurrency=1,
                           kind=ProbeKind.CORPUS) for row in GSIM_CORPUS]
    return fit_cost_law(samples, substrate=substrate)


def test_reproduces_the_fast_simulator_fixture():
    law = _gsim_law(1, "fast-firrtl-sim")
    assert law.cycles_per_second == pytest.approx(377_334.7, rel=1e-4)
    assert law.per_cycle.value * 1e3 == pytest.approx(0.00265017, rel=1e-4)
    assert law.words_per_second == pytest.approx(405_167.6, rel=1e-4)
    assert law.fixed.value == pytest.approx(0.005345818, rel=1e-4)
    assert law.per_cycle.r2 == pytest.approx(0.9225, abs=5e-4)
    assert law.per_word.r2 == pytest.approx(0.9692, abs=5e-4)
    # no cycle ladder here: the cycle term rests on the word term having been removed correctly.
    assert law.per_cycle.provenance is Provenance.DERIVED
    assert law.per_word.provenance is Provenance.MEASURED


def test_the_remeasure_of_the_slow_simulator_lands_within_three_percent():
    law = _gsim_law(2, "elaborated_rtl_remeasured")
    assert law.cycles_per_second == pytest.approx(7441.5, rel=1e-3)
    assert law.fixed.value == pytest.approx(0.0282753, rel=1e-4)
    prior = PUBLISHED["elaborated_rtl"]["cycles_per_s"]
    assert abs(law.cycles_per_second - prior) / prior < 0.03, (
        "an independent re-measure of the same substrate must land inside a few percent")


def test_the_speed_ratio_between_the_two_simulators():
    fast, slow = _gsim_law(1, "fast"), _gsim_law(2, "slow")
    assert fast.cycles_per_second / slow.cycles_per_second == pytest.approx(50.7, abs=0.1)
    assert fast.words_per_second / slow.words_per_second == pytest.approx(130.1, abs=0.2)


# --- concurrency is structurally impossible to omit -------------------------------------------------

def test_a_sample_cannot_be_built_without_its_concurrency():
    with pytest.raises(TypeError):
        CostSample(seconds=1.0, cycles=10, words=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        CostSample(seconds=1.0, cycles=10, words=10, concurrency=0)


def test_pooling_two_concurrencies_into_one_fit_is_refused():
    samples = _ladder_samples("elaborated_rtl")
    samples.append(CostSample(seconds=23.4, cycles=1090, words=414, concurrency=16,
                              kind=ProbeKind.CORPUS))
    with pytest.raises(MixedConcurrency):
        fit_cost_law(samples, substrate="mixed")


def test_validating_a_serial_law_against_contended_samples_is_refused():
    law = _law("elaborated_rtl")
    contended = [CostSample(seconds=23.4, cycles=1090, words=414, concurrency=16)]
    with pytest.raises(MixedConcurrency):
        law.validate(contended)


def test_every_cost_number_carries_its_concurrency():
    law = _law("elaborated_rtl")
    est = law.estimate(cycles=100_000, words=1_000)
    assert est.concurrency == 1
    assert "concurrency=1" in str(est)
    assert est.as_dict()["concurrency"] == 1
    assert "CONCURRENCY = 1" in render_law(law)


def test_the_16_worker_inflation_fixture():
    """3.7 s serial vs 23.4 s under 16 workers — 6.3x, measured on the same query."""
    rep = concurrency_inflation(serial_seconds=3.724, observed_seconds=23.433, workers=16)
    assert rep.inflation_x == pytest.approx(6.29, abs=0.01)
    assert "throughput figure" in rep.note
    fast = concurrency_inflation(serial_seconds=0.27, observed_seconds=0.545, workers=16)
    assert fast.inflation_x == pytest.approx(2.02, abs=0.01)


# --- extrapolation ----------------------------------------------------------------------------------

def test_a_projection_reports_how_far_past_the_evidence_it_reaches():
    slow, fast = _law("cycle_model"), _law("elaborated_rtl")
    # a real layer: 198,000 cycles. Beyond the slow substrate's largest timed run, well inside the fast
    # substrate's -- the same projection is an extrapolation on one and interpolation on the other.
    s_est = slow.estimate(cycles=198_000, words=1_000)
    f_est = fast.estimate(cycles=198_000, words=1_000)
    assert s_est.extrapolation["cycles"] == pytest.approx(1.4, abs=0.05)
    assert not s_est.within_measured_domain
    assert "EXTRAPOLATED" in str(s_est)
    assert f_est.extrapolation["cycles"] == pytest.approx(0.09, abs=0.01)
    assert f_est.within_measured_domain
    assert s_est.seconds / f_est.seconds == pytest.approx(
        PUBLISHED["cycle_model"]["ms_per_cycle"] / PUBLISHED["elaborated_rtl"]["ms_per_cycle"], rel=0.1)


def test_the_measured_versus_assumed_split_is_explicit():
    law = _law("elaborated_rtl", with_floor=True)
    est = law.estimate(cycles=1_000, words=500)
    assert set(est.measured) == {"fixed", "per_cycle", "per_word"}
    assert est.assumed == () and est.excluded == ()
    assert set(est.by_term) == {"fixed", "per_cycle", "per_word"}
    assert est.seconds == pytest.approx(sum(est.by_term.values()))


def test_an_unmeasured_axis_is_infinite_extrapolation_not_a_pass():
    law = fit_cost_law(
        [CostSample(seconds=s, cycles=2, words=w, concurrency=1, kind=ProbeKind.LOAD)
         for w, s in LOAD_LADDER["elaborated_rtl"]],
        substrate="words-only")
    assert law.per_cycle.provenance is Provenance.UNKNOWN
    est = law.estimate(cycles=1_000_000, words=100)
    assert est.extrapolation["cycles"] == float("inf")
    assert "per_cycle" in est.excluded


# --- the measurement driver -------------------------------------------------------------------------

class _FakeSubstrate:
    """A stand-in device: 1 ms per word loaded, 0.1 ms per cycle, 20 ms fixed."""

    name = "fake"
    concurrency = 1

    def run(self, program) -> RunOutcome:
        words, cycles = program
        return RunOutcome(seconds=0.020 + 1e-3 * words + 1e-4 * cycles, cycles=cycles, words=words)


def test_the_driver_recovers_a_known_law_end_to_end():
    sub = _FakeSubstrate()
    probes = halt_first_probes(lambda n: (n, 1), [1, 100, 1_000, 10_000])
    probes += [Probe(program=(64, c), kind=ProbeKind.CYCLE, label=f"loop_{c}")
               for c in (100, 1_000, 10_000, 100_000)]
    law = fit_cost_law(measure(sub, probes, reps=2), substrate=sub.name)
    assert law.per_word.value == pytest.approx(1e-3, rel=1e-6)
    assert law.per_cycle.value == pytest.approx(1e-4, rel=1e-6)
    assert law.fixed.value == pytest.approx(0.020, abs=1e-6)
    assert law.fixed.provenance is Provenance.MEASURED
    assert law.concurrency == 1


def test_the_driver_stamps_the_substrate_concurrency_on_every_sample():
    sub = _FakeSubstrate()
    sub.concurrency = 8
    samples = measure(sub, halt_first_probes(lambda n: (n, 1), [1, 10]))
    assert {s.concurrency for s in samples} == {8}
    assert fit_cost_law(
        samples + [CostSample(seconds=1.0, cycles=c, words=64, concurrency=8, kind=ProbeKind.CYCLE)
                   for c in (100, 1000)],
        substrate="fake").concurrency == 8


def test_a_load_ladder_needs_two_distinct_rungs():
    with pytest.raises(NotEnoughEvidence):
        halt_first_probes(lambda n: (n, 1), [10, 10])
    with pytest.raises(NotEnoughEvidence):
        fit_cost_law([], substrate="empty")


# --- replicate averaging ----------------------------------------------------------------------------

def test_replicates_of_one_query_collapse_to_their_mean():
    reps = [CostSample(seconds=2.7803, cycles=1478, words=138, concurrency=1, label="k"),
            CostSample(seconds=2.7632, cycles=1478, words=138, concurrency=1, label="k")]
    merged = average_replicates(reps)
    assert len(merged) == 1
    assert merged[0].seconds == pytest.approx(2.77175)


@pytest.mark.skipif(not GSIM.exists(), reason=f"DID NOT RUN: fixture absent at {GSIM}")
def test_the_inlined_fixture_rows_match_the_published_artifact():
    """Guards the inlined ladders against drift from the artifact they were transcribed from."""
    raw = json.loads(GSIM.read_text())
    pad = [(r["words"], r["gsim_wall_s"], r["vl_wall_s"]) for r in raw["phases"]["B_inert_pad_word_ladder"]]
    assert pad == GSIM_PAD_LADDER
    assert {r["gsim_cycles"] for r in raw["phases"]["B_inert_pad_word_ladder"]} == {1478}
    corpus = [(r["gsim_cycles"], r["words"], r["gsim_wall_s"], r["vl_wall_s"])
              for r in raw["phases"]["A_corpus_head_to_head"]]
    assert corpus == GSIM_CORPUS
    assert raw["Q1_speed"]["gsim"]["cycles_per_s"] == pytest.approx(377_334.7, rel=1e-5)
    assert raw["Q1_speed"]["verilator_remeasured"]["cycles_per_s"] == pytest.approx(7441.5, rel=1e-4)


@pytest.mark.skipif(not LAYERSCALE.exists(), reason=f"DID NOT RUN: fixture absent at {LAYERSCALE}")
def test_the_inlined_ladders_match_the_published_artifact():
    raw = json.loads(LAYERSCALE.read_text())
    tier_kind = {"L3": "cycle_model", "L4": "elaborated_rtl"}
    load = defaultdict(list)
    for s in raw["phases"]["D_imem_load"]["samples"]:
        if s["kind"] == "load_only":
            load[tier_kind[s["tier"]]].append((s["program_words"], s["wall_s"]))
    assert {k: sorted(v) for k, v in load.items()} == {k: sorted(v) for k, v in LOAD_LADDER.items()}
    cyc = defaultdict(list)
    for s in raw["phases"]["C_ladder"]["samples"]:
        cyc[tier_kind[s["tier"]]].append((s["cycles"], s["program_words"], s["sim_s"]))
    assert {k: sorted(v) for k, v in cyc.items()} == {k: sorted(v) for k, v in CYCLE_LADDER.items()}
