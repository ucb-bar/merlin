"""Every way a two-arm comparison has already lied on this tree, made unrepeatable.

Two are measured history rather than hypotheses:

* the last phase-2 campaign sealed a performance hypothesis on a candidate whose own receipt said
  ``command_buffer_identical: True`` and ``lowered_identical: True`` -- the same emitted program;
* a 1.642x improvement was measured on host instruction count across arms whose traffic was
  byte-identical (169.52 MiB on every one), while the axis under optimization was data movement.

The rest pin the licences the instruments themselves declare: a band eliminates and never certifies,
an unmeasured change is not an ineffective one, and a single flattering instrument may not carry a
verdict another usable one contradicts.
"""
from __future__ import annotations

import pytest

from merlin.perf import candidate_decision as CD
from merlin.perf.candidate_decision import Axis, CandidateFacts, Metric


def _arm(name, buffer="", lowered=""):
    return CandidateFacts(name=name, command_buffer_sha256=buffer, lowered_sha256=lowered)


BASE = _arm("baseline", "a" * 64, "b" * 64)
CAND = _arm("candidate", "c" * 64, "d" * 64)


def _metric(name="cycles", *, base=100.0, cand=50.0, blind=(), instrument="firesim"):
    return Metric(name=name, instrument=instrument, baseline=base, candidate=cand, blind_to=blind)


class TestTheSameProgramIsNotAComparison:
    def test_an_identical_command_buffer_ends_the_comparison(self):
        """What the last campaign shipped a performance hypothesis on top of."""
        same = _arm("candidate", BASE.command_buffer_sha256, "d" * 64)
        got = CD.compare(BASE, same, axes=[Axis("host_work", True)], metrics=[_metric()])
        assert got.verdict == CD.IDENTICAL_EMISSION
        assert "same emitted program" in got.why
        assert got.decided_by == "", "no instrument may be credited for an identical pair"

    def test_an_identical_lowered_module_ends_it_too(self):
        same = _arm("candidate", "c" * 64, BASE.lowered_sha256)
        got = CD.compare(BASE, same, axes=[Axis("host_work", True)], metrics=[_metric()])
        assert got.verdict == CD.IDENTICAL_EMISSION and "lowered module" in got.why

    def test_it_is_checked_BEFORE_any_metric_however_good_the_ratio(self):
        same = _arm("candidate", BASE.command_buffer_sha256)
        got = CD.compare(BASE, same, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric(base=1000.0, cand=1.0)])
        assert got.verdict == CD.IDENTICAL_EMISSION

    def test_absent_digests_do_not_count_as_matching(self):
        """Two arms with no digest recorded are UNKNOWN emission, not identical emission."""
        got = CD.compare(_arm("b"), _arm("c"), axes=[Axis("host_work", True)],
                         metrics=[_metric()])
        assert got.verdict != CD.IDENTICAL_EMISSION


class TestAnInstrumentBlindToTheChangeCannotDecide:
    def test_a_metric_blind_to_every_moved_axis_is_excluded_and_the_change_is_BLIND(self):
        """THE 1.642x FAILURE. Host instructions cannot decide a traffic change."""
        got = CD.compare(BASE, CAND, axes=[Axis("traffic", True, "169.52 MiB on both arms")],
                         metrics=[_metric("host_instructions", base=914191598.0,
                                          cand=556761305.0, blind=("traffic",),
                                          instrument="spike")])
        assert got.verdict == CD.BLIND
        assert "unmeasured change, NOT an ineffective one" in got.why
        assert got.excluded[0]["metric"] == "host_instructions"
        # The number is kept, so a reader can see what was excluded and why.
        assert got.excluded[0]["ratio"] == pytest.approx(914191598.0 / 556761305.0)

    def test_a_partially_blind_metric_is_kept_but_flagged(self):
        got = CD.compare(BASE, CAND,
                         axes=[Axis("traffic", True), Axis("host_work", True)],
                         metrics=[_metric("host_instructions", blind=("traffic",))])
        assert got.verdict == CD.BETTER
        flagged = [e for e in got.excluded if e.get("kept")]
        assert flagged and "partially blind" in flagged[0]["reason"]

    def test_a_sighted_metric_beside_a_blind_one_still_decides(self):
        got = CD.compare(BASE, CAND, axes=[Axis("traffic", True)],
                         metrics=[_metric("host_instructions", blind=("traffic",)),
                                  _metric("bytes_moved", base=200.0, cand=100.0,
                                          instrument="lane_cost")])
        assert got.verdict == CD.BETTER and got.decided_by == "lane_cost"

    def test_an_undeclared_axis_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="is not one of"):
            Axis("vibes", True)
        with pytest.raises(ValueError, match="is not one of"):
            Metric("m", "i", 1.0, 1.0, blind_to=("vibes",))

    def test_a_duplicated_axis_is_refused(self):
        with pytest.raises(ValueError, match="declared twice"):
            CD.compare(BASE, CAND, axes=[Axis("traffic", True), Axis("traffic", False)],
                       metrics=[_metric()])


class TestBandsEliminateAndNeverCertify:
    def _band(self, *, b, c, name="band"):
        return Metric(name=name, instrument="compose_estimate.band", baseline=sum(b) / 2,
                      candidate=sum(c) / 2, baseline_interval=b, candidate_interval=c)

    def test_overlapping_bands_are_UNKNOWN_not_a_tie(self):
        got = CD.compare(BASE, CAND, axes=[Axis("offload", True)],
                         metrics=[self._band(b=(100.0, 5000.0), c=(120.0, 6000.0))])
        assert got.verdict == CD.UNKNOWN
        assert any("may never certify one" in e["reason"] for e in got.excluded)

    def test_a_disjoint_band_ABOVE_the_baseline_eliminates_the_candidate(self):
        got = CD.compare(BASE, CAND, axes=[Axis("offload", True)],
                         metrics=[self._band(b=(100.0, 200.0), c=(300.0, 400.0))])
        assert got.verdict == CD.WORSE
        assert got.decided_by == "compose_estimate.band"
        assert "eliminates it" in got.why

    def test_a_disjoint_band_FAVOURING_the_candidate_is_still_not_a_win(self):
        """It eliminates the baseline; certifying the candidate is outside the licence."""
        got = CD.compare(BASE, CAND, axes=[Axis("offload", True)],
                         metrics=[self._band(b=(300.0, 400.0), c=(100.0, 200.0))])
        assert got.verdict == CD.UNKNOWN
        assert "may not certify a candidate" in got.why

    def test_a_point_metric_beside_a_favouring_band_does_decide(self):
        got = CD.compare(BASE, CAND, axes=[Axis("offload", True)],
                         metrics=[self._band(b=(300.0, 400.0), c=(100.0, 200.0)),
                                  _metric("cycles", instrument="gsim")])
        assert got.verdict == CD.BETTER and got.decided_by == "gsim"

    def test_the_real_band_width_cannot_separate_two_close_arms(self):
        """At the measured 211.5x width, a 2x difference is invisible -- and must read as UNKNOWN."""
        got = CD.compare(BASE, CAND, axes=[Axis("offload", True)],
                         metrics=[self._band(b=(1000.0, 211500.0), c=(500.0, 105750.0))])
        assert got.verdict == CD.UNKNOWN


class TestTheVerdictQuotesTheWeakestImprovement:
    def test_agreeing_instruments_yield_BETTER_quoting_the_weakest(self):
        got = CD.compare(BASE, CAND, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric("a", base=100.0, cand=10.0, instrument="flattering"),
                                  _metric("b", base=100.0, cand=95.0, instrument="strict")])
        assert got.verdict == CD.BETTER and got.decided_by == "strict"
        assert "1.053x" in got.why and "cannot carry the verdict" in got.why

    def test_disagreeing_instruments_yield_UNKNOWN_and_name_both(self):
        got = CD.compare(BASE, CAND, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric("a", base=100.0, cand=50.0, instrument="one"),
                                  _metric("b", base=100.0, cand=200.0, instrument="two")])
        assert got.verdict == CD.UNKNOWN and got.decided_by == ""
        assert "the instruments disagree" in got.why and "happened to be quoted" in got.why

    def test_every_instrument_regressing_yields_WORSE(self):
        got = CD.compare(BASE, CAND, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric(base=100.0, cand=105.0),
                                  _metric("b", base=100.0, cand=200.0)])
        assert got.verdict == CD.WORSE and "the mildest is" in got.why

    def test_a_measured_zero_is_NO_EFFECT_and_says_it_is_not_unmeasured(self):
        got = CD.compare(BASE, CAND, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric(base=100.0, cand=100.0)])
        assert got.verdict == CD.NO_EFFECT
        assert "not the same as unmeasured" in got.why

    def test_no_axis_moving_is_NO_EFFECT_rather_than_a_ratio(self):
        got = CD.compare(BASE, CAND, axes=[Axis("traffic", False), Axis("offload", False)],
                         metrics=[_metric(base=100.0, cand=10.0)])
        assert got.verdict == CD.NO_EFFECT and "nothing for a metric to be about" in got.why

    def test_a_non_positive_reading_yields_no_ratio_and_no_verdict(self):
        got = CD.compare(BASE, CAND, axes=[Axis("wall_cycles", True)],
                         metrics=[_metric(base=0.0, cand=50.0)])
        assert got.verdict == CD.UNKNOWN and "usable ratio" in got.why


class TestTheRecordIsReadable:
    def test_the_decision_serialises_with_its_licence_and_moved_axes(self):
        got = CD.compare(BASE, CAND,
                         axes=[Axis("traffic", True, "169.52 -> 84.0 MiB"), Axis("offload", False)],
                         metrics=[_metric("bytes_moved", base=169.52, cand=84.0,
                                          instrument="lane_cost")])
        d = got.to_dict()
        assert d["schema"] == "merlin_candidate_decision_v1"
        assert d["verdict"] == CD.BETTER and d["moved_axes"] == ["traffic"]
        assert "may eliminate a candidate and may never certify one" in d["licence"]
        assert d["metrics"][0]["ratio"] == pytest.approx(169.52 / 84.0)

    def test_every_verdict_is_in_the_declared_vocabulary(self):
        cases = [
            (BASE, _arm("c", BASE.command_buffer_sha256), [Axis("host_work", True)], [_metric()]),
            (BASE, CAND, [Axis("traffic", False)], [_metric()]),
            (BASE, CAND, [Axis("traffic", True)], [_metric(blind=("traffic",))]),
            (BASE, CAND, [Axis("wall_cycles", True)], [_metric()]),
            (BASE, CAND, [Axis("wall_cycles", True)], [_metric(base=100.0, cand=200.0)]),
        ]
        for base, cand, axes, metrics in cases:
            assert CD.compare(base, cand, axes=axes, metrics=metrics).verdict in CD.VERDICTS
