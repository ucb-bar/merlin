"""A ledger row that asserts a measurement must carry one, and `unmeasured` is not `no_effect`.

The failure this guards is the one the previous campaign shipped: receipts saying
`global_speedup_proven: False` with `probe_receipts: []`, where nothing distinguished "measured and
it did not help" from "never measured". A report in which those look alike is worse than no report.
"""
from __future__ import annotations

from merlin.perf.optimization_ledger import (SCOPES, VERDICTS, Attempt, Delta, Ledger,
                                             arithmetic_intensity)


def _delta(**kw):
    base = {"workload": "w", "metric": "cycles", "before": 100.0, "after": 50.0,
            "instrument": "spike"}
    base.update(kw)
    return Delta(**base)


def test_a_cost_ratio_above_one_means_the_cost_fell():
    assert _delta().ratio == 2.0


def test_a_coverage_metric_inverts_so_more_is_still_better():
    d = _delta(metric="offload", before=0.5, after=1.0, lower_is_better=False)
    assert d.ratio == 2.0


def test_a_corrected_measurement_has_no_improvement_ratio():
    """Two beliefs about one program are not two programs."""
    d = _delta(metric="priced_macs", before=2048000, after=4089184256, lower_is_better=None)
    assert d.ratio is None, "a bug fix must not report as a 2000x regression"


def test_a_verdict_asserting_a_measurement_needs_a_delta_and_an_instrument():
    for verdict in ("helped", "no_effect", "refuted"):
        bare = Attempt("m", "local", "tool", verdict)
        assert any("carries no delta" in why for why in bare.problems()), verdict
        nameless = Attempt("m", "local", "", verdict, deltas=(_delta(),))
        assert any("names no instrument" in why for why in nameless.problems()), verdict
        sound = Attempt("m", "local", "tool", verdict, deltas=(_delta(),))
        assert sound.problems() == (), verdict


def test_unmeasured_needs_no_delta_and_is_not_no_effect():
    a = Attempt("m", "local", "", "unmeasured")
    assert a.problems() == ()
    assert "unmeasured" in VERDICTS and "no_effect" in VERDICTS
    assert a.verdict != "no_effect"


def test_blocked_must_say_what_blocks_it():
    assert any("must say what blocks" in why
               for why in Attempt("m", "local", "t", "blocked").problems())
    assert Attempt("m", "local", "t", "blocked", blocked_by="upstream has no scales").problems() == ()


def test_an_unknown_scope_or_verdict_is_a_problem_not_a_silent_pass():
    assert any("scope" in w for w in Attempt("m", "nowhere", "t", "unmeasured").problems())
    assert any("verdict" in w for w in Attempt("m", "local", "t", "great").problems())


def test_the_ledger_audits_verdicts_scopes_and_instruments():
    L = Ledger("t")
    L.add(Attempt("a", "local", "histogram", "helped", deltas=(_delta(),)))
    L.add(Attempt("b", "build", "histogram", "refuted", deltas=(_delta(after=200.0),)))
    L.add(Attempt("c", "global", "", "unmeasured"))
    assert L.by_verdict() == {"helped": 1, "refuted": 1, "unmeasured": 1}
    assert L.by_scope() == {"local": 1, "build": 1, "global": 1}
    assert L.instruments() == {"histogram": 2}, "an unmeasured row names no instrument"
    assert L.problems() == ()


def test_the_series_shows_advance_in_recorded_order():
    L = Ledger("t")
    for i, after in ((1, 90.0), (2, 80.0), (3, 75.0)):
        L.add(Attempt(f"m{i}", "local", "t", "helped", iteration=i,
                      deltas=(_delta(after=after),)))
    assert L.series("cycles", "w") == ((1, 90.0), (2, 80.0), (3, 75.0))
    assert L.series("cycles", "other") == ()


def test_a_ledger_problem_is_surfaced_with_its_row():
    L = Ledger("t")
    L.add(Attempt("unevidenced", "local", "t", "helped"))
    problems = L.problems()
    assert problems and "unevidenced" in problems[0]
    assert "PROBLEM" in L.format_table()


def test_bound_ness_is_refused_without_a_measured_machine_balance():
    got = arithmetic_intensity(4089184256, 177_000_000)
    assert got["status"] == "derived"
    assert got["bound_by"] == "UNKNOWN", "a ridge point is hardware, not an assumption"
    assert abs(got["macs_per_byte"] - 4089184256 / 177_000_000) < 1e-9


def test_bound_ness_is_decided_only_when_the_balance_is_supplied():
    dense = arithmetic_intensity(1000, 10, machine_macs_per_byte=50.0)
    sparse = arithmetic_intensity(10, 1000, machine_macs_per_byte=50.0)
    assert dense["bound_by"] == "compute" and sparse["bound_by"] == "memory"


def test_a_program_with_no_traffic_has_no_intensity():
    assert arithmetic_intensity(100, 0)["status"] == "unavailable"


def test_the_vocabularies_are_closed():
    assert "host_lane" in SCOPES and "frontend" in SCOPES and "transformation" in SCOPES
    assert set(VERDICTS) == {"helped", "no_effect", "refuted", "blocked", "unmeasured"}
