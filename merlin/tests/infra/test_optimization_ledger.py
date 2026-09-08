"""A ledger row that asserts a measurement must carry one, and `unmeasured` is not `no_effect`.

The failure this guards is the one the previous campaign shipped: receipts saying
`global_speedup_proven: False` with `probe_receipts: []`, where nothing distinguished "measured and
it did not help" from "never measured". A report in which those look alike is worse than no report.
"""
from __future__ import annotations

import json

import pytest

from merlin.perf import optimization_ledger as OL
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


class TestPersistence:
    """A ledger that cannot be written and read back is not a campaign artifact."""

    def _attempt(self, **kw):
        base = dict(mechanism="m", scope="local", found_by="work_volume", verdict="helped",
                    deltas=(OL.Delta(workload="w", metric="cycles", before=100.0, after=50.0,
                                     instrument="firesim"),))
        base.update(kw)
        return OL.Attempt(**base)

    def test_a_ledger_round_trips_through_disk_unchanged(self, tmp_path):
        led = OL.Ledger(target="t", attempts=[self._attempt(iteration=0),
                                              self._attempt(verdict="blocked", blocked_by="upstream",
                                                            deltas=(), iteration=1)])
        path = led.write(tmp_path / "ledger.json")
        assert OL.read_ledger(path).to_dict() == led.to_dict()

    def test_a_missing_file_is_an_empty_campaign_not_an_error(self, tmp_path):
        led = OL.read_ledger(tmp_path / "absent.json", target="t")
        assert led.attempts == [] and led.target == "t"

    def test_a_corrupt_file_raises_rather_than_reading_as_empty(self, tmp_path):
        """"The campaign tried nothing" and "the record is corrupt" must not look the same."""
        path = tmp_path / "ledger.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="exists but could not be read"):
            OL.read_ledger(path)

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({"schema": "something_else_v9", "attempts": []}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="refusing to read a ledger"):
            OL.read_ledger(path)

    def test_appending_accumulates_across_immutable_iterations(self, tmp_path):
        """The campaign's iteration records are chmod 0444; the history has to live somewhere else."""
        path = tmp_path / "ledger.json"
        OL.append_attempts(path, [self._attempt(iteration=0)], target="t")
        led = OL.append_attempts(path, [self._attempt(iteration=1)], target="t")
        assert len(led.attempts) == 2
        assert [a.iteration for a in OL.read_ledger(path).attempts] == [0, 1]

    def test_appending_a_different_target_refuses(self, tmp_path):
        path = tmp_path / "ledger.json"
        OL.append_attempts(path, [self._attempt()], target="t")
        with pytest.raises(ValueError, match="mix two machines"):
            OL.append_attempts(path, [self._attempt()], target="other")

    def test_a_stored_problems_list_cannot_launder_an_unsound_row(self, tmp_path):
        """`problems` is DERIVED on read, so a row written unsound stays unsound."""
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({
            "schema": OL.SCHEMA, "target": "t",
            "attempts": [{"mechanism": "m", "scope": "local", "found_by": "",
                          "verdict": "helped", "deltas": [], "problems": []}]}), encoding="utf-8")
        led = OL.read_ledger(path)
        assert led.problems(), "an evidence-free 'helped' row must still report a problem"

    def test_a_stored_ratio_is_recomputed_never_trusted(self, tmp_path):
        """Otherwise the number a reader reports is whichever the writer's arithmetic produced."""
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({
            "schema": OL.SCHEMA, "target": "t",
            "attempts": [{"mechanism": "m", "scope": "local", "found_by": "i", "verdict": "helped",
                          "deltas": [{"workload": "w", "metric": "cycles", "before": 100,
                                      "after": 50, "instrument": "i", "ratio": 999.0}]}]}),
            encoding="utf-8")
        assert OL.read_ledger(path).attempts[0].deltas[0].ratio == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", ["not-a-number", float("nan"), float("inf"), True])
    def test_a_non_numeric_metric_value_is_refused_never_coerced(self, tmp_path, bad):
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({
            "schema": OL.SCHEMA, "target": "t",
            "attempts": [{"mechanism": "m", "scope": "local", "found_by": "i", "verdict": "helped",
                          "deltas": [{"workload": "w", "metric": "cycles", "before": 100,
                                      "after": bad, "instrument": "i"}]}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a number or null|must be finite"):
            OL.read_ledger(path)

    def test_a_write_is_atomic_so_a_reader_never_sees_a_partial_ledger(self, tmp_path):
        path = tmp_path / "ledger.json"
        OL.Ledger(target="t", attempts=[self._attempt()]).write(path)
        assert not list(tmp_path.glob("*.partial")), "the temporary must be replaced, not left"
        assert OL.read_ledger(path).attempts
