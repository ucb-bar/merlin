"""The per-capsule verdict must decide on evidence, and refuse rather than assume."""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import perf_capsule_verdict as V  # noqa: E402


def _call(**over):
    args = dict(capsule="PK00_k16", declared_macs=4096, achievable_rate=80.0,
                baseline_cycles=303, candidate_cycles=303, dispersion=0.01)
    args.update(over)
    return V.capsule_verdict(**args)


def test_a_capsule_at_the_ceiling_is_finished_not_a_failure():
    # ideal == baseline, so share is 1.0
    out = _call(declared_macs=8000, achievable_rate=80.0, baseline_cycles=100, candidate_cycles=100)
    assert out["verdict"] == V.NO_HEADROOM
    assert "nothing on this machine" in out["reason"]


def test_a_real_saving_is_improved_and_reports_the_gap_it_closed():
    out = _call(baseline_cycles=400, candidate_cycles=300)
    assert out["verdict"] == V.IMPROVED
    assert out["cycles_saved"] == 100
    assert 0.0 < out["gap_closed"] <= 1.0


def test_a_slower_candidate_is_regressed_never_improved():
    out = _call(baseline_cycles=300, candidate_cycles=400)
    assert out["verdict"] == V.REGRESSED
    assert out["verdict"] != V.IMPROVED


def test_no_change_leaves_headroom_open_and_quantifies_what_remains():
    out = _call(baseline_cycles=400, candidate_cycles=400)
    assert out["verdict"] == V.HEADROOM_OPEN
    assert out["factor_to_achievable"] > 1.0


def test_a_saving_inside_the_replicate_dispersion_is_not_a_win():
    out = _call(baseline_cycles=400, candidate_cycles=398, replicate_dispersion=5.0)
    assert out["verdict"] == V.HEADROOM_OPEN


def test_a_deterministic_oracle_counts_every_saved_cycle():
    # replicate dispersion 0 (cycle-accurate, deterministic) => one cycle is a real saving
    out = _call(baseline_cycles=400, candidate_cycles=399, replicate_dispersion=0.0)
    assert out["verdict"] == V.IMPROVED


@pytest.mark.parametrize("field", ["declared_macs", "achievable_rate", "baseline_cycles"])
def test_an_underivable_input_refuses_and_never_substitutes(field):
    out = _call(**{field: None})
    assert out["verdict"] == V.REFUSED
    assert field in out["reason"]
    assert "baseline_share_of_achievable" not in out


def test_an_unmeasurable_dispersion_refuses_rather_than_assuming_zero():
    out = _call(dispersion=None)
    assert out["verdict"] == V.REFUSED
    assert "tolerance" in out["reason"]


def test_a_missing_candidate_is_open_headroom_not_a_pass():
    out = _call(candidate_cycles=None)
    assert out["verdict"] == V.HEADROOM_OPEN


def test_dispersion_is_measured_from_the_points_and_refuses_below_two():
    assert V.ceiling_dispersion([{"macs": 100, "cycles": 10}]) is None
    d = V.ceiling_dispersion([{"macs": 100, "cycles": 10}, {"macs": 90, "cycles": 10},
                              {"macs": 50, "cycles": 10}])
    assert d is not None and 0.0 <= d < 1.0


def test_the_rollup_never_lets_a_refusal_read_as_decided():
    rows = [_call(baseline_cycles=400, candidate_cycles=300), _call(declared_macs=None)]
    s = V.summarize(rows)
    assert s["n_capsules"] == 2 and s["n_refused"] == 1 and s["n_decided"] == 1


def test_worst_first_ranks_by_remaining_factor():
    a = _call(capsule="lots", baseline_cycles=4000, candidate_cycles=4000)
    b = _call(capsule="little", baseline_cycles=200, candidate_cycles=200)
    order = V.summarize([b, a])["worst_first"]
    assert order[0] == "lots"
