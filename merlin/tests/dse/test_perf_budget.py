"""Unit tests for :mod:`merlin.perf.budget` -- the measured scarce unit.

The regression these lock in: the budget unit is a MEASUREMENT, not a constant. The original design
rationed simulator queries; measurement says that on one target the oracle is a fraction of a percent
of what a datapoint costs, while on another the deep simulator dominates. Both directions are
exercised here with the same code, because "the tool reads its scarce unit from measurement" is the
generalizable claim and a test that only shows one direction does not test it.

No oracle runs, no simulator, no network: every price here is a supplied sample.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.perf import budget as bud
from merlin.perf.decompose import Unavailable, is_unknown
from merlin.perf.oracle_cost import CostLaw, Provenance, Term
from merlin.targetgen import tier_policy


# --- pricing a channel ------------------------------------------------------------------------------

def test_channel_from_samples_uses_the_median_not_the_mean():
    """Long-tailed by construction: the observed agent runs span 900-65,401 s. A mean over that is a
    number no single run ever cost."""
    c = bud.channel_from_samples("synthesis_call", seconds=[900.0, 2184.0, 65401.0],
                                 dollars=[0.51, 25.33, 103.17])
    assert c.seconds_per_item == 2184.0
    assert c.dollars_per_item == 25.33
    assert c.provenance == bud.MEASURED
    assert c.priced


def test_a_channel_with_no_samples_is_unpriced_not_zero():
    c = bud.channel_from_samples("deep_sim", seconds=[])
    assert c.seconds_per_item is None
    assert c.provenance == bud.UNPRICED
    assert not c.priced
    assert is_unknown(c.seconds_per_datapoint)


def test_items_per_datapoint_is_separate_from_the_per_item_price():
    """A cheap item run many times per datapoint is not cheap. Comparing per-item prices alone is how
    a fast oracle called 26 times reads as free."""
    c = bud.channel_from_samples("oracle_query", seconds=[0.276], items_per_datapoint=26)
    assert c.seconds_per_item == pytest.approx(0.276)
    assert c.seconds_per_datapoint == pytest.approx(0.276 * 26)


# --- which unit is scarce ---------------------------------------------------------------------------

def _synthesis_dominated() -> list[bud.Channel]:
    """One regime: a fast oracle, an expensive synthesis call."""
    return [
        bud.channel_from_samples("synthesis_call", seconds=[2184.0], dollars=[25.33]),
        bud.channel_from_samples("oracle_query", seconds=[0.276], items_per_datapoint=26),
    ]


def _simulation_dominated() -> list[bud.Channel]:
    """The other regime, same code: a slow deep simulator and the same synthesis call."""
    return [
        bud.channel_from_samples("synthesis_call", seconds=[2184.0], dollars=[25.33]),
        bud.channel_from_samples("deep_sim_query", seconds=[2700.0], items_per_datapoint=26),
    ]


def test_scarce_unit_is_read_from_measurement_and_flips_with_the_regime():
    """THE generalization claim: nothing about which unit is scarce is written down in the module."""
    a = bud.scarce_unit(_synthesis_dominated())
    b = bud.scarce_unit(_simulation_dominated())
    assert isinstance(a, bud.Channel) and a.name == "synthesis_call"
    assert isinstance(b, bud.Channel) and b.name == "deep_sim_query"


def test_the_measured_oracle_share_of_a_datapoint_is_a_fraction_of_a_percent():
    rep = bud.unit_report(_synthesis_dominated())
    assert rep.established
    share = rep.ratios["oracle_query"]
    assert 0.002 < float(share) < 0.004      # the measured 0.2-0.4% band


def test_scarce_unit_refuses_while_any_channel_is_unpriced():
    """An unpriced channel cannot be ruled out as the expensive one. Refusing is the whole point."""
    chans = [bud.channel_from_samples("synthesis_call", seconds=[2184.0]),
             bud.unpriced_channel("deep_sim", missing="never timed on this target")]
    out = bud.scarce_unit(chans)
    assert isinstance(out, Unavailable)
    assert any("deep_sim" in m for m in out.missing)


def test_scarce_unit_refuses_with_only_one_priced_channel():
    out = bud.scarce_unit([bud.channel_from_samples("only", seconds=[1.0])])
    assert isinstance(out, Unavailable)
    assert "comparison" in str(out)


def test_unit_report_ratios_are_unknown_when_the_unit_is_not_established():
    rep = bud.unit_report([bud.unpriced_channel("a", missing="x"),
                           bud.unpriced_channel("b", missing="y")])
    assert not rep.established
    assert all(is_unknown(v) for v in rep.ratios.values())


# --- pricing from a fitted cost law -------------------------------------------------------------------

def _law(*, with_word_term: bool = True, cycle_domain: float = 1_000_000.0) -> CostLaw:
    """A two-term law standing in for one fitted by ``oracle_cost.fit_cost_law``."""
    word = (Term("per_word", 0.000264, "s/word", Provenance.MEASURED, "halt-first probe", n=6,
                 domain=(0.0, 4096.0))
            if with_word_term else
            Term("per_word", None, "s/word", Provenance.UNKNOWN, "not isolated"))
    return CostLaw(
        substrate="fast_sim", concurrency=1,
        fixed=Term("fixed", 0.01, "s", Provenance.MEASURED, "floor probe", n=3),
        per_cycle=Term("per_cycle", 0.000131, "s/cycle", Provenance.MEASURED, "trip-count sweep",
                       n=8, domain=(178.0, cycle_domain)),
        per_word=word, n_samples=17)


def test_channel_from_cost_law_prices_a_query_from_the_fit():
    c = bud.channel_from_cost_law(_law(), cycles=10_000, words=512, items_per_datapoint=26)
    assert c.provenance == bud.MEASURED
    assert c.seconds_per_item == pytest.approx(0.01 + 0.000131 * 10_000 + 0.000264 * 512)
    assert c.notes == ()


def test_a_law_missing_a_term_prices_a_lower_bound_and_says_so():
    c = bud.channel_from_cost_law(_law(with_word_term=False), cycles=10_000, words=512)
    assert c.provenance == bud.PROJECTED
    assert any("LOWER BOUND" in n for n in c.notes)


def test_projecting_past_the_measured_domain_is_flagged_as_extrapolation():
    c = bud.channel_from_cost_law(_law(cycle_domain=1_000.0), cycles=100_000, words=512)
    assert c.provenance == bud.PROJECTED
    assert any("EXTRAPOLATED" in n for n in c.notes)


# --- the ledger --------------------------------------------------------------------------------------

def _budget(**limits) -> bud.Budget:
    b = bud.budget_from_channels(_synthesis_dominated(), **limits)
    assert isinstance(b, bud.Budget)
    return b


def test_budget_is_denominated_in_the_scarce_unit():
    b = _budget(limit_items=4)
    assert b.unit.name == "synthesis_call"


def test_budget_from_channels_propagates_the_refusal():
    out = bud.budget_from_channels([bud.unpriced_channel("a", missing="x"),
                                    bud.channel_from_samples("b", seconds=[1.0])], limit_items=3)
    assert isinstance(out, Unavailable)


def test_charging_spends_seconds_and_dollars_at_the_measured_price():
    b = _budget(limit_items=3)
    b.charge(label="cand_a")
    assert b.spent_items == 1.0
    assert b.spent_seconds == pytest.approx(2184.0)
    assert b.spent_dollars == pytest.approx(25.33)


def test_budget_exhausts_on_whichever_cap_binds_first():
    b = _budget(limit_items=10, limit_dollars=30.0)
    b.charge(label="a")
    assert not b.exhausted
    b.charge(label="b")
    assert b.exhausted
    assert "$" in b.exhausted_reason


def test_can_afford_refuses_without_calling_it_a_verdict_on_the_candidate():
    b = _budget(limit_items=1)
    b.charge(label="a")
    ok, why = b.can_afford()
    assert not ok
    assert "did not run" in why


def test_charging_also_credits_the_shared_tier_policy_ledger():
    tier_policy.reset_spend()
    b = bud.Budget(unit=_synthesis_dominated()[0], limit_items=2, target="tgt_under_test")
    b.charge(label="a")
    assert tier_policy.spent("tgt_under_test") == pytest.approx(2184.0)
    tier_policy.reset_spend()


# --- tier-price persistence + explicit uncalibrated state ---------------------------------------------

@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    tier_policy.reset_costs()
    yield tmp_path
    tier_policy.reset_costs()


def test_uncalibrated_is_an_explicit_state_never_a_silent_assumption(out_root):
    cal = bud.calibration("tgt", ["screen", "certify"])
    assert cal.calibrated is False
    assert cal.unpriced == ("certify", "screen")
    assert "no price" in cal.note


def test_tier_prices_round_trip_through_disk_so_a_new_process_starts_calibrated(out_root):
    assert repo_root().is_dir()
    tier_policy.record_cost("tgt", "screen", 0.3)
    tier_policy.record_cost("tgt", "certify", 3.7)
    path = bud.save_tier_costs("tgt", ["screen", "certify"])
    assert json.loads(path.read_text())["median_seconds"] == {"certify": 3.7, "screen": 0.3}

    tier_policy.reset_costs()                       # a fresh process
    assert bud.calibration("tgt", ["screen", "certify"], load=False).calibrated is False

    primed = bud.load_tier_costs("tgt")
    assert set(primed) == {"screen", "certify"}
    cal = bud.calibration("tgt", ["screen", "certify"], load=False)
    assert cal.calibrated is True
    assert bud.tier_costs("tgt", ["screen", "certify"]) == {"certify": 3.7, "screen": 0.3}


def test_loading_a_missing_file_primes_nothing_rather_than_inventing_prices(out_root):
    assert bud.load_tier_costs("never_measured") == ()
    assert bud.tier_costs("never_measured", ["a"]) == {}


def test_an_unpriced_tier_is_absent_from_tier_costs_not_defaulted(out_root):
    tier_policy.record_cost("tgt", "screen", 0.3)
    assert bud.tier_costs("tgt", ["screen", "certify"]) == {"screen": 0.3}


def test_channels_from_tiers_leaves_an_untimed_tier_unpriced_and_blocks_the_unit(out_root):
    tier_policy.record_cost("tgt", "screen", 0.3)
    chans = bud.channels_from_tiers("tgt", ["screen", "certify"], items_per_datapoint={"screen": 26})
    assert {c.name: c.priced for c in chans} == {"screen": True, "certify": False}
    assert isinstance(bud.scarce_unit(chans), Unavailable)


def test_budget_from_channels_inherits_the_repo_wide_certify_budget_cap(monkeypatch):
    """One budget knob, not two: the env cap that already bounds certify-tier spend bounds this loop."""
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "5000")
    b = bud.budget_from_channels(_synthesis_dominated(), limit_items=10)
    assert isinstance(b, bud.Budget)
    assert b.limit_seconds == 5000.0
    b.charge(label="a")
    b.charge(label="b")
    b.charge(label="c")               # 3 x 2184 s = 6552 s
    assert b.exhausted and "s of" in b.exhausted_reason


def test_an_explicit_seconds_limit_wins_over_the_env_cap(monkeypatch):
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "5000")
    b = bud.budget_from_channels(_synthesis_dominated(), limit_items=10, limit_seconds=100.0)
    assert isinstance(b, bud.Budget) and b.limit_seconds == 100.0
