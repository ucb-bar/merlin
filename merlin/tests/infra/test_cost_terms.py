"""An edit that trades one resource for another must be legible as a trade, not a regression.

Grounded in measured optimizations of one whole model (2026-09-09). An authoring loop scored on four
command-buffer quantities while the host lane was 95.8% of measured cycles; six of seven hand-found
optimizations were invisible or scored as regressions, and every iteration reported an exact zero
delta. Adding host counters fixed the invisible ones. The two below are the harder half -- they are
real wins that SPEND one resource to buy another, and no rule that treats a raised term as bad can
judge them:

* device epilogue: host operations -146,185,268, accelerator dispatches 23,875 -> 276,189 (11.6x);
* implicit GEMM: host operations -553,346,427 (-32.75%), DRAM +24%, mesh issue cycles +21.6%.

The load-bearing property is `test_a_term_that_moved_without_a_rate_withholds_the_composite`:
weighting an unpriced term as zero would hide a resource the edit actually spent, which is precisely
how a scoreboard stops being able to say "no".
"""

from __future__ import annotations

import pytest

from merlin.perf.cost_terms import DERIVED, MEASURED, UNPRICED, Rate, compare_terms, unpriced_terms

HOST = "host_dynamic_operations_total"
MESH = "mesh_issue_cycles"
DISPATCH = "dispatches"

# The one rate this project actually measured: resnet50 q535's host cycle residue over its host ops.
HOST_RATE = Rate(0.556, MEASURED, "resnet50 firesim q535: host residue / host dynamic operations")


def test_per_term_deltas_are_always_reported() -> None:
    out = compare_terms({HOST: 100, MESH: 10}, {HOST: 60, MESH: 14}, {HOST: HOST_RATE})
    by = {row["term"]: row for row in out["terms"]}
    assert by[HOST]["delta"] == -40 and by[MESH]["delta"] == 4
    assert by[HOST]["cycle_delta"] == pytest.approx(-40 * 0.556)
    assert by[MESH]["cycle_delta"] is None


def test_a_term_that_moved_without_a_rate_withholds_the_composite() -> None:
    """THE property: an unpriced moved term must not be silently weighted zero."""
    out = compare_terms({HOST: 100, MESH: 10}, {HOST: 60, MESH: 14}, {HOST: HOST_RATE})
    assert out["composite_status"] == UNPRICED
    assert out["composite_cycle_delta"] is None
    assert out["unpriced_moved_terms"] == [MESH]


def test_an_unpriced_term_that_did_not_move_does_not_withhold_the_composite() -> None:
    out = compare_terms({HOST: 100, MESH: 10}, {HOST: 60, MESH: 10}, {HOST: HOST_RATE})
    assert out["composite_status"] == "priced"
    assert out["composite_cycle_delta"] == pytest.approx(-40 * 0.556)
    assert out["unpriced_moved_terms"] == []


def test_the_device_epilogue_trade_is_reported_as_a_trade() -> None:
    """Measured: host -146,185,268 while dispatches rise 23,875 -> 276,189."""
    before = {HOST: 1_686_424_109, DISPATCH: 23_875}
    after = {HOST: 1_540_238_841, DISPATCH: 276_189}
    out = compare_terms(before, after, {HOST: HOST_RATE})
    assert out["terms_increased"] == [DISPATCH]
    assert out["composite_status"] == UNPRICED  # no measured cycles-per-dispatch exists
    assert out["unpriced_moved_terms"] == [DISPATCH]
    assert "it is a bet" in out["reading"]


def test_the_implicit_gemm_trade_names_both_resources_it_spent() -> None:
    """Measured: host -553,346,427 while DRAM and mesh issue cycles both rise."""
    before = {HOST: 1_689_656_515, MESH: 21_383_000, "dram_bytes": 134_970_000}
    after = {HOST: 1_136_310_088, MESH: 25_999_000, "dram_bytes": 167_310_000}
    out = compare_terms(before, after, {HOST: HOST_RATE})
    assert out["terms_increased"] == ["dram_bytes", MESH]
    assert set(out["unpriced_moved_terms"]) == {"dram_bytes", MESH}
    assert out["composite_cycle_delta"] is None


def test_a_pure_win_on_priced_terms_gets_a_composite() -> None:
    """Levers A/E/F move only the host lane, so they are fully priceable."""
    out = compare_terms({HOST: 2_384_685_111}, {HOST: 728_401_808}, {HOST: HOST_RATE})
    assert out["composite_status"] == "priced"
    assert out["composite_cycle_delta"] == pytest.approx((728_401_808 - 2_384_685_111) * 0.556)
    assert out["terms_increased"] == []


def test_a_rate_without_provenance_is_refused() -> None:
    """A number with no evidence behind it silently decides what the loop pursues."""
    with pytest.raises(ValueError, match="no provenance"):
        compare_terms({HOST: 2}, {HOST: 1}, {HOST: Rate(0.5, MEASURED, "   ")})


def test_an_explicitly_unpriced_rate_still_withholds_the_composite() -> None:
    """Declaring a term unpriced is honest; it must not become a zero weight."""
    rates = {HOST: HOST_RATE, MESH: Rate(0.0, UNPRICED, "no measurement exists")}
    out = compare_terms({HOST: 100, MESH: 10}, {HOST: 60, MESH: 14}, rates)
    assert out["composite_status"] == UNPRICED and out["unpriced_moved_terms"] == [MESH]


def test_derived_rates_are_usable_but_labelled() -> None:
    rates = {HOST: HOST_RATE, MESH: Rate(1.0, DERIVED, "one issue slot per mesh cycle, from RTL")}
    out = compare_terms({HOST: 100, MESH: 10}, {HOST: 60, MESH: 14}, rates)
    assert out["composite_status"] == "priced"
    by = {row["term"]: row for row in out["terms"]}
    assert by[MESH]["rate_status"] == DERIVED
    assert out["composite_cycle_delta"] == pytest.approx(-40 * 0.556 + 4 * 1.0)


def test_unpriced_terms_helper_ignores_non_numeric_and_unchanged_fields() -> None:
    before = {HOST: 5, "note": "text", MESH: 3}
    after = {HOST: 4, "note": "other", MESH: 3}
    assert unpriced_terms(before, after, {HOST: HOST_RATE}) == []
