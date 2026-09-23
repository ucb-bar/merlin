"""Exchange rates: per device, and refused unless they close.

WHY THIS EXISTS. `cost_terms` withholds a composite when a moved term has no measured rate, which is
why the loop here can name a trade and not score one. The missing half was never a model — it was a
measurement nobody wrote down. The project's single measured rate lived in
`merlin/tests/infra/test_cost_terms.py`, as a fixture: readable by one test, citable by nothing, and
carrying no record of which machine produced it.

Two refusals are what this module adds, and both are tested by mutation because both fail silently
if they are wrong: a rate from another device, and a rate that does not close.
"""

from __future__ import annotations

import copy

import pytest

from merlin.perf import exchange_rates as X
from merlin.perf.cost_terms import MEASURED

DEVICE = "some_device"
OTHER = "another_device"
TERM = "host_dynamic_operations_total"


def _ledger(**entry):
    base = {
        "cycles_per_unit": 0.5,
        "status": MEASURED,
        "provenance": "a named run",
        "closure_residual_fraction": 0.01,
    }
    base.update(entry)
    return {
        "policy": {"max_closure_residual_fraction": 0.05, "unmeasured_closure_is_unpriced": True},
        "rates": {DEVICE: {TERM: base}},
    }


class TestARateIsUsableOnlyWhenItCloses:
    def test_a_measured_rate_within_the_bound_is_returned(self):
        got = X.rates_for_design(DEVICE, ledger=_ledger())
        assert got[TERM].cycles_per_unit == 0.5 and got[TERM].status == MEASURED

    def test_a_residual_over_the_declared_bound_is_withheld(self):
        """THE MUTATION. A sweep can always fit a slope through two points; only closure says the
        slope means anything, and a rate that does not close would quietly weight every composite."""
        ledger = _ledger(closure_residual_fraction=0.20)
        assert X.rates_for_design(DEVICE, ledger=ledger) == {}
        assert "exceeds the declared bound" in X.unpriced_reasons(DEVICE, ledger=ledger)[TERM]

    def test_a_residual_exactly_at_the_bound_is_allowed(self):
        assert X.rates_for_design(DEVICE, ledger=_ledger(closure_residual_fraction=0.05))

    def test_a_negative_residual_is_judged_by_magnitude(self):
        """Under-predicting by 20% is not better than over-predicting by 20%."""
        assert X.rates_for_design(DEVICE, ledger=_ledger(closure_residual_fraction=-0.20)) == {}

    def test_an_unmeasured_closure_is_not_assumed_to_close(self):
        ledger = _ledger(closure_residual_fraction=None)
        assert X.rates_for_design(DEVICE, ledger=ledger) == {}
        assert X.unpriced_reasons(DEVICE, ledger=ledger)[TERM] == "closure was never measured"

    def test_the_policy_can_declare_that_an_unmeasured_closure_is_tolerated(self):
        """The bound is a review decision, so the ledger gets to make it — but explicitly."""
        ledger = _ledger(closure_residual_fraction=None)
        ledger["policy"]["unmeasured_closure_is_unpriced"] = False
        assert X.rates_for_design(DEVICE, ledger=ledger)


class TestWhatIsNotARate:
    def test_a_rate_without_provenance_is_withheld(self):
        ledger = _ledger(provenance="   ")
        assert X.rates_for_design(DEVICE, ledger=ledger) == {}
        assert "provenance" in X.unpriced_reasons(DEVICE, ledger=ledger)[TERM]

    def test_a_derived_rate_is_not_a_measured_one(self):
        assert X.rates_for_design(DEVICE, ledger=_ledger(status="derived")) == {}

    def test_a_non_numeric_rate_is_withheld(self):
        assert X.rates_for_design(DEVICE, ledger=_ledger(cycles_per_unit="fast")) == {}

    def test_a_boolean_is_not_a_number(self):
        assert X.rates_for_design(DEVICE, ledger=_ledger(cycles_per_unit=True)) == {}


class TestRatesDoNotCrossDevices:
    def test_another_devices_rate_is_not_returned(self):
        """A rate from a different machine is not a weaker answer, it is a wrong one."""
        assert X.rates_for_design(OTHER, ledger=_ledger()) == {}

    def test_an_unknown_device_yields_no_rates_rather_than_raising(self):
        """Every moved term then goes unpriced and the composite is withheld -- which is the truth
        when no rate exists, and is what the caller would do with a raise anyway."""
        assert X.rates_for_design("never_registered", ledger=_ledger()) == {}


class TestThePolicyIsDeclaredNotDefaulted:
    def test_a_ledger_with_no_bound_raises(self):
        ledger = _ledger()
        del ledger["policy"]["max_closure_residual_fraction"]
        with pytest.raises(X.ExchangeRatesError) as excinfo:
            X.rates_for_design(DEVICE, ledger=ledger)
        assert "no default" in str(excinfo.value).lower() or "nobody chose" in str(excinfo.value)

    def test_a_ledger_with_no_policy_block_raises(self):
        ledger = _ledger()
        del ledger["policy"]
        with pytest.raises((X.ExchangeRatesError, KeyError)):
            X.rates_for_design(DEVICE, ledger=ledger)


class TestTheShippedLedger:
    def test_it_loads_and_declares_a_bound(self):
        assert isinstance(X.policy()["max_closure_residual_fraction"], (int, float))

    def test_every_device_it_names_is_a_registered_bitstream(self):
        """Keyed on the pin registry's artifact NAME, which is unique per device -- never on the
        config string, because two bitstreams here share one. A typo would otherwise create a device
        that silently has no rates."""
        from merlin.common.provenance import load_artifacts

        registered = {n for n, a in load_artifacts().items() if a.role == "firesim_bitstream"}
        named = set(X.load_ledger()["rates"])
        assert named <= registered, f"ledger names devices the registry does not: {sorted(named - registered)}"

    def test_the_one_measured_rate_is_carried_but_withheld(self):
        """The honest current state, pinned so a later change is deliberate: the project's only
        measured rate has no closure measurement, so it cannot weigh a composite -- and it is kept
        rather than deleted, because it is real evidence about that device."""
        device = "firesim_gemmini_rocket_u250"
        assert X.rates_for_design(device) == {}
        assert "closure" in X.unpriced_reasons(device).get("host_dynamic_operations_total", "")

    def test_the_device_this_weeks_numbers_ran_on_has_no_rates_at_all(self):
        """Also pinned deliberately. Phase 2 cannot score a trade on this device today, and a test
        that says so is better than discovering it inside a campaign."""
        assert X.rates_for_design("firesim_gemmini_rocket_u250_30mhz") == {}

    def test_the_ledger_is_not_silently_empty(self):
        """`rates: {}` would make every device unpriced and every test above vacuous."""
        assert X.load_ledger()["rates"], "an empty ledger is not a clean bill of health"


def test_the_measured_rate_is_no_longer_only_in_a_test_file():
    """The point of the move. A number only a test knows cannot be cited, checked, or attributed."""
    ledger = X.load_ledger()
    entry = ledger["rates"]["firesim_gemmini_rocket_u250"]["host_dynamic_operations_total"]
    assert entry["cycles_per_unit"] == 0.556
    assert "535" in entry["provenance"], "the rate must name the run that produced it"
    assert copy.deepcopy(entry) == entry
