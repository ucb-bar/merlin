"""Three perf instruments, each called from the path it was built for.

``check_wiring``'s premise: an instrument nothing calls reads exactly like a passing one. These
modules each had tests and no production caller, so the defects they exist to surface could not
reach anybody. The tests below pin the CALL, not just the module -- each one asserts the behaviour
the caller gained, with a mutation that must make it fail.

* :mod:`merlin.perf.cycle_bound` owns what a capsule may put in its cycle-ceiling slot;
  :func:`merlin.perf.cost_plane.declared_ceiling` is the reader its own docstring names, and it used
  to walk the field itself and read ANY non-integer as "no ceiling".
* :mod:`merlin.perf.exchange_rates` is the measured ledger; ``agent_guidance``'s resource scorer is
  the loop that could name a trade and not score one for want of it.
* :mod:`merlin.perf.exchange_rate_windows` proves an arm isolates the term it measures;
  ``build_tools/scripts/check_exchange_rate_windows.py`` is what runs that proof before a campaign
  is spent measuring two things at once.
"""

from __future__ import annotations

import copy
import subprocess
import sys

import yaml

from merlin.common.paths import repo_root
from merlin.perf import agent_guidance as AG
from merlin.perf import cost_plane as CP
from merlin.perf import cycle_bound as CB
from merlin.perf import exchange_rate_windows as XW
from merlin.perf.cost_terms import MEASURED, UNPRICED

GATE = repo_root() / "build_tools" / "scripts" / "check_exchange_rate_windows.py"


# --------------------------------------------------------------------------------------------
# cycle_bound -> cost_plane.declared_ceiling
# --------------------------------------------------------------------------------------------


def _capsule(value):
    return {"name": "m", "performance": {"cost": {"projected_cycles": value}}}


def test_the_cost_plane_reads_the_cycle_slot_through_the_vocabulary_that_owns_it():
    """A word all three readers admit must read the same way; one none of them admits must refuse."""
    for word, reason in CB.NO_CYCLE_BOUND.items():
        ceiling, basis = CP.declared_ceiling(_capsule(word))
        assert ceiling is None and reason in basis, word

    # MUTATION: a word the vocabulary does not know is a refusal that says so, not a silent "no
    # ceiling" -- which is what made a typo indistinguishable from a declaration.
    ceiling, basis = CP.declared_ceiling(_capsule("derived_at_preflght"))
    assert ceiling is None and "neither a cycle count nor a declared reason" in basis

    # MUTATION: zero is not a ceiling any run can meet, and it is the one malformed value that
    # reaches a reader looking exactly like a bound.
    ceiling, basis = CP.declared_ceiling(_capsule(0))
    assert ceiling is None and "not a bound any run can meet" in basis

    # ...and an integer is still the bound it is, so the delegation did not swallow the real case.
    assert CP.declared_ceiling(_capsule(4096))[0] == 4096
    assert CP.declared_ceiling({"name": "m"}) == (None, "the capsule declares no performance.cost.projected_cycles")


# --------------------------------------------------------------------------------------------
# exchange_rates -> the optimisation loop's resource scorer
# --------------------------------------------------------------------------------------------

_BEFORE = {"dispatches": 100.0}
_AFTER = {"dispatches": 140.0}


TERM = "host_dynamic_operations_total"


def _synthetic_ledger(**entry):
    base = {
        "cycles_per_unit": 0.5,
        "status": MEASURED,
        "provenance": "a named run on that device",
        "closure_residual_fraction": 0.01,
    }
    base.update(entry)
    return {
        "policy": {"max_closure_residual_fraction": 0.05, "unmeasured_closure_is_unpriced": True},
        "rates": {"a_device": {TERM: base}},
    }


def test_the_resource_scorer_prices_from_the_measured_ledger_for_the_named_device(monkeypatch):
    """The loop could NAME a trade and not score one; the ledger is the half that was missing.

    The ledger CONTENTS are substituted at the loader, not the lookup, so the call path under test is
    the real one: ``_resource_cost`` -> ``exchange_rates.rates_for_design`` -> the closure policy.
    """
    from merlin.perf import exchange_rates

    monkeypatch.setattr(exchange_rates, "load_ledger", _synthetic_ledger)
    before = {TERM: 1000.0, "dispatches": 100.0}
    after = {TERM: 800.0, "dispatches": 140.0}

    priced = AG._resource_cost(before, after, None, "a_device")
    assert priced["rates_from_ledger"] == [TERM] and priced["rates_design"] == "a_device"
    row = next(r for r in priced["terms"] if r["term"] == TERM)
    assert row["rate_status"] == MEASURED and row["cycle_delta"] == -100.0

    # MUTATION 1: a device the ledger does not name yields nothing. A rate from another device is
    # not a weaker answer, it is a wrong one, so nothing is borrowed.
    elsewhere = AG._resource_cost(before, after, None, "another_device")
    assert elsewhere["rates_from_ledger"] == []
    assert next(r for r in elsewhere["terms"] if r["term"] == TERM)["rate_status"] == UNPRICED

    # MUTATION 2: with no device resolved the behaviour is exactly what it was -- unpriced, and the
    # per-term table still reported. Naming no device must not start pricing from somewhere.
    anonymous = AG._resource_cost(before, after, None)
    assert anonymous["rates_from_ledger"] == [] and anonymous["rates_design"] is None
    assert anonymous["terms"] == elsewhere["terms"]

    # MUTATION 3: a rate the ANALYSIS declares is about this measurement and wins over the device's
    # general one, rather than being silently replaced by it.
    declared = {TERM: {"cycles_per_unit": 99.0, "status": MEASURED, "provenance": "this run"}}
    won = AG._resource_cost(before, after, declared, "a_device")
    assert won["rates_from_ledger"] == []
    assert next(r for r in won["terms"] if r["term"] == TERM)["rate_provenance"] == "this run"


def test_a_ledger_entry_that_does_not_close_reaches_the_scorer_as_a_named_reason(monkeypatch):
    """A withheld composite must say what is missing, not merely that something is."""
    from merlin.perf import exchange_rates

    monkeypatch.setattr(exchange_rates, "load_ledger", lambda: _synthetic_ledger(closure_residual_fraction=0.2))
    out = AG._resource_cost({TERM: 1000.0}, {TERM: 800.0}, None, "a_device")
    assert out["rates_from_ledger"] == []
    assert "exceeds the declared bound" in out["rates_unpriced"][TERM]
    assert next(r for r in out["terms"] if r["term"] == TERM)["rate_status"] == UNPRICED

    # MUTATION: the same entry with a closure inside the bound prices, so the refusal above is about
    # the closure and not about the plumbing.
    monkeypatch.setattr(exchange_rates, "load_ledger", _synthetic_ledger)
    closing = AG._resource_cost({TERM: 1000.0}, {TERM: 800.0}, None, "a_device")
    assert closing["rates_from_ledger"] == [TERM] and closing["rates_unpriced"] == {}


def test_the_shipped_ledger_reaches_the_scorer_with_its_real_verdict():
    """No monkeypatch: whatever the ledger says today must arrive, priced or refused with a reason."""
    from merlin.perf import exchange_rates

    designs = sorted(exchange_rates.load_ledger()["rates"])
    assert designs, "the shipped ledger names no device, so this wiring cannot be exercised"
    for design in designs:
        out = AG._resource_cost(_BEFORE, _AFTER, None, design)
        declared = set(exchange_rates.load_ledger()["rates"][design])
        # Every declared term is accounted for: usable, or unpriced with a reason. Neither list may
        # silently drop one, because a dropped entry is indistinguishable from an absent measurement.
        assert declared == set(exchange_rates.rates_for_design(design)) | set(out["rates_unpriced"])
        for term, reason in out["rates_unpriced"].items():
            assert reason.strip(), term


# --------------------------------------------------------------------------------------------
# exchange_rate_windows -> the gate that runs the isolation proof
# --------------------------------------------------------------------------------------------


def _gate():
    return subprocess.run(
        [sys.executable, str(GATE)],
        capture_output=True,
        text=True,
        cwd=str(repo_root()),
        timeout=180,
    )


def test_the_window_isolation_proof_is_run_by_a_gate_and_the_gate_can_fail(tmp_path, monkeypatch):
    """A check nothing runs is a check that cannot fail; this is the thing that runs it."""
    done = _gate()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "isolate what they claim" in done.stdout

    # MUTATION: flatten a fit arm's windows so the term it CLAIMS to vary is constant across them.
    # That is the exact failure the module exists to catch -- an arm that measures nothing while
    # reading as a measurement -- and the proof must refuse it. Done in memory against the checker
    # the gate calls, so the shipped plan is never edited.
    plan = yaml.safe_load((repo_root() / "merlin" / "contract" / "phase2_exchange_rate_windows.yaml").read_text())
    assert XW.check_plan(body=plan)["status"] == "consistent"

    broken = copy.deepcopy(plan)
    arm = next(a for a in broken["arms"] if str(a.get("role", "")) != "closure" and len(a["windows"]) > 1)
    template = {k: v for k, v in arm["windows"][0].items() if k != "id"}
    for window in arm["windows"]:
        window.update(copy.deepcopy(template))
    try:
        XW.check_plan(body=broken)
    except XW.WindowPlanError as refusal:
        assert "CONSTANT" in str(refusal), refusal
        return
    raise AssertionError("an arm whose varied term never moves must be refused, not reported")
