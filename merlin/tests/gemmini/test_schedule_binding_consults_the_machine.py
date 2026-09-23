"""The measured schedules depend on the derived machine, and break if it is wrong.

This is the test that stops the machine model being infrastructure nobody uses. The layer-bench slice
builds its schedules from geometry read out of the C headers the programs compile against;
`merlin.sched.mach` derives the same geometry from the target's extracted RTL. Nothing compared them, so
a header regenerated against a different elaboration would schedule against a machine that is not the
one being measured -- and every cycle count would still look plausible, because the program would run.

The binding now cross-checks. What makes that more than a comment is the sabotage below: a derivation
that returns the wrong edge must make the binding REFUSE. If that test passes with the cross-check
deleted, the derivation is decoration and the slice's numbers are not attributable to it.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest

pytestmark = pytest.mark.target("gemmini")


def _binding():
    """The target's schedule binding, reached the way production reaches it.

    Through the registered backend rather than by path: the package is an out-of-tree plugin whose
    modules use relative imports, so importing the file directly gets a different thing than what runs.
    """
    from merlin.runtime.backends import base

    try:
        backend = base.get_backend("gemmini")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the backend is not resolvable in this checkout: {exc}")
    package = sys.modules[backend.__name__].__package__
    gemmini = __import__(f"{package}.gemmini", fromlist=["x"])
    sched = __import__(f"{package}.gemmini_sched", fromlist=["x"])
    if not hasattr(gemmini, "rocc_tests_dir"):
        pytest.skip("this backend exposes no curated harness headers")
    header = str(gemmini.rocc_tests_dir() / "include" / "gemmini.h")
    params = str(gemmini._params_header())
    return sched, header, params


def _facts(sched, header, params):
    sched.schedule_facts.cache_clear()
    try:
        return sched.schedule_facts(header, params, target="gemmini")
    except Exception as exc:  # noqa: BLE001 - a missing curated header is not this test's subject
        pytest.skip(f"the headers are not readable here: {exc}")


def test_the_headers_and_the_derived_machine_agree():
    """Both routes to the same silicon, compared. The recorded digest is which machine agreed."""
    sched, header, params = _binding()
    facts = _facts(sched, header, params)
    verdict = facts["machine_crosscheck"]
    if verdict.startswith("unavailable:"):
        pytest.skip(f"no machine derives in this checkout, so there is nothing to compare: {verdict}")
    assert verdict.startswith("agrees with"), verdict
    assert facts["dim"] > 0 and facts["spad_rows"] > 0 and facts["acc_rows"] > 0


def test_a_wrong_derived_machine_makes_the_binding_refuse():
    """THE mutation.

    Without this, the cross-check could return a constant string and every other test here would pass.
    A derivation that says the array edge is 8 where the headers say 16 must stop the binding, because a
    schedule built on the wrong edge addresses the wrong rows, runs anyway, and reports a number.
    """
    sched, header, params = _binding()
    if _facts(sched, header, params)["machine_crosscheck"].startswith("unavailable:"):
        pytest.skip("no machine derives in this checkout; the cross-check cannot be exercised")

    import merlin.sched.mach.derive as derive_module

    real = derive_module.derive
    try:
        derive_module.derive = lambda target, **kw: dataclasses.replace(real(target, **kw), array_edge=8)
        sched.schedule_facts.cache_clear()
        with pytest.raises(sched.IsaError) as excinfo:
            sched.schedule_facts(header, params, target="gemmini")
    finally:
        derive_module.derive = real
        sched.schedule_facts.cache_clear()

    message = str(excinfo.value)
    assert "block declared=16" in message and "derived=8" in message, (
        f"the refusal does not name both numbers, so a reader cannot tell which is stale: {message}"
    )
    assert "refuses rather than picking" in message
    assert "DIM" in message, (
        "the refusal speaks the shared vocabulary but does not say how THESE headers spell it, so a "
        "reader cannot find the declaration that disagrees"
    )


def test_an_underivable_machine_is_recorded_rather_than_fatal():
    """A missing artifact is not a contradiction.

    Inside the agent sandbox the RTL facts are not mounted. Refusing there would make "we could not
    look" indistinguishable from "the two disagree", and would break a binding that is otherwise fine.
    It is recorded instead, so a reader can tell agreement from silence.
    """
    sched, header, params = _binding()

    import merlin.sched.mach.derive as derive_module

    real = derive_module.derive
    try:

        def _absent(target, **kw):
            raise FileNotFoundError("no facts artifact in this checkout")

        derive_module.derive = _absent
        sched.schedule_facts.cache_clear()
        facts = sched.schedule_facts(header, params, target="gemmini")
    finally:
        derive_module.derive = real
        sched.schedule_facts.cache_clear()

    assert facts["machine_crosscheck"].startswith("unavailable:")
    assert "FileNotFoundError" in facts["machine_crosscheck"], "the reason it could not look is not recorded"
