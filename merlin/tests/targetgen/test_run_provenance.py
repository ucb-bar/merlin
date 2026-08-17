"""A run record must say which hardware revision its result belongs to — for every target.

The runner did record this, but through a helper that named one target's checkouts directly (a chipyard
root and a generator subdirectory). Every other target's run therefore recorded merlin's sha and nothing
else, and did so silently: the record looked complete. Which pins a result depends on is now the
target's own declaration (``hardware_pins`` in its contract), read against the reviewed registry.
"""
from __future__ import annotations

from merlin.targetgen import provenance as P


def test_a_target_that_declares_no_pins_still_records_merlin():
    """Not an error: a pure-simulation or host target has no external checkout to pin."""
    assert P.declared_pins(None) == ()
    assert set(P.toolchain_shas(None)) == {"merlin"}


def test_an_unresolvable_target_degrades_instead_of_raising():
    assert P.declared_pins("no_such_target_exists") == ()
    assert set(P.toolchain_shas("no_such_target_exists")) == {"merlin"}


def test_the_reference_target_declares_its_rtl_pin_and_resolves_it():
    """The regression for the migration: if the contract loses the declaration, the run record silently
    reverts to merlin-only, which is exactly the unattributable state this replaced."""
    assert "gemmini_rtl" in P.declared_pins("gemmini")
    shas = P.toolchain_shas("gemmini")
    assert "merlin" in shas and "gemmini_rtl" in shas
    assert len(shas["gemmini_rtl"]) == 40 or shas["gemmini_rtl"] == "UNKNOWN"


def test_an_unreadable_pin_is_recorded_as_unknown_never_omitted():
    """A missing key reads as 'this run had no such dependency'; what happened was 'nobody could tell'.

    Driven through a target whose contract declares a pin name the registry does not define, which is
    the same failure path as a checkout that is absent on this machine.
    """
    import merlin.targetgen.provenance as mod

    real = mod.declared_pins
    try:
        mod.declared_pins = lambda _t: ("not_a_registered_pin",)
        shas = mod.toolchain_shas("anything")
    finally:
        mod.declared_pins = real
    assert shas["not_a_registered_pin"] == "UNKNOWN"
