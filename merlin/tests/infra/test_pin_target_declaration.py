"""Which revisions a target's results depend on — and why the registry, not the contract, holds it.

A verdict claiming a hardware tier must record the revision it came from. Before pins named their
targets, three of the four targets declared none, so ``toolchain_shas`` returned merlin's own commit
and nothing else: every hardware-tier verdict they produced was attributed to no RTL revision, and it
looked exactly like a target with no hardware dependency.
"""
from __future__ import annotations

import shutil

import pytest

from merlin.common import provenance as PROV
from merlin.targetgen import provenance as TPROV


@pytest.fixture
def registry(tmp_path):
    dst = tmp_path / "hardware_pins.yaml"
    shutil.copy(PROV.pins_path(), dst)
    PROV._PINS_MEMO.clear()
    return dst


def test_every_pin_that_names_a_target_is_reachable_from_it():
    """The paired direction: a declaration nobody can read is not a declaration."""
    pins = PROV.load_pins()
    declared = {t for p in pins.values() for t in p.targets}
    assert declared, "no pin names any target; hardware attribution cannot be derived"
    for target in sorted(declared):
        names = TPROV.declared_pins(target)
        expected = {n for n, p in pins.items() if target in p.targets}
        assert expected <= set(names), f"{target}: registry names {expected - set(names)} unreachably"


def test_a_target_nobody_declared_gets_nothing():
    """UNKNOWN must stay distinguishable from "no hardware dependency" — this is the negative case."""
    assert TPROV.declared_pins("no_such_target_anywhere") == ()


def test_every_declared_target_resolves_a_real_revision():
    """A 40-hex sha is what `tier_cache._valid_pin` requires; anything else cannot key a certificate,
    so a pin that resolves to UNKNOWN leaves the target exactly as unattributed as before."""
    from merlin.targetgen import tier_cache as TC
    pins = PROV.load_pins()
    for target in sorted({t for p in pins.values() for t in p.targets}):
        shas = TPROV.toolchain_shas(target)
        usable = [k for k, v in shas.items() if k.lower() != "merlin" and TC._valid_pin(k, v)]
        assert usable, f"{target} declares pins but none resolve to a usable revision: {shas}"


def test_merlins_own_commit_is_never_the_hardware_attribution():
    """A source edit that emits identical code has not changed the device. If merlin's sha were the
    only entry, the identity would be about the wrong thing entirely."""
    shas = TPROV.toolchain_shas("gemmini")
    assert "merlin" in shas
    assert [k for k in shas if k.lower() != "merlin"], "no hardware entry beside merlin's own commit"


# --------------------------------------------------------------------------------------------
# parsing the declaration
# --------------------------------------------------------------------------------------------

def test_a_declaration_is_parsed_into_the_pin(registry):
    pins = PROV.load_pins(registry)
    assert any(p.targets for p in pins.values())
    for p in pins.values():
        assert isinstance(p.targets, tuple)
        assert all(isinstance(t, str) and t for t in p.targets)


def test_an_absent_declaration_is_not_an_error(registry):
    """"Not stated" is a legitimate state -- some pins are tooling, not a device revision."""
    text = registry.read_text().replace("    targets: [gemmini]\n", "", 1)
    registry.write_text(text)
    PROV._PINS_MEMO.clear()
    pins = PROV.load_pins(registry)                     # must not raise
    assert pins


@pytest.mark.parametrize("bad", ["targets: gemmini", "targets: []", "targets: [\"\"]", "targets: [3]"])
def test_a_malformed_declaration_raises_rather_than_being_dropped(registry, bad):
    """Dropped silently, it would read as "this target has no hardware dependency" -- the exact
    confusion this field exists to remove."""
    text = registry.read_text().replace("    targets: [gemmini]", f"    {bad}", 1)
    registry.write_text(text)
    PROV._PINS_MEMO.clear()
    with pytest.raises(PROV.PinsError):
        PROV.load_pins(registry)


def test_the_contract_and_the_registry_are_unioned_without_duplicates(monkeypatch):
    """A target contract is the natural place to declare this, but only one target's contract is a
    tracked file -- the rest are generated under out/. Both sources count; neither duplicates."""
    monkeypatch.setattr(TPROV, "PINS_CONTRACT_KEY", "hardware_pins")

    class _R:
        def load_contract(self):
            return {"hardware_pins": ["gemmini_rtl", "some_contract_only_pin"]}
    monkeypatch.setattr("merlin.targetgen.target_registry.resolve", lambda t: _R())
    got = TPROV.declared_pins("gemmini")
    assert got.count("gemmini_rtl") == 1, f"unioned with a duplicate: {got}"
    assert "some_contract_only_pin" in got, "the contract's own declaration was dropped"
    assert "gemmini_isa_headers" in got, "the registry's declaration was dropped"


def test_an_unreadable_registry_leaves_the_contract_working(monkeypatch):
    """The registry is additive. If it cannot be read, a contract declaration must still stand."""
    class _R:
        def load_contract(self):
            return {"hardware_pins": ["only_from_the_contract"]}
    monkeypatch.setattr("merlin.targetgen.target_registry.resolve", lambda t: _R())
    monkeypatch.setattr("merlin.common.provenance.load_pins",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no registry")))
    assert TPROV.declared_pins("gemmini") == ("only_from_the_contract",)
