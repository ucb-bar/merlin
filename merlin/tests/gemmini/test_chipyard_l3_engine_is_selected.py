"""A chipyard target's L3 engine is SELECTED by policy, not hardcoded to one binary.

`rtl_engine_policy` exists because "a tier index is a FIDELITY, not a simulator": VCS, GSIM and Verilator
all run the elaborated design and all produce an `elaborated_rtl` verdict, so which one answers is a cost
and availability decision. Binding `L3 = verilator` made that decision invisible and unchangeable.

The program-oracle path already selected this way. The chipyard path did NOT -- it returned
`{"L2": spike, "L3": verilator}` outright -- so on a chipyard target the policy never ran, and a faster
engine could not be adopted without editing the function. That matters concretely: GSIM is measured at
~23x Verilator on the SIMT target (25 capsules, mean 115 s vs ~45 min/capsule), which is the difference
between a cert tier that runs per-capsule and one affordable once per run.

This does not make GSIM certify gemmini. It makes gemmini ASK, and record the answer.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import rtl_engine_policy as POL


def test_the_selection_is_made_in_priority_order_and_records_what_it_passed_over():
    sel = CR.chipyard_l3_selection("gemmini")
    assert sel["fidelity"] == POL.ELABORATED_RTL
    considered = [c["engine"] for c in sel["considered"]]
    # probes are tried cheapest-first and STOP at the first available one
    assert considered == [e for e in POL.ENGINE_PRIORITY][:len(considered)]
    assert sel["engine"] == considered[-1], "the selected engine is the one that probed available"


def test_every_engine_passed_over_carries_a_reason():
    """An engine skipped for no recorded reason is unauditable; the policy refuses that shape."""
    sel = CR.chipyard_l3_selection("gemmini")
    for c in sel["considered"]:
        assert str(c["reason"]).strip(), f"{c['engine']} was considered with no reason recorded"


def test_a_backend_that_does_not_know_an_engine_is_unavailable_not_a_crash():
    """Measured: `available('gsim')` raises GemminiError('unknown simulator'). A raising probe is an
    availability answer, not a broken run -- otherwise adding an engine to the priority list would break
    every target that lacks it."""
    sel = CR.chipyard_l3_selection("gemmini")
    by = {c["engine"]: c for c in sel["considered"]}
    if "gsim" in by and not by["gsim"]["available"]:
        assert "unknown simulator" in by["gsim"]["reason"] or "unavailable" in by["gsim"]["reason"]


def test_gsim_is_preferred_over_verilator_when_both_are_available():
    """The ordering claim itself, exercised without needing a real GSIM build."""
    probes = {
        "vcs": lambda: (False, "no license"),
        "gsim": lambda: (True, "gsim emulator present"),
        "verilator": lambda: (True, "verilator binary present"),
    }
    sel = POL.select("t", probes)
    assert sel["engine"] == "gsim", sel
    assert "verilator" not in sel["passed_over"], "the policy stops at the first available engine"


def test_no_engine_available_is_reported_not_downgraded():
    """A tier that cannot run must come back absent. Silently resolving L3 to the model tier below is how
    a functional result gets read as an RTL certification."""
    with pytest.raises(POL.NoEngineAvailable):
        POL.select("t", {"verilator": lambda: (False, "binary absent")})


def test_the_adapters_map_contains_only_tiers():
    """Every key is a tier name and every value a callable: `min(full)` picks the fastest tier and tier
    filters compare keys against a capsule's declared tiers, so a metadata key would read as a tier."""
    ad = CR._sim_engine_adapters("chipyard", "gemmini")
    assert set(ad) <= {"L0", "L1", "L2", "L3", "L4", "L5"}, sorted(ad)
    for tier, fn in ad.items():
        assert callable(fn), f"{tier} is not callable: {type(fn).__name__}"
