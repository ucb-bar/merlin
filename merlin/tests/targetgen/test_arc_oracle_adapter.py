"""P2: the mlc arc oracle as the DEFAULT cross-target grading tier + the sim_via-keyed registry.

The arc model (RTL-derived functional model) lets ANY mlc target be graded with no bespoke sim; a target
that declares a bespoke sim (chipyard) additionally gets spike/verilator. These are board-free structural
tests (the full cb-round-trip grade is exercised in the 2nd-target cross-target proof, P4)."""
from __future__ import annotations

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.rtl import mlc_bridge as B


def test_arc_is_the_default_tier_for_any_target():
    ad = CR.oracle_adapters("atlas", sim_via=None)          # a target with NO bespoke sim
    assert "L3" in ad                                       # arc supplies the RTL tier
    assert ad["L3"].__qualname__.startswith("mlc_arc_adapter")


def test_bespoke_sim_overrides_when_declared():
    ad = CR.oracle_adapters("gemmini", sim_via="chipyard")  # gemmini keeps its spike/verilator sims
    assert "L2" in ad and "L3" in ad
    # the chipyard sim adapters are the _spike_verilator_adapter closures, not the arc adapter
    assert ad["L3"].__qualname__.startswith("_spike_verilator_adapter")


def test_arc_adapter_fails_closed_for_unknown_target():
    run = CR.mlc_arc_adapter("definitely_not_a_target")
    try:
        run(cb={}, llvm_text="", workdir="/tmp", timeout=5)
        assert False, "arc adapter should raise OracleUnavailable for an unknown/absent target"
    except CR.OracleUnavailable as e:
        assert "arc model unavailable" in str(e)


def test_arc_adapter_available_for_gemmini_when_mlc_present():
    # gemmini has a prebuilt arc model; if mlc is present, arc_available is True (gate the assertion).
    if B.mlc_available()[0] and B.arc_available("gemmini"):
        assert CR.mlc_arc_adapter("gemmini") is not None    # constructs; the run needs a real cb (P4)
