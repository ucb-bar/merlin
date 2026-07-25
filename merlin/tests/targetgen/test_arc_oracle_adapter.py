"""P2: the mlc arc oracle as the DEFAULT cross-target grading tier + the sim_via-keyed registry.

The arc model (RTL-derived functional model) lets ANY mlc target be graded with no bespoke sim; a target
that declares a bespoke sim (chipyard) additionally gets spike/verilator. These are board-free structural
tests (the full cb-round-trip grade is exercised in the 2nd-target cross-target proof, P4)."""
from __future__ import annotations

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.rtl import mlc_bridge as B


def test_arc_is_the_default_tier_for_any_target():
    # arc is the default RTL tier for a target with NO bespoke sim AND a command_buffer/.insn endpoint
    # (radiance = simt/.insn). A self-hosted-ISA (external_backend) target instead routes to the program
    # oracle — see test_external_backend_target_uses_the_program_oracle.
    ad = CR.oracle_adapters("radiance", sim_via=None)
    assert "L3" in ad                                       # arc supplies the RTL tier
    assert ad["L3"].__qualname__.startswith("mlc_arc_adapter")


def test_external_backend_target_uses_the_program_oracle():
    # atlas is a self-hosted ISA core (endpoint_kind=external_backend): its emitted kernel is assembled +
    # run on the target's cosim by the generic program oracle, NOT the command_buffer arc path.
    ad = CR.oracle_adapters("atlas", sim_via=None)
    assert "L3" in ad
    assert ad["L3"].__module__ == "merlin.targetgen.program_oracle"


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


# --- the QA-gate loop/checkpoint split is manifest/descriptor-resolved, not hardwired ---------------

def _factory(adp) -> str:
    return adp.__qualname__.split(".")[0]


def test_qa_loop_gate_is_fastest_tier_only_for_chipyard():
    # gemmini/chipyard: the per-round loop grades on spike (L2) ONLY; verilator (L3) is held back.
    loop = CR.qa_loop_adapters("gemmini", "chipyard")
    assert set(loop) == {"L2"}
    assert _factory(loop["L2"]) == "_spike_verilator_adapter"
    assert loop["L2"].__closure__[0].cell_contents == "spike"


def test_qa_checkpoint_is_full_ladder_for_chipyard():
    # gemmini/chipyard: the cycle-accurate checkpoint = spike (L2) + verilator (L3), same as before.
    ckpt = CR.qa_checkpoint_adapters("gemmini", "chipyard")
    assert set(ckpt) == {"L2", "L3"}
    assert [c.cell_contents for c in ckpt["L2"].__closure__] == ["spike"]
    assert [c.cell_contents for c in ckpt["L3"].__closure__] == ["verilator"]


def test_qa_adapters_are_arc_for_a_non_chipyard_target():
    # a target with a different sim_via resolves ITS adapters (the RTL-derived arc tier) with no gemmini
    # path — the loop and checkpoint both use the arc oracle at L3, and NO spike/verilator appears.
    loop = CR.qa_loop_adapters("radiance", "cyclotron")
    ckpt = CR.qa_checkpoint_adapters("radiance", "cyclotron")
    assert set(loop) == {"L3"} and _factory(loop["L3"]) == "mlc_arc_adapter"
    assert set(ckpt) == {"L3"} and _factory(ckpt["L3"]) == "mlc_arc_adapter"
    assert loop["L3"].__closure__[0].cell_contents == "radiance"
