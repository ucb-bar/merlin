"""P2: the mlc arc oracle as the DEFAULT cross-target grading tier + the sim_via-keyed registry.

The arc model (RTL-derived functional model) lets ANY mlc target be graded with no bespoke sim; a target
that declares a bespoke sim (chipyard) additionally gets spike/verilator. These are board-free structural
tests (the full cb-round-trip grade is exercised in the 2nd-target cross-target proof, P4)."""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.rtl import mlc_bridge as B


def test_arc_is_the_default_tier_for_a_command_buffer_target(monkeypatch):
    # arc is the default RTL tier for a command_buffer target with NO bespoke sim — proven on a synthetic
    # target (no descriptor sim_via, no external_backend). radiance is NOT this case: it is a self-hosted
    # SIMT target graded on its emitted kernel ELF by cyclotron, so the arc COMMAND-BUFFER adapter (wrong
    # artifact) must not be its default — see test_simt_cyclotron_target_routes_to_cyclotron_not_arc.
    monkeypatch.setattr(CR, "_endpoint_of", lambda t: ("inline_asm_insn", None))
    monkeypatch.setattr(CR, "_bespoke_sim_via", lambda t: "")
    ad = CR.oracle_adapters("synth_arc_only", sim_via=None)
    assert "L3" in ad and ad["L3"].__qualname__.startswith("mlc_arc_adapter")


def test_simt_cyclotron_target_routes_to_cyclotron_not_arc():
    # radiance is a self-hosted SIMT core (sim_via=cyclotron): its emitted kernel ELF is graded by the
    # bespoke cyclotron/VCS oracle, NOT the arc command-buffer adapter (which grades the wrong artifact).
    ad = CR.oracle_adapters("radiance", sim_via=None)
    assert set(ad) >= {"L2"} and all(v.__module__ == "merlin._oot_backends.muon.muon_oracles" for v in ad.values())
    assert not any(v.__qualname__.startswith("mlc_arc_adapter") for v in ad.values())


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


# --- AW3: the external_backend kernel is assembled by STOCK LLVM (.word/.insn), not a bespoke assembler -

def test_stock_llvm_assembles_word_insn_kernel(tmp_path):
    # the program oracle assembles the agent's emitted `.word`/`.insn` kernel with the prebuilt stock LLVM
    # (llvm-mc + llvm-objcopy). This is the target-agnostic assembly path — merlin holds no opcode table;
    # the encoding lives in the emitted directives. `#`/`//` comments + labels are accepted (no bytes).
    from merlin.targetgen.contract import toolchain as mlir_tc
    if not (mlir_tc.mlir_bin("llvm-mc").is_file() and mlir_tc.mlir_bin("llvm-objcopy").is_file()):
        pytest.skip("prebuilt stock LLVM (llvm-mc/llvm-objcopy) unavailable")
    from merlin.targetgen.program_oracle import _assemble_kernel_words
    ks = tmp_path / "kernel.S"
    ks.write_text(".text\nmain:  # label\n"
                  "  .word 0x00000013   // encoded insn\n"
                  "  .insn r 0x77, 0x0, 0x0a, x0, x1, x2\n"
                  "  .word 0xdeadbeef\n")
    words = _assemble_kernel_words(ks, tmp_path)
    # little-endian u32 stream; labels/comments emit no .text bytes
    assert words == [0x00000013, 0x14208077, 0xDEADBEEF]


def test_stock_llvm_rejects_empty_kernel(tmp_path):
    # an all-comment / empty kernel assembles to zero .text words -> fail closed (never a false green).
    from merlin.targetgen.contract import toolchain as mlir_tc
    if not mlir_tc.mlir_bin("llvm-mc").is_file():
        pytest.skip("prebuilt stock LLVM unavailable")
    from merlin.targetgen.program_oracle import _assemble_kernel_words, OracleUnavailable
    ks = tmp_path / "empty.S"
    ks.write_text("# only comments\n.text\n")
    with pytest.raises(OracleUnavailable):
        _assemble_kernel_words(ks, tmp_path)


# --- AW5: the program oracle preloads the capsule's CANONICAL operands (golden raws), not int 0..3 ----

def test_program_oracle_preloads_canonical_cb_operands():
    # the grader attaches each leaf's canonical bytes to the cb as `preload_b64`; the program oracle
    # turns those into (base, bytes) DRAM preload keyed by the cb-declared base. Output tensors + tensors
    # without preload_b64/base are ignored.
    from merlin.targetgen.program_oracle import _preload_from_cb
    import base64
    a, w = bytes([0x38, 0x40, 0x44, 0x42]), bytes([0x44, 0x44])
    cb = {"tensors": {
        "A0": {"role": "input", "base": 0x0, "preload_b64": base64.b64encode(a).decode()},
        "W":  {"role": "weight", "base": 0x400, "preload_b64": base64.b64encode(w).decode()},
        "Y0": {"role": "output", "base": 0x800},                       # output: no preload
        "S":  {"role": "input", "base": 0x900},                        # input w/o bytes: skipped
    }}
    pre = dict(_preload_from_cb(cb))
    assert pre == {0x0: a, 0x400: w}


def test_canonical_input_raws_reads_golden_fp8():
    # canonical_input_raws pulls the exact fp8 operand bytes the independent golden used from golden.yaml
    # (NOT Tensor.deterministic's int 0..3). Gated on the atlas corpus being present.
    from pathlib import Path
    from merlin.common.paths import repo_root
    from merlin.targetgen import capsule_golden as CG
    cdir = repo_root() / "merlin/contract/capsules/atlas/isa/AT3_k_accumulation"
    if not (cdir / "golden.yaml").is_file():
        pytest.skip("atlas corpus not present")
    raws = CG.canonical_input_raws({}, cdir)
    assert set(raws) >= {"A0", "W"} and len(raws["A0"]) == 32 * 64 and len(raws["W"]) == 64 * 32
    # exact-fp8 palette, not the degenerate 0..3 int fill (which would be mostly 0x00..0x03)
    assert any(b not in (0, 1, 2, 3) for b in raws["A0"])


def test_arc_adapter_available_for_gemmini_when_mlc_present():
    # gemmini has a prebuilt arc model; if mlc is present, arc_available is True (gate the assertion).
    if B.mlc_available()[0] and B.arc_available("gemmini"):
        assert CR.mlc_arc_adapter("gemmini") is not None    # constructs; the run needs a real cb (P4)


# --- the QA-gate loop/checkpoint split is manifest/descriptor-resolved, not hardwired ---------------

def _factory(adp) -> str:
    return adp.__qualname__.split(".")[0]


def _closed_over(adapter) -> set[str]:
    """The string values an adapter closure captures — its simulator AND its target.

    A set, not a list: closure cells are ordered by free-variable name, so indexing them
    made this test fail the moment the adapter took a second argument, for a reason that had
    nothing to do with what it was checking. Both values matter — an adapter bound to the
    wrong target would grade one accelerator against another's toolchain.
    """
    return {c.cell_contents for c in (adapter.__closure__ or ())
            if isinstance(c.cell_contents, str)}


def test_qa_loop_gate_is_fastest_tier_only_for_chipyard():
    # gemmini/chipyard: the per-round loop grades on spike (L2) ONLY; verilator (L3) is held back.
    loop = CR.qa_loop_adapters("gemmini", "chipyard")
    assert set(loop) == {"L2"}
    assert _factory(loop["L2"]) == "_spike_verilator_adapter"
    assert _closed_over(loop["L2"]) == {"spike", "gemmini"}


def test_qa_checkpoint_is_full_ladder_for_chipyard():
    """gemmini/chipyard: the cycle-accurate checkpoint = spike (L2) + the elaborated-RTL tier (L3).

    L3 is a FIDELITY, not a simulator. This used to assert the literal "verilator" and broke the day
    the engine policy started resolving a faster engine of the same fidelity -- for a reason that had
    nothing to do with what it was checking, which is the same mistake the closure-indexing note above
    records. Which engine answers is an availability choice, so the expectation is derived from the
    policy's own selection; what this test actually guards is that the ladder has both rungs, that L3
    is an elaborated-RTL engine rather than a quietly demoted one, and that both adapters are bound to
    the right target.
    """
    ckpt = CR.qa_checkpoint_adapters("gemmini", "chipyard")
    assert set(ckpt) == {"L2", "L3"}
    assert _closed_over(ckpt["L2"]) == {"spike", "gemmini"}
    selected = CR.describe_l3_engine("gemmini", "chipyard")
    assert selected["fidelity"] == "elaborated_rtl"
    assert _closed_over(ckpt["L3"]) == {selected["engine"], "gemmini"}


def test_qa_adapters_are_cyclotron_for_a_simt_target():
    # a SIMT target (sim_via=cyclotron) resolves the bespoke cyclotron/VCS oracle — NOT the arc
    # command-buffer path and NO gemmini spike/verilator. Loop = the fast cyclotron tier; checkpoint adds
    # the VCS tier. Both are the muon_oracles adapters.
    loop = CR.qa_loop_adapters("radiance", "cyclotron")
    ckpt = CR.qa_checkpoint_adapters("radiance", "cyclotron")
    assert loop and all(v.__module__ == "merlin._oot_backends.muon.muon_oracles" for v in loop.values())
    assert ckpt and all(v.__module__ == "merlin._oot_backends.muon.muon_oracles" for v in ckpt.values())
    assert not any(_factory(v) == "mlc_arc_adapter" for v in {**loop, **ckpt}.values())
