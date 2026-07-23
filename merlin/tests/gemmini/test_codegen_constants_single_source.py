"""Anti-triplication guard for the codegen EMITTER.

rocc_decode now sources its ABI constants from the capability manifest (the single source of truth). The
emitter gemmini_codegen_mlir still writes its own literals (it is the live .insn path — a runtime->targetgen
import-sourcing + spike bit-exact re-verification is a deliberate follow-on). This test pins the invariant
that the emitter's constants EQUAL the manifest source, so the two can never silently drift: any change to
the ISA encoding must go through the manifest, and this fails if the emitter is edited out of step.
"""
from __future__ import annotations

from merlin.runtime.backends import gemmini_codegen_mlir as gm
from merlin.targetgen.target_experiment import load_capability_manifest


def _enc():
    return load_capability_manifest("gemmini").encoding


def test_readout_bits_match_the_manifest():
    rb = _enc()["readout_bits"]
    assert gm.F1 == rb["f1"]
    assert gm.C_ACC == rb["c_acc"]
    assert gm.ACC_ACCUM == rb["acc_accum"]
    assert gm.ACC_I8 == rb["acc_i8"]


def test_funct_codes_match_the_semantic_class_map():
    # the emitter's K_* funct codes are the inverse of the manifest's RTL-code -> class map
    code_of = {cls: code for code, cls in _enc()["semantic_class"].items()}
    assert gm.K_CONFIG == code_of["CONFIG"]
    assert gm.K_MVIN == code_of["MVIN"]
    assert gm.K_MVOUT == code_of["MVOUT"]
    assert gm.K_COMPUTE_PRELOADED == code_of["COMPUTE_PRELOADED"]
    assert gm.K_PRELOAD == code_of["PRELOAD"]
    assert gm.K_FLUSH == code_of["FLUSH"]


def test_dim_matches_the_contract_mesh():
    m = load_capability_manifest("gemmini")
    dim = (m.contract["capabilities"]["mesh"])["rows"]
    assert gm.DIM == dim
