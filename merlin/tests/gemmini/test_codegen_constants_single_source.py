"""Anti-triplication guard for the codegen EMITTER.

rocc_decode now sources its ABI constants from the capability manifest (the single source of truth). The
emitter gemmini_codegen_mlir still writes its own literals (it is the live .insn path — a runtime->targetgen
import-sourcing + spike bit-exact re-verification is a deliberate follow-on). This test pins the invariant
that the emitter's constants EQUAL the manifest source, so the two can never silently drift: any change to
the ISA encoding must go through the manifest, and this fails if the emitter is edited out of step.
"""
from __future__ import annotations

from merlin.runtime.backends import base as _bk
from merlin.targetgen.target_experiment import load_capability_manifest

gm = _bk.get_backend("gemmini").gemmini_codegen_mlir


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


def test_dim_matches_the_extracted_mesh_fact():
    # DIM is a CIRCT-extracted FACT (arrays[mesh]), not a manifest field — the emitter and the decoder
    # both read it from the fact bundle, so it can never drift from the extracted mesh geometry.
    from merlin.targetgen.rtl.facts import load_facts
    mesh = next(a for a in load_facts("gemmini")["facts"]["arrays"] if a["name"] == "mesh")
    assert gm.DIM == mesh["rows"]


def test_scratchpad_rows_match_the_extracted_memory_fact():
    # The capacity-fit residency lever bounds resident operands by the scratchpad DEPTH — a
    # CIRCT-extracted memory fact (memories[scratchpad].depth), never a hardcoded capacity. Pin it so
    # the emitter's capacity budget can never drift from the extracted on-chip store geometry.
    from merlin.targetgen.rtl.facts import load_facts
    sp = next(m for m in load_facts("gemmini")["facts"]["memories"] if m["name"] == "scratchpad")
    assert gm.SCRATCHPAD_ROWS == sp["depth"]
