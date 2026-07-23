"""Byte-parity: gemmini's manifest `encoding` block must equal the frozen hand-coded constants it is
replacing. This is the gate that lets the triplicated ABI constants (rocc_decode / gemmini_codegen_mlir
/ GemminiToLLVM.cpp) be retired in favor of the single manifest-sourced generated ISA module — no
guessed encoding: the manifest is proven to reproduce today's numbers exactly.
"""
from __future__ import annotations

from merlin.targetgen.target_experiment import load_capability_manifest
from merlin.targetgen import rocc_decode as RD


def test_readout_bits_match_the_decoder_constants():
    enc = load_capability_manifest("gemmini").encoding
    rb = enc["readout_bits"]
    assert rb["f1"] == RD.F1
    assert rb["c_acc"] == RD.C_ACC
    assert rb["acc_i8"] == RD.ACC_I8
    assert rb["acc_accum"] == RD.ACC_ACCUM
    assert rb["full_c_bit"] == RD.FULL_C_BIT


def test_semantic_class_map_matches_funct_class():
    enc = load_capability_manifest("gemmini").encoding
    # the manifest's RTL-code -> compiler-class map IS the decoder's _FUNCT_CLASS (the shared vocabulary)
    assert enc["semantic_class"] == RD._FUNCT_CLASS


def test_config_subtype_matches():
    enc = load_capability_manifest("gemmini").encoding
    assert enc["config_subtype"] == RD._CONFIG_SUBTYPE


def test_dim_and_opcode_stay_in_the_fact_bundle_not_the_encoding_block():
    # DIM/custom-opcode/funct3/legal-funct are RTL-derived (fact bundle), so they must NOT be duplicated
    # into the declarative encoding block — only the un-groundable ABI bits live there.
    enc = load_capability_manifest("gemmini").encoding
    assert "dim" not in enc and "custom_opcode" not in enc and "legal_funct" not in enc
