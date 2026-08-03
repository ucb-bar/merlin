"""Byte-parity: gemmini's manifest `encoding` block must equal the frozen hand-coded constants it is
replacing. This is the gate that lets the triplicated ABI constants (rocc_decode / gemmini_codegen_mlir
/ GemminiToLLVM.cpp) be retired in favor of the single manifest-sourced generated ISA module — no
guessed encoding: the manifest is proven to reproduce today's numbers exactly.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.target_experiment import load_capability_manifest, derived_readout_bits
from merlin.targetgen import rocc_decode as RD
from merlin.targetgen.rtl import circt_introspect as CI
from merlin.targetgen.rtl import mlc_bridge as MB


def test_readout_bits_match_the_decoder_constants():
    enc = load_capability_manifest("gemmini").encoding
    rb = enc["readout_bits"]
    assert rb["f1"] == RD.isa_constants("gemmini")["F1"]
    assert rb["c_acc"] == RD.isa_constants("gemmini")["C_ACC"]
    assert rb["acc_i8"] == RD.isa_constants("gemmini")["ACC_I8"]
    assert rb["acc_accum"] == RD.isa_constants("gemmini")["ACC_ACCUM"]
    assert rb["full_c_bit"] == RD.isa_constants("gemmini")["FULL_C_BIT"]


def test_semantic_class_map_matches_funct_class():
    enc = load_capability_manifest("gemmini").encoding
    # the manifest's RTL-code -> compiler-class map IS the decoder's _FUNCT_CLASS (the shared vocabulary)
    assert enc["semantic_class"] == RD.isa_constants("gemmini")["FUNCT_CLASS"]


def test_config_subtype_matches():
    enc = load_capability_manifest("gemmini").encoding
    assert enc["config_subtype"] == RD.isa_constants("gemmini")["CONFIG_SUBTYPE"]


def test_dim_and_opcode_stay_in_the_fact_bundle_not_the_encoding_block():
    # DIM/custom-opcode/funct3/legal-funct are RTL-derived (fact bundle), so they must NOT be duplicated
    # into the declarative encoding block — only the un-groundable ABI bits live there.
    enc = load_capability_manifest("gemmini").encoding
    assert "dim" not in enc and "custom_opcode" not in enc and "legal_funct" not in enc


# --- Step D: readout_bits are DERIVED (addr_len + flag-bit convention + 1.0f), not hand-declared hex ----
_FROZEN_HEX = {"f1": 0x3F800000, "c_acc": 0xA0000000, "acc_i8": 0x80000000,
               "acc_accum": 0x40000000, "full_c_bit": 0x20000000}


def test_readout_bits_are_not_declared_in_the_contract_yaml():
    # The magic hex block is retired from the contract: it is synthesized from addr_len at load time.
    contract = load_capability_manifest("gemmini").contract
    assert "readout_bits" not in (contract.get("encoding") or {})


def test_derived_readout_bits_equal_the_frozen_hex():
    # DERIVED from addr_len alone == the former hand-declared hex, byte-for-byte (discovery, not change).
    enc = load_capability_manifest("gemmini").encoding
    assert derived_readout_bits(int(enc["addr_len"])) == _FROZEN_HEX
    assert enc["readout_bits"] == _FROZEN_HEX  # what consumers actually read, via the loader synthesis


def test_readout_bit_roles_follow_the_addr_len_convention():
    # The 3 flag bits are the top 3 bits of the addr_len-wide field; c_acc composes two of them; f1 is 1.0f.
    rb = derived_readout_bits(32)
    assert rb["acc_i8"] == 1 << 31 and rb["acc_accum"] == 1 << 30 and rb["full_c_bit"] == 1 << 29
    assert rb["c_acc"] == rb["acc_i8"] | rb["full_c_bit"]
    import struct
    assert rb["f1"] == struct.unpack("<I", struct.pack("<f", 1.0))[0]


# --- Step A: semantic_class NAMES are HEADER-DERIVED (== gemmini.h `#define k_*`) -----------------------
def test_semantic_class_names_are_header_derived():
    res = CI.crosscheck_semantic_class_names("gemmini")
    if not (res["exact"] or res["alias"]):
        pytest.skip("gemmini ISA headers not staged — nothing to cross-check")
    # gate: every declared class is a real header funct, named by the header (modulo a reviewed alias)
    assert res["missing"] == [] and res["mismatch"] == []
    # the only non-exact name is the reviewed compiler alias LOOP_CONV for the header's LOOP_CONV_WS
    assert res["alias"] == [(15, "LOOP_CONV", "LOOP_CONV_WS")]


# --- Step C: config_subtype NAMES header-derived + CODES structurally decoded on rs1[1:0] ---------------
def test_config_subtype_names_are_header_derived():
    res = CI.crosscheck_config_subtype_names("gemmini")
    if not res["exact"]:
        pytest.skip("gemmini ISA headers not staged — nothing to cross-check")
    assert res["missing"] == [] and res["mismatch"] == []
    assert sorted(res["exact"]) == [(0, "CONFIG_EX"), (1, "CONFIG_LD"), (2, "CONFIG_ST")]


@pytest.mark.skipif(not MB.mlc_available()[0], reason="mlc unavailable (no HW dialect to decode)")
def test_config_subtype_codes_are_structurally_decoded():
    field = MB.discover_config_subtype_field("gemmini")
    assert field["values"] is not None, field["evidence"]
    enc = load_capability_manifest("gemmini").encoding
    # every hand config-subtype code is a real value the RTL decoder fans out on the rs1[1:0] sub-field
    assert set(int(c) for c in enc["config_subtype"]).issubset(set(field["values"]))


@pytest.mark.skipif(not MB.mlc_available()[0], reason="mlc unavailable")
def test_crosscheck_config_subtype_agrees():
    # combined names-from-headers + codes-from-structural-decode cross-check: no disagreements
    assert MB.crosscheck_config_subtype("gemmini") == []
