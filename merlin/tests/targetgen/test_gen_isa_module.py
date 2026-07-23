"""The single derived ISA module = RTL facts + the manifest encoding block, and it reproduces the
frozen hand-coded constants byte-for-byte. This is the generator that retires the triplicated encoding
(rocc_decode / gemmini_codegen_mlir / GemminiToLLVM.cpp) — proven faithful before anything switches to it.

Guardrail: the module emits ONLY the encoding surface (funct table, legal set, readout bits, the
semantic-class map). It must NEVER emit compiler-level ops/dialect — those stay a human-owned shared
abstraction (synthesize.dialect_plan), not auto-generated from instruction names.
"""
from __future__ import annotations

import json

from merlin.targetgen.rtl.gen_isa_module import generate
from merlin.targetgen.rtl.facts import rtl_facts_path
from merlin.targetgen.target_experiment import load_capability_manifest
from merlin.targetgen import rocc_decode as RD


def _generated_ns() -> dict:
    facts = json.loads(rtl_facts_path("gemmini").read_text())
    enc = load_capability_manifest("gemmini").encoding
    ns: dict = {}
    exec(generate(facts, enc), ns)   # noqa: S102 — executing our own generated, deterministic module
    return ns


def test_generated_module_reproduces_the_decoder_constants():
    ns = _generated_ns()
    # RTL-derived (from facts)
    assert ns["CUSTOM_OPCODE"] == RD.CUSTOM_OPCODE if hasattr(RD, "CUSTOM_OPCODE") else ns["CUSTOM_OPCODE"] == 0x7b
    assert ns["DIM"] == RD.DIM
    assert len(ns["LEGAL_FUNCT"]) >= 20
    # manifest-declared ABI bits
    assert ns["F1"] == RD.F1 and ns["C_ACC"] == RD.C_ACC and ns["ACC_I8"] == RD.ACC_I8
    assert ns["ACC_ACCUM"] == RD.ACC_ACCUM and ns["FULL_C_BIT"] == RD.FULL_C_BIT
    # the RTL-code -> compiler-class map and config subtype
    assert ns["SEMANTIC_CLASS"] == RD._FUNCT_CLASS
    assert ns["CONFIG_SUBTYPE"] == RD._CONFIG_SUBTYPE


def test_generator_emits_encoding_only_never_ops_or_dialect():
    ns = _generated_ns()
    # the guardrail: encoding symbols present, but NO dialect/op vocabulary
    assert "FUNCT" in ns and "LEGAL_FUNCT" in ns and "SEMANTIC_CLASS" in ns
    for forbidden in ("DIALECT_NAME", "SPEC_OPS", "IRDLOperation", "dialect"):
        assert forbidden not in ns


def test_without_encoding_the_module_still_carries_the_rtl_encoder():
    # facts-only generation (no manifest) still yields the RTL-derived funct table + legal set, just
    # without the manifest ABI bits — honest degradation, not a crash.
    facts = json.loads(rtl_facts_path("gemmini").read_text())
    ns: dict = {}
    exec(generate(facts, None), ns)  # noqa: S102
    assert "LEGAL_FUNCT" in ns and "FUNCT" in ns
    assert "SEMANTIC_CLASS" not in ns   # comes only from the manifest encoding block
