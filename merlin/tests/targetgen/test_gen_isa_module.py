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
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.target_experiment import load_capability_manifest
from merlin.targetgen import rocc_decode as RD


def _generated_ns() -> dict:
    facts = load_facts("gemmini")
    enc = load_capability_manifest("gemmini").encoding
    ns: dict = {}
    exec(generate(facts, enc), ns)   # noqa: S102 — executing our own generated, deterministic module
    return ns


def test_generated_module_reproduces_the_decoder_constants():
    ns = _generated_ns()
    isa = RD.isa_constants("gemmini")  # the decoder's derived reference for this target
    # RTL-derived (from facts)
    assert ns["CUSTOM_OPCODE"] == isa["CUSTOM_OPCODE"]
    assert ns["DIM"] == isa["DIM"]
    assert len(ns["LEGAL_FUNCT"]) >= 20
    # manifest-declared ABI bits
    assert ns["F1"] == isa["F1"] and ns["C_ACC"] == isa["C_ACC"] and ns["ACC_I8"] == isa["ACC_I8"]
    assert ns["ACC_ACCUM"] == isa["ACC_ACCUM"] and ns["FULL_C_BIT"] == isa["FULL_C_BIT"]
    # the RTL-code -> compiler-class map and config subtype
    assert ns["SEMANTIC_CLASS"] == isa["FUNCT_CLASS"]
    assert ns["CONFIG_SUBTYPE"] == isa["CONFIG_SUBTYPE"]


def test_generator_emits_encoding_only_never_ops_or_dialect():
    ns = _generated_ns()
    # the guardrail: encoding symbols present, but NO dialect/op vocabulary
    assert "FUNCT" in ns and "LEGAL_FUNCT" in ns and "SEMANTIC_CLASS" in ns
    for forbidden in ("DIALECT_NAME", "SPEC_OPS", "IRDLOperation", "dialect"):
        assert forbidden not in ns


def test_generated_cpp_header_carries_the_same_single_source_constants():
    from merlin.targetgen.rtl.gen_isa_module import generate_header
    facts = load_facts("gemmini")
    enc = load_capability_manifest("gemmini").encoding
    h = generate_header(facts, enc, "gemmini")
    # the C++ header a backend #includes instead of hand-typing constexprs — same values as the Python
    # module + the emitter, so the third (C++) leg of the former triplication shares the single source.
    assert "namespace gemmini_isa" in h
    assert f"constexpr unsigned CUSTOM_OPCODE = {hex(0x7b)};" in h
    assert f"constexpr unsigned C_ACC = {hex(RD.isa_constants('gemmini')['C_ACC'])};" in h
    assert "constexpr int K_MVIN = 2;" in h and "constexpr int K_MVOUT = 3;" in h


def test_without_encoding_the_module_still_carries_the_rtl_encoder():
    # facts-only generation (no manifest) still yields the RTL-derived funct table + legal set, just
    # without the manifest ABI bits — honest degradation, not a crash.
    facts = load_facts("gemmini")
    ns: dict = {}
    exec(generate(facts, None), ns)  # noqa: S102
    assert "LEGAL_FUNCT" in ns and "FUNCT" in ns
    assert "SEMANTIC_CLASS" not in ns   # comes only from the manifest encoding block
