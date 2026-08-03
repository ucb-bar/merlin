"""rocc_decode must SEE every inline-asm spelling — never silently drop one.

Regression for the silent-drop bug: the decoder gated instruction parsing on the pretty op spelling
``llvm.inline_asm`` only, so a backend that emitted the generic-op spelling ``"llvm.intr.inlineasm"()``
produced an EMPTY trace (zero instructions), and ``trace_check`` then reported every required class
"missing" — a misleading verdict that hid the real problem (a mis-encoded, but present, RoCC stream).

Invariant asserted here (not per-capsule goldens): if inline-asm ops are present, the decoded trace is
non-empty; a recognized canonical form decodes to its class; an unrecognized-but-present form is
recorded as UNKNOWN (fail-closed), never dropped.
"""
from __future__ import annotations

from merlin.targetgen import rocc_decode as RD

# Canonical emitter form the certified native path (gemmini_codegen_mlir.py) produces.
_CANON_MVIN = (
    '  %a = llvm.ptrtoint %arg0 : !llvm.ptr to i64\n'
    '  %rs2 = llvm.mlir.constant(4503668346847232 : i64) : i64\n'
    '  llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 2, x0, $0, $1", "r,r" %a, %rs2'
    ' : (i64, i64) -> ()\n'
)

# The generic-op spelling a conformant-but-different backend may emit (the form GLM-5 emitted in the
# gemmini arm-4 run). Different op name, ``%N`` operand refs, collapsed 5-field .insn: not the canonical
# encoding, so it must land as UNKNOWN — but it must NOT vanish.
_GENERIC_MVIN = (
    '  %mvin_rs2 = llvm.mlir.constant(4503668346847232 : i64) : i64\n'
    '  "llvm.intr.inlineasm"() {asm = ".insn r 0x7b, 2, x0, %0, %1", constraints = "r,r",'
    ' has_side_effects = true} (%arg0, %mvin_rs2) : (!llvm.ptr, i64) -> ()\n'
)


def test_canonical_inline_asm_decodes_to_class():
    trace = RD.decode_text(_CANON_MVIN, target="gemmini")
    classes = [i["class"] for i in trace["instructions"]]
    assert classes == ["MVIN"], classes


def test_generic_op_spelling_is_not_silently_dropped():
    # The core regression: an entire generic-spelled asm stream must be SEEN, not skipped.
    trace = RD.decode_text(_GENERIC_MVIN, target="gemmini")
    ins = trace["instructions"]
    assert len(ins) == 1, f"generic-op asm was dropped (empty trace): {ins}"
    assert ins[0]["class"] == "UNKNOWN", ins  # non-canonical encoding -> fail-closed, not a fake MVIN
    assert ins[0].get("raw"), "UNKNOWN instruction must carry the raw asm for actionable feedback"


def test_present_asm_never_yields_empty_trace():
    # Whatever the spelling, if inline-asm ops are present the trace is non-empty (no silent zero that
    # would make trace_check falsely report all-classes-missing).
    for text in (_CANON_MVIN, _GENERIC_MVIN, _CANON_MVIN + _GENERIC_MVIN):
        assert RD.decode_text(text, target="gemmini")["instructions"], f"empty trace for present asm:\n{text}"


def test_decoded_trace_conforms_to_instruction_trace_schema():
    # The trace the runner writes MUST validate against the instruction_trace schema, else every graded
    # capsule crashes runner_internal. Regression for: abi codes emitted as raw ints (123/3) instead of
    # the schema-required hex STRINGS ("0x7b"/"0x3").
    from merlin.targetgen.contract import schemas as S
    trace = RD.decode_text(_CANON_MVIN, source="lowered.llvm.mlir", target="gemmini")
    abi = trace["abi"]
    assert isinstance(abi["custom_opcode"], str) and abi["custom_opcode"].startswith("0x"), abi
    assert isinstance(abi["funct3"], str), abi
    S.validate(trace, "instruction_trace")  # must not raise


def test_fence_recognized_in_both_spellings():
    for op in ('llvm.inline_asm has_side_effects "fence", "" : () -> ()',
               '"llvm.intr.inlineasm"() {asm = "fence", constraints = ""} () : () -> ()'):
        trace = RD.decode_text("  " + op + "\n", target="gemmini")
        assert [i["class"] for i in trace["instructions"]] == ["FENCE"], op
