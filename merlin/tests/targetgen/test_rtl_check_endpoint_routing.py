"""Arm-4 RTL checks are ENDPOINT-routed and fully DERIVED — no target literals, no gemmini checks leaking
onto a self-hosted-ISA target.

The bug these lock: the check compiler keyed "is this RoCC?" on the presence of a ``funct_decode_table``,
but the mlc icmp-fanout extractor synthesises one for a self-hosted decoder too (atlas gets a table with
custom_opcode 0x7b) — so atlas was handed gemmini dialect/trace FileCheck (gemmini.matmul, ABI 0x7b). The
fix routes on the DERIVED endpoint_kind and gives external_backend its own kernel opcode-LEGALITY check
(every emitted opcode ∈ the RTL-discovered legal set), derived from facts at run time.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import rtl_check_compiler as CC
from merlin.targetgen import rtl_check_runner as RUN


def _facts(target):
    f = RUN.load_facts(target)
    if not f or not (f.get("facts") or f).get("interfaces"):
        pytest.skip(f"{target} RTL facts not derivable (mlc/circt absent)")
    return f


def test_gemmini_is_rocc_routed_and_atlas_is_not():
    fg, fa = _facts("gemmini"), _facts("atlas")
    assert CC._is_rocc_target("gemmini", fg) is True
    assert CC._is_rocc_target("atlas", fa) is False           # endpoint external_backend, NOT decode-table


def test_compile_checks_routes_by_endpoint_not_decode_table():
    cap = {"name": "m", "operation": {"op": "matmul"}, "inputs": [{"role": "output", "shape": [16, 16]}]}
    g = CC.compile_checks(_facts("gemmini"), cap, "gemmini")
    a = CC.compile_checks(_facts("atlas"), cap, "atlas")
    # gemmini: RoCC dialect+trace, no kernel check
    assert g["dialect"] and g["trace"] and g["kernel"] is None
    # atlas: kernel opcode-legality check ONLY — no gemmini dialect/trace leaking on
    assert a["kernel"] and a["dialect"] is None and a["trace"] is None
    # and no gemmini literal anywhere in the atlas checks
    assert "gemmini" not in (a["kernel"] or "")


def test_kernel_legality_passes_legal_and_catches_illegal_opcode():
    fa = _facts("atlas")
    cap = {"name": "x", "operation": {"op": "matmul"}, "inputs": []}
    checks = CC.compile_checks(fa, cap, "atlas")["kernel"]
    fc = RUN.find_filecheck()
    if not fc:
        pytest.skip("FileCheck absent")
    # a kernel whose opcodes are all in the RTL legal set -> PASS
    legal = sorted((next(i for i in (fa.get("facts", fa).get("interfaces"))
                         if i.get("name") == "funct_decode_table")).get("legal_funct"))
    good = "k:\n" + "\n".join(f"  .word {v}" for v in legal[:3]) + "\n"
    ok, _ = RUN.run_filecheck(fc, checks, RUN.render_kernel_decode(good, fa), "KERNEL")
    assert ok is True
    # an opcode outside the legal set -> ILLEGAL_OPCODE_COUNT>0 -> FileCheck FAIL (the real RTL insight)
    illegal_val = next(v for v in range(1, 1 << 14) if v not in set(legal))
    bad = f"k:\n  .word {illegal_val}\n"
    dec = RUN.render_kernel_decode(bad, fa)
    assert "ILLEGAL_OPCODE_COUNT 1" in dec
    ok2, _ = RUN.run_filecheck(fc, checks, dec, "KERNEL")
    assert ok2 is False


def test_no_hardcoded_abi_fallback():
    # a facts record with no funct_decode_table yields no ABI (never the old 0x7b/0x3 gemmini default)
    assert CC._facts_abi({"interfaces": []}) is None


def test_class_coverage_decodes_words_and_catches_missing_required_class():
    """The field-decode class-coverage check classifies each emitted word via the ISA-def decode
    signatures and asserts the capsule's required classes were actually emitted — so a matmul kernel that
    emitted the wrong ops (no MXU matmul) fails, which opcode-legality alone passes. Fully derived."""
    from merlin.targetgen import isa_taxonomy as IT
    IT.clear_cache()
    tax = IT.taxonomy_for_target("atlas")
    if not tax or not any("fixed_mask" in e for e in (tax.get("by_mnemonic") or {}).values()):
        pytest.skip("atlas taxonomy / decode signatures not derivable (model venv absent)")
    # decode is unambiguous on the real op encodings (the ISA-def signatures are disjoint)
    fa = _facts("atlas")
    fc = RUN.find_filecheck()
    if not fc:
        pytest.skip("FileCheck absent")
    # a matmul capsule requiring the MXU sequence, graded against a kernel that emitted only weight-pushes
    cap = {"name": "m", "operation": {"op": "matmul"},
           "expected": {"instruction_classes": ["TensorBaseOffset", "MXUWeightPush",
                                                "MXUMatMul", "MXUAccumulatorPop"]}}
    checks = CC.compile_checks(fa, cap, "atlas")["kernel"]
    assert "CLASS_PRESENT MXUMatMul" in checks           # coverage assertions are compiled in
    # find a word that decodes to MXUWeightPush and one that decodes to nothing-MXUMatMul
    push_word = next((w for w in range(0, 1 << 16)
                      if IT.decode_word(w, tax) == ["MXUWeightPush"]), None)
    if push_word is None:
        pytest.skip("no MXUWeightPush encoding found")
    kernel = f"k:\n  .word {push_word}\n"                # only a weight-push; no load/matmul/pop
    dec = RUN.render_kernel_decode(kernel, fa, tax)
    assert "CLASS_PRESENT MXUWeightPush" in dec and "CLASS_PRESENT MXUMatMul" not in dec
    ok, _ = RUN.run_filecheck(fc, checks, dec, "KERNEL")
    assert ok is False                                   # missing MXUMatMul/TensorBaseOffset/pop → FAIL
