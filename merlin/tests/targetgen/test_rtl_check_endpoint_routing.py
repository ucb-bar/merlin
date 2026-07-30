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
    # gemmini: RoCC trace check over the decoded stream, no kernel check; no dialect-MLIR family exists
    assert g["trace"] and g["kernel"] is None and "dialect" not in g
    # atlas: kernel opcode-legality check ONLY — no RoCC trace leaking on
    assert a["kernel"] and a["trace"] is None and "dialect" not in a
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


def _canonical_word(tax, cls, nonzero_operands=False):
    """A canonical valid encoding of some op in `cls` = its derived fixed_value, optionally with one
    operand bit set (so its operand payload is non-zero for the field-sanity check)."""
    from merlin.targetgen import isa_taxonomy as IT
    for e in (tax.get("by_mnemonic") or {}).values():
        if e.get("class") == cls and e.get("fixed_value") is not None:
            w = int(e["fixed_value"])
            if nonzero_operands:
                opbits = (~int(e["fixed_mask"])) & 0xFFFFFFFF
                w |= (opbits & (-opbits))                 # lowest operand bit
            assert IT.decode_word(w, tax) == [cls]        # decodes back unambiguously
            return w
    return None


def test_kernel_order_tiling_and_field_sanity_checks():
    """The three structural checks — required-class ORDER, mesh-TILING count, memory-op field-sanity —
    pass a correct kernel and fail order/tiling/base-zero violations. Derived + target-agnostic."""
    from merlin.targetgen import isa_taxonomy as IT
    IT.clear_cache()
    tax = IT.taxonomy_for_target("atlas")
    if not tax or not any("fixed_mask" in e for e in (tax.get("by_mnemonic") or {}).values()):
        pytest.skip("atlas decode signatures not derivable (model venv absent)")
    fa = _facts("atlas")
    fc = RUN.find_filecheck()
    if not fc:
        pytest.skip("FileCheck absent")
    import yaml
    from merlin.common.paths import merlin_dir
    capd = yaml.safe_load((merlin_dir()
                           / "contract/capsules/atlas/isa/AT2_single_tile_matmul/capsule.yaml").read_text())
    seq = ["TensorBaseOffset", "MXUWeightPush", "MXUMatMul", "MXUAccumulatorPop"]
    W = {c: _canonical_word(tax, c, nonzero_operands=(c == "TensorBaseOffset")) for c in seq}
    if any(w is None for w in W.values()):
        pytest.skip("atlas taxonomy missing an expected MXU class")
    checks = CC.compile_checks(fa, capd, "atlas")["kernel"]
    assert "CLASS_COUNT MXUMatMul 1" in checks and "CLASS_ZEROOPS TensorBaseOffset 0" in checks
    assert "KORDER: class=TensorBaseOffset" in checks

    def run(word_seq):
        k = "atlas_kernel:\n" + "".join(f"  .word 0x{W[c]:x}\n" for c in word_seq) + "  ret\n"
        ok, _ = RUN.run_filecheck(fc, checks, RUN.render_kernel_decode(k, fa, tax), ["KERNEL", "KORDER"])
        return ok

    assert run(seq) is True                                          # correct load→push→matmul→pop
    assert run(["MXUMatMul", "MXUWeightPush", "TensorBaseOffset", "MXUAccumulatorPop"]) is False  # bad order
    assert run(["TensorBaseOffset", "MXUWeightPush", "MXUMatMul", "MXUMatMul",
                "MXUAccumulatorPop"]) is False                       # 2 matmuls != 1 tile
    # base-zero load (addresses DRAM 0) → field-sanity FAIL
    zero_load = _canonical_word(tax, "TensorBaseOffset", nonzero_operands=False)
    k = ("atlas_kernel:\n" + f"  .word 0x{zero_load:x}\n"
         + "".join(f"  .word 0x{W[c]:x}\n" for c in ("MXUWeightPush", "MXUMatMul", "MXUAccumulatorPop"))
         + "  ret\n")
    okz, _ = RUN.run_filecheck(fc, checks, RUN.render_kernel_decode(k, fa, tax), ["KERNEL", "KORDER"])
    assert okz is False
