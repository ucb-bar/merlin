"""RTL checks route only through a verified executable endpoint.

An RTL equality fan-out observes one decoder field; it does not establish an
instruction set, an endpoint, or whole-word legality.
"""

from __future__ import annotations

import external_sources
import pytest

from merlin.targetgen import rtl_check_compiler as CC
from merlin.targetgen import rtl_check_runner as RUN

pytestmark = pytest.mark.target("atlas", "gemmini", "mx_gemmini", "radiance")


def _facts(target):
    f = RUN.load_facts(target)
    if not f or not (f.get("facts") or f).get("interfaces"):
        pytest.skip(f"{target} RTL facts not derivable (mlc/circt absent)")
    return f


@external_sources.requires_rtl("gemmini", "atlas")
def test_gemmini_is_rocc_routed_and_atlas_is_not():
    fg, fa = _facts("gemmini"), _facts("atlas")
    assert CC._is_rocc_target("gemmini", fg) is True
    assert CC._is_rocc_target("atlas", fa) is False  # field-local observations are not RoCC evidence


def test_mx_gemmini_routes_trace_and_radiance_routes_kernel():
    """RoCC transport routes MX; SIMT routes only when its selected provider supplies facts."""
    from merlin.targetgen.rtl import mlc_bridge

    ek_mx = CC._endpoint_kind_for("mx_gemmini", {})
    ek_rad = CC._endpoint_kind_for("radiance", {})
    assert ek_mx == "inline_asm_insn" and CC._is_rocc_target("mx_gemmini", {}) is True
    if mlc_bridge.simt_facts("radiance"):
        assert ek_rad == "external_backend"
    else:
        assert ek_rad == "unresolved"
    assert CC._is_rocc_target("radiance", {}) is False


@external_sources.requires_rtl("atlas")
def test_compile_checks_does_not_route_on_decoder_field_alone():
    cap = {"name": "m", "operation": {"op": "matmul"}, "inputs": [{"role": "output", "shape": [16, 16]}]}
    a = CC.compile_checks(_facts("atlas"), cap, "atlas")
    # Atlas's observed decoder field alone must not authorize either endpoint.
    assert a["kernel"] is None and a["trace"] is None and "dialect" not in a
    assert a["endpoint_status"] == "unverified"
    assert "gemmini" not in (a["kernel"] or "")


@external_sources.requires_rtl("atlas")
def test_observed_decoder_values_cannot_validate_entire_kernel_words():
    fa = _facts("atlas")
    observed = next(i for i in (fa.get("facts", fa).get("interfaces")) if i.get("name") == "funct_decode_table")
    assert observed["scope"] == "observed_decode_field"
    # The 14-bit field concatenates distant instruction bits. A matching low
    # word cannot be called legal, and an arbitrary word cannot be rejected.
    word = observed["legal_funct"][0]
    dec = RUN.render_kernel_decode(f"k:\n  .word {word}\n  .word 0xdeadbeef\n", fa)
    assert "ILLEGAL_OPCODE_COUNT -" in dec
    assert "legal=?" in dec


def test_no_hardcoded_abi_fallback():
    from gemmini_rtl_test_support import checks

    # a facts record with no funct_decode_table yields no ABI (never the old 0x7b/0x3 gemmini default)
    assert checks._facts_abi({"interfaces": []}) is None


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
    cap = {
        "name": "m",
        "operation": {"op": "matmul"},
        "expected": {"instruction_classes": ["TensorBaseOffset", "MXUWeightPush", "MXUMatMul", "MXUAccumulatorPop"]},
    }
    checks = CC.compile_checks(fa, cap, "atlas")["kernel"]
    assert "CLASS_PRESENT MXUMatMul" in checks  # coverage assertions are compiled in
    # find a word that decodes to MXUWeightPush and one that decodes to nothing-MXUMatMul
    push_word = next((w for w in range(0, 1 << 16) if IT.decode_word(w, tax) == ["MXUWeightPush"]), None)
    if push_word is None:
        pytest.skip("no MXUWeightPush encoding found")
    kernel = f"k:\n  .word {push_word}\n"  # only a weight-push; no load/matmul/pop
    dec = RUN.render_kernel_decode(kernel, fa, tax)
    assert "CLASS_PRESENT MXUWeightPush" in dec and "CLASS_PRESENT MXUMatMul" not in dec
    ok, _ = RUN.run_filecheck(fc, checks, dec, "KERNEL")
    assert ok is False  # missing MXUMatMul/TensorBaseOffset/pop → FAIL


def _canonical_word(tax, cls, nonzero_operands=False):
    """A canonical valid encoding of some op in `cls` = its derived fixed_value, optionally with one
    operand bit set (so its operand payload is non-zero for the field-sanity check)."""
    from merlin.targetgen import isa_taxonomy as IT

    for e in (tax.get("by_mnemonic") or {}).values():
        if e.get("class") == cls and e.get("fixed_value") is not None:
            w = int(e["fixed_value"])
            if nonzero_operands:
                opbits = (~int(e["fixed_mask"])) & 0xFFFFFFFF
                w |= opbits & (-opbits)  # lowest operand bit
            assert IT.decode_word(w, tax) == [cls]  # decodes back unambiguously
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

    capd = yaml.safe_load(
        (merlin_dir() / "contract/capsules/atlas/isa/AT2_single_tile_matmul/capsule.yaml").read_text()
    )
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

    assert run(seq) is True  # correct load→push→matmul→pop
    assert run(["MXUMatMul", "MXUWeightPush", "TensorBaseOffset", "MXUAccumulatorPop"]) is False  # bad order
    assert (
        run(["TensorBaseOffset", "MXUWeightPush", "MXUMatMul", "MXUMatMul", "MXUAccumulatorPop"]) is False
    )  # 2 matmuls != 1 tile
    # base-zero load (addresses DRAM 0) → field-sanity FAIL
    zero_load = _canonical_word(tax, "TensorBaseOffset", nonzero_operands=False)
    k = (
        "atlas_kernel:\n"
        + f"  .word 0x{zero_load:x}\n"
        + "".join(f"  .word 0x{W[c]:x}\n" for c in ("MXUWeightPush", "MXUMatMul", "MXUAccumulatorPop"))
        + "  ret\n"
    )
    okz, _ = RUN.run_filecheck(fc, checks, RUN.render_kernel_decode(k, fa, tax), ["KERNEL", "KORDER"])
    assert okz is False
