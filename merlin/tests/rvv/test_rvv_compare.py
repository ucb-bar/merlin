"""Structural generated-vs-curated comparison — the gap-router's diagnostic signal."""
from merlin.kernels.compare import (RvvFingerprint, compare_fingerprints,
                                    _decisions_from_asm, _canon_op)

_KEY = {"op": "gemm", "dtype": "f32", "shape_regime": "square_small"}

# expert: e32m4 + scalar-broadcast fused fma + vl-polymorphic loop
_CURATED_C = ("vl = __riscv_vsetvl_e32m4(n);\n"
              "vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, a0, vb, vl);\n")
# ours: separate fmul+fadd (no fma), fixed immediate VL, lower LMUL
_GEN_ASM = (
    "   10:\t00\tvsetivli\ta0,8,e32,m1,ta,ma\n"
    "   14:\t00\tvle32.v\tv8,(a1)\n"
    "   18:\t00\tvfmul.vv\tv8,v16,v24\n"
    "   1c:\t00\tvfadd.vv\tv8,v8,v0\n"
)


def test_canon_op_unifies_c_and_asm():
    assert _canon_op("__riscv_vfmacc_vf_f32m4") == "vfmacc"
    assert _canon_op("vfmacc.vv") == "vfmacc"
    assert _canon_op("vle32.v") == _canon_op("__riscv_vle32_v_f32m4") == "vle32"
    assert _canon_op("vsetivli") == _canon_op("vsetvli") == "vsetvl"


def test_decisions_from_asm_detects_no_fma_gap():
    d = _decisions_from_asm(_GEN_ASM)
    assert d["fma_form"] is None          # vfmul+vfadd, NOT fused -> the measured gap
    assert d["lmul_class"] == "m1"
    assert d["vl_strategy"] == "vsetivli_fixed"


def test_compare_surfaces_the_three_gaps():
    cur = RvvFingerprint.from_curated(_CURATED_C, _KEY, "xnnpack")
    gen = RvvFingerprint.from_objdump(_GEN_ASM, _KEY, "hand_v0")
    r = compare_fingerprints(cur, gen)
    assert r["decision_match"]["fma_form"] is False
    assert r["decision_match"]["lmul_class"] is False
    assert r["decision_match"]["vl_strategy"] is False
    assert 0.0 <= r["structural_match"] <= 1.0
    keys = {d.split(":")[0] for d in r["divergences"]}
    assert {"fma_form", "lmul_class", "vl_strategy"} <= keys


def test_identical_fingerprints_match_fully():
    cur = RvvFingerprint.from_curated(_CURATED_C, _KEY, "xnnpack")
    r = compare_fingerprints(cur, cur)
    assert r["structural_match"] == 1.0
    assert r["divergences"] == []
