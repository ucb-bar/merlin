"""RVV intrinsic decision extractor — the feature layer the kernel miner needs.

Synthetic-snippet units (deterministic) + a no-op guard for non-RVV kernels. Real-corpus
extraction is exercised by the mining run (S3.4); here we lock the decision logic, including
OpenBLAS #define-macro resolution.
"""
from merlin.kernels.types import NormalizedKernel
from merlin.kernels.features.rvv_intrinsics import extract_rvv_intrinsics


def _nk(text, dtype="f32", target="rvv"):
    return NormalizedKernel(source="x", target=target, path="k.c", op="gemm",
                            dtype=dtype, raw_text=text)


def test_xnnpack_style_f32_gemm():
    text = ("size_t vl = __riscv_vsetvlmax_e32m4();\n"
            "vl = __riscv_vsetvl_e32m4(n);\n"
            "vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, a0, vb, vl);\n")
    r = extract_rvv_intrinsics(_nk(text, "f32"), {})["rvv"]
    assert r["lmul_class"] == "m4"
    assert r["fma_form"] == "vf"           # scalar-broadcast GEMM idiom
    assert r["int_widening"] is False
    assert r["accumulator_dtype"] == "f32"
    assert r["vl_strategy"] == "vsetvl_loop"
    assert r["register_block"]["mr"] == 1  # one vacc register


def test_int8_widening_gemm():
    text = ("vl = __riscv_vsetvl_e32m4(n);\n"
            "vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, a0, vb, vl);\n"
            "out = __riscv_vnclip_wx_i8m1(acc, shift, vl);\n"
            "__riscv_vse8_v_i8m1(o, out, vl);\n")
    r = extract_rvv_intrinsics(_nk(text, "i8"), {})["rvv"]
    assert r["int_widening"] is True
    assert r["accumulator_dtype"] == "i32"
    assert r["requant_epilogue"] is True   # vnclip + vse8 narrowing store


def test_openblas_macro_resolution():
    # OpenBLAS hides LMUL + fma form behind #define aliases; the body uses the alias.
    text = ("#define VSETVL(n)  __riscv_vsetvl_e32m2(n)\n"
            "#define VFMACCVF   __riscv_vfmacc_vf_f32m2\n"
            "for (...) { vl = VSETVL(n); vr = VFMACCVF(vr, a, vb, vl); }\n")
    r = extract_rvv_intrinsics(_nk(text, "f32"), {})["rvv"]
    assert r["lmul_class"] == "m2"
    assert r["fma_form"] == "vf"
    assert r["vl_strategy"] == "vsetvl_loop"


def test_reduction_form_and_fixed_vl():
    text = ("vl = __riscv_vsetvlmax_e32m1();\n"
            "acc = __riscv_vfredusum_vs_f32m1_f32m1(acc, v, z, vl);\n")
    r = extract_rvv_intrinsics(_nk(text, "f32"), {})["rvv"]
    assert r["reduction_form"] == "vfredusum"
    assert r["vl_strategy"] == "vsetvlmax_fixed"


def test_non_rvv_is_noop():
    assert extract_rvv_intrinsics(_nk("_mm256_fmadd_ps(a,b,c);", target="avx2"), {}) == {}
    assert extract_rvv_intrinsics(_nk("plain C, no intrinsics", target="rvv"), {}) == {}
