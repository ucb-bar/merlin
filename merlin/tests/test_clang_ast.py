"""R2-enrich: typed C-intrinsic source extractor + source-level CCA + asm-vs-source agreement.

Core tests use a synthetic clang-AST JSON fixture (no clang/headers needed). The point: read RVV
decisions from RESOLVED intrinsic types, not regex over C.
"""
from __future__ import annotations

from merlin.kernels import cca
from merlin.kernels.decode import clang_ast as ca

# minimal clang -ast-dump=json shape: a CallExpr to __riscv_vfmacc returning vfloat32m4_t.
_AST = {
    "kind": "TranslationUnitDecl",
    "inner": [{
        "kind": "FunctionDecl", "name": "gemm",
        "inner": [{
            "kind": "CallExpr",
            "type": {"qualType": "vfloat32m4_t"},
            "inner": [{
                "kind": "ImplicitCastExpr",
                "inner": [{"kind": "DeclRefExpr",
                           "referencedDecl": {"kind": "FunctionDecl",
                                              "name": "__riscv_vfmacc_vf_f32m4"}}],
            }],
        }],
    }],
}


def test_vtype_from_typename():
    assert ca._vtype_from_typename("vfloat32m4_t") == (32, 4.0)
    assert ca._vtype_from_typename("vint8mf2_t") == (8, 0.5)
    assert ca._vtype_from_typename("vfloat16m1_t") == (16, 1.0)


def test_extract_typed_intrinsic_from_ast():
    facts = ca.facts_from_ast_json(_AST)
    assert facts.ok and len(facts.intrinsics) == 1
    c = facts.intrinsics[0]
    assert c.name == "__riscv_vfmacc_vf_f32m4" and c.ret_type == "vfloat32m4_t"
    assert c.sew == 32 and c.lmul == 4.0
    assert facts.has("vfmacc") == 1


def test_lift_source_and_agreement():
    facts = ca.facts_from_ast_json(_AST)
    src = cca.lift_source(facts, op="matmul", source="xnnpack_gemm")
    assert src.compute.contraction_form == "fused_fma"   # from the typed vfmacc call
    assert src.vector.sew == 32 and src.vector.lmul == 4.0
    # asm-lifted "ours" that's mul_add/m2 disagrees -> the validity gate flags it
    ours = cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form="mul_add"),
                   vector=cca.VectorFacet(sew=32, lmul=2.0),
                   provenance={"level": "asm"})
    rep = cca.cca_agree(src, ours)
    assert not rep.agree
    assert any("contraction_form" in d for d in rep.disagreements)


def test_extract_degrades_without_clang(tmp_path):
    # a non-C path / no headers -> ok=False, never raises (asm path is independent)
    facts = ca.extract(tmp_path / "nope.c")
    assert facts.ok is False
