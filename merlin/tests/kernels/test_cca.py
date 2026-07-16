"""R3: Common Compute Abstraction lift + cross-level agreement (target-agnostic)."""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir

import pytest

from merlin.kernels import cca
from merlin.kernels.decode import objdump, rvv

_ASM_DIR = merlin_dir() / "tests" / "data" / "cca_asm"

# matmul-ish stream: mul_add (vfmul+vfadd), no vfmacc, fixed vsetivli, e32m2.
_MUL_ADD = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmul.vv\tv8, v8, v9
       c: 00     \tvfadd.vv\tv10, v10, v8
"""
# fused stream: vfmacc present, e32m4.
_FUSED = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m4, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmacc.vv\tv12, v8, v9
"""


# scalar activation: a per-element libm call loop (jal to <expf>), scalar float, no vfmacc.
_SCALAR_LIBM_ACT = """\

Disassembly of section .text:
0 <gelu>:
       0: 00     \tflw\tfa0, 0x0(a0)
       4: 00     \tjal\tra, 0x100 <expf>
       8: 00     \tfmul.s\tfa0, fa0, fa1
       c: 00     \tfsw\tfa0, 0x0(a1)
"""
# vectorized activation: an inline minimax polynomial (vfmacc chain), no libm call.
_VECTOR_POLY_ACT = """\

Disassembly of section .text:
0 <gelu>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmacc.vv\tv12, v8, v9
       c: 00     \tvfmacc.vv\tv12, v12, v10
"""


def _cca(monkeypatch, snippet, source):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: snippet)
    return cca.lift_asm(rvv.decode("x.o"), op="matmul", source=source)


def _cca_op(monkeypatch, snippet, op):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: snippet)
    return cca.lift_asm(rvv.decode("x.o"), op=op, source="act")


def test_activation_vectorization_scalar_libm(monkeypatch):
    c = _cca_op(monkeypatch, _SCALAR_LIBM_ACT, "gelu")
    assert c.compute.activation_vectorization == "scalar_libm_call"   # calls <expf>


def test_activation_vectorization_vectorized_poly(monkeypatch):
    c = _cca_op(monkeypatch, _VECTOR_POLY_ACT, "gelu")
    assert c.compute.activation_vectorization == "vectorized_polynomial"  # vfmacc poly, no libm call


def test_activation_vectorization_none_for_matmul(monkeypatch):
    # a plain matmul (not a transcendental activation) must NOT be classified as an activation.
    c = _cca(monkeypatch, _FUSED, "expert")
    assert c.compute.activation_vectorization is None


def test_lift_asm_mul_add(monkeypatch):
    c = _cca(monkeypatch, _MUL_ADD, "ours")
    assert c.backend == ["rvv"]
    assert c.compute.contraction_form == "mul_add"   # no vfmacc -> mul_add
    assert c.vector.sew == 32 and c.vector.lmul == 2.0
    assert c.vector.vl_strategy == "vsetivli_fixed"


def test_lift_asm_fused(monkeypatch):
    c = _cca(monkeypatch, _FUSED, "expert")
    assert c.compute.contraction_form == "fused_fma"
    assert c.vector.lmul == 4.0


def test_agreement_flags_divergences(monkeypatch):
    ours = _cca(monkeypatch, _MUL_ADD, "ours")
    expert = _cca(monkeypatch, _FUSED, "expert")
    rep = cca.cca_agree(expert, ours)
    assert not rep.agree
    axes = {d.split(":")[0] for d in rep.disagreements}
    assert "compute.contraction_form" in axes   # the vfmacc gap
    assert "vector.lmul" in axes                 # 4 vs 2
    # self-agreement is the validity baseline
    assert cca.cca_agree(ours, ours).agree


def test_composite_backend_supported():
    # a heterogeneous region (NPU+RVV) is just a backend list — not a special case
    c = cca.CCA(op="attention", backend=["npu", "rvv"])
    assert c.backend == ["npu", "rvv"]
    assert c.spatial is None and c.dataflow is None   # facets populated only when relevant


# ---- accumulator-residency / register-block / VL-NR: the abstraction reads the expert-win
# properties faithfully off REAL disassembly (no regex; via decode.rvv). These pin the ABSTRACTION
# (does the CCA *see* the gap?), not a memorized shape — the asm fixtures are whole-kernel
# disassembly built from the expert ceiling drivers + our own baseline / impr-feature codegen
# (provenance in data/cca_asm/AGENT.md).

def _lift_fixture(monkeypatch, name: str) -> cca.CCA:
    text = (_ASM_DIR / name).read_text()
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: text)
    return cca.lift_asm(rvv.decode(name), op="matmul", source=name)


def test_lift_reads_accumulator_resident_on_experts(monkeypatch):
    # Both expert GEMMs keep the accumulator in vector registers across the whole K loop (no in-loop
    # spill) and commit C once after — the abstraction must read accumulator_resident=True on each.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.compute.contraction_form == "fused_fma", fx
        assert c.compute.accumulator_resident is True, fx
        # a register block is recovered (MR distinct accumulators, NR lmul-scaled lanes)
        assert c.compute.register_block is not None, fx
        mr, nr = c.compute.register_block
        assert isinstance(mr, int) and mr >= 1, fx
        assert nr[0] == "vsetvlmax", fx


def test_lift_reads_accumulator_dtype(monkeypatch):
    # The accumulate width is captured (ISA-grounded): the f32 GEMM experts accumulate in f32. This is
    # the dtype-datapath axis the compiler exposes via the dtype_strategy knob.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.compute.accumulator_dtype == "f32", fx


def test_decode_text_matches_object_path(monkeypatch):
    # a CCA lifted from objdump TEXT (rvv.decode_text — what the beam has in objdump.txt) equals the
    # object-file path (rvv.decode). Enables lifting a fork's CCA with no toolchain.
    text = (_ASM_DIR / "openblas_sgemm_rvv.objdump").read_text()
    from_text = cca.lift_asm(rvv.decode_text(text), op="matmul", source="openblas")
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: text)
    from_obj = cca.lift_asm(rvv.decode("x.o"), op="matmul", source="openblas")
    assert from_text.compute.contraction_form == from_obj.compute.contraction_form == "fused_fma"
    assert from_text.compute.accumulator_resident == from_obj.compute.accumulator_resident
    assert from_text.vector.tail == from_obj.vector.tail


def _matmul_record(dtype: str):
    from merlin.frontends.linalg_mlir import MatmulRecord
    return MatmulRecord(kind="linalg.matmul", m=64, k=64, n=64, lhs_shape=(64, 64), rhs_shape=(64, 64),
                        dtype=dtype, weight_arg_index=1, weight_name="w", prov={"prov.op": "matmul"})


def test_lift_graph_partial_from_dtype():
    # the flat-graph analyzer derives only op + dtype datapath facets (partial, by design)
    g = cca.lift_graph(_matmul_record("f32"))
    assert g.compute.op == "matmul" and g.compute.accumulator_dtype == "f32"
    assert g.compute.widening is None                          # f32 -> not a widening MAC
    g8 = cca.lift_graph(_matmul_record("i8"))
    assert g8.compute.accumulator_dtype == "i32" and g8.compute.widening is True


def test_asm_and_graph_analyzers_agree(monkeypatch):
    # the two DETERMINISTIC analyzers (asm decode + flat graph) must agree on the shared populated
    # facets — cca_agree is the validity gate that quarantines a bad reconstruction on either side.
    asm = _lift_fixture(monkeypatch, "openblas_sgemm_rvv.objdump")   # f32 GEMM -> acc f32
    graph = cca.lift_graph(_matmul_record("f32"))
    rep = cca.cca_agree(asm, graph)
    assert rep.agree, rep.disagreements
    assert "compute.accumulator_dtype" in rep.compared_fields    # the shared facet actually compared


def test_lift_reads_vector_tail(monkeypatch):
    # The tail policy (ta|tu) is captured from the decoded vsetvl vtype state (not guessed). The GEMM
    # fixtures run tail-agnostic (ta). Populating it feeds the eventual tail route + cca_agree.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.vector.tail in ("ta", "tu"), fx


def test_lift_reads_xnnpack_nr_tracks_vsetvlmax(monkeypatch):
    # XNNPACK 1x4v is the VL-adaptive expert: a polymorphic vsetvli VL-loop, so NR tracks vsetvlmax.
    c = _lift_fixture(monkeypatch, "xnnpack_f32_gemm_rvv.objdump")
    assert c.vector.vl_strategy == "vsetvl_loop"
    assert c.compute.nr_is_vsetvlmax is True
    assert c.compute.register_block[0] == 1            # MR=1 (one accumulator), 1x4v


def test_lift_reads_ours_baseline_not_resident(monkeypatch):
    # Our FROZEN baseline lowering does not even form a fused MAC (vfmul+vfadd) — the deepest gap.
    c = _lift_fixture(monkeypatch, "ours_baseline_matmul.objdump")
    assert c.compute.contraction_form == "mul_add"
    assert c.compute.accumulator_resident is not True   # None/False — never the expert's True


def test_lift_reads_ours_accum_feature_still_not_resident(monkeypatch):
    # The accumulator_resident_microkernel feature DOES form vfmacc, but the emitted asm still spills
    # the accumulator through the stack inside the K loop (whole-register vsNr/vlNre). The abstraction
    # must honestly read accumulator_resident=False — i.e. it SEES the residual gap to the experts.
    c = _lift_fixture(monkeypatch, "ours_accum_resident_matmul.objdump")
    assert c.compute.contraction_form == "fused_fma"
    assert c.compute.accumulator_resident is False


def test_accumulator_residency_divergence_expert_vs_ours(monkeypatch):
    # End-to-end of the abstraction: expert(resident=True) vs ours(resident=False) DISAGREE on the
    # shared compute.accumulator_resident axis — the gap is now a typed, comparable CCA field.
    expert = _lift_fixture(monkeypatch, "openblas_sgemm_rvv.objdump")
    ours = _lift_fixture(monkeypatch, "ours_accum_resident_matmul.objdump")
    rep = cca.cca_agree(expert, ours)
    assert "compute.accumulator_resident" in rep.compared_fields
    assert any("accumulator_resident" in d for d in rep.disagreements)


def test_accumulator_resident_is_target_agnostic_compute_field():
    # Promoted onto the SHARED ComputeFacet (not just SpatialFacet) — every backend answers the same
    # "is the accumulator resident across the reduction" question on the same axis.
    assert "accumulator_resident" in cca.ComputeFacet().__dataclass_fields__
    # the Gemmini/spatial view of the SAME concept still exists (compared per-facet)
    assert "accumulator_resident" in cca.SpatialFacet().__dataclass_fields__


@pytest.mark.skipif(not _ASM_DIR.is_dir(), reason="cca asm fixtures absent")
def test_fixtures_present():
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump",
               "ours_baseline_matmul.objdump", "ours_accum_resident_matmul.objdump"):
        assert (_ASM_DIR / fx).is_file()
