"""R4: CCA divergences -> typed compiler-action catalog (FLAG/HEURISTIC/PASS/KNOB)."""
from __future__ import annotations

from merlin.kernels import action_catalog as ac
from merlin.kernels import cca, cca_compare


def _pair():
    ours = cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form="mul_add",
                                            widening=False, epilogue="none"),
                   vector=cca.VectorFacet(sew=32, lmul=2.0, vl_strategy="vsetivli_fixed"),
                   provenance={"level": "asm", "source": "ours"})
    expert = cca.CCA(op="matmul", backend=["rvv"],
                     compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma",
                                              widening=False, epilogue="none"),
                     vector=cca.VectorFacet(sew=32, lmul=4.0, vl_strategy="vsetvl_loop"),
                     provenance={"level": "asm", "source": "xnnpack_rvv_gemm"})
    return expert, ours


def test_divergences_route_to_typed_actions():
    expert, ours = _pair()
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack_rvv_gemm"])
    actions, unrouted = ac.build_catalog(divs)
    assert unrouted == []
    by_axis = {a.divergence_axis: a for a in actions}
    cf = by_axis["compute.contraction_form"]
    assert cf.action_class == "PASS" and cf.forkable_now
    assert cf.target_seam == "impr_features:fused_vfmacc_contraction"   # the R1 feature
    assert by_axis["vector.lmul"].action_class == "KNOB" and by_axis["vector.lmul"].forkable_now
    vl = by_axis["vector.vl_strategy"]
    assert vl.action_class == "PASS" and not vl.forkable_now            # deferred work-item


def test_evidence_carried_through():
    expert, ours = _pair()
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack_rvv_gemm", "openblas_rvv_gemm"])
    actions, _ = ac.build_catalog(divs)
    assert all("xnnpack_rvv_gemm" in a.evidence for a in actions)


def test_no_divergence_when_equal():
    expert, _ = _pair()
    assert cca_compare.compare(expert, expert) == []


def test_unrouted_reported_not_dropped():
    # an axis with no route is returned as unrouted, never silently dropped
    d = cca_compare.Divergence(axis="compute.made_up_axis", expert="x", ours="y", backend="rvv")
    actions, unrouted = ac.build_catalog([d])
    assert actions == [] and len(unrouted) == 1
