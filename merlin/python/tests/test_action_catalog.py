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
    # Evidence-driven, RESOLVED: knob/flag attempts measured no-op -> demoted to PASS; the PASS
    # was then implemented + certified (impr_rvv_v5: vfmacc=8065, correct) -> re-promoted forkable.
    assert cf.action_class == "PASS" and cf.forkable_now
    assert cf.target_seam == "impr_features:fused_vfmacc_contraction"
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


def test_accumulator_residency_routes_to_deferred_pass():
    # expert keeps the accumulator resident, ours does not -> a PASS action at the impr feature
    # seam. forkable_now is HONEST: the transform-dialect feature does not yet fully close it (still
    # spills the accumulator per K-tile), so it is a deferred work-item, not a green fork.
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                               backend="rvv", evidence=["openblas_rvv_gemm"])
    a = ac.route(d)
    assert a is not None and a.action_class == "PASS"
    assert a.target_seam == "impr_features:accumulator_resident_microkernel"
    assert a.forkable_now is False                       # deferred: transform path doesn't close it


def test_accumulator_residency_codegen_when_ours_unknown():
    # when ours can't even be judged (no fma loop), route to the dedicated micro-kernel CODEGEN
    # closer (the intrinsic_microkernel ceiling target), also a deferred work-item.
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=None,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "CODEGEN" and a.forkable_now is False


def test_vl_nr_routes_to_forkable_heuristic():
    # NR=vsetvlmax (VL-adaptive output tile + N-tail) is expressible today (the N-tail-safe feature
    # vectorizes small-N attention), so it is a forkable HEURISTIC.
    d = cca_compare.Divergence(axis="compute.nr_is_vsetvlmax", expert=True, ours=False,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "HEURISTIC" and a.forkable_now is True
    assert "vsetvlmax" in a.target_seam


def test_mtail_routes_to_forkable_heuristic():
    # MR=min(MR,M) (matmul M-tail clamp) is the M-side analog of nr_is_vsetvlmax: a forkable
    # HEURISTIC (the accumulator_resident_mtail feature vectorizes the M=1 token-decode matmul).
    d = cca_compare.Divergence(axis="compute.mr_adapts_to_m", expert=True, ours=False,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "HEURISTIC" and a.forkable_now is True
    assert "MR=min(MR,M)" in a.target_seam
    assert "accumulator_resident_mtail" in a.target_seam


def test_unrouted_reported_not_dropped():
    # an axis with no route is returned as unrouted, never silently dropped
    d = cca_compare.Divergence(axis="compute.made_up_axis", expert="x", ours="y", backend="rvv")
    actions, unrouted = ac.build_catalog([d])
    assert actions == [] and len(unrouted) == 1
