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


def test_dtype_axes_route_to_dtype_strategy_knob():
    # WS-C: the accumulate-width and element-width axes close via the existing dtype_strategy knob
    # (the datapath lever), a distinct axis from `widening`. Both are forkable KNOB registrations.
    for axis, expert, ours in (("compute.accumulator_dtype", "i32", "f32"), ("vector.sew", 8, 32)):
        d = cca_compare.Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=["x"])
        a = ac.route(d)
        assert a is not None, axis
        assert a.action_class == "KNOB" and a.forkable_now, axis
        assert "dtype_strategy" in a.target_seam, axis


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


# --- intended-vs-achieved + escalation ladder (the closed-loop methodology fix) ---

def test_routes_carry_machine_readable_intended_facet():
    # register_block: the promise is "reach the EXPERT's MR" (derived from the divergence, not a const)
    d = cca_compare.Divergence(axis="compute.register_block", expert=(7, ("vsetvlmax", 4)),
                               ours=None, backend="rvv")
    a = ac.route(d)
    assert a.intended_facet == {"compute.register_block": 7}
    # accumulator_resident: the cheapest matching class is PASS, promising resident=True
    d2 = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                                backend="rvv")
    a2 = ac.route(d2)
    assert a2.action_class == "PASS" and a2.intended_facet == {"compute.accumulator_resident": True}


def test_achieved_residual_detects_unmet_promise():
    a = ac.route(cca_compare.Divergence(axis="compute.accumulator_resident", expert=True,
                                        ours=False, backend="rvv"))
    bad = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(accumulator_resident=False))
    ok = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(accumulator_resident=True))
    assert ac.achieved_residual(a, bad) == ["compute.accumulator_resident"]
    assert ac.achieved_residual(a, ok) == []
    # register_block: emitted MR must be >= promised MR
    arb = ac.route(cca_compare.Divergence(axis="compute.register_block", expert=(7, None),
                                          ours=None, backend="rvv"))
    mr4 = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(register_block=(4, None)))
    mr7 = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(register_block=(7, None)))
    assert ac.achieved_residual(arb, mr4) == ["compute.register_block"]
    assert ac.achieved_residual(arb, mr7) == []


def test_escalation_ladder_walks_up_classes():
    # PASS was insufficient for accumulator_resident -> escalate to the CODEGEN microkernel route
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                               backend="rvv")
    esc = ac.route_escalated(d, prior_class="PASS")
    assert esc is not None and esc.action_class == "CODEGEN"
    # CODEGEN is the top of the ladder -> nothing stronger
    assert ac.route_escalated(d, prior_class="CODEGEN") is None
