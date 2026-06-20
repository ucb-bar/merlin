"""R1: fork-scoped default-off compiler features must not perturb the baseline.

The baseline RVV compiler is frozen; an ``impr_`` fork enables named features. With no features
enabled, the emitted pipeline string and schedule are byte-identical to today's, so any fork can
be measured against an unchanged baseline.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower import pipeline as P


def test_empty_features_pipeline_byte_identical():
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base


def test_empty_features_schedule_byte_identical():
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE


def test_feature_changes_only_when_enabled():
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["fused_vfmacc_contraction"]))
    assert on != P.RVV_TRANSFORM_SCHEDULE
    assert "lower_outerproduct" in on   # the vfmacc-forming recipe (outerproduct -> vector.fma)


def test_unknown_feature_rejected():
    with pytest.raises(KeyError):
        F.normalize(["does_not_exist"])


def test_registered_feature_is_typed():
    f = F.get("fused_vfmacc_contraction")
    assert f.action_class in ("PASS", "HEURISTIC", "PATTERN")


def test_activation_schedule_targets_tagged_generics_no_blanket_no_suppress():
    # PRECISE TARGETING (the openvla cos-0.541 fix): the activation schedule must vectorize ONLY the
    # generics the poly rewriter tagged (merlin.act_vectorize) — not blanket-foreach over EVERY
    # linalg.generic — and must NOT use failures(suppress) (which masked a miscompile). It also must
    # carry no hard-coded rank-1 tile (the earlier "too many tiles, expected at most 0" regression).
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE,
                          frozenset(["vectorized_transcendental_activation"]))
    assert "tile_sizes [16]" not in on        # the rank-1 tile that broke real (rank-0/rank-N) models
    assert "vector_sizes [16]" not in on
    # match ONLY the tagged activation generics (precise), not every generic...
    assert 'attributes{"merlin.act_vectorize"}' in on
    # ...and NO failures(suppress) transform op (no masked miscompile). The comment may mention the
    # word; assert the actual op-construct `transform.sequence ... failures(suppress)` is absent.
    assert "transform.sequence" not in on
    assert "failures(suppress) {" not in on
    assert "transform.structured.vectorize %one_eg" in on


def test_packed_feature_packs_and_forms_vfmacc():
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["vfmacc_packed"]))
    assert on != P.RVV_TRANSFORM_SCHEDULE
    assert "transform.structured.pack" in on        # operands packed into contiguous panels
    assert "pack_transpose" in on                   # B pre-transposed (no runtime vector.transpose)
    assert "lower_outerproduct" in on               # outerproduct -> vector.fma -> vfmacc


def test_packed_feature_inserts_eliminate_empty_tensors():
    # The packed pipeline edit must insert eliminate-empty-tensors right before bufferize (so the
    # A-pack and C-pack dests do not CSE-alias onto one buffer); baseline pipeline is untouched.
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    on = P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset(["vfmacc_packed"]))
    assert "eliminate-empty-tensors" not in base
    assert "eliminate-empty-tensors,one-shot-bufferize" in on


def test_intrinsic_microkernel_registered_and_baseline_safe():
    # The scalable-gap winner: a compiler-emitted register-blocked RVV intrinsic micro-kernel
    # (1.7x faster than OpenBLAS, pack-excluded, spill-free; see scalable_gap_result.md). It is a
    # CODEGEN-class marker with no MLIR schedule/pipeline edit, so enabling it must leave the
    # baseline pipeline AND schedule byte-identical.
    f = F.get("intrinsic_microkernel")
    assert f.action_class == "CODEGEN"
    assert f.edit_pipeline is None and f.edit_schedule is None
    feats = frozenset(["intrinsic_microkernel"])
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=feats) == P.build_rvv_pipeline("/tmp/s.mlir")
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, feats) == P.RVV_TRANSFORM_SCHEDULE


def test_accumulator_resident_features_baseline_safe():
    # The accumulator-resident PASS features (+ the N-tail-safe variant) are default-off: enabling
    # NONE leaves the baseline byte-identical, and each known one is a registered PASS.
    base_pipe = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base_pipe
    for nm in ("accumulator_resident_microkernel", "accumulator_resident_ntail"):
        f = F.get(nm)
        assert f.action_class == "PASS"
        # enabling it DOES change the schedule (forms the accumulator-resident recipe)
        on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset([nm]))
        assert on != P.RVV_TRANSFORM_SCHEDULE
        assert "bufferize_to_allocation" in on


def test_ntail_clamps_batch_matmul_nr():
    # The N-tail-safe feature differs from the default accumulator-resident schedule ONLY in the
    # batch_matmul N tile: it clamps NR_bmm to 8 (<= small attention N) so the inner vectorize is
    # full (no masked transfer_write -> no LLVM-23 PipelineError). The matmul NR stays 16.
    default = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_microkernel"]))
    ntail = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_ntail"]))
    assert default != ntail
    # the batch_matmul tile/vectorize in the n-tail schedule uses NR_bmm=8
    assert "[1, 4, 8, 0]" in ntail and "[1, 4, 8, 1]" in ntail
    # but the matmul path keeps NR=16 in both
    assert "[4, 16, 0]" in ntail and "[4, 16, 1]" in ntail


def test_intrinsic_microkernel_labeled_as_ceiling_reference():
    # Honest labeling: intrinsic_microkernel is a CEILING REFERENCE (hand-written driver), NOT a
    # compiler-emitted feature. Its description must say so, and it must remain a no-edit marker.
    f = F.get("intrinsic_microkernel")
    assert f.edit_pipeline is None and f.edit_schedule is None
    assert "CEILING REFERENCE" in f.description and "hand-written" in f.description.lower()


def test_mtail_clamps_matmul_mr():
    # The M-tail-safe feature differs from the default accumulator-resident schedule ONLY in the
    # matmul M tile: it clamps MR_mm to 1 (<= a small leading M, e.g. the M=1 token-decode matmul)
    # so the inner vectorize is full (no masked transfer_write -> no LLVM-23 PipelineError). The
    # batch_matmul MR stays 4. This is the M-side analog of the N-tail clamp.
    default = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_microkernel"]))
    mtail = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_mtail"]))
    assert default != mtail
    # the matmul tile/vectorize in the m-tail schedule uses MR_mm=1
    assert "[1, 16, 0]" in mtail and "[1, 16, 1]" in mtail
    # but the batch_matmul path keeps MR=4 (the m-tail clamp only touches the matmul M tile)
    assert "[1, 4, 16, 0]" in mtail and "[1, 4, 16, 1]" in mtail


def test_wholemodel_composes_mtail_and_ntail_in_one_schedule():
    # The whole-model-safe composed feature carries BOTH clamps inherent in ONE schedule: matmul
    # MR_mm=1 (M-tail) AND batch_matmul NR_bmm=8 (N-tail). This is the inherent-clamp answer to the
    # composition problem (two full-schedule replacements would clobber each other).
    wm = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_wholemodel"]))
    assert wm != P.RVV_TRANSFORM_SCHEDULE
    assert "bufferize_to_allocation" in wm
    # matmul M-tail clamp (MR_mm=1)
    assert "[1, 16, 0]" in wm and "[1, 16, 1]" in wm
    # batch_matmul N-tail clamp (NR_bmm=8)
    assert "[1, 4, 8, 0]" in wm and "[1, 4, 8, 1]" in wm


def test_two_full_schedule_replacements_refuse_to_compose():
    # WORK-ITEM 2: two features that each FULLY REPLACE the transform schedule cannot compose — the
    # last in sorted order would silently clobber the other's clamp. apply_schedule must refuse
    # (raise CompositionError) rather than pick a winner. The whole-model-safe config is ONE feature.
    with pytest.raises(F.CompositionError):
        F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE,
                         frozenset(["fused_vfmacc_tiled", "accumulator_resident_ntail"]))
    with pytest.raises(F.CompositionError):
        F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE,
                         frozenset(["accumulator_resident_mtail", "accumulator_resident_ntail"]))


def test_additive_edit_layers_on_full_replacement():
    # An additive schedule edit (schedule_replace=False, e.g. lmul_widen_n's text substitution) must
    # layer cleanly on top of a single full-replacement feature, not be rejected by the guard.
    assert F.get("lmul_widen_n").schedule_replace is False
    out = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE,
                           frozenset(["fused_vfmacc_tiled", "lmul_widen_n"]))
    assert "lower_outerproduct" in out   # the full replacement applied
    # (lmul_widen_n's substitution targets the baseline [4,8,1] tile, absent in the tiled recipe,
    # so it is a no-op here; the point is the guard does NOT raise on replacement + additive.)


def test_single_full_replacement_still_allowed():
    # The guard must only fire on >1 full replacement; a single one (the common case) is fine.
    for nm in ("fused_vfmacc_tiled", "accumulator_resident_mtail",
               "accumulator_resident_wholemodel"):
        on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset([nm]))
        assert on != P.RVV_TRANSFORM_SCHEDULE


def test_mtail_wholemodel_features_baseline_safe():
    # The new M-tail + composed features are default-off: enabling NONE leaves baseline byte-identical.
    base_pipe = P.build_rvv_pipeline("/tmp/s.mlir")
    base_sched = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset())
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base_pipe
    assert base_sched == P.RVV_TRANSFORM_SCHEDULE
    for nm in ("accumulator_resident_mtail", "accumulator_resident_wholemodel"):
        f = F.get(nm)
        assert f.action_class == "PASS"
        assert f.schedule_replace is True


def test_wholemodel_vf_mr4_register_block_for_a_reuse():
    # ITERATION-3 packing/memory residual feature: the MR=4 register-block variant of the vf kernel
    # for A-operand REUSE (the OpenBLAS lever). It must be registered, default-off-safe, and its
    # schedule must clamp the matmul M tile to MR_mm=4 (4 accumulator rows -> 1 B-load shared across
    # 4 vfmacc.vf) while keeping the batch_matmul N-tail clamp NR_bmm=8. The MR=1 sibling
    # (accumulator_resident_wholemodel_vf) keeps the M=1 tile for the small-M VLA matmuls.
    f = F.get("accumulator_resident_wholemodel_vf_mr4")
    assert f.action_class == "PASS" and f.schedule_replace is True
    # default-off: baseline byte-identical
    base_pipe = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base_pipe
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE
    mr4 = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_wholemodel_vf_mr4"]))
    mr1 = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_wholemodel_vf"]))
    assert mr4 != mr1 and mr4 != P.RVV_TRANSFORM_SCHEDULE
    # MR=4 register block on the matmul path (4 output rows -> 4 accumulators, A-reuse)
    assert "[4, 16, 0]" in mr4 and "[4, 16, 1]" in mr4
    # the MR=1 sibling clamps the matmul M tile to 1 (small-M VLA safe, no A-reuse)
    assert "[1, 16, 0]" in mr1 and "[1, 16, 1]" in mr1
    # both keep the batch_matmul N-tail clamp NR_bmm=8
    assert "[1, 4, 8, 0]" in mr4 and "[1, 4, 8, 0]" in mr1
    # it rides the v3 (vfmacc.vf) runner so the A-scalarization rewrite fires
    assert "accumulator_resident_wholemodel_vf_mr4" in F.ACCUM_RESIDENT_V3_NAMES


def test_memory_traffic_facet_quantifies_loads_per_fma():
    # The iteration-3 memory-traffic decode facet (decode/memory.analyze_memory) must structurally
    # classify K-loop loads (unit-stride vle vs strided vlse vs scalar flw) over the FMA loop and
    # compute loads/useful-FMA — the data-movement metric the CCA vector/compute facets are blind to.
    # Synthetic stream is built straight from RawInsn (no toolchain needed) so the test is hermetic.
    from merlin.kernels.decode.objdump import RawInsn
    from merlin.kernels.decode.rvv import VInsn, VType, InsnStream
    from merlin.kernels.decode.memory import analyze_memory

    vt = VType(sew=32, lmul=4.0, tail="ta", mask="ma")

    def vi(addr, mn, *ops):
        is_vec = mn.startswith("v")
        return VInsn(raw=RawInsn(addr=addr, mnemonic=mn, operands=list(ops)),
                     is_vector=is_vec, vtype=vt if is_vec else None)

    # An MR=1 .vf K-loop (the XNNPACK / wholemodel_vf shape): 1 unit-stride B load (vle32.v) + 1
    # scalar A load (flw) + 1 vfmacc.vf + a back-edge branch to the loop top.
    mr1 = InsnStream(insns=[
        vi(0x100, "vle32.v", "v12", "(a3)"),
        vi(0x104, "flw", "fa5", "0x0(s1)"),
        vi(0x108, "vfmacc.vf", "v8", "fa5", "v12"),
        vi(0x10c, "bne", "s1", "a5", "0x100"),   # back-edge (target < addr)
    ])
    m = analyze_memory(mr1)
    assert m is not None
    assert m.fma_in_loop == 1
    assert m.vec_unit_loads == 1 and m.scalar_loads == 1
    assert m.vec_strided_loads == 0 and m.broadcast_ladder_ops == 0
    assert m.loads_per_fma == 2.0 and m.unit_stride_only is True

    # An MR=4 register block (the vf_mr4 A-reuse shape): ONE B-row load shared across 4 vfmacc.vf
    # (4 A scalars) -> loads/FMA drops to (1 vle + 4 flw)/4 = 1.25.
    mr4 = InsnStream(insns=[
        vi(0x200, "vle32.v", "v12", "(a3)"),
        vi(0x204, "flw", "fa0", "0x0(s1)"),
        vi(0x208, "flw", "fa1", "0x4(s1)"),
        vi(0x20c, "flw", "fa2", "0x8(s1)"),
        vi(0x210, "flw", "fa3", "0xc(s1)"),
        vi(0x214, "vfmacc.vf", "v8", "fa0", "v12"),
        vi(0x218, "vfmacc.vf", "v12", "fa1", "v12"),
        vi(0x21c, "vfmacc.vf", "v16", "fa2", "v12"),
        vi(0x220, "vfmacc.vf", "v20", "fa3", "v12"),
        vi(0x224, "bne", "s1", "a5", "0x200"),
    ])
    m4 = analyze_memory(mr4)
    assert m4.fma_in_loop == 4
    assert m4.vec_unit_loads == 1 and m4.scalar_loads == 4
    assert m4.loads_per_fma == 1.25   # the A-reuse win: 1 B-load amortized over MR=4 FMAs

    # A strided / vfmacc.vv broadcast-ladder loop (the pre-iteration-2 .vv shape) must be flagged:
    # vlse32.v (strided) + a vslideup/vmv broadcast ladder, NOT unit_stride_only.
    vv = InsnStream(insns=[
        vi(0x300, "vlse32.v", "v12", "(a3)", "a1"),
        vi(0x304, "vslideup.vi", "v4", "v2", "0x1"),
        vi(0x308, "vmv1r.v", "v6", "v4"),
        vi(0x30c, "vfmacc.vv", "v8", "v6", "v12"),
        vi(0x310, "bne", "s1", "a5", "0x300"),
    ])
    mv = analyze_memory(vv)
    assert mv.vec_strided_loads == 1 and mv.unit_stride_only is False
    assert mv.broadcast_ladder_ops == 2 and mv.a_broadcast_per_fma == 2.0


def test_activation_feature_baseline_safe_and_typed():
    # vectorized_transcendental_activation is default-off: enabling NONE leaves the pipeline AND
    # schedule byte-identical; it is a registered PASS with both an edit_pipeline and edit_schedule.
    base_pipe = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base_pipe
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE
    f = F.get("vectorized_transcendental_activation")
    assert f.action_class == "PASS"
    # the feature is a SCHEDULE edit only; it adds NO pipeline-pass edit (the pure-arith poly needs no
    # convert-math-to-llvm, and adding one converted the softmax exp to llvm.intr.exp -> spike crash).
    assert f.edit_pipeline is None and f.edit_schedule is not None


def test_activation_feature_adds_no_pipeline_edit_softmax_keeps_libm():
    # The activation feature must NOT edit the pass list. The earlier version spliced
    # convert-math-to-llvm before convert-math-to-libm (to vector-lower the poly's math.fma/absf/
    # roundeven); that same pass converted the un-rewritten SOFTMAX math.exp to llvm.intr.exp
    # (llvm.exp.f32), which the freestanding spike/RVV runtime cannot legalize -> the openvla
    # whole-model 'bad syscall' CRASH. The poly is now pure arith, so no math-to-llvm pass is needed
    # and the softmax exp keeps the baseline convert-math-to-libm -> scalar expf path. Assert the
    # pipeline is byte-identical to the baseline when the feature is on (only the schedule changes).
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    on = P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset(["vectorized_transcendental_activation"]))
    assert on == base
    assert "convert-math-to-llvm,convert-math-to-libm" not in on


def test_activation_feature_vectorizes_elementwise_generic_in_schedule():
    # The schedule edit must add an elementwise linalg.generic tile+vectorize on top of the baseline
    # matmul/batch_matmul vectorization (so the activation generic — carrying the rewritten polynomial
    # — vectorizes to vector ops instead of falling through convert-linalg-to-loops to a scalar loop).
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["vectorized_transcendental_activation"]))
    assert on != P.RVV_TRANSFORM_SCHEDULE
    assert 'ops{["linalg.generic"]}' in on        # the activation generic is matched + vectorized
    assert 'ops{["linalg.matmul"]}' in on          # baseline matmul vectorization preserved
    assert "tile_using_for" in on and "vectorize" in on


def test_activation_runner_embeds_polynomial_rewriter():
    # The lowering runner used when the feature is on must embed the math.exp/erf/tanh -> arith
    # polynomial rewriter (act_poly). The plain baseline runner must NOT (byte-identical lowering).
    runner = P._activation_poly_runner()
    assert "apply_activation_polynomial" in runner
    assert "math.exp" in runner and "math.erf" in runner and "math.tanh" in runner
    assert "apply_activation_polynomial" not in P._RUNNER   # baseline runner unchanged


def test_baseline_package_has_no_features():
    # The immutable baseline must carry zero compiler_features -> byte-identical lowering.
    from pathlib import Path

    from merlin.rvvgen.registry import load_rvv_package
    base = Path(__file__).resolve().parents[3] / "generated_targets" / "rvv" / "hand_v0"
    if not base.is_dir():
        pytest.skip("hand_v0 package not present")
    assert load_rvv_package(base).compiler_features == []
