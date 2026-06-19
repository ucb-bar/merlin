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


def test_activation_feature_baseline_safe_and_typed():
    # vectorized_transcendental_activation is default-off: enabling NONE leaves the pipeline AND
    # schedule byte-identical; it is a registered PASS with both an edit_pipeline and edit_schedule.
    base_pipe = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base_pipe
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE
    f = F.get("vectorized_transcendental_activation")
    assert f.action_class == "PASS"
    assert f.edit_pipeline is not None and f.edit_schedule is not None


def test_activation_feature_inserts_math_to_llvm_before_libm():
    # When enabled, the pipeline edit must splice convert-math-to-llvm IMMEDIATELY before
    # convert-math-to-libm so the polynomial's vector math.absf/roundeven/fma lower as vector LLVM
    # intrinsics (not scalarized lane-by-lane by libm). Baseline pipeline does NOT have that order.
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    on = P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset(["vectorized_transcendental_activation"]))
    assert "convert-math-to-llvm,convert-math-to-libm" not in base
    assert "convert-math-to-llvm,convert-math-to-libm" in on


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
