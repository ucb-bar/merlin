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


def test_baseline_package_has_no_features():
    # The immutable baseline must carry zero compiler_features -> byte-identical lowering.
    from pathlib import Path

    from merlin.rvvgen.registry import load_rvv_package
    base = Path(__file__).resolve().parents[3] / "generated_targets" / "rvv" / "hand_v0"
    if not base.is_dir():
        pytest.skip("hand_v0 package not present")
    assert load_rvv_package(base).compiler_features == []
