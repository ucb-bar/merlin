"""R1: fork-scoped default-off compiler features must not perturb the baseline.

The baseline RVV compiler is frozen; an ``impr_`` fork enables named features. With no features
enabled, the emitted pipeline string and schedule are byte-identical to today's, so any fork can
be measured against an unchanged baseline.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

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

    from merlin.mining.registry import load_rvv_package
    base = repo_root() / "out/artifacts/targets" / "rvv" / "hand_v0"
    if not base.is_dir():
        pytest.skip("hand_v0 package not present")
    assert load_rvv_package(base).compiler_features == []


# ---------------------------------------------------------------------------------------
# ImprFeature.implies -- lowering hygiene a recipe cannot be MEASURED without.
#
# A default-off feature whose payoff is cancelled by a SEPARATE default-off fix is an inert
# lever in practice: the beam has to discover the conjunction, and anyone naming the feature
# directly in `compiler_features` gets the cancelled version. Measured on spike at 64^3,
# bit-identical output on every arm: bare MR=4 is 2.0-2.1x SLOWER than MR=1 (the per-tile
# @memrefCopy costs 187,520 instructions), and 1.27-1.35x FASTER with the erase.
# ---------------------------------------------------------------------------------------

def test_normalize_is_closed_under_implies_and_leaves_the_baseline_empty():
    from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY
    mr4 = "accumulator_resident_wholemodel_vf_mr4"
    assert F.get(mr4).implies == frozenset({SELF_COPY})
    assert F.normalize([mr4]) == frozenset({mr4, SELF_COPY})
    # the frozen baseline must not acquire anything
    assert F.normalize(frozenset()) == frozenset()
    assert F.normalize(None) == frozenset()
    # a feature with no implications is returned unchanged (no accidental growth)
    assert F.normalize([SELF_COPY]) == frozenset({SELF_COPY})


def test_only_the_mr_gt_1_register_blocks_imply_the_tile_epilogue_hygiene():
    """The rule is keyed on the recipe's MATMUL register block, not on a name list.

    MR=1 is excluded not because the erase would be unsafe but because MR=1+erase measured
    BYTE-IDENTICAL (both dtypes, same cycle count to the digit) -- there is no self-copy at MR=1, so
    implying it would move the validated MR_mm=1 control's declared feature set for no effect.
    """
    from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY
    assert F.get("accumulator_resident_wholemodel_vf").implies == frozenset()      # MR_mm=1 control
    for mr_gt_1 in ("accumulator_resident_wholemodel_vf_mr4",
                    "accumulator_resident_wholemodel_vf_mrpad"):
        assert F.get(mr_gt_1).implies == frozenset({SELF_COPY}), mr_gt_1
    # ...and on the on-demand v3 grid the rule follows MR, both ways
    assert F.get(F.ensure_v3_microkernel(4, 16, 16)).implies == frozenset({SELF_COPY})
    assert F.get(F.ensure_v3_microkernel(1, 16, 16)).implies == frozenset()
    # per-op tables key on the matmul MR specifically
    assert F.get(F.ensure_v3_perop_microkernel(4, 16, 4, 8, 16)).implies == frozenset({SELF_COPY})
    assert F.get(F.ensure_v3_perop_microkernel(1, 16, 4, 8, 16)).implies == frozenset()


def test_implied_hygiene_reaches_the_runner_argv_gate():
    """The implication is worthless unless the argv[4] gate sees it -- that gate is what actually
    splits the pass pipeline and erases the copy. `normalize` is the single choke point every
    consumer reads, so closing there is what keeps the gate, the runner selection and the schedule
    edits from disagreeing about which features are on."""
    from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY, needs_canonicalize
    feats = F.normalize(["accumulator_resident_wholemodel_vf_mr4"])
    assert SELF_COPY in feats
    # the erase depends on a canonicalize+cse after bufferization; the pipeline must be able to add it
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    assert needs_canonicalize(base) or "canonicalize" in base


def test_the_broadcast_pricing_and_its_decision_are_recorded_not_re_derivable():
    """`linalg.broadcast` surfaced as 8.95% of the profiled accelerator device leg -- third behind
    generic and transpose -- and looks like a candidate for the non-contraction vectorize lever. It is
    not: 3.3 MiB read vs 171.1 MiB WRITTEN across its 567 ops (51.9x amplification, zero arithmetic),
    so wider stores cannot reduce the bytes stored. That is the same reason
    `vectorize_non_contraction_generics` measured 1.28x SLOWER at 4.9x more vector instructions. The
    finding and the decision must live next to that lever, or the next reader reaches for it."""
    import inspect

    src = inspect.getsource(F)
    assert "PRICED AND DECLINED" in src
    assert "51.9x amplification" in src
    assert "171.1 MiB" in src and "3.3 MiB read" in src
    assert "1.00000" in src                       # the profiler coverage the pricing rests on
    # ...and it must say what the real fix is, and that it is out of scope here
    assert "broadcasting indexing map" in src
    assert "out of scope" in src
    # the negative result it would otherwise be confused with is still recorded
    assert "4.9x more vector instructions" in src and "1.28x SLOWER" in src


def test_the_implies_closure_terminates_and_validates_what_it_adds():
    """`normalize` closes the feature set under `implies`, so it has to be a real fixpoint loop and not
    a single pass. Pinned because the next person to declare an implication will rely on all four:
    a chain closes fully, a cycle terminates instead of spinning, precedence is not accidental
    (`fs | X if fs else fs` must parse as `(fs | X) if fs else fs`), and an implied name that is not
    registered RAISES rather than being silently added to the set the whole pipeline then keys off.
    """
    probes = ("_t_cycle_a", "_t_cycle_b", "_t_chain_1", "_t_chain_2", "_t_bad")
    try:
        F.register(F.ImprFeature(name="_t_cycle_a", action_class="PASS", description="t",
                                 implies=frozenset({"_t_cycle_b"})))
        F.register(F.ImprFeature(name="_t_cycle_b", action_class="PASS", description="t",
                                 implies=frozenset({"_t_cycle_a"})))
        F.register(F.ImprFeature(name="_t_chain_1", action_class="PASS", description="t",
                                 implies=frozenset({"_t_chain_2"})))
        F.register(F.ImprFeature(name="_t_chain_2", action_class="PASS", description="t",
                                 implies=frozenset({"erase_self_copy"})))
        F.register(F.ImprFeature(name="_t_bad", action_class="PASS", description="t",
                                 implies=frozenset({"_t_does_not_exist"})))

        assert F.normalize(["_t_cycle_a"]) == frozenset({"_t_cycle_a", "_t_cycle_b"})
        assert F.normalize(["_t_chain_1"]) == frozenset(
            {"_t_chain_1", "_t_chain_2", "erase_self_copy"})
        with pytest.raises(KeyError, match="_t_does_not_exist"):
            F.normalize(["_t_bad"])
        # precedence / empty in -> empty out, and a feature with no implications is unchanged
        assert F.normalize(frozenset()) == frozenset()
        assert F.normalize(["erase_self_copy"]) == frozenset({"erase_self_copy"})
    finally:
        for n in probes:
            F._REGISTRY.pop(n, None)


def test_every_mr_gt_1_recipe_implies_the_tile_epilogue_hygiene():
    """Enumerated by CONSTRUCTION, not by hand — because hand-enumeration already failed once.

    The `implies` mechanism was added to eight v3 registration sites and MISSED `ensure_perop_block`,
    which is the whole-model per-op path: the one the board builds by default and the one every
    measurement in this line of work went through. Cost, measured on small_llama int8 with a spike PC
    histogram over the linked ELF: `memrefCopy` was 28.15% of all retired instructions -- more than
    `forward` itself at 27.09% -- and removing it took total instructions 5,574,201 -> 3,523,665
    (1.58x), with `forward` unchanged.

    So this test does not check a list of names. It EXERCISES each on-demand registrar at MR>1 and at
    MR=1 and asserts the rule, so a ninth registrar added later fails here rather than silently
    shipping the defect a tenth time.
    """
    from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY

    mr_gt_1 = {
        "ensure_v3_microkernel":            lambda: F.ensure_v3_microkernel(4, 16, 16),
        "ensure_v3_scalable_microkernel":   lambda: F.ensure_v3_scalable_microkernel(4, 16, 16),
        "ensure_v3_kblocked_microkernel":   lambda: F.ensure_v3_kblocked_microkernel(4, 16, 16),
        "ensure_v3_unrolled_microkernel":   lambda: F.ensure_v3_unrolled_microkernel(4, 16, 16),
        "ensure_v3_perop_microkernel":      lambda: F.ensure_v3_perop_microkernel(4, 16, 4, 8, 16),
        "ensure_perop_block":               lambda: F.ensure_perop_block(
            {"linalg.matmul:128x128:128": (4, 16)}, 16),
    }
    mr_eq_1 = {
        "ensure_v3_microkernel":            lambda: F.ensure_v3_microkernel(1, 16, 16),
        "ensure_v3_scalable_microkernel":   lambda: F.ensure_v3_scalable_microkernel(1, 16, 16),
        "ensure_v3_kblocked_microkernel":   lambda: F.ensure_v3_kblocked_microkernel(1, 16, 16),
        "ensure_v3_unrolled_microkernel":   lambda: F.ensure_v3_unrolled_microkernel(1, 16, 16),
        "ensure_v3_perop_microkernel":      lambda: F.ensure_v3_perop_microkernel(1, 16, 4, 8, 16),
        "ensure_perop_block":               lambda: F.ensure_perop_block(
            {"linalg.matmul:17x128:128": (1, 16)}, 16),
    }

    for who, make in mr_gt_1.items():
        feat = F.get(make())
        assert SELF_COPY in feat.implies, (
            f"{who} registers an MR>1 recipe that does NOT imply {SELF_COPY!r}: it will emit a per-tile "
            f"memref.copy %x, %x and pay for it (measured at 28% of retired instructions on the path "
            f"that missed this)")
    for who, make in mr_eq_1.items():
        feat = F.get(make())
        assert SELF_COPY not in feat.implies, (
            f"{who} implies the erase at MR=1, where MR=1+erase measured byte-identical — that moves a "
            f"validated control for no effect")


def test_the_statically_registered_mr_gt_1_features_imply_it_too():
    """The named whole-model points, which are not reached through a registrar."""
    from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY

    for name in ("accumulator_resident_wholemodel_vf_mr4",
                 "accumulator_resident_wholemodel_vf_mrpad",
                 "accumulator_resident_microkernel_v3"):
        assert SELF_COPY in F.get(name).implies, name
    # ...and the MR_mm=1 control still does not
    assert SELF_COPY not in F.get("accumulator_resident_wholemodel_vf").implies


def test_stack_promotion_lands_after_hoisting_and_leaves_the_baseline_alone():
    """Order matters: promoting AFTER hoisting means a buffer already lifted out of a loop is
    promoted once rather than per iteration, and deallocation runs earlier in this pipeline (a stack
    buffer needs none), so inserting after it is what keeps the two consistent."""
    from merlin.llvmlower import impr_features as F, pipeline as P
    base = list(P._UPSTREAM_PASSES)
    out = F._REGISTRY[F.PROMOTE_STACK_NAME].edit_pipeline(base)
    i = out.index("func.func(buffer-hoisting,buffer-loop-hoisting)")
    assert out[i + 1].startswith("func.func(promote-buffers-to-stack{")
    assert base == list(P._UPSTREAM_PASSES), "the baseline pipeline was mutated in place"
    assert len(out) == len(base) + 1


def test_stack_promotion_refuses_to_guess_when_its_anchor_is_gone():
    """Fail closed. If the pipeline is reshaped, inserting this pass at the wrong point silently
    changes which buffers are promoted -- and a promotion before deallocation would leave a stack
    buffer with a dealloc against it."""
    import pytest
    from merlin.llvmlower import impr_features as F
    with pytest.raises(ValueError, match="anchor"):
        F._REGISTRY[F.PROMOTE_STACK_NAME].edit_pipeline(["canonicalize", "cse"])


def test_the_promotion_cap_is_env_overridable_and_fails_closed():
    """The cap is a stack-overflow guard: a promoted buffer lives for the whole frame, so an
    uncapped promotion of a large intermediate overruns it. A bad env value must not silently
    disable or unbound the guard."""
    import os
    from merlin.llvmlower import impr_features as F
    old = os.environ.get(F.PROMOTE_STACK_BYTES_ENV)
    try:
        for bad in ("0", "-1", "nonsense", ""):
            os.environ[F.PROMOTE_STACK_BYTES_ENV] = bad
            assert F._promote_stack_bytes() == F._PROMOTE_STACK_DEFAULT_BYTES, bad
        os.environ[F.PROMOTE_STACK_BYTES_ENV] = "262144"
        assert F._promote_stack_bytes() == 262144
    finally:
        if old is None:
            os.environ.pop(F.PROMOTE_STACK_BYTES_ENV, None)
        else:
            os.environ[F.PROMOTE_STACK_BYTES_ENV] = old


def test_a_sized_promotion_variant_names_its_cap_in_the_emitted_pass():
    """The cap must reach the emitted pass, because that is the whole point of the sized variant.

    The cap was reachable ONLY through MERLIN_PROMOTE_STACK_BYTES, and no fork can vary an
    environment variable -- so the beam always built at the 16 KB default (measured 1.03x) and ranked
    the lever below levers it should beat, while the same feature at 256 KB is 1.34x on that model.
    """
    from merlin.llvmlower import impr_features as F, pipeline as pl
    base = list(pl._UPSTREAM_PASSES)

    sized = F.apply_pipeline(base, frozenset({F.ensure_promote_stack(262144)}))
    assert [p for p in sized if "promote-buffers" in p] == [
        "func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=262144})"]

    # the variant AT the default value must emit exactly what the unsized feature emits -- if those
    # two ever diverge, the ladder's bottom rung stops being a control arm for the plain feature.
    at_default = F.apply_pipeline(base, frozenset({F.ensure_promote_stack(
        F._PROMOTE_STACK_DEFAULT_BYTES)}))
    plain = F.apply_pipeline(base, frozenset({F.PROMOTE_STACK_NAME}))
    assert ([p for p in at_default if "promote-buffers" in p]
            == [p for p in plain if "promote-buffers" in p])


def test_sized_variants_leave_the_frozen_baseline_byte_identical():
    """Registering on-demand variants must not perturb the empty-feature pipeline."""
    from merlin.llvmlower import impr_features as F, pipeline as pl
    base = list(pl._UPSTREAM_PASSES)
    F.ensure_promote_stack(1048576)
    F.perop_mr_sentinel(16)
    assert F.apply_pipeline(base, frozenset()) == base


def test_a_nonpositive_cap_is_refused_rather_than_silently_defaulted():
    import pytest
    from merlin.llvmlower import impr_features as F
    for bad in (0, -1):
        with pytest.raises(ValueError):
            F.ensure_promote_stack(bad)
        with pytest.raises(ValueError):
            F.perop_mr_sentinel(bad)


def test_the_mr_sentinel_parser_does_not_swallow_a_resolved_block_feature():
    """`ensure_perop_block` mints names in the SAME `perop_register_block_*` namespace, so a
    prefix-only test would read a RESOLVED feature as an unresolved sentinel and re-derive the block
    table underneath it. The suffix must be all digits."""
    from merlin.llvmlower import impr_features as F
    assert F.parse_perop_mr_sentinel(F.perop_mr_sentinel(8)) == 8
    assert F.parse_perop_mr_sentinel("perop_register_block") is None
    assert F.parse_perop_mr_sentinel("perop_register_block_3b_128_deadbeef12") is None
    assert F.parse_perop_mr_sentinel("perop_register_block_mr") is None
    assert F.parse_perop_mr_sentinel("perop_register_block_mrx") is None


def test_an_unresolved_mr_sentinel_raises_instead_of_lowering_untagged():
    """Same contract as the plain sentinel: reaching lowering means nothing tagged the IR, every
    contraction falls to convert-linalg-to-loops, and the build still reports success with correct
    numbers -- the measured deepjscc '2.56x regression that is actually an untagged build'."""
    import pytest
    from merlin.llvmlower import impr_features as F
    feat = F._REGISTRY[F.perop_mr_sentinel(8)]
    with pytest.raises(RuntimeError, match="unresolved"):
        feat.edit_pipeline(["canonicalize"])


def test_the_default_mrpad_schedule_is_unchanged():
    """Frozen-baseline invariant: parameterizing the pad literals must not move the f32 default."""
    from merlin.llvmlower.impr_features import _accumulator_resident_v3_mrpad_pre_schedule as sched
    txt = sched(4, 16, 16, NR_bmm=8)
    assert "padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]" in txt


def test_a_pad_value_carries_the_type_of_the_operand_it_pads():
    """transform.structured.pad HARD-ERRORS on a wrongly-typed pad value:

        'transform.structured.pad' op expects a padding value of type 'i8', got 0.000000e+00 : f32

    which the lowering catches as a PipelineError and turns into a whole-model SCALAR FALLBACK, so
    the lever presents as 'applied but slow' rather than 'rejected'. Verified against mlir-opt.
    """
    from merlin.llvmlower.impr_features import (ensure_mrpad_for_elem_types, get,
                                                _accumulator_resident_v3_mrpad_pre_schedule as sched)
    txt = sched(4, 16, 16, NR_bmm=8, elem_types=("i8", "i8", "i32"))
    assert "padding_values = [0 : i8, 0 : i8, 0 : i32]" in txt
    name = ensure_mrpad_for_elem_types("i8", "i8", "i32")
    assert name == "accumulator_resident_wholemodel_vf_mrpad_i8_i8_i32"
    assert "padding_values = [0 : i8, 0 : i8, 0 : i32]" in get(name).edit_schedule("")
    # idempotent registration
    assert ensure_mrpad_for_elem_types("i8", "i8", "i32") == name


def test_a_zero_literal_is_derived_and_refuses_what_it_cannot_derive():
    """Deriving beats tabulating: a table goes stale on the first new element type, and the
    consequence of a wrong literal here is silent (scalar fallback), not loud."""
    import pytest
    from merlin.llvmlower.impr_features import _zero_attr, ensure_mrpad_for_elem_types
    assert _zero_attr("i8") == "0 : i8"
    assert _zero_attr("i32") == "0 : i32"
    assert _zero_attr("f32") == "0.000000e+00 : f32"
    assert _zero_attr("f16") == "0.000000e+00 : f16"
    for bad in ("weird", "tensor<4xi8>", ""):
        with pytest.raises(ValueError):
            _zero_attr(bad)
    # and the refusal happens at REGISTRATION, not inside the transform interpreter
    with pytest.raises(ValueError):
        ensure_mrpad_for_elem_types("i8", "i8", "nonsense")
