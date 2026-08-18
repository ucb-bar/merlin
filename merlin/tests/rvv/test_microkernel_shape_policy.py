"""The SHAPE-AWARE micro-kernel selection: pick a register block that masks no parallel dim.

The predicate and the ranking these tests pin are MEASURED, not guessed — 57 synthetic
``linalg.matmul`` lowerings through ``llvmlower.lower.lower_model_file`` on the int8 (W8A8)
pipeline (MR in {1,2,4,8} x NR in {8,16,32} x M in {2,6,8,17} x N in {8,24,40,128,344}) plus 10
whole-model lowerings (small_llama / openvla / bitvla / rdt2 / smolvla, pinned block vs derived).
The cells reproduced below are the ones that DISCRIMINATE between candidate rules; the tests are
pure-Python (no MLIR, no board), so they stay a fast regression on the policy, not on the toolchain.
"""
from __future__ import annotations

import pytest

from merlin.kernels.microkernel import (ContractionShape, MicrokernelSpec, VL_DYNAMIC,
                                        largest_divisor_at_most, masked_parallel_dims,
                                        resolve, resolve_for_shapes)
from merlin.kernels.shapes import contraction_shapes
from merlin.rvvgen.from_strategy import _rvv_best_block, _rvv_blocking_lowers


# (MR, NR, M, N, lowers) — measured cells of the int8 lowering grid.
MEASURED_CELLS = [
    # the champion block: fine when both parallel dims divide ...
    (4, 16, 8, 128, True),
    (4, 16, 128, 128, True),
    (4, 16, 20, 192, True),
    # ... masking N breaks it (this is the half the package manifest misses: M=8 divides MR=4)
    (4, 16, 8, 344, False),
    (4, 16, 8, 24, False),
    # ... and masking M breaks it
    (4, 16, 17, 128, False),
    (4, 16, 2, 128, False),
    (4, 16, 6, 128, False),
    # narrowing NR to a divisor of N rescues the N case but not the M case
    (4, 8, 8, 344, True),
    (4, 8, 17, 128, False),
    # MR=1 is the rank-1 escape: a masked N tail DOES lower ...
    (1, 16, 8, 344, True),
    (1, 16, 8, 24, True),
    (1, 16, 17, 128, True),
    (1, 32, 8, 40, True),
    # ... until the tile EXCEEDS the extent, where it fails again
    (1, 16, 8, 8, False),
    (1, 32, 8, 24, False),
    # MR>1 has no such escape at any MR
    (2, 16, 8, 344, False),
    (8, 16, 8, 344, False),
    (2, 8, 17, 128, False),
    (8, 8, 17, 344, False),
    # a fully dividing narrow block is always fine
    (1, 8, 8, 8, True),
    (8, 8, 8, 344, True),
]


@pytest.mark.parametrize("MR,NR,M,N,expected", MEASURED_CELLS)
def test_predicate_matches_measured_lowering(MR, NR, M, N, expected):
    assert _rvv_blocking_lowers(MR, NR, M, N) is expected


def test_reduction_extent_is_not_a_hazard():
    """Masking K is harmless — M=8, N=128, K=344 lowers at the champion block (measured)."""
    assert _rvv_blocking_lowers(4, 16, 8, 128)          # K plays no part in the predicate
    assert masked_parallel_dims((4, 16), (8, 128)) == ()
    assert masked_parallel_dims((4, 16), (8, 344)) == (1,)
    assert masked_parallel_dims((4, 16), (17, 128)) == (0,)
    assert masked_parallel_dims((0, 16), (17, 344)) == (1,)   # tile 0 = "do not tile" = no mask


@pytest.mark.parametrize("n,cap,expected", [(8, 16, 8), (344, 16, 8), (64, 16, 16),
                                            (17, 4, 1), (1, 16, 1), (32, 4, 4)])
def test_largest_divisor_at_most(n, cap, expected):
    assert largest_divisor_at_most(n, cap) == expected


def test_best_block_keeps_the_requested_block_when_every_extent_divides():
    """bitvla's shapes (M=32, N in {128..1024}) leave the champion block untouched."""
    ext = [(32, 128), (32, 256), (32, 512), (32, 1024)]
    assert _rvv_best_block(4, 16, ext) == (4, 16)


def test_best_block_narrows_n_when_only_n_is_indivisible():
    """small_llama: gcd(N) = 8 over {128, 344, 256} — derived, not a shape literal."""
    ext = [(8, 128), (8, 344), (8, 256)]
    assert _rvv_best_block(4, 16, ext) == (4, 8)


def test_best_block_falls_to_the_rank1_escape_when_m_is_indivisible():
    """openvla: gcd(M) = 1 over {20, 17, 16} forces MR=1, and NR=16 survives (all N >= 16).

    This DERIVES the clamp `accumulator_resident_wholemodel_vf` pins by hand (MR_mm=1)."""
    ext = [(20, 512), (17, 576), (16, 2304), (20, 128)]
    assert _rvv_best_block(4, 16, ext) == (1, 16)


def test_best_block_respects_the_tile_le_extent_bound_of_the_escape():
    """A dim smaller than the block cannot use the rank-1 escape; the block must divide it."""
    assert _rvv_best_block(4, 16, [(8, 8), (8, 32)]) == (4, 8)


def test_best_block_never_returns_an_illegal_block():
    """Exhaustive over a small extent lattice: whatever is chosen must satisfy the predicate."""
    for m1 in (1, 2, 8, 17, 32):
        for n1 in (8, 17, 24, 128, 344):
            for m2 in (8, 20):
                for n2 in (16, 192):
                    ext = [(m1, n1), (m2, n2)]
                    mr, nr = _rvv_best_block(4, 16, ext)
                    assert all(_rvv_blocking_lowers(mr, nr, m, n) for m, n in ext), (ext, mr, nr)


def test_shape_blind_resolution_is_unchanged_without_shapes():
    """No observed shapes -> byte-identical to the shape-blind resolver (the default path)."""
    spec = MicrokernelSpec(MR=4, NR=16, KC=16)
    assert resolve_for_shapes("rvv", spec, ()) == resolve("rvv", spec)


def test_policy_is_a_noop_when_the_pinned_block_already_fits():
    spec = MicrokernelSpec(MR=4, NR=16, KC=16)
    shapes = [ContractionShape("linalg.matmul", (32, 256), (256,)),
              ContractionShape("linalg.batch_matmul", (8, 32, 32), (32,))]
    assert resolve_for_shapes("rvv", spec, shapes) == resolve("rvv", spec)


def test_policy_solves_each_op_class_independently():
    """small_llama's matmuls want (4, 8); a batch_matmul with N=32 could keep 16 — each op class
    carries its own tile factors in the emitted schedule, so they are solved separately."""
    spec = MicrokernelSpec(MR=4, NR=16, KC=16)
    shapes = [ContractionShape("linalg.matmul", (8, 344), (128,)),
              ContractionShape("linalg.batch_matmul", (4, 32, 32), (32,))]
    feats = resolve_for_shapes("rvv", spec, shapes)
    assert feats[0] == "accum_resident_v3p_4_8_4_16_16"


def test_policy_defers_to_the_blind_recipe_for_axes_it_does_not_author():
    """vl_strategy / pack / k_block each REPLACE the schedule and have no per-op-class arm —
    an unrealized axis must stay an honest divergence, never a silent fixed-width substitution."""
    spec = MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_DYNAMIC)
    shapes = [ContractionShape("linalg.matmul", (8, 344), (128,))]
    assert resolve_for_shapes("rvv", spec, shapes) == resolve("rvv", spec)


def test_contraction_shape_rejects_degenerate_extents():
    with pytest.raises(ValueError):
        ContractionShape("linalg.matmul", (8, 0), (128,))


def test_shape_observer_reads_named_and_generic_contractions():
    """The int8 rewrite leaves contraction GENERICS behind; the observer must see both forms.

    (The RVV pipeline recovers the named op with `linalg-specialize-generic-ops` just before the
    transform interpreter, so an observer that only matched named ops would report zero
    contractions for every int8 whole model — exactly the case the policy exists for.)"""
    named = """
    builtin.module {
      func.func @f(%a: tensor<8x128xf32>, %b: tensor<128x344xf32>,
                   %c: tensor<8x344xf32>) -> tensor<8x344xf32> {
        %0 = linalg.matmul ins(%a, %b : tensor<8x128xf32>, tensor<128x344xf32>)
             outs(%c : tensor<8x344xf32>) -> tensor<8x344xf32>
        return %0 : tensor<8x344xf32>
      }
    }
    """
    got = contraction_shapes(named)
    # The observer also keeps the element types it reads, positionally (lhs, rhs, out): they are the
    # other half of a legality question, and a policy that only saw extents could not tell an int8
    # contraction from an f32 one of the same size.
    assert got == [ContractionShape("linalg.matmul", (8, 344), (128,), ("f32", "f32", "f32"))]

    generic = """
    builtin.module {
      func.func @f(%a: tensor<8x128xi8>, %b: tensor<128x344xi8>,
                   %c: tensor<8x344xi32>) -> tensor<8x344xi32> {
        %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d2, d1)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"]}
             ins(%a, %b : tensor<8x128xi8>, tensor<128x344xi8>)
             outs(%c : tensor<8x344xi32>) {
        ^bb0(%x: i8, %y: i8, %acc: i32):
          %xe = arith.extsi %x : i8 to i32
          %ye = arith.extsi %y : i8 to i32
          %m = arith.muli %xe, %ye : i32
          %s = arith.addi %acc, %m : i32
          linalg.yield %s : i32
        } -> tensor<8x344xi32>
        return %0 : tensor<8x344xi32>
      }
    }
    """
    assert contraction_shapes(generic) == [
        ContractionShape("linalg.matmul", (8, 344), (128,), ("i8", "i8", "i32"))]


def test_shape_observer_degrades_to_empty_on_unreadable_input():
    """A failed observation must mean "I observed nothing" (-> shape-blind fallback), never a raise."""
    assert contraction_shapes("this is not mlir {{{") == []


# ---------------------------------------------------------------------------
# Frozen-block adaptation. A package's register block is a claim about extents, so a workload with
# a small or awkward parallel dim cannot build with a block chosen on transformer shapes. These pin
# the two halves of that: adaptation must fire where the block fails, and must NOT fire where it
# holds (which is what keeps existing measurements comparable).
# ---------------------------------------------------------------------------


def test_frozen_block_caps_match_the_registered_point():
    """The caps table must describe the feature the registration actually emits."""
    from merlin.llvmlower import impr_features as impr
    from merlin.llvmlower.frozen_blocks import frozen_block_caps, frozen_block_per_class

    caps = frozen_block_caps(impr.WHOLEMODEL_VF_NAME)
    blocks = frozen_block_per_class(impr.WHOLEMODEL_VF_NAME)
    assert caps == {"MR": impr.WHOLEMODEL_VF_CAPS[0], "NR": impr.WHOLEMODEL_VF_CAPS[1],
                    "KC": impr.WHOLEMODEL_VF_CAPS[2]}
    # The schedule tiles matmul [MR_mm, NR] and batch_matmul [1, MR, NR_bmm].
    assert blocks["linalg.matmul"] == (impr.WHOLEMODEL_VF_MR_MM, impr.WHOLEMODEL_VF_CAPS[1])
    assert blocks["linalg.batch_matmul"] == (impr.WHOLEMODEL_VF_CAPS[0],
                                            impr.WHOLEMODEL_VF_NR_BMM)
    assert frozen_block_caps("not_a_frozen_point") is None


def test_adaptation_is_a_noop_when_the_frozen_block_lowers():
    """Extents the frozen block already fits must keep the frozen feature name.

    This is the property that protects existing results: substituting an equivalent-but-differently-
    named point would change the emitted kernel for models whose numbers are already published.
    """
    from merlin.llvmlower import impr_features as impr
    from merlin.rvvgen.apply import _adapt_frozen_points

    # matmul M%1==0 and N%16==0; batch_matmul M%4==0 and N%8==0 -> both frozen blocks lower.
    shapes = [
        _Shape("linalg.matmul", (8, 256)),
        _Shape("linalg.matmul", (8, 2048)),
        _Shape("linalg.batch_matmul", (4, 8, 64)),
    ]
    feats = _adapt_frozen_points([impr.WHOLEMODEL_VF_NAME], shapes, target="rvv")
    assert feats == [impr.WHOLEMODEL_VF_NAME]


def test_adaptation_fires_only_for_the_failing_op_class():
    """A class whose frozen block cannot lower is re-derived; the other class keeps its block."""
    from merlin.llvmlower import impr_features as impr
    from merlin.rvvgen.apply import _adapt_frozen_points

    # matmul N=8 with the frozen NR=16: MR==1 but NR > N, so the fully-masked single iteration
    # does not lower (the measured rank-1 escape needs NR <= N). batch_matmul is left fitting.
    shapes = [
        _Shape("linalg.matmul", (1, 8)),
        _Shape("linalg.matmul", (1, 1024)),
        _Shape("linalg.batch_matmul", (4, 8, 64)),
    ]
    feats = _adapt_frozen_points([impr.WHOLEMODEL_VF_NAME], shapes, target="rvv")
    assert feats != [impr.WHOLEMODEL_VF_NAME]
    assert len(feats) == 1
    name = feats[0]
    # accum_resident_v3p_<MR_mm>_<NR_mm>_<MR_bmm>_<NR_bmm>_<KC>
    mr_mm, nr_mm, mr_bmm, nr_bmm, kc = (int(p) for p in name.rsplit("_", 5)[1:])
    assert (mr_mm, nr_mm) == (1, 8), f"matmul block not re-derived to fit N=8: {name}"
    assert (mr_bmm, nr_bmm) == (impr.WHOLEMODEL_VF_CAPS[0], impr.WHOLEMODEL_VF_NR_BMM), (
        f"batch_matmul block changed even though the frozen one lowers: {name}")
    assert kc == impr.WHOLEMODEL_VF_CAPS[2]


def test_a_class_with_no_multi_lane_block_is_left_unclaimed():
    """An extent of N=1 admits only a 1-lane block, which is not a vectorization at all.

    Vectorizing that tile emits a parallel-dim-free ``vector.contract`` (a ``vector<1xT>`` dot into a
    scalar) that no ``lower_contraction`` strategy matches, so the build fails at LLVM translation
    after a full compile. Measured on whisper_tiny, whose decode-step attention has N=1 alongside a
    1500-wide encoder attention in the SAME op class. The policy must decline the class (leaving it
    for convert-linalg-to-loops) and keep the other class vectorized.
    """
    from merlin.llvmlower import impr_features as impr
    from merlin.rvvgen.apply import _adapt_frozen_points

    shapes = [
        _Shape("linalg.matmul", (1, 384)),
        _Shape("linalg.matmul", (1500, 1536)),
        _Shape("linalg.batch_matmul", (6, 1, 1)),          # the decode step: N=1
        _Shape("linalg.batch_matmul", (6, 1500, 1500)),
    ]
    feats = _adapt_frozen_points([impr.WHOLEMODEL_VF_NAME], shapes, target="rvv")
    assert len(feats) == 1
    name = feats[0]
    assert name.endswith(f"_x_x_{impr.WHOLEMODEL_VF_CAPS[2]}"), (
        f"batch_matmul should be unclaimed, not blocked at 1 lane: {name}")
    # the matmul class still gets real lanes
    assert name.startswith("accum_resident_v3p_1_16_"), name
    sched = impr.apply_schedule("", frozenset([name]))
    assert "linalg.matmul" in sched
    assert "linalg.batch_matmul" not in sched, (
        "the unclaimed class must not appear in the schedule (it goes to convert-linalg-to-loops)")


def test_claiming_no_class_at_all_is_rejected():
    """Both blocks None would vectorize nothing — that is the scalar backend, not a micro-kernel."""
    import pytest

    from merlin.llvmlower.impr_features import ensure_v3_perop_microkernel

    with pytest.raises(ValueError):
        ensure_v3_perop_microkernel(None, None, None, None, 16)


def test_skipping_a_class_leaves_the_other_arm_byte_identical():
    """The skip must remove only the skipped class's arms, not perturb the surviving ones."""
    from merlin.llvmlower.impr_features import _accumulator_resident_v3_pre_schedule as sched

    full = sched(4, 16, 16, NR_bmm=8, MR_mm=1)
    no_bmm = sched(4, 16, 16, NR_bmm=8, MR_mm=1, skip_bmm=True)
    no_mm = sched(4, 16, 16, NR_bmm=8, MR_mm=1, skip_mm=True)
    # every line of the reduced schedules appears verbatim in the full one
    for reduced in (no_bmm, no_mm):
        for line in reduced.splitlines():
            assert line in full, f"skipping a class changed an unrelated line: {line!r}"
    assert "linalg.batch_matmul" not in no_bmm and "linalg.matmul" in no_bmm
    assert "linalg.matmul" not in no_mm.replace("linalg.batch_matmul", "")


def test_schedule_pinned_blocks_are_read_by_following_handles():
    """The block is recovered from a schedule's own tile_sizes, not from a fixed spelling.

    Packages that carry no compiler feature keep their register block in the schedule text, where the
    shape resolver cannot re-derive it. Reading it back is what lets a caller warn instead of silently
    running ~34x slow.
    """
    from merlin.rvvgen.apply import _schedule_pinned_blocks

    sched = """
    module {
      transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
        %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op)
        %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 2, 16, 1] : (!transform.any_op) -> (!transform.any_op)
        transform.yield
      }
    }
    """
    blocks = _schedule_pinned_blocks(sched)
    assert blocks["linalg.matmul"] == (4, 8)
    # batch_matmul tiles [B, M, N, K] -> the two PARALLEL tiles, not the leading batch tile
    assert blocks["linalg.batch_matmul"] == (2, 16)
    assert _schedule_pinned_blocks("not a schedule") == {}


def test_a_schedule_pinned_block_that_masks_is_reported_not_swallowed():
    """M=1/M=3 against an MR=4 schedule block must produce a message naming the op class."""
    from merlin.rvvgen.apply import blocking_risks

    class _Pkg:
        compiler_features: tuple = ()
        schedule_text = ('%mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 '
                         ': (!transform.any_op) -> !transform.any_op\n'
                         '%t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] '
                         ': (!transform.any_op) -> (!transform.any_op)\n')

    import merlin.rvvgen.apply as apply_mod

    shapes = [_Shape("linalg.matmul", (1, 64)), _Shape("linalg.matmul", (3, 4096)),
              _Shape("linalg.matmul", (64, 256))]
    orig = apply_mod.__dict__.get("contraction_shapes")
    import merlin.kernels.shapes as shapes_mod
    saved = shapes_mod.contraction_shapes
    shapes_mod.contraction_shapes = lambda _p: shapes
    try:
        msgs = blocking_risks(_Pkg(), "/nonexistent")
    finally:
        shapes_mod.contraction_shapes = saved
        assert orig is None or apply_mod.contraction_shapes is orig
    assert len(msgs) == 1 and "linalg.matmul" in msgs[0]
    assert "[4, 8]" in msgs[0]


def test_a_frozen_feature_package_reports_no_schedule_risk():
    """The resolver already adapts a frozen point, so warning about it would be noise."""
    from merlin.llvmlower import impr_features as impr
    from merlin.rvvgen.apply import blocking_risks

    class _Pkg:
        compiler_features = (impr.WHOLEMODEL_VF_NAME,)
        schedule_text = ""

    assert blocking_risks(_Pkg(), "/nonexistent") == []


class _Shape:
    """Minimal stand-in for kernels.microkernel.ContractionShape (op + parallel extents)."""

    def __init__(self, op, parallel):
        self.op = op
        self.parallel = parallel


def test_the_per_hart_tile_is_what_a_multicore_block_must_cover():
    """``harts`` re-expresses each matmul's N as the tile ONE hart gets, not the model's N.

    The multicore stage wraps every contraction in an ``scf.forall`` over the harts BEFORE the
    package's schedule runs, so the register block sees ``ceil(N / harts)`` (and the smaller
    remainder tile). Resolving against the unsplit N is how a block that lowers at 1 and 4 harts
    failed at 3.
    """
    from merlin.rvvgen.apply import _harts_split_shapes

    shapes = [_Shape("linalg.matmul", (32, 2)), _Shape("linalg.batch_matmul", (6, 8, 64))]
    assert [s.parallel for s in _harts_split_shapes(shapes, 1)] == [(32, 2), (6, 8, 64)]
    split = _harts_split_shapes(shapes, 3)
    mm = sorted(s.parallel[-1] for s in split if s.op == "linalg.matmul")
    assert mm == [1], f"N=2 over 3 harts is a 1-wide tile, got {mm}"
    # batch_matmul splits over its BATCH dim, which is not part of the (M, N) block
    assert [s.parallel for s in split if s.op == "linalg.batch_matmul"] == [(6, 8, 64)]


def test_a_remainder_tile_is_checked_too_not_just_the_ceiling():
    """The last hart's tile is smaller than the others; a block legal only for the big tile is wrong."""
    from merlin.rvvgen.apply import _harts_split_shapes

    # N=10 over 4 harts -> ceil = 3 for three harts, remainder 1 for the fourth.
    tiles = sorted({s.parallel[-1] for s in
                    _harts_split_shapes([_Shape("linalg.matmul", (8, 10))], 4)})
    assert tiles == [1, 3], tiles
