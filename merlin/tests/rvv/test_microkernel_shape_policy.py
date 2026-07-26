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
    assert got == [ContractionShape("linalg.matmul", (8, 344), (128,))]

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
    assert contraction_shapes(generic) == [ContractionShape("linalg.matmul", (8, 344), (128,))]


def test_shape_observer_degrades_to_empty_on_unreadable_input():
    """A failed observation must mean "I observed nothing" (-> shape-blind fallback), never a raise."""
    assert contraction_shapes("this is not mlir {{{") == []
