"""Per-op register blocking: one block per CONTRACTION, not per op class.

The class-wide policy is one decision too coarse. whisper_tiny's batch_matmul class holds a 1500-wide
encoder attention and a single-token decode step whose N=1; the only block legal for both is one lane
wide, so the policy declines the class and loses 34% of the model's MACs. Blocking per op recovers it.

Two measured facts shape the implementation and are pinned here, because both were wrong on the first
attempt:
  * the tag must be applied AFTER linalg-specialize-generic-ops (which renames the capture's contraction
    generics and drops discardable attributes: 20 renamed, 0 kept the tag), and
  * the tag must name the OP CLASS, or a batch_matmul arm (4 tile sizes) matches rank-2 matmul ops and
    the schedule dies with "too many tiles provided, expected at most 3 found 4".
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import perop_blocks as pb


class _S:
    """Stand-in for a ContractionShape (op + parallel + reduction extents)."""

    def __init__(self, op, parallel, reduction=()):
        self.op = op
        self.parallel = tuple(parallel)
        self.reduction = tuple(reduction)


def test_each_contraction_gets_its_own_block():
    """A wide op and a narrow op in the SAME class must not be clamped to one block."""
    shapes = [_S("linalg.batch_matmul", (6, 1500, 1500), (64,)),
              _S("linalg.batch_matmul", (6, 8, 64), (64,))]
    t = pb.block_table(shapes, nr_cap=16)
    blocks = {v for v in t.values()}
    assert (1, 16) in blocks, f"the 1500-wide op must get the full N tile: {t}"
    assert len(t) == 2, "both geometries must be claimed"


def test_a_one_lane_op_is_left_out_not_forced_on_the_class():
    """An N=1 op has no multi-lane block; it must drop out ALONE, not take its class with it."""
    shapes = [_S("linalg.batch_matmul", (6, 1500, 1500), (64,)),
              _S("linalg.batch_matmul", (6, 1, 1), (64,))]
    t = pb.block_table(shapes, nr_cap=16)
    cov = pb.coverage(shapes, t)
    assert len(t) == 1, "the N=1 op must be excluded"
    assert cov["claimed_mac_fraction"] > 0.999, (
        f"the wide op must still be claimed; got {cov['claimed_mac_fraction']}")
    assert len(cov["unclaimed"]) == 1


def test_mr_is_pinned_at_one_by_default():
    """MR>1 is 2.56x SLOWER on this micro-kernel (measured, deepjscc): the vfmacc.vf form needs a
    SCALAR A operand, and MR>1 rebuilds the A column with a vmv/vslideup ladder that spills."""
    assert pb.DEFAULT_MR == 1
    shapes = [_S("linalg.matmul", (64, 256), (288,))]      # M=64 would admit MR=4 on area alone
    assert set(pb.block_table(shapes, nr_cap=16).values()) == {(1, 16)}


def test_the_tag_names_the_op_class():
    """A class-agnostic tag lets a 4-tile bmm arm match a rank-2 matmul -> 'too many tiles provided'."""
    assert pb.tag_for("linalg.matmul", 1, 16) != pb.tag_for("linalg.batch_matmul", 1, 16)
    assert "mm_1x16" in pb.tag_for("linalg.matmul", 1, 16)
    assert "bmm_1x16" in pb.tag_for("linalg.batch_matmul", 1, 16)


def test_the_schedule_emits_one_arm_per_block_with_the_right_rank():
    t = {"linalg.matmul:64x256:288": (1, 16), "linalg.batch_matmul:6x1500x1500:64": (1, 8)}
    s = pb.schedule_text(t, 16)
    assert s.count("transform.structured.match attributes{") == 2
    assert "tile_sizes [1, 16, 0]" in s          # matmul: 3 tile sizes
    assert "tile_sizes [1, 1, 8, 0]" in s        # batch_matmul: 4 tile sizes
    assert s.count("transform.structured.vectorize") == 2


def test_the_k_tile_chains_the_handle_instead_of_rematching():
    """Re-matching by op name after tiling is ambiguous -- it selects that class's ops again. Chaining
    the returned handle targets exactly the op the first tile produced, and needs no attribute to
    survive tiling. Measured: the chained form makes deepjscc BIT-EXACT (w8a8_rel 0.0) where the
    re-matching v3 schedule scores cos 0.9176."""
    t = {"linalg.matmul:64x256:288": (1, 16)}
    s = pb.schedule_text(t, 16)
    assert 'match ops{["linalg.matmul"]}' not in s, "no re-match by op name"
    assert "%b0k, %b0kl = transform.structured.tile_using_for %b0t" in s, "K tile must chain %b0t"


def test_shape_key_survives_a_square_contraction():
    """K must be operand 0's last dim, not 'the dim that is not a result dim' -- a square matmul would
    otherwise key as K=1 and never be tagged."""
    k1 = pb.shape_key("linalg.matmul", (256, 256), (256,))
    k2 = pb.shape_key("linalg.matmul", (256, 256), (128,))
    assert k1 != k2


def test_coverage_is_mac_weighted():
    """One huge claimed op must outweigh a dozen tiny unclaimed ones, or the metric mis-ranks the loss."""
    shapes = [_S("linalg.matmul", (1024, 1024), (1024,)), _S("linalg.matmul", (1, 1), (1,))]
    cov = pb.coverage(shapes, pb.block_table(shapes, nr_cap=16))
    assert cov["claimed_mac_fraction"] > 0.9999


def test_an_empty_table_claims_nothing_and_says_so():
    shapes = [_S("linalg.matmul", (1, 1), (1,))]
    t = pb.block_table(shapes, nr_cap=16)
    assert t == {}
    assert pb.coverage(shapes, t)["claimed_mac_fraction"] == 0.0


@pytest.mark.parametrize("bundle,expect_claimed", [
    ("whisper_tiny_int8_full", 0.999),      # was 0.659 per op CLASS
    ("spectformer_int8_full", 0.999),
    ("deepjscc_int8_full", 0.999),
])
def test_real_bundles_are_fully_claimed_per_op(bundle, expect_claimed):
    """The headline: per-op blocking claims essentially every MAC of every captured workload."""
    from merlin.common.artifacts import recaptures_dir
    from merlin.kernels.shapes import contraction_shapes

    p = recaptures_dir() / bundle / "model.mlir"
    if not p.is_file():
        pytest.skip(f"{bundle} not captured")
    shapes = contraction_shapes(p)
    cov = pb.coverage(shapes, pb.block_table(shapes, nr_cap=16))
    assert cov["claimed_mac_fraction"] >= expect_claimed, cov
