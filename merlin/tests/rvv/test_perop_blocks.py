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

from merlin.common.paths import repo_root
from merlin.llvmlower import perop_blocks as pb


class _S:
    """Stand-in for a ContractionShape (op + parallel + reduction extents)."""

    def __init__(self, op, parallel, reduction=(), dtypes=()):
        self.op = op
        self.parallel = tuple(parallel)
        self.reduction = tuple(reduction)
        self.dtypes = tuple(dtypes)


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


def test_mr_defaults_to_one_but_the_cap_is_a_cap_not_a_pin():
    """The default stays 1 so a caller that passes no cap does not move.

    This test used to be titled "MR is PINNED at one by default" and justified by "MR>1 is 2.56x
    SLOWER (measured, deepjscc)". That reading is superseded: the ladder it blamed was a real defect in
    `accum_microkernel` (no integer path to a scalar A operand) and is fixed, and the LARGER cost was a
    separate per-tile `@memrefCopy` self-copy now implied by every MR>1 recipe. With both fixed, MR=4
    beats MR=1 on the live K1 at 128^3: f32 3.20x, int8 1.58x, cos-gated. So what must hold is not that
    MR is pinned, but that the cap behaves as an upper BOUND -- honouring a caller that raises it, and
    still returning MR=1 where no clean M-tile exists.
    """
    assert pb.DEFAULT_MR == 1
    shapes = [_S("linalg.matmul", (64, 256), (288,))]      # M=64 admits MR=4
    assert set(pb.block_table(shapes, nr_cap=16).values()) == {(1, 16)}          # default unchanged
    assert set(pb.block_table(shapes, mr_cap=4, nr_cap=16).values()) == {(4, 16)}  # cap honoured
    # ...and a shape with no clean M-tile is NOT forced up to the cap
    odd = [_S("linalg.matmul", (17, 256), (288,))]
    assert set(pb.block_table(odd, mr_cap=4, nr_cap=16).values()) == {(1, 16)}


def test_the_whole_model_backend_offers_the_measured_mr_cap():
    """The cap is only a lever if the whole-model path actually passes it -- a default-off knob nobody
    passes is the failure mode this whole line of work is about."""
    from merlin.runtime.backends import zephyr_model as zm
    assert zm.perop_mr_cap() == 4
    src = (repo_root() / "merlin/python/merlin/runtime/backends/zephyr_model.py").read_text()
    assert "mr_cap=perop_mr_cap()" in src, "block_table must be called with the MR cap"


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


def test_a_per_op_block_covers_the_per_hart_tile_not_the_whole_extent():
    """The multicore stage wraps each matmul in an scf.forall over N BEFORE the package schedule
    runs, so a block chosen from the unsplit extent can exceed the tile a hart actually gets. That
    is not a slowdown, it is a build failure: `'vector.mask' op expects only one operation to mask`,
    measured on lstmnetvit at --harts 3 (an N=2 and an N=3 contraction) while 1 hart built fine."""
    from merlin.llvmlower import perop_blocks as pb

    class S:
        def __init__(self, op, par, red):
            self.op, self.parallel, self.reduction = op, par, red

    narrow = S("linalg.matmul", (1, 3), (128,))       # N=3: one lane per hart at 3 harts
    wide = S("linalg.matmul", (64, 96), (288,))       # N=96: 32 per hart, still blockable

    assert pb.block_table([narrow], nr_cap=16, harts=1)                  # blockable alone
    assert not pb.block_table([narrow], nr_cap=16, harts=3), \
        "a 3-wide N split over 3 harts leaves one lane; the op must be declined, not masked"
    assert pb.block_table([wide], nr_cap=16, harts=3), "a wide N must stay on the vector path"

    # The KEY must stay the unsplit geometry: the tag is applied to the op before the forall split,
    # so a key computed from the tile would never match anything.
    key = next(iter(pb.block_table([wide], nr_cap=16, harts=3)))
    assert key == pb.shape_key("linalg.matmul", (64, 96), (288,))


def test_batch_matmul_blocks_are_unaffected_by_the_hart_count():
    """batch_matmul splits over BATCH, which is not part of the (M, N) block."""
    from merlin.llvmlower import perop_blocks as pb

    class S:
        op, parallel, reduction = "linalg.batch_matmul", (2, 96, 32), (6,)

    assert (pb.block_table([S()], nr_cap=16, harts=1)
            == pb.block_table([S()], nr_cap=16, harts=3))


def test_the_block_cap_follows_the_boards_vector_length():
    """A fixed ELEMENT COUNT does not scale with the vector unit: a wider machine spends it as a
    smaller LMUL rather than as more work per instruction.

    Measured on the same model built two ways, with the cap fixed at 16:

        VLEN=128:  e16,m2  / e8,m1  / e16,m1     -- 16 elements across one or two whole registers
        VLEN=512:  e16,mf2 / e8,mf4 / e16,mf4    -- the same 16 elements in HALF or a QUARTER of one

    i.e. the 512-bit machine issued the same count of vector instructions doing the same 16 elements
    each as the 128-bit one; three quarters of its datapath went unused. With the cap scaled to 32 the
    dominant ops became e32,m2 and e16,m1 (whole registers) and the total vector-op count fell 1202 ->
    1122, and the result stayed bit-exact on spike at VLEN=512 (tier_ok=w8a8, cos 1.0, max_rel 0.0).
    """
    from merlin.runtime.backends.zephyr_model import (_PEROP_NR_CAP, _PEROP_NR_CAP_REF_VLEN,
                                                      perop_nr_cap)

    # Scales UP only, so nothing already measured moves as a side effect: the champion was tuned at
    # the reference width and keeps its value there and below.
    assert perop_nr_cap(_PEROP_NR_CAP_REF_VLEN) == _PEROP_NR_CAP
    assert perop_nr_cap(128) == _PEROP_NR_CAP
    assert perop_nr_cap(None) == _PEROP_NR_CAP, "an unknown VLEN must not widen the block"
    # A wider unit gets a proportionally wider tile.
    assert perop_nr_cap(512) == 2 * _PEROP_NR_CAP
    assert perop_nr_cap(1024) == 4 * _PEROP_NR_CAP


def test_the_vector_length_reaches_the_block_table():
    """The cap is only useful if the value threads through; a parameter accepted and dropped is the
    failure mode that shipped a wrong block table once already."""
    import inspect

    from merlin.runtime.backends import zephyr_model as zm

    prep = inspect.getsource(zm.prepare_for_lowering)
    assert "nr_cap=perop_nr_cap(vlen)" in prep, "the cap must be derived from the board's vlen"
    assert "vlen: int | None" in prep, "prepare_for_lowering must accept it"
    build = inspect.getsource(zm.build_app)
    assert "vlen=vlen" in build, "build_app must pass it down"


# ---------------------------------------------------------------------------------------
# Priced-vs-tagged agreement. The two sides are computed at DIFFERENT points: `block_table`
# prices `contraction_shapes` of the PREPARED module; `tag_prepared_mlir` tags the module
# AFTER `linalg-specialize-generic-ops`. A contraction priced but not tagged matches no
# schedule arm and falls to `convert-linalg-to-loops` -- producing CORRECT numbers, so no
# correctness gate catches it. That is the measured deepjscc "2.56x regression that looks
# like a bad block but is an untagged build". Hence: hard failure.
# ---------------------------------------------------------------------------------------

def test_a_priced_but_untagged_contraction_is_a_hard_failure():
    table = {"linalg.matmul:64x256:288": (1, 16), "linalg.matmul:32x32:32": (4, 16)}
    stdout = ('OK perop_blocks tagged 1\n'
              'MERLIN_PEROP_AGREEMENT {"hit": ["linalg.matmul:64x256:288"], '
              '"untagged": ["linalg.matmul:31x32:32"]}\n')
    with pytest.raises(pb.BlockAgreementError) as e:
        pb._assert_priced_is_tagged(table, stdout)
    assert "linalg.matmul:32x32:32" in str(e.value)
    assert "scalar" in str(e.value)                      # says what the consequence IS
    assert "linalg.matmul:31x32:32" in str(e.value)      # ...and what the tagger saw instead


def test_full_agreement_passes():
    table = {"linalg.matmul:64x256:288": (1, 16)}
    pb._assert_priced_is_tagged(
        table, 'MERLIN_PEROP_AGREEMENT {"hit": ["linalg.matmul:64x256:288"], "untagged": []}\n')


def test_a_guard_that_cannot_run_must_not_report_success():
    """The repo's recurring failure: a check that could not run reported SUCCESS. If the tagger emits
    no agreement line, the guard has no evidence and must refuse -- not pass."""
    with pytest.raises(pb.BlockAgreementError) as e:
        pb._assert_priced_is_tagged({"linalg.matmul:64x256:288": (1, 16)},
                                    "OK perop_blocks tagged 1\n")
    assert "cannot verify" in str(e.value)


def test_the_tagger_reports_both_sides_of_the_agreement():
    """The runner source must actually produce the line the guard parses, and report the key sets
    (a count alone cannot distinguish 'tagged a different op' from 'tagged the priced one')."""
    src = pb.runner_rewrite_src({"linalg.matmul:64x256:288": (1, 16)})
    assert "return n, hit, seen_untagged" in src
    assert "hit.add(key)" in src
    assert "seen_untagged.add(str(key))" in src


# ---------------------------------------------------------------------------------------
# Dtype-aware N cap. NR is an ELEMENT count, so one number is a different fraction of the
# register file at each element width: at VLEN=256, NR=16 is m2 at e32, m1 at e16 and only
# mf2 -- half a register -- at e8. `perop_nr_cap` already scales with VLEN; it could not
# see the element width because `_rvv_best_block` discarded ContractionShape.dtypes.
# ---------------------------------------------------------------------------------------

def test_the_n_cap_widens_for_narrow_elements_and_never_narrows():
    # e8 on a 256-bit unit needs 32 elements to fill one register; e32 already fills two at 16.
    assert pb.nr_cap_for_dtypes(16, 256, ("i8", "i8", "i32")) == 32
    assert pb.nr_cap_for_dtypes(16, 256, ("f32", "f32", "f32")) == 16
    assert pb.nr_cap_for_dtypes(16, 256, ("bf16", "bf16", "f32")) == 16
    # never LOWER what the caller asked for, even where the width would allow less
    assert pb.nr_cap_for_dtypes(32, 256, ("f32", "f32", "f32")) == 32
    # ...and it is the NARROWEST element that decides (an i8 x i8 -> i32 op is an e8 op)
    assert pb.narrowest_elem_bits(("i8", "i8", "i32")) == 8


def test_an_unreadable_dtype_falls_back_to_the_blind_cap_rather_than_guessing():
    """A synthetic shape and an observer that could not read the types look the same here. Inventing a
    width would silently pick a wrong N tile, so both must fall back to the caller's cap."""
    assert pb.narrowest_elem_bits(()) is None
    assert pb.narrowest_elem_bits(("not_a_type",)) is None
    assert pb.nr_cap_for_dtypes(16, 256, ()) == 16
    assert pb.nr_cap_for_dtypes(16, 256, ("not_a_type", "x", "y")) == 16
    assert pb.nr_cap_for_dtypes(16, None, ("i8", "i8", "i32")) == 16   # no vlen -> unchanged


def test_block_table_without_vlen_is_byte_identical_to_the_dtype_blind_behaviour():
    """The widening is opt-in: omitting vlen must not move a single block."""
    shapes = [_S("linalg.matmul", (64, 256), (288,)),
              _S("linalg.batch_matmul", (6, 1500, 1500), (64,))]
    assert (pb.block_table(shapes, mr_cap=4, nr_cap=16)
            == pb.block_table(shapes, mr_cap=4, nr_cap=16, vlen=None))


def test_the_widened_cap_reaches_the_chosen_block_for_an_int8_contraction():
    """End to end through block_table: an e8 contraction whose N admits 32 must GET 32 once the board's
    vlen is known, and an f32 one at the same extents must not move."""
    i8 = [_S("linalg.matmul", (64, 256), (288,), dtypes=("i8", "i8", "i32"))]
    f32 = [_S("linalg.matmul", (64, 256), (288,), dtypes=("f32", "f32", "f32"))]
    assert set(pb.block_table(i8, mr_cap=4, nr_cap=16, vlen=256).values()) == {(4, 32)}
    assert set(pb.block_table(i8, mr_cap=4, nr_cap=16).values()) == {(4, 16)}      # opt-in only
    assert set(pb.block_table(f32, mr_cap=4, nr_cap=16, vlen=256).values()) == {(4, 16)}


def test_a_shape_that_cannot_take_the_wider_tile_keeps_the_narrower_one():
    """It is a CAP, not a pin: N=24 has no legal 32-wide tile, so the widening must not force one."""
    odd = [_S("linalg.matmul", (64, 24), (288,), dtypes=("i8", "i8", "i32"))]
    blocks = set(pb.block_table(odd, mr_cap=4, nr_cap=16, vlen=256).values())
    assert blocks and all(nr <= 24 for _mr, nr in blocks), blocks


def test_the_n_fill_knob_is_off_by_default_and_only_turns_on_by_request():
    """Its SIGN is model-dependent (K1: 1.160x faster on spectformer int8, 1.196x SLOWER on small_llama
    int8, both outside the 2.6% band), so it must be a search knob and NOT a default. The mechanism is
    visible in the object: the i32 accumulator sets LMUL, so NR=16 is already e32,m4 with zero
    accumulator spills and NR=32 is e32,m8 with six. This pins the wiring that keeps it opt-in."""
    import inspect

    from merlin.llvmlower.impr_features import PEROP_NR_FILL_NAME
    from merlin.runtime.backends import zephyr_model as zm

    prep = inspect.getsource(zm.prepare_for_lowering)
    # the ONLY thing that turns it on
    assert "nr_fill_vlen = vlen if PEROP_NR_FILL_NAME in features else None" in prep
    assert "vlen=nr_fill_vlen" in prep, "block_table must receive the GATED vlen, not the raw one"
    # and the sentinel is stripped so it can never reach lowering
    assert "features = features - {PEROP_NR_FILL_NAME}" in prep


def test_the_n_fill_knob_implies_the_blocking_it_has_no_meaning_without():
    from merlin.llvmlower import impr_features as F
    from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME, PEROP_NR_FILL_NAME

    assert F.get(PEROP_NR_FILL_NAME).implies == frozenset({PEROP_BLOCK_NAME})
    assert F.normalize([PEROP_NR_FILL_NAME]) == frozenset({PEROP_NR_FILL_NAME, PEROP_BLOCK_NAME})
    # it changes the TABLE, not the schedule shape, so it must not claim a schedule replacement
    # (two replacements cannot compose, and the block feature it implies is already one)
    assert F.get(PEROP_NR_FILL_NAME).schedule_replace is False


def test_the_n_fill_measurement_is_recorded_with_BOTH_signs():
    """A lever measured faster on one model and slower on another must record both, or the next reader
    turns it on citing half the evidence."""
    from merlin.llvmlower import impr_features as F
    from merlin.llvmlower.impr_features import PEROP_NR_FILL_NAME

    desc = F.get(PEROP_NR_FILL_NAME).description
    assert "1.160x faster" in desc and "1.196x slower" in desc
    assert "m4" in desc and "m8" in desc          # the mechanism, not just the numbers
    assert "search knob, not a default" in desc
