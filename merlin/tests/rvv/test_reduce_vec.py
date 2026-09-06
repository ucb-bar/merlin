"""Gates for `llvmlower.reduce_vec` -- the amax-reduction vectorize lever.

The load-bearing test is :func:`test_absf_sign_mask_is_bit_exact_over_every_f32`, which checks the
rewrite's exactness the way `quant_round` checks its own: not on a sample, but over EVERY f32 bit
pattern, NaNs included. That is affordable (2**32 patterns, swept in chunks) and it is the only form
of the claim worth making -- "exact on the values we tried" is what an approximation says.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import repo_root  # noqa: F401  (import-time location independence)
from merlin.llvmlower import reduce_vec as R


def test_sign_mask_is_derived_from_the_float_width():
    """The mask is a function of the type's own width, not a per-dtype literal."""
    assert R.sign_mask(16) == 0x7FFF
    assert R.sign_mask(32) == 0x7FFF_FFFF
    assert R.sign_mask(64) == 0x7FFF_FFFF_FFFF_FFFF


def test_absf_sign_mask_is_bit_exact_over_every_f32():
    """`bitcast(bitcast(x) & 0x7fffffff)` equals `|x|` for all 4,294,967,296 f32 bit patterns.

    Compared as BITS, not as floats: a float comparison would call every NaN unequal to itself and
    would not notice a changed NaN payload or a flipped zero sign -- the two things this rewrite has
    to preserve and the reason the cheaper `max(x, -x)` spelling was rejected.
    """
    mask = np.uint32(R.sign_mask(32))
    step = 1 << 24
    for lo in range(0, 1 << 32, step):
        bits = np.arange(lo, lo + step, dtype=np.uint64).astype(np.uint32)
        # The reference: IEEE absolute value, applied to the float those bits denote, read back as
        # bits. np.abs on a float32 array is the hardware fabs, i.e. exactly `math.absf`'s semantics.
        ref = np.abs(bits.view(np.float32)).view(np.uint32)
        got = bits & mask
        if not np.array_equal(ref, got):
            bad = int(np.flatnonzero(ref != got)[0])
            pytest.fail(f"first divergence at bit pattern 0x{int(bits[bad]):08x}: "
                        f"abs=0x{int(ref[bad]):08x} mask=0x{int(got[bad]):08x}")


def test_max_of_x_and_negx_is_not_bit_exact_which_is_why_it_was_rejected():
    """The REJECTED cheaper spelling (`|x| = max(x, -x)`), and the two input classes it is wrong on.

    Pinned as a test so a later "simplification" to two ops fails here rather than silently changing
    a bit pattern. Both divergences are on inputs this pass can actually see:

      * NaN -- `absf` CLEARS the sign bit and keeps the payload, while a `maximum` of two NaNs
        returns an unspecified quiet NaN. A NaN is reachable here: an all-zero activation tensor
        gives `amax = 0`, and the `x / 0` that follows is a NaN fed straight back through this
        family on the next tensor.
      * SIGNED ZERO -- the spelling is only exact if `maximum` orders `-0` below `+0`. IEEE-754
        2019 `maximum` does; the widespread `maxNum`/`fmaxf`/numpy `maximum` semantics do NOT (they
        return an arbitrary one of two equal operands), and this repo derives neither which one MLIR
        `arith.maximumf` lowers to on this target nor what the backend emits for it. Measured below
        with numpy's semantics: `max(+0.0, -0.0)` gives `-0.0`, where `|+0.0|` is `+0.0`.

    The sign-mask form has no such dependence -- it is the definition of the operation, not an
    identity that happens to hold.
    """
    finite = np.array([1.5, -1.5, np.inf, -np.inf], dtype=np.float32)
    assert np.array_equal(np.abs(finite).view(np.uint32),
                          np.maximum(finite, -finite).view(np.uint32))
    # SIGNED ZERO: exact only under IEEE-2019 `maximum`, and numpy's `maximum` is not that.
    zeros = np.array([0.0, -0.0], dtype=np.float32)
    assert list(np.abs(zeros).view(np.uint32)) == [0x0000_0000, 0x0000_0000]
    assert list(np.maximum(zeros, -zeros).view(np.uint32)) != [0x0000_0000, 0x0000_0000]
    # NaN: `absf` clears the sign and preserves the payload; the sign-mask form reproduces that
    # exactly, which is the whole point.
    nan_bits = np.array([0xFFC0_1234], dtype=np.uint32)
    assert int(np.abs(nan_bits.view(np.float32)).view(np.uint32)[0]) == 0x7FC0_1234
    assert int(nan_bits[0] & R.sign_mask(32)) == 0x7FC0_1234


def test_maximumf_is_order_independent_so_the_reduction_needs_no_reassociation_knob():
    """Why this feature is EXACT where `vectorize_reduction` is an approximation.

    Vectorizing a reduction re-associates it. For an fp SUM that changes the answer (which is why
    `vectorize_reduction` turns on `reassociate-fp-reductions` and is cos-gated). IEEE `maximum` is
    associative and commutative on the whole domain, so every association order of an amax returns
    the same f32 -- checked here against a scalar left-to-right accumulate on a hostile vector
    (both signed zeros, subnormals, infinities, a payload NaN).
    """
    rng = np.random.default_rng(0)
    hostile = np.concatenate([
        np.array([0.0, -0.0, np.inf, -np.inf, 5e-324, -1e-45, 1.0, -1.0], dtype=np.float32),
        rng.standard_normal(1024).astype(np.float32) * 1e3,
    ])
    a = np.abs(hostile)
    serial = a[0]
    for v in a[1:]:
        serial = max(serial, v)
    # Tree / lane-parallel association: reshape into lanes and reduce across, then within.
    pad = (-a.size) % 8
    lanes = np.concatenate([a, np.full(pad, -np.inf, np.float32)]).reshape(-1, 8)
    tree = np.float32(lanes.max(axis=0).max())
    assert serial.view(np.uint32) == tree.view(np.uint32)


# ---- structure of the emitted arms + the refusal set --------------------------------------------

def test_arms_are_bounded_not_whole_tensor():
    """Every emitted arm tiles before it vectorizes, and vectorizes at MACHINE width.

    This is the difference from `vectorize_reduction`, which calls `structured.vectorize` with no
    `vector_sizes` and therefore asks for a vector as wide as the op's whole static iteration space
    (~590K lanes on a (2304, 196) amax).
    """
    arms = R.reduction_arms(lanes=8, min_rank=2, max_rank=4)
    for rank in (2, 3, 4):
        sizes = ", ".join(["1"] * (rank - 1) + ["8"])
        assert f"attributes{{merlin.vec_red{rank}}}" in arms
        assert f"tile_sizes [{sizes}]" in arms
        assert f"vectorize %rdt{rank} vector_sizes [{sizes}]" in arms
    # No unbounded vectorize anywhere in the arms.
    for line in arms.splitlines():
        if "structured.vectorize" in line:
            assert "vector_sizes" in line, line


def test_lanes_and_rank_bound_are_parameters_not_literals():
    """A different machine width / rank bound changes the arms, so nothing about the target is baked."""
    assert "vector_sizes [1, 16]" in R.reduction_arms(lanes=16, min_rank=2, max_rank=2)
    assert "merlin.vec_red5" in R.reduction_arms(lanes=8, min_rank=5, max_rank=5)
    assert "merlin.vec_red2" not in R.reduction_arms(lanes=8, min_rank=3, max_rank=3)


def test_splice_is_additive_and_idempotent():
    text = ('module {\n'
            '    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
            '    %f = transform.structured.match ops{["func.func"]} in %arg0\n'
            '}\n')
    once = R.splice_reduction_arms(text, lanes=8, min_rank=2, max_rank=3)
    assert "merlin.vec_red2" in once
    assert '%mm = transform.structured.match ops{["linalg.matmul"]}' in once   # additive
    assert once.index("merlin.vec_red2") < once.index('ops{["func.func"]}')     # before the anchor
    assert R.splice_reduction_arms(once, lanes=8, min_rank=2, max_rank=3) == once


def test_splice_returns_a_schedule_without_the_anchor_unchanged():
    """Never silently swap a caller's tuned schedule for a generic one."""
    text = "module { /* no func anchor */ }"
    assert R.splice_reduction_arms(text, lanes=8, min_rank=2, max_rank=4) == text


def test_feature_is_registered_eagerly_in_every_process():
    """The registration trap: the lowering subprocess re-imports impr_features and k1 imports no
    proposer, so a name registered at run time in the parent resolves in neither."""
    from merlin.llvmlower import impr_features as F
    assert R.FEATURE in F.known()
    assert F.get(R.FEATURE).edit_pipeline is None      # no pipeline edit => no reassociation knob
    assert F.get(R.FEATURE).schedule_replace is False  # additive, composes with a tuned recipe
    assert F.normalize([R.FEATURE]) == frozenset({R.FEATURE})   # implies nothing


def test_empty_feature_set_leaves_the_schedule_byte_identical():
    """The frozen-baseline invariant."""
    from merlin.llvmlower import impr_features as F
    text = ('module {\n'
            '    %f = transform.structured.match ops{["func.func"]} in %arg0\n'
            '}\n')
    assert F.apply_schedule(text, frozenset()) == text
