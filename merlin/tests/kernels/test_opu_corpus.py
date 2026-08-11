"""The corpus is the only correctness oracle this delta has, so it must not be able to pass vacuously.

With no functional simulator for the unit and no bitstream, whole-model numerics have no oracle in this
pass — correctness rests entirely on these shapes. That makes two failure modes worth defending against
specifically: a case that claims to exercise something it does not (the reason the overflow case was
removed rather than kept), and a case silently disappearing from the frozen file.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.kernels import opu_corpus as OC


class TestReference:
    def test_contracts_two_k_major_operands(self):
        # Both operands K-major is the layout the hardware forces; the result is (M, N).
        lhs = np.array([[1, 2], [3, 4]], dtype=np.int8)     # (K=2, M=2)
        rhs = np.array([[5], [6]], dtype=np.int8)           # (K=2, N=1)
        got = OC.reference(lhs, rhs)
        assert got.shape == (2, 1)
        assert got.tolist() == [[1 * 5 + 3 * 6], [2 * 5 + 4 * 6]]

    def test_matches_a_plain_transpose_matmul_in_a_wide_type(self):
        rng = np.random.default_rng(7)
        a = rng.integers(-8, 9, size=(13, 5), dtype=np.int8)
        b = rng.integers(-8, 9, size=(13, 3), dtype=np.int8)
        assert (OC.reference(a, b) == a.astype(np.int64).T @ b.astype(np.int64)).all()

    def test_adds_bias_across_the_output_columns(self):
        a = np.ones((2, 3), dtype=np.int8)
        b = np.ones((2, 4), dtype=np.int8)
        bias = np.array([10, 20, 30, 40], dtype=np.int32)
        got = OC.reference(a, b, bias)
        assert got.tolist() == [[12, 22, 32, 42]] * 3

    def test_wraps_instead_of_saturating(self):
        # The hardware has no saturation logic, so where it overflows it wraps; a saturating reference
        # would disagree with correct hardware. Needs a K no model produces, which is the point.
        k = 200_000
        a = np.full((k, 1), 127, dtype=np.int8)
        got = OC.reference(a, a)
        exact = k * 127 * 127
        assert exact > 2**31 - 1, "this case must actually exceed the accumulator"
        assert int(got[0, 0]) == exact - 2**32
        assert got.dtype == np.int32

    def test_rejects_operands_that_do_not_share_a_reduction_extent(self):
        with pytest.raises(ValueError):
            OC.reference(np.ones((4, 2), dtype=np.int8), np.ones((5, 2), dtype=np.int8))

    def test_rejects_a_non_matrix_operand(self):
        with pytest.raises(ValueError):
            OC.reference(np.ones((4,), dtype=np.int8), np.ones((4, 2), dtype=np.int8))


class TestOverflowIsUnreachable:
    def test_the_bound_is_derived_from_the_operand_and_accumulator_widths(self):
        assert OC.ACC_OVERFLOW_UNREACHABLE_ABOVE_K == ((1 << 31) - 1) // (127 * 127)

    def test_no_corpus_case_comes_near_the_accumulator_limit(self):
        # The claim in the module docstring, checked rather than asserted: if a future case did overflow,
        # the "retired hazard" framing would be wrong and this fails.
        worst = 0
        for case in OC.load_corpus():
            lhs, rhs, bias = case.operands()
            wide = lhs.astype(np.int64).T @ rhs.astype(np.int64)
            if bias is not None:
                wide = wide + bias[None, :]
            worst = max(worst, int(np.abs(wide).max()))
        assert worst < 2**31 - 1
        assert worst * 100 < 2**31 - 1, f"corpus worst magnitude {worst} is within 100x of the limit"

    def test_the_longest_corpus_reduction_is_far_below_the_bound(self):
        longest = max(c.k for c in OC.load_corpus())
        assert longest < OC.ACC_OVERFLOW_UNREACHABLE_ABOVE_K / 100


class TestCorpusContents:
    def test_the_narrow_parallel_extents_are_present_by_name(self):
        # The historical failure. These must be findable by name, not merely happen to be covered.
        names = {c.name for c in OC.load_corpus()}
        assert {"narrow_m_1", "narrow_n_1", "narrow_both_1"} <= names

    def test_both_asymmetric_orientations_are_present(self):
        # An operand swap in the accumulate is shape-safe on a square tile, which is how one hid.
        cases = {c.name: c for c in OC.load_corpus()}
        assert cases["asymmetric_m_gt_n"].m > cases["asymmetric_m_gt_n"].n
        assert cases["asymmetric_n_gt_m"].n > cases["asymmetric_n_gt_m"].m

    def test_every_case_states_why_it_exists(self):
        for case in OC.load_corpus():
            assert len(case.why) > 20, f"{case.name} has no reason it may not be deleted"

    def test_every_case_has_positive_extents(self):
        for c in OC.load_corpus():
            assert c.m > 0 and c.n > 0 and c.k > 0, c.name

    def test_case_names_are_unique(self):
        names = [c.name for c in OC.load_corpus()]
        assert len(names) == len(set(names))

    def test_a_bias_case_exists_including_one_with_a_single_output_row(self):
        # The hardware initialises bias by broadcasting it across ALL rows, whether or not they are read.
        biased = [c for c in OC.load_corpus() if c.bias]
        assert biased
        assert any(c.m == 1 for c in biased)

    def test_the_reduction_extent_is_swept_including_one(self):
        ks = {c.k for c in OC.load_corpus()}
        assert 1 in ks and 2 in ks and 3 in ks
        assert max(ks) > 100, "a long reduction must be covered too"


class TestFrozenOnDisk:
    def test_the_corpus_file_is_committed_and_is_what_loads(self):
        # Frozen as data so a case cannot vanish because the enumeration logic changed.
        assert OC.corpus_is_frozen(), f"expected a frozen corpus at {OC.CORPUS_PATH}"

    def test_the_frozen_file_matches_the_declared_cases(self, tmp_path):
        # If these diverge, the file is stale and the gate is running against yesterday's shapes.
        written = OC.load_corpus(OC.write_corpus(tmp_path / "corpus.json"))
        assert written == OC.load_corpus()

    def test_operands_are_deterministic_from_the_seed(self):
        case = OC.load_corpus()[0]
        first, second = case.operands()[0], case.operands()[0]
        assert (first == second).all(), "a failing case must be reproducible"

    def test_two_cases_do_not_share_operand_data(self):
        cases = {c.name: c for c in OC.load_corpus()}
        a = cases["m_16"].operands()[0]
        b = cases["n_16"].operands()[0]
        assert a.shape != b.shape or not (a == b).all()

    def test_the_lhs_is_generated_k_major(self):
        # Handing back an M-major LHS would hide the transpose/packing cost the routing decision prices.
        case = next(c for c in OC.load_corpus() if c.name == "asymmetric_m_gt_n")
        lhs, rhs, _ = case.operands()
        assert lhs.shape == (case.k, case.m)
        assert rhs.shape == (case.k, case.n)
