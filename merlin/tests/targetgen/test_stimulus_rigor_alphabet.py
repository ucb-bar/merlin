"""Operand rigor must be judged against what the operand's ALPHABET makes achievable.

Two contracts, deliberately distinct:

* `rigor_findings` demands EXACT distinctness of every row and column. Correct for the operands
  `corpus_operands` synthesizes, which draw on a prime palette large enough to make it achievable, and
  relied on by the salt search in `operand_values`.
* `stimulus_rigor_findings` judges the integer stimulus `Tensor.deterministic` produces. That fill draws
  from lo..hi = 0..3 -- a FOUR-symbol alphabet -- so a 16-column single-row operand has at most 4 distinct
  columns no matter what the fill does.

Measured 2026-09-01: applying the exact contract to every shipped capsule root produced 78 findings, of
which the overwhelming majority were of that second kind. The hazard being guarded is a COLLAPSE (the
historical period-4 fill: 8 rows, 8 achievable, 1 observed), not a collision (1017 distinct of 1024).
"""
from __future__ import annotations

from merlin.common import stimulus as ST
from merlin.targetgen.corpus_operands import (
    MIN_ACHIEVED_FRACTION,
    rigor_findings,
    stimulus_rigor_findings,
)


class TestItCatchesTheHistoricalCollapse:
    def test_the_period_four_fill_over_four_columns_is_flagged(self):
        """Exactly the old failure: periodic in the FLAT index, so every row is identical."""
        vals = [i % 4 for i in range(8 * 4)]
        found = stimulus_rigor_findings(vals, (8, 4), alphabet=4)
        assert found, "1 distinct row of 8 achievable must be flagged"
        assert "row collapse" in found[0] and "alphabet allows 8" in found[0]

    def test_a_constant_operand_is_flagged(self):
        found = stimulus_rigor_findings([2] * 64, (8, 8), alphabet=4)
        assert any("constant" in f for f in found), found

    def test_a_symmetric_square_operand_is_flagged(self):
        grid = [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
        vals = [v for row in grid for v in row]
        found = stimulus_rigor_findings(vals, (3, 3), alphabet=4)
        assert any("symmetric" in f for f in found), found


class TestItDoesNotBlameArithmetic:
    """Every one of these is a shape the alphabet makes impossible to satisfy exactly."""

    def test_a_single_row_operand_wider_than_its_alphabet(self):
        assert stimulus_rigor_findings(ST.fill("G", (1, 16)), (1, 16), alphabet=4) == []

    def test_a_single_column_operand_deeper_than_its_alphabet(self):
        assert stimulus_rigor_findings(ST.fill("W", (64, 1)), (64, 1), alphabet=4) == []

    def test_a_one_by_one_operand_is_not_symmetric_in_any_meaningful_sense(self):
        assert stimulus_rigor_findings([1], (1, 1), alphabet=4) == []

    def test_birthday_collisions_in_a_wide_operand_are_not_a_collapse(self):
        """1017 distinct of 1024 is a hash colliding, not a fill failing."""
        shape = (1024, 8)
        assert stimulus_rigor_findings(ST.fill("W", shape), shape, alphabet=4) == []

    def test_a_batched_rank_three_operand_is_checked_not_skipped(self):
        shape = (2, 16, 1)
        assert stimulus_rigor_findings(ST.fill("V", shape), shape, alphabet=4) == []
        # and it really was compared: a collapsed version of the same shape is caught
        assert stimulus_rigor_findings([1] * 32, shape, alphabet=4)


class TestUndeterminableIsNotAPass:
    def test_a_shape_mismatch_is_a_finding_in_both_checks(self):
        a = stimulus_rigor_findings([1, 2, 3], (4, 4), alphabet=4)
        b = rigor_findings([1.0, 2.0, 3.0], (4, 4))
        for found in (a, b):
            assert found and "UNDETERMINABLE" in found[0]
            assert "not a clean bill of health" in found[0]

    def test_the_strict_check_no_longer_raises(self):
        """It raised IndexError, which inside a try/except becomes 'no findings' == rigorous."""
        rigor_findings([], (0, 0))          # must not raise
        rigor_findings([1.0], (3, 3))


class TestTheFloor:
    def test_the_floor_is_a_fraction_of_achievable_not_of_the_extent(self):
        """32 rows of 2 columns over a 4-symbol alphabet: 16 achievable, so the floor is 8 -- not 16."""
        assert MIN_ACHIEVED_FRACTION == 0.5
        rows = [[a, b] for a in range(4) for b in range(4)] * 2      # 32 rows, all 16 combos twice
        vals = [v for r in rows for v in r]
        assert stimulus_rigor_findings(vals, (32, 2), alphabet=4) == [], "16 of 16 achievable is rigorous"
        few = [[0, 0], [0, 1]] * 16                                   # 32 rows, 2 distinct of 16
        vals2 = [v for r in few for v in r]
        assert stimulus_rigor_findings(vals2, (32, 2), alphabet=4), "2 of 16 achievable is a collapse"
