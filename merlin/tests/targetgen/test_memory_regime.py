"""The memory-mapping axis: which regime a program puts the on-chip operand store in.

The measured motivation. Against the operand store derived for the interlocked target here (16384 rows
of 16 bytes): 90.1% of 1829 contraction regions across 20 real captures exceed it however they are
allocated, while 100% of that target's 37 public capsules fit it TWICE over and the largest uses 2.34%
of capacity. The corpus exercises the rare regime exclusively and the common one never.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import memory_regime as MR


class TestClassification:
    @pytest.mark.parametrize("live,total,cap,regime", [
        (10, 10, 100, MR.FITS_DOUBLE),      # fits twice: staging is possible, so not staging is a defect
        (60, 60, 100, MR.FITS_SINGLE),      # fits once: staging is impossible, so serialising is correct
        (60, 300, 100, MR.FITS_ON_REUSE),   # only an allocator that reuses freed rows works
        (300, 300, 100, MR.SPILLS),         # exceeds capacity however it is allocated
        (50, 50, 100, MR.FITS_DOUBLE),      # exactly half still fits twice
        (51, 51, 100, MR.FITS_SINGLE),
    ])
    def test_each_regime_is_named(self, live, total, cap, regime):
        assert MR.classify(live, total, cap) == regime

    def test_an_underivable_capacity_is_unknown_not_a_fit(self):
        # An unmeasurable capacity reported as satisfied is exactly how a graded backend reached the
        # simulator with nothing recorded and aborted three layers away in a range check.
        assert MR.classify(10, 10, None) == MR.UNKNOWN
        assert MR.classify(10, 10, 0) == MR.UNKNOWN

    def test_an_unsized_program_is_unknown_not_empty(self):
        assert MR.classify(None, None, 100) == MR.UNKNOWN

    def test_reuse_is_only_claimed_when_the_totals_actually_differ(self):
        # If peak-live equals total there is nothing for an allocator to reuse, so demanding reuse would
        # be an obligation no program in that shape could ever fail.
        assert MR.classify(60, 60, 100) == MR.FITS_SINGLE


class TestGapReport:
    def test_a_regime_real_models_occupy_and_no_capsule_reaches_is_reported(self):
        req = {"by_regime": {MR.FITS_DOUBLE: ["m"], MR.SPILLS: ["m"], MR.FITS_SINGLE: ["m"]}}
        corpus = {"by_regime": {MR.FITS_DOUBLE: ["c0"]}, "capacity_rows": 16384,
                  "largest_working_set": {"name": "c0", "rows": 384, "fraction_of_capacity": 0.023}}
        gap = MR.uncovered_regimes(req, corpus)
        assert gap["uncovered"] == [MR.FITS_SINGLE, MR.SPILLS]     # reported weakest-demand first
        assert gap["n_covered"] == 1 and gap["n_required"] == 3

    def test_unknown_is_never_a_requirement_and_never_a_coverage(self):
        # UNKNOWN means we could not tell. Demanding it would make a corpus chase an unmeasurable cell;
        # crediting it would let an unmeasurable capsule discharge a real one.
        req = {"by_regime": {MR.UNKNOWN: ["m"], MR.SPILLS: ["m"]}}
        corpus = {"by_regime": {MR.UNKNOWN: ["c"]}}
        gap = MR.uncovered_regimes(req, corpus)
        assert gap["n_required"] == 1
        assert gap["uncovered"] == [MR.SPILLS]

    def test_a_corpus_that_covers_everything_reports_no_gap(self):
        req = {"by_regime": {MR.FITS_DOUBLE: ["m"]}}
        corpus = {"by_regime": {MR.FITS_DOUBLE: ["c"], MR.SPILLS: ["c2"]}}
        assert MR.uncovered_regimes(req, corpus)["uncovered"] == []


class TestOperandStoreSelection:
    def test_the_operand_store_is_the_narrowest_row_not_a_named_one(self):
        # A separate accumulator space exists precisely BECAUSE its row is wider (it holds the
        # accumulate type). Selecting by name would bake one target's spelling into shared code, which
        # the no-target-name gate rightly rejects.
        store, cap = MR.operand_store("gemmini")
        if store is None:
            pytest.skip("this target declares no derivable operand store")
        assert store.row_bytes == min(
            s.row_bytes for s in __import__(
                "merlin.targetgen.address_space", fromlist=["x"]
            ).derive_address_space("gemmini").stores if s.row_bytes)
        assert cap and cap > 0

    def test_an_unresolvable_target_yields_no_store_rather_than_a_default(self):
        store, cap = MR.operand_store("definitely_not_a_target")
        assert store is None and cap is None

    def test_a_target_with_no_store_requires_no_regime_and_says_why(self):
        got = MR.required_regimes({}, "definitely_not_a_target")
        assert got["by_regime"] == {}
        assert "we do not know" in got.get("why", "")
