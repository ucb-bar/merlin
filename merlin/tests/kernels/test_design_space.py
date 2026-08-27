"""The corpus as a search space, and the ledger of what did NOT work.

A mined corpus reads as a set of good kernels. It is also a factorial design somebody already ran, and
the interesting half is what they did not run. Separately, a loop that sees only winners over-proposes:
the corpus records what an expert KEPT, and a transform ledger is the only record of what was tried and
discarded.
"""
from __future__ import annotations

import pytest

from merlin.kernels import space as S
from merlin.kernels.ingest import autocomp as A

_REPO = "/scratch/agustin/projects/autocomp"

# The measured shape of a real sweep: deliberately IRREGULAR, which is the normal case.
_SWEEP = ([{"m": 32, "n": 32, "k": k} for k in (32, 64, 128, 256)]
          + [{"m": 64, "n": 64, "k": k} for k in (32, 64, 128)]
          + [{"m": 96, "n": 96, "k": 96}])


class TestTheDesignIsDerivedNotDeclared:
    def test_axes_and_levels_come_from_the_records(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        assert sp.axes["m"] == (32, 64, 96)
        assert sp.axes["k"] == (32, 64, 96, 128, 256)

    def test_a_constant_axis_is_dropped_with_a_reason(self):
        """A constant is not a dimension. Leaving it in inflates the cross product, so coverage reads
        far worse than it is — and the note is what stops that being read as missing data."""
        sp = S.space_from_records([{"m": 32, "dtype": "int8"}, {"m": 64, "dtype": "int8"}],
                                  axes=("m", "dtype"), target="t")
        assert "dtype" not in sp.axes
        assert any("constant" in n for n in sp.notes)

    def test_an_absent_axis_is_reported_rather_than_invented(self):
        sp = S.space_from_records([{"m": 1}], axes=("m", "nope"), target="t")
        assert any("absent from every record" in n for n in sp.notes)


class TestTheUnobservedCellsAreTheProduct:
    def test_an_irregular_sample_yields_the_cells_nobody_tried(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        assert len(sp.observed) == 8
        assert len(sp.unobserved()) == 3 * 3 * 5 - 8

    def test_the_unobserved_list_is_complete_not_truncated(self):
        """A top-N list presented as the whole space is how a sweep gets reported as covered when most
        of it was never run."""
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        d = sp.to_dict()
        assert len(d["unobserved"]) == d["n_cells_unobserved"]

    def test_candidates_are_ordered_nearest_to_something_already_run(self):
        # A cell one step from a known point is a cheaper and more interpretable probe than one that
        # varies every axis at once.
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        cands = S.candidates(sp)
        steps = [c["steps_from_observed"] for c in cands]
        assert steps == sorted(steps)

    def test_a_truncated_candidate_list_says_how_many_it_dropped(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        cands = S.candidates(sp, limit=3)
        assert "more unobserved cells not listed" in cands[-1]["cell"]

    def test_coverage_is_reported_and_low_is_informative(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        assert 0.0 < sp.coverage < 0.25


class TestTheLedgerRecordsWhatFailed:
    def _have(self):
        from pathlib import Path
        if not (Path(_REPO) / "output" / "transform_ledger.jsonl").is_file():
            pytest.skip("autocomp ledger not present in this checkout")

    def test_the_base_rate_is_measurable(self):
        """The prior a proposal loop needs. Without it the loop keeps re-proposing transforms somebody
        already refuted."""
        self._have()
        s = A.ledger_summary(_REPO)
        assert s["total"] > 1000
        assert 0.0 < s["improvement_rate"] < 0.5, s["improvement_rate"]

    def test_every_outcome_value_is_classified(self):
        """An outcome vocabulary that grows without the map noticing would silently shrink the
        denominator, so an unknown value is counted separately rather than folded into failures."""
        self._have()
        assert A.ledger_summary(_REPO)["unclassified"] == 0

    def test_failures_outnumber_wins(self):
        self._have()
        c = A.ledger_summary(_REPO)["outcomes"]
        assert sum(v for k, v in c.items() if k != "improved") > c.get("improved", 0)

    def test_rows_map_onto_the_search_record(self):
        self._have()
        steps = list(A.ledger_search_steps(_REPO, target="muon"))
        assert len(steps) == A.ledger_summary(_REPO)["total"]

    def test_no_ledger_row_claims_its_promise_was_audited(self):
        """`achieved` means the emitted asm delivered the action's intended facet. The ledger records an
        outcome, not that audit, and claiming otherwise would fabricate the one check the whole search
        discipline rests on."""
        self._have()
        assert not any(s.achieved for s in A.ledger_search_steps(_REPO, target="muon"))

    def test_speedup_is_never_credited_without_correctness(self):
        self._have()
        bad = [s for s in A.ledger_search_steps(_REPO, target="muon")
               if s.speedup is not None and not s.correctness_ok]
        assert bad == [], f"{len(bad)} rows credited a speedup for a broken attempt"

    def test_a_missing_ledger_yields_nothing_rather_than_raising(self):
        assert list(A.ledger_rows("/no/such/repo")) == []
        assert A.ledger_summary("/no/such/repo")["total"] == 0


class TestAttemptsFoldIntoTheDesign:
    def test_a_tried_and_rejected_cell_is_not_the_same_as_never_tried(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        sp = S.merge_ledger(sp, [{"cell": "k=256 m=64 n=64", "outcome": "regressed"}])
        cell = sp.cells["k=256 m=64 n=64"]
        assert cell.observed is False and cell.regressed == 1
        assert cell.attempted == 1

    def test_outcomes_accumulate_on_an_observed_cell(self):
        sp = S.space_from_records(_SWEEP, axes=("m", "n", "k"), target="t")
        key = sorted(c.key for c in sp.observed)[0]
        sp = S.merge_ledger(sp, [{"cell": key, "outcome": "improved"},
                                 {"cell": key, "outcome": "compile_error"}])
        assert sp.cells[key].improved == 1 and sp.cells[key].failed == 1
        assert sp.cells[key].observed is True, "merging attempts must not un-observe a kept cell"
