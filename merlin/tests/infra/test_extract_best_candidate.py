"""The shipped compiler must be the best one that was MEASURED, not the last one that was edited.

The stage seals the agent's final tree and grades nothing, so which compiler gets shipped is decided
by where the agent happened to stop. Measured over six completed trials: in one, the sealed tree was
never measured at all -- it was an edit made after the last measurement, so its speed is unknown.

Three ways of getting "best" wrong are pinned here, because each produces a confident wrong answer:
a tie reported as a winner by recency, a short sweep's smaller total read as faster, and a cell the
sweep never paid for read as zero cycles.
"""
from __future__ import annotations

import sys

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import extract_best_candidate as EX  # noqa: E402


def _cell(capsule, base, cand, *, measured=True, comparable=True):
    return {"family": "PK", "capsule": capsule, "measured": measured, "comparable": comparable,
            "baseline_gsim_cycles": base if measured else None,
            "candidate_gsim_cycles": cand if measured else None}


def _row(call, cells, sha):
    cs = EX._measured_cells({"cells": cells})
    return {"document": f"d{call}", "round": 0, "call": call, "candidate_sha256": sha,
            "members": EX._member_key(cs), "n_members": len(cs),
            "baseline_total_cycles": sum(c["baseline_gsim_cycles"] for c in cs),
            "candidate_total_cycles": sum(c["candidate_gsim_cycles"] for c in cs),
            "snapshot": None}


def test_a_cell_the_sweep_never_paid_for_is_not_a_zero():
    cells = [_cell("a", 100, 90), _cell("b", 100, 0, measured=False)]
    kept = EX._measured_cells({"cells": cells})
    assert [c["capsule"] for c in kept] == ["a"], "an unmeasured cell must not enter a total"


def test_a_document_predating_the_measured_field_still_counts():
    """Absence of the key means the schema predates the early stop, not that nothing ran."""
    old = [{"family": "PK", "capsule": "a", "comparable": True,
            "baseline_gsim_cycles": 100, "candidate_gsim_cycles": 90}]
    assert len(EX._measured_cells({"cells": old})) == 1


def test_a_short_sweep_is_never_compared_against_a_full_one():
    """Its total is smaller because it measured less, which has nothing to do with being faster."""
    full_a = _row(1, [_cell("a", 100, 95), _cell("b", 100, 95)], "sha_a")
    full_b = _row(2, [_cell("a", 100, 90), _cell("b", 100, 90)], "sha_b")
    short = _row(3, [_cell("a", 100, 99)], "sha_short")
    verdict = EX.choose_best([full_a, full_b, short])
    assert verdict["best_total_cycles"] == 180, "the winner must come from the full-sweep cohort"
    assert verdict["winners"][0]["candidate_sha256"] == "sha_b"
    assert len(verdict["excluded_from_comparison"]) == 1
    assert verdict["excluded_from_comparison"][0]["call"] == 3


def test_a_tie_is_reported_as_a_tie_and_not_broken_by_recency():
    first = _row(1, [_cell("a", 100, 90)], "sha_first")
    later = _row(9, [_cell("a", 100, 90)], "sha_later")
    verdict = EX.choose_best([first, later])
    assert verdict["status"] == "tie"
    assert {w["candidate_sha256"] for w in verdict["winners"]} == {"sha_first", "sha_later"}


def test_no_measurement_is_said_rather_than_guessed():
    assert EX.choose_best([])["status"] == "no_measurement"
