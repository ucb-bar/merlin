"""A numeric failure must record WHERE it diverged, not only how many elements did.

A count plus one index cannot distinguish a CLUSTERED failure (a whole row/column/tile wrong -- a scale,
stride or tail-handling bug) from a SCATTERED one (a few elements a couple of ULP out -- rounding or
accumulation order). Those need completely different fixes, so reading the count alone leaves the
diagnosis a guess.

Measured on a real capsule: 8 of 256 elements diverged by exactly 3/32 and 5/32 -- even multiples of the
bf16 ULP and sub-ULP in the operand format -- and nothing in the record said whether those 8 shared a row.

Target-agnostic: the comparison is over flat element lists and knows nothing about any device.
"""
from __future__ import annotations

from merlin.targetgen.capsule_golden import _MISMATCH_INDEX_CAP, compare

N = 16
POLICY = {"compare": "tolerance_float", "atol": 0.0, "rtol": 0.0}
BASE = [float(i % 7) + 1.0 for i in range(N * N)]
DELTA = 0.09375                                   # the measured 3/32 divergence


def _run(bad_positions):
    obs = list(BASE)
    for i in bad_positions:
        obs[i] += DELTA
    return compare({"Y0": BASE}, {"Y0": obs}, POLICY)["per_output"]["Y0"]


def test_a_clustered_failure_is_visibly_one_row():
    po = _run(range(2 * N, 2 * N + 8))
    assert po["mismatch_count"] == 8
    rows = {i // N for i in po["mismatch_indices"]}
    assert rows == {2}, f"a single-row failure should show one row, got {sorted(rows)}"


def test_a_scattered_failure_is_visibly_many_rows():
    """Same COUNT as the clustered case — only the indices separate them."""
    po = _run([5, 23, 47, 88, 130, 171, 202, 250])
    assert po["mismatch_count"] == 8
    rows = {i // N for i in po["mismatch_indices"]}
    assert len(rows) > 3, f"a scattered failure should span rows, got {sorted(rows)}"


def test_a_passing_output_records_no_index_list():
    """No noise on the common path."""
    po = compare({"Y0": BASE}, {"Y0": list(BASE)}, POLICY)["per_output"]["Y0"]
    assert po["status"] == "pass"
    assert "mismatch_indices" not in po


def test_the_list_is_bounded_and_says_so():
    """A saturated output must not balloon the record, and a partial list must announce itself."""
    po = compare({"Y0": BASE}, {"Y0": [v + 1.0 for v in BASE]}, POLICY)["per_output"]["Y0"]
    assert len(po["mismatch_indices"]) == _MISMATCH_INDEX_CAP
    assert po["mismatch_indices_truncated"] is True


def test_the_true_total_is_never_the_capped_length():
    """`mismatch_count` stays exact — the cap is a reporting bound, not a measurement bound."""
    po = compare({"Y0": BASE}, {"Y0": [v + 1.0 for v in BASE]}, POLICY)["per_output"]["Y0"]
    assert po["mismatch_count"] == N * N > _MISMATCH_INDEX_CAP


def test_indices_are_recorded_for_the_integer_policy_too():
    """The gate has two modes; a diagnostic that only works for floats helps half the targets."""
    exp = [1, 2, 3, 4]
    obs = [1, 9, 3, 9]
    po = compare({"Y0": exp}, {"Y0": obs}, {"compare": "exact_int"})["per_output"]["Y0"]
    assert po["mismatch_indices"] == [1, 3]
