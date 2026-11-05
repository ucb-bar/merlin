"""The label-free design check applies exactly the fit's identifiability gates."""

import pytest
from merlin_timing import contracts as C
from merlin_timing import fitting as F


def _rows(n, slope_rows):
    return [[1.0, float(i + 1), float((i + 1) * slope_rows + (i % 3))] for i in range(n)]


def test_design_check_accepts_what_the_fit_accepts_and_rejects_what_it_rejects():
    good = _rows(8, 4)
    verdict = F.design_check(good, maximum_condition=1e6)
    assert verdict["identifiable"] is True
    labels = [10.0 + 2.0 * row[1] + 0.5 * row[2] for row in good]
    F.nnls(good, labels, maximum_condition=1e6)  # the fit agrees

    collinear = [[1.0, float(i + 1), float(2 * (i + 1))] for i in range(8)]
    verdict = F.design_check(collinear, maximum_condition=1e6)
    assert verdict["identifiable"] is False and "rank deficient" in verdict["reason"]
    with pytest.raises(C.ContractError, match="rank deficient"):
        F.nnls(
            [[v / max(1.0, max(r[i] for r in collinear)) for i, v in enumerate(row)] for row in collinear],
            [1.0] * 8,
            maximum_condition=1e6,
        )


def test_design_check_requires_two_distinct_points_per_parameter():
    few = _rows(5, 4)  # 5 distinct rows < 2 * 3 parameters
    verdict = F.design_check(few, maximum_condition=1e6)
    assert verdict == {"identifiable": False, "reason": "fit requires two distinct points per parameter", "rows": 5}


def test_design_check_refuses_a_malformed_matrix():
    with pytest.raises(C.ContractError):
        F.design_check([], maximum_condition=1e6)
    with pytest.raises(C.ContractError):
        F.design_check([[1.0, 2.0], [1.0]], maximum_condition=1e6)
