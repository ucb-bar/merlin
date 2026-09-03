"""The affine reduction-depth claim is REFUTED, and this pins WHY rather than only THAT.

Every number below is a measurement, not a fixture chosen to make an assertion land. The cycle counts
are the citable Verilator L3 counts of the four frozen ``PK`` capsules -- one interlocked
command-driven accelerator, ``matmul`` at fixed single-tile M and N, reduction depth doubling from one
tile to eight. The counter readings are what those same bracketed runs printed through
``MERLIN_HWCOUNTER``. Provenance: merlin ``06dd9d56dab0947e0c1e571c22e624068f178e3c``, RTL
``8c3f9923a44a2fe2c7930587be297d6d4f8c09ca``, oracle ``verilator`` (``derived_from_rtl``).

⚠️ **THE REFUTATION IS THRESHOLD-FREE, and that is the point of this file.** The frozen acceptance
block says in its own comment that changing any of its values is a new experiment contract, so the
tempting repair -- widen ``r_squared_min_inclusive`` until 0.977 passes -- is forbidden. It is also
WRONG, and these tests are what make that checkable: the marginal cost per unit of the fit axis falls
strictly at every step, and an affine law asserts that it is constant. No tolerance repairs a
contradicted form. What the overlap counters add is the MECHANISM: realised overlap is still rising at
the deepest depth measured, so no point in the cohort priced a settled machine.

The second half pins the road not taken. A successor family whose model is affine-plus-overlap was the
other candidate resolution once the hardware began exposing realised overlap. Fitted honestly on this
same evidence, its two natural spellings each miss a DIFFERENT half of the same frozen contract --
overlap cycles miss r^2, the overlap ratio misses the residual bound -- and both give the overlap term
a coefficient whose sign reverses the mechanism it was introduced to express. With four design points
and a regressor nearly collinear with the axis, those coefficients are not identifiable. The option is
closed here with arithmetic rather than with an opinion.
"""
from __future__ import annotations

from fractions import Fraction

from merlin.perf import fill_transient as FT
from merlin.perf import hw_counters as HC

#: The counter block this target's own shipped header declares, reduced to the ``#define`` lines the
#: derivation reads. Stated here rather than read from an external checkout so the test says what it
#: measured on; the ENGINE set is still derived from this text by the module under test, never typed.
COUNTER_HEADER = """
#define MAIN_LD_CYCLES 1
#define MAIN_ST_CYCLES 2
#define MAIN_EX_CYCLES 3
#define MAIN_LD_ST_CYCLES 4
#define MAIN_LD_EX_CYCLES 5
#define MAIN_ST_EX_CYCLES 6
#define MAIN_LD_ST_EX_CYCLES 7
"""

#: capsule -> (fit-axis value, measured Verilator L3 cycles, the counter readings that run printed)
MEASURED = {
    "PK00_k16": (16, 301, {"MAIN_EX_CYCLES": 70, "MAIN_LD_CYCLES": 39, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 28, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK01_k32": (32, 373, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 46, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 51, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK02_k64": (64, 471, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 43, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 128, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK03_k128": (128, 604, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 42, "MAIN_ST_CYCLES": 43,
                             "MAIN_LD_EX_CYCLES": 281, "MAIN_ST_EX_CYCLES": 0,
                             "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
}

#: The frozen contract's own two bounds, restated so a test that quotes them is greppable from either
#: side. They are NEVER modified here; every assertion below holds them fixed and moves nothing.
R_SQUARED_MIN = Fraction(995, 1000)
RESIDUAL_FLOOR_CYCLES = Fraction(8)
RESIDUAL_FRACTION_OF_OBSERVED = Fraction(3, 100)


def counters():
    return HC.derive_occupancy_counters(COUNTER_HEADER)


def readings() -> dict:
    return {name: dict(values) for name, (_axis, _cycles, values) in MEASURED.items()}


def points() -> list[FT.Point]:
    derived = counters()
    return [FT.point_from_counter_values(name, axis, cycles, values, derived)
            for name, (axis, cycles, values) in MEASURED.items()]


def _bound(observed: int) -> Fraction:
    return max(RESIDUAL_FLOOR_CYCLES, RESIDUAL_FRACTION_OF_OBSERVED * observed)


def _affine_fit(xs, ys):
    """The claim analyzer's own ordinary least squares, in exact rationals."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    slope = Fraction(n * sum(x * y for x, y in zip(xs, ys, strict=True)) - sx * sy,
                     n * sum(x * x for x in xs) - sx * sx)
    intercept = Fraction(sy, n) - slope * Fraction(sx, n)
    residuals = [Fraction(y) - (slope * x + intercept) for x, y in zip(xs, ys, strict=True)]
    mean = Fraction(sy, n)
    ss_res = sum((r * r for r in residuals), Fraction())
    ss_tot = sum(((Fraction(y) - mean) ** 2 for y in ys), Fraction())
    return slope, intercept, residuals, Fraction(1) - ss_res / ss_tot


def _ols(columns, ys):
    """Exact multivariate least squares through the normal equations. Small by design."""
    n = len(ys)
    design = [[Fraction(col[i]) for col in columns] + [Fraction(1)] for i in range(n)]
    width = len(columns) + 1
    matrix = [[sum(design[i][r] * design[i][s] for i in range(n)) for s in range(width)]
              for r in range(width)]
    rhs = [sum(design[i][r] * Fraction(ys[i]) for i in range(n)) for r in range(width)]
    for r in range(width):
        pivot = next(k for k in range(r, width) if matrix[k][r] != 0)
        matrix[r], matrix[pivot] = matrix[pivot], matrix[r]
        rhs[r], rhs[pivot] = rhs[pivot], rhs[r]
        divisor = matrix[r][r]
        matrix[r] = [v / divisor for v in matrix[r]]
        rhs[r] = rhs[r] / divisor
        for k in range(width):
            if k != r and matrix[k][r] != 0:
                factor = matrix[k][r]
                matrix[k] = [matrix[k][j] - factor * matrix[r][j] for j in range(width)]
                rhs[k] = rhs[k] - factor * rhs[r]
    beta = rhs
    predicted = [sum(beta[j] * design[i][j] for j in range(width)) for i in range(n)]
    residuals = [Fraction(ys[i]) - predicted[i] for i in range(n)]
    mean = Fraction(sum(ys), n)
    ss_res = sum((r * r for r in residuals), Fraction())
    ss_tot = sum(((Fraction(y) - mean) ** 2 for y in ys), Fraction())
    return beta, residuals, Fraction(1) - ss_res / ss_tot


# ---------------------------------------------------------------------------------------------------
# what the frozen contract decides on this evidence
# ---------------------------------------------------------------------------------------------------


def test_the_frozen_affine_contract_is_refuted_on_the_measured_cohort():
    """Reproduces the recorded verdict, so the mechanism below has something to explain."""
    xs = [axis for axis, _c, _v in MEASURED.values()]
    ys = [cycles for _a, cycles, _v in MEASURED.values()]
    _slope, _intercept, residuals, r_squared = _affine_fit(xs, ys)

    assert r_squared < R_SQUARED_MIN, f"r^2 is {float(r_squared)}"
    missed = [y for y, r in zip(ys, residuals, strict=True) if abs(r) > _bound(y)]
    assert len(missed) == 2, (
        f"the recorded refutation is two residuals past their predeclared bound; got {len(missed)}")


# ---------------------------------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------------------------------


def test_the_marginal_cost_falls_at_every_step_so_the_affine_FORM_is_contradicted():
    """The finding that decides what the repair is -- and it involves no threshold at all."""
    marginals = FT.marginal_costs(points())
    rates = [Fraction(m["exact_numerator"], m["exact_denominator"]) for m in marginals]

    assert len(rates) == 3
    assert all(later < earlier for earlier, later in zip(rates, rates[1:], strict=False)), rates
    # Per 16-wide tile these deltas are 72, 49 and 33.25 cycles: one extra tile of reduction depth
    # costs less than half at eight tiles what it cost at two.
    assert [m["delta_cycles"] for m in marginals] == [72, 98, 133]


def test_realised_overlap_is_still_rising_at_the_deepest_declared_depth():
    """So no point in the cohort priced a settled machine, and no two priced the same one."""
    trend = FT.overlap_trend(points())

    assert trend["state"] == "measured"
    assert trend["monotonically_rising"] is True
    assert trend["still_rising_at_deepest_point"] is True
    etas = [row["eta"] for row in trend["eta_by_point"]]
    assert etas == sorted(etas)
    assert etas[0] < 0.3 < etas[-1] < 1.0, etas


def test_the_cohort_verdict_is_in_fill_transient():
    verdict = FT.transient_verdict(points())

    assert verdict["state"] == FT.IN_FILL_TRANSIENT
    assert verdict["affine_form_contradicted"] is True


def test_a_point_with_no_overlap_reading_is_unknown_and_not_zero_overlap():
    """The recurring bug class in this package: the unmeasured reported as the measured-and-zero."""
    partial = points()
    partial[2] = FT.Point(label=partial[2].label, axis=partial[2].axis, cycles=partial[2].cycles,
                          overlap_detail="the bracket did not run for this point")
    verdict = FT.transient_verdict(partial)

    assert verdict["state"] == FT.UNDETERMINABLE
    assert verdict["overlap"]["unread"] == [partial[2].label]
    # The arithmetic still stands on its own: the marginals never needed an overlap reading.
    assert verdict["affine_form_contradicted"] is True


def test_a_saturated_cohort_is_not_reported_as_a_transient():
    """The verdict must be able to come out the other way, or it is not measuring anything."""
    derived = counters()
    settled = [
        FT.point_from_counter_values("A", 16, 300, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 40,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
        FT.point_from_counter_values("B", 32, 340, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 80,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
        FT.point_from_counter_values("C", 64, 420, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 80,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
    ]
    verdict = FT.transient_verdict(settled)

    assert verdict["overlap"]["still_rising_at_deepest_point"] is False
    assert verdict["state"] == FT.SATURATED
    # Constant marginal cost, so the affine form is not contradicted either.
    assert verdict["affine_form_contradicted"] is False


def test_overlap_that_falls_somewhere_refuses_rather_than_calling_the_cohort_settled():
    derived = counters()
    unordered = [
        FT.point_from_counter_values("A", 16, 300, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 80,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
        FT.point_from_counter_values("B", 32, 340, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 20,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
        FT.point_from_counter_values("C", 64, 420, {"MAIN_EX_CYCLES": 60, "MAIN_LD_CYCLES": 60,
                                                    "MAIN_ST_CYCLES": 0, "MAIN_LD_EX_CYCLES": 90,
                                                    "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_ST_CYCLES": 0,
                                                    "MAIN_LD_ST_EX_CYCLES": 0}, derived),
    ]
    verdict = FT.transient_verdict(unordered)

    assert verdict["overlap"]["monotonically_rising"] is False
    assert verdict["state"] == FT.UNDETERMINABLE


# ---------------------------------------------------------------------------------------------------
# the successor that was NOT minted, and the evidence for not minting it
# ---------------------------------------------------------------------------------------------------


def test_an_overlap_term_does_not_rescue_the_claim_and_comes_out_wrong_signed():
    """``cycles = rate*axis + weight*overlap + intercept`` -- the obvious successor, measured.

    It still misses the frozen r^2 bound, still leaves a residual past its bound, and its overlap
    coefficient is NEGATIVE while its rate more than triples. That is not a mechanism; it is two
    nearly-collinear regressors sharing one slope between four design points.
    """
    xs = [axis for axis, _c, _v in MEASURED.values()]
    ys = [cycles for _a, cycles, _v in MEASURED.values()]
    overlaps = [p.realised_overlap for p in points()]

    (rate, weight, _intercept), residuals, r_squared = _ols([xs, overlaps], ys)

    assert r_squared < R_SQUARED_MIN, (
        f"the overlap-term model reaches r^2 {float(r_squared)}; if it ever clears the frozen bound "
        "this test is the place to reopen the successor question")
    assert any(abs(r) > _bound(y) for r, y in zip(residuals, ys, strict=True))
    assert weight < 0, (
        "the fitted overlap coefficient is negative -- each measured overlap cycle would ADD to the "
        "predicted cost, reversing the mechanism the term was introduced to express")
    affine_rate, _i, _r, _r2 = _affine_fit(xs, ys)
    assert rate > 3 * affine_rate, (
        "and the per-axis-unit rate more than triples against the plain affine fit, which is what "
        "unidentifiable coefficients look like")


def test_regressing_on_eta_moves_which_threshold_is_missed_and_reverses_the_mechanism():
    """Why "it fits better" cannot be the test for admitting a successor.

    Swapping the overlap CYCLES for the overlap RATIO is the same idea in different units, and lands
    somewhere else entirely: it CLEARS the global r^2 bound the cycles spelling misses, then misses
    the local residual bound the cycles spelling clears. Two spellings of one successor, each failing
    a different half of the same contract -- which is what an unidentifiable model looks like from
    outside.

    And its eta coefficient is large and POSITIVE: "the more the engines overlapped, the more it
    cost". No fit statistic distinguishes that from the mechanism, so no fit statistic may be what
    admits a successor. This is the decisive argument against versioning one off this cohort.
    """
    xs = [axis for axis, _c, _v in MEASURED.values()]
    ys = [cycles for _a, cycles, _v in MEASURED.values()]
    etas = [float(p.eta) for p in points()]

    (_rate, eta_weight, _intercept), residuals, r_squared = _ols([xs, etas], ys)

    assert r_squared >= R_SQUARED_MIN, float(r_squared)
    assert any(abs(r) > _bound(y) for r, y in zip(residuals, ys, strict=True)), (
        "the eta spelling clears r^2 and must still miss the residual bound; if it clears both, this "
        "test is the place to reopen the successor question")
    assert eta_weight > 0, (
        "the coefficient says overlap ADDS cycles, reversing the mechanism the term exists to "
        "express, while the fit statistic improves")
