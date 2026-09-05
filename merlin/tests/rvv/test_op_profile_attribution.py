"""The per-op profiler's ARITHMETIC: what it divides by, what it refuses to say, how it buckets.

These are the three defects the whole-model profile shipped with, and each one produced a number
that was quoted:

1. It divided the accumulated ticks by ``--iters`` while the shim had accumulated them over
   ``warmup + iters`` executions of ``@forward``. Measured on ``small_llama_int8_consistent`` with
   ``--warmup 2 --iters 5``: every one of 996 ops reported ``hits=7``, the divisor was 5, and the
   attributed total came back 6.0 ms against a 4.2 ms wall (``profiler_cov=1.4407``, and
   1.4407 / (7/5) = 1.029). The perturbation guard was CLEAN through all of it.
2. It printed those inflated numbers as percentages of runtime with no gate.
3. It bucketed families on ``prov.family``, a tag the quantization and blocking rewrites drop, so
   776 of the 996 ops fell into ``(none)`` and the summary announced ``contraction = 0.0 ms`` for a
   model whose matmuls measured ~1 ms.

Board-free: everything below is the pure arithmetic in ``merlin.llvmlower.op_profile``.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import op_profile as opf

TIMEBASE_HZ = 24_000_000.0


def _rec(oid, mlir_op, ticks, hits, family=None):
    return {"id": oid, "mlir_op": mlir_op, "ticks": ticks, "hits": hits, "family": family}


# =================================================================================================
# 1. The divisor is the MEASURED execution count, not the requested iteration count.
# =================================================================================================

def test_per_execution_divides_by_the_measured_hits_not_the_requested_iters():
    # 700 ticks accumulated over 7 executions is 100 ticks per execution -- whatever the CLI asked
    # for. Dividing by a requested `iters=5` would say 140, a 1.4x over-credit.
    assert opf.per_execution_ticks(700, 7) == 100
    assert opf.per_execution_ticks(700, 5) == 140  # what the old code effectively did


def test_an_op_that_never_ran_is_unknown_not_zero():
    """0 hits is 'not measured', which must not enter a sum as a measured 0.0."""
    assert opf.per_execution_ticks(0, 0) is None
    table = [_rec(0, "linalg.matmul", 700, 7), _rec(1, "linalg.transpose", 0, 0)]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    assert table[1]["ticks_avg"] is None and table[1]["ms_avg"] is None
    assert opf.sum_attributed_ticks(table) == 100


def test_the_warmup_ratio_reproduces_the_measured_144_percent():
    """The whole 144 % is the window mismatch -- reproduce it, then show the fix removes it."""
    wall_ticks = 100_964.0                       # the harness's per-timed-iteration wall
    table = [_rec(0, "linalg.matmul", 722_604, 7)]   # the artifact's total, over 7 executions
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    good = opf.coverage_report(opf.sum_attributed_ticks(table), wall_ticks,
                               executions=7, timed_iterations=5)
    bad_ticks = 722_604 / 5                      # the old divisor
    bad = opf.coverage_report(bad_ticks, wall_ticks, executions=7, timed_iterations=5)
    assert bad["profiler_coverage"] == pytest.approx(1.44, abs=0.01)
    assert good["profiler_coverage"] == pytest.approx(1.02, abs=0.01)
    # the inflation is EXACTLY the ratio of the two windows
    assert bad["profiler_coverage"] / good["profiler_coverage"] == pytest.approx(7 / 5, rel=1e-3)
    assert bad["runtime_shares_reportable"] is False
    assert good["runtime_shares_reportable"] is True
    # and the wider-than-timed window is named even when the coverage passes
    assert good["executions_exceed_timed_iterations"] is True


# =================================================================================================
# 2. Coverage outside the band refuses to express anything as a percentage of RUNTIME.
# =================================================================================================

def test_coverage_above_the_band_refuses_runtime_shares():
    cov = opf.coverage_report(144.0, 100.0)
    assert cov["runtime_shares_reportable"] is False
    assert cov["share_denominator"] == "attributed"
    assert cov["refusal"] and "above" in cov["refusal"]


def test_coverage_below_the_band_refuses_runtime_shares():
    """Under-attribution is just as unquotable: most of the wall went somewhere unnamed."""
    cov = opf.coverage_report(30.0, 100.0)
    assert cov["runtime_shares_reportable"] is False
    assert cov["refusal"] and "below" in cov["refusal"]


def test_no_wall_at_all_refuses_rather_than_dividing_by_nothing():
    cov = opf.coverage_report(100.0, None)
    assert cov["profiler_coverage"] is None
    assert cov["runtime_shares_reportable"] is False
    assert cov["refusal"]


def test_coverage_in_band_permits_runtime_shares():
    cov = opf.coverage_report(102.0, 100.0)
    assert cov["runtime_shares_reportable"] is True
    assert cov["share_denominator"] == "wall"
    assert cov["refusal"] is None


def test_a_refused_profile_still_ranks_but_carries_no_runtime_percentage():
    """The ranking was never the broken part -- the DENOMINATOR was. Keep one, withhold the other."""
    table = [_rec(0, "linalg.transpose", 900, 1), _rec(1, "linalg.matmul", 100, 1)]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    rows = opf.rollup(table, lambda r: r["family_resolved"], "family", wall_ms=None)
    assert [r["family"] for r in rows] == ["linalg.transpose", "contraction"]
    assert rows[0]["share_of_attributed"] == pytest.approx(0.9)
    assert all(r["share_of_runtime"] is None for r in rows)


def test_shares_of_attributed_are_invariant_to_the_bad_divisor():
    """Why the 45.9 % 'felt right': a uniform over-credit cancels in a RATIO of attributed time.

    The old bug multiplied every op by the same 7/5, so the ranking and the shares-of-attributed
    survived it untouched. Only the comparison against the WALL was wrong -- which is exactly the
    one the tool printed. This is the reason the fence has to sit on the denominator, not on the
    ranking.
    """
    table = [_rec(0, "linalg.transpose", 900, 7), _rec(1, "linalg.matmul", 100, 7)]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    right = opf.rollup(table, lambda r: r["family_resolved"], "family")
    for r in table:                       # re-annotate with the WRONG divisor
        r["hits"] = 5
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    wrong = opf.rollup(table, lambda r: r["family_resolved"], "family")
    assert [r["share_of_attributed"] for r in right] == pytest.approx(
        [r["share_of_attributed"] for r in wrong])
    assert wrong[0]["ms"] > right[0]["ms"]          # the ABSOLUTE ms did move


# =================================================================================================
# 3. The family fallback: no false zero for a family whose tag was dropped.
# =================================================================================================

def test_prov_family_wins_when_present():
    assert opf.resolve_family({"mlir_op": "linalg.generic", "family": "elementwise"}) == (
        "elementwise", "prov.family")


def test_an_untagged_named_contraction_is_still_a_contraction():
    """The whole false zero in one assertion."""
    assert opf.resolve_family({"mlir_op": "linalg.matmul", "family": None}) == (
        "contraction", "mlir_op")
    assert opf.resolve_family({"mlir_op": "linalg.batch_matmul"})[0] == "contraction"


def test_an_untagged_unnamed_op_buckets_under_its_op_name():
    fam, src = opf.resolve_family({"mlir_op": "linalg.transpose", "family": None})
    assert (fam, src) == ("linalg.transpose", "mlir_op")


def test_untagged_generic_is_unknown_and_says_so():
    """A generic with no tag may BE a contraction (the int8 datapath makes them); it is not
    evidence that contraction time is zero."""
    assert opf.is_unclassified_generic({"mlir_op": "linalg.generic", "family": None}) is True
    assert opf.is_unclassified_generic({"mlir_op": "linalg.generic",
                                        "family": "elementwise"}) is False
    assert opf.is_unclassified_generic({"mlir_op": "linalg.matmul", "family": None}) is False


def test_contraction_ops_covers_everything_the_mac_pricer_prices():
    """Kept in sync with the structural pricer rather than drifting from it."""
    from merlin.xdsl_dialects.lowering.contraction_coverage import MATMUL_OPS
    assert set(MATMUL_OPS) <= opf.CONTRACTION_OPS


# =================================================================================================
# 4. A synthetic table with NO family tags at all still ranks correctly and reports a contraction.
# =================================================================================================

#: Shaped like the real int8 whole-model capture: every tag dropped by the quant/blocking rewrites,
#: ticks in the measured proportions (transpose > matmul > generic > concat).
_UNTAGGED = [
    *[_rec(i, "linalg.transpose", 2000, 7) for i in range(10)],
    *[_rec(100 + i, "linalg.matmul", 1000, 7) for i in range(10)],
    *[_rec(200 + i, "linalg.generic", 700, 7) for i in range(10)],
    *[_rec(300 + i, "tensor.concat", 400, 7) for i in range(10)],
    *[_rec(400 + i, "tensor.empty", 1, 7) for i in range(50)],
]


def test_untagged_table_ranks_the_way_the_tagged_one_would():
    table = [dict(r) for r in _UNTAGGED]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    rows = opf.rollup(table, lambda r: r["family_resolved"], "family")
    assert [r["family"] for r in rows[:4]] == [
        "linalg.transpose", "contraction", "linalg.generic", "tensor.concat"]
    assert all(r["family_sources"] == ["mlir_op"] for r in rows)


def test_untagged_table_does_not_report_a_zero_contraction():
    """The exact headline that made the tool worse than no tool."""
    table = [dict(r) for r in _UNTAGGED]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    contraction_ms = sum(r["ms_avg"] for r in table if r["vectorized"])
    assert contraction_ms > 0
    # ... and bucketing on the raw tag is what produced the zero.
    tag_only = sum(r["ms_avg"] for r in table if r.get("family") == "contraction")
    assert tag_only == 0


def test_every_untagged_row_is_marked_as_derived():
    """A reader must be able to tell a measured tag from an inferred one."""
    table = [dict(r) for r in _UNTAGGED]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    assert {r["family_source"] for r in table} == {"mlir_op"}
    table2 = [_rec(0, "linalg.generic", 10, 1, family="elementwise")]
    opf.annotate_table(table2, timebase_hz=TIMEBASE_HZ)
    assert table2[0]["family_source"] == "prov.family"
