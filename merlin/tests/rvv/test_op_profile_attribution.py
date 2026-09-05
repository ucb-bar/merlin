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


# =================================================================================================
# 5. The window is per EXECUTION, and a session bundle runs @forward many times per timed iteration.
#
# The 144 % artifact was one direction of a window mismatch. This is the other: a version-1 session
# bundle declares `steps: 256` and its harness calls merlin_run once per step INSIDE one timed
# iteration, so the shim accumulates 256 executions against a wall that reports one iteration.
# Comparing the per-execution attributed total against that wall gives ~1/256 and refuses a profile
# that is in fact sound -- an under-attribution refusal manufactured by arithmetic, not measurement.
# =================================================================================================

def test_a_plain_model_runs_forward_once_per_iteration():
    assert opf.executions_per_iteration(7, 5, 2) == 1      # the 144 % artifact's own run
    assert opf.executions_per_iteration(1, 1, 0) == 1


def test_a_session_bundle_runs_forward_once_per_declared_step():
    """256 steps in one launch of one timed iteration is 256 executions per iteration."""
    assert opf.executions_per_iteration(256, 1, 0) == 256
    assert opf.executions_per_iteration(768, 2, 1) == 256   # 3 launches x 256 steps


def test_an_underivable_divisor_is_unknown_not_guessed():
    """A count that is not a whole multiple of the launches means the model did something the
    profiler cannot account for -- refuse rather than round."""
    assert opf.executions_per_iteration(7, 5, 0) is None    # 7/5 is not whole
    assert opf.executions_per_iteration(0, 1, 0) is None
    assert opf.executions_per_iteration(1, 0, 0) is None


def test_a_session_profile_is_not_refused_for_a_window_it_did_not_have():
    """Same ticks, same wall: dividing the wall by the step count is the difference between a
    coverage of ~0.004 (refused) and ~1.0 (reportable)."""
    wall_ticks = 256_000.0                       # one timed iteration = a 256-step session
    attributed = 1_000.0                         # per ONE execution of @forward
    naive = opf.coverage_report(attributed, wall_ticks, executions=256, timed_iterations=1)
    fixed = opf.coverage_report(attributed, wall_ticks, executions=256, timed_iterations=1,
                                executions_per_timed_iteration=256)
    assert naive["profiler_coverage"] == pytest.approx(1 / 256, abs=1e-4)
    assert naive["runtime_shares_reportable"] is False
    assert fixed["profiler_coverage"] == pytest.approx(1.0)
    assert fixed["runtime_shares_reportable"] is True
    assert fixed["wall_ticks_per_execution"] == 1_000.0
    assert fixed["executions_per_timed_iteration"] == 256


def test_an_underivable_divisor_refuses_runtime_shares():
    cov = opf.coverage_report(100.0, 100.0, executions_per_timed_iteration=0)
    assert cov["runtime_shares_reportable"] is False
    assert cov["refusal"] and "divisor" in cov["refusal"]


# =================================================================================================
# 6. Actionable categories: the bucket a lever is aimed at, not the frontend's family name.
# =================================================================================================

def test_the_requant_epilogue_is_not_counted_as_a_contraction():
    """The int8 rewrite splits one matmul into a contraction and a requant epilogue that CARRY THE
    SAME fqn and family. Bucketing on family charges the quantize chain's cost to the matmul --
    which is precisely the claim ('the quantize chain costs more than every matmul combined') this
    profile exists to test."""
    contraction = {"mlir_op": "linalg.generic", "family": "contraction", "op": "matmul",
                   "role": "contraction", "fqn": "layers.0.attn.q_proj"}
    requant = {"mlir_op": "linalg.generic", "family": "contraction", "op": "matmul",
               "role": "requant", "fqn": "layers.0.attn.q_proj"}
    assert opf.resolve_category(contraction) == ("contraction", "prov.role")
    assert opf.resolve_category(requant) == ("quantize_requant", "prov.role")
    # ... while the FAMILY still calls them both a contraction, which is why the split is needed.
    assert opf.resolve_family(contraction)[0] == opf.resolve_family(requant)[0] == "contraction"


def test_the_whole_activation_quantize_chain_lands_in_one_bucket():
    """`passes_quant_int._carry_prov` stamps act_amax/act_scale/act_quantize/gather/requant. The
    first four are the per-row dynamic activation quantization; they must roll up together or the
    chain's cost is reported as five unrelated slivers."""
    roles = ["act_amax", "act_scale", "act_quantize", "requant"]
    cats = {opf.resolve_category({"mlir_op": "linalg.generic", "role": r})[0] for r in roles}
    assert cats == {"quantize_requant"}
    assert opf.resolve_category({"mlir_op": "linalg.generic", "role": "gather"})[0] == "gather"


def test_softmax_is_a_reduction_not_a_normalization():
    """`prov.family` calls softmax and layer_norm both `normalization`, and they cost differently:
    one is a reduction chain, the other is not."""
    sm = {"mlir_op": "linalg.generic", "family": "normalization", "op": "softmax"}
    ln = {"mlir_op": "linalg.generic", "family": "normalization", "op": "layer_norm"}
    assert opf.resolve_category(sm) == ("reduction_softmax", "prov.op")
    assert opf.resolve_category(ln) == ("normalization", "prov.family")


def test_a_metadata_view_is_not_a_layout_copy():
    """`family = layout` covers both a free reshape and a transpose that materializes a buffer.
    Charging a lever aimed at layout copies with the cost of free views would misprice it."""
    view = {"mlir_op": "tensor.expand_shape", "family": "layout", "op": "view"}
    transpose = {"mlir_op": "linalg.transpose", "family": "layout", "op": "transpose"}
    assert opf.resolve_category(view)[0] == "layout_view"
    assert opf.resolve_category(transpose)[0] == "layout_copy"


def test_an_untagged_op_the_map_does_not_know_is_named_not_folded():
    """Fail closed: an op none of role/op/family/mlir_op places gets its OWN bucket, so it shows up
    in the ranking as unaccounted rather than inflating a neighbour."""
    cat, src = opf.resolve_category({"mlir_op": "some.future_op"})
    assert cat == "unclassified:some.future_op" and src == "unknown"
    # the one genuinely unknowable case keeps its dedicated bucket
    assert opf.resolve_category({"mlir_op": "linalg.generic"}) == ("unclassified_generic", "unknown")


def test_annotate_table_stamps_the_category_and_its_source():
    table = [{"id": 0, "mlir_op": "linalg.generic", "family": "contraction", "op": "matmul",
              "role": "requant", "ticks": 100, "hits": 1}]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    assert table[0]["category"] == "quantize_requant"
    assert table[0]["category_source"] == "prov.role"


# =================================================================================================
# 7. Every row carries its own denominator: a share quoted alone still says what it is a share of.
# =================================================================================================

def test_every_rollup_row_states_its_denominator_and_coverage():
    """The 144 % profile was quoted one row at a time. A caveat that lives only in a sibling block
    is a caveat that does not travel with the number."""
    table = [_rec(0, "linalg.matmul", 900, 1), _rec(1, "linalg.transpose", 100, 1)]
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    refused = opf.rollup(table, lambda r: r["category"], "category", wall_ms=None, coverage=1.44)
    assert all(r["share_denominator"] == "attributed" for r in refused)
    assert all(r["profiler_coverage"] == 1.44 for r in refused)
    ok = opf.rollup(table, lambda r: r["category"], "category",
                    wall_ms=opf.sum_attributed_ticks(table) * (1e9 / TIMEBASE_HZ) / 1e6,
                    coverage=1.0)
    assert all(r["share_denominator"] == "wall" for r in ok)
    assert sum(r["share_of_runtime"] for r in ok) == pytest.approx(1.0)


def test_what_the_profiler_cannot_attribute_is_named_rather_than_missing():
    """A bucket the mark interval structurally cannot see must not read as a measured zero."""
    assert set(opf.CATEGORIES_NOT_ATTRIBUTABLE) == {"allocator", "fork_join", "intra_op"}
    for why in opf.CATEGORIES_NOT_ATTRIBUTABLE.values():
        assert len(why) > 60          # each says WHY, and what to do instead


# =================================================================================================
# 8. Body evidence: an untagged `linalg.generic` is not a mystery, it is an unread one.
#
# The quantization rewrites rebuild ops without re-stamping provenance. Measured on lstmnetvit's
# prepared module, 261 of 2656 marks reach the table as `linalg.generic` with no family — and 76 of
# those bodies carry `math.roundeven` + `arith.fptosi` (the activation-quantize step) and 80 carry
# `math.absf` + `arith.maximumf` (the per-row amax that finds its scale). Bucketing all of them as
# UNKNOWN reports the quantize chain as 71 ops when it is ~250, against a contraction total of 224 —
# which is exactly the comparison the profile is being run to settle.
# =================================================================================================

def test_round_to_int_is_the_quantize_step():
    assert opf.classify_generic_body(["math.roundeven", "arith.fptosi", "arith.divf"]) == (
        "quantize_requant", "body:round_to_int")
    # a narrowing convert alone is enough; not every spelling rounds explicitly
    assert opf.classify_generic_body(["arith.fptosi"])[0] == "quantize_requant"


def test_int_to_float_times_a_scale_is_the_dequantize_step():
    assert opf.classify_generic_body(["arith.sitofp", "arith.mulf"]) == (
        "quantize_requant", "body:int_to_float_scale")


def test_abs_max_is_the_scale_search_and_keeps_its_own_bucket():
    """An abs-then-max reduction is ALSO a legitimate model op, so it gets a bucket a reader can
    discount separately instead of being folded into the quantize total."""
    cat, rule = opf.classify_generic_body(["math.absf", "arith.maximumf"])
    assert (cat, rule) == ("quantize_scale_search", "body:abs_max")
    assert cat in opf.QUANTIZE_CHAIN_CATEGORIES and "quantize_requant" in opf.QUANTIZE_CHAIN_CATEGORIES


def test_a_body_no_rule_reads_stays_unknown():
    """Fail closed. A bare `arith.divf` is the scale-apply AND an ordinary elementwise divide; the
    profiler cannot tell, so it must not claim to."""
    assert opf.classify_generic_body(["arith.divf"]) is None
    assert opf.classify_generic_body([]) is None
    assert opf.classify_generic_body(None) is None
    assert opf.resolve_category({"mlir_op": "linalg.generic", "body_ops": ["arith.divf"]}) == (
        "unclassified_generic", "unknown")


def test_body_evidence_names_itself_so_it_can_be_audited():
    cat, src = opf.resolve_category({"mlir_op": "linalg.generic",
                                     "body_ops": ["math.roundeven", "arith.fptosi"]})
    assert cat == "quantize_requant" and src.startswith("body:")


def test_a_tagged_op_is_decided_by_its_tag_not_its_body():
    """Body evidence is the LAST resort, not a competing signal: a contraction whose epilogue was
    fused into it still carries a convert, and must not be re-bucketed as a quantize op."""
    rec = {"mlir_op": "linalg.generic", "family": "contraction", "op": "matmul",
           "role": "contraction", "body_ops": ["math.roundeven", "arith.fptosi"]}
    assert opf.resolve_category(rec) == ("contraction", "prov.role")


def test_body_ops_are_attached_only_where_they_can_answer_something():
    """Carrying a fingerprint for all 5000+ ops would bloat every artifact for no gain: an op the
    frontend already tagged needs no body evidence."""
    mlir = """module {
  func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = linalg.generic {indexing_maps = []} ins(%a : tensor<4xf32>) outs(%a : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32):
      %r = math.roundeven %x : f32
      %i = arith.fptosi %r : f32 to i32
      linalg.yield %x : f32
    } -> tensor<4xf32>
    %1 = linalg.generic {prov.family = "elementwise", prov.op = "add"} ins(%a : tensor<4xf32>) outs(%a : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"""
    _, _, ops = opf.find_forward_ops(mlir)
    untagged, tagged = ops[0], ops[1]
    assert "body_ops" in untagged and "math.roundeven" in untagged["body_ops"]
    assert "arith.fptosi" in untagged["body_ops"]
    assert "body_ops" not in tagged
    assert opf.resolve_category(untagged)[0] == "quantize_requant"


def test_body_ops_reads_both_print_forms():
    """Bundles ship GENERIC form and the printer emits CUSTOM; a scan that reads only one silently
    finds no body at all, which reads as 'unknown' rather than as a failed match."""
    assert "arith.mulf" in opf._body_ops('      %3 = arith.mulf %1, %2 : f32')
    assert "arith.mulf" in opf._body_ops('      %3 = "arith.mulf"(%1, %2) : (f32, f32) -> f32')
