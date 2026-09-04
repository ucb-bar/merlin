"""Preflight: can a transform lever reach the IR at all, before board time is spent on it."""
from __future__ import annotations

from merlin.mining.lever_applicability import (all_matched_op_names, applicability,
                                               matched_op_names)

# A schedule shaped like the real ones: two payload matches plus a CONTAINER match that exists only
# to receive apply_patterns.
SCHED = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %l:2 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:3 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.yield
  }
}
"""


def test_a_container_match_is_not_a_work_target():
    """func.func is matched only to receive apply_patterns, and it is ALWAYS present.

    Counting it as a work target is what made the first version of this check report 'applicable'
    for the exact case it was written to catch: on the int8 datapath every payload op had vanished,
    but func.func was there, so `any(present > 0)` said the lever was fine.
    """
    assert matched_op_names(SCHED) == ("linalg.batch_matmul", "linalg.matmul")
    assert "func.func" in all_matched_op_names(SCHED)
    assert "func.func" not in matched_op_names(SCHED)


def test_a_lever_whose_payload_ops_are_all_absent_is_inapplicable():
    """This is the int8 case: apply_quant leaves 0 linalg.matmul, so the match yields an empty
    handle, every downstream op is a vacuous no-op, and the lever still builds and gates clean."""
    a = applicability(SCHED, {"func.func": 1, "linalg.generic": 280})
    assert a["status"] == "inapplicable"
    assert a["present"] == {"linalg.batch_matmul": 0, "linalg.matmul": 0}
    assert "empty handle" in a["reason"] and "reports as applied" in a["reason"].replace(
        "report as applied", "reports as applied")


def test_one_present_payload_op_is_enough():
    """A model with matmuls but no batch_matmuls can still use the lever."""
    a = applicability(SCHED, {"func.func": 1, "linalg.matmul": 15})
    assert a["status"] == "applicable"
    assert a["present"]["linalg.matmul"] == 15


def test_a_schedule_that_does_not_match_by_name_is_unknown_not_inapplicable():
    """Fail OPEN where we cannot judge: refusing a lever this check simply cannot analyse would be
    worse than running it. Reported honestly rather than guessed."""
    a = applicability("transform.structured.match interface{LinalgOp} in %arg0", {"func.func": 1})
    assert a["status"] == "unknown"
    assert a["needs"] == ()
    for empty in ("", None):
        assert applicability(empty, {})["status"] == "unknown"


def test_the_real_mrpad_schedule_is_judged_on_its_payload_ops():
    from merlin.llvmlower.impr_features import _accumulator_resident_v3_mrpad_pre_schedule as sched
    txt = sched(4, 16, 16, NR_bmm=8)
    assert matched_op_names(txt) == ("linalg.batch_matmul", "linalg.matmul")
    # the int8 datapath as it lowers by default: inapplicable
    assert applicability(txt, {"func.func": 1, "linalg.generic": 280})["status"] == "inapplicable"
    # with named_int8_contraction on: applicable
    assert applicability(txt, {"func.func": 1, "linalg.matmul": 15})["status"] == "applicable"


def test_the_check_has_a_caller_at_the_seam_that_knows_both_things():
    """A detector nobody calls is the defect it was written to catch.

    prepare_for_lowering is the one place that holds BOTH the requested features and the prepared
    module, so that is where a lever which cannot fire has to be named.
    """
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python" / "merlin" / "runtime" / "backends" / "zephyr_model.py").read_text()
    assert "inapplicable_features as _inapplicable" in src
    assert "op_counts_out=_op_counts" in src
    assert "[lever] INAPPLICABLE" in src
    # and it must never break a build: a diagnostic that can fail a compile is worse than none
    assert "a diagnostic must never break a build" in src


def test_named_features_resolve_their_own_schedules():
    """No table of its own: the check reads each lever's schedule through impr_features, so it stays
    correct as levers are added."""
    from merlin.llvmlower.impr_features import MRPAD_INT8_NAME
    from merlin.mining.lever_applicability import inapplicable_features
    feats = {MRPAD_INT8_NAME, "promote_buffers_to_stack", "named_int8_contraction"}
    # int8 as it lowers by DEFAULT: the register block cannot fire
    bad = inapplicable_features(feats, {"func.func": 1, "linalg.generic": 280})
    assert MRPAD_INT8_NAME in bad
    assert bad[MRPAD_INT8_NAME]["present"] == {"linalg.batch_matmul": 0, "linalg.matmul": 0}
    # with the named contraction restored, nothing is inapplicable
    assert inapplicable_features(feats, {"func.func": 1, "linalg.matmul": 15}) == {}
    # unknown names are skipped rather than guessed at
    assert inapplicable_features({"no_such_feature"}, {"func.func": 1}) == {}
