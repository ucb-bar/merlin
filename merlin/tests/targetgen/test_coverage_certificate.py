"""What the ARR coverage certificate is evidence OF.

Every number in the certificate is derived from the routing PLAN, which lists what the router assigned
and not what ran. That is the same conflation already fixed on the lane side, where one submission
assigned 15 matmuls to the mesh and fell back on all 15 at run time -- and the certificate, which is the
surface people actually quote "how much of what it could accelerate did it accelerate" from, had never
been told about it.

So it must state its own evidence, and carry the run beside the plan when a run exists.
"""
from __future__ import annotations

from merlin.targetgen import coverage_certificate as CC

# A plan with no regions: these tests are about the evidence FIELDS, which must be present and honest
# whatever the regions say. The recall arithmetic itself is exercised elsewhere.
_EMPTY_PLAN = {"mesh": [], "fallback": [], "scalar_rvv": [], "results": []}


def test_the_certificate_states_that_its_numbers_are_plan_derived():
    assert CC.build(_EMPTY_PLAN, {}, target="t")["arr_evidence"] == "routing_plan"


def test_without_an_execution_record_the_crosscheck_is_absent_not_agreeing():
    assert CC.build(_EMPTY_PLAN, {}, target="t")["execution_crosscheck"] is None


def test_a_plan_the_run_did_not_carry_out_is_reported_as_disagreeing():
    """15 assigned, 0 executed: the recalls describe an intent that did not happen."""
    x = CC.build(_EMPTY_PLAN, {}, target="t",
                 execution={"matmul_layers_routed": 15, "matmul_layers_on_mesh": 0,
                            "matmul_layers_host_fallback": 15})["execution_crosscheck"]
    assert x["agrees"] is False
    assert x["matmul_layers_on_mesh"] == 0 and x["matmul_layers_host_fallback"] == 15


def test_a_plan_the_run_carried_out_agrees():
    x = CC.build(_EMPTY_PLAN, {}, target="t",
                 execution={"matmul_layers_routed": 15, "matmul_layers_on_mesh": 15,
                            "matmul_layers_host_fallback": 0})["execution_crosscheck"]
    assert x["agrees"] is True


def test_an_unknown_count_leaves_agreement_undecided_rather_than_true():
    """`UNKNOWN` is a sentinel string, not a number, and "nobody could tell" is not "they agree"."""
    x = CC.build(_EMPTY_PLAN, {}, target="t",
                 execution={"matmul_layers_routed": "UNKNOWN",
                            "matmul_layers_on_mesh": 3})["execution_crosscheck"]
    assert x["agrees"] is None
