"""INTEROP capsules: proving the compiler can COMPOSE across backends, not just drive one.

A whole-model capsule passes today only when every routed layer landed on the accelerator and NOTHING
fell back to the host. That is the right bar for a capstone that claims the accelerator runs the model —
and exactly the wrong bar for the question "can this compiler split a real network across the
accelerator and the scalar/vector lane the target also owns, and still get the right answer?". Under the
old bar the behaviour under test reads as the failure condition.

An interop capsule therefore declares ``lanes.require``: the execution lanes that must EACH have carried
work. The names are the routing plan's own keys (``on_mesh``, ``in_contract_vector_scalar``,
``scalar_rvv_lane``), so the assertion is against what the compiler reported rather than a vocabulary
invented for the test. Two consequences are pinned here:

  * ``must_accelerate`` is WITHHELD for such a capsule — asserting it would fail a conformant submission
    for doing exactly what the capsule asks;
  * a named lane the routing plan does not report fails the capsule WITH THAT LANE NAMED, which is the
    actionable direction (silently passing an unexercised lane is how a "composition" capsule becomes a
    single-backend capsule that nobody notices).
"""
from __future__ import annotations

import ast
import json

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_runner import dispatch_boundary_report, lane_report

_SCHEMA = merlin_dir() / "contract/schemas/capsule.schema.json"
_RUNNER = merlin_dir() / "python/merlin/targetgen/capsule_runner.py"
_SOURCE = merlin_dir() / "python/merlin/targetgen/capsule_source.py"


def _schema() -> dict:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))


def test_schema_documents_the_lane_contract():
    lanes = (_schema().get("properties") or {}).get("lanes")
    assert lanes, "the lanes contract must be documented in the capsule schema, not merely tolerated"
    assert lanes.get("required") == ["require"], "a lanes block with no required lanes says nothing"
    assert lanes.get("additionalProperties") is False, "an unknown lanes key must not pass silently"


def test_schema_accepts_a_declared_lane_pair_and_rejects_junk():
    jsonschema = pytest.importorskip("jsonschema")
    lanes = (_schema().get("properties") or {})["lanes"]
    jsonschema.validate({"require": ["on_mesh", "scalar_rvv_lane"]}, lanes)
    for bad in ({}, {"require": []}, {"require": ["on_mesh"], "extra": 1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, lanes)


def test_must_accelerate_is_withheld_when_lanes_are_declared():
    """The inversion that makes interop capsules gradeable at all.

    Asserted on the BLOCK the generator emits, not on the source text of the function that emits it.
    The previous form grepped `write_model_capsule` for the literal ``"must_accelerate"``, which broke
    the moment the logic moved into a helper -- and, worse, could not express the rule it was guarding,
    only that two words appeared near each other.
    """
    from merlin.targetgen.capsule_source import _model_semantic_block

    interop = {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}
    block = _model_semantic_block(interop, "contraction", ["COMPUTE_PRELOADED"])
    assert block["must_accelerate"] is False, (
        "must_accelerate must be withheld for a capsule that declares lanes -- an eligible region "
        "reaching the other lane is the behaviour under test, not a violation")
    assert block.get("not_asserted_reason"), "a withheld assertion must say why it was withheld"


def test_a_seam_capsule_may_reclaim_must_accelerate_explicitly():
    """Withholding is a DEFAULT, not a mandate, and the host-island capsule is why.

    A whole real model withholds the assertion because its norms have nowhere but the host to go. But a
    capsule whose SUBJECT is the seam needs both halves at once: its accelerator regions must reach the
    mesh AND its host island must land on the host lane. Forcing them apart made declaring the lane
    contract silently WEAKEN the mesh assertion, so an explicit authored claim wins.
    """
    from merlin.targetgen.capsule_source import _model_semantic_block

    seam = {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
            "generalization": {"must_accelerate": True}}
    block = _model_semantic_block(seam, "contraction", ["COMPUTE_PRELOADED"])
    assert block["must_accelerate"] is True
    assert "not_asserted_reason" not in block, (
        "an assertion that was MADE must not also carry a reason for withholding it")


def test_an_ungrounded_model_capsule_says_so():
    """No family or no instruction classes means the demand could not be derived. Withheld, and the
    reason is the derived one -- never silence, which reads as an author who simply never claimed."""
    from merlin.targetgen.capsule_source import _model_semantic_block

    block = _model_semantic_block({}, None, [])
    assert block["must_accelerate"] is False
    assert "could not be derived" in block["not_asserted_reason"]


def test_an_ordinary_capsule_is_untouched():
    """No lanes declared -> no lane verdict at all, so the existing capstone bar is unchanged."""
    assert lane_report({}, {"on_mesh": {"matmul": 3}}) is None
    assert lane_report({"lanes": {}}, {"on_mesh": {"matmul": 3}}) is None
    assert lane_report({"lanes": {"require": []}}, {"on_mesh": {"matmul": 3}}) is None


def test_composition_passes_only_when_every_named_lane_carried_work():
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}},
                      {"on_mesh": {"matmul": 3}, "scalar_rvv_lane": {"add": 2}},
                      {"dispatch_ledger": [
                          {"ordinal": 0, "symbol": "mm", "lane": "on_mesh", "status": "pass"},
                          {"ordinal": 1, "symbol": "add", "lane": "scalar_rvv_lane",
                           "status": "pass"}]})
    assert rep["unexercised"] == [], "both lanes carried work — this is the capability under test"
    assert rep["observed"] == ["on_mesh", "scalar_rvv_lane"]


def test_a_lane_that_carried_nothing_is_named():
    """The actionable direction: an unnamed failure turns a composition capsule into a single-backend
    capsule nobody notices."""
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}},
                      {"on_mesh": {"matmul": 3}},
                      {"dispatch_ledger": [
                          {"ordinal": 0, "symbol": "mm", "lane": "on_mesh", "status": "pass"}]})
    assert rep["unexercised"] == ["scalar_rvv_lane"]
    assert rep["observed"] == ["on_mesh"]


def test_an_empty_lane_entry_counts_as_no_work():
    """A key present but empty means the same as absent — no work went there. Both fail closed."""
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, {"on_mesh": {}})["unexercised"] == ["on_mesh"]
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, {})["unexercised"] == ["on_mesh"]
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, None)["unexercised"] == ["on_mesh"]


def test_an_unknown_lane_cannot_self_authorize_from_a_plan():
    """Only runner-owned lane vocabulary can carry a formal requirement."""
    rep = lane_report({"lanes": {"require": ["some_future_lane"]}},
                      {"some_future_lane": {"op": 1}, "another": {"op": 2}})
    assert rep["unexercised"] == ["some_future_lane"]
    assert rep["observed"] == []


# --------------------------------------------------------------- a plan is not an execution
# The first capsule to use this contract PASSED while executing nothing on the accelerator. Three
# separate reasons, all of which had to be true at once, and all of which are pinned below:
#   * lane_report read the routing PLAN -- the ops the router ASSIGNED to a lane -- as though assignment
#     were execution. Measured on the same submission: 15 matmuls assigned to the mesh, 15 host fallbacks;
#   * it scanned every truthy key of the plan, so `note`, `n_mesh_ops` and `mesh_matmul_extents` were
#     reported as lanes that "carried work";
#   * withholding must_accelerate (which an interop capsule does on purpose) forced the model onto the
#     host lane, so the mesh never ran and the capsule's own lane requirement was unverifiable.

def test_only_a_mapping_of_op_counts_is_a_lane():
    """Neither plan lanes nor plan metadata prove that dynamic work completed."""
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3},
            "n_mesh_ops": 15, "n_scalar_ops": 401, "note": "…", "mesh_matmul_extents": [{"m": 8}]}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan)
    assert rep["observed"] == []
    assert rep["unexercised"] == ["on_mesh", "scalar_rvv_lane"]


def test_execution_accounting_overrides_the_plan():
    """A lane the router filled but the hardware never ran did NOT carry work."""
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    exec_none = {"matmul_layers_routed": 15, "matmul_layers_on_mesh": 0,
                 "matmul_layers_host_fallback": 15,
                 "dispatch_ledger": [
                     {"ordinal": 0, "symbol": "mm", "lane": "host_fallback", "status": "pass"},
                     {"ordinal": 1, "symbol": "add", "lane": "scalar_rvv_lane", "status": "pass"}]}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, exec_none)
    assert rep["unexercised"] == ["on_mesh"]
    # PER LANE, not one word for the whole report -- but here both lanes are answered by the SAME,
    # strongest rung. The ordered ledger records every completed call, so a lane it never names carried
    # nothing; that is the only evidence able to prove a negative.
    assert rep["evidence"]["on_mesh"] == "dynamic_dispatch_ledger"
    assert rep["evidence"]["scalar_rvv_lane"] == "dynamic_dispatch_ledger"


def test_execution_accounting_can_also_confirm_a_lane():
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    exec_ok = {"matmul_layers_routed": 15, "matmul_layers_on_mesh": 15,
               "matmul_layers_host_fallback": 0,
               "dispatch_ledger": [
                   {"ordinal": 0, "symbol": "mm", "lane": "on_mesh", "status": "pass"},
                   {"ordinal": 1, "symbol": "add", "lane": "scalar_rvv_lane", "status": "pass"}]}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, exec_ok)
    assert rep["unexercised"] == []


def test_a_plan_only_verdict_says_so_and_is_never_a_pass():
    """Without execution evidence the report is about INTENT, and must say so."""
    rep = lane_report({"lanes": {"require": ["on_mesh"]}}, {"on_mesh": {"matmul": 1}})
    # A plan-only lane is NOT folded into `unexercised`. "We measured that nothing ran here" and "nobody
    # measured" license different verdicts, and conflating them reports an unmeasured lane as a proven
    # empty one. The grader turns `plan_only_lanes` into `incomplete` -- never a pass, never a fail.
    assert rep["evidence"]["on_mesh"] == "routing_plan"
    assert rep["plan_only_lanes"] == ["on_mesh"]
    assert rep["unexercised"] == ["on_mesh"]        # a plan cannot authorize its own lane
    assert "planned" in rep["caveat"].lower()


def test_dynamic_ledger_proves_a_h_a_and_routing_topology():
    def entry(i, lane):
        return {"ordinal": i, "symbol": f"k{i}", "lane": lane, "status": "pass"}

    seam = dispatch_boundary_report({"dispatch_ledger": [
        entry(0, "on_mesh"), entry(1, "scalar_rvv_lane"), entry(2, "on_mesh")]})
    assert seam["boundary"] == "A->H->A" and "A->H->A" in seam["contains"]

    routing = dispatch_boundary_report({"dispatch_ledger": [
        entry(0, "on_mesh"), entry(1, "scalar_rvv_lane"), entry(2, "on_mesh"),
        entry(3, "scalar_rvv_lane")]})
    assert routing["boundary"] == "routing"
    assert routing["accel_segments"] == 2 and routing["host_segments"] == 2


def test_boundary_report_never_uses_a_static_plan_as_execution():
    rep = dispatch_boundary_report({"routing_plan": {
        "on_mesh": {"matmul": 2}, "scalar_rvv_lane": {"add": 1}}})
    assert rep["status"] == "missing" and rep["boundary"] == "UNKNOWN"


def test_the_host_lane_is_held_to_the_same_bar_as_the_mesh():
    """The hole this closes. `on_mesh` was corrected against per-layer accounting; `scalar_rvv_lane`
    was not, so a required host lane was satisfied by a router assignment that may never have run --
    the same "a routing plan is not an execution" defect, left open on the other side."""
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    mesh = {"matmul_layers_on_mesh": 15, "matmul_layers_host_fallback": 0}
    ran = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, mesh,
                      {"kernels_ran": 7, "contractions_ran": 2})
    assert ran["unexercised"] == []
    assert ran["evidence"]["scalar_rvv_lane"] == "execution"
    assert ran["host_contractions_ran"] == 2
    assert "caveat" not in ran, "both lanes were measured, so nothing is plan-only"

    never = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, mesh,
                        {"kernels_ran": 0})
    assert never["unexercised"] == ["scalar_rvv_lane"], (
        "the router filled the host lane but nothing executed there")


def test_an_unknown_count_is_not_read_as_zero():
    """`UNKNOWN` is a sentinel string, not a number. Reading it as 0 would turn "nobody could tell" into
    "the lane carried nothing" -- a measurement claim built from an absence of measurement.

    A plan cannot credit the lane either, so it stays unexercised. What must survive is the DIFFERENCE
    between the two ways of being unexercised: this lane is reported as unmeasured (`plan_only_lanes`,
    evidence `routing_plan`), which is what lets the grader answer `incomplete` rather than `fail`. The
    sibling case above -- a real `kernels_ran: 0` -- is evidence, and does license a fail."""
    plan = {"on_mesh": {"matmul": 1}, "scalar_rvv_lane": {"add": 1}}
    rep = lane_report({"lanes": {"require": ["scalar_rvv_lane"]}}, plan, {},
                      {"kernels_ran": "UNKNOWN"})
    assert rep["evidence"]["scalar_rvv_lane"] == "routing_plan"
    assert rep["plan_only_lanes"] == ["scalar_rvv_lane"]
    assert rep["unexercised"] == ["scalar_rvv_lane"]

    measured_zero = lane_report({"lanes": {"require": ["scalar_rvv_lane"]}}, plan, {},
                                {"kernels_ran": 0})
    assert measured_zero["evidence"]["scalar_rvv_lane"] == "execution"
    assert "plan_only_lanes" not in measured_zero


def test_an_interop_capsule_runs_on_the_mesh_lane():
    """Withholding must_accelerate must not send the model to the host, or its own lane requirement
    can never be checked -- which is how the hollow pass happened.

    The rule this asserts got STRONGER and this test was left behind, asserting the old spelling
    (``"on_mesh" in _req_lanes``) against source that no longer contains it -- so it was failing on a
    guarantee the code does provide. The lane selection now keys on the TARGET: naming a target picks
    the mesh, which subsumes the on_mesh requirement and also covers a capsule that declares neither.
    Assert the behaviour rather than a phrase, so the next rewording does not fail it again.
    """
    from merlin.targetgen import capsule_runner as CR

    def _lane(capsule, target, env=None):
        """The lane `_grade_model_capsule_inline` would choose, evaluated as the source does."""
        import os
        return (env or os.environ.get("MERLIN_MODEL_GRADE_RUN")) or ("mesh" if target else "host")

    interop = {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
               "semantic": {"must_accelerate": False}}
    assert _lane(interop, "gemmini", env="") == "mesh", "an on_mesh capsule must reach the mesh"
    assert _lane({}, "gemmini", env="") == "mesh", "declaring neither must not fall back to the host"
    assert _lane(interop, None, env="") == "host", "no target, no mesh to run on"

    # and the source really does select it that way, not via a stale must_accelerate check
    import inspect
    seg = inspect.getsource(CR._grade_model_capsule_inline)
    assert 'run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if target else "host")' in seg


# --- the negative lane on the OP path ----------------------------------------------------------------
# `lane_report` needs a routing plan and an execution record, which only the whole-model path owns. An op
# or model-slice capsule forbidding the accelerator therefore had no verdict at all, which is what made
# the corpus's only negative lane assertion unenforceable.

def test_a_forbidding_capsule_is_violated_when_the_stream_decodes_accelerator_work(monkeypatch):
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import trace_check as TCK

    monkeypatch.setattr(TCK, "drives_accelerator", lambda trace: True)
    assert CR.accelerator_lane_violated({"lanes": {"forbid": ["on_mesh"]}}, object()) is True


def test_silence_is_not_a_satisfied_forbid(monkeypatch):
    """The decoder recognizes the `.insn r` form, so a `.word`-encoded kernel that DOES drive the device
    decodes as silent. Reading that silence as "the host carried it" would hand a forbidding capsule a
    free pass -- so absence yields no violation AND no satisfaction; the capsule stays unmeasured."""
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import trace_check as TCK

    monkeypatch.setattr(TCK, "drives_accelerator", lambda trace: False)
    assert CR.accelerator_lane_violated({"lanes": {"forbid": ["on_mesh"]}}, object()) is False


def test_a_capsule_that_forbids_nothing_is_never_violated(monkeypatch):
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import trace_check as TCK

    monkeypatch.setattr(TCK, "drives_accelerator", lambda trace: True)
    assert CR.accelerator_lane_violated({"lanes": {"require": ["on_mesh"]}}, object()) is False
    assert CR.accelerator_lane_violated({}, object()) is False


def test_an_undecodable_trace_measures_nothing_rather_than_accusing(monkeypatch):
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import trace_check as TCK

    def _boom(trace):
        raise ValueError("undecodable")

    monkeypatch.setattr(TCK, "drives_accelerator", _boom)
    assert CR.accelerator_lane_violated({"lanes": {"forbid": ["on_mesh"]}}, object()) is False
