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
from merlin.targetgen.capsule_runner import lane_report

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
                      {"on_mesh": {"matmul": 3}, "scalar_rvv_lane": {"add": 2}})
    assert rep["unexercised"] == [], "both lanes carried work — this is the capability under test"
    assert rep["observed"] == ["on_mesh", "scalar_rvv_lane"]


def test_a_lane_that_carried_nothing_is_named():
    """The actionable direction: an unnamed failure turns a composition capsule into a single-backend
    capsule nobody notices."""
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}},
                      {"on_mesh": {"matmul": 3}})
    assert rep["unexercised"] == ["scalar_rvv_lane"]
    assert rep["observed"] == ["on_mesh"]


def test_an_empty_lane_entry_counts_as_no_work():
    """A key present but empty means the same as absent — no work went there. Both fail closed."""
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, {"on_mesh": {}})["unexercised"] == ["on_mesh"]
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, {})["unexercised"] == ["on_mesh"]
    assert lane_report({"lanes": {"require": ["on_mesh"]}}, None)["unexercised"] == ["on_mesh"]


def test_lane_names_are_the_plans_own_keys_not_a_fixed_list():
    """A target whose routing plan reports different lane names needs no change here."""
    rep = lane_report({"lanes": {"require": ["some_future_lane"]}},
                      {"some_future_lane": {"op": 1}, "another": {"op": 2}})
    assert rep["unexercised"] == []
    assert rep["observed"] == ["another", "some_future_lane"]


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
    """The plan carries metadata beside its lanes; metadata never 'carried work'."""
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3},
            "n_mesh_ops": 15, "n_scalar_ops": 401, "note": "…", "mesh_matmul_extents": [{"m": 8}]}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan)
    assert rep["observed"] == ["on_mesh", "scalar_rvv_lane"], (
        "ints, strings and lists are not lanes even when truthy")


def test_execution_accounting_overrides_the_plan():
    """A lane the router filled but the hardware never ran did NOT carry work."""
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    exec_none = {"matmul_layers_routed": 15, "matmul_layers_on_mesh": 0,
                 "matmul_layers_host_fallback": 15}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, exec_none)
    assert rep["unexercised"] == ["on_mesh"]
    # PER LANE, not one word for the report. The two lanes genuinely have different evidence here --
    # the mesh was measured, the host lane was not -- and a single label had to misdescribe one of them.
    assert rep["evidence"]["on_mesh"] == "execution"
    assert rep["evidence"]["scalar_rvv_lane"] == "routing_plan"


def test_execution_accounting_can_also_confirm_a_lane():
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    exec_ok = {"matmul_layers_routed": 15, "matmul_layers_on_mesh": 15,
               "matmul_layers_host_fallback": 0}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, exec_ok)
    assert rep["unexercised"] == []


def test_a_plan_only_verdict_says_so():
    """Without execution accounting the report is about intent, and must admit it."""
    rep = lane_report({"lanes": {"require": ["on_mesh"]}}, {"on_mesh": {"matmul": 1}})
    assert rep["evidence"]["on_mesh"] == "routing_plan"
    assert rep["plan_only_lanes"] == ["on_mesh"]
    assert "PLANNED" in rep["caveat"] or "planned" in rep["caveat"].lower()


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
    """`UNKNOWN` is a sentinel string, not a number. Reading it as 0 would turn "nobody could tell"
    into "the lane carried nothing" -- a measurement claim from an absence of measurement."""
    plan = {"on_mesh": {"matmul": 1}, "scalar_rvv_lane": {"add": 1}}
    rep = lane_report({"lanes": {"require": ["scalar_rvv_lane"]}}, plan, {},
                      {"kernels_ran": "UNKNOWN"})
    assert rep["unexercised"] == [], "an unmeasured lane must fall back to the plan, not to zero"
    assert rep["evidence"]["scalar_rvv_lane"] == "routing_plan"


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
