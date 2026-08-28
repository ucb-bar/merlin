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
    """The inversion that makes interop capsules gradeable at all."""
    src = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "write_model_capsule")
    seg = ast.get_source_segment(src, fn) or ""
    assert '"must_accelerate"' in seg
    # the must_accelerate expression must consult the lanes declaration
    ma = seg.split('"must_accelerate"', 1)[1].split("\n\n", 1)[0]
    assert "lanes" in ma, (
        "must_accelerate must be withheld for a capsule that declares lanes — an eligible region "
        "reaching the other lane is the behaviour under test, not a violation")


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
    assert rep["evidence"] == "execution"


def test_execution_accounting_can_also_confirm_a_lane():
    plan = {"on_mesh": {"matmul": 15}, "scalar_rvv_lane": {"add": 3}}
    exec_ok = {"matmul_layers_routed": 15, "matmul_layers_on_mesh": 15,
               "matmul_layers_host_fallback": 0}
    rep = lane_report({"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}, plan, exec_ok)
    assert rep["unexercised"] == []


def test_a_plan_only_verdict_says_so():
    """Without execution accounting the report is about intent, and must admit it."""
    rep = lane_report({"lanes": {"require": ["on_mesh"]}}, {"on_mesh": {"matmul": 1}})
    assert rep["evidence"] == "routing_plan"
    assert "PLANNED" in rep["caveat"] or "planned" in rep["caveat"].lower()


def test_an_interop_capsule_runs_on_the_mesh_lane():
    """Withholding must_accelerate must not send the model to the host, or its own lane requirement
    can never be checked -- which is how the hollow pass happened."""
    import ast
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python/merlin/targetgen/capsule_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_grade_model_capsule")
    seg = ast.get_source_segment(src, fn) or ""
    assert '"on_mesh" in _req_lanes' in seg, (
        "a capsule REQUIRING on_mesh must run on the mesh lane so the requirement is verifiable")
