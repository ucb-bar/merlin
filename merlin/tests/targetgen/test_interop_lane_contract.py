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


def test_the_runner_checks_every_required_lane_against_the_routing_plan():
    src = _RUNNER.read_text(encoding="utf-8")
    assert "lanes" in src and "routing_plan" in src, "the runner must honor the lane contract"
    # the check must consult the routing plan the COMPILER reported, not a hardcoded lane list
    seg = src.split("_req_lanes", 1)[1][:1200]
    assert "routing_plan" in seg, "lane verification must read the reported routing plan"
    assert "unexercised" in seg, (
        "a required lane that carried no work must be NAMED in the result — an unnamed failure turns a "
        "composition capsule into a single-backend capsule nobody notices")


def test_lane_names_are_not_invented_here():
    """The vocabulary must be the routing plan's, so a target that reports different lanes still works."""
    src = _RUNNER.read_text(encoding="utf-8")
    seg = src.split("_req_lanes", 1)[1][:1200]
    # the required lanes come from the capsule; the observed set comes from the plan's own keys
    assert "capsule.get(\"lanes\")" in src or "capsule.get('lanes')" in src
    assert ".items()" in seg or "_plan.get" in seg, (
        "observed lanes must be read off the plan's keys rather than compared to a fixed list")
