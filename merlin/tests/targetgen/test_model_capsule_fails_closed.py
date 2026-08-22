"""A whole-model capsule must not report a verdict no oracle backed.

Measured: model capsules returned before the tier ladder, so their declared required_oracle_tiers were
dead metadata and every result carried `tiers == {}`. The functional gate is the HOST x86 run unless
mesh execution is requested, so a `pass` could mean "the CPU computed the model correctly" — not a
statement about the accelerator at all.
"""

from __future__ import annotations

import types

import pytest

from merlin.targetgen import capsule_runner as CR


def _capsule(tiers):
    return {"name": "M0_x", "kind": "model", "label": "public",
            "required_oracle_tiers": list(tiers),
            "operation": {"op": "model", "attributes": {"model": "m", "compile_dtype": "int8",
                                                        "dtype": "int8"}}}


def _grade(monkeypatch, capsule, out):
    """Drive _grade_model_capsule with a stubbed compile_model returning *out*."""
    import merlin.compile_cli as cc
    monkeypatch.setattr(cc, "compile_model", lambda *a, **k: out, raising=False)
    return CR._grade_model_capsule(capsule, target="gemmini", timeout=10)


def test_no_declared_tier_ran_is_never_a_pass(monkeypatch):
    """The core case: the host run verified the numbers, but no declared oracle tier executed."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True}})
    assert r["status"] == "incomplete"                    # NOT pass
    assert r["tiers"] == {}
    assert r["numeric"]["status"] == "not_compared"       # no comparison is not a passing comparison
    assert r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert "ran NONE of them" in r["failure"]["detail"]


def test_a_tier_that_actually_ran_can_pass(monkeypatch):
    # the TILE certification lives under its own key; `mesh_execution` is the separate record of what
    # happened to the model's own layers (they shared one key, and the tile record clobbered the model one)
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_tile_verification": {"n_tiles": 15, "ok": True}})
    assert r["status"] == "pass"
    assert r["tiers"] == {"L3": "pass"}                   # the declared RTL tier, named from the capsule
    assert r.get("tiers_unexercised") == ["L0", "L1", "L2"]   # honest about what did NOT run


def test_a_failing_mesh_execution_is_recorded_as_a_failing_tier(monkeypatch):
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L3", "L4"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_tile_verification": {"n_tiles": 15, "ok": False}})
    assert r["tiers"] == {"L4": "fail"}                   # last non-structural declared tier


def test_a_routing_plan_alone_is_not_a_tier(monkeypatch):
    """A plan is a plan. Counting it would let 'we intend to use the mesh' read as 'we used the mesh'."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "routing_plan": {"n_mesh_ops": 15}})
    assert r["status"] == "incomplete" and r["tiers"] == {}


def test_zero_tiles_is_not_an_exercised_tier(monkeypatch):
    """The vacuous-mesh trap: with 0 mesh ops _mesh_verify returns n_tiles=0 and reads clean."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L3", "L4"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_execution": {"n_tiles": 0, "ok": True}})
    assert r["status"] == "incomplete" and r["tiers"] == {}


def test_the_op_pass_fraction_is_labelled_op_coverage(monkeypatch):
    r = _grade(monkeypatch, _capsule(["L0"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_execution": {"n_tiles": 4, "ok": True}})
    assert "OP COVERAGE" in r["op_coverage"]["note"]
    assert "not a verdict" in r["op_coverage"]["note"]


def test_an_incomplete_capstone_deliberately_blocks_all_pass():
    """POLICY, recorded so it is not "fixed" by accident.

    `not_graded` (hardware cannot) and `gated` (not yet attempted) are excluded from the denominator so
    all_pass stays reachable. `incomplete` is NOT excluded, and that is deliberate: excluding it would let
    a submission claim all_pass while the whole-model capstone silently never ran — the exact vacuity this
    whole change exists to remove. The cost is that all_pass stays unreachable until the capstone really
    executes; the round-budget economics are handled by --plateau-rounds, not by relaxing this.
    """
    results = ([{"capsule": f"p{i}", "status": "pass"} for i in range(22)]
               + [{"capsule": f"b{i}", "status": "not_graded"} for i in range(12)]
               + [{"capsule": "M0", "status": "incomplete"}, {"capsule": "M1", "status": "incomplete"}])
    graded = [r for r in results if r["status"] not in ("not_graded", "gated")]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    assert (len(graded), n_pass) == (24, 22)
    assert n_pass != len(graded), "an unrun capstone must not be silently forgiven"
