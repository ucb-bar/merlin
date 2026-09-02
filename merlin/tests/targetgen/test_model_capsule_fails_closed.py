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


@pytest.fixture(autouse=True)
def _descriptor_pinned_host_lane(monkeypatch):
    """These tests isolate tier accounting from the independently tested host-lane resolver."""
    from merlin.common.paths import repo_root
    monkeypatch.setattr(CR, "_resolve_model_host_lane", lambda target, dtype: (
        None, repo_root() / "frozen-test-host", {
            "package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8",
        }))


#: The frozen model capsule these tests grade, and the assets materialization needs from it.
#:
#: A whole-model grade materializes the capsule from its frozen source directory BEFORE any tier
#: accounting happens. A synthetic dict has no `__dir__`, so production correctly refused with
#: "model capsule has no frozen source directory" and every assertion below then fell over on that
#: error instead of on the tier ladder it is testing. Production is right to fail closed -- a grade
#: with no frozen source cannot be attributed -- so these tests take a REAL capsule and override the
#: one field they vary. Same fix as test_model_host_lane_pin.py.
_FROZEN_MODEL_DIR = "merlin/contract/capsules/model/M3_host_island_seam_gemmini"

#: `golden.yaml` is untracked by design (answer surfaces never enter the public repo), so a fresh
#: worktree has the capsule but not its golden. Skip and name the absent asset rather than failing
#: for a reason that has nothing to do with tier accounting.
_REQUIRED_ASSETS = ("capsule.yaml", "capsule.interface.mlir", "capsule.pytorch.py",
                    "capsule.weights.safetensors", "golden.yaml")


def _capsule(tiers):
    from merlin.common.paths import repo_root

    root = repo_root() / _FROZEN_MODEL_DIR
    missing = [n for n in _REQUIRED_ASSETS if not (root / n).is_file()]
    if missing:
        pytest.skip(f"frozen model capsule at {_FROZEN_MODEL_DIR} is missing {missing} "
                    f"(answer surfaces are untracked by design, so a fresh worktree has none)")
    cap = dict(CR.load_capsule(root))          # copy: these tests mutate the tier list per case
    cap["required_oracle_tiers"] = list(tiers)
    return cap


def _grade(monkeypatch, capsule, out):
    """Drive _grade_model_capsule with a stubbed compile_model returning *out*."""
    import merlin.compile_cli as cc
    monkeypatch.setattr(cc, "compile_model", lambda *a, **k: out, raising=False)
    return CR._grade_model_capsule(capsule, target="gemmini", timeout=10)


def _statuses(r):
    """Tier -> status, from the rich per-tier objects the merged row carries.

    The row is the same shape an op capsule produces, which is what capsule_result.schema.json requires
    and what routes it through the shared fail-closed gates. A tier that is honestly NOT APPLICABLE to a
    whole model (L0/L1 interpret a command buffer; a model has none) is reported as such rather than
    omitted, so `passed()` below asks the question that matters: did any declared tier actually certify?
    """
    return {t: (v or {}).get("status") for t, v in (r.get("tiers") or {}).items()}


def _passed(r):
    """The tiers that actually certified — the guarantee, independent of how the row is shaped."""
    return {t: v for t, v in _statuses(r).items() if v == "pass"}


def test_no_declared_tier_ran_is_never_a_pass(monkeypatch):
    """The core case: the host run verified the numbers, but no declared oracle tier executed."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True}})
    assert r["status"] == "incomplete"                    # NOT pass
    assert _passed(r) == {}, "no tier certified this model"
    assert r["numeric"]["status"] == "not_compared"       # no comparison is not a passing comparison
    assert r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert "ran NONE of them" in r["failure"]["detail"]


def test_a_tier_that_actually_ran_can_pass(monkeypatch):
    # the TILE certification lives under its own key; `mesh_execution` is the separate record of what
    # happened to the model's own layers (they shared one key, and the tile record clobbered the model one)
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                # A REAL tile record carries its counts. `ok: True` alone is an aggregate with no
                # evidence under it, and the grader is right to refuse one -- that is the same shape as
                # every other "the summary says pass" defect this suite exists to catch.
                "mesh_tile_verification": {"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                                           "n_unavailable": 0, "n_unsynthesizable": 0, "ok": True},
                # ...and the MODEL's own layers. A tile record alone must not carry a model capsule:
                # certifying a synthesized tile proves the SHAPE is runnable, while the capstone is a
                # claim about THIS model, and the two came apart once already -- a run with all 15
                # layers on the host reported "15 of 15 tiles passed".
                "mesh_execution": {"target": "gemmini", "matmul_layers_routed": 15,
                                   "matmul_layers_on_mesh": 15,
                                   "matmul_layers_host_fallback": 0},
                # The frozen capsule this fixture loads is an INTEROP capstone: it requires the host
                # lane as well as the mesh, because composition across the two is the behaviour under
                # test. A stub that reports only the mesh leaves the other required lane unmeasured,
                # and an unmeasured required lane is not a pass.
                "host_execution": {"kernels_ran": 12, "contractions_ran": 0}})
    assert r["status"] == "pass"
    assert _passed(r) == {"L3": "pass"}                   # the declared RTL tier, named from the capsule
    # honest about what did NOT run, AND why. A bare list read as "we skipped three of the four tiers
    # this capsule declares", which is indistinguishable from a grade that cut corners; the real reason
    # is that those tiers grade a per-op command buffer a model capsule never produces.
    unex = r.get("tiers_unexercised")
    assert sorted(unex) == ["L0", "L1", "L2"]
    assert all(v and "no whole-model analogue" in v for v in unex.values()), unex


def test_a_failing_mesh_execution_is_recorded_as_a_failing_tier(monkeypatch):
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L3", "L4"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_tile_verification": {"n_tiles": 15, "ok": False}})
    assert _statuses(r).get("L4") == "fail"               # last non-structural declared tier
    assert _passed(r) == {}, "a failing mesh run certifies nothing"


def test_a_routing_plan_alone_is_not_a_tier(monkeypatch):
    """A plan is a plan. Counting it would let 'we intend to use the mesh' read as 'we used the mesh'."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "routing_plan": {"n_mesh_ops": 15}})
    assert r["status"] == "incomplete" and _passed(r) == {}


def test_zero_tiles_is_not_an_exercised_tier(monkeypatch):
    """The vacuous-mesh trap: with 0 mesh ops _mesh_verify returns n_tiles=0 and reads clean."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L3", "L4"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "mesh_execution": {"n_tiles": 0, "ok": True}})
    assert r["status"] == "incomplete" and _passed(r) == {}


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


def test_declining_a_region_the_hardware_admits_is_a_failure(monkeypatch):
    """The coverage certificate counted `false_fallback` -- regions the manifest ADMITS that the router
    left on the host -- and no verdict ever read the count. That is a claim about the router's own
    choice, so plan-derived evidence is the right evidence for it."""
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "coverage_certificate": {"false_fallback_count": 3},
                "mesh_tile_verification": {"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                                           "n_unavailable": 0, "n_unsynthesizable": 0, "ok": True},
                "mesh_execution": {"target": "gemmini", "matmul_layers_routed": 15,
                                   "matmul_layers_on_mesh": 15, "matmul_layers_host_fallback": 0},
                "host_execution": {"kernels_ran": 12, "contractions_ran": 0}})
    assert r["status"] == "fail"
    assert r["failure"]["category"] == "FALLBACK_ON_ELIGIBLE_REGION"
    assert "ADMITS" in r["failure"]["detail"]


def test_no_declined_region_leaves_the_verdict_alone(monkeypatch):
    r = _grade(monkeypatch, _capsule(["L0", "L1", "L2", "L3"]),
               {"status": "verified", "verify": {"gate_ok": True},
                "coverage_certificate": {"false_fallback_count": 0},
                "mesh_tile_verification": {"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                                           "n_unavailable": 0, "n_unsynthesizable": 0, "ok": True},
                "mesh_execution": {"target": "gemmini", "matmul_layers_routed": 15,
                                   "matmul_layers_on_mesh": 15, "matmul_layers_host_fallback": 0},
                "host_execution": {"kernels_ran": 12, "contractions_ran": 0}})
    assert r["status"] == "pass"
