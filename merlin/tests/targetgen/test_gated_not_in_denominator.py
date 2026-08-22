"""A capsule deferred by its own gate is in neither the numerator nor the denominator.

The agent loop's ONLY early exit is a genuine all_pass. A whole-model capstone that is permanently
deferred (its op-pass gate never clears) therefore makes all_pass unreachable and forces every run to
buy its entire round budget. Measured: a 28-capsule grade of {pass 14, fail 12, not_graded 1, gated 2}
could not reach all_pass no matter what the agent produced.
"""

from __future__ import annotations


def _score(results):
    """The numerator/denominator arithmetic under test, mirroring capsule_grade."""
    ungraded = [r for r in results if r.get("status") == "not_graded"]
    deferred = [r for r in results if r.get("status") == "gated"]
    graded = [r for r in results if r.get("status") not in ("not_graded", "gated")]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    return {"n_capsules": len(graded), "n_passed": n_pass,
            "n_not_graded_ineligible": len(ungraded), "n_gated_deferred": len(deferred),
            "functional_pass": int(n_pass == len(graded) and len(graded) > 0)}


def test_the_measured_atlas_shape_becomes_reachable():
    results = ([{"capsule": f"p{i}", "status": "pass"} for i in range(14)]
               + [{"capsule": f"f{i}", "status": "fail"} for i in range(12)]
               + [{"capsule": "AF12", "status": "not_graded"}]
               + [{"capsule": "M0", "status": "gated"}, {"capsule": "M1", "status": "gated"}])
    s = _score(results)
    assert s["n_capsules"] == 26 and s["n_passed"] == 14        # was 28 -- the 2 gated inflated it
    assert s["n_gated_deferred"] == 2 and s["n_not_graded_ineligible"] == 1
    assert s["functional_pass"] == 0                            # 14 != 26, still honestly failing


def test_all_pass_is_reachable_once_the_gated_are_excluded():
    """The economic point: with every graded capsule passing, all_pass is TRUE and the loop can exit."""
    results = ([{"capsule": f"p{i}", "status": "pass"} for i in range(22)]
               + [{"capsule": "M0", "status": "gated"}, {"capsule": "M1", "status": "gated"}])
    s = _score(results)
    assert s["n_capsules"] == 22 and s["n_passed"] == 22
    assert s["functional_pass"] == 1                            # reachable
    # counting the gated capsules instead: 22 != 24, unreachable forever
    assert 22 != len(results)


def test_a_gated_capsule_is_never_counted_as_a_pass():
    s = _score([{"capsule": "M0", "status": "gated"}])
    assert s["n_passed"] == 0 and s["n_capsules"] == 0
    assert s["functional_pass"] == 0                            # empty suite is not a phantom pass


def test_gated_and_not_graded_are_reported_under_distinct_names():
    """They are different facts: one the hardware cannot do, one has not been attempted yet."""
    s = _score([{"capsule": "a", "status": "pass"},
                {"capsule": "b", "status": "not_graded"},
                {"capsule": "c", "status": "gated"}])
    assert s["n_not_graded_ineligible"] == 1 and s["n_gated_deferred"] == 1
