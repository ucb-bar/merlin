"""A run whose oracle never ran must not present n_passed as a capability number.

The scorecard already fails closed on an EMPTY suite ("the exact vacuous-pass trap"). It had no
equivalent for the case where capsules WERE selected and graded but their mandatory tier came back
`unavailable` — the oracle was missing, so nothing was learned about the submission.

Measured: re-grading a frozen submission whose own run scored 33/36 produced `n_passed: 6` with 29
capsules incomplete ("no derived ISA encoding fact for target 'radiance'"). The six that "passed" were
MX fixtures, which need no oracle at all. `n_not_gradeable_no_oracle` stayed 0 and nothing in the
rollup said the run was invalid — a reader would have quoted 6/36 as a capability collapse when it was
an environment gap.
"""
from __future__ import annotations

import merlin.targetgen.capsule_grade as CG


def _score_from(rows, *, no_oracle=False):
    """Drive only the rollup: build a score dict the way grade() would, from canned per-capsule rows."""
    score = {"per_capsule": rows, "n_capsules": len(rows),
             "n_passed": sum(1 for r in rows if r["status"] == "pass")}
    graded = [r for r in rows if r["status"] in ("pass", "fail", "error", "incomplete",
                                                 "not_gradeable_no_oracle")]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    n_ng = sum(1 for r in graded if r["status"] == "not_gradeable_no_oracle")
    _empty = len(graded) == 0
    inc = [r for r in graded if r.get("status") == "incomplete"]
    score["n_incomplete"] = len(inc)
    if inc:
        score["measurement_incomplete"] = {
            "n": len(inc), "of": len(graded),
            "reasons": sorted({(r.get("failure") or {}).get("tier_reason") or "unknown" for r in inc}),
        }
    score["gradeable"] = (not no_oracle) and not _empty and not inc
    score["n_structural_pass"] = n_pass + n_ng
    return score


def _row(name, status, reason=None):
    r = {"capsule": name, "status": status}
    if reason:
        r["failure"] = {"tier_reason": reason, "tier_status": "unavailable"}
    return r


def test_a_healthy_run_is_gradeable_and_carries_no_marker():
    s = _score_from([_row("A", "pass"), _row("B", "pass"), _row("C", "fail")])
    assert s["gradeable"] is True
    assert s["n_incomplete"] == 0
    assert "measurement_incomplete" not in s


def test_a_missing_oracle_makes_the_run_not_gradeable():
    """The shape that actually happened: a few fixtures pass, everything real is incomplete."""
    rows = [_row("FIXTURE1", "pass"), _row("FIXTURE2", "pass")]
    rows += [_row(f"REAL{i}", "incomplete", "no derived ISA encoding fact for target 'radiance'")
             for i in range(29)]
    s = _score_from(rows)
    assert s["n_passed"] == 2                 # the number a reader would otherwise quote
    assert s["gradeable"] is False            # ...and the flag that says not to
    assert s["n_incomplete"] == 29
    assert "no derived ISA encoding fact" in s["measurement_incomplete"]["reasons"][0]


def test_incomplete_is_not_confused_with_a_withheld_verdict():
    """`not_gradeable_no_oracle` is --no-oracle DELIBERATELY withholding a verdict; `incomplete` is a run
    that believed it was grading and found the tool absent. Conflating them would let a broken
    environment read as an intentional structure-only smoke."""
    s = _score_from([_row("A", "not_gradeable_no_oracle"), _row("B", "not_gradeable_no_oracle")],
                    no_oracle=True)
    assert s["n_incomplete"] == 0
    assert "measurement_incomplete" not in s
    assert s["n_structural_pass"] == 2


def test_a_failure_is_still_a_real_measurement():
    """A FAIL means the submission was wrong — that IS a measurement and must not be suppressed."""
    s = _score_from([_row("A", "pass"), _row("B", "fail"), _row("C", "fail")])
    assert s["gradeable"] is True and s["n_incomplete"] == 0


def test_the_real_grader_carries_the_same_guard():
    """Pin it in grade() itself, not only in this local re-implementation."""
    import inspect
    src = inspect.getsource(CG.grade)
    assert "n_incomplete" in src and "measurement_incomplete" in src
    assert "not _incomplete" in src, "gradeable must be False when a mandatory oracle never ran"
