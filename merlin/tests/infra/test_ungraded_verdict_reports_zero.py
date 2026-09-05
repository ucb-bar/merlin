"""A verdict for a submission that was never graded must not look like a grade.

MEASURED COST: the two failure stubs in the QA loop reported ``n_capsules: 4``. That renders in the
round log as ``0/4``, which is byte-identical to a genuine grade of four capsules, and the only thing
distinguishing them is an empty ``per_capsule``. A reader watching a live run spent an hour believing
its graded denominator had collapsed from 96 to 4 and raised it as a possible launch defect. Nothing
was wrong with the run; the number was a placeholder.

This is the same family as every other defect this corpus keeps finding: a value that cannot be told
apart from a real one, in a path nobody exercises deliberately.
"""
from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir

# Resolved through the repo's own path helper, never `Path(__file__).parents[N]`: this file moved once
# already while I was writing it and the index arithmetic silently pointed at a directory that does not
# exist, which the vacuity guard below caught and a `parents[N]` would not have.
DRIVER = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "run_baseline_qa_loop.py"


def _stub_verdicts() -> list[dict]:
    """Every dict literal in the driver that carries a ``package_failure`` — the ungraded stubs.

    Parsed structurally rather than matched as text: a stub spelled across different lines, or a third
    one added later, must be covered by this test the day it appears.
    """
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "package_failure" not in keys:
            continue
        out.append({k: v for k, v in zip(keys, node.values)})
    return out


def test_the_driver_has_ungraded_stubs_to_check():
    """A test that found nothing would pass vacuously and report the property as verified."""
    assert _stub_verdicts(), "no package_failure verdict literal found; this test cannot fail"


def test_a_verdict_that_graded_nothing_reports_zero_capsules():
    for stub in _stub_verdicts():
        node = stub.get("n_capsules")
        assert isinstance(node, ast.Constant), "n_capsules must be a literal in a failure stub"
        assert node.value == 0, (
            f"an ungraded verdict reports n_capsules={node.value!r}; it renders as '0/{node.value}' "
            f"in the round log and is indistinguishable from a real grade of that many capsules")


def test_an_ungraded_verdict_carries_no_per_capsule_rows():
    """The count and the rows must agree, or one of them is lying about what happened."""
    for stub in _stub_verdicts():
        rows = stub.get("per_capsule")
        assert isinstance(rows, ast.List) and not rows.elts, (
            "an ungraded verdict must carry no per-capsule rows")
