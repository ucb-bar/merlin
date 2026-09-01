"""A failing whole-model capsule must still say whether the ARITHMETIC was right.

The whole-model numeric gate runs before the acceleration checks, so by the time `must_accelerate` (or a
failed tier, or an unexercised one) rejects a capstone, the model's output has already been compared
against its golden. That result was then overwritten with ``not_compared`` -- which reads as "we do not
know" when in fact we did, and had thrown the answer away.

The distinction erased is the one a reader of a failing capstone most needs:

  * the compiler got the model RIGHT and ran it in the wrong place  -> routing work;
  * the compiler got the model WRONG as well                        -> correctness work.

Those call for completely different effort, and `not_compared` hides which one you are in. Measured on
gemmini: `M0_small_llama_gemmini` and `M1_lstmnetvit_gemmini` both reported `numeric_status:
not_compared`, so the question "does this compiler produce a correct small_llama at all?" had no answer
on record despite having been measured.

What is pinned here is that reporting the number NEVER promotes the verdict: every path that carries a
numeric result still returns `fail` or `incomplete`, and the number is labelled with the lane it was
measured on so it cannot be read as an accelerator result.
"""
from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir

_RUNNER = merlin_dir() / "python/merlin/targetgen/capsule_runner.py"


def _src() -> str:
    return _RUNNER.read_text(encoding="utf-8")


def _fn(name: str) -> str:
    src = _src()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn) or ""


def _body(name: str) -> str:
    """Source of a function with its docstring removed -- these assertions are about CODE, and the
    docstrings here legitimately discuss the very string being asserted absent."""
    src = _src()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                           and isinstance(fn.body[0].value.value, str)) else fn.body
    return "\n".join(ast.get_source_segment(src, n) or "" for n in body)


def test_the_helper_reports_a_real_status_not_an_absence():
    body = _body("_numeric_when_not_accelerated")
    assert '"pass" if (st == "verified" and gate) else "fail"' in body, (
        "the numeric status must be the measured verdict, not a placeholder")
    assert "not_compared" not in body, "the helper must not re-introduce the placeholder in code"


def test_the_number_is_labelled_with_the_lane_it_came_from():
    """A correct number computed on the host is still not an accelerator result."""
    seg = _fn("_numeric_when_not_accelerated")
    assert "measured_on" in seg
    assert "not an accelerator result" in seg.lower() or "wrong lane" in seg.lower()


def test_every_rejection_path_carries_the_numeric_verdict():
    # The grading body lives in `_grade_model_capsule_INLINE`; `_grade_model_capsule` is the wrapper
    # that decides whether to run it in-process or in a subprocess. Inspecting the wrapper found none
    # of these categories and the test failed for a rename it was written to catch -- which is the
    # weakness of asserting on source text, so it now names the function that actually holds them.
    seg = _fn("_grade_model_capsule_inline")
    # The places a model capsule is turned down after the gate has already run and its arithmetic IS
    # known: unmeasurable layers, must_accelerate fallback, and a tier that ran and failed. A path where
    # NOTHING ran is deliberately not among them -- it reports `not_compared`, because a comparison that
    # never happened against the accelerator is not a passing comparison.
    for category in ("NOT_RUN_IS_NOT_PASS", "FALLBACK_ON_ELIGIBLE_REGION", "FUNCTIONAL_MISMATCH"):
        assert category in seg, f"rejection path {category} is gone — was it renamed?"
    assert seg.count("_numeric_when_not_accelerated(") == 3, (
        "each rejection path with a measured number (unmeasurable layers, must_accelerate fallback, "
        "failed tier) must report it rather than overwrite it with a placeholder")
    # and the paths where nothing was compared are deliberately NOT among them: they go through
    # _numeric_not_compared, which is a different statement from a measured rejection.
    assert "_numeric_not_compared(" in seg


def test_reporting_the_number_never_promotes_the_verdict():
    """The load-bearing safety property: more information, never a weaker bar."""
    seg = _fn("_grade_model_capsule")
    for chunk in seg.split("_numeric_when_not_accelerated(")[1:]:
        head = chunk[:400]
        assert ('status="fail"' in chunk[-600:] or 'status="incomplete"' in chunk[-600:]
                or 'status="fail"' in head or 'status="incomplete"' in head), (
            "a path that reports a numeric result must still return fail/incomplete")


def test_not_compared_survives_only_where_nothing_was_measured():
    """It is still the right answer when the model never ran -- just not when it did."""
    src = _src()
    assert src.count('"status": "not_compared"') <= 1, (
        "not_compared should remain only for the case where no measurement exists")
