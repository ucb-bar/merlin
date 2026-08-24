"""The launch preflight must be able to see what it checks, and its verdict must stop the launch.

Both halves failed on a live launch, and they hid each other:

  * ``_run_preflight`` locked every answer surface (``chmod -R 000``) BEFORE running verify_no_cheat. The
    held-out check derives holdout capsule NAMES by walking ``hidden/*/capsule.yaml``, so after the lock it
    saw nothing. While "no names" was read as "nothing to check" this was invisible; once the check was made
    to fail closed it became a guaranteed failure -- the honest symptom of an ordering bug always present.
  * ``chia_ab_batch`` called ``LB._run_preflight()`` and discarded the return code, so the run printed
    "VERIFY_NO_CHEAT: FAIL -- DO NOT launch" and launched.

A gate that cannot see, whose verdict is thrown away, is decoration.
"""
from __future__ import annotations

import ast
import sys

from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"


def _src(name: str) -> str:
    return (HARNESS / name).read_text(encoding="utf-8")


def _fn(name: str, mod: str) -> ast.FunctionDef:
    tree = ast.parse(_src(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {mod}")


def test_verification_runs_before_the_answer_lock():
    """Structural, not textual: compare the LINE where the chmod happens to where verify runs."""
    fn = _fn("_run_preflight", "launch_ab_batch.py")
    chmod_lines, vnc_lines = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "000":
                chmod_lines.append(node.lineno)
            elif node.value == "verify_no_cheat.py":
                vnc_lines.append(node.lineno)
    assert chmod_lines, "the answer-surface lock disappeared from the preflight"
    assert vnc_lines, "verify_no_cheat is no longer invoked by the preflight"
    assert min(vnc_lines) < min(chmod_lines), (
        "verify_no_cheat must run BEFORE the chmod 000 lock — locking first blinds the held-out check, "
        "which must read hidden/*/capsule.yaml to know what to look for")


def test_the_lock_still_happens_in_the_preflight():
    """Verifying first must not become 'not locking': the surfaces are locked before any agent spends."""
    fn = _fn("_run_preflight", "launch_ab_batch.py")
    assert any(isinstance(n, ast.Constant) and n.value == "000" for n in ast.walk(fn))


def test_the_preflight_verdict_gates_the_launch():
    """The return value must be bound and returned, not called for its side effects."""
    tree = ast.parse(_src("chia_ab_batch.py"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_run_preflight"]
    assert calls, "chia_ab_batch no longer runs the preflight at all"
    # every call must be part of an assignment (its value used), never a bare expression statement
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "_run_preflight"]
    assert not bare, ("_run_preflight's return code is discarded — a failed preflight would print "
                      "'DO NOT launch' and then launch")
