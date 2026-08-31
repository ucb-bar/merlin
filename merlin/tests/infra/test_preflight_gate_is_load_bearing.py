"""The launch preflight must be able to see what it checks, and its verdict must stop the launch.

Two independent failures occurred on a live launch:

  * ``_run_preflight`` chmod-000-locked every answer surface. The host anti-cheat and hidden grader run as
    the same user, so this blinded both of them. The real isolation boundary is bwrap; host trees must stay
    owner-readable while the sandbox mount table masks them from the agent.
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


def test_host_access_is_restored_before_verification():
    """A stale mode-000 run must be repaired before the host tries to enumerate hidden capsules."""
    fn = _fn("_run_preflight", "launch_ab_batch.py")
    prepare_lines, vnc_lines = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_make_host_owner_only":
                prepare_lines.append(node.lineno)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "verify_no_cheat.py":
                vnc_lines.append(node.lineno)
    assert prepare_lines, "the host-readable answer-surface preparation disappeared"
    assert vnc_lines, "verify_no_cheat is no longer invoked by the preflight"
    assert min(prepare_lines) < min(vnc_lines), (
        "host access must be restored before verify_no_cheat walks hidden/*/capsule.yaml")


def test_host_protection_is_owner_only_never_mode_zero():
    """The host grader retains access; the agent is isolated by the separately tested bwrap masks."""
    fn = _fn("_make_host_owner_only", "launch_ab_batch.py")
    modes = {node.value for node in ast.walk(fn)
             if isinstance(node, ast.Constant) and isinstance(node.value, int)}
    assert 0o700 in modes and 0o600 in modes
    assert 0 not in modes, "mode 000 blinds same-UID host grading and is not an agent security boundary"


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
