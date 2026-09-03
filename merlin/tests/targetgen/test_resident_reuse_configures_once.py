"""A resident-weight program configures the execution mode ONCE, and reconfigures only on a change.

The point of a resident-weight capsule is that the weight-stationary mode is established once and then
reused by every activation that consumes the resident weight. A backend that re-issues the mode
configuration per command still computes the right answer -- the second write is bit-identical to the
first -- so numerics, L0/L1/L2 and the L3 RTL oracle all pass, and only the trace conformance rule
notices. That is exactly how it was found: capsule ``A6_resident_reuse`` passed every tier and reported
``mode resident_reuse: expected a single weight-stationary config, saw 2 CONFIG_EX``, which held the
functional freeze on one row.

These tests pin the rule that caught it, in both directions:

* the pre-fix instruction sequence (two identical CONFIG_EX around two output commits) is REJECTED;
* the post-fix sequence (one CONFIG_EX, still two commits) is ACCEPTED;
* and the rule does not degenerate into "one CONFIG_EX is always fine" -- a single-commit trace still
  fails the ``resident_reuse`` declaration, because no reuse is visible in it.

Class names here are the decoder's own derived classes, and the expectations come from a capsule's
``expected`` block, so nothing in this file assumes an opcode, a funct value or a target.
"""
from __future__ import annotations

from merlin.targetgen import trace_check as TCK

# The declaration a resident-reuse capsule carries: the classes its trace must contain, and the mode
# whose *shape* is being asserted.
_EXPECTED = {
    "instruction_classes": ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
                            "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"],
    "modes": {"resident_reuse": True},
}

# Every class carries a funct so ``drives_accelerator`` (the one gating signal) is satisfied and the
# only findings under test are the mode findings.
_FUNCT = {"FENCE": None, "FLUSH": 7, "CONFIG_EX": 0, "CONFIG_LD": 0, "CONFIG_ST": 0,
          "MVIN": 2, "MVOUT": 3, "PRELOAD": 6, "COMPUTE_PRELOADED": 4}


def _trace(classes: list[str]) -> dict:
    return {"instructions": [{"index": i, "class": c, "funct": _FUNCT[c], "decoded": {}}
                             for i, c in enumerate(classes)]}


def _commit(*, config_ex: bool) -> list[str]:
    """One output commit: optionally (re)configure the mode, then load, compute, store."""
    return (["CONFIG_EX"] if config_ex else []) + [
        "CONFIG_ST", "CONFIG_LD", "MVIN", "CONFIG_LD", "MVIN",
        "PRELOAD", "COMPUTE_PRELOADED", "MVOUT",
    ]


def _program(*, config_ex_per_commit: bool, commits: int = 2) -> list[str]:
    head = ["FENCE", "FLUSH"] + ([] if config_ex_per_commit else ["CONFIG_EX"])
    body: list[str] = []
    for n in range(commits):
        body += _commit(config_ex=config_ex_per_commit)
    return head + body + ["FENCE"]


def _mode_violations(result: dict) -> list[str]:
    return [v for v in result["violations"] if v.startswith("mode resident_reuse")]


def test_a_per_command_execution_mode_config_is_rejected():
    """The DEFECT, verbatim: two output commits, each opening with its own CONFIG_EX."""
    trace = _trace(_program(config_ex_per_commit=True))
    assert [i["class"] for i in trace["instructions"]].count("CONFIG_EX") == 2
    result = TCK.check(trace, _EXPECTED)
    assert result["status"] == "fail"
    assert _mode_violations(result) == [
        "mode resident_reuse: expected a single weight-stationary config, saw 2 CONFIG_EX"]


def test_a_program_scope_execution_mode_config_is_accepted():
    """The FIX: the mode is hoisted to program scope, so both commits share one CONFIG_EX."""
    trace = _trace(_program(config_ex_per_commit=False))
    classes = [i["class"] for i in trace["instructions"]]
    assert classes.count("CONFIG_EX") == 1
    assert classes.count("MVOUT") == 2, "the reuse must still be visible as two output commits"
    result = TCK.check(trace, _EXPECTED)
    assert _mode_violations(result) == []
    assert result["status"] == "pass", result["violations"]


def test_hoisting_the_config_does_not_by_itself_satisfy_resident_reuse():
    """One CONFIG_EX is not the whole rule: a single-commit program shows no reuse and still fails.

    Without this, "emit one CONFIG_EX" would be satisfiable by a backend that lowered a resident-reuse
    program to a single matmul and dropped the second activation entirely.
    """
    trace = _trace(_program(config_ex_per_commit=False, commits=1))
    result = TCK.check(trace, _EXPECTED)
    assert _mode_violations(result) == [
        "mode resident_reuse declared but <2 output commits (no reuse visible)"]


def test_the_single_config_still_precedes_the_first_compute():
    """Hoisting must not move the configuration after the work it configures."""
    classes = _program(config_ex_per_commit=False)
    late = [c for c in classes if c != "CONFIG_EX"]
    late.insert(late.index("MVOUT") + 1, "CONFIG_EX")  # after the first commit's readout
    result = TCK.check(_trace(late), _EXPECTED)
    assert any("CONFIG_EX appears after first PRELOAD/COMPUTE" in v for v in result["violations"])
