"""A self-check reply the shim cannot read must never read as a clean run.

The grader prints human diagnostics ("tier plan: ...", "model gate: ...") on the SAME stdout that
carries the verdict JSON, and the broker forwards that stream verbatim. So the reply the shim parses is
routinely `<diagnostics>\n{...}` rather than a bare document.

The shim used to `json.loads` the whole thing and, on failure, `return 0` -- success. That is the
absent-measurement-reads-as-a-pass bug: the agent's own `echo $?`, the conformance probe, and the
shape-coverage gate all consume that exit code, so on any target whose grader printed a diagnostic the
self-check reported clean no matter what the capsules did.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))

import selfcheck_shim as S  # noqa: E402

_VERDICT = {"sim": "spike", "n_passed": 0, "n_capsules": 3, "all_pass": False}
_PLAN = "  tier plan: 6/23 eligible capsule(s) form the derived covering set (certified first)"


def test_a_bare_document_still_parses():
    assert S._verdict(json.dumps(_VERDICT))["n_capsules"] == 3


def test_the_verdict_survives_a_diagnostic_preamble():
    """The case that actually shipped."""
    got = S._verdict(f"{_PLAN}\n{json.dumps(_VERDICT, indent=2)}")
    assert got is not None and got["all_pass"] is False


def test_it_survives_diagnostics_on_both_sides():
    got = S._verdict(f"{_PLAN}\n{json.dumps(_VERDICT)}\n  model gate: denominator is 6 CERTIFIED\n")
    assert got == _VERDICT


def test_a_brace_in_the_preamble_does_not_derail_the_scan():
    """A `{` that starts no document must be stepped over, not fatal."""
    got = S._verdict(f"  note: shapes {{m,k,n}} were covered\n{json.dumps(_VERDICT)}")
    assert got == _VERDICT


@pytest.mark.parametrize("txt", ["", "   ", "Traceback (most recent call last):\n  boom\n", "[1, 2]"])
def test_an_unreadable_reply_is_none_not_an_empty_verdict(txt):
    """None is the signal to fail closed; a `{}` here would read as `all_pass` absent == falsy pass."""
    assert S._verdict(txt) is None
