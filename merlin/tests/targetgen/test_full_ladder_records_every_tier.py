"""A capsule's grade must say what EVERY declared tier thought, not just the first refuter.

`tier_order` runs the ladder in COST order, and a mandatory failure used to raise from inside the tier
loop — so every tier ordered after the refuting one was left with no record at all. Not "skipped":
absent. Measured on `merlincirct_atlassg1` (26 public capsules):

    pass  L3=pass    L4=pass     14
    fail  L3=ABSENT  L4=fail     11
    fail  L3=fail    L4=ABSENT    1

Atlas's Verilator tier costs 0.29 s and its arc cosim 24.5 s, so the cheap tier refuted first and the
expensive one never ran. That makes "these 12 capsules failed" unanswerable at the other tier — which
is exactly the question to ask once a shared defect is fixed. `not_run_is_not_pass` already says an
unrun tier is not a pass; this says it must also not be INVISIBLE.
"""
from __future__ import annotations

import inspect
import os

from merlin.targetgen import capsule_runner as CR


def test_the_ladder_completes_by_default():
    """A grade whose numbers get quoted must be complete unless someone opts out explicitly."""
    os.environ.pop("MERLIN_FULL_LADDER", None)
    assert CR._full_ladder_enabled() is True


def test_the_opt_out_is_explicit_and_narrow(monkeypatch):
    for off in ("0", "false", "no", "FALSE"):
        monkeypatch.setenv("MERLIN_FULL_LADDER", off)
        assert CR._full_ladder_enabled() is False, off
    for on in ("1", "true", "", "anything"):
        monkeypatch.setenv("MERLIN_FULL_LADDER", on)
        assert CR._full_ladder_enabled() is True, on


def test_a_mandatory_failure_does_not_raise_from_inside_the_tier_loop():
    """The literal regression: `raise CertFailure` under `if not okt and mand` inside the loop."""
    src = inspect.getsource(CR.run_capsule)
    i = src.index("if not okt and mand:")
    seg = src[i:i + 1600]
    assert "_first_cert_failure" in seg, "the deferral is gone — the ladder aborts again"
    assert "if not _complete_ladder:" in seg, "the abort is no longer conditional on the opt-out"


def test_the_deferred_failure_is_still_raised():
    """Completing the ladder must not turn a failing capsule into a pass."""
    src = inspect.getsource(CR.run_capsule)
    assert "raise _first_cert_failure" in src, "a deferred failure is never raised — capsules would pass"
    # and it must be raised AFTER the tier loop, not before it
    assert src.index("for tier in _tier_seq:") < src.index("raise _first_cert_failure")


def test_the_first_refuting_plane_is_the_one_reported():
    """Completing the ladder must not relabel WHICH tier refuted the capsule."""
    src = inspect.getsource(CR.run_capsule)
    seg = src[src.index("if not okt and mand:"):src.index("if not _complete_ladder:")]
    assert "if _first_cert_failure is None:" in seg, \
        "a later tier could overwrite the first refuter and misreport the failure plane"
