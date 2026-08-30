"""A capsule's grade must say what EVERY declared tier thought, not just the first refuter.

`tier_order` runs the ladder in COST order, and a mandatory failure used to raise from inside the tier
loop — so every tier ordered after the refuting one was left with no record at all. Not "skipped":
absent. Measured on `merlincirct_atlassg1` (26 public capsules):

    pass  L3=pass    L4=pass     14
    fail  L3=ABSENT  L4=fail     11
    fail  L3=fail    L4=ABSENT    1

Atlas's Verilator tier is the cheaper of the two (measured serially over 42 samples: Verilator median
0.276 s, arc cosim median 3.68 s -- NOT the 24.5 s that appears in tier_policy's docstring, which is a
throughput figure under 16-way parallelism), so the cheap tier refuted first and the dearer one never
ran. That makes "these 12 capsules failed" unanswerable at the other tier — which
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


# --- the OPT-OUT path must still record ------------------------------------------------------------
# MERLIN_FULL_LADDER=0 restores the short-circuit for a fast iteration loop. That is a cost choice, not
# a licence to record nothing: a tier the ladder declined to run must still SAY it was not run.
# `suppressed_tier_result` is what says it, and it was silently dropped by a merge while its test kept
# importing it -- so the invariant it encodes (a skipped tier says `skipped`, never `fail`) went
# unenforced. Recording `fail` would be worse than recording nothing: `not_run_is_not_pass` reads a
# recorded fail as evidence the capsule WAS certified at that tier and found wrong, which would put a
# cycle-accurate verdict on a capsule no RTL ever saw.

def test_the_short_circuit_path_fills_the_tiers_it_skipped():
    src = inspect.getsource(CR.run_capsule)
    seg = src[src.index("if not _complete_ladder:"):src.index("raise _cf")]
    assert "suppressed_tier_result" in seg, \
        "the opt-out path raises without recording the tiers it skipped — absent tiers are back"


def test_a_suppressed_tier_is_skipped_never_failed():
    r = CR.suppressed_tier_result("L3", mandatory=True, failed_tier="L2", from_rtl=True)
    assert r.status == "skipped", "a tier that never executed must not be recorded as a failure"
    assert r.mandatory is True and r.derived_from_rtl is True
    assert "not run" in (r.reason or ""), "the record must say it was not run, and why"
