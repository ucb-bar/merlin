"""A ``CheckReport`` in which nothing could be checked must not read as a clean bill of health.

``CheckReport.verdict`` counted only ``status == "fail"``. Skips landed in a sibling ``"skipped"``
key that no consumer branched on (established 2026-09-21: ``circt_gate`` reads ``rep.verdict``,
``rtl_check_runner`` reads ``rep.verdict``, and the harness's ``qa_check_rtlchecks._redact_rtl``
forwards the skip list to the agent as ``not_run`` text but gates on nothing), and about a dozen of
the screen's checks skip whenever an RTL fact is UNKNOWN. A target with no derivable facts therefore
serialized ``verdict: "ok"`` with zero checks having run.
"""

from __future__ import annotations

from merlin.targetgen import rtl_checks as RC


def _check(status, severity="error", cid="T0.x"):
    return RC.Check(cid, "T0", severity, status, "why")


def test_a_report_whose_every_check_skipped_is_not_ok():
    """THE MUTATION TEST for fix 2."""
    rep = RC.CheckReport(
        capsule="C",
        source_trace=None,
        rtl_facts={},
        checks=[
            _check("skipped", "error", "T0.a"),
            _check("skipped", "warn", "T0.b"),
            _check("skipped", "info", "T0.c"),
        ],
    )
    assert rep.verdict != "ok"
    assert rep.verdict == RC.CheckReport.VACUOUS
    assert (rep.n_ran, rep.n_skipped) == (0, 3)


def test_a_report_with_no_checks_at_all_is_not_ok():
    rep = RC.CheckReport(capsule=None, source_trace=None, rtl_facts={}, checks=[])
    assert rep.verdict == RC.CheckReport.VACUOUS


def test_one_check_that_actually_ran_restores_a_real_verdict():
    rep = RC.CheckReport(capsule=None, source_trace=None, rtl_facts={}, checks=[_check("skipped"), _check("pass")])
    assert rep.verdict == "ok"
    assert (rep.n_ran, rep.n_skipped) == (1, 1)


def test_a_failure_still_outranks_vacuity():
    """A check that DID run and failed is a real finding whatever the others did."""
    err = RC.CheckReport(
        capsule=None, source_trace=None, rtl_facts={}, checks=[_check("skipped"), _check("fail", "error")]
    )
    assert err.verdict == "reject"
    warn = RC.CheckReport(
        capsule=None, source_trace=None, rtl_facts={}, checks=[_check("skipped"), _check("fail", "warn")]
    )
    assert warn.verdict == "warn"


def test_the_serialized_report_carries_the_run_counts_beside_the_verdict():
    rep = RC.CheckReport(
        capsule="C",
        source_trace=None,
        rtl_facts={},
        checks=[_check("skipped", "error", "T0.a"), _check("pass", "warn", "T0.b")],
    )
    d = rep.to_dict()
    assert d["verdict"] == "ok"
    assert (d["n_ran"], d["n_skipped"]) == (1, 1)
    # The skip list is still there, now next to a count a reader cannot miss.
    assert [s["id"] for s in d["skipped"]] == ["T0.a"]


def test_an_unfactsed_screen_does_not_report_ok(monkeypatch):
    """End to end: with every RTL fact UNKNOWN and an empty trace, the screen must not read clean."""
    monkeypatch.setattr(RC, "load_default_facts", lambda target: {"from": "UNKNOWN (RTL facts not derivable)"})
    rep = RC.screen({"instructions": []}, None, None, target="a-target-with-no-facts")
    assert rep.verdict != "ok"
    # Every check that could not run is accounted for.
    assert rep.n_skipped >= 1
    assert rep.n_ran + rep.n_skipped == len(rep.checks)
