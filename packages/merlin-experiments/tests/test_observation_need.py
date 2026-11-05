"""A timed observation's budget scales with the case's declared work and this run's observed rate.

A fake observer stands in for the count runtime: it records the budget it was given and reports a
wall time proportional to the case's work, as the real one does.
"""

from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import joint_performance as J
from merlin_experiments.phase1.feedback import joint_performance_report as P
from merlin_experiments.phase1.feedback import observation_need as ON

# Recorded observations: 20 s of setup, then one second per 1000 floor cycles of declared work.
SETUP, PER_WORK = 20.0, 1 / 1000


def _roofline(floor):
    return {"structural_compute_floor_cycles": floor, "useful_macs": floor * 256, "structural_peak_macs_per_cycle": 256}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / ON.LEDGER
    monkeypatch.setattr(ON, "_ledger_path", lambda: path)
    for floor in (1_000, 2_000, 5_000, 10_000, 50_000, 100_000, 200_000, 400_000):
        ON.record(_roofline(floor), {"status": "observed", "wall_seconds": SETUP + floor * PER_WORK})
    return path


def test_need_follows_the_recorded_rate_and_is_unknown_without_evidence(ledger, tmp_path):
    rate = ON.fit()
    assert rate["observations"] == 8 and rate["setup_s"] == pytest.approx(SETUP, rel=0.2)
    assert rate["seconds_per_work"] == pytest.approx(PER_WORK, rel=0.05)
    assert ON.need_seconds(_roofline(2_000_000)) == pytest.approx(SETUP + 2000, rel=0.05)
    assert ON.need_seconds({"useful_macs": None}) is None  # no declared work
    assert ON.need_seconds(_roofline(10), path=tmp_path / "absent.json") is None  # no evidence: floor applies
    ON.record(_roofline(10), {"status": "UNKNOWN", "wall_seconds": 900})  # a failed observation is no rate
    assert ON.fit()["observations"] == 8


class _Observer:
    """Records the budget each observation receives; reports ``wall_seconds`` for its work."""

    def __init__(self, floor):
        self.floor, self.budgets = floor, []

    def __call__(self, root, *, remaining_s, **kwargs):
        self.budgets.append(remaining_s)
        return None, {"status": "observed", "wall_seconds": SETUP + self.floor * PER_WORK}


def _attach(monkeypatch, floor, *, remaining_s=30, request_deadline=None, ceiling=None):
    choice = ({"sha256": "a" * 64}, {"facts": {"sha256": "f" * 64}, "runtime": {"path": "untuned"}, "per_request_seconds": 900})
    monkeypatch.setattr(J, "selected", lambda: choice)
    monkeypatch.setattr(J, "structural_feedback", lambda *_a, **_k: _roofline(floor))
    record = (lambda: {"observation_resources": {"per_command_wall_seconds": ceiling}}) if ceiling else None
    monkeypatch.setattr(J, "_runtime", lambda *_a, **_k: SimpleNamespace(review_id="r", record=record))
    observer = _Observer(floor)
    monkeypatch.setattr(J, "_observe", observer)
    from merlin_experiments.phase1.feedback import joint_timing_selection as S

    monkeypatch.setattr(S, "choose", lambda *_: {"epoch": 0, "mode": "untuned", "runtime": {"path": "untuned"}})
    row = {}
    result = {"numeric": {"status": "pass"}, "tiers": {"L2": {"status": "pass", "engine": "spike"}}}
    J.attach(row, "unused", result=result, context=SimpleNamespace(target="neutral"), compiler="unused",
             capsule={"name": "neutral"}, remaining_s=remaining_s, request_deadline=request_deadline)
    return row["performance"], observer


def test_a_large_case_gets_the_budget_its_work_needs_within_its_request(ledger, monkeypatch):
    import time

    performance, observer = _attach(monkeypatch, 2_000_000, request_deadline=time.monotonic() + 7200)
    need = performance["observation_need_s"]
    assert need > 900 and observer.budgets == [pytest.approx(need, abs=1)]  # beyond the 30 s floor
    assert performance["observation_deferred"] is False
    assert ON.fit()["observations"] == 9  # its own completed observation joins the rate


def test_a_small_case_keeps_the_floor_and_a_grade_defers_what_it_cannot_fit(ledger, monkeypatch):
    performance, observer = _attach(monkeypatch, 1_000)
    assert observer.budgets == [30] and performance["observation_deferred"] is False
    performance, observer = _attach(monkeypatch, 2_000_000)  # a grade passes no request deadline
    assert observer.budgets == [] and performance["observation_deferred"] is True
    assert performance["reason"].startswith("observation budget too small (needed ~")
    assert f"~{performance['observation_need_s']} s" in performance["reason"]


def test_a_need_beyond_the_count_runtimes_own_bound_is_reported_not_started(ledger, monkeypatch):
    import time

    performance, observer = _attach(monkeypatch, 2_000_000, request_deadline=time.monotonic() + 7200, ceiling=900)
    assert observer.budgets == []
    assert "beyond the selected count runtime's own bound (1800 s" in performance["reason"]


def test_a_model_coverage_limit_is_named_in_the_reason(monkeypatch):
    events = ["loader bootstrap directory-event bound exceeded", "posted dispatch budget",
              "posted_target:unfinished_raw_ingress", "posted_timeline:posted dispatch budget"]
    original = P.project

    def project(*args, **kwargs):
        return {**original(*args, **kwargs), "unresolved_events": events}

    monkeypatch.setattr(P, "project", project)
    performance, _ = _attach(monkeypatch, 1_000)
    assert performance["model_coverage_limit"] == events[:2]
    assert performance["reason"] == ("model coverage limit: loader bootstrap directory-event bound exceeded; "
                                     "posted dispatch budget")


def test_background_requests_carry_the_need_of_cases_a_grade_deferred(monkeypatch):
    monkeypatch.delenv("MERLIN_IDLE_PERF_TIMEOUT_S", raising=False)
    base = I.knob("MERLIN_IDLE_PERF_TIMEOUT_S")
    performance = {"big": {"observation_deferred": True, "observation_need_s": 2500},
                   "small": {"observation_deferred": False, "observation_need_s": 40}}
    assert I.request_timeout(["big", "small", "unknown"], performance) == (base + 2500, ["big"])
    assert I.request_timeout(["small"], performance) == (base, [])


def test_a_case_starting_with_no_budget_left_says_so(ledger, monkeypatch):
    performance, observer = _attach(monkeypatch, 1_000, remaining_s=0)
    assert observer.budgets == [] and performance["reason"] == "observation not started: shared budget exhausted"


def test_the_providers_stated_refusal_reaches_the_row_without_host_paths(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import archive_count_observation as A

    stage = tmp_path / "annotated" / "invocations" / "abc"
    stage.mkdir(parents=True)
    (stage / "stderr.bin").write_bytes(b"trace line\nerror: completion outside dispatch or repeated identity "
                                       b"at /work/merlin/out/x/program.elf\n\n")
    reason = A.provider_reason(tmp_path / "annotated")
    assert reason.startswith("error: completion outside dispatch or repeated identity")
    assert "/work/merlin" not in reason
    assert A.provider_reason(tmp_path / "functional") is None

    class Refusing(_Observer):
        def __call__(self, root, *, remaining_s, **kwargs):
            return None, {"status": "UNKNOWN", "provider_reason": reason,
                          "reason": "original boxed archive count observation unavailable: CensusUnavailable"}

    monkeypatch.setattr(ON, "_ledger_path", lambda: tmp_path / ON.LEDGER)
    monkeypatch.setattr(J, "_observe", Refusing(1))
    choice = ({"sha256": "a" * 64}, {"facts": {"sha256": "f" * 64}, "runtime": {"path": "u"}, "per_request_seconds": 900})
    monkeypatch.setattr(J, "selected", lambda: choice)
    monkeypatch.setattr(J, "structural_feedback", lambda *_a, **_k: _roofline(10))
    monkeypatch.setattr(J, "_runtime", lambda *_a, **_k: SimpleNamespace(review_id="r"))
    from merlin_experiments.phase1.feedback import joint_timing_selection as S

    monkeypatch.setattr(S, "choose", lambda *_: {"epoch": 0, "mode": "untuned", "runtime": {"path": "u"}})
    row = {}
    J.attach(row, "unused", result={"numeric": {"status": "pass"}, "tiers": {"L2": {"status": "pass",
             "engine": "spike"}}}, context=SimpleNamespace(target="neutral"), compiler="unused",
             capsule={"name": "neutral"}, remaining_s=30)
    assert row["performance"]["unavailability_detail"] == reason
