"""Idle-time gSIM measurement of passing scalability cases: budget, admission, lane and consumers."""

import json
import os
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1 import joint_gsim_measurement as M
from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import idle_gsim as G
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import model_calibration, scratch_retention
from merlin_experiments.phase1.feedback import performance_headline as HL
from merlin_experiments.phase1.feedback import performance_headroom as H

ROOFLINE = {"structural_compute_floor_cycles": 100, "useful_macs": 25600, "structural_peak_macs_per_cycle": 256}


def _performance(cycles, elf):
    return {
        "status": "PROVISIONAL",
        "complete_event_accounting": True,
        "estimated_whole_soc_cycles": cycles,
        "structural_roofline": ROOFLINE,
        "elf_sha256": elf,
    }


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A started joint campaign whose gSIM budget is 20 x 1000 s (the lane's default share: 5000 s)."""
    run = tmp_path / "run"
    run.mkdir()
    policy = {
        "version": 1,
        "duration_s": 28800,
        "final_model_s": 7200,
        "drain_s": 900,
        "gsim": {"max_jobs": 20, "per_job_s": 1200, "worker_budget_s": 20000, "witness_capsules": []},
        "gsim_measurement": {"enabled": True},  # the lane is off unless a campaign enables it
    }
    state = run / "joint_campaign.json"
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "policy": policy,
                "policy_sha256": "p",
                "started_at_epoch": time.time(),
                "performance_selection": None,
            }
        )
    )
    monkeypatch.setenv(joint_control.ENV, str(state))
    monkeypatch.delenv(M.ENV, raising=False)
    return run


def _workspace(tmp_path, cohort=("big", "mid", "small")):
    ws = tmp_path / "ws"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (ws / "submission").mkdir()
    # Recorded single-case gSIM jobs: 40 s of setup, then 1000 cycles per second.
    for index, cycles in enumerate((100, 200, 300, 400, 20_000, 40_000, 60_000, 80_000)):
        jid = f"j{index}"
        (channel / f"simtiming_{jid}.json").write_text(json.dumps({"worker_runtime_s": 40 + cycles / 1000}))
        (channel / f"simresp_{jid}.json").write_text(
            json.dumps(
                {
                    "sim": "gsim",
                    "per_capsule": [{"capsule": f"w{index}", "barrier_cycles": cycles, "barrier_tier": "L3"}],
                }
            )
        )
    sizes = {"big": 90_000, "mid": 9_000, "small": 900}
    cases = [
        {"capsule": name, "tiers": {"L2": "pass"}, "performance": _performance(sizes[name], f"elf-{name}")}
        for name in cohort
    ]
    (channel / "resp_1_a.json").write_text(
        json.dumps(
            {"submission_sha256": "d1", "graded_at_unix_ns": 1, "selfcheck_submission": "submission", "cases": cases}
        )
    )
    return ws


def _measurement(ws, name, *, cycles, elf, rid="op_idle_gsim_1_a", passed=True):
    row = {
        "capsule": name,
        "pass": passed,
        "barrier_tier": "L3",
        "barrier_status": "pass" if passed else "fail",
        "barrier_cycles": cycles,
        "execution_digest": f"exe-{elf}",
        "performance": _performance(cycles * 2, elf),
    }
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(
        json.dumps(
            {
                "sim": "gsim",
                "suite": "scalability",
                "graded_at_unix_ns": 5,
                "grade_started_at_unix_ns": 4,
                "selfcheck_submission": "submission",
                "per_capsule": [row],
            }
        )
    )


def test_throughput_is_fitted_from_recorded_jobs_and_sets_the_timeout(tmp_path, campaign):
    ws = _workspace(tmp_path)
    fit = M.throughput(ws / ".qa_channel", min_observations=4, quantile=0.95)
    assert fit["observations"] == 8 and fit["setup_s"] == pytest.approx(40.25)
    assert fit["cycles_per_s"] == pytest.approx(1000, rel=0.02) and fit["margin"] >= 1
    timeout = M.job_timeout(fit, 120_000, spike_seconds=30)
    assert timeout == pytest.approx(fit["margin"] * (fit["setup_s"] + 30 + 120_000 / fit["cycles_per_s"]), abs=1)
    assert M.job_timeout(fit, 120_000, spike_seconds=30, retries=1) >= 2 * timeout - 1
    assert M.job_timeout(fit, None) is None
    assert M.throughput(ws / ".qa_channel", min_observations=9, quantile=0.95) is None  # too few: no guess


def test_authorization_reserves_the_lane_budget_and_only_authorized_requests_pass(tmp_path, campaign):
    knobs = M.settings(joint_control.selected()["policy"])
    assert knobs["budget_s"] == 5000 and knobs["max_jobs"] is None  # concurrency is derived per submission
    rid = M.REQUEST_PREFIX + "1_x"

    def authorize(request_id, name, timeout, free=None):
        return M.authorize(request_id, name, timeout_s=timeout, executable="e", modeled_cycles=1, free_slots=free)

    assert authorize(rid, "big", 6000) == "gSIM measurement budget exhausted"
    assert authorize("op_idle_1_x", "big", 10) == "invalid measurement request"
    assert authorize(rid, "big", 1000, free=0) == "no free idle slot"
    assert authorize(rid, "big", 1000, free=3) is None
    assert authorize(rid + "y", "mid", 10, free=3) == "measurement jobs in flight"  # 0.25 x (1 + 3) idle = 1
    request = {"origin": M.ORIGIN, "suite": "scalability", "sim": "gsim", "capsules": "big"}
    assert M.broker_admission(rid, request)["timeout_s"] == 1000
    for forged in ({**request, "capsules": "mid"}, {**request, "origin": I.ORIGIN}, {**request, "sim": "spike"}):
        assert M.broker_admission(rid, forged) is None
    assert M.broker_admission(M.REQUEST_PREFIX + "2_never", request) is None
    with pytest.raises(ValueError):  # a child without the broker's authorization id
        with M.admission("big"):
            pass
    os.environ[M.ENV] = rid
    try:
        with pytest.raises(ValueError, match="not authorized"):
            with M.admission("mid"):
                pass
        with M.admission("big") as budget:
            assert 1 <= budget <= 1000
            assert M.read()["jobs"][rid]["status"] == "running"
            M.note_result(rid, {"outcome": "measured", "barrier_cycles": 77, "ignored": 1})
            assert M.broker_admission(rid, request) is None  # claimed: a replay is refused
    finally:
        del os.environ[M.ENV]
    job = M.read()["jobs"][rid]
    assert job["status"] == "done" and job["outcome"] == "measured" and job["barrier_cycles"] == 77
    assert "ignored" not in job and M.read()["reserved_s"] < 1000  # the unused reservation is refunded


def test_lane_measures_largest_headroom_first_and_rotates_with_the_executable(tmp_path, campaign):
    ws = _workspace(tmp_path)
    lane = G.GsimLane(ws, campaign)
    known = I.estimates(ws / ".qa_channel", with_status=True)
    l2 = {"big": "pass", "mid": "pass", "small": "pass"}
    assert (
        lane.tick(ws / ".qa_channel", cohort=["small", "mid", "big"], l2=l2, known=known, digest="d1")
        == "submitted gSIM measurement"
    )
    (request,) = (ws / ".qa_channel").glob(f"req_{M.REQUEST_PREFIX}*.json")
    body = json.loads(request.read_text())
    rid = body["request_id"]
    assert body["capsules"] == "big" and body["sim"] == "gsim" and body["suite"] == "scalability"
    assert body["origin"] == M.ORIGIN and body["origin"] in I.OPERATOR_ORIGINS and rid.startswith(I.REQUEST_PREFIX)
    assert (ws / ".qa_channel" / f"detached_{rid}").exists()
    job = M.read()["jobs"][rid]
    assert job["executable"] == "elf-big" and body["timeout"] == job["timeout_s"]
    assert lane.inflight(ws / ".qa_channel") == 1
    # Finished: the measured executable is not measured again; the next largest case is.
    _measurement(ws, "big", cycles=50_000, elf="elf-big", rid=rid)
    with M._ledger() as data:
        data["jobs"][rid].update(status="done", outcome="measured")
        data["reserved_s"] = 0
    assert lane.tick(ws / ".qa_channel", cohort=["small", "mid", "big"], l2=l2, known=known, digest="d1")
    assert {json.loads(p.read_text())["capsules"] for p in (ws / ".qa_channel").glob("req_op_idle_gsim_*")} == {
        "big",
        "mid",
    }
    ranked = G.candidates(
        ["big", "mid"],
        l2,
        {"big": [(2, "d2", _performance(90_000, "elf-big-2"), True)]},
        "d2",
        measured=G.measured_executables(ws / ".qa_channel"),
    )
    assert [c["capsule"] for c in ranked] == ["big"]  # a new executable of a measured case is eligible again


def test_concurrency_follows_the_idle_capacity_seen_at_submission(tmp_path, campaign, monkeypatch):
    monkeypatch.setattr(M, "_slot_capacity", lambda: 20)
    granted = []
    for index in range(8):  # 16 idle slots: the lane may hold 0.25 x 16 = 4 jobs; each started job uses one
        rid = M.REQUEST_PREFIX + f"9_{index}"
        if (
            M.authorize(rid, f"c{index}", timeout_s=100, executable="e", modeled_cycles=1, free_slots=16 - len(granted))
            is None
        ):
            granted.append(rid)
    assert len(granted) == 4
    monkeypatch.setattr(M, "_slot_capacity", lambda: 4)  # the gSIM slots still bound it
    assert (
        M.authorize(M.REQUEST_PREFIX + "9_x", "cx", timeout_s=100, executable="e", modeled_cycles=1, free_slots=40)
        == "gSIM slots held by correctness or witness work"
    )


def test_cases_without_an_estimate_are_timed_from_their_structural_floor(tmp_path, campaign):
    ws = _workspace(tmp_path)
    channel = ws / ".qa_channel"
    # Measured gSIM rows of this run: 200 cycles of overhead plus 3x the structural floor.
    for index, floor in enumerate((10, 20, 30, 40, 1000, 2000, 3000, 4000)):
        roofline = {"structural_compute_floor_cycles": floor}
        row = {
            "capsule": f"w{index}",
            "pass": True,
            "barrier_tier": "L3",
            "barrier_cycles": 200 + 3 * floor,
            "performance": {"structural_roofline": roofline},
        }
        (channel / f"simresp_r{index}.json").write_text(json.dumps({"sim": "gsim", "per_capsule": [row]}))
    model = M.inefficiency(channel, min_observations=4, quantile=0.95)
    fit = model["classes"]["compute"]
    assert fit["overhead_cycles"] == 275 and fit["factor"] == pytest.approx(2.975, abs=0.01)  # median of the small half
    # The floor is the larger of the compute floor and the ideal request service.
    huge = {
        "useful_macs": 25600 * 256,
        "structural_peak_macs_per_cycle": 256,
        "logical_request_ideal_service": {
            "read": {"requested_bytes": 16 * 50_000, "ideal_requested_service_cycles": 50_000},
            "write": {"requested_bytes": 16 * 70_000},
        },
    }
    assert M.structural_floor(huge) == (70_000, "write requests")  # write rate taken from the read side
    records = G.verdict_performance(
        [
            {
                "graded_source_sha256": "s" * 64,
                "per_capsule": [
                    {
                        "capsule": "noest",
                        "tiers": {"L2": "pass"},
                        "performance": {"status": "UNKNOWN", "structural_roofline": huge},
                    }
                ],
            }
        ]
    )
    ranked = G.candidates(
        ["small", "noest"],
        {"small": "pass", "noest": "pass"},
        I.estimates(channel, with_status=True),
        "d1",
        records=records,
        model=model,
    )
    assert [c["capsule"] for c in ranked] == ["noest", "small"]  # largest headroom over the bound first
    (noest,) = [c for c in ranked if c["capsule"] == "noest"]
    assert noest["modeled_cycles"] is None and noest["structural_floor_cycles"] == 70_000
    assert noest["predicted_cycles"] == pytest.approx(275 + 2.975 * 70_000, rel=0.01)
    assert noest["cycles"] >= noest["predicted_cycles"] and noest["executable"] == "source:" + "s" * 64
    # Too few rows for a fit: the worst observed cycles / floor ratio stands in.
    sparse = M.inefficiency(channel, min_observations=100, quantile=0.95)
    assert sparse["classes"] == {} and M.predicted_cycles(sparse, 1000, "compute")["cycles"] == pytest.approx(
        sparse["worst_ratio"] * 1000
    )
    lane = G.GsimLane(ws, campaign)
    assert (
        lane.tick(channel, cohort=["noest"], l2={"noest": "pass"}, known={}, digest="d1", free=4, records=records)
        == "submitted gSIM measurement"
    )
    (request,) = channel.glob(f"req_{M.REQUEST_PREFIX}*.json")
    detail = json.loads(request.read_text())["measurement"]
    assert detail["structural_floor_cycles"] == 70_000 and "inefficiency" in detail


def test_a_long_job_may_run_past_the_drain_boundary_only_when_it_is_expected_to_end_in_time():
    fit = {"setup_s": 0.0, "cycles_per_s": 1.0, "margin": 4.0}
    window = dict(remaining_drain_s=1000, remaining_end_s=1200, final_window_s=400)

    def plan(cycles, central, **kw):
        return M.job_plan(fit, cycles, central, **{**window, "long_allowed": True, "final_slots_free": True, **kw})

    assert plan(100, 100)[0] == {"timeout_s": 400, "long": False, "expected_s": 100.0}
    # Timeout 2000 s > the 1000 s before drain; expected 500 s ends before the final window (at 800 s).
    assert plan(500, 500)[0] == {"timeout_s": 1200, "long": True, "expected_s": 500.0}  # up to the end
    assert plan(500, 900)[1] == "expected runtime does not end before the final-model window"
    assert plan(500, 500, long_allowed=False)[0] is None
    # A job still running in the final window (timeout > 800 s) needs slots beyond the grade reserve.
    assert plan(250, 250, final_slots_free=False)[1] == "final-window slots are reserved for final grading"
    assert plan(150, 150, final_slots_free=False)[0]["long"] is False  # 600 s ends before it


def test_one_long_measurement_at_a_time_bounded_by_the_campaign_end(tmp_path, campaign):
    from merlin_experiments.phase1 import joint_control as JC

    end, drain = JC.remaining(), JC.remaining(drain=True)
    beyond = int(drain) + 60  # past the drain boundary, before the end
    first, second = M.REQUEST_PREFIX + "7_a", M.REQUEST_PREFIX + "7_b"
    budget = dict(executable="e", modeled_cycles=1)
    assert M.authorize(first, "big", timeout_s=beyond, **budget) == (
        "remaining campaign window shorter than the derived timeout"
    )
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim"]["worker_budget_s"] = 400_000
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    assert M.authorize(first, "big", timeout_s=beyond, long=True, **budget) is None
    assert M.authorize(second, "mid", timeout_s=beyond, long=True, **budget) == "long measurement in flight"
    assert M.authorize(second, "mid", timeout_s=int(end) + 60, long=True, **budget) == (
        "remaining campaign window shorter than the derived timeout"
    )
    os.environ[M.ENV] = first
    try:
        with M.admission("big") as granted:
            assert granted > drain  # the child's own bound is the campaign end, not the drain boundary
    finally:
        del os.environ[M.ENV]
    assert M.read()["jobs"][first]["long"] is True


def test_the_lane_starts_a_long_job_for_a_case_too_large_for_the_drain_window(tmp_path, campaign, monkeypatch):
    from merlin_experiments.phase1 import joint_control as JC

    ws = _workspace(tmp_path, cohort=("big",))
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim"]["worker_budget_s"] = 400_000
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    monkeypatch.setattr(JC, "remaining", lambda drain=False, now=None: 3000.0 if drain else 12_000.0)
    lane = G.GsimLane(ws, campaign)
    known = I.estimates(ws / ".qa_channel", with_status=True)
    args = dict(cohort=["big"], l2={"big": "pass"}, known=known, digest="d1", free=8)
    monkeypatch.setattr(M, "job_timeout", lambda *a, **k: 9000)  # beyond the 3000 s before drain
    assert lane.tick(ws / ".qa_channel", **args, final_reserve=8) is None  # final-window slots kept
    assert lane.last == "final-window slots are reserved for final grading"
    assert lane.tick(ws / ".qa_channel", **args, final_reserve=2) == "submitted gSIM measurement"
    (request,) = (ws / ".qa_channel").glob(f"req_{M.REQUEST_PREFIX}*.json")
    body = json.loads(request.read_text())
    assert body["measurement"]["long"] is True and body["timeout"] == 9000
    assert M.read()["jobs"][body["request_id"]]["long"] is True


def test_lane_stands_aside_without_a_campaign_throughput_or_enablement(tmp_path, campaign, monkeypatch):
    ws = _workspace(tmp_path)
    for path in (ws / ".qa_channel").glob("simtiming_*"):
        path.unlink()
    lane = G.GsimLane(ws, campaign)
    known = I.estimates(ws / ".qa_channel", with_status=True)
    args = dict(cohort=["big"], l2={"big": "pass"}, known=known, digest="d1")
    assert lane.tick(ws / ".qa_channel", **args) is None and lane.last == "no recorded gSIM throughput yet"
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim_measurement"] = {"enabled": False}
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    assert lane.tick(ws / ".qa_channel", **args) is None and lane.last == "disabled by policy"
    monkeypatch.delenv(joint_control.ENV)
    assert lane.tick(ws / ".qa_channel", **args) is None and lane.last == "no joint campaign"


def test_the_idle_measurer_submits_the_gsim_lane_on_open_gates(tmp_path, campaign, monkeypatch):
    for name in (I.ENV, "MERLIN_SELFCHECK_BROKER_JOBS", *I.KNOBS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 20)
    monkeypatch.setattr(scratch_retention, "free_space_ok", lambda *a, **k: True)
    ws = _workspace(tmp_path)
    (campaign / "qa_history").mkdir()
    verdict = {
        "joint_test_selection": {"performance_cohorts": {"primary": ["big", "mid", "small"]}},
        "per_capsule": [{"capsule": n, "tiers": {"L2": "pass"}} for n in ("big", "mid", "small")],
    }
    (campaign / "qa_history" / "verdict_fast_1.json").write_text(json.dumps(verdict))
    measurer = I.IdleMeasurer(ws, campaign, digest=lambda _root: "d1", clock=lambda: 1000.0)
    assert measurer.tick(grading=False) == "submitted gSIM measurement"
    assert len(list((ws / ".qa_channel").glob("req_op_idle_gsim_*"))) == 1


def test_measurements_feed_headroom_history_headline_and_calibration(tmp_path, monkeypatch):
    monkeypatch.delenv(joint_control.ENV, raising=False)
    ws = _workspace(tmp_path)
    (ws / "qa").mkdir()
    _measurement(ws, "big", cycles=60_000, elf="elf-big")
    row = {"capsule": "big", "status": "pass", "tiers": {"L2": "pass"}, "performance": _performance(90_000, "elf-big")}
    other = {"capsule": "mid", "status": "pass", "tiers": {"L2": "pass"}, "performance": _performance(9_000, "elf-mid")}
    entries = {e["test_case"]: e for e in H.build(ws, {"per_capsule": [row, other]})["largest_headroom"]}
    assert entries["big"]["evidence"] == H.ADVISORY and entries["big"]["cycles"] == 60_000
    assert "measured_on_earlier_source" not in entries["big"] and entries["mid"]["evidence"].startswith("modeled")
    stale = {**row, "performance": _performance(90_000, "elf-big-newer")}
    (entry,) = [e for e in H.build(ws, {"per_capsule": [stale]})["largest_headroom"]]
    assert entry["evidence"] == H.ADVISORY and entry["measured_on_earlier_source"] is True
    pairs = model_calibration.pairs(ws)
    assert any(p["test_case"] == "big" and p["measured"] == 60_000 and p["modeled"] == 120_000 for p in pairs)
    assert CG.record(
        ws,
        {"per_capsule": [row, other], "graded_source_sha256": "a" * 64, "graded_at": "2026-10-26T00:00:00+00:00"},
        100,
    )
    ledger = [json.loads(line) for line in (ws / "qa" / "perf_history.jsonl").read_text().splitlines()]
    measured = [r for r in ledger if r["kind"] == "measured"]
    assert [(r["test_case"], r["cycles"], r["evidence"]) for r in measured] == [("big", 60_000, H.ADVISORY)]
    assert any(r["kind"] == "pair" and r["test_case"] == "big" for r in ledger)
    headline = json.loads((ws / "qa" / CG.FILENAME).read_text())[HL.KEY]
    assert headline["performance"]["measured"]["cases"] == 1


def test_the_joint_wrapper_admits_only_an_authorized_measurement_child(tmp_path, campaign):
    calls = []

    @joint_control.selfcheck_policy
    def selfcheck(argv=None, **kwargs):
        calls.append(argv)
        return 0

    argv = ["--suite", "scalability", "--sim", "gsim", "--capsules", "big", "--timeout", "5000"]
    assert selfcheck(argv) == 2 and not calls  # an author request: the small-witness roster refuses it
    rid = M.REQUEST_PREFIX + "3_z"
    assert M.authorize(rid, "big", timeout_s=900, executable="e", modeled_cycles=1) is None
    os.environ[M.ENV] = rid
    try:
        assert selfcheck(argv) == 0
    finally:
        del os.environ[M.ENV]
    (forwarded,) = calls
    assert forwarded[forwarded.index("--sim") + 1] == "gsim" and "--suite" in forwarded
    assert int(forwarded[len(forwarded) - 1 - forwarded[::-1].index("--timeout") + 1]) <= 900
    assert M.read()["jobs"][rid]["status"] == "done"


def test_the_measurement_child_refuses_without_authorization():
    from merlin_experiments.phase1.feedback import public_scalability as P

    with pytest.raises(ValueError, match="only Spike functional"):
        P.cli(SimpleNamespace(sim="gsim", out="x", capsules="big"), context=None)


def test_the_broker_forwards_only_an_authorized_scalability_gsim_request(tmp_path, campaign, monkeypatch):
    from merlin_experiments.phase1 import frozen_facts
    from merlin_experiments.phase1.brokers import selfcheck as B
    from merlin_experiments.phase1.feedback import public_scalability as P

    launched = []

    class Launched(Exception):
        pass

    def popen(command, **kwargs):
        launched.append((command, kwargs["env"]))
        raise Launched("worker started")

    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setattr(frozen_facts, "select", lambda *args: None)
    monkeypatch.setattr(B, "resolve_context", lambda *args: SimpleNamespace(repo=tmp_path))
    monkeypatch.setattr(B, "worker_command", lambda *a, **k: ["worker"])
    monkeypatch.setattr(B.subprocess, "Popen", popen)
    monkeypatch.setattr(P, "selected", lambda context: (tmp_path, {"cases": ["big"]}))
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    channel = ws / ".qa_channel"
    channel.mkdir()
    pool = tmp_path / "pool"
    pool.mkdir()

    def serve(rid, body):
        (channel / f"req_{rid}.json").write_text(json.dumps({"request_id": rid, "protocol": 3, **body}))
        B.main(["--ws", str(ws), "--capsules-root", str(pool), "--one-request", f"req_{rid}.json"])
        return json.loads((channel / f"resp_{rid}.json").read_text())

    deadline = time.time_ns() + 600_000_000_000
    base = {
        "suite": "scalability",
        "sim": "gsim",
        "capsules": "big",
        "workers": 0,
        "timeout": 5000,
        "deadline_unix_ns": deadline,
        "origin": M.ORIGIN,
    }
    refused = serve(M.REQUEST_PREFIX + "4_forged", base)  # never authorized
    assert "public scalability request refused" in refused["error"] and not launched
    rid = M.REQUEST_PREFIX + "5_ok"
    assert M.authorize(rid, "big", timeout_s=700, executable="e", modeled_cycles=1) is None
    (channel / f"detached_{rid}").write_text("{}")
    serve(rid, base)
    ((command, env),) = launched
    flat = " ".join(map(str, command))
    assert "--suite scalability" in flat and "--sim gsim" in flat and "--timeout 700" in flat
    assert env[M.ENV] == rid
    from merlin.common.cpu_pool import AUXILIARY, PRIORITY_ENV

    assert env[PRIORITY_ENV] == str(AUXILIARY)


def test_only_a_long_measurement_survives_the_broker_stop_and_it_yields_to_waiting_grading(tmp_path, monkeypatch):
    from merlin_experiments.phase1.brokers import selfcheck as B

    channel = tmp_path / "ch"
    channel.mkdir()
    request = channel / "req_x.json"
    request.write_text("{}")
    later = time.monotonic() + 600
    contended = [False]
    monkeypatch.setattr(M, "shared_pool_contended", lambda: contended[0])
    (channel / "STOP").write_text("stop")
    assert B._abort_reason(channel, os.getppid(), request, later) == "STOP"  # today's behaviour
    assert B._abort_reason(channel, os.getppid(), request, later, survive_stop=True) is None
    assert B._abort_reason(channel, -1, request, later, survive_stop=True) is None  # parent exited
    (channel / "STOP").unlink()
    stop = channel / ".handler-stop"
    stop.write_text("stop")
    assert B._abort_reason(channel, os.getppid(), request, later, request_stop=stop) == "dispatcher stopped"
    assert B._abort_reason(channel, os.getppid(), request, later, request_stop=stop, survive_stop=True) is None
    contended[0] = True  # final grading waits for a shared slot while none is free
    assert "wanted while none is free" in B._abort_reason(
        channel, os.getppid(), request, later, request_stop=stop, survive_stop=True
    )
    contended[0] = False
    assert B._abort_reason(channel, os.getppid(), request, time.monotonic() - 1, survive_stop=True) == "timed out"
    request.unlink()
    assert B._abort_reason(channel, os.getppid(), request, later, survive_stop=True) == "the requester cancelled it"


def test_the_dispatcher_leaves_only_a_running_long_measurement_at_its_exit(tmp_path):
    from merlin_experiments.phase1.brokers import selfcheck as B

    class Process:
        def __init__(self, running):
            self.running, self.waited = running, False

        def poll(self):
            return None if self.running else 0

        def wait(self, timeout=None):
            self.waited = True
            return 0

    jobs = {
        name: {"long": long, "process": Process(running), "stop": tmp_path / f"stop_{name}"}
        for name, long, running in (("long", True, True), ("done_long", True, False), ("normal", False, True))
    }
    assert B._drain_handlers(jobs) == ["long"]
    assert not jobs["long"]["process"].waited and not jobs["long"]["stop"].exists()
    assert jobs["normal"]["process"].waited and jobs["done_long"]["process"].waited


def test_a_running_long_measurement_is_not_dispatched_again_and_ends_by_the_campaign_end(tmp_path, campaign):
    from merlin_experiments.phase1.brokers import selfcheck as B

    rid = M.REQUEST_PREFIX + "8_long"
    assert M.authorize(rid, "big", timeout_s=1000, executable="e", modeled_cycles=1, long=True) is None
    channel = tmp_path / "ch"
    channel.mkdir()
    (channel / f"req_{rid}.json").write_text(json.dumps({"deadline_unix_ns": time.time_ns() + 10**12}))
    (channel / f"detached_{rid}").write_text("{}")
    assert M.long_request(rid) and not M.long_request(rid, running=True)
    assert [p.name for p in B._pending_requests(channel, set())] == [f"req_{rid}.json"]  # not yet claimed
    with M._ledger() as data:
        data["jobs"][rid].update(status="running", pid=os.getpid())
    assert M.long_request(rid, running=True)
    assert B._pending_requests(channel, set()) == []  # a restarted broker leaves it to its handler
    with M._ledger() as data:
        data["jobs"][rid].update(pid=2**22 + 12345)  # its handler died: the ledger marks it lost
    assert not M.long_request(rid, running=True)
    assert len(B._pending_requests(channel, set())) == 1  # re-offered; its claim is gone, so it is refused
    with M._ledger():
        pass  # the next ledger access settles the dead claim
    assert not M.long_request(rid) and M.read()["jobs"][rid]["status"] == "lost"
    assert not M.long_request(M.REQUEST_PREFIX + "never") and not M.long_request("1_author")


def test_the_pool_counts_as_contended_only_with_a_waiter_and_no_free_slot(monkeypatch, tmp_path):
    from merlin.common import cpu_pool

    root = tmp_path / "pool"
    root.mkdir()
    state = {"free": (), "waiters": ["w"]}
    monkeypatch.setattr(cpu_pool, "_shared_compute_selection", lambda environ: ((1, 2), root))
    monkeypatch.setattr(cpu_pool, "_free_cpus", lambda r, cpus, fresh=False: state["free"])
    monkeypatch.setattr(cpu_pool, "_queued_keys", lambda directory: state["waiters"])
    assert M.shared_pool_contended()
    state["free"] = (2,)
    assert not M.shared_pool_contended()
    state.update(free=(), waiters=[])
    assert not M.shared_pool_contended()
    monkeypatch.setattr(cpu_pool, "_shared_compute_selection", lambda environ: None)
    assert not M.shared_pool_contended()


@pytest.mark.parametrize("long", [True, False])
def test_a_handler_serves_its_long_measurement_through_a_stop(tmp_path, campaign, monkeypatch, long):
    from merlin_experiments.phase1 import frozen_facts
    from merlin_experiments.phase1.brokers import selfcheck as B
    from merlin_experiments.phase1.feedback import public_scalability as P

    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    channel = ws / ".qa_channel"
    channel.mkdir()

    class Worker:
        """A measurement child that runs four polls; the round's STOP arrives during the first."""

        pid = 2**22 + 4321  # no such process: an abort reaps nothing real
        polls = 0

        def poll(self):
            Worker.polls += 1
            (channel / "STOP").write_text("stop")
            return None if Worker.polls < 4 else 0

        def wait(self, timeout=None):
            return 0

        terminate = kill = lambda self: None

    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setattr(frozen_facts, "select", lambda *args: None)
    monkeypatch.setattr(B, "resolve_context", lambda *args: SimpleNamespace(repo=tmp_path))
    monkeypatch.setattr(B, "worker_command", lambda *a, **k: ["worker"])
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: Worker())
    monkeypatch.setattr(P, "selected", lambda context: (tmp_path, {"cases": ["big"]}))
    monkeypatch.setattr(M, "shared_pool_contended", lambda: False)
    rid = M.REQUEST_PREFIX + "6_stop"
    assert M.authorize(rid, "big", timeout_s=700, executable="e", modeled_cycles=1, long=long) is None
    (channel / f"detached_{rid}").write_text("{}")
    body = {
        "suite": "scalability",
        "sim": "gsim",
        "capsules": "big",
        "workers": 0,
        "timeout": 700,
        "deadline_unix_ns": time.time_ns() + 10**12,
        "origin": M.ORIGIN,
        "request_id": rid,
        "protocol": 3,
    }
    (channel / f"req_{rid}.json").write_text(json.dumps(body))
    pool = tmp_path / "pool"
    pool.mkdir()
    B.main(["--ws", str(ws), "--capsules-root", str(pool), "--one-request", f"req_{rid}.json", "--poll", "0"])
    response = json.loads((channel / f"resp_{rid}.json").read_text())
    if long:  # served to completion although the round's STOP arrived
        assert Worker.polls == 4 and "aborted" not in str(response.get("error"))
    else:  # every other request keeps today's behaviour
        assert Worker.polls == 1 and "self-check aborted (STOP)" in response["error"]
