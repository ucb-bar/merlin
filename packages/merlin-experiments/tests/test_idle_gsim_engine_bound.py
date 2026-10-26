"""The idle gSIM lane carries each job's wall budget to the emulator's simulated-cycle bound.

In one campaign 20 of 34 lane jobs were stopped by the selected harness command's ``+max-cycles`` cap
after ~700 s, whatever their budget. These tests hold the lane to: derive the bound from the budget and
the engine rate earlier jobs recorded; skip, and say why, a job no run under the selected process ceiling
can complete; treat a cap stop as a timed-out attempt.
"""

import json

import pytest
from test_idle_gsim_measurement import _workspace, campaign  # noqa: F401 -- the shared campaign fixture

from merlin_experiments.phase1 import gsim_engine_bound as E
from merlin_experiments.phase1 import joint_gsim_measurement as M
from merlin_experiments.phase1.feedback import idle_gsim as G
from merlin_experiments.phase1.feedback import idle_performance as I


def _record_job(rid, **fields):
    with M._ledger() as data:
        data["jobs"][rid] = {"capsule": "older", "executable": "old", "status": "done", "timeout_s": 1, **fields}


def _tick(lane, ws, **kw):
    known = I.estimates(ws / ".qa_channel", with_status=True)
    return lane.tick(ws / ".qa_channel", cohort=["big"], l2={"big": "pass"}, known=known, digest="d1", **kw)


def test_job_plan_skips_a_job_the_selected_ceiling_cannot_hold_and_holds_one_that_fits():
    fit = {"setup_s": 0.0, "cycles_per_s": 1.0, "margin": 4.0}
    window = dict(remaining_drain_s=10_000, remaining_end_s=12_000, final_window_s=400, long_allowed=True,
                  final_slots_free=True)
    plan, why = M.job_plan(fit, 1000, 1000, engine_ceiling_s=500, **window)
    assert plan is None and why.startswith(M.CEILING_SKIP) and "ceiling 500 s" in why
    plan, why = M.job_plan(fit, 100, 100, engine_ceiling_s=300, **window)
    assert plan["timeout_s"] == 300 and why == ""  # 400 s timeout held to the ceiling; 100 s expected fits
    assert M.job_plan(fit, 100, 100, **window)[0]["timeout_s"] == 400  # no known ceiling: unchanged


def test_the_lane_derives_the_cycle_bound_from_the_budget_and_records_it(tmp_path, campaign, capsys):  # noqa: F811
    ws = _workspace(tmp_path, cohort=("big",))
    # An earlier child recorded one capped run: 20,000 simulated cycles in 10 s, i.e. 2,000 cycles/s.
    _record_job(M.REQUEST_PREFIX + "0_old", engine_runs=[{"cycles": 20_000, "wall_s": 10.0, "capped": True}])
    lane = G.GsimLane(ws, campaign)
    assert _tick(lane, ws) == "submitted gSIM measurement"
    (request,) = (ws / ".qa_channel").glob(f"req_{M.REQUEST_PREFIX}*.json")
    body = json.loads(request.read_text())
    job = M.read()["jobs"][body["request_id"]]
    margin = body["measurement"]["throughput"]["margin"]
    assert job["engine_max_cycles"] == pytest.approx(job["timeout_s"] * 2000 * margin, abs=2)
    assert body["measurement"]["engine_bound"]["max_cycles"] == job["engine_max_cycles"]
    assert "fastest observed engine rate 2000 cycles/s" in job["engine_bound_basis"]
    assert f"engine cycle bound {job['engine_max_cycles']}" in capsys.readouterr().out


def test_without_a_recorded_rate_the_lane_says_the_selected_bound_stays(tmp_path, campaign, capsys):  # noqa: F811
    ws = _workspace(tmp_path, cohort=("big",))
    assert _tick(G.GsimLane(ws, campaign), ws) == "submitted gSIM measurement"
    (job,) = M.read()["jobs"].values()
    assert job["engine_max_cycles"] is None and "no gSIM engine rate observed" in job["engine_bound_basis"]
    assert "engine cycle bound selected" in capsys.readouterr().out


def test_a_case_beyond_the_selected_ceiling_is_skipped_once_with_its_reason(tmp_path, campaign, capsys):  # noqa: F811
    ws = _workspace(tmp_path, cohort=("big",))
    _record_job(M.REQUEST_PREFIX + "0_old", engine_timeout_ceiling_s=60, finished_at_epoch=1.0)
    lane = G.GsimLane(ws, campaign)
    assert _tick(lane, ws) is None and lane.last.startswith(M.CEILING_SKIP)
    assert _tick(lane, ws) is None
    lines = [line for line in capsys.readouterr().out.splitlines() if "not measured" in line]
    assert lines == [f"[idle-gsim] big not measured: {lane.last}"]  # said once, not every tick
    assert not list((ws / ".qa_channel").glob(f"req_{M.REQUEST_PREFIX}*.json"))  # no slot burned


def test_a_cycle_cap_stop_counts_as_a_timed_out_attempt(tmp_path, campaign):  # noqa: F811
    ws = _workspace(tmp_path, cohort=("big",))
    lane = G.GsimLane(ws, campaign)
    _record_job(M.REQUEST_PREFIX + "0_old", capsule="big", executable="older-elf", outcome="cycle_cap")
    assert _tick(lane, ws) == "submitted gSIM measurement"
    first = [j for j in M.read()["jobs"].values() if j.get("executable") == "elf-big"][0]
    fit = M.throughput(ws / ".qa_channel", min_observations=4, quantile=0.95)
    assert first["timeout_s"] >= 2 * M.job_timeout(fit, 90_000) - 1  # doubled, as after a wall timeout


def test_child_results_carry_engine_runs_and_ceiling_into_the_ledger(tmp_path, campaign, monkeypatch):  # noqa: F811
    rid = M.REQUEST_PREFIX + "5_child"
    assert M.authorize(rid, "big", timeout_s=100, executable="e", modeled_cycles=1,
                       engine_bound={"max_cycles": 123_456, "basis": "b"}) is None
    monkeypatch.setenv(M.ENV, rid)
    runs = [{"cycles": 300, "wall_s": 0.3, "capped": True}]
    with M.admission("big"):
        M.note_result(rid, {"engine_runs": runs, "engine_stop": E.engine_stop(runs), "engine_timeout_ceiling_s": 50})
        M.note_result(rid, {"outcome": "cycle_cap"})
    job = M.read()["jobs"][rid]
    assert job["engine_max_cycles"] == 123_456 and job["engine_runs"] == runs and job["outcome"] == "cycle_cap"
    assert "cycle bound (300 simulated cycles)" in job["engine_stop"]
    assert E.engine_ceiling(M.read()["jobs"]) == 50
    assert E.engine_rate(M.read()["jobs"])["max_cycles_per_s"] == pytest.approx(1000)
