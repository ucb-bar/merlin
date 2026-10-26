"""Command console: wall-time attribution, the machine core map, console placement and their findings."""

import json
import os
import time
from pathlib import Path

from merlin_experiments.tracking.console import diagnose, discovery, placement
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import jobs, machine, timeline


def _ctx(run: Path, shared: dict, *, live=False, started=1000.0, last=2000.0, now=5000.0) -> Context:
    experiment = {
        "id": str(run),
        "run_id": run.name,
        "run_dir": str(run),
        "started_at": started,
        "last_record_at": last,
    }
    return Context(
        experiment=experiment, reader=Reader(), state={}, operator_private=True, now=now, shared=shared, live=live
    )


def test_interval_algebra():
    assert timeline.merge([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]
    assert timeline.subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert timeline.intersect([(0, 4), (6, 9)], [(3, 7)]) == [(3, 4), (6, 7)]
    assert timeline.length([(0, 2), (5, 6)]) == 3


def test_lead_time_splits_into_disjoint_shares(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    agents = {
        "agents": [
            {
                "id": "lead",
                "role": "lead",
                "spans": [[1100, 1900, "turn"], [1200, 1300, "tool"], [1250, 1400, "wait"], [1500, 1600, "tool"]],
            },
            {"id": "h1", "role": "helper", "spans": [[1200, 1500, "turn"]]},
        ],
        "totals": {"helper_ceiling": 16},
    }
    jobs_section = {
        "spans": [{"lane": "L3", "start": 1250, "end": 1700}, {"lane": "queue", "start": 1200, "end": 1250}]
    }
    section = timeline.read(_ctx(run, {"agents": agents, "jobs": jobs_section}))
    lead = section["lead"]
    assert section["window"] == {"start": 1000.0, "end": 2000.0, "seconds": 1000.0}
    assert (lead["setup_s"], lead["tool_s"], lead["wait_s"]) == (100.0, 200.0, 100.0)
    assert lead["think_s"] == 800 - 200 - 100
    assert lead["idle_s"] == 100.0  # 1900..2000
    # Blocked on simulation = (tool ∪ wait) ∩ L3 = 1250..1400 and 1500..1600.
    assert section["overlap"]["lead_blocked_on_sim_s"] is None
    assert section["overlap"]["command_lifetime_during_sim_s"] == 150.0
    assert section["overlap"]["queue_s"] == 50.0
    assert section["parallelism"]["helpers_used"] == 1
    assert section["verdict"]["bound"] == "unclassified_agent_time"


def test_recorded_telemetry_is_reported_and_open_jobs_end_at_the_last_record(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    telemetry = {
        "method": "arrival_stamps",
        "think_generate_s": 900.0,
        "tool_and_wait_s": 100.0,
        "measured_span_s": 1000,
    }
    (run / "joint_campaign_summary.json").write_text(
        json.dumps({"completion_reason": "freeze_drain", "telemetry": telemetry})
    )
    os.utime(run / "joint_campaign_summary.json", (1990, 1990))
    agents = {"agents": [{"id": "lead", "role": "lead", "spans": [[1000, 1500, "turn"]]}]}
    open_job = {"spans": [{"lane": "L3", "start": 1800, "end": None}]}
    section = timeline.read(_ctx(run, {"agents": agents, "jobs": open_job}, now=99999))
    assert section["recorded"]["think_s"] == 900.0
    # The derived split covers the whole window, so it decides while the lead has spans: 500 s of
    # turn in a 1000 s window, after 0 s of setup, is half thinking and half idle.
    assert section["verdict"]["basis"] == "rollout spans of the lead agent"
    assert section["verdict"]["share_pct"] == 50.0
    # Without lead spans, the recorded in-session split is all there is.
    alone = timeline._verdict(None, section["recorded"], {}, 1000.0)
    assert alone["bound"] == "unclassified_agent_time" and alone["share_pct"] == 90.0
    assert alone["basis"] == "recorded arrival-stamp telemetry (inside sessions only)"
    # A job still "running" in a run that stopped ends at the run's last record, not now.
    assert section["window"]["end"] == 2000.0
    assert section["overlap"]["sim_busy_s"] == 200.0
    assert any(m["kind"] == "finish" for m in section["markers"])


def _cgroups(root: Path, groups: dict[str, tuple[str, int]]) -> None:
    for rel, (cpus, procs) in groups.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "cpuset.cpus").write_text(cpus + "\n")
        (d / "cpuset.cpus.effective").write_text(cpus + "\n")
        (d / "cgroup.procs").write_text("".join(f"{100 + i}\n" for i in range(procs)))


def _fake_proc(root: Path, cpus: int, runner: tuple[int, str, str] | None, own: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["cpu  0 0 0 0 0 0 0 0"] + [f"cpu{i} 10 0 10 80 0 0 0 0" for i in range(cpus)]
    (root / "stat").write_text("\n".join(lines) + "\n")
    (root / "self").mkdir()
    (root / "self" / "cgroup").write_text(f"0::{own}\n")
    if runner:
        pid, run_id, allowed = runner
        d = root / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\tpython\nCpus_allowed_list:\t{allowed}\n")
        (d / "stat").write_text(f"{pid} (python) S 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
        argv = ["/usr/bin/python3", "-m", "merlin_experiments.phase1", "--run-id", run_id]
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_allocations_are_derived_not_named(tmp_path):
    """No slice names or CPU numbers are assumed: the console's own cgroup defines the general pool."""
    cg, proc = tmp_path / "cg", tmp_path / "proc"
    _cgroups(
        cg,
        {
            "pool.slice": ("0-3", 5),  # holds the console: general
            "pool.slice/me.scope": ("0-3", 1),
            "services.slice": ("0-1", 3),  # confined within the pool: general
            "deep/one/held.service": ("4-7", 2),  # carved out, holds processes: reservation in use
            "spare.slice": ("6-7", 0),  # carved out, empty: reservation
        },
    )
    _fake_proc(proc, 8, (77, "exp-a", "4-5"), "/pool.slice/me.scope")
    allocs = {a["name"]: a for a in placement.allocations(proc_root=proc, cgroup_root=cg)}
    assert allocs["pool.slice"]["kind"] == "general"
    assert allocs["services.slice"]["kind"] == "general"
    assert allocs["deep/one/held.service"]["kind"] == "reservation"
    assert allocs["deep/one/held.service"]["processes"] == 2
    assert allocs["spare.slice"]["kind"] == "reservation"
    assert allocs["exp-a"] == {**allocs["exp-a"], "kind": "run", "cpus": [4, 5], "pid": 77}
    # Held = runner affinity plus reservations with processes; the empty reservation is not held.
    assert placement.held_cpus(list(allocs.values())) == {4, 5, 6, 7}


def test_machine_section_marks_the_experiment_and_finds_an_idle_reservation(tmp_path, monkeypatch):
    cg, proc = tmp_path / "cg", tmp_path / "proc"
    _cgroups(cg, {"pool.slice": ("0-3", 4), "spare.slice": ("6-7", 0)})
    _fake_proc(proc, 8, (77, "exp-a", "4-5"), "/pool.slice")
    allocs = placement.allocations(proc_root=proc, cgroup_root=cg)
    monkeypatch.setattr(placement, "allocations", lambda: allocs)
    monkeypatch.setattr(machine, "FIRST_SAMPLE_S", 0)
    ctx = _ctx(tmp_path / "exp-a", {}, live=True)
    ctx.experiment["runner_pid"] = 77
    section = machine.read(ctx)
    assert section["experiment_cpus"] == [4, 5]
    by_cpu = {c["cpu"]: c for c in section["cores"]}
    assert by_cpu[4]["mine"] and by_cpu[4]["holders"][0] == "exp-a"
    assert by_cpu[6]["holders"] == ["spare.slice"]
    findings = diagnose.diagnose({"experiment": ctx.experiment, "sections": {"machine": section}})
    idle = [f for f in findings if f["code"] == "RESERVATION_IDLE"]
    assert len(idle) == 1 and "spare.slice" in idle[0]["title"]


def test_wall_time_findings(tmp_path):
    snap = {
        "experiment": {"state": "FINISHED"},
        "sections": {
            "timeline": {
                "window": {"seconds": 20000},
                "lead": {"think_s": 2000, "tool_s": 12000, "wait_s": 0, "idle_s": 6000, "setup_s": 0},
                "verdict": {"bound": "outer_tools", "share_pct": 60.0, "basis": "matched call/return spans"},
                "overlap": {"lead_blocked_on_sim_s": 8000},
                "parallelism": {"agents_avg": 1.1, "helpers_avg": 0.1, "helper_ceiling": 16, "helpers_used": 2},
            },
            "jobs": {"running": [{"stage": "gsim simulation"}]},
        },
    }
    codes = {f["code"] for f in diagnose.diagnose(snap)}
    assert {"TOOL_BOUND", "HELPERS_UNDERUSED", "UNFINISHED_JOBS"} <= codes
    assert "SIM_ON_CRITICAL_PATH" not in codes  # Coincident intervals do not prove an awaited job.


def test_findings_with_one_title_merge_their_details():
    snap = {
        "experiment": {},
        "sections": {
            "progress": {
                "alerts": [
                    {"code": "X", "severity": "warn", "title": "Repeated failure", "detail": "plane a"},
                    {"code": "X", "severity": "warn", "title": "Repeated failure", "detail": "plane b"},
                ]
            }
        },
    }
    (finding,) = [f for f in diagnose.diagnose(snap) if f["code"] == "X"]
    assert finding["occurrences"] == 2 and finding["items"] == ["plane a", "plane b"]
    assert finding["detail"] == "plane a (+1 more)"


def test_subset_simulation_request_is_judged_by_what_it_returned(tmp_path):
    ws = tmp_path / "ws"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (channel / "simreq_j1.json").write_text(json.dumps({"tiers": "L3", "sim": "gsim"}))
    (channel / "simdone_j1").write_text("")
    # all_pass is False for the whole suite, but the one capsule asked about passed.
    (channel / "simresp_j1.json").write_text(json.dumps({"all_pass": False, "n_result_rows": 1, "n_passed": 1}))
    run = tmp_path / "run"
    run.mkdir()
    ctx = _ctx(run, {})
    ctx.experiment["workspace"] = str(ws)
    section = jobs.read(ctx)
    assert section["sim"]["done"] == 1 and section["sim"]["failed"] == 0


def test_a_live_runner_is_found_before_its_run_has_rounds(tmp_path):
    tasks, proc = tmp_path / "tasks", tmp_path / "proc"
    run = tasks / "t" / "merlin" / "out" / "runs" / "tgt" / "phase1" / "exp-new"
    run.mkdir(parents=True)
    _fake_proc(proc, 4, (88, "exp-new", "0-1"), "/")
    d = discovery.Discovery(tasks, proc_root=proc)
    (item,) = d.scan(now=time.time())
    assert item["state"] == "LIVE" and item["runner_pid"] == 88
    # The same directory gains its markers later; a cached listing must not hide it.
    later = run.parent / "exp-later"
    later.mkdir()
    assert {e["run_id"] for e in d.scan()} == {"exp-new"}
    (later / "joint_campaign.json").write_text("{}")
    assert {e["run_id"] for e in d.scan()} == {"exp-new", "exp-later"}


def test_inspecting_a_timeline_lane_profiles_its_jobs(tmp_path):
    ws = tmp_path / "ws"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (channel / "simreq_j1.json").write_text(json.dumps({"tiers": "L3", "sim": "gsim", "capsules": ["c1"]}))
    (channel / "simdone_j1").write_text("")
    (channel / "simresp_j1.json").write_text(json.dumps({"n_result_rows": 1, "n_passed": 0}))
    run = tmp_path / "run"
    run.mkdir()
    ctx = _ctx(run, {})
    ctx.experiment["workspace"] = str(ws)
    result = jobs.inspect(ctx, "L3")
    assert result["title"] == "lane L3"
    sims = next(s for s in result["sections"] if s["title"] == "simulation jobs")
    assert sims["lines"][0].startswith("1 simulation request(s): fail 1")


def test_background_command_is_not_outer_tool_blocking_and_peak_is_exact(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    agents = {"agents": [
        {"id": "lead", "role": "lead", "outer_tool_calls_recorded": True,
         "spans": [[1000, 2000, "turn"], [1100, 1900, "tool"], [1100, 1110, "tool_call"]]},
        {"id": "h1", "role": "subagent", "spans": [[1500, 1500.01, "turn"]]},
        {"id": "h2", "role": "subagent", "spans": [[1500, 1500.01, "turn"]]}]}
    result = timeline.read(_ctx(run, {"agents": agents}))
    assert result["lead"]["command_lifetime_s"] == 800
    assert result["lead"]["outer_tool_call_s"] == 10
    assert result["lead"]["unclassified_non_tool_s"] == 990
    assert result["lead"]["pure_deliberation_s"] is None
    assert "not separated" in result["verdict"]["explanation"]
    assert result["parallelism"]["agents_peak"] == 3
    assert result["parallelism"]["helpers_peak"] == 2
    assert result["parallelism"]["agents_peak_sampled"] == 1
    assert result["parallelism"]["helpers_avg"] == 0.0
    assert result["overlap"]["lead_blocked_on_sim_s"] is None


def test_half_open_helper_intervals_do_not_overlap():
    assert timeline._peak([[(0, 1)], [(1, 2)]]) == 1


def test_helper_background_command_does_not_extend_its_task_lifetime(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    agents = {"agents": [{"id": "lead", "role": "lead", "spans": [[1000, 2000, "turn"]]},
                         {"id": "h", "role": "subagent", "spans": [[1100, 1200, "turn"],
                                                                          [1150, 1900, "tool"]]}]}
    result = timeline.read(_ctx(run, {"agents": agents}))
    assert result["parallelism"]["helpers_avg"] == 0.1
    assert result["parallelism"]["agents_avg"] == 1.1


def test_relaunched_round_leads_all_count_as_lead_time(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    agents = {
        "agents": [
            {"id": "lead-r0", "role": "lead", "spans": [[1100, 1400, "turn"], [1200, 1300, "tool"]]},
            {"id": "lead-r1", "role": "lead", "spans": [[1500, 1900, "turn"], [1600, 1700, "tool"]]},
        ],
        "totals": {"helper_ceiling": 16},
    }
    lead = timeline.read(_ctx(run, {"agents": agents, "jobs": {"spans": []}}))["lead"]
    assert lead["sessions"] == 2
    assert (lead["setup_s"], lead["tool_s"]) == (100.0, 200.0)
    assert lead["think_s"] == 500.0
    assert lead["idle_s"] == 200.0  # only the relaunch gap 1400..1500 and the tail 1900..2000
