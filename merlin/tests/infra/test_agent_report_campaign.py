"""The campaign figure (merlin.agentreport.campaign): synthetic records in, figure and notes out.

Paired where a reader could silently return nothing: the performance binding FINDS a bound row and
REFUSES (and counts) every unbound one; the stitched pass curve CARRIES a test case across attempts
and DROPS it when a later grade fails it.
"""

from __future__ import annotations

import json

import pytest

from merlin.agentreport.campaign.figure import Stitch, pass_curve
from merlin.agentreport.campaign.manifest import cohort_of, from_dict
from merlin.agentreport.campaign.records import activity_of_command, bind_perf, read_campaign

T0 = 1_800_000_000.0
SRC = "a" * 64


def _row(**over):
    row = {
        "capsule": "case_a",
        "barrier_tier": "L3",
        "barrier_status": "pass",
        "barrier_cycles": 1000,
        "barrier_timing_identity": {"engine": "rtlsim"},
        "tier_boundary": {"common": {"compiler_sha256": SRC}},
        "performance": {
            "capsule_sha256": "d1",
            "structural_comparison_binding": {"capsule_sha256": "d1"},
            "structural_roofline": {"useful_macs": 25600, "structural_peak_macs_per_cycle": 256,
                                    "structural_compute_floor_cycles": 100},
        },
    }
    row.update(over)
    return row


def test_binding_finds_a_bound_row():
    excluded = {}
    from collections import defaultdict

    excluded = defaultdict(int)
    rec = bind_perf(_row(), T0, "rtlsim", excluded)
    assert rec is not None and rec["cycles"] == 1000 and rec["floor_cycles"] == 100
    assert rec["model_cycles"] is None and excluded["model_estimate_unbound"] == 1


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"barrier_timing_identity": {"engine": "other"}}, "no_rtl_cycles"),
        ({"barrier_cycles": 0}, "no_rtl_cycles"),
        ({"tier_boundary": {"common": {"compiler_sha256": "short"}}}, "unbound_digest"),
        ({"performance": {"capsule_sha256": "d1", "structural_comparison_binding": {"capsule_sha256": "d2"}}}, "unbound_digest"),
    ],
)
def test_binding_refuses_and_counts_unbound_rows(over, reason):
    from collections import defaultdict

    excluded = defaultdict(int)
    assert bind_perf(_row(**over), T0, "rtlsim", excluded) is None
    assert excluded[reason] == 1


def test_roofline_above_measured_cycles_is_not_used():
    from collections import defaultdict

    excluded = defaultdict(int)
    perf = _row()["performance"]
    perf["structural_roofline"]["structural_compute_floor_cycles"] = 5000  # floor > measured: unbound
    rec = bind_perf(_row(performance=perf), T0, "rtlsim", excluded)
    assert rec is not None and rec["floor_cycles"] is None and excluded["roofline_unavailable"] == 1


def _stitch_data(events):
    attempts = [
        {"id": "c1", "live": False, "agent_start": T0, "last_record": T0 + 600},
        {"id": "c2", "live": True, "agent_start": T0 + 3600, "last_record": T0 + 4200},
    ]
    return {"attempts": attempts, "events": events, "grades": [], "rtl_jobs": []}


def test_stitched_passes_carry_across_attempts():
    ev = [
        {"t": T0 + 60, "test_case": "x", "tier": "L2", "status": "pass", "attempt": "c1"},
        {"t": T0 + 120, "test_case": "x", "tier": "L3", "status": "pass", "attempt": "c1"},
        {"t": T0 + 3700, "test_case": "y", "tier": "L2", "status": "pass", "attempt": "c2"},
    ]
    data = _stitch_data(ev)
    S = Stitch(data)
    assert S.total == pytest.approx(20.0)  # two 10-minute windows, the hour between them dropped
    curve, state = pass_curve(data, S)
    assert curve[-1][1:] == (2, 1)  # x is still counted (and still L3) in the second attempt
    assert state == {"x": "L3", "y": "L2"}


def test_stitched_passes_drop_on_a_later_failure():
    ev = [
        {"t": T0 + 60, "test_case": "x", "tier": "L2", "status": "pass", "attempt": "c1"},
        {"t": T0 + 3700, "test_case": "x", "tier": "L2", "status": "fail", "attempt": "c2"},
    ]
    data = _stitch_data(ev)
    curve, state = pass_curve(data, Stitch(data))
    assert max(c[1] for c in curve) == 1 and curve[-1][1] == 0 and state == {}


def test_activity_rule():
    assert activity_of_command("sed -n '1,80p' a.py | head -5") == "reading"
    assert activity_of_command("python3 - <<'PY'\nopen('a').write_text('x')\nPY") == "writing"
    assert activity_of_command("python3 agent_selfcheck.py --attach 3_1") == "checks"
    assert activity_of_command("make -j4") == "shell"


def test_cohort_priority_beats_display_order():
    rules = [{"id": "dense", "default": True, "priority": 9}, {"id": "cal", "calibration_profile": True, "priority": 0}]
    assert cohort_of(rules, "m1", {}, {"m1"}) == "cal"
    assert cohort_of(rules, "m2", {}, {"m1"}) == "dense"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _iso(t):
    import datetime as dt

    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def test_synthetic_records_build_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from merlin.agentreport.campaign.cli import run

    run_dir, ws, home = tmp_path / "run", tmp_path / "ws", tmp_path / "home"
    # one grade: case_a passes L2, timed by its stage record
    tok = str(int((T0 + 30) * 1e9))
    (run_dir / "qa_history").mkdir(parents=True)
    (run_dir / "qa_history" / f"verdict_fast_{tok}.json").write_text(json.dumps({
        "graded_at": _iso(T0 + 300),
        "per_capsule": [{"capsule": "case_a", "status": "pass", "tiers": {"L0": "pass", "L1": "pass", "L2": "pass"}}],
    }))
    timing = run_dir / "_qa_work" / f"fruns_{tok}" / "host-stage-timing" / "setup-timing" / "t.json"
    timing.parent.mkdir(parents=True)
    ns = lambda t: int(t * 1e9)  # noqa: E731
    timing.write_text(json.dumps({"capsule": "case_a", "stages": [
        {"stage": "capsule_execution", "stage_id": 1, "parent_stage_id": None, "started_ns": ns(T0 + 40), "finished_ns": ns(T0 + 200)},
        {"stage": "oracle_compile_link_admit", "stage_id": 2, "parent_stage_id": 1, "started_ns": ns(T0 + 50), "finished_ns": ns(T0 + 120)},
        {"stage": "oracle_simulation_readback.spike", "stage_id": 3, "parent_stage_id": 1, "started_ns": ns(T0 + 120), "finished_ns": ns(T0 + 190)},
    ]}))
    # two RTL certifications of case_a, the second faster
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    for i, (t, cyc) in enumerate(((T0 + 400, 2000), (T0 + 500, 1000))):
        ch.joinpath(f"simtiming_j{i}.json").write_text(json.dumps({"started_at_unix": t - 60, "finished_at_unix": t}))
        ch.joinpath(f"simreq_j{i}.json").write_text(json.dumps({"submitted_at": t - 70}))
        ch.joinpath(f"simresp_j{i}.json").write_text(json.dumps({"per_capsule": [_row(barrier_cycles=cyc)]}))
    # a lead agent: one turn, one command, token counters
    _write_jsonl(home / "sessions" / "rollout-1.jsonl", [
        {"type": "session_meta", "timestamp": _iso(T0), "payload": {"id": "lead", "timestamp": _iso(T0)}},
        {"type": "event_msg", "timestamp": _iso(T0), "payload": {"type": "task_started"}},
        {"type": "event_msg", "timestamp": _iso(T0 + 100), "payload": {"type": "item_completed", "started_at_ms": (T0 + 90) * 1000,
         "completed_at_ms": (T0 + 100) * 1000, "item": {"type": "CommandExecution", "command": ["bash", "-lc", "rg foo"]}}},
        {"type": "event_msg", "timestamp": _iso(T0 + 60), "payload": {"type": "token_count", "info": {"total_token_usage": {
            "input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50}}}},
        {"type": "event_msg", "timestamp": _iso(T0 + 590), "payload": {"type": "token_count", "info": {"total_token_usage": {
            "input_tokens": 5000, "cached_input_tokens": 4000, "output_tokens": 300}}}},
        {"type": "event_msg", "timestamp": _iso(T0 + 600), "payload": {"type": "task_complete"}},
    ])
    m = from_dict({
        "campaign": "synthetic", "rtl_engine": "rtlsim", "reserved_cores": 4,
        "attempts": [{"continuation": "c1", "run_dir": str(run_dir), "workspace": str(ws), "agent_home": str(home)}],
        "cohorts": [{"id": "all", "label": "All", "default": True}],
    })
    data = read_campaign(m)
    assert data["test_cases"]["case_a"]["cohort"] == "all"
    out = run(m, str(tmp_path / "out"), formats=("png",))
    notes = json.loads((out / "notes.json").read_text())
    assert (out / "campaign_figure.png").stat().st_size > 0
    assert notes["final"] == {"test_cases": 1, "l2": 1, "l3": 1}
    assert notes["performance"]["speedup_geomean"] == pytest.approx(2.0)
    assert notes["performance"]["speedup_test_cases"] == 1
    assert notes["tokens"]["output"] == pytest.approx(250)  # deltas inside the window only
    assert "capsule" not in (out / "README.md").read_text().lower()
