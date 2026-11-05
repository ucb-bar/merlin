"""The insight panels: one generic shape, numbers from the run's own records, private names gated."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml
from merlin_experiments.tracking.console import diagnose, model
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import INSIGHTS, SOURCES, Context
from merlin_experiments.tracking.console.sources import (
    agents,
    attempts,
    certification,
    cost,
    focus,
    grader,
    insight_common,
    integrity,
    performance,
    tools,
)

from merlin.common.paths import repo_root

NOW = time.time()


def _write(path: Path, document, at: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    if at is not None:
        os.utime(path, (at, at))
    return path


def _ctx(experiment: dict, *, private: bool = False, shared: dict | None = None, state: dict | None = None):
    return Context(
        experiment=experiment,
        reader=Reader(),
        state=state if state is not None else {},
        operator_private=private,
        now=NOW,
        shared=shared or {},
        live=True,
    )


def _layout(tmp_path: Path, name: str = "run-v2") -> dict:
    """A run laid out as ``out/runs/<target>/<suite>/<run>`` with its agent workspace."""
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / name
    ws = tmp_path / "out" / "build" / "ws" / name / "workspace"
    (run / "qa_history").mkdir(parents=True)
    (ws / ".qa_channel").mkdir(parents=True)
    return {
        "id": str(run),
        "run_id": name,
        "run_dir": str(run),
        "workspace": str(ws),
        "state": "LIVE",
        "task": str(tmp_path),
        "runner_pid": None,
    }


def _manifest(tmp_path: Path, attempts_: list[dict]) -> None:
    _write(
        tmp_path / "out" / "artifacts" / "agentic-report" / "campaign-figures" / "tgt" / "c.campaign.yaml",
        yaml.safe_dump(
            {"schema": "merlin.agentreport.campaign_manifest.v1", "root": str(tmp_path), "attempts": attempts_}
        ),
    )


def _simresp(ws: str, i: int, cycles: dict[str, float], at: float) -> None:
    rows = [{"capsule": name, "barrier_cycles": c} for name, c in cycles.items()]
    _write(Path(ws) / ".qa_channel" / f"simresp_promo0_{i:04d}.json", {"sim": "gsim", "per_capsule": rows}, at)


def _shape(section: dict) -> None:
    assert {"title", "summary", "table", "series", "notes"} <= set(section)
    for line in section["summary"]:
        assert set(line) == {"label", "value", "level", "key"} and isinstance(line["value"], str)
        assert line["value"] != "0" or "not recorded" not in line["label"]
    if section["table"]:
        width = len(section["table"]["columns"])
        assert all(len(r["cells"]) == width for r in section["table"]["rows"])


def _values(section: dict) -> dict[str, str]:
    return {line["label"]: line["value"] for line in section["summary"]}


# -- contract -----------------------------------------------------------------------------------------
def test_every_insight_is_registered_and_mirrored_in_the_frontend():
    names = [name for name, _ in SOURCES]
    assert all(n in names for n in INSIGHTS)
    # Dependencies come first: an insight reads the sections it summarises through ctx.shared.
    for insight, needs in (
        ("performance", "perf"),
        ("tools", "timeline"),
        ("grader", "progress"),
        ("certification", "capsules"),
        ("cost", "certification"),
        ("focus", "agent_view"),
    ):
        assert names.index(needs) < names.index(insight)
    proto = repo_root() / "packages" / "merlin-experiments" / "console" / "internal" / "proto" / "proto.go"
    declared = next(line for line in proto.read_text().splitlines() if line.startswith("var InsightNames"))
    assert declared.split("{", 1)[1].rstrip("}").replace('"', "").replace(" ", "").split(",") == list(INSIGHTS)


def test_an_empty_run_renders_every_panel_without_inventing_zeros(tmp_path):
    experiment = _layout(tmp_path)
    collector = model.Collector(experiment, clock=lambda: NOW)
    snapshot = collector.refresh()
    for name in INSIGHTS:
        section = snapshot["sections"][name]
        assert "error" not in section, (name, section)
        _shape(section)
    g = snapshot["sections"]["grader"]
    assert g["verdicts"]["p50_s"] is None and _values(g)["grade time p50 / p95"] == "—"
    assert _values(snapshot["sections"]["performance"])["median array utilization"] == "—"
    assert snapshot["sections"]["cost"]["per_new_l3"] == {"attempt": None, "watched": None}


# -- performance --------------------------------------------------------------------------------------
def test_performance_speedups_families_and_the_previous_attempt(tmp_path):
    previous = _layout(tmp_path, "run-v1")
    current = _layout(tmp_path, "run-v2")
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "run_dir": os.path.relpath(previous["run_dir"], tmp_path),
                "workspace": os.path.relpath(previous["workspace"], tmp_path),
            },
            {
                "continuation": "v2",
                "live": True,
                "run_dir": os.path.relpath(current["run_dir"], tmp_path),
                "workspace": os.path.relpath(current["workspace"], tmp_path),
            },
        ],
    )
    _simresp(previous["workspace"], 0, {"mm_a": 4000.0, "conv_b": 9000.0}, NOW - 9000)
    _simresp(current["workspace"], 0, {"mm_a": 2000.0, "conv_b": 8000.0, "add_c": 500.0}, NOW - 7200)
    _simresp(current["workspace"], 1, {"mm_a": 1000.0, "conv_b": 8000.0}, NOW - 600)
    _write(
        Path(current["workspace"]) / ".qa_channel" / "simresp_promo0_spike.json",
        {"sim": "spike", "per_capsule": [{"capsule": "mm_a", "barrier_cycles": 1.0}]},
        NOW - 60,
    )
    ctx = _ctx(current)
    insight_common.share_rows(
        ctx,
        "capsules",
        [{"name": "mm_a", "op": "matmul"}, {"name": "conv_b", "op": "conv2d"}, {"name": "add_c", "op": "residual_add"}],
    )
    insight_common.share_rows(
        ctx, "perf", [{"capsule": "mm_a", "useful_macs": 25600, "peak_macs_per_cycle": 256, "spike_model_cycles": 3000}]
    )
    section = performance.read(ctx)
    _shape(section)
    # Geomean over re-measured cases: mm_a 2000->1000 (x2), conv_b unchanged (x1); spike never counts.
    assert abs(section["speedup"] - 2**0.5) < 1e-9
    assert abs(section["vs_previous"] - (4 * 9 / 8) ** 0.5) < 1e-9
    families = {f["family"]: f for f in section["families"]}
    assert families["dense"]["cases"] == 1 and families["conv"]["cases"] == 1 and families["epilogue"]["cases"] == 1
    assert abs(families["dense"]["util_median_pct"] - 10.0) < 1e-9
    assert section["model_gap"] == {"compared": 1, "median_ratio": 3.0, "over_2x": 1}
    # Public mode names no test case among the movers.
    assert section["movers"]["up"] == [[2.0, None]]
    assert all("mm_a" not in str(r["cells"]) for r in section["table"]["rows"])


def test_reference_targets_are_operator_private(tmp_path):
    experiment = _layout(tmp_path)
    _simresp(experiment["workspace"], 0, {"mm_a": 1500.0}, NOW - 60)
    _write(
        tmp_path / "out" / "artifacts" / "perf-studies" / "study" / "pairs.json",
        {
            "pairs": [
                {
                    "id": "p",
                    "agent_case": "mm_a",
                    "exo_kernel": "ref_mm",
                    "exo_metric": "call1_cycles",
                    "exo_scale": 2,
                    "exo_gsim": {"call1_cycles": 500},
                }
            ]
        },
    )
    public = performance.read(_ctx(experiment))
    assert public["targets"] == [] and "ref_mm" not in json.dumps(public)
    private = performance.read(_ctx(experiment, private=True))
    assert private["targets"][0]["reference_cycles"] == 1000 and private["targets"][0]["ratio"] == 1.5


# -- agent tools and focus ----------------------------------------------------------------------------
def _rollout(path: Path, ws: str, commands: list[str], edits: list[tuple[float, str]]) -> None:
    def stamp(at):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(at)) + ".000Z"

    rows = []
    for i, text in enumerate(commands):
        rows.append(
            {
                "timestamp": stamp(NOW - 4000 + i * 10),
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {"type": "CommandExecution", "command": ["bash", "-lc", text]},
                },
            }
        )
    for at, rel in edits:
        rows.append(
            {
                "timestamp": stamp(at),
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "FileChange",
                        "changes": {
                            f"{ws}/{rel}": {"type": "update", "unified_diff": "+from merlin.sched import plan"}
                        },
                    },
                },
            }
        )
    rows.append(
        {
            "timestamp": stamp(NOW - 100),
            "type": "response_item",
            "payload": {"type": "function_call", "name": "spawn_agent", "arguments": "{}"},
        }
    )
    _write(path, "\n".join(json.dumps(r) for r in rows) + "\n")


def test_tool_usage_counts_windows_and_unused_performance_tools(tmp_path):
    experiment = _layout(tmp_path)
    ws = experiment["workspace"]
    commands = ["python3 agent_selfcheck.py --sim gsim --suite scalability"] * 3 + ["ls"] * 60
    commands += ["python3 -c 'print(estimated_whole_soc_cycles)'"]
    _rollout(
        tmp_path / "rollout.jsonl",
        ws,
        commands,
        [(NOW - 300, "submission/compiler/conv_lower.py"), (NOW - 200, "iteration_notes.md")],
    )
    ctx = _ctx(
        experiment,
        shared={
            "agents": {"totals": {"working": 1, "helper_ceiling": 16}},
            "timeline": {"lead": {"think_s": 90, "tool_s": 10, "wait_s": 0}, "concurrency": {"agents": [1, 1, 2]}},
        },
    )
    agents.index_for(ctx).files = [tmp_path / "rollout.jsonl"]
    section = tools.read(ctx)
    _shape(section)
    counts = section["counts"]["all"]
    assert counts["command"] == 64 and counts["gsim_check"] == 3 and counts["scalability_check"] == 3
    assert counts["perf_fields"] == 1 and counts["sched_library"] == 2 and counts["iteration_notes"] == 1
    assert counts["spawn"] == 1 and counts["edit"] == 2
    # The commands ran over an hour ago except the last few; the 30-minute window sees only edits/spawn.
    assert section["counts"]["30m"].get("command", 0) == 0 and section["counts"]["30m"]["edit"] == 2
    assert (
        "isa_tools debug" in section["unused_perf_tools"]
        and "cycle / roofline fields" not in section["unused_perf_tools"]
    )
    snapshot = {"generated_at": NOW, "experiment": {}, "sections": {"tools": section}}
    assert "PERF_TOOLS_UNUSED" in {f["code"] for f in diagnose.diagnose(snapshot)}
    # The same streams again: nothing is counted twice.
    assert tools.read(ctx)["counts"]["all"]["command"] == 64

    shared = {
        "agents": {
            "agents": [
                {"role": "lead", "state": "working", "commands": 64, "last_message": "tuning conv", "current": "pytest"}
            ]
        },
        "agent_view": {"selfchecks": {"pending": 2, "oldest_pending_at": NOW - 1200}},
    }
    public = focus.read(_ctx(experiment, shared=shared))
    _shape(public)
    assert {f["path"] for f in public["files"]} == {"submission/compiler/…", "./…"}
    assert public["families"] == {"conv": 1} and "conv_lower" not in json.dumps(public)
    assert _values(public)["lead says"] == "operator-private"
    private = focus.read(_ctx(experiment, private=True, shared=shared))
    assert {f["path"] for f in private["files"]} == {"submission/compiler/conv_lower.py", "iteration_notes.md"}
    assert _values(private)["lead says"] == "tuning conv"


# -- grader -------------------------------------------------------------------------------------------
def test_grader_durations_superseded_verdicts_and_closures(tmp_path):
    experiment = _layout(tmp_path)
    run, ws = Path(experiment["run_dir"]), Path(experiment["workspace"])
    for i, (start, took) in enumerate(((NOW - 7200, 1200), (NOW - 5000, 1500), (NOW - 3000, 1800))):
        tick = int(start * 1e9)
        graded = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(start + took))
        _write(
            run / "qa_history" / f"verdict_fast_{tick}.json",
            {
                "graded_at": graded,
                "snapshot_current": False,
                "n_capsules": 3,
                "n_passed": i,
                "background_grade": {"status": "superseded_snapshot", "reason": "background source superseded"},
                "per_capsule": [{"capsule": "a", "status": "pass"}, {"capsule": "b", "status": "gated"}],
            },
        )
        _write(
            run / "_qa_work" / "background_snapshot_owners" / f"{tick}.json",
            {
                "token": str(tick),
                "started_at_epoch": start,
                "state": "complete",
                "admission_closed": True,
                "admission_reason": "snapshot admission deadline reached",
            },
            start + took,
        )
    for i in range(3):
        rid = f"op_idle_{int((NOW - 600 + i) * 1e9)}_{i:024x}"
        _write(ws / ".qa_channel" / f"req_{rid}.json", {"request_id": rid})
        if i:
            _write(ws / ".qa_channel" / f"resp_{rid}.json", {"passed": False})
    _write(
        ws / ".qa_channel" / "simdefer_promo0_x.json",
        {"reason": "protected outstanding promotion identity unavailable"},
    )
    shared = {
        "lifecycle": {"turns": [{"startup_end": NOW - 7300}]},
        "progress": {"grades": [{"state": "grading", "n_selected": 10, "n_completed": 4}]},
    }
    section = grader.read(_ctx(experiment, shared=shared))
    _shape(section)
    v = section["verdicts"]
    assert v["n"] == 3 and v["delivered"] == 0 and v["superseded"] == 3
    assert abs(v["p50_s"] - 1500) < 1 and abs(v["first_after_s"] - 1300) < 1
    assert section["owners"]["admission"] == {"snapshot admission deadline reached": 3}
    assert section["owners"]["deferred"] == {"protected outstanding promotion identity unavailable": 1}
    assert section["idle"]["requests"] == 3 and section["idle"]["answered"] == 2
    assert abs(section["idle"]["last_at"] - (NOW - 598)) < 1
    assert section["backlog"] == {"grading": 1, "remaining": 6}
    assert section["owners"]["max_concurrent"] == 1
    snapshot = {"generated_at": NOW, "experiment": {}, "sections": {"grader": section}}
    assert "GRADES_UNDELIVERED" in {f["code"] for f in diagnose.diagnose(snapshot)}

    # Once the agent's completed-grade file records merging two of them, those count as delivered.
    ticks = sorted(int(p.stem[len("verdict_fast_") :]) for p in (run / "qa_history").glob("verdict_fast_*.json"))
    _write(
        ws / "qa" / "latest_completed_grade.json",
        {"completed_rows": {"recorded_grades": [str(ticks[0]), str(ticks[2])]}},
    )
    merged = grader.read(_ctx(experiment, shared=shared))
    assert merged["verdicts"]["delivered"] == 2 and merged["verdicts"]["superseded"] == 1
    assert [r["cells"][3] for r in merged["table"]["rows"]] == ["merged", "superseded", "merged"]
    snapshot = {"generated_at": NOW, "experiment": {}, "sections": {"grader": merged}}
    assert "GRADES_UNDELIVERED" not in {f["code"] for f in diagnose.diagnose(snapshot)}


# -- integrity ----------------------------------------------------------------------------------------
def test_integrity_counts_every_grade_failure_spelling(tmp_path):
    experiment = _layout(tmp_path)
    run = Path(experiment["run_dir"])
    log = _write(
        tmp_path / "campaign.log",
        "\n".join(
            [
                "[first-grade] grade error: RuntimeError: 3 completion processing operations failed",
                "[snapshot 1792901853332202555] grade failed: RuntimeError: original snapshot read lease custody changed",
                "[verilator attempt 2] grade error: timeout",
                "[promote] L2: 1 pass, 0 fail",
            ]
        )
        + "\n",
    )
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "live": True,
                "run_dir": os.path.relpath(run, tmp_path),
                "launch_log": os.path.relpath(log, tmp_path),
            }
        ],
    )
    public = integrity.read(_ctx(experiment, shared={"machine": {}}))
    assert public["log"]["counts"]["grade_error"] == 3


def test_integrity_counts_log_signals_and_hides_their_text_in_public(tmp_path):
    experiment = _layout(tmp_path)
    run = Path(experiment["run_dir"])
    log = _write(
        tmp_path / "campaign.log",
        "\n".join(
            [
                json.dumps({"at": "2026-10-24T23:45:40+00:00", "event": "cpu_reservation_verified"}),
                "[first-grade] grade error: RuntimeError: original snapshot input content differs from its marker",
                "[first-grade] completed promotion skipped: ValueError: secret_case lacks admission",
                "[promote] L2: 1 pass, 0 fail",
            ]
        )
        + "\n",
    )
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "live": True,
                "run_dir": os.path.relpath(run, tmp_path),
                "launch_log": os.path.relpath(log, tmp_path),
            }
        ],
    )
    _write(run / "feedback_health.json", {"healthy": True, "broker_starts": 2, "expired": 0, "stranded": 0})
    _write(run / "grade_health.json", {"alert": False, "consecutive_identical_failures": 0})
    shared = {
        "machine": {
            "experiment_cpus": [4, 5],
            "volume": {"free_pct": 40.0, "free_bytes": 2**38, "level": "ok"},
            "allocations": [
                {"name": "other-run", "kind": "run", "cpus": "0-3"},
                {"name": "run-v2", "kind": "run", "cpus": "4-5"},
            ],
            "cores": [
                {"cpu": 0, "util_pct": 50.0, "holders": ["other-run"]},
                {"cpu": 4, "util_pct": 100.0, "holders": ["run-v2"]},
            ],
        }
    }
    public = integrity.read(_ctx(experiment, shared=shared))
    _shape(public)
    assert public["log"]["counts"]["marker"] == 1 and public["log"]["counts"]["grade_error"] == 1
    assert public["log"]["counts"]["promotion_skipped"] == 1 and "restart" not in public["log"]["counts"]
    assert "secret_case" not in json.dumps(public) and public["log"]["path"] is None
    assert public["neighbours"] == [{"name": "other-run", "cpus": "0-3", "busy_cores": 0.5, "overlap": 0}]
    assert public["broker"]["broker_starts"] == 2
    private = integrity.read(_ctx(experiment, private=True, shared=shared))
    assert any("secret_case" in r["cells"][2] for r in private["table"]["rows"])
    snapshot = {"generated_at": NOW, "experiment": {}, "sections": {"integrity": public}}
    assert "RUN_INTEGRITY" in {f["code"] for f in diagnose.diagnose(snapshot)}


# -- attempts -----------------------------------------------------------------------------------------
def test_runtime_text_parses_like_the_service_accounting():
    assert attempts.runtime_seconds("1d 5h 28min 55.000s") == 86400 + 5 * 3600 + 28 * 60 + 55
    assert attempts.runtime_seconds("25min 670ms") == 25 * 60 + 0.67
    assert attempts.runtime_seconds("not recorded") is None


def test_attempt_timeline_from_the_campaign_manifest(tmp_path):
    first = _layout(tmp_path, "attempt1")
    current = _layout(tmp_path, "attempt2")
    start1 = NOW - 9000
    log1 = _write(
        tmp_path / "a1.log",
        json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(start1))})
        + "\nmerlin_experiments.spec.SpecError: public scalability source/input/reference bytes changed\n",
    )
    log2 = _write(
        tmp_path / "a2.log",
        json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(NOW - 3600))}) + "\n",
    )
    accounting = _write(
        tmp_path / "accounting.log",
        "\n".join(
            [
                "Running as unit: x.service",
                "Finished with result: exit-code",
                "Main processes terminated with: code=exited/status=1",
                "Service runtime: 15min 30.000s",
                "Running as unit: x.service",
            ]
        )
        + "\n",
    )
    rel = lambda p: os.path.relpath(p, tmp_path)  # noqa: E731
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v8",
                "attempt": 1,
                "run_dir": rel(first["run_dir"]),
                "workspace": rel(first["workspace"]),
                "launch_log": rel(log1),
                "accounting_log": rel(accounting),
                "accounting_entry": 0,
            },
            {
                "continuation": "v8",
                "attempt": 2,
                "live": True,
                "run_dir": rel(current["run_dir"]),
                "workspace": rel(current["workspace"]),
                "launch_log": rel(log2),
                "accounting_log": rel(accounting),
                "accounting_entry": 1,
            },
        ],
    )
    _write(Path(current["run_dir"]) / "qa_history" / "verdict_fast_1.json", {}, NOW - 3000)
    _write(Path(current["workspace"]) / "PREVIOUS_SESSION.md", "# previous\n")
    section = attempts.read(_ctx(current))
    _shape(section)
    one, two = section["attempts"]
    assert one["label"] == "v8.1" and one["exit"] == "exited/1" and one["runtime_s"] == 930
    assert abs(one["start"] - start1) < 1 and abs(one["end"] - (start1 + 930)) < 1
    assert one["stop_detail"].startswith("merlin_experiments.spec.SpecError") and one["grades"] == 0
    assert two["live"] and two["end"] is None and abs(two["runtime_s"] - 3600) < 2 and two["grades"] == 1
    assert abs(two["first_verdict_s"] - 600) < 2 and two["handoff"] is True and one["handoff"] is False
    assert section["totals"]["lost_s"] == 930
    assert _values(section)["attempts"] == "2 (this is v8.2)"


# -- certification and cost ---------------------------------------------------------------------------
def test_certification_matrix_by_cohort_and_cost_per_new_certificate(tmp_path):
    previous = _layout(tmp_path, "run-v1")
    current = _layout(tmp_path, "run-v2")
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "run_dir": os.path.relpath(previous["run_dir"], tmp_path),
                "workspace": os.path.relpath(previous["workspace"], tmp_path),
            },
            {
                "continuation": "v2",
                "live": True,
                "run_dir": os.path.relpath(current["run_dir"], tmp_path),
                "workspace": os.path.relpath(current["workspace"], tmp_path),
            },
        ],
    )
    certs = {"<certs>": {"L2": {"e1": {"status": "pass"}}, "L3": {"e1": {"status": "pass"}}}}
    _write(
        Path(current["workspace"]) / "qa" / "tier_state.json",
        {
            "a": certs,
            "b": {"<certs>": {"L2": {"e2": {"status": "pass"}}, "L3": {"e2": {"status": "joined"}}}},
            "c": certs,
        },
    )
    _write(Path(previous["workspace"]) / "qa" / "tier_state.json", {"a": certs})
    _write(
        Path(current["workspace"]) / "qa" / "joint_physical_correctness.json",
        {
            "schema": "merlin.joint_physical_correctness.v1",
            "obligations": {"a": {"status": "pass"}, "b": {"status": "pending"}},
        },
    )
    _write(
        Path(current["run_dir"]) / "qa_history" / "verdict_fast_5.json",
        {
            "per_capsule": [
                {"capsule": "a", "status": "pass"},
                {"capsule": "b", "status": "incomplete", "failure_category": "NOT_RUN_IS_NOT_PASS"},
                {"capsule": "d", "status": "gated"},
            ]
        },
    )
    ctx = _ctx(current)
    insight_common.share_rows(
        ctx,
        "capsules",
        [
            {"name": "a", "cohort": "functional"},
            {"name": "b", "cohort": "functional"},
            {"name": "c", "cohort": "calibration"},
            {"name": "d", "cohort": "scalability"},
        ],
    )
    section = certification.read(ctx)
    _shape(section)
    rows = {c["cohort"]: c for c in section["cohorts"]}
    assert [c["cohort"] for c in section["cohorts"]] == ["functional", "calibration", "scalability"]
    assert rows["functional"] == {
        "cohort": "functional",
        "cases": 2,
        "L2": 2,
        "L3": 1,
        "L2_reproving": 0,
        "L3_reproving": 0,
        "tier_records": 2,
        "tier_certified_in_loop": True,
        "physical_required": 2,
        "physical_passed": 1,
        "newest": {"pass": 1, "incomplete": 1},
    }
    assert rows["calibration"]["newest"] == {"not graded": 1} and rows["scalability"]["newest"] == {"gated": 1}
    assert section["reasons"] == {"NOT_RUN_IS_NOT_PASS": 1}
    public = certification.inspect(ctx, "functional")
    assert all(line.endswith(" test cases") for sec in public["sections"] for line in sec["lines"])
    ctx.operator_private = True
    assert any("a" in s["lines"] for s in certification.inspect(ctx, "functional")["sections"])

    t = [NOW - 3600 + 600 * i for i in range(7)]
    tokens = {
        "total": {"total": 9_000_000, "fresh_input": 300_000, "output": 30_000},
        "series": {
            "t": t,
            "fresh_input": [i * 10_000 for i in range(7)],
            "cache_read": [i * 1_000_000 for i in range(7)],
            "cache_write": [0] * 7,
            "output": [i * 1_000 for i in range(7)],
        },
        "budget": {"total": 40_000_000, "used_pct": 22.5, "mode": "soft"},
    }
    spent = cost.read(_ctx(current, shared={"tokens": tokens, "certification": section}))
    _shape(spent)
    assert spent["new_l3"]["attempt"] == 1 and spent["new_l3"]["previous"] == 1
    assert spent["per_new_l3"]["attempt"] == 9_000_000
    assert abs(spent["per_hour"]["total"] - 6 * 1_011_000) < 1 and abs(spent["per_hour"]["uncached"] - 6 * 11_000) < 1


def _certification_panel(tmp_path, ledger, obligations, cohorts):
    current = _layout(tmp_path, "run-v1")
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "live": True,
                "run_dir": os.path.relpath(current["run_dir"], tmp_path),
                "workspace": os.path.relpath(current["workspace"], tmp_path),
            }
        ],
    )
    _write(Path(current["workspace"]) / "qa" / "tier_state.json", ledger)
    _write(
        Path(current["workspace"]) / "qa" / "joint_physical_correctness.json",
        {"schema": "merlin.joint_physical_correctness.v1", "obligations": obligations},
    )
    ctx = _ctx(current)
    insight_common.share_rows(ctx, "capsules", [{"name": n, "cohort": c} for n, c in cohorts.items()])
    section = certification.read(ctx)
    _shape(section)
    return ctx, section, {s["label"]: s["value"] for s in section["summary"]}


def test_certification_headline_counts_only_cohorts_the_loop_certifies(tmp_path):
    held = {"L2": {"status": "pass"}, "L3": {"status": "pass"}}
    cohorts = {
        "f1": "functional",
        "f2": "functional",
        "f3": "functional",
        "f4": "functional",
        "k1": "calibration",
        "k2": "calibration",
        "s1": "scalability",
        "s2": "scalability",
        "s3": "scalability",
        "u1": None,
    }
    # Tier records and obligations exist only for the functional cases; nothing names a cohort.
    _, section, summary = _certification_panel(
        tmp_path,
        {"f1": held, "f2": held, "f3": {"L2": {"status": "pass"}}},
        {name: {"status": "pass" if name == "f1" else "pending"} for name in ("f1", "f2", "f3", "f4")},
        cohorts,
    )
    loop = {c["cohort"]: c["tier_certified_in_loop"] for c in section["cohorts"]}
    assert loop == {"functional": True, "calibration": False, "scalability": False, "not recorded": False}
    # The headline is every case's numerics, all cohorts (the CASES panel's count); tiers follow on one line.
    assert section["summary"][0]["label"] == "cases passing numerics"
    assert summary["cases passing numerics"] == (
        "3/10 · calibration 0/2 · functional 3/4 · not in catalog 0/1 · scalability 0/3"
    )
    assert summary["functional tier certification"] == (
        "L2 3/4 (75%) · L3 2/4 (50%) · L3 physical newest source 1/4 (25%)"
    )
    assert "L2 certified" not in summary and "L3 certified" not in summary
    assert summary["not tier-certified in the loop"] == "calibration 2 · scalability 3 (final gate)"
    assert summary["no cohort recorded"] == "not recorded 1"  # unknown membership is not "final gate"
    assert section["totals"]["certifiable_cases"] == 4 and section["totals"]["cases"] == 10
    rows = {r["key"]: r["cells"] for r in section["table"]["rows"]}
    assert rows["scalability"][2:4] == ["final gate", "final gate"] and rows["functional"][2:4] == ["3", "2"]
    assert rows["not recorded"][2:4] == ["—", "—"]


def test_certification_repeats_the_cases_panel_count_and_shows_carried_cases_apart(tmp_path):
    from merlin_experiments.tracking.console.sources import agent_view, case_history

    held = {"L2": {"status": "pass"}, "L3": {"status": "pass"}}
    current = _layout(tmp_path)
    _write(Path(current["workspace"]) / "qa" / "tier_state.json", {"f1": held, "f2": held})
    rows = [
        {"capsule": "f1", "status": "pass", "tiers": {"L2": "pass"}},
        {"capsule": "f2", "status": "carried", "tiers": {}, "carried": {"reason": "sampled"}},
    ]
    _write(Path(current["run_dir"]) / "qa_history" / "verdict_fast_7.json", {"per_capsule": rows})
    shared = {"cases": {"totals": {"cases": 252, "passing": 249, "by_cohort": {"functional": [150, 153]}}}}
    ctx = _ctx(current, shared=shared)
    insight_common.share_rows(
        ctx, "capsules", [{"name": "f1", "cohort": "functional"}, {"name": "f2", "cohort": "functional"}]
    )
    section = certification.read(ctx)
    _shape(section)
    summary = _values(section)
    assert section["summary"][0]["label"] == "cases passing numerics"
    assert summary["cases passing numerics"] == "249/252 · functional 150/153"  # the CASES panel's own totals
    assert sorted(summary["newest verdict"].split(" · ")) == ["carried (sampled) 1", "pass 1"]
    (row,) = section["table"]["rows"]
    assert row["cells"][section["table"]["columns"].index("carried")] == "1"
    # Neither reader takes a carried row for a result of that grade.
    assert agent_view.executed_rows(rows) == 1
    digest = case_history.verdict_digest("v", 7)({"per_capsule": rows})
    assert set(digest["status"]) == {"f1"}


def test_an_invalidated_l3_reads_as_re_proving_not_a_drop(tmp_path):
    def case(newest, *earlier):
        return {
            "L2": {"status": "pass"},
            "L3": {"status": newest},
            "<certs>": {"L3": {f"e{i}": {"status": s} for i, s in enumerate(earlier)}},
        }

    ledger = {
        "current": case("pass", "pass"),
        "invalidated": case("pending", "pass", "pending"),  # passed for earlier bytes, now re-checking
        "never": case("pending", "pending"),  # no certificate yet: neither certified nor re-proving
        "only_ledger": {"<certs>": {"L3": {"e0": {"status": "pending"}, "e1": {"status": "pass"}}}},
    }
    cohorts = {name: "functional" for name in ledger}
    ctx, section, summary = _certification_panel(tmp_path, ledger, {}, cohorts)
    totals = section["totals"]
    assert (totals["L3"], totals["L3_reproving"], totals["L3_held"]) == (2, 1, 3)
    assert summary["functional tier certification"] == "L2 3/4 (75%) · L3 2/4 (50%) · re-proving 1"
    assert "L3 physical obligation" not in summary["functional tier certification"]
    assert any("re-proving" in note and "execution artifact" in note for note in section["notes"])
    # The held count (certified + re-proving) is what a previous attempt is compared against.
    assert certification._held(ledger["invalidated"], "L3") == "pass"
    assert certification.tier_status(ledger["invalidated"], "L3") == "reproving"
    assert certification.tier_status(ledger["never"], "L3") == "pending"
    ctx.operator_private = True
    groups = {s["title"]: s["lines"] for s in certification.inspect(ctx, "functional")["sections"]}
    assert groups["L3 re-proving (1)"] == ["invalidated"]


def test_an_attempt_read_over_the_byte_budget_says_so_until_it_is_whole(tmp_path):
    first = _layout(tmp_path, "attempt1")
    current = _layout(tmp_path, "attempt2")
    at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(NOW - 5000))
    log = _write(tmp_path / "a1.log", json.dumps({"at": at}) + "\n" + "[promote] L2: 1 pass\n" * 200)
    _manifest(
        tmp_path,
        [
            {
                "continuation": "v1",
                "run_dir": os.path.relpath(first["run_dir"], tmp_path),
                "launch_log": os.path.relpath(log, tmp_path),
            },
            {"continuation": "v2", "live": True, "run_dir": os.path.relpath(current["run_dir"], tmp_path)},
        ],
    )
    ctx = Context(
        experiment=current, reader=Reader(budget=4096), state={}, operator_private=False, now=NOW, shared={}, live=True
    )
    # The manifest itself fits; the log does not fit in one refresh.
    section = attempts.read(ctx)
    archived = section["attempts"][0]
    assert archived["reading"] and section["table"]["rows"][-1]["cells"][3] == "reading…"
    assert any("still being read" in n for n in section["notes"])
    for _ in range(4):
        ctx.reader.begin()
        section = attempts.read(ctx)
    archived = section["attempts"][0]
    assert not archived["reading"] and abs(archived["start"] - (NOW - 5000)) < 1
