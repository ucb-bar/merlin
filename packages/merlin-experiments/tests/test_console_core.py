"""Command console core: discovery states, the collector, diagnostics, the serve protocol and fallbacks."""

import io
import json
import shutil
import os
import time
from pathlib import Path

import pytest
from merlin_experiments.tracking.console import diagnose, discovery, launch, model, plain, serve


def _proc(root: Path, pid: int, argv: list[str], state: str = "S") -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (python) {state} 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def _run(task: Path, name: str, *, rounds=True) -> Path:
    run = task / "merlin" / "out" / "runs" / "tgt" / "phase1" / name
    (run / "rounds").mkdir(parents=True) if rounds else run.mkdir(parents=True)
    return run


def _owner(task: Path, launch_name: str, run: Path, pid: int, argv: list[str]) -> None:
    d = task / ".tmp" / launch_name / "process"
    d.mkdir(parents=True)
    (d / "owner.json").write_text(
        json.dumps(
            {
                "schema": discovery.OWNER_SCHEMA,
                "pid": pid,
                "argv": argv,
                "native_run_dir": str(run),
                "started_at_unix": time.time() - 100,
                "status": "native_runner_started",
                "selection": {"path": "/nowhere", "sha256": "0" * 64},
            }
        )
    )


@pytest.fixture
def machine(tmp_path):
    tasks, proc = tmp_path / "tasks", tmp_path / "proc"
    proc.mkdir()
    a = tasks / "task-a"
    live = _run(a, "exp-live")
    argv = ["/usr/bin/python3", "-B", "-m", "merlin_experiments.phase1", "--run-id", "exp-live"]
    _owner(a, "launch-live", live, 4242, ["/usr/bin/taskset", "-c", "1,2", *argv])
    _proc(proc, 4242, argv)
    dead = _run(a, "exp-dead")
    _owner(a, "launch-dead", dead, 999999, ["/usr/bin/python3"])
    done = _run(a, "exp-done")
    (done / "joint_campaign_summary.json").write_text(
        json.dumps({"completion_reason": "freeze_drain", "elapsed_wall_s": 60})
    )
    stopped = _run(a, "exp-stopped")
    (stopped / "joint_epoch_stop.json").write_text(json.dumps({"reason": "operator stop"}))
    old = _run(a, "exp-old")
    past = time.time() - 86400
    for path in (old, old / "rounds"):
        os.utime(path, (past, past))
    _run(a, "exp-runner")
    _proc(proc, 5151, ["/usr/bin/python3", "-m", "merlin_experiments.phase1", "--run-id", "exp-runner"])
    # A wrapper whose argv merely carries the runner command is not the runner.
    _proc(
        proc,
        5152,
        ["/usr/bin/python3", "reserve.py", "--", "python", "-m", "merlin_experiments.phase1", "--run-id", "exp-old"],
    )
    gone = tasks / "task-b" / "merlin" / "out" / "runs" / "tgt" / "phase1" / "exp-gone"
    _owner(tasks / "task-b", "launch-gone", gone, 999998, ["/usr/bin/python3"])
    study = _run(a, "exp-study", rounds=False)
    (study / "construction_study_result.json").write_text(json.dumps({"status": "completed"}))
    return tasks, proc


def test_discovery_states_carry_their_deciding_record(machine):
    tasks, proc = machine
    found = {e["run_id"]: e for e in discovery.Discovery(tasks, proc_root=proc).scan()}
    assert found["exp-live"]["state"] == "LIVE" and "4242" in found["exp-live"]["state_basis"]
    assert found["exp-runner"]["state"] == "LIVE" and found["exp-runner"]["runner_pid"] == 5151
    assert found["exp-dead"]["state"] == "INTERRUPTED" and "gone" in found["exp-dead"]["state_basis"]
    assert found["exp-done"]["state"] == "FINISHED"
    assert found["exp-stopped"]["state"] == "INTERRUPTED" and "operator stop" in found["exp-stopped"]["state_basis"]
    assert found["exp-old"]["state"] == "ENDED"
    assert found["exp-gone"]["state"] == "ENDED" and "no longer present" in found["exp-gone"]["state_basis"]
    assert found["exp-study"]["state"] == "FINISHED" and "completed" in found["exp-study"]["state_basis"]
    order = [e["state"] for e in discovery.Discovery(tasks, proc_root=proc).scan()]
    assert order == sorted(order, key=discovery.STATES.index)
    assert discovery.default_selection(list(found.values()))["state"] == "LIVE"


def test_stale_live_run_is_stalled(machine):
    tasks, proc = machine
    run = tasks / "task-a" / "merlin" / "out" / "runs" / "tgt" / "phase1" / "exp-live"
    past = time.time() - 7200
    os.utime(run / "rounds", (past, past))
    found = {e["run_id"]: e for e in discovery.Discovery(tasks, proc_root=proc, stale_s=600).scan()}
    assert found["exp-live"]["state"] == "STALLED"


def test_collector_never_crashes_on_empty_records_and_reports_reader_errors(machine, monkeypatch):
    tasks, proc = machine
    item = next(e for e in discovery.Discovery(tasks, proc_root=proc).scan() if e["run_id"] == "exp-done")
    snapshot = model.Collector(item, operator_private=True).refresh()
    assert snapshot["schema"] == "merlin.console.snapshot.v1"
    assert set(snapshot["sections"]) >= {"agents", "tokens", "progress", "capsules", "perf", "passes", "stages", "jobs"}
    json.dumps(snapshot, allow_nan=False, default=str)
    assert "golden" not in model.Collector(item, operator_private=False).refresh()["sections"]

    class Broken:
        @staticmethod
        def read(ctx):
            raise RuntimeError("boom")

    real = model._module
    monkeypatch.setattr(model, "_module", lambda name: Broken if name == "tokens" else real(name))
    snapshot = model.Collector(item).refresh()
    assert snapshot["sections"]["tokens"]["error"].startswith("RuntimeError")
    assert any(d["code"] == "READER_ERROR" and d["scope"] == "tokens" for d in snapshot["diagnostics"])


def _snap(**sections):
    return {"experiment": {"state": "LIVE"}, "generated_at": 10_000.0, "sections": sections}


def test_diagnostics_fire_only_on_recorded_numbers():
    assert diagnose.diagnose(_snap()) == []
    found = diagnose.diagnose(
        _snap(
            jobs={"sim": {"queue_wait_p50": 50, "runtime_p50": 10, "queued": 4}},
            resources={"mode": "live", "busy_cpus": 1.0, "total_cpus": 20, "queue": {"waiters": 2}},
            perf={
                "rows": [
                    {
                        "capsule": "c",
                        "efficiency_pct": 0.3,
                        "comparable": True,
                        "divergence_pct": 40,
                        "top_resource": {"name": "dma", "activity_pct": 80},
                    }
                ]
            },
            capsules={
                "capsules": [{"name": f"c{i}", "status": "fail", "failure": {"category": "X"}} for i in range(6)]
            },
            stages={"stages": [{"id": "lowering", "state": "blocked", "items": ["0/96 passing"]}]},
            tokens={
                "series": {"t": [0, 4000, 9000], "output": [0, 100_000, 400_000]},
                "budget": {"total": 40_000_000, "used_pct": 106.2},
                "rate_limit": {"used_percent": 85},
            },
            progress={
                "series": {"t": [0, 10], "n_passed": [1, 1]},
                "alerts": [{"code": "A", "severity": "warn", "title": "t", "detail": "d"}] * 3,
            },
        )
    )
    codes = {d["code"] for d in found}
    assert {
        "QUEUE_DOMINATED",
        "CPU_IDLE_WITH_BACKLOG",
        "LOW_ROOFLINE_EFFICIENCY",
        "MODEL_GAP",
        "FAILURE_CLUSTER",
        "STAGE_BLOCKED",
        "TOKENS_WITHOUT_PROGRESS",
        "TOKEN_BUDGET",
        "RATE_LIMIT_PRESSURE",
        "A",
    } <= codes
    assert next(d for d in found if d["code"] == "TOKEN_BUDGET")["severity"] == "error"
    # A soft target is advice, never an error: it is reported as info with what it is.
    for budget in ({"total": 40_000_000, "used_pct": 112.0, "mode": "soft", "enforced": False},
                   {"total": 40_000_000, "used_pct": 112.0, "mode": "soft"}):
        soft = next(d for d in diagnose.diagnose(_snap(tokens={"budget": budget})) if d["code"] == "TOKEN_BUDGET")
        assert soft["severity"] == "info" and "soft target exceeded (not enforced)" in soft["title"]
    enforced = {"total": 40_000_000, "used_pct": 95.0, "mode": "soft", "enforced": True}
    found_enforced = diagnose.diagnose(_snap(tokens={"budget": enforced}))
    assert next(d for d in found_enforced if d["code"] == "TOKEN_BUDGET")["severity"] == "warn"
    assert next(d for d in found if d["code"] == "A")["occurrences"] == 3
    severities = [diagnose.SEVERITY_ORDER[d["severity"]] for d in found]
    assert severities == sorted(severities)
    stalled = {"experiment": {"state": "STALLED", "state_basis": "quiet"}, "sections": {}}
    assert diagnose.diagnose(stalled)[0]["code"] == "RECORD_STALE"


def test_serve_sends_full_then_delta_snapshots_and_answers_commands(machine, monkeypatch):
    tasks, proc = machine
    monkeypatch.setattr(discovery, "runner_processes", lambda proc_root=None, limit=0: {})
    args = serve.parser().parse_args(["--tasks-root", str(tasks), "--experiment", "exp-done"])
    out = io.StringIO()
    server = serve.Server(args, out=out)
    server.experiments = server.discovery.scan()
    server.select("exp-done")
    server.snapshot("all")
    server.snapshot("all")
    server.inspect({"request_id": 7, "kind": "agents", "key": "nobody"})
    server.inspect({"request_id": 8, "kind": "nonsense", "key": "x"})
    messages = [json.loads(line) for line in out.getvalue().splitlines()]
    first, second, inspected, refused = messages
    assert first["type"] == "snapshot" and first["full"] and first["experiment"]["run_id"] == "exp-done"
    # Only live host telemetry (the machine's core map) changes between two reads of a finished run.
    assert second["full"] is False and set(second["sections"]) <= {"machine"}
    assert inspected["type"] == "inspect" and inspected["request_id"] == 7
    assert "not a console source" in refused["title"]


def test_plain_text_names_every_panel_without_escapes(machine):
    tasks, proc = machine
    item = next(e for e in discovery.Discovery(tasks, proc_root=proc).scan() if e["run_id"] == "exp-done")
    text = plain.render(model.Collector(item, operator_private=True).refresh(), color=False, width=150)
    for title in (
        "PROGRESS",
        "BOTTLENECKS",
        "AGENTS",
        "TOKENS",
        "CORES",
        "PERFORMANCE",
        "CAPSULES",
        "PASSES",
        "EVENTS",
    ):
        assert title in text
    assert "\x1b" not in text
    listing = plain.experiments_text(discovery.Discovery(tasks, proc_root=proc).scan(), color=False)
    assert "LIVE" in listing and "INTERRUPTED" in listing and "exp-done" in listing


def test_launch_prefers_the_environment_binary_and_passes_this_interpreter(tmp_path, monkeypatch):
    fake = tmp_path / "merlin-console"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MERLIN_CONSOLE_BIN", str(fake))
    assert launch.binary() == fake
    argv = launch.frontend_argv(fake, operator_private=True, tasks=[tmp_path], experiment="exp")
    assert argv[:3] == [str(fake), "--python", os.sys.executable]
    assert "--operator-private" in argv and argv[argv.index("--experiment") + 1] == "exp"


def test_console_cli_once_json_prints_a_snapshot(machine, capsys, monkeypatch):
    from merlin_experiments import cli

    tasks, proc = machine
    monkeypatch.setattr(discovery, "runner_processes", lambda proc_root=None, limit=0: {})
    assert cli.main(["console", "--tasks-root", str(tasks), "--experiment", "exp-done", "--once", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["experiment"]["run_id"] == "exp-done"
    assert cli.main(["console", "--tasks-root", str(tasks), "--list", "--json"]) == 0
    assert {e["run_id"] for e in json.loads(capsys.readouterr().out)} >= {"exp-done", "exp-dead"}


def test_serve_follows_a_relaunch_that_reuses_the_run_directory(machine, monkeypatch):
    tasks, proc = machine
    monkeypatch.setattr(discovery, "runner_processes", lambda proc_root=None, limit=0: {})
    args = serve.parser().parse_args(["--tasks-root", str(tasks), "--experiment", "exp-done"])
    server = serve.Server(args, out=io.StringIO())
    server.experiments = server.discovery.scan()
    server.select("exp-done")
    first = server.collector
    server.scan()
    assert server.collector is first  # same run: caches and history are kept
    run_dir = Path(first.experiment["run_dir"])
    run_dir.rename(run_dir.with_name(run_dir.name + ".archived"))
    shutil.copytree(run_dir.with_name(run_dir.name + ".archived"), run_dir)
    server.scan()
    assert server.collector is not first and server.sent == {}  # a new run at the same path starts fresh


def test_live_gsim_slots_are_counted_from_their_holders_own_lines(tmp_path):
    # The holder writes "pid starttime" into its locked slot file and clears it before releasing;
    # the console never reads the kernel lock table to find it.
    from merlin_experiments.tracking.console.sources import resources

    proc = tmp_path / "proc"
    fields = ["S", "1", *(["0"] * 17), "777", *(["0"] * 20)]
    (proc / "41").mkdir(parents=True)
    (proc / "41" / "stat").write_text(f"41 (holder) {' '.join(fields)}\n")
    slots = [tmp_path / f"slot_{index}.lock" for index in range(4)]
    for path, line in zip(slots, ("41 777\n", "41 778\n", "", "99 777\n"), strict=True):
        path.write_text(line)
    # Only the line naming pid 41 at its own birth counts; a reused pid, an empty and a dead holder do not.
    assert resources._held_slots(slots, proc_root=proc) == 1
