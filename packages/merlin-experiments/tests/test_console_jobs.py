"""Console resources, jobs and stages read fixture records only; no process, agent or target runs."""

import hashlib
import json

import pytest
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import jobs, resources, stages
from merlin_experiments.tracking.terminal_probe import Sampler


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def ctx_for(experiment, *, private=True, live=False, shared=None, now=2_000_000_000.0, state=None):
    return Context(
        experiment=experiment,
        reader=Reader(),
        state={} if state is None else state,
        operator_private=private,
        now=now,
        shared=shared or {},
        live=live,
    )


def stat(pid, ticks=10, last_cpu=0, ppid=1, start=10):
    fields = ["0"] * 40
    fields[0], fields[1], fields[11], fields[19], fields[36] = "R", str(ppid), str(ticks), str(start), str(last_cpu)
    return f"{pid} (fixture) " + " ".join(fields)


def timeline(capsule, start_ns, *, tier=None, running=False):
    stages_ = [
        {
            "stage": "capsule_execution",
            "stage_id": 0,
            "parent_stage_id": None,
            "started_ns": start_ns,
            "finished_ns": None if running else start_ns + 20_000_000_000,
            "elapsed_s": None if running else 20.0,
            "status": "running" if running else "returned",
        },
        {
            "stage": "emission",
            "stage_id": 1,
            "parent_stage_id": 0,
            "started_ns": start_ns,
            "finished_ns": start_ns + 10_000_000_000,
            "elapsed_s": 10.0,
            "status": "returned",
        },
        {
            "stage": "oracle_adapter.L2",
            "stage_id": 2,
            "parent_stage_id": 0,
            "started_ns": start_ns + 10_000_000_000,
            "finished_ns": start_ns + 15_000_000_000,
            "elapsed_s": 5.0,
            "status": "returned",
        },
        {
            "stage": "package_policy",
            "stage_id": 3,
            "parent_stage_id": None,
            "started_ns": start_ns,
            "finished_ns": start_ns + 1_000_000,
            "elapsed_s": 0.001,
            "status": "returned",
        },
    ]
    return {
        "schema": "merlin.compiler_setup_timing.v1",
        "capsule": capsule,
        "tier": tier,
        "operation": "native_capsule",
        "stages": stages_,
    }


def jobs_fixture(tmp_path):
    run, ws = tmp_path / "run", tmp_path / "ws"
    base = 1_900_000_000_000_000_000
    write(run / "_qa_work/profile_package_execution/setup-timing/a.json", timeline("CAP_A", base))
    write(
        run / "_qa_work/fruns_2/host-stage-timing/setup-timing/b.json",
        timeline("CAP_B", base + 30 * 10**9, running=True),
    )
    channel = ws / ".qa_channel"
    write(
        channel / "simreq_j1.json", {"sim": "gsim", "tiers": "L3", "capsules": "CAP_A", "submitted_at": 1_900_000_100.0}
    )
    write(
        channel / "simtiming_j1.json",
        {
            "schema": "simjob_resource_timing_v1",
            "queue_wait_s": 2.0,
            "worker_runtime_s": 40.0,
            "started_at_unix": 1_900_000_102.0,
            "finished_at_unix": 1_900_000_142.0,
            "returncode": 1,
        },
    )
    write(channel / "simdone_j1", "ok")
    write(channel / "simresp_j1.json", {"all_pass": True})
    write(
        channel / "simreq_j2.json",
        {"sim": "spike", "tiers": "L2", "capsules": "CAP_B", "submitted_at": 1_900_000_200.0},
    )
    write(channel / "simrun_j2", "running")
    write(
        channel / "simreq_j3.json", {"sim": "gsim", "tiers": "L3", "capsules": "CAP_C", "submitted_at": 1_900_000_300.0}
    )
    write(channel / "simerr_j3", "boom")
    return {"run_dir": str(run), "workspace": str(ws), "state": "LIVE"}


def test_lane_rules_are_first_token_match_with_tier_fallback():
    assert jobs.lane_of("oracle_adapter.L2") == "L2"
    assert jobs.lane_of("oracle_simulation_readback.spike") == "L2"
    assert jobs.lane_of("scheduler_submission_wait") == "queue"
    assert jobs.lane_of("emission") == "build"
    assert jobs.lane_of("package_policy.apply_masks") == "verify"
    assert jobs.lane_of("capsule_execution", "L3") == "L3"
    assert jobs.lane_of("result_processing") == "setup"
    assert jobs.percentile([], 0.5) is None
    assert jobs.percentile([1, 2, 3, 4], 0.5) == 3


def test_jobs_draw_leaf_spans_stage_stats_and_broker_verdicts(tmp_path):
    experiment = jobs_fixture(tmp_path)
    section = jobs.read(ctx_for(experiment, live=True))
    labels = {(s["lane"], s["label"]) for s in section["spans"]}
    assert ("build", "emission") in labels and ("L2", "oracle_adapter.L2") in labels
    # Parents are not drawn over their children; sub-threshold stages are not drawn at all.
    assert not any(s["label"] in ("capsule_execution", "package_policy") for s in section["spans"])
    stats = {r["stage"]: r for r in section["stages"]}
    assert stats["emission"]["count"] == 2 and stats["emission"]["p50_s"] == 10.0
    assert stats["capsule_execution"]["running"] == 1
    assert [r["capsule"] for r in section["running"] if r["stage"] == "capsule_execution"] == ["CAP_B"]
    sim = section["sim"]
    # A worker returncode of 1 is not a verdict: the broker marker and response decide.
    assert (sim["requests"], sim["running"], sim["failed"], sim["errors"]) == (3, 1, 0, 1)
    assert sim["queue_wait_p50"] == 2.0 and sim["runtime_p50"] == 40.0
    l3 = [s for s in section["spans"] if s["lane"] == "L3"]
    assert l3 and l3[0]["status"] == "pass" and l3[0]["capsule"] == "CAP_A"
    assert any(s["lane"] == "queue" and s["end"] == 1_900_000_102.0 for s in section["spans"])


def test_jobs_public_mode_hides_capsules_and_inspect_profiles_stages(tmp_path):
    experiment = jobs_fixture(tmp_path)
    section = jobs.read(ctx_for(experiment, private=False))
    assert all(s["capsule"] is None for s in section["spans"])
    assert "no longer live" in " ".join(section["notes"])  # ctx.live, not the row text, decides staleness
    detail = jobs.inspect(ctx_for(experiment), "emission")
    assert detail["sections"][0]["lines"][0].startswith("occurrences 2")
    assert jobs.inspect(ctx_for(experiment, private=False), "CAP_A")["notice"]
    capsule = jobs.inspect(ctx_for(experiment), "CAP_A")
    assert any("emission" in line for line in capsule["sections"][0]["lines"])


def test_jobs_index_incrementally_within_a_parse_budget(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for i in range(5):
        write(
            run / f"_qa_work/profile_package_execution/setup-timing/{i}.json",
            timeline(f"C{i}", 1_900_000_000_000_000_000 + i * 10**9),
        )
    monkeypatch.setattr(jobs, "PARSES_PER_REFRESH", 2)
    state, experiment = {}, {"run_dir": str(run), "workspace": None}
    counts = [jobs.read(ctx_for(experiment, state=state))["records"] for _ in range(4)]
    assert counts == [2, 4, 5, 5]


def test_resources_recorded_series_from_resource_samples(tmp_path):
    run = tmp_path / "run"
    rows = [
        {
            "sampled_at": f"2026-10-23T19:00:0{i}+00:00",
            "user_cpu_s": 2.0 * i,
            "system_cpu_s": 0.0,
            "rss_bytes": 100 + i,
            "host_load_1m": 1.5,
            "processes": 3,
            "threads": 9,
        }
        for i in range(3)
    ]
    write(run / "rounds/round_00.turn00.resource_samples.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    section = resources.read(ctx_for({"run_dir": str(run), "state": "FINISHED"}))
    assert section["mode"] == "recorded" and section["cores"] == []
    assert section["series"]["cpu_pct"] == [None, 200.0, 200.0]
    assert section["process"]["rss_bytes"] == 102
    assert any("not simulation" in note for note in section["notes"])
    empty = resources.read(ctx_for({"run_dir": str(tmp_path / "none"), "state": "ENDED"}))
    assert empty["mode"] == "none" and empty["busy_cpus"] is None


def test_tree_sampler_follows_descendants_and_cpu_deltas(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    write(proc / "100/stat", stat(100))
    write(proc / "100/cmdline", "python\0-m\0merlin_experiments.phase1\0")
    write(proc / "100/status", "Cpus_allowed_list:\t0-1\n")
    write(proc / "100/task/100/stat", stat(100, 10, last_cpu=1))
    write(proc / "101/stat", stat(101, ppid=100))
    write(proc / "101/cmdline", "spike\0prog.elf\0")
    write(proc / "101/task/101/stat", stat(101, 10, last_cpu=1, ppid=100))
    write(proc / "999/stat", stat(999))  # unrelated process: never sampled
    write(proc / "999/task/999/stat", stat(999))
    write(proc / "stat", "cpu0 100 0 0 100 0 0 0 0\ncpu1 0 0 0 200 0 0 0 0\n")
    clock = iter([10.0, 11.0])
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(clock))
    sampler = resources.TreeSampler(100, proc_root=proc)
    assert sorted(sampler.descendants()) == [100, 101]
    first = sampler.sample()
    assert [c["measured_utilization_pct"] for c in first["cores"]] == [None, None]
    write(proc / "stat", "cpu0 100 0 0 200 0 0 0 0\ncpu1 100 0 0 200 0 0 0 0\n")
    write(proc / "101/task/101/stat", stat(101, 110, last_cpu=1, ppid=100))
    second = sampler.sample()
    assert [c["measured_utilization_pct"] for c in second["cores"]] == [0.0, 100.0]
    rows = resources._core_rows(second["cores"], True)
    assert rows[1]["kind"] == "L2" and rows[0]["kind"] == "idle"


def test_sampler_owner_path_binds_one_launch(tmp_path):
    task = tmp_path / "task"
    for name, started in (("old", 1), ("new", 2)):
        raw = json.dumps({"run_id": name, "run_dir": str(tmp_path / name)})
        write(tmp_path / f"{name}.json", raw)
        write(
            task / f".tmp/{name}/process/owner.json",
            {
                "pid": 1,
                "started_at_unix": started,
                "argv": ["x"],
                "selection": {
                    "path": str(tmp_path / f"{name}.json"),
                    "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                },
            },
        )
    proc = tmp_path / "proc"
    proc.mkdir()
    assert Sampler(task, proc_root=proc).selected()[1]["run_id"] == "new"
    bound = Sampler(task, owner_path=task / ".tmp/old/process/owner.json", proc_root=proc)
    assert bound.selected()[1]["run_id"] == "old"


def contract_or_skip(tmp_path):
    doc, path = stages.contract(ctx_for({"run_dir": str(tmp_path)}))
    if doc is None:
        pytest.skip("contract directory not resolvable in this environment")
    return doc


def test_stage_contract_maps_every_boundary_and_token_exactly_once(tmp_path):
    doc = contract_or_skip(tmp_path)
    entries = stages._entries(doc)
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    assert ids[:8] == [
        "frontend",
        "global_optimization",
        "interface",
        "lowering",
        "isa_dialect",
        "oot_integration",
        "transformations",
        "tuning",
    ]
    boundaries = [b for e in entries for b in e["construction_boundaries"]]
    assert sorted(boundaries) == sorted(
        ["graph", "numeric_layout", "schedule_storage", "isa_lowering", "host_runtime", "integration"]
    )
    tokens = [t for e in entries for t in e["pass_requirement_tokens"]]
    assert len(tokens) == len(set(tokens))
    assert {
        "region-partition",
        "tile-schedule",
        "block-movement",
        "target-isa-lowering",
        "host-seam",
        "model-boundary-capstone",
    } <= set(tokens)
    assert doc["ledger_schema"] == "merlin.compiler_stage_ledger.v1"


def test_stages_derive_from_actions_and_capsules_then_prefer_a_ledger(tmp_path):
    contract_or_skip(tmp_path)
    run, ws = tmp_path / "run", tmp_path / "ws"
    events = [
        {"action_id": "src", "boundary": "graph", "capability": "fusion", "state": "running", "time_ns": 1_000_000_000},
        {
            "action_id": "src",
            "boundary": "graph",
            "capability": "fusion",
            "state": "completed",
            "returncode": 0,
            "time_ns": 3_000_000_000,
        },
        {
            "action_id": "isa",
            "boundary": "isa_lowering",
            "capability": "isa",
            "state": "running",
            "time_ns": 2_000_000_000,
        },
        {
            "action_id": "int",
            "boundary": "integration",
            "capability": "x",
            "state": "failed",
            "returncode": 1,
            "time_ns": 2_000_000_000,
        },
    ]
    write(ws / "construction_study/events.jsonl", "".join(json.dumps(e) + "\n" for e in events))
    capsules = {
        "capsules": [
            {"name": "a", "pass_requirements": ["tile-schedule"], "status": "pass"},
            {"name": "b", "pass_requirements": ["tile-schedule", "target-isa-lowering"], "status": "fail"},
        ]
    }
    experiment = {"run_dir": str(run), "workspace": str(ws), "state": "FINISHED"}
    section = stages.read(ctx_for(experiment, shared={"capsules": capsules}))
    by_id = {s["id"]: s for s in section["stages"]}
    assert by_id["global_optimization"]["state"] == "done"
    assert by_id["global_optimization"]["wall_s"] == 2.0
    assert by_id["isa_dialect"]["state"] == "active"
    assert by_id["lowering"]["state"] == "blocked"  # its only action failed
    assert by_id["oot_integration"]["capsules"] == {"required": 2, "passing": 1}
    assert by_id["oot_integration"]["state"] == "pending"
    assert by_id["tuning"]["state"] == "unknown" and by_id["frontend"]["basis"] == "derived"
    write(
        ws / "compiler_stage_ledger.json",
        {
            "schema": "merlin.compiler_stage_ledger.v1",
            "stages": [
                {
                    "stage": "frontend",
                    "state": "done",
                    "started_ns": 1_000_000_000,
                    "finished_ns": 4_000_000_000,
                    "metrics": {"parsed_capsules": 7},
                }
            ],
        },
    )
    ledger = stages.read(ctx_for(experiment, shared={"capsules": capsules}))
    by_id = {s["id"]: s for s in ledger["stages"]}
    assert by_id["frontend"]["basis"] == "ledger" and by_id["frontend"]["state"] == "done"
    assert by_id["frontend"]["metrics"] == {"parsed_capsules": 7} and by_id["frontend"]["wall_s"] == 3.0
    assert by_id["tuning"]["state"] == "unknown"
    detail = stages.inspect(ctx_for(experiment), "frontend")
    assert detail["title"] == "stage Frontend"
