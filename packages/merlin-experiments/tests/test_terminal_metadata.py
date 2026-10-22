"""The operator view never upgrades incomplete or mismatched evidence."""

import copy
import json

from merlin_experiments.tracking.terminal_metadata import Metadata, capsule_metric


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def performance():
    pins = {key: key + "-pin" for key in (
        "capsule_sha256", "facts_sha256", "selection_sha256", "model_sha256",
        "comparison_domain_sha256", "scope_sha256")}
    return {"schema": "merlin.phase1.joint_performance.v1", **pins,
            "structural_comparison_binding": pins.copy(), "status": "PROVISIONAL",
            "complete_event_accounting": True, "estimated_whole_soc_cycles": 1000,
            "structural_roofline": {"structural_compute_floor_cycles": 200},
            "structural_floor_to_estimate_ratio": .2, "timing_qualified": False,
            "scope": "complete kernel call"}


def test_ratios_require_exact_native_domain_and_remain_provisional():
    perf = performance()
    row = {"capsule": "public_case", "performance": perf}
    result = capsule_metric(row, source_sha256="compiler")
    assert result["cycles"] == 1000
    assert result["roofline_efficiency_pct"] == 20
    assert result["roofline_gap_pct"] == 80
    assert not result["timing_qualified"] and not result["promotion_authority"]
    assert result["model_status"] == "PROVISIONAL"
    for bad in ("model_sha256", "scope_sha256", "comparison_domain_sha256"):
        changed = copy.deepcopy(row)
        changed["performance"][bad] = "other"
        metric = capsule_metric(changed, source_sha256="compiler")
        assert metric["cycles"] is None and metric["roofline_cycles"] is None
    assert capsule_metric(row)["cycles"] is None
    for key, value in (("schema", "foreign"), ("status", "REFUTED"),
                       ("complete_event_accounting", False), ("estimated_whole_soc_cycles", float("nan"))):
        changed = copy.deepcopy(row)
        changed["performance"][key] = value
        assert capsule_metric(changed, source_sha256="compiler")["cycles"] is None
    changed = copy.deepcopy(row)
    changed["performance"]["structural_floor_to_estimate_ratio"] = .3
    metric = capsule_metric(changed, source_sha256="compiler")
    assert metric["cycles"] == 1000 and metric["roofline_efficiency_pct"] is None


def test_rollouts_track_own_history_and_only_complete_lines(tmp_path):
    run, workspace = tmp_path / "run", tmp_path / "workspace"
    rollout = tmp_path / "helper.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "helper", "subagent_history_start_ordinal": 10,
         "source": {"subagent": {"thread_spawn": {"parent_thread_id": "lead", "agent_nickname": "fusion"}}}}},
        {"type": "session_meta", "payload": {"id": "lead"}},
        {"type": "event_msg", "ordinal": 5, "payload": {"type": "task_complete"}},
        {"type": "event_msg", "ordinal": 10, "payload": {"type": "task_started", "thread_id": "helper"}},
    ]
    rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
    partial = json.dumps({"type": "event_msg", "ordinal": 11, "payload": {"type": "task_complete"}})
    with rollout.open("a") as out:
        out.write(partial)
    reader = Metadata(tmp_path, operator_private=True)
    core = {"campaign": {"run_dir": str(run), "workspace": str(workspace), "rollout_paths": [str(rollout)]}}
    first = reader.sample(core)
    agent, = first["agents"]
    assert (agent["id"], agent["parent_id"], agent["name"], agent["status"]) == ("helper", "lead", "fusion", "working")
    assert agent["branch"] is None
    offset = first["sources"][0]["consumed_offset"]
    assert reader.sample(core)["sources"][0]["consumed_offset"] == offset
    with rollout.open("a") as out:
        out.write("\n")
    assert reader.sample(core)["agents"][0]["status"] == "completed"


def test_live_home_discovery_progress_and_native_attempt_joins(tmp_path):
    run, workspace, homes = tmp_path / "run", tmp_path / "workspace", tmp_path / "homes"
    write(run / "environment.yaml", {"codex_runtime_policy": {"home_root": str(homes)}})
    rollout = homes / "run_r0/sessions/2026/10/22/rollout-own.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": "lead", "git": {"branch": "experiment"}}}) + "\n")
    write(workspace / "qa/progress.json", {"schema": "merlin.phase1.grade_progress.v1", "n_completed": 1,
          "graded_source_sha256": "current", "per_capsule": [{"capsule": "case", "performance": performance()}]})
    write(workspace / ".qa_channel/resp_1.json", {"all_pass": False, "selfcheck_request_id": "one",
          "submission_sha256": "older", "requested_submission_sha256": "older", "graded_at_unix_ns": 123})
    core = {"campaign": {"run_dir": str(run), "workspace": str(workspace)}}
    sampled = Metadata(tmp_path, operator_private=True).sample(core)
    assert sampled["agents"][0]["branch"] == "experiment"
    assert sampled["coverage"]["n_completed"] == 1
    assert sampled["capsules"][0]["source_sha256"] == "current"
    attempt, = sampled["attempts"]
    assert attempt["source_sha256"] == "older" and attempt["status"] == "fail"
    assert attempt["hypothesis"] is None and attempt["pass_diff"] is None
    assert attempt["source"]["live"] and len(attempt["source"]["sha256"]) == 64
    redacted = Metadata(tmp_path).sample(core)
    assert redacted["agents"] == redacted["capsules"] == redacted["attempts"] == []
    assert redacted["coverage"] is None


def test_foreign_partial_and_oversized_records_are_unknown(tmp_path):
    workspace, run = tmp_path / "workspace", tmp_path / "run"
    write(workspace / "qa/progress.json", {"schema": "foreign", "n_completed": 100})
    write(workspace / ".qa_channel/resp_large.json", {"payload": "x" * 4096})
    write(workspace / ".qa_channel/resp_partial.json", {})
    (workspace / ".qa_channel/resp_partial.json").write_text('{"all_pass":')
    result = Metadata(tmp_path, operator_private=True, max_bytes=1024).sample(
        {"campaign": {"run_dir": str(run), "workspace": str(workspace), "rollout_paths": []}})
    assert result["coverage"] is None and result["attempts"] == []
    assert any("byte bound" in notice for notice in result["notices"])
    assert capsule_metric({"performance": [1], "numeric": "not an object"})["cycles"] is None
