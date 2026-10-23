"""Neutral asynchronous author routing; these fixtures establish no performance."""

import threading
import time
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import lean_authoring as L
from merlin_experiments.phase2.contracts import StageGateError


def test_author_executable_uses_worker_path_or_explicit_selection(tmp_path, monkeypatch):
    default = tmp_path / "codex"
    explicit = tmp_path / "selected-author"
    for path, marker in ((default, "worker-path"), (explicit, "explicit-selection")):
        path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{marker}'\n")
        path.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    for config, marker in (({}, "worker-path"), ({"codex_binary": str(explicit)}, "explicit-selection")):
        binary = L._codex_binary(config)
        assert binary.is_absolute()
        result = subprocess.run([str(binary)], capture_output=True, text=True, check=True, timeout=5)
        assert result.stdout.strip() == marker
    with pytest.raises(StageGateError, match="unavailable"):
        L._codex_binary({"codex_binary": str(tmp_path / "missing")})
    default.chmod(0o600)
    with pytest.raises(StageGateError, match="unavailable"):
        L._codex_binary({})


def count(cycles=100):
    return dict(status="observed", native_outputs_complete=True, feature_complete=True,
                complete_counts=True, timing_complete=True, unresolved_events=[], estimated_cycles=cycles,
                attainment_fraction=40 / cycles, model_sha256="a" * 64, scope_sha256="b" * 64)


def arm(cycles=100):
    return dict(status="numeric_pass", complete_outputs_checked=True, count_observation=count(cycles),
                native_result={"path": "/PRIVATE/output", "golden": [1234567]})


def test_complete_native_counts_project_only_unqualified_aggregate_information():
    result = L._public_arm(arm(), 40, "a" * 64)
    assert result["attainment_fraction"] == .4 and result["estimated_cycles"] == 100
    assert result["qualification"] == "UNQUALIFIED" and result["source_effects"] == "PENDING"
    assert result["promotion_authority"] == "NONE" and result["roofline_convergence"] is False
    assert "PRIVATE" not in str(result) and "1234567" not in str(result)


def test_safe_failure_and_nested_original_event_counts_reach_author():
    original = arm()
    original.update(failure_category="compiler_error", public_failure_summary="Unsupported public scalar operation")
    original["count_observation"]["count_observation"] = dict(feature_counts={"target.commands": 100},
                                                                report={"path": "PRIVATE"})
    result = L._public_arm(original, 40, "a" * 64)
    assert result["feature_counts"] == {"target.commands": 100}
    assert result["failure_category"] == "compiler_error"
    assert result["public_failure_summary"] == "Unsupported public scalar operation"
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("fault", ["numeric", "outputs", "features", "counts", "timing", "unresolved",
                                   "model", "scope", "underfloor", "attainment", "inconsistent"])
def test_partial_or_inconsistent_observations_cannot_become_attainment(fault):
    value = arm()
    c = value["count_observation"]
    if fault == "numeric":
        value["status"] = "failure"
    elif fault == "outputs":
        value["complete_outputs_checked"] = False
    elif fault in {"features", "counts", "timing"}:
        c[{"features": "feature_complete", "counts": "complete_counts", "timing": "timing_complete"}[fault]] = False
    elif fault == "unresolved":
        c["unresolved_events"] = ["unknown"]
    elif fault in {"model", "scope"}:
        c[fault + "_sha256"] = "wrong"
    elif fault == "underfloor":
        c.update(estimated_cycles=20, attainment_fraction=None)
    elif fault == "attainment":
        c["attainment_fraction"] = .9
    else:
        c["comparison_consistent"] = False
    assert L._public_arm(value, 40, "a" * 64)["attainment_fraction"] is None


@pytest.fixture
def workflow(tmp_path):
    now = datetime.now(UTC)
    config = dict(output=str(tmp_path), members=[dict(name=n, compute_floor_cycles=40) for n in ("x", "y")],
                  frozen_model_sha256="a" * 64, deadline_utc=(now + timedelta(minutes=10)).isoformat(),
                  admission_cutoff_utc=(now + timedelta(minutes=5)).isoformat(), target_fraction=.3,
                  beam_width=1, max_jobs=4)
    revisions = {name: SimpleNamespace(revision_id=name, source_sha256=name, source=tmp_path,
                                       verify=lambda: None, public_record=lambda name=name: dict(revision_id=name))
                 for name in ("base", "one", "two")}
    store = SimpleNamespace(get=lambda name: revisions[name])
    gate = threading.Event()

    def evaluate(**kwargs):
        assert gate.wait(2)
        return dict(members=[dict(id=m["name"], baseline=arm(200), candidate=arm(100))
                             for m in kwargs["selected_members"]])

    owner = L.LeanWorkflow(config, candidate=tmp_path, target=object(), receipt=tmp_path / "receipts",
                           store=store, baseline=revisions["base"], evaluate_candidate=evaluate)
    # The executing-revisions owner is separately tested with real retained sources.
    owner._evaluate = lambda revision, selected, root, timeout: evaluate(
        selected_members=tuple(owner.members[n] for n in selected))
    yield owner, revisions, gate
    gate.set()
    owner.close()


def execute(owner, name, **bindings):
    return owner.execute({}, name, bindings, 0, 10, time.monotonic())[1]


def collect(owner):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = owner.status()
        if all(j["state"] != "running" for j in status["jobs"]):
            return status
        time.sleep(.005)
    pytest.fail("neutral asynchronous job did not complete")


def test_asynchronous_selected_jobs_do_not_block_author_and_respect_beam(workflow):
    owner, revisions, gate = workflow
    owner.latest = revisions["one"]
    job = execute(owner, "evaluate-selected", revision_id="one", members="x")
    assert job["state"] == "running" and owner.status()["target_revision"] is None
    assert execute(owner, "evaluate-selected", revision_id="two", members="y")["status"] == "BUSY"
    gate.set()
    status = collect(owner)
    assert status["jobs"][0]["state"] == "completed"
    assert status["priorities"][0]["id"] == "x" and status["priorities"][1]["status"] == "UNKNOWN"
    assert status["stop_reason"] is None


def test_target_requires_every_member_under_same_retained_revision(workflow):
    owner, revisions, gate = workflow
    gate.set()
    execute(owner, "evaluate-selected", revision_id="one", members="x")
    collect(owner)
    execute(owner, "evaluate-selected", revision_id="two", members="y")
    assert collect(owner)["target_revision"] is None
    execute(owner, "evaluate-selected", revision_id="one", members="y")
    status = collect(owner)
    assert status["target_revision"] == "one"
    assert status["stop_reason"].startswith("estimated_structural_target_met")
    assert status["roofline_convergence"] is False


def test_current_candidate_priorities_do_not_mix_in_slower_baseline_estimates(workflow):
    owner, revisions, gate = workflow
    owner.config["target_fraction"] = .8
    gate.set()
    execute(owner, "evaluate-selected", revision_id="base", members="x,y")
    collect(owner)
    owner.latest = revisions["one"]
    execute(owner, "evaluate-selected", revision_id="one", members="x")
    priorities = {row["id"]: row for row in collect(owner)["priorities"]}
    assert priorities["x"]["attainment_fraction"] == .4
    assert priorities["x"]["observation_origin"] == "current_candidate"
    assert priorities["y"]["attainment_fraction"] == .2
    assert priorities["y"]["observation_origin"] == "original_control"


@pytest.mark.parametrize("failed", [("x", "y"), ("y",)])
def test_automatic_fallback_never_selects_a_numerically_failed_revision(workflow, failed):
    owner, revisions, _ = workflow
    owner._evaluate = lambda *_: dict(members=[dict(id=name, baseline=arm(),
        candidate=dict(status="numeric_failure", complete_outputs_checked=False)
        if name in failed else arm()) for name in ("x", "y")])
    execute(owner, "evaluate-selected", revision_id="one", members="x,y")
    collect(owner)
    selected, reason = owner.selection()
    assert selected is revisions["base"]
    assert reason == "no completed candidate evidence; original control retained"


def test_finish_cannot_end_open_exploration_and_unknown_members_are_refused(workflow):
    owner, *_ = workflow
    with pytest.raises(StageGateError, match="remains open"):
        execute(owner, "finish", revision_id="one")
    with pytest.raises(StageGateError, match="public member"):
        execute(owner, "evaluate-selected", revision_id="one", members="private")


def test_original_deadline_stops_even_without_a_target(workflow):
    owner, *_ = workflow
    owner.config["deadline_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    assert owner.status()["stop_reason"] == "original campaign deadline reached"


def test_drain_collects_jobs_and_selects_exact_retained_evidence(workflow):
    owner, revisions, gate = workflow
    owner.latest = revisions["two"]
    execute(owner, "evaluate-selected", revision_id="one", members="x,y")
    gate.set()
    owner.close()
    owner.close()  # Existing broker and outer finally can both close the lifetime.
    selected, reason = owner.selection()
    assert selected is revisions["one"] and reason == "estimated_structural_target_met"
    assert owner.status()["jobs"][0]["state"] == "completed"


def test_target_can_use_complete_candidate_estimate_when_baseline_counts_are_unknown(workflow):
    owner, _, _ = workflow
    owner._evaluate = lambda *_: dict(members=[dict(id=n, baseline=dict(status="numeric_pass",
        complete_outputs_checked=True, count_observation=dict(status="UNKNOWN")), candidate=arm()) for n in ("x", "y")])
    execute(owner, "evaluate-selected", revision_id="one", members="x,y")
    status = collect(owner)
    assert status["target_revision"] == "one"
    assert all(r["baseline"]["attainment_fraction"] is None for r in status["jobs"][0]["result"]["members"])


def test_public_source_view_contains_ir_only_and_detects_later_changes(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    snapshot = L._source_view([dict(name="x", public_source_interface=str(source),
                                    private_golden=[999], capsule_dir="PRIVATE")], tmp_path / "public")
    L.AW.verify_answer_free_agent_inputs(snapshot)
    assert "PRIVATE" not in snapshot.manifest_path.read_text()
    output = snapshot.root / "x.mlir"
    output.chmod(0o644)
    output.write_text("changed")
    with pytest.raises(StageGateError, match="changed"):
        L.AW.verify_answer_free_agent_inputs(snapshot)
