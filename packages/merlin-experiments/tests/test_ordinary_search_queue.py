"""Normal admission and queue/thread wiring with synthetic engine observations."""

import os
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import evaluation_pool as EP
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator
from merlin_experiments.phase2.search_inputs import ORDINARY_KIND
from test_compiler_libraries import selected as select_library
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs
from test_ordinary_search_policy import revisions

from merlin.targetgen.compiler_library import CompilerLibraryError, active_compiler_library, scoped_compiler_library


def test_parallel_members_keep_fixed_additional_library_selection(ordinary_inputs, tmp_path, monkeypatch):
    selection, _ = select_library(tmp_path / "additional")
    execute = PME.run_execution
    seen = []

    def observe(*args, **kwargs):
        assert active_compiler_library() is selection
        seen.append(args[0].arm)
        return execute(*args, **kwargs)

    monkeypatch.setattr(PME, "run_execution", observe)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments, compiler_library_selection=selection)
    assert policy.binding()["additional_compiler_library"] == selection.record()
    candidate, baseline = revisions(policy, tmp_path)
    monkeypatch.setenv("MERLIN_PERF_SWEEP_WORKERS", "2")
    feedback = policy.evaluate_retained(
        candidate, baseline, work_root=tmp_path / "parallel-library", call_index=0, timeout_s=20, workers=2
    )
    assert feedback["summary"]["all_correct"]
    assert set(seen) == {"baseline", "candidate"}
    assert active_compiler_library() is None
    (selection.root / "merlin/portable.py").write_text("changed library bytes")
    ordinary_inputs.executions.clear()
    with pytest.raises(CompilerLibraryError, match="changed|digest"):
        policy.evaluate_retained(
            candidate, baseline, work_root=tmp_path / "changed-library", call_index=1, timeout_s=20
        )
    assert not ordinary_inputs.executions


def test_policy_without_library_cannot_inherit_unbound_ambient_selection(ordinary_inputs, tmp_path):
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    assert "additional_compiler_library" not in policy.binding()
    candidate, baseline = revisions(policy, tmp_path)
    selection, _ = select_library(tmp_path / "additional")
    with scoped_compiler_library(selection):
        with pytest.raises(StageGateError, match="unbound"):
            policy.evaluate_retained(candidate, baseline, work_root=tmp_path / "unbound", call_index=0, timeout_s=10)
    assert not ordinary_inputs.executions


def test_parallel_normal_members_keep_both_retained_compiler_contexts(ordinary_inputs, tmp_path, monkeypatch):
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    monkeypatch.setenv("MERLIN_PERF_SWEEP_WORKERS", "2")
    (policy.candidate / "compiler.py").write_text("unfinished author edit: [\n")
    feedback = policy.evaluate_retained(
        candidate, baseline, work_root=tmp_path / "parallel", call_index=0, timeout_s=20, workers=2
    )
    assert feedback["summary"]["all_correct"]
    assert {spec.arm for spec, *_ in ordinary_inputs.executions} == {"baseline", "candidate"}
    assert all(
        issued is (baseline if spec.arm == "baseline" else candidate)
        for spec, _, _, issued in ordinary_inputs.executions
    )


def test_normal_queue_uses_original_feedback_and_reaps_before_selection(ordinary_inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two allocated CPUs required for harmless queue routing")
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    store = RetainedRevisionStore(
        tmp_path / "sources",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    pool = policy.enable_evaluation_pool(
        revisions=store,
        resources=AssignedResources(cpus, 8 << 30, 16 << 20, 2, author_cores=0),
        job_resources=JobResources(2, 2 << 30, 1 << 20, 2),
        root=tmp_path / "queue",
    )
    published = []

    def publish(root):
        issued = next(row for row in pool._jobs.values() if row.root == root)
        assert issued.state == "completed" and issued.result is not None
        assert issued.guardian_fd is None and issued.pins == []
        assert (root / "cleanup.json").is_file() and (root / "feedback.json").is_file()
        with pytest.raises(ChildProcessError):
            os.waitpid(issued.guardian_pid, os.WNOHANG)
        published.append(root)

    # Observe the original completion seam only; this spy grants no replay role.
    monkeypatch.setattr(policy, "publish_completed_artifacts", publish)
    try:
        policy.search_coordinator = ComponentSearchCoordinator(pool)
        revision = store.freeze(policy.candidate)
        job = pool.submit(revision.revision_id, kind=ORDINARY_KIND, timeout_s=30)
        (policy.candidate / "compiler.py").write_text("def schedule():\n    return 99\n")
        deadline = time.monotonic() + 35
        while not job["terminal"] and time.monotonic() < deadline:
            time.sleep(0.02)
            job = pool.status(job["job_id"])
        assert job["state"] == "completed", job
        assert job["kind"] == ORDINARY_KIND
        assert job["result"]["kind"] == "host_owned_tuning_gsim_feedback"
        assert job["result"]["candidate_sha256"] == revision.source_sha256
        assert published == [pool.root / job["job_id"]]
        policy.search_coordinator.finalize(timeout_s=10)
        selected = policy.search_coordinator.select()
        assert selected["selected"]["source_sha256"] == revision.source_sha256
        assert (policy.candidate / "compiler.py").read_text() == "def schedule():\n    return 1\n"
        assert "tuning-gsim-feedback" in {action.name for action in policy.build_registry() if action.required}
    finally:
        pool.close()


@pytest.mark.parametrize("disposition", ["cleanup_unknown", "refused", "cancelled", "storage_failure"])
def test_unsuccessful_finish_cannot_publish_artifact_archive(disposition, tmp_path, monkeypatch):
    # Isolate the original parent branch; synthetic cleanup packets issue no
    # process, kernel or cache authority. The positive route above reaps a real
    # harmless child through the original guardian.
    published, discarded = [], []
    pool = EP.ComponentEvaluationPool.__new__(EP.ComponentEvaluationPool)
    pool.inputs = SimpleNamespace(ordinary=True)
    pool.policy = SimpleNamespace(
        publish_completed_artifacts=published.append, discard_cleaned_artifacts=discarded.append
    )
    pool._save = lambda job: None

    def release(job, **kwargs):
        if disposition == "storage_failure":
            raise OSError("injected bounded backing exhaustion")

    pool._release = release
    reader, writer = os.pipe()
    job = SimpleNamespace(
        root=tmp_path,
        state="running",
        refusal=None,
        result=None,
        guardian_pid=123,
        guardian_fd=reader,
        control=SimpleNamespace(close=lambda: None),
        cancel_requested=disposition == "cancelled",
    )
    monkeypatch.setattr(EP, "_close_guardian", lambda *args: ({}, True, 0))
    monkeypatch.setattr(EP, "_complete_cleanup", lambda *args: disposition != "cleanup_unknown")
    try:
        pool._finish(job, error=RuntimeError("injected incomplete evaluation"))
        assert published == []
        assert discarded == ([tmp_path] if disposition in {"refused", "cancelled"} else [])
        assert job.state == (
            "cleanup_unknown" if disposition in {"cleanup_unknown", "storage_failure"} else disposition
        )
    finally:
        os.close(writer)
        if job.guardian_fd is not None:
            os.close(reader)
