"""Normal admission and queue/thread wiring with synthetic engine observations."""

import os
import time

import pytest
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator
from merlin_experiments.phase2.search_inputs import ORDINARY_KIND
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs
from test_ordinary_search_policy import revisions


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
    try:
        policy.search_coordinator = ComponentSearchCoordinator(pool)
        revision = store.freeze(policy.candidate)
        job = pool.submit(revision.revision_id, timeout_s=30)
        (policy.candidate / "compiler.py").write_text("def schedule():\n    return 99\n")
        deadline = time.monotonic() + 35
        while not job["terminal"] and time.monotonic() < deadline:
            time.sleep(0.02)
            job = pool.status(job["job_id"])
        assert job["state"] == "completed", job
        assert job["kind"] == ORDINARY_KIND
        assert job["result"]["kind"] == "host_owned_tuning_gsim_feedback"
        assert job["result"]["candidate_sha256"] == revision.source_sha256
        policy.search_coordinator.finalize(timeout_s=10)
        selected = policy.search_coordinator.select()
        assert selected["selected"]["source_sha256"] == revision.source_sha256
        assert (policy.candidate / "compiler.py").read_text() == "def schedule():\n    return 1\n"
        assert "tuning-gsim-feedback" in {action.name for action in policy.build_registry() if action.required}
    finally:
        pool.close()
