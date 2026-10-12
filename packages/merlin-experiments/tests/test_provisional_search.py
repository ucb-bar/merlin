"""Normal exploration and parent restoration, with original harmless fake tools."""

import os
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import ordinary_queue_calibration as CALIBRATION
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import provisional_search as PROV
from merlin_experiments.phase2 import queued_search_actions as QSA
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from merlin_experiments.phase2.search_inputs import ORDINARY_FAST_KIND, SearchInputs
from test_ordinary_fast_feedback import cheap_inputs as cheap_inputs
from test_ordinary_fast_feedback import extend_corpus
from test_ordinary_fast_feedback import ordinary_inputs as ordinary_inputs


def test_unknown_cheap_evidence_is_provisional_and_never_promotes(cheap_inputs, tmp_path):
    inputs, _ = cheap_inputs
    policy = OS.prepare_ordinary_search_policy(**inputs.arguments)
    store = RetainedRevisionStore(
        tmp_path / "revisions",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    baseline = store.freeze(policy.search_baseline)
    revision = store.freeze(
        policy.candidate,
        parent_revision=baseline.revision_id,
        options={"hypothesis": "own schedule test", "edit_family": "schedule"},
    )
    feedback = policy.evaluate_retained(
        revision,
        baseline,
        kind=ORDINARY_FAST_KIND,
        work_root=tmp_path / "cheap",
        call_index=0,
        timeout_s=20,
    )
    job = {
        "state": "completed",
        "kind": ORDINARY_FAST_KIND,
        "revision_id": revision.revision_id,
        "job_id": "own-job",
        "result": feedback,
        "source_sha256": revision.source_sha256,
    }
    coordinator = SimpleNamespace(
        pool=SimpleNamespace(policy=policy, revisions=store, inputs=SearchInputs(policy)),
        roster=tuple(sorted(policy.validate_search_inputs())),
        settings=SC.SearchSettings(),
    )
    projection = PROV.project(coordinator, [job])
    assert projection["ranking_authority"] == "NONE"
    candidate = projection["candidates"][0]
    assert candidate["parent_revision"] == baseline.revision_id
    assert candidate["conservative_model_saved_cycles"] is None
    assert candidate["emission_status"] == "UNKNOWN"
    assert candidate["promotion"] == "NONE"
    measured = SC.rank_measured_contenders(
        [job],
        roster=coordinator.roster,
        corpus_sha256=policy.search_corpus.capsules_sha256,
        configuration_sha256=policy.feedback_evaluator.certificate.sha256,
        timing_kind="corpus-rtl-feedback",
    )
    assert measured["ranked"] == []
    failed = {**job, "state": "refused", "refusal": "incomplete tool observation", "result": None}
    assert PROV.project(coordinator, [failed])["history"][0]["outcome"] == "unmeasured"


def test_original_normal_parent_action_restores_bytes_without_promoting(cheap_inputs, tmp_path, monkeypatch):
    inputs, _ = cheap_inputs
    extend_corpus(inputs, tmp_path)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    policy = OS.prepare_ordinary_search_policy(**inputs.arguments)
    store = RetainedRevisionStore(
        tmp_path / "parents",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two CPUs needed to prepare the original harmless pool")
    pool = policy.enable_evaluation_pool(
        revisions=store,
        resources=AssignedResources(cpus, 8 << 30, 16 << 20, 1, author_cores=1),
        job_resources=JobResources(1, 1 << 30, 1 << 20),
        root=tmp_path / "queue",
    )
    try:
        coordinator = policy.search_coordinator = SC.ComponentSearchCoordinator(pool)
        original = (policy.candidate / "compiler.py").read_bytes()
        parent = store.freeze(policy.candidate, parent_revision=pool.baseline_revision.revision_id)
        (policy.candidate / "compiler.py").write_text("def schedule():\n    return 77\n")
        assert PROV.RESUME_ACTION in {row.name for row in policy.build_registry()}
        assert CALIBRATION.ACTION in {row.name for row in policy.build_registry()}
        closure = policy.search_closure_sha256
        with pytest.raises(StageGateError, match="lacks complete"):
            QSA.execute(
                policy,
                CALIBRATION.ACTION,
                {"revision_ids": parent.revision_id + "," + pool.baseline_revision.revision_id},
                20,
                time.monotonic(),
            )
        assert policy.search_closure_sha256 == closure
        assert coordinator.calibration is None
        transport, receipt = QSA.execute(
            policy, PROV.RESUME_ACTION, {"revision_id": parent.revision_id}, 20, time.monotonic()
        )
        import json

        result = json.loads(transport["stdout"])
        assert receipt is None
        assert (policy.candidate / "compiler.py").read_bytes() == original
        assert result["parent"]["revision_id"] == parent.revision_id
        assert result["promotion"] == "NONE"
        assert coordinator.status()["incumbent"] == pool.baseline_revision.public_record()
        coordinator.finalize(timeout_s=2)
        with pytest.raises(StageGateError, match="remaining time"):
            coordinator.resume_parent(parent.revision_id)
    finally:
        pool.close()


def test_artifact_signature_requires_complete_observation_and_preserves_source():
    cell = {
        "family": "public",
        "capsule": "tail",
        "source_sha256": "a" * 64,
        "candidate": {"compiled": {"status": "observed", "elf_sha256": "b" * 64, "llvm_mlir_sha256": "c" * 64}},
    }
    original = PROV._signature([cell], "candidate")
    assert original is not None
    assert PROV._signature([{**cell, "source_sha256": "d" * 64}], "candidate") != original
    assert PROV._signature([{**cell, "candidate": {"compiled": {"status": "UNKNOWN"}}}], "candidate") is None
