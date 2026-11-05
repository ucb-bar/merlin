"""Exact post-run source freezing; no performance or qualification is established."""

from pathlib import Path

import pytest
from merlin_experiments.phase2 import lean_authoring as L
from merlin_experiments.phase2 import lean_finalization as F
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, mapping_file, write_json
from merlin_experiments.phase2.retained_revisions import source_inventory


@pytest.fixture
def campaign(tmp_path):
    retained = tmp_path / "author" / "retained"
    retained.mkdir(parents=True)
    sources, revisions = {}, {}
    for name in ("control", "candidate"):
        source = retained / "sources" / name
        source.mkdir(parents=True)
        (source / "compiler.py").write_text(f"# {name}\n")
        before = source_inventory(source, max_bytes=4096, max_files=10)
        source = source.rename(source.with_name(before["sha256"]))
        (source / "compiler.py").chmod(0o444)
        source.chmod(0o555)
        inventory = source_inventory(source, max_bytes=4096, max_files=10, readonly=True)
        record = dict(schema="merlin.retained_revision.v1", source_sha256=inventory["sha256"],
                      source_inventory_sha256=document_sha256(inventory),
                      parent_revision=None if name == "control" else revisions["control"],
                      options=dict(hypothesis="immutable original control" if name == "control" else "candidate"),
                      edit_authority_sha256="c" * 64)
        identity = document_sha256(record)
        write_json(retained / (identity + ".json"), record)
        (retained / (identity + ".json")).chmod(0o444)
        sources[name], revisions[name] = source, identity
    config = dict(schema=L.PROTOCOL, output=str(tmp_path), members=[dict(name="x", compute_floor_cycles=40)],
                  frozen_model_sha256="a" * 64, target_fraction=.3, max_jobs=4)
    path = tmp_path / "config.json"
    write_json(path, config)
    job = dict(job_id="job", revision_id=revisions["candidate"], source_sha256=sources["candidate"].name,
               members=["x"], state="running", result=None)
    write_json(tmp_path / "author" / "result.json", dict(schema=L.PROTOCOL, final_status=dict(jobs=[job])))
    directory = tmp_path / "jobs" / "job"
    directory.mkdir(parents=True)
    write_json(directory / "worker-request.json", dict(evaluation=dict(candidate_source=str(sources["candidate"]),
        baseline_source=str(sources["control"]), selected_members=[dict(id="x")])))
    counts = dict(status="observed", native_outputs_complete=True, feature_complete=True, complete_counts=True,
                  timing_complete=True, unresolved_events=[], estimated_cycles=100, attainment_fraction=.4,
                  model_sha256="a" * 64, scope_sha256="b" * 64)
    arm = dict(status="numeric_pass", complete_outputs_checked=True, count_observation=counts)
    result = dict(members=[dict(id="x", baseline=arm, candidate=arm)])
    write_json(directory / "worker-result.json", result)
    return path, tmp_path, sources, revisions, result


def test_late_worker_result_is_recovered_and_exact_source_frozen_without_rewriting_author(campaign):
    config, root, sources, revisions, _ = campaign
    original = (root / "author" / "result.json").read_bytes()
    final = F.finalize_lean_campaign(config, root / "final", execution_closed=True)
    assert final["selected_revision"]["revision_id"] == revisions["candidate"]
    assert final["estimated_structural_target_met"] is True
    assert final["qualification"] == "UNQUALIFIED" and final["source_effects"] == "PENDING"
    assert final["roofline_convergence"] is False and final["promotion_authority"] == "NONE"
    assert (root / "author" / "result.json").read_bytes() == original
    frozen = Path(final["frozen_compiler"]) / "compiler.py"
    assert frozen.read_bytes() == (sources["candidate"] / "compiler.py").read_bytes()
    assert not Path(final["frozen_compiler"]).stat().st_mode & 0o222
    assert not (root / "final" / "result.json").stat().st_mode & 0o222


@pytest.mark.parametrize("fault", ["pending", "numeric", "incomplete_counts"])
def test_missing_or_failed_evidence_never_invents_a_complete_target(campaign, fault):
    config, root, _, revisions, result = campaign
    path = root / "jobs" / "job" / "worker-result.json"
    if fault == "pending":
        path.unlink()
    elif fault == "numeric":
        result["members"][0]["candidate"] = dict(status="numeric_failure", complete_outputs_checked=False)
        write_json(path, result)
    else:
        result["members"][0]["candidate"]["count_observation"]["timing_complete"] = False
        write_json(path, result)
    final = F.finalize_lean_campaign(config, root / "final", execution_closed=True)
    assert final["estimated_structural_target_met"] is False
    if fault != "incomplete_counts":
        assert final["selected_revision"]["revision_id"] == revisions["control"]


def test_shutdown_assertion_and_changed_source_or_worker_binding_are_refused(campaign):
    config, root, sources, _, _ = campaign
    with pytest.raises(StageGateError, match="processes have ended"):
        F.finalize_lean_campaign(config, root / "not-final")
    request = root / "jobs" / "job" / "worker-request.json"
    document = mapping_file(request)
    document["evaluation"]["candidate_source"] = str(sources["control"])
    write_json(request, document)
    with pytest.raises(StageGateError, match="binding changed"):
        F.finalize_lean_campaign(config, root / "not-final", execution_closed=True)
    document["evaluation"]["candidate_source"] = str(sources["candidate"])
    write_json(request, document)
    source = sources["candidate"] / "compiler.py"
    source.chmod(0o644)
    source.write_text("changed\n")
    source.chmod(0o444)
    with pytest.raises(StageGateError, match="inventory changed"):
        F.finalize_lean_campaign(config, root / "not-final", execution_closed=True)
