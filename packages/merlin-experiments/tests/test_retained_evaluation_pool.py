"""Harmless fork workers check retained revisions and actual cleanup ownership."""

import ast
import json
import os
import threading
import time
from dataclasses import replace

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from component_baseline_fixture import unissued_baseline
from merlin_experiments.phase2 import broker
from merlin_experiments.phase2 import component_analytical as CA
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import evaluation_pool as EP
from merlin_experiments.phase2 import retained_revisions as RR
from merlin_experiments.phase2.contracts import StageGateError, canonical_json
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import (
    RetainedRevisionStore,
    executing_candidate_revision,
    executing_revision,
    executing_revisions,
)
from test_component_analytical import provider
from test_component_workflow import context

from merlin.benchharness import hash_tree
from merlin.common import storage_lifecycle as storage
from merlin.perf.phase2_edit_contract import seal
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def authority_for(candidate, root):
    root.mkdir()
    authority = FrozenEditAuthority(root)
    authority.freeze(
        candidate,
        seal(
            {
                "schema": "compiler_edit_contract_v1",
                "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
                "helper_extensions": [],
            }
        ),
        has_iterations=False,
    )
    return authority


def synthetic_feedback(*, policy, revision, root, **_kwargs):
    """Only test execution: no compiler, paid agent, simulator or target process."""
    revision.verify()
    (root / "entered").write_text("started")
    while not (root / "continue").exists():
        time.sleep(0.01)
    revision.verify()
    p = policy.component_analytical
    return CW.validate_component_feedback(
        {
            "schema": CW.SCHEMA,
            "workflow_id": policy.workflow_id,
            "tier": "calibrated_component_analytical",
            "candidate_sha256": revision.source_sha256,
            "corpus_sha256": policy.component_corpus.capsules_sha256,
            "manifest_sha256": policy.component_corpus.manifest_sha256,
            "provider_sha256": p.implementation_sha256,
            "configuration_sha256": p.configuration_sha256,
            "promotion": "NO_FINAL_ACCEPTANCE",
            "evidence": [
                {
                    "family": member.family,
                    "capsule": member.capsule,
                    "baseline": CycleInterval.point(10, "test").to_dict(),
                    "candidate": CycleInterval.point(9, "test").to_dict(),
                }
                for member in policy.component_corpus.capsules
            ],
        }
    )


def descendants(*, root, **_kwargs):
    first = os.fork()
    if first == 0:
        os.setsid()
        second = os.fork()
        if second == 0:
            (root / "descendant").write_text(str(os.getpid()))
            while True:
                time.sleep(0.1)
        os._exit(0)
    os.waitpid(first, 0)
    while True:
        time.sleep(0.1)


def crashing(**_kwargs):
    os._exit(9)


def completing_feedback(**kwargs):
    (kwargs["root"] / "continue").touch()
    return synthetic_feedback(**kwargs)


def contextual_feedback(**kwargs):
    with executing_revisions(kwargs["revision"], kwargs["baseline_revision"]):
        return synthetic_feedback(**kwargs)


def wait_for(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    pytest.fail("synthetic evaluation did not reach its bounded observation")


@pytest.fixture
def selected_pool(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    selected = context(tmp_path)
    baseline = selected["baseline_admission"].baseline
    for compiler in (selected["candidate"], baseline):
        (compiler / "compiler.py").write_text("def schedule():\n    return 1\n")
    selected["baseline_admission"] = unissued_baseline(
        baseline=baseline, corpus=selected["component_corpus"], target_descriptor=selected["target_experiment"].path
    )
    selected["component_analytical"] = provider(tmp_path, monkeypatch, selected)
    policy = CW.ComponentOnlyPolicy(**selected)
    authority = authority_for(selected["candidate"], tmp_path / "authority")
    revisions = RetainedRevisionStore(
        tmp_path / "revisions", edit_authority=authority, max_bytes=4096, max_files=20, max_total_bytes=8192
    )
    worker = getattr(request, "param", synthetic_feedback)
    if worker is not None:
        monkeypatch.setattr(EP, "_fixed_evaluator", worker)
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:3]
    if len(cpus) < 2:
        pytest.skip("two assigned CPUs are needed for this harmless fan-out test")
    pool = policy.enable_evaluation_pool(
        revisions=revisions,
        resources=EP.AssignedResources(cpus, 8 << 30, 1 << 22, 2, author_cores=0),
        job_resources=EP.JobResources(1, 2 << 30, 1 << 18),
        root=tmp_path / "pool",
        beam=2,
        max_pending=3,
    )
    yield policy, pool, selected, authority
    pool.close()


def test_exact_sources_shared_options_separate_and_new_live_edit_does_not_change_a(selected_pool):
    policy, pool, selected, _authority = selected_pool
    a = pool.revisions.freeze(policy.candidate)
    annotated = pool.revisions.freeze(
        policy.candidate, parent_revision=a.revision_id, options={"hypothesis": "overlap"}
    )
    assert annotated.revision_id != a.revision_id and annotated.source == a.source
    assert a.source_sha256 == hash_tree(a.source)["sha256"]
    job = pool.submit(a.revision_id)
    root = pool.root / job["job_id"]
    wait_for(lambda: (root / "entered").exists())
    (selected["candidate"] / "compiler.py").write_text("def schedule():\n    return 2\n")
    b = pool.revisions.freeze(policy.candidate, parent_revision=a.revision_id)
    assert b.source_sha256 != a.source_sha256 and (a.source / "compiler").read_text() == "# synthetic tool\n"
    (root / "continue").touch()
    finished = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert finished["state"] == "completed" and finished["result"]["candidate_sha256"] == a.source_sha256
    assert finished["result"]["promotion"] == "NO_FINAL_ACCEPTANCE"
    assert json.loads((root / "cleanup.json").read_text())["cleanup"]["complete"]
    assert list(a.source.rglob("compiler")) == [a.source / "compiler"]


def test_verification_context_never_aliases_the_mutable_execution_path(selected_pool):
    policy, pool, _selected, authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    (policy.candidate / "compiler.py").write_text("def schedule(\n")
    assert executing_candidate_revision(authority) is None
    with executing_revisions(revision, pool.baseline_revision):
        assert executing_candidate_revision(authority) is revision
        assert executing_revision(policy.candidate) is None
        assert executing_revision(revision.source) is revision
        with pytest.raises(StageGateError, match="authority"):
            executing_candidate_revision(object())


@pytest.mark.parametrize("selected_pool", [None, contextual_feedback], indirect=True)
def test_live_syntax_edit_during_evaluation_keeps_every_result_check_in_issued_context(selected_pool, monkeypatch):
    policy, pool, _selected, authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    validate = CW.ComponentOnlyPolicy._validate
    configuration = CW.ComponentAnalyticalProvider.configuration_sha256.fget
    evaluate = CA._evaluate

    def validate_in_context(self):
        if self is policy and executing_candidate_revision(authority) is None:
            ast.parse((policy.candidate / "compiler.py").read_text())
        return validate(self)

    def configuration_in_context(self):
        if self is policy.component_analytical and executing_candidate_revision(authority) is None:
            ast.parse((policy.candidate / "compiler.py").read_text())
        return configuration(self)

    def gated_evaluate(binding, *args, **kwargs):
        root = binding.output.parent
        (root / "entered").touch()
        while not (root / "continue").exists():
            time.sleep(0.01)
        return evaluate(binding, *args, **kwargs)

    monkeypatch.setattr(CW.ComponentOnlyPolicy, "_validate", validate_in_context)
    monkeypatch.setattr(CW.ComponentAnalyticalProvider, "configuration_sha256", property(configuration_in_context))
    monkeypatch.setattr(CA, "_evaluate", gated_evaluate)
    job = pool.submit(revision.revision_id)
    root = pool.root / job["job_id"]
    wait_for(lambda: (root / "entered").exists())
    (policy.candidate / "compiler.py").write_text("def schedule(\n")
    (root / "continue").touch()
    final = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert final["state"] == "completed", final["refusal"]
    assert final["result"]["candidate_sha256"] == revision.source_sha256


@pytest.mark.parametrize("selected_pool", [completing_feedback], indirect=True)
def test_bounded_storage_delay_keeps_cleanup_ownership_until_release(selected_pool, monkeypatch):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    unpin, entered = storage.unpin, threading.Event()
    observed, done = {}, threading.Event()

    def delayed_unpin(path, token):
        if not entered.is_set():
            entered.set()
            time.sleep(0.5)
        return unpin(path, token)

    monkeypatch.setattr(storage, "unpin", delayed_unpin)
    job = pool.submit(revision.revision_id)
    assert entered.wait(timeout=10)
    lifecycle = next(row for row in storage.inventory()["paths"] if row["path"] == str(pool.root / job["job_id"]))
    assert lifecycle["leases"] == 1

    def observe_status():
        started = time.monotonic()
        observed["row"] = pool.status(job["job_id"])
        observed["seconds"] = time.monotonic() - started
        done.set()

    observer = threading.Thread(target=observe_status, daemon=True)
    observer.start()
    status_waited = not done.wait(timeout=0.05)
    assert done.wait(timeout=60)
    observer.join()
    print(f"[queue-storage-delay] status_waited={status_waited} response_seconds={observed['seconds']:.3f}")
    assert observed["row"]["state"] == "completed" and observed["row"]["result"]
    lifecycle = next(row for row in storage.inventory()["paths"] if row["path"] == str(pool.root / job["job_id"]))
    assert lifecycle["leases"] == 0
    assert json.loads((pool.root / job["job_id"] / "cleanup.json").read_text())["cleanup"]["complete"]


def test_drift_refuses_queued_and_previously_completed_results(selected_pool):
    policy, pool, selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    job = pool.submit(revision.revision_id)
    root = pool.root / job["job_id"]
    wait_for(lambda: (root / "entered").exists())
    selected["target_experiment"].path.write_text("changed configuration")
    (root / "continue").touch()
    finished = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert finished["state"] == "refused" and finished["result"] is None


@pytest.mark.parametrize("selected_pool", [descendants], indirect=True)
def test_backpressure_and_cancel_release_only_after_double_fork_reaping(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    jobs = [pool.submit(revision.revision_id) for _ in range(3)]
    root = pool.root / jobs[0]["job_id"]
    wait_for(lambda: (root / "descendant").exists())
    with pytest.raises(StageGateError, match="backpressure"):
        pool.submit(revision.revision_id)
    descendant = int((root / "descendant").read_text())
    initial = pool.cancel(jobs[0]["job_id"])
    assert initial["state"] == "cancelling" and not initial["terminal"]
    assert storage.blockers(root)
    final = wait_for(lambda: pool.status(jobs[0]["job_id"]) if pool.status(jobs[0]["job_id"])["terminal"] else None)
    assert final["state"] == "cancelled" and final["result"] is None
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)
    lifecycle = next(row for row in storage.inventory()["paths"] if row["path"] == str(root))
    assert lifecycle["leases"] == 0 and lifecycle["state"] == "failed"
    # Other jobs retain the pool/cache root until their own reaping completes.
    assert storage.blockers(root)
    for job in jobs[1:]:
        pool.cancel(job["job_id"])


def test_failed_snapshot_copy_rolls_back_before_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("def schedule():\n    return 1\n")
    authority = authority_for(candidate, tmp_path / "authority")
    revisions = RetainedRevisionStore(tmp_path / "revisions", edit_authority=authority, max_bytes=4096, max_files=20)
    validate = authority.validate_candidate

    def refuse(_candidate):
        raise StageGateError("injected frozen authority refusal")

    monkeypatch.setattr(authority, "validate_candidate", refuse)
    with pytest.raises(StageGateError, match="injected"):
        revisions.freeze(candidate)
    assert not list((revisions.root / "sources").iterdir())
    assert all(row["leases"] == 0 for row in storage.inventory()["paths"])
    monkeypatch.setattr(authority, "validate_candidate", validate)
    revision = revisions.freeze(candidate)
    assert revision.verify() == hash_tree(candidate)["sha256"]


def test_partial_enqueue_pin_failure_releases_its_issued_lease(selected_pool, monkeypatch):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    before = sum(len(row["pins"]) for row in storage.inventory()["paths"])
    pin, calls = storage.pin, 0

    def refuse_second_pin(path, *, reason):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected pin storage refusal")
        return pin(path, reason=reason)

    monkeypatch.setattr(storage, "pin", refuse_second_pin)
    with pytest.raises(OSError, match="injected"):
        pool.submit(revision.revision_id)
    assert pool.observations() == ()
    inventory = storage.inventory()["paths"]
    assert sum(len(row["pins"]) for row in inventory) == before
    assert all(row["leases"] == 0 for row in inventory)


@pytest.mark.parametrize("selected_pool", [crashing], indirect=True)
def test_crash_cannot_become_feedback(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    job = pool.submit(revision.revision_id)
    final = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert final["state"] == "refused" and final["result"] is None


def test_freeze_caps_and_issued_authority_cannot_be_bypassed(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    with pytest.raises(StageGateError, match="issued"):
        replace(revision, revision_id="0" * 64).verify()
    with pytest.raises(StageGateError, match="file budget"):
        from merlin_experiments.phase2.retained_revisions import source_inventory

        source_inventory(policy.candidate, max_bytes=4096, max_files=1)
    with pytest.raises(StageGateError, match="byte budget"):
        from merlin_experiments.phase2.retained_revisions import source_inventory

        source_inventory(policy.candidate, max_bytes=1, max_files=20)
    with pytest.raises(StageGateError, match="assigned capacity"):
        EP.ComponentEvaluationPool(
            policy=policy,
            revisions=pool.revisions,
            resources=pool.resources,
            job_resources=EP.JobResources(1, pool.resources.ram_bytes, 4096),
            root=pool.root.parent / "oversized",
        )


def test_actual_broker_control_receipt_cannot_claim_scientific_completion(selected_pool):
    policy, pool, selected, _authority = selected_pool
    engine = broker.Broker(
        object(),
        selected["target_experiment"],
        policy.candidate,
        policy.build_registry(),
        policy.receipt_path,
        workflow=policy,
        deadline=time.monotonic() + 30,
        max_calls=5,
        max_tool_seconds=10,
    )
    response = engine.execute({"action": CW.REVISION_FREEZE_ACTION, "bindings": {"parent_revision": "none"}})
    revision = json.loads(response["stdout"])
    queued = engine.execute(
        {
            "action": CW.EVALUATION_SUBMIT_ACTION,
            "bindings": {"revision_id": revision["revision_id"], "kind": EP.KINDS[0]},
        }
    )
    status = json.loads(queued["stdout"])
    assert status["state"] == "queued" and not status["terminal"] and status["result"] is None
    rows = [json.loads(line) for line in policy.receipt_path.read_text().splitlines()]
    assert all("feedback_receipt_path" not in row for row in rows)
    assert canonical_json(pool.observations()[0]) == canonical_json(pool.status(status["job_id"]))
    pool.cancel(status["job_id"])


def test_completed_result_is_reopened_and_source_drift_revokes_it(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    job = pool.submit(revision.revision_id)
    root = pool.root / job["job_id"]
    wait_for(lambda: (root / "entered").exists())
    (root / "continue").touch()
    final = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert final["state"] == "completed"
    (root / "feedback.json").chmod(0o600)
    (root / "feedback.json").write_text("{}")
    assert pool.status(job["job_id"])["state"] == "refused"
    assert pool.status(job["job_id"])["result"] is None


def test_explicit_hard_limits_refuse_without_enforcing_envelope(selected_pool):
    _policy, pool, _selected, _authority = selected_pool
    with pytest.raises(StageGateError, match="hard limits"):
        replace(pool.resources, require_hard_limits=True).verify()
    assert pool.enforcement["hard_aggregate_limits"] == "UNKNOWN"


@pytest.mark.parametrize("selected_pool", [None], indirect=True)
def test_fixed_analytical_route_uses_the_retained_source_without_member_copies(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    # Virtual address space is explicit and generous for inherited Python imports;
    # this fixture creates only tiny text products and executes no target tool.
    revision = pool.revisions.freeze(policy.candidate)
    job = pool.submit(revision.revision_id)
    final = wait_for(lambda: pool.status(job["job_id"]) if pool.status(job["job_id"])["terminal"] else None)
    assert final["state"] == "completed", final["refusal"]
    assert final["result"]["tier"] == "calibrated_component_analytical"
    assert not list((pool.root / job["job_id"]).glob("**/candidate/compiler.py"))
    assert len(list(pool.revisions.root.glob("sources/*/compiler.py"))) == 1


@pytest.mark.parametrize("selected_pool", [completing_feedback], indirect=True)
def test_stop_submissions_preserves_queued_work_for_drain(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    jobs = [pool.submit(revision.revision_id) for _ in range(3)]
    pool.stop_submissions()
    with pytest.raises(StageGateError, match="submissions stopped"):
        pool.submit(revision.revision_id)
    rows = pool.drain(timeout_s=600)
    assert {row["job_id"] for row in rows} == {job["job_id"] for job in jobs}
    assert all(row["terminal"] and row["state"] == "completed" and row["result"] for row in rows)
    assert pool._engine_lease is None


@pytest.mark.parametrize("selected_pool", [descendants], indirect=True)
def test_drain_timeout_cancels_and_reaps_remaining_jobs(selected_pool):
    policy, pool, _selected, _authority = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    jobs = [pool.submit(revision.revision_id) for _ in range(3)]
    root = pool.root / jobs[0]["job_id"]
    wait_for(lambda: (root / "descendant").exists())
    descendant = int((root / "descendant").read_text())
    rows = pool.drain(timeout_s=0)
    assert all(row["terminal"] and row["state"] == "cancelled" and row["result"] is None for row in rows)
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)
    assert not storage.blockers(pool.root)


@pytest.fixture
def restorable_source(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("def schedule():\n    return 1\n")
    (candidate / "compiler.py").chmod(0o740)
    (candidate / "helpers").mkdir()
    (candidate / "helpers").chmod(0o750)
    (candidate / "helpers" / "original.py").write_text("def original():\n    return 1\n")
    authority = FrozenEditAuthority(tmp_path / "authority")
    authority.output.mkdir()
    authority.freeze(
        candidate,
        seal(
            {
                "schema": "compiler_edit_contract_v1",
                "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
                "helper_extensions": [{"directory": "helpers", "surface_ids": ["schedule"]}],
                "ancillary_paths": ["docs"],
            }
        ),
        has_iterations=False,
    )
    revisions = RetainedRevisionStore(tmp_path / "revisions", edit_authority=authority, max_bytes=4096, max_files=20)
    revision = revisions.freeze(candidate)
    return candidate, authority, revisions, revision


def test_restore_replays_exact_source_in_same_root_and_removes_only_authorized_extras(restorable_source):
    candidate, authority, revisions, revision = restorable_source
    identity = candidate.stat().st_dev, candidate.stat().st_ino
    (candidate / "compiler.py").write_text("def schedule():\n    return 2\n")
    (candidate / "helpers" / "extra.py").write_text("def extra():\n    return 2\n")
    (candidate / "docs").mkdir()
    (candidate / "docs" / "note.txt").write_text("author note")
    authority.validate_candidate(candidate)
    result = revisions.restore(revision.revision_id, candidate)
    assert result == revision.public_record() and "source" not in result
    assert hash_tree(candidate)["sha256"] == revision.source_sha256
    assert (candidate.stat().st_dev, candidate.stat().st_ino) == identity
    assert (candidate / "compiler.py").stat().st_mode & 0o111 == 0o100
    assert (revision.source / "compiler.py").stat().st_mode & 0o111 == 0o100
    assert (candidate / "helpers").stat().st_mode & 0o111 == 0o110
    assert not (candidate / "helpers" / "extra.py").exists() and not (candidate / "docs").exists()
    assert not list(candidate.rglob(".restore-*"))


def test_restore_refuses_foreign_candidate_and_replaced_original_root(restorable_source):
    candidate, _authority, revisions, revision = restorable_source
    foreign = candidate.parent / "foreign"
    foreign.mkdir()
    with pytest.raises(StageGateError, match="exact original"):
        revisions.restore(revision.revision_id, foreign)
    original = candidate.parent / "original"
    candidate.rename(original)
    candidate.mkdir()
    with pytest.raises(StageGateError, match="root inode"):
        revisions.restore(revision.revision_id, candidate)
    assert hash_tree(original)["sha256"] == revision.source_sha256


@pytest.mark.parametrize("kind", ["symlink", "fifo", "unauthorized-directory"])
def test_restore_refuses_linked_special_and_unauthorized_extra_target_entries(restorable_source, kind):
    candidate, _authority, revisions, revision = restorable_source
    if kind == "symlink":
        (candidate / "helpers" / "linked").symlink_to(candidate.parent)
    elif kind == "fifo":
        os.mkfifo(candidate / "helpers" / "fifo")
    else:
        (candidate / "foreign").mkdir()
    with pytest.raises(StageGateError, match="linked or special|lacks frozen"):
        revisions.restore(revision.revision_id, candidate)


def test_restore_detects_concurrent_disagreement_without_accepting_selection(restorable_source, monkeypatch):
    candidate, _authority, revisions, revision = restorable_source
    (candidate / "compiler.py").write_text("def schedule():\n    return 2\n")
    replace_file = RR.os.replace

    def concurrent_edit(source, destination, **kwargs):
        result = replace_file(source, destination, **kwargs)
        if destination == "original.py":
            (candidate / "compiler.py").write_text("def schedule():\n    return 3\n")
        return result

    monkeypatch.setattr(RR.os, "replace", concurrent_edit)
    with pytest.raises(StageGateError, match="differs from the exact"):
        revisions.restore(revision.revision_id, candidate)
    assert hash_tree(candidate)["sha256"] != revision.source_sha256
    assert not list(candidate.rglob(".restore-*"))
