"""Launcher input/custody controls; these tests do not execute a target."""

import hashlib
import importlib.util
import json
import time

import pytest
from merlin_experiments.phase2 import public_diagnostic_launch as L
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen.compiler_pass_selection import OriginalPassFactory, restore_original_factory


def pin(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def selected(tmp_path):
    leaf = pin(tmp_path / "metadata.json", {})
    now = time.time()
    clock = pin(
        tmp_path / "clock.json",
        {
            "schema": "phase2.public_diagnostic_infrastructure_packet.v1",
            "start_unix": now,
            "deadline_unix": now + 1200,
            "target_admission_cutoff_unix": now + 720,
            "cleanup_reserve_seconds": 480,
            "total_cpu_seconds": 600,
            "peak_rss_bytes": 1 << 30,
            "per_file_bytes": 64 << 20,
            "total_artifact_bytes": 768 << 20,
            "performance_campaign_restarted": False,
        },
    )
    plan = {name: leaf for name in L.FIELDS}
    plan.update(
        schema=L.SCHEMA,
        sources=[str(tmp_path)],
        cpu_ids=[6, 7],
        board_config="explicit-fixture-board",
        clock=clock,
        profile=dict(
            archive=leaf, workspace=str(tmp_path), repo=str(tmp_path), profile_sha256="a" * 64, snapshot_sha256="b" * 64
        ),
    )
    return plan, tmp_path


@pytest.mark.parametrize("fault", [None, "extra", "cpu", "schema", "sha", "link", "expired", "budget"])
def test_explicit_bounded_plan_pins_and_packet(selected, fault):
    plan, root = selected
    if fault == "extra":
        plan["command"] = ["unreviewed"]
    elif fault == "cpu":
        plan["cpu_ids"] = [7, 7]
    elif fault == "schema":
        plan["schema"] = "formal_phase2"
    elif fault in {"expired", "budget"}:
        value = json.loads((root / "clock.json").read_bytes())
        value["target_admission_cutoff_unix" if fault == "expired" else "total_cpu_seconds"] = (
            1 if fault == "expired" else 601
        )
        plan["clock"] = pin(root / "clock.json", value)
    record = pin(root / "plan.json", plan)
    if fault == "sha":
        record["sha256"] = "0" * 64
    elif fault == "link":
        link = root / "linked.json"
        link.symlink_to(root / "plan.json")
        record["path"] = str(link)
    if fault is None:
        actual, clock = L.read_plan(record["path"], record["sha256"])
        assert actual == plan and clock["performance_campaign_restarted"] is False
    else:
        with pytest.raises(StageGateError):
            L.read_plan(record["path"], record["sha256"])


def test_factory_source_restoration_cannot_restore_saved_owner_or_changed_source(tmp_path):
    source = tmp_path / "original.py"
    source.write_text("def restore(inputs, *, work_root):\n raise AssertionError('not admission yet')\n")
    spec = importlib.util.spec_from_file_location("original_launch_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = OriginalPassFactory(module.restore, {"public": True})
    restored = restore_original_factory(original.record())
    assert restored.record() == original.record()
    with pytest.raises(AssertionError, match="not admission yet"):
        restored.create(work_root=tmp_path / "not-created")
    record = original.record()
    record["successful_owner"] = {}
    with pytest.raises(ValueError, match="transport"):
        restore_original_factory(record)
    record = original.record()
    source.write_text("def restore(inputs, *, work_root):\n return {'success': True}\n")
    with pytest.raises(ValueError, match="source changed"):
        restore_original_factory(record)


def test_contract_projection_preserves_exact_leaves_and_refuses_unselected_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "schemas").mkdir()
    (source / "schemas" / "model.yaml").write_text("exact numerical schema")
    (source / "examples").symlink_to(tmp_path / "missing")
    rows = L.project_contract(source, tmp_path / "first")
    assert len(rows) == 1
    assert (tmp_path / "first/schemas/model.yaml").read_bytes() == (source / "schemas/model.yaml").read_bytes()
    assert not (tmp_path / "first/examples").exists()
    (source / "schemas/foreign").symlink_to(tmp_path / "missing")
    with pytest.raises(StageGateError, match="linked"):
        L.project_contract(source, tmp_path / "second")


def test_no_parent_cpu_lease_may_create_queue_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "current_thread_owns_cpu_lease", lambda: True)
    with pytest.raises(StageGateError, match="child CPU lease"):
        L.run_plan({}, {}, output=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_installed_entry_passes_only_pinned_plan_to_original_api(selected, monkeypatch):
    plan, root = selected
    record = pin(root / "plan.json", plan)
    calls = []
    monkeypatch.setattr(L, "run_plan", lambda actual, clock, **kw: calls.append((actual, clock, kw)))
    assert L.main(["--plan", record["path"], "--plan-sha256", record["sha256"], "--output", str(root / "fresh")]) == 0
    assert calls[0][0] == plan and calls[0][2]["output"] == root / "fresh"


@pytest.mark.parametrize("failed", [False, True])
def test_terminal_reporting_uses_original_pool_finalization(tmp_path, monkeypatch, failed):
    import threading

    import test_evaluation_costs as C

    fixture = C.parent.__wrapped__(tmp_path, monkeypatch)
    parent = next(fixture)
    try:
        parent.pool._lock = threading.RLock()
        job = parent.job()
        parent.pool._finish(job, error=StageGateError("source refused") if failed else None)
        receipt = (job.root / "cleanup.json").read_bytes()
        events = []
        status = L._wait_for_job(
            parent.pool,
            {"job_id": job.job_id},
            output=tmp_path,
            report=lambda stage, **values: events.append((stage, values)),
            remaining=lambda: pytest.fail("terminal job must not wait or reset its deadline"),
        )
        assert status["state"] == ("refused" if failed else "completed")
        assert events[-1][1] == {"queue_state": status["state"]}
        assert (job.root / "cleanup.json").read_bytes() == receipt
        assert json.loads((tmp_path / "queue-status.json").read_bytes())["terminal"]
    finally:
        fixture.close()


@pytest.mark.parametrize("damage", [None, "metadata_pin", "expired", "plan_sha"])
def test_plan_only_uses_existing_validation_and_never_issues_or_queues(selected, monkeypatch, capsys, damage):
    plan, root = selected
    if damage == "metadata_pin":
        plan["target_experiment"] = dict(plan["target_experiment"], sha256="0" * 64)
    elif damage == "expired":
        clock = json.loads((root / "clock.json").read_bytes())
        clock["target_admission_cutoff_unix"] = 1
        plan["clock"] = pin(root / "clock.json", clock)
    chosen = pin(root / "plan.json", plan)
    monkeypatch.setattr(L, "run_plan", lambda *a, **kw: pytest.fail("setup must not execute any queue"))
    monkeypatch.setattr(L, "restore_original_factory", lambda *a, **kw: pytest.fail("setup must not issue a factory"))
    argv = [
        "--check-plan",
        "--plan",
        chosen["path"],
        "--plan-sha256",
        "0" * 64 if damage == "plan_sha" else chosen["sha256"],
    ]
    if damage:
        with pytest.raises(StageGateError):
            L.main(argv)
        assert capsys.readouterr().out == ""
    else:
        before = sorted(p.name for p in root.iterdir())
        assert L.main(argv) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "NOT_EXECUTED" and report["authority"] == "NONE"
        assert report["selected"]["public_sources"] == plan["sources"]
        assert report["clock"]["admission_remaining_seconds"] > 0
        assert "queue" in report["not_checked"] and "timing" in report["not_checked"]
        assert sorted(p.name for p in root.iterdir()) == before


def test_setup_and_execution_output_modes_are_explicit(selected):
    plan, root = selected
    chosen = pin(root / "plan.json", plan)
    base = ["--plan", chosen["path"], "--plan-sha256", chosen["sha256"]]
    for extra in ([], ["--check-plan", "--output", str(root / "forbidden")]):
        with pytest.raises(SystemExit) as error:
            L.main([*base, *extra])
        assert error.value.code == 2
    assert not (root / "forbidden").exists()


@pytest.fixture
def issued_boundary(tmp_path, monkeypatch):
    """Original issuance with neutral kernel observations; not enforcement proof."""
    from merlin_experiments.phase2 import external_resources as R

    monkeypatch.setattr(R.kernel, "observe", lambda **kw: ("neutral-kernel-state",))
    return R.prepare_external_resource_boundary(
        cpu_ids=tuple(sorted(__import__("os").sched_getaffinity(0))),
        ram_bytes=1 << 30,
        disk_bytes=768 << 20,
        writable_roots=(tmp_path,),
    )


@pytest.mark.parametrize("fault", ["json", "reconstructed", "binding", "output"])
def test_optional_boundary_refuses_before_setup_or_output(tmp_path, monkeypatch, issued_boundary, fault):
    from dataclasses import replace

    from merlin_experiments.phase2 import external_resources as R

    owner = issued_boundary
    output = tmp_path / "fresh"
    if fault == "json":
        owner = {"status": "verified"}
    elif fault == "reconstructed":
        owner = replace(owner)
    elif fault == "binding":
        owner = R.prepare_external_resource_boundary(
            cpu_ids=owner.cpu_ids, ram_bytes=1, disk_bytes=768 << 20, writable_roots=(tmp_path,)
        )
    else:
        output = tmp_path.parent / "foreign-output"
    monkeypatch.setattr(L, "_execute_plan", lambda *a, **kw: pytest.fail("invalid owner cannot begin setup"))
    with pytest.raises(StageGateError):
        L.run_plan({"cpu_ids": list(issued_boundary.cpu_ids)}, {}, output=output, external_boundary=owner)
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_same_original_boundary_covers_setup_and_pool(tmp_path, monkeypatch, issued_boundary, selected):
    from types import SimpleNamespace

    from merlin_experiments.phase2 import evaluation_pool as E
    from merlin_experiments.phase2 import external_resources as R

    owner = issued_boundary if selected else None
    calls = []

    def pool(**kwargs):
        assert R.active_resource_boundary() is owner
        assert kwargs["external_boundary"] is owner
        assert kwargs["resources"].require_hard_limits is selected
        calls.append(kwargs)
        return "neutral-pool-routing-only"

    def execute(plan, clock, **kwargs):
        assert R.active_resource_boundary() is owner
        assert kwargs["external_boundary"] is owner
        policy = SimpleNamespace(feedback_evaluator=SimpleNamespace(work_root=tmp_path))
        return L._create_pool(
            policy=policy,
            revisions=object(),
            resources=kwargs["resources"],
            external_boundary=kwargs["external_boundary"],
        )

    monkeypatch.setattr(E, "ComponentEvaluationPool", pool)
    monkeypatch.setattr(L, "_execute_plan", execute)
    assert (
        L.run_plan({"cpu_ids": list(issued_boundary.cpu_ids)}, {}, output=tmp_path / "fresh", external_boundary=owner)
        == "neutral-pool-routing-only"
    )
    assert len(calls) == 1 and R.active_resource_boundary() is None
    assert not (tmp_path / "fresh").exists()
