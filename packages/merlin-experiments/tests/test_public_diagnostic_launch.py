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
