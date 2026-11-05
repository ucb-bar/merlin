"""Campaign bookkeeping refuses incomplete metrics and unmatched measurements.

Synthetic receipts test joins only; they establish no execution qualification.
"""

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def campaign(tmp_path):
    path = repo_root() / "examples/gemmini/phase2/continuing_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.CONFIG = {
        "target": {"required_members": 1, "attainable_roofline_fraction": 0.3},
        "deadline_utc": "later",
        "compiler_freeze_utc": "later",
        "compiler_freeze_epoch": 1e20,
    }
    return module


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


def fixture_candidate(campaign, tmp_path, cycles=80):
    elf = tmp_path / "candidate.elf"
    elf.write_bytes(b"synthetic identity only")
    numerical = dump(
        tmp_path / "numerical.json",
        {
            "status": "complete_outputs_passed",
            "checked_elements": 15,
            "elf": campaign.pin(elf),
        },
    )
    report = dump(
        tmp_path / "report.json",
        {
            "feature_observation": {"complete": True},
            "timing_complete": True,
            "estimated_cycles": cycles,
            "scope_binding": {"elf_sha256": campaign.pin(elf)["sha256"]},
            "model_sha256": "fixed",
        },
    )
    evidence = tmp_path / "facts.txt"
    evidence.write_text("independent fixture")
    request = {
        "id": "trial",
        "member": "capsule",
        "elf": campaign.pin(elf),
        "numerical": {
            "status": "pass",
            "complete_outputs": True,
            "elf_sha256": campaign.pin(elf)["sha256"],
            "artifact": campaign.pin(numerical),
        },
        "spike": {"estimated_cycles": cycles, "model_sha256": "fixed", "report": campaign.pin(report)},
        "roofline": {
            "floor_cycles": 10,
            "kind": "partial_compute",
            "assumptions": ["conditional"],
            "uncertainty": "unqualified",
            "evidence": [campaign.pin(evidence)],
        },
    }
    board = {
        "members": [
            {
                "member": "capsule",
                "checked_elements": 15,
                "model_sha256": "fixed",
                "baseline_estimated_cycles": 100,
                "current_estimated_cycles": 100,
                "best_estimated_cycles": 100,
            }
        ]
    }
    return dump(tmp_path / "request.json", request), board, request


@pytest.mark.parametrize("missing", ["cycles", "coverage", "outputs", "roofline", "same_elf"])
def test_incomplete_observations_cannot_become_incumbent(campaign, tmp_path, missing):
    path, board, request = fixture_candidate(campaign, tmp_path)
    if missing == "cycles":
        request["spike"]["estimated_cycles"] = None
    elif missing == "coverage":
        report_path = tmp_path / "report.json"
        report = json.loads(report_path.read_text())
        report["timing_complete"] = False
        dump(report_path, report)
        request["spike"]["report"] = campaign.pin(report_path)
    elif missing == "outputs":
        request["numerical"]["complete_outputs"] = False
    elif missing == "roofline":
        request["roofline"]["uncertainty"] = None
    else:
        request["numerical"]["elf_sha256"] = "other"
    dump(path, request)
    with pytest.raises(ValueError):
        campaign.ingest(path, board)
    assert board["members"][0]["current_estimated_cycles"] == 100


@pytest.mark.parametrize("cycles,status", [(80, "provisional_estimated_incumbent"), (120, "rejected_cycle_regression")])
def test_cycles_select_provisional_candidates_and_partial_floor_stays_partial(campaign, tmp_path, cycles, status):
    path, board, _ = fixture_candidate(campaign, tmp_path, cycles)
    result = campaign.ingest(path, board)
    assert result["status"] == status
    assert result["formal_qualification"] is False
    assert board["members"][0].get("attainable_roofline_attainment") is None
    assert board["members"][0]["current_estimated_cycles"] == min(cycles, 100)


def test_changed_model_requires_baseline_replay(campaign, tmp_path):
    path, board, _ = fixture_candidate(campaign, tmp_path)
    board["members"][0]["model_sha256"] = "different"
    assert campaign.ingest(path, board)["status"] == "baseline_replay_required"
    assert board["members"][0]["current_estimated_cycles"] == 100


def test_changed_evidence_is_reopened(campaign, tmp_path):
    path, board, _ = fixture_candidate(campaign, tmp_path)
    (tmp_path / "candidate.elf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        campaign.ingest(path, board)


def test_missing_case_or_insufficient_measured_attainment_cannot_meet_target(campaign):
    row = {
        "member": "capsule",
        "measured_speedup": 2.1,
        "current_estimated_cycles": 80,
        "attainable_roofline_attainment": 0.4,
        "measured_attainable_roofline_attainment": 0.29,
    }
    board = {"members": [row]}
    assert campaign.summary(board, [])["hardware_target_met"] is False
    row["measured_attainable_roofline_attainment"] = 0.31
    assert campaign.summary(board, [])["hardware_target_met"] is True
    campaign.CONFIG["target"]["required_members"] = 2
    result = campaign.summary(board, [])
    assert result["hardware_target_met"] is False
    assert result["measured_geomean_speedup"] is None


def fixture_measurement(campaign, tmp_path):
    elf = tmp_path / "batch.elf"
    elf.write_bytes(b"synthetic hardware identity only")
    cases = [
        dict(member="capsule", arm=arm, checked_elements=15, complete_outputs=True, measured_cycles=cycles)
        for arm, cycles in (("baseline", [200, 204]), ("trial", [100, 102]))
    ]
    manifest = dump(tmp_path / "manifest.json", dict(elf=campaign.pin(elf), cases=cases))
    artifact = tmp_path / "HW_CFG_SUMMARY"
    artifact.write_text("synthetic configuration identity")
    value = dict(
        schema="merlin.phase2.firesim_batch_result.v1",
        status="pass",
        elf=campaign.pin(elf),
        uploaded_elf=campaign.pin(elf),
        source_manifest=campaign.pin(manifest),
        artifacts={"HW_CFG_SUMMARY": campaign.pin(artifact)},
        cases=cases,
    )
    board = dict(members=[dict(member="capsule", checked_elements=15, baseline_measured_cycles=None)])
    return dump(tmp_path / "measurement.json", value), board, value


def test_hardware_speedup_uses_baseline_in_actual_paired_batch(campaign, tmp_path):
    path, board, _ = fixture_measurement(campaign, tmp_path)
    board["members"][0]["baseline_measured_cycles"] = [10000]
    result = campaign.review(campaign.ingest_measurement, path, board)
    assert result["formal_qualification"] is False
    assert board["members"][0]["measured_speedup"] == 2
    assert board["members"][0]["matched_baseline_measured_cycles"] == [200, 204]


@pytest.mark.parametrize("mutation", ["duplicate", "dropped_outputs", "invalid_later_row", "changed_upload"])
def test_invalid_hardware_batch_never_partially_updates_scoreboard(campaign, tmp_path, mutation):
    import copy

    path, board, value = fixture_measurement(campaign, tmp_path)
    if mutation == "duplicate":
        value["cases"][1] = value["cases"][0]
    elif mutation == "dropped_outputs":
        for case in value["cases"]:
            case["checked_elements"] = 14
        manifest = dump(tmp_path / "manifest.json", dict(elf=value["elf"], cases=value["cases"]))
        value["source_manifest"] = campaign.pin(manifest)
    elif mutation == "invalid_later_row":
        value["cases"][1]["measured_cycles"] = [0]
    else:
        other = tmp_path / "other.elf"
        other.write_bytes(b"different upload")
        value["uploaded_elf"] = campaign.pin(other)
    dump(path, value)
    original = copy.deepcopy(board)
    with pytest.raises(ValueError):
        campaign.review(campaign.ingest_measurement, path, board)
    assert board == original


def test_priority_refresh_uses_current_measured_conservative_floor(campaign):
    row = dict(
        conditional_necessary_floor_cycles=10,
        independent_frozen_feasible_cycles_upper=100,
        current_estimated_cycles=40,
        current_measured_cycles=[50, 50],
        compute_peak_utilization=0.01,
    )
    campaign.refresh_roofline_ratios(row)
    assert campaign.priority_fraction(row) == 0.2
    assert row["conditional_roofline_attainment_lower"] == 0.25
    row["current_measured_cycles"] = [25, 25]
    campaign.refresh_roofline_ratios(row)
    assert campaign.priority_fraction(row) == 0.4


@pytest.mark.parametrize("version", [1, 2])
def test_conditional_roofline_pack_keeps_qualification_pending(campaign, tmp_path, version):
    evidence = dump(
        tmp_path / "envelope.json",
        dict(
            member="capsule",
            theoretical_necessary_floor_cycles=10,
            independent_frozen_complete_feasible_cycles_upper=100,
            assumptions=["conditional algorithm family"],
            uncertainty="reference is a feasible witness",
            full_attainable_qualification="PENDING",
        ),
    )
    pack = dump(
        tmp_path / "pack.json",
        dict(
            schema=f"merlin.phase2.conditional_roofline_pack.v{version}",
            members=[dict(member="capsule", artifact=campaign.pin(evidence))],
        ),
    )
    board = dict(members=[dict(member="capsule", current_estimated_cycles=50)])
    campaign.review(campaign.ingest_roofline_pack, pack, board)
    assert board["members"][0]["conditional_roofline_attainment_lower"] == 0.2
    assert board["members"][0]["full_attainable_qualification"] == "PENDING"


def test_interrupted_external_action_requires_declared_resume(campaign, tmp_path):
    request = dict(argv=["external-action"])
    assert campaign.job_argv(tmp_path, request) == ["external-action"]
    dump(tmp_path / "request.json", request)
    with pytest.raises(campaign.RecoveryRequired):
        campaign.job_argv(tmp_path, request)
    with pytest.raises(ValueError, match="immutable"):
        campaign.job_argv(tmp_path, dict(argv=["different-action"]))
    request["resume_argv"] = ["resume-exact-owned-command", "command-id"]
    dump(tmp_path / "request.json", request)
    assert campaign.job_argv(tmp_path, request) == request["resume_argv"]


@pytest.mark.parametrize("mutation", [None, "missing_member", "changed_baseline", "changed_input", "missing_replay"])
def test_model_revision_requires_complete_original_baseline_pairs(campaign, tmp_path, mutation):
    import copy

    model = dump(tmp_path / "model.json", dict(identity="new source-derived scenario"))
    model_sha = campaign.pin(model)["sha256"]
    packets = []
    for name, cycles in (("baseline", 200), ("current", 120)):
        folder = tmp_path / name
        folder.mkdir()
        path, board, request = fixture_candidate(campaign, folder, cycles)
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["model_sha256"] = model_sha
        report["scope_binding"].update(input_sha256="unchanged", scope_sha256="whole-call")
        if mutation == "changed_input" and name == "current":
            report["scope_binding"]["input_sha256"] = "different"
        if mutation == "missing_replay" and name == "baseline":
            report["timing_complete"] = False
        dump(report_path, report)
        request["id"] = name
        request["spike"].update(model_sha256=model_sha, report=campaign.pin(report_path))
        dump(path, request)
        packets.append(path)
    member = board["members"][0]
    member.update(current_estimated_cycles=80, best_estimated_cycles=80)
    manifest = dump(
        tmp_path / "manifest.json",
        dict(
            cases=[
                dict(
                    member="capsule",
                    arm="baseline",
                    source_elf=json.loads(packets[0].read_text())["elf"],
                )
            ]
        ),
    )
    if mutation == "changed_baseline":
        value = json.loads(manifest.read_text())
        value["cases"][0]["source_elf"]["sha256"] = "original baseline differs"
        dump(manifest, value)
    hardware = dump(tmp_path / "hardware.json", dict(source_manifest=campaign.pin(manifest)))
    member["baseline_hardware_artifact"] = campaign.pin(hardware)
    pack = dump(
        tmp_path / "revision.json",
        dict(
            schema="merlin.phase2.complete_model_replay.v1",
            model=campaign.pin(model),
            evidence=[campaign.pin(model)],
            members=[]
            if mutation == "missing_member"
            else [
                dict(
                    member="capsule",
                    previous_model_sha256="fixed",
                    baseline=campaign.pin(packets[0]),
                    current=campaign.pin(packets[1]),
                )
            ],
        ),
    )
    original = copy.deepcopy(board)
    if mutation:
        with pytest.raises(ValueError):
            campaign.review(campaign.ingest_model_replay, pack, board)
        assert board == original
    else:
        result = campaign.review(campaign.ingest_model_replay, pack, board)
        assert result["formal_qualification"] is False
        assert member is not board["members"][0]
        member = board["members"][0]
        assert member["model_sha256"] == model_sha
        assert member["baseline_estimated_cycles"] == 200
        assert member["current_estimated_cycles"] == member["best_estimated_cycles"] == 120
        assert member["estimated_speedup"] == 200 / 120


def test_watchdog_recovers_after_controller_crash_without_reissuing_action(tmp_path):
    import sqlite3
    import sys
    import time

    from merlin.common import owned_process

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "job-requests").mkdir()
    (root / "checkpoints").mkdir()
    dump(
        root / "manifest.json",
        dict(
            task_dir=str(tmp_path),
            deadline_epoch=time.time() + 8,
            compiler_freeze_epoch=1e20,
            deadline_utc="fixture",
            compiler_freeze_utc="fixture",
            checkpoint_seconds=1,
            target=dict(required_members=1, attainable_roofline_fraction=0.3),
        ),
    )
    dump(
        root / "scoreboard.json",
        dict(
            members=[
                dict(
                    member="capsule",
                    current_estimated_cycles=100,
                    attainable_roofline_attainment=None,
                )
            ]
        ),
    )
    with sqlite3.connect(root / "queue.sqlite") as db:
        db.execute("CREATE TABLE work (id INTEGER PRIMARY KEY, priority REAL, status TEXT, lease_until REAL)")
    action = (
        "from pathlib import Path; import os,signal,time; "
        f"r=Path({str(root)!r}); (r/'issued').write_text('once'); "
        "os.kill(int((r/'controller.lock').read_text()),signal.SIGKILL); time.sleep(30)"
    )
    resume = f"from pathlib import Path; Path({str(root / 'resumed')!r}).write_text('same owned action')"
    dump(
        root / "job-requests/job.json",
        dict(argv=[sys.executable, "-c", action], resume_argv=[sys.executable, "-c", resume], timeout_seconds=20),
    )
    result = owned_process.run(
        [
            sys.executable,
            str(repo_root() / "examples/gemmini/phase2/continuing_campaign.py"),
            "--campaign-root",
            str(root),
            "--watchdog",
        ],
        timeout=30,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.owned_process_cleanup["complete"] is True
    assert (root / "issued").read_text() == "once"
    assert (root / "resumed").read_text() == "same owned action"
    assert json.loads((root / "jobs/job/result.json").read_text())["status"] == "complete"
    assert len(list((root / "watchdog").iterdir())) >= 2
