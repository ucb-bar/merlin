"""Freeze the joint experiment's final author bytes without legacy certification."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import yaml

from merlin.common.tree_hash import hash_tree
from . import joint_control


def completion_reason(run_dir):
    # A reviewed epoch checkpoint never consumes the final validation budget,
    # including a checkpoint requested just as the drain boundary is crossed.
    if (run_dir / "joint_epoch_stop.json").is_file():
        return "library_epoch"
    return "freeze_drain" if joint_control.remaining(drain=True) <= 0 else "interrupted"


def complete(prepared, *, verdict, rounds, workflow_conformant, final_coverage=None):
    run_dir, workspace = prepared.run_dir, prepared.workspace
    source = workspace / "submission"
    destination = run_dir / "joint_frozen_submission"
    prepared.verify_inputs()
    if destination.exists():
        raise ValueError("joint completion already froze a submission")
    from .run_inputs import SNAPSHOT_IGNORED, strip_build_state
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*SNAPSHOT_IGNORED))
    strip_build_state(destination)
    digest = hash_tree(destination)["sha256"]
    for member in destination.rglob("*"):
        if member.is_file():
            member.chmod(member.stat().st_mode & ~0o222)
    state = joint_control.selected()
    from .study_protocol import validate as study_protocol
    study = study_protocol(state.get("policy", {}))
    summary = {
        "schema": "merlin.joint_campaign_completion.v1",
        "converged": False, "formal_complete": False, "legacy_certificate_issued": False,
        "numeric_all_pass": verdict.get("all_pass"),
        "workflow_conformant": workflow_conformant,
        "compiler_sha256": digest, "frozen_compiler": str(destination),
        "rounds": rounds, "elapsed_wall_s": time.time() - state["started_at_epoch"],
        "policy_sha256": state["policy_sha256"],
        "development_verdict": verdict,
        "performance_qualification": "reported separately by versioned timing evidence",
        "final_validation": {"required": True, "engine": "firesim", "status": "pending",
                             "scope": ["all_admitted_capsules", "full_models"], "complete": False,
                             "configuration": "exact selected unmodified hardware; no alternate configuration"},
        "completion_reason": completion_reason(run_dir),
    }
    summary["final_cheap_coverage"] = {
        "required": True, "complete": False, "passed": False,
        "status": "not_run", "reason": summary["completion_reason"],
    }
    if study is not None:
        summary["final_validation"]["scope"] = ["selected_capsules", "frozen_transfer_panel"]
        summary["final_cheap_coverage"].update(
            status="deferred", reason="operator evaluation after all authors freeze",
            frozen_compiler=str(destination), compiler_sha256=digest,
        )
        summary["numeric_all_pass"] = None
        summary["study_protocol"] = study
    elif summary["completion_reason"] == "freeze_drain":
        try:
            if final_coverage is None:
                raise RuntimeError("native final coverage callback is missing")
            summary["final_cheap_coverage"] = final_coverage(destination, digest)
        except Exception as exc:
            summary["final_cheap_coverage"].update(status="error", error=f"{type(exc).__name__}: {exc}")
        summary["numeric_all_pass"] = summary["final_cheap_coverage"].get("passed") is True
    live = workspace / "qa" / "verdict.json"
    if live.is_file():
        summary["latest_feedback"] = json.loads(live.read_text())
    try:
        from .telemetry.report import _emit_run_timing
        summary["telemetry"] = _emit_run_timing(run_dir, rounds)
    except Exception as exc:
        summary["telemetry"] = {"complete": False, "failure": type(exc).__name__}
    (run_dir / "joint_campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "qa_loop_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"[joint] compiler frozen: {digest}; no historical Phase 1 certificate issued", flush=True)
    if summary["completion_reason"] == "library_epoch":
        return 0  # An epoch checkpoint is explicitly not campaign completion.
    if study is not None:
        return 0 if summary["completion_reason"] == "freeze_drain" else 1
    coverage = summary["final_cheap_coverage"]
    return 0 if (summary["completion_reason"] == "freeze_drain"
                 and coverage.get("complete") is True and coverage.get("passed") is True) else 1
