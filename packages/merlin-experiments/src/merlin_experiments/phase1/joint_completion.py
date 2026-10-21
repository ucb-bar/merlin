"""Freeze the joint experiment's final author bytes without legacy certification."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import yaml

from merlin.common.tree_hash import hash_tree
from . import joint_control


def complete(prepared, *, verdict, rounds, workflow_conformant):
    run_dir, workspace = prepared.run_dir, prepared.workspace
    source = workspace / "submission"
    destination = run_dir / "joint_frozen_submission"
    prepared.verify_inputs()
    if destination.exists():
        raise ValueError("joint completion already froze a submission")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    digest = hash_tree(destination)["sha256"]
    for member in destination.rglob("*"):
        if member.is_file():
            member.chmod(member.stat().st_mode & ~0o222)
    state = joint_control.selected()
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
        "completion_reason": "freeze_drain" if joint_control.remaining(drain=True) <= 0 else "interrupted",
    }
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
    return 0 if summary["completion_reason"] == "freeze_drain" else 1
