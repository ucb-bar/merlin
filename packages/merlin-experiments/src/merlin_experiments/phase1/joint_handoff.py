"""Freeze-ready source custody for an operator epoch stop, without new grading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common import oot_repo
from . import joint_epochs, oot_history


def drain_required(run_dir):
    """True at the joint freeze boundary when the turn end should checkpoint instead of grading.

    Final coverage of the frozen submission supersedes the post-turn development grade; running it
    first spends the freeze/drain window (it produced 14 gated smoke rows after the deadline).
    """
    from . import joint_control

    return (
        joint_control.selected() is not None
        and joint_control.drain_checkpoint_enabled()
        and joint_epochs.stop_reason(run_dir) is None
        and joint_control.freeze_reached()
    )


def checkpoint(workspace, run_dir, round_index, *, drain=False):
    """Called only after the author and its existing background grade have joined."""
    workspace, run_dir = Path(workspace), Path(run_dir)
    if drain:
        if not drain_required(run_dir):
            raise ValueError("drain checkpoint requires the joint freeze boundary")
    elif joint_epochs.stop_reason(run_dir) is None:
        raise ValueError("fast source handoff requires an explicit operator epoch stop")
    kind = "drain" if drain else "epoch"
    label = "drain_checkpoint" if drain else "epoch_checkpoint"
    source = workspace / "submission"
    digest = oot_repo.package_digest(source)
    live = workspace / 'qa/verdict.json'
    previous = json.loads(live.read_bytes()) if live.is_file() else {}
    archive = run_dir / "qa_history" / f"{kind}_handoff_{round_index:02d}_prior.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(previous, sort_keys=True, indent=2).encode() + b'\n'
    with archive.open('xb') as stream:
        stream.write(raw)
    archive.chmod(0o400)
    total = previous.get('n_required_development')
    if total is not None and (type(total) is not int or total < 0):
        raise ValueError('prior development denominator is invalid')
    verdict = {
        "all_pass": None,
        "gradeable": False,
        "formal_complete": False,
        "n_passed": 0,
        "n_execution_passed": 0,
        "n_capsules": 0,
        "n_required_development": total,
        "n_selected_development": 0,
        "n_pending_development": total,
        "per_capsule": [],
        "count_scope": (
            "freeze/drain handoff: no post-turn development grade; the frozen submission is covered by final coverage"
            if drain
            else "operator handoff: no new checks; previous evidence retained separately"
        ),
        "status": "ungraded_drain_checkpoint" if drain else "ungraded_operator_checkpoint",
        "current_source_eligible": False,
        "current_source_n_passed": 0,
        "checkpoint_source_sha256": digest,
        "prior_evidence": {"path": str(archive), "sha256": hashlib.sha256(raw).hexdigest()},
        "note": (
            "Joint freeze/drain: retained source and existing evidence; post-turn grading skipped so "
            "the guaranteed final coverage budget is not consumed."
            if drain
            else "Operator epoch stop: retained source and existing evidence; post-turn grading deferred."
        ),
    }
    record = oot_history.commit_graded(
        run_dir, source, label=label, key=f"{round_index:02d}", sandbox_roots=(workspace,)
    )
    if record is None:
        raise ValueError('operator checkpoint requires original OOT source custody')
    oot_history.verify_turn_end(run_dir, source, record.commit)
    if oot_repo.package_digest(source) != digest:
        raise ValueError("compiler changed during operator checkpoint")
    verdict["oot_commit"] = record.commit
    oot_history.record_round(run_dir, record, label=label, key=f"{round_index:02d}", verdict=verdict)
    print(f"[round {round_index}] {kind} checkpoint retained source; no post-turn jobs started", flush=True)
    return verdict
