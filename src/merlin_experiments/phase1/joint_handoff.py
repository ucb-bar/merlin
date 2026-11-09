"""Freeze-ready source custody for an operator epoch stop or the joint freeze, without new grading.

At the joint freeze the final coverage grades the frozen bytes itself; a post-turn development grade
there has no L3 oracle (0/N by construction), is recorded only as the development verdict, and would
spend the final window. Both stops therefore retain the source and the existing evidence instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common import oot_repo
from . import joint_epochs, oot_history


FREEZE = 'joint campaign freeze/drain boundary reached'


def stop(run_dir):
    """``(kind, reason)`` for an operator epoch stop or the joint freeze, else ``(None, None)``."""
    reason = joint_epochs.stop_reason(run_dir)
    if reason is not None:
        return 'epoch', reason
    from . import joint_control

    if joint_control.selected() is not None and joint_control.remaining(drain=True) <= 0:
        return 'freeze', FREEZE
    return None, None


def checkpoint(workspace, run_dir, round_index):
    """Called only after the author and its existing background grade have joined."""
    workspace, run_dir = Path(workspace), Path(run_dir)
    kind, reason = stop(run_dir)
    if kind is None:
        raise ValueError('fast source handoff requires an explicit operator epoch stop or the joint freeze')
    source = workspace / 'submission'
    digest = oot_repo.package_digest(source)
    live = workspace / 'qa/verdict.json'
    previous = json.loads(live.read_bytes()) if live.is_file() else {}
    archive = run_dir / 'qa_history' / f'{kind}_handoff_{round_index:02d}_prior.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(previous, sort_keys=True, indent=2).encode() + b'\n'
    with archive.open('xb') as stream:
        stream.write(raw)
    archive.chmod(0o400)
    total = previous.get('n_required_development')
    if total is not None and (type(total) is not int or total < 0):
        raise ValueError('prior development denominator is invalid')
    verdict = {
        'all_pass': None, 'gradeable': False, 'formal_complete': False,
        'n_passed': 0, 'n_execution_passed': 0, 'n_capsules': 0,
        'n_required_development': total, 'n_selected_development': 0,
        'n_pending_development': total, 'per_capsule': [],
        'count_scope': f'{kind} handoff: no new checks; previous evidence retained separately',
        'status': 'ungraded_operator_checkpoint' if kind == 'epoch' else 'ungraded_freeze_checkpoint',
        'stop_reason': reason, 'current_source_eligible': False,
        'current_source_n_passed': 0, 'checkpoint_source_sha256': digest,
        'prior_evidence': {'path': str(archive), 'sha256': hashlib.sha256(raw).hexdigest()},
        'note': ('Operator epoch stop: retained source and existing evidence; post-turn grading deferred.'
                 if kind == 'epoch' else
                 'Joint freeze: retained source and existing evidence; final coverage grades the frozen bytes.'),
    }
    label = f'{kind}_checkpoint'
    record = oot_history.commit_graded(run_dir, source, label=label,
                                      key=f'{round_index:02d}', sandbox_roots=(workspace,))
    if record is None:
        raise ValueError('operator checkpoint requires original OOT source custody')
    oot_history.verify_turn_end(run_dir, source, record.commit)
    if oot_repo.package_digest(source) != digest:
        raise ValueError('compiler changed during operator checkpoint')
    verdict['oot_commit'] = record.commit
    oot_history.record_round(run_dir, record, label=label, key=f'{round_index:02d}', verdict=verdict)
    print(f'[round {round_index}] {kind} checkpoint retained source; no post-turn jobs started', flush=True)
    return verdict
