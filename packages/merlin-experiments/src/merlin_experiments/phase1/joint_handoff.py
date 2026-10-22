"""Freeze-ready source custody for an operator epoch stop, without new grading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common import oot_repo
from . import joint_epochs, oot_history


def checkpoint(workspace, run_dir, round_index):
    """Called only after the author and its existing background grade have joined."""
    workspace, run_dir = Path(workspace), Path(run_dir)
    if joint_epochs.stop_reason(run_dir) is None:
        raise ValueError('fast source handoff requires an explicit operator epoch stop')
    source = workspace / 'submission'
    digest = oot_repo.package_digest(source)
    live = workspace / 'qa/verdict.json'
    previous = json.loads(live.read_bytes()) if live.is_file() else {}
    archive = run_dir / 'qa_history' / f'epoch_handoff_{round_index:02d}_prior.json'
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
        'count_scope': 'operator handoff: no new checks; previous evidence retained separately',
        'status': 'ungraded_operator_checkpoint', 'current_source_eligible': False,
        'current_source_n_passed': 0, 'checkpoint_source_sha256': digest,
        'prior_evidence': {'path': str(archive), 'sha256': hashlib.sha256(raw).hexdigest()},
        'note': 'Operator epoch stop: retained source and existing evidence; post-turn grading deferred.',
    }
    record = oot_history.commit_graded(run_dir, source, label='epoch_checkpoint',
                                      key=f'{round_index:02d}', sandbox_roots=(workspace,))
    if record is None:
        raise ValueError('operator checkpoint requires original OOT source custody')
    oot_history.verify_turn_end(run_dir, source, record.commit)
    if oot_repo.package_digest(source) != digest:
        raise ValueError('compiler changed during operator checkpoint')
    verdict['oot_commit'] = record.commit
    oot_history.record_round(run_dir, record, label='epoch_checkpoint', key=f'{round_index:02d}', verdict=verdict)
    print(f'[round {round_index}] operator checkpoint retained source; no post-turn jobs started', flush=True)
    return verdict
