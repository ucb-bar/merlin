"""Optional off-path cleanup of newly evaluated rejected/duplicate products.

Reuse the candidate's existing completion receipt and storage lifecycle. No new
registry, executor, cache scan, reference counter or task/worktree cleanup. The
producer opts in once; active evaluation and authoring never wait for cleanup.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from merlin.common import storage_lifecycle
from merlin.common.paths import out_dir

_BUDGET = 64 << 10


def _owned(path):
    path = Path(os.path.abspath(path))
    if out_dir().resolve() not in path.parents or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('candidate artifacts must stay strictly within current output, without symlinks')
    return path


@contextmanager
def hold_candidate_artifacts(directory):
    """One lease for the whole candidate job, including its completion callback.

    Use ``hold_inputs=lambda request: hold_candidate_artifacts(owners[request.key])``
    with evaluate_emitted. Acquire before submit, release after the callback and
    all child workers have stopped. No artifact walks or hashes occur here.
    """
    directory = _owned(directory)
    with storage_lifecycle.lease(directory, owner='candidate evaluation and completion callback'):
        if not directory.is_dir():
            raise FileNotFoundError('candidate artifact directory is absent')
        yield directory


def _decision(receipt):
    with receipt.open('rb') as stream:
        payload = stream.read(_BUDGET + 1)
    if len(payload) > _BUDGET:
        raise ValueError('existing candidate completion receipt exceeds small-summary budget')
    record = json.loads(payload)
    if record.get('schema') != 'merlin.emitted_evaluation_completion.v1':
        raise ValueError('original emitted evaluation completion receipt required')
    for key in ('evaluation_identity', 'program_sha256', 'feedback_sha256'):
        value = record.get(key)
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('completion receipt lacks exact evaluation/program/feedback identity')
    retirement = record.get('retention')
    if (type(retirement) is not dict or retirement.get('decision') not in ('rejected', 'duplicate')
            or type(retirement.get('reason')) is not str or not retirement['reason']
            or len(retirement['reason']) > 1024):
        raise ValueError('only explicit terminal rejected/duplicate decisions can retire products')
    commit, patch = retirement.get('source_commit'), retirement.get('source_patch')
    valid_commit = (type(commit) is str and len(commit) in (40, 64)
                    and all(c in '0123456789abcdef' for c in commit))
    if not valid_commit and not (type(patch) is str and patch and len(patch.encode()) <= _BUDGET // 2):
        raise ValueError('retain original source commit or bounded patch in existing result')
    supported = (record.get('status') == 'rankable' and record.get('numerical_status') == 'pass'
                 and record.get('timing_status') == 'complete' and record.get('complete_roi') is True
                 and type(record.get('cycles')) is int and record['cycles'] > 0)
    if not supported and record.get('numerical_status') != 'fail':
        raise ValueError('pending/UNKNOWN products stay deferred, not prematurely rejected')
    if retirement['decision'] == 'duplicate' and retirement.get('duplicate_of') != record['program_sha256']:
        raise ValueError('duplicate cleanup needs byte-identical original program identity')
    return retirement['decision']


def cleanup_retired_candidate(products, *, receipt, apply=False):
    """Dry-run by default; enqueue on the owner's existing maintenance executor.

    ``receipt`` is the EXISTING compact EvaluationCompletion.record() with one
    caller-owned ``retention`` mapping: decision, reason, source_commit (or
    source_patch), and duplicate_of for duplicate products. Keep it and any
    rollback/source files OUTSIDE products. Winners/incumbents/controls never
    enter this terminal-rejection queue. Their on_completion owner keeps them
    with the existing storage_lifecycle.pin(..., reason=...) API until explicit
    replacement/rollback; this helper never releases those pins.

    Only the explicit rebuildable products subtree is inspected/deleted. Shared
    compiler/toolchain sources remain outside it. Final removal uses the existing
    lease/pin/citation/Git checks, outside evaluation's critical path. No historical
    receipt adoption or automatic UNKNOWN expiry is performed.
    """
    from merlin.common import storage_cli

    if type(apply) is not bool:
        raise TypeError('cleanup opt-in must be an explicit Boolean')
    products, receipt = _owned(products), _owned(receipt)
    if products == receipt or products in receipt.parents or receipt in products.parents:
        raise ValueError('retain the small original result outside rebuildable products')
    decision = _decision(receipt)
    reasons = storage_cli._protection_reasons(products, require_terminal=True)
    size = storage_cli.measure(products).occupied
    result = dict(decision=decision, blockers=reasons, product_bytes=size,
                  reclaimable_bytes=0 if reasons else size, applied=False, summary_preserved=str(receipt))
    if apply and not reasons and products.exists():
        # Rechecks leases/pins/citations under the same lifecycle lock as job
        # admission. The earlier dry-run is never deletion authority.
        storage_cli._remove(products, require_terminal=True)
        result['applied'] = True
    return result
