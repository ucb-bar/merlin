"""Operator-reviewed authored checkpoints in a new joint campaign clock.

This selects source bytes only. It carries no previous results, physical budget,
clock, model qualification or empty-backend assertion into the new run.

Two optional, separately reviewed additions (each must appear identically in the
operator review, and both default to off):

* ``inherit_certificates_for_identical_executables: true`` lets a certificate earned
  by byte-identical program bytes under byte-identical grading code carry into this
  run through :mod:`merlin.targetgen.tier_cache` (its portable instrument mode,
  ``MERLIN_TIER_CERT_PORTABLE=1``, which the launch must also set). A carried verdict
  is labelled as carried with its source record; anything whose program or judge
  differs is executed again. Verdict-level credit is still not inherited.
* ``previous_session: {run_dir, workspace}`` stages ``PREVIOUS_SESSION.md``
  (:mod:`.continuation_handoff`), built from the previous session's own outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import run_inputs

SCHEMA = 'merlin.joint_reviewed_seed.v1'
CARRY = 'inherit_certificates_for_identical_executables'
PREVIOUS = 'previous_session'


def selected(options, policy=None):
    """Reopen a trusted policy checkpoint; candidate manifests cannot select it."""
    if not getattr(options, 'joint_campaign', ''):
        return None
    if policy is None:
        from .joint_control import policy as read_policy
        policy = read_policy(options.joint_campaign)
    origin = policy.get('compiler_origin')
    if origin is None:
        return None
    if (not isinstance(origin, dict) or origin.get('schema') != SCHEMA
            or not isinstance(origin.get('review_id'), str) or not origin['review_id'].strip()
            or origin.get('inherit_clock') is not False or origin.get('inherit_results') is not False
            or getattr(options, 'fresh_backend', False) or getattr(options, 'joint_epoch', '')
            or getattr(options, 'resume', False) or getattr(options, 'qualify_submission', '')
            or getattr(options, 'sandbox', '') != 'bwrap' or getattr(options, 'allow_unsandboxed', False)):
        raise ValueError('reviewed joint seed requires a new isolated source-only campaign')
    source = origin.get('source')
    if not isinstance(source, str) or not source or getattr(options, 'seed_submission', '') != source:
        raise ValueError('joint seed differs from its operator-selected checkpoint')
    path = Path(source)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('joint seed requires a canonical checkpoint path')
    if run_inputs.graded_source_hash(path) != origin.get('sha256'):
        raise ValueError('reviewed joint seed source bytes changed')
    pin = origin.get('review', {})
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('joint seed lacks an exact operator review')
    review_path = Path(pin['path'])
    if review_path.is_symlink() or not review_path.is_absolute() or review_path.resolve(strict=True) != review_path:
        raise ValueError('joint seed review is indirect')
    payload = review_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != pin['sha256']:
        raise ValueError('joint seed review changed')
    review = json.loads(payload)
    if (review.get('schema') != 'operator.joint_seed_review.v1'
            or review.get('approved') is not True
            or any(review.get(key) != origin[key] for key in ('review_id', 'source', 'sha256', 'inherit_clock', 'inherit_results', 'variants'))
            or any(review.get(key) != origin.get(key) for key in (CARRY, PREVIOUS))):
        raise ValueError('joint seed review does not approve this exact source-only origin')
    if origin.get(CARRY) not in (None, False, True):
        raise ValueError('certificate carry-over approval must be a boolean')
    previous = origin.get(PREVIOUS)
    if previous is not None and (
            not isinstance(previous, dict) or set(previous) != {'run_dir', 'workspace'}
            or not all(isinstance(v, str) and Path(v).is_absolute() for v in previous.values())):
        raise ValueError('previous session handoff requires absolute run_dir and workspace')
    variants = origin.get('variants')
    if not isinstance(variants, list) or len(variants) > 64:
        raise ValueError('joint seed authored alternatives must be bounded')
    return origin


def stage(options, workspace, run_dir):
    """Use the existing native copy/strip path with source checks on both sides."""
    origin = selected(options)
    if origin is None:
        raise ValueError('reviewed joint seed is not selected')
    result = run_inputs.seed_submission(workspace, origin['source'], run_dir,
                                       require_stable_source=True, require_lineage=False)
    if selected(options) != origin:
        raise ValueError('joint seed selection changed during staging')
    from .joint_interrupted_epoch import seed_variants
    result['authored_alternatives'] = seed_variants(origin, workspace)
    result['reviewed_joint_seed'] = origin
    result['current_credit_granted'] = False
    result['inherited_clock'] = False
    from merlin.targetgen import tier_cache
    carry = origin.get(CARRY) is True
    if carry != tier_cache.portable():
        raise ValueError(f'certificate carry-over needs both the reviewed {CARRY} approval and '
                         f'{tier_cache.PORTABLE_ENV}=1 at launch; one without the other is refused')
    result['certificate_carry_over'] = {
        'enabled': carry,
        'mechanism': 'merlin.targetgen.tier_cache portable instrument identity',
        'scope': ('a tier verdict carries only when the program execution identity and the grading '
                  'instrument are byte-identical; carried verdicts are labelled carried with their source '
                  'record, everything else is executed again') if carry else 'off: every tier is executed again',
    }
    if origin.get(PREVIOUS) is not None:
        from .continuation_handoff import write as write_handoff
        result['previous_session'] = write_handoff(workspace, origin[PREVIOUS], Path(workspace) / 'submission')
    (Path(run_dir) / 'seed_submission.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
