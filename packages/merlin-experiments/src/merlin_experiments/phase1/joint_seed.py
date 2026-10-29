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
* ``seed_evidence: {path, sha256}`` stages an operator-prepared evidence directory (the seed
  run's optimization rules, experiment records, variants, performance history and notes moved out
  of the compiler) at the workspace root, outside ``submission/``, after checking its tree digest.
  It is reference material only and grants nothing. Its ``RUN_PRIVATE/`` entry is never staged in
  the workspace: its files (the seed's model cycles by executable bytes, its fast-estimate
  calibration pairs and its per-case baseline) go to the run directory, where the grader reads them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import run_inputs

SCHEMA = 'merlin.joint_reviewed_seed.v1'
CARRY = 'inherit_certificates_for_identical_executables'
PREVIOUS = 'previous_session'
EVIDENCE = 'seed_evidence'
#: Evidence entries the author keeps maintaining; everything else is staged read-only.
EVIDENCE_EDITABLE = frozenset({'OPTIMIZATION_RULES.json'})
#: Evidence entry copied into the run directory (read by the grader), never into the workspace.
RUN_PRIVATE = 'RUN_PRIVATE'


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
            or any(review.get(key) != origin.get(key) for key in (CARRY, PREVIOUS, EVIDENCE))):
        raise ValueError('joint seed review does not approve this exact source-only origin')
    if origin.get(CARRY) not in (None, False, True):
        raise ValueError('certificate carry-over approval must be a boolean')
    previous = origin.get(PREVIOUS)
    if previous is not None and (
            not isinstance(previous, dict) or set(previous) != {'run_dir', 'workspace'}
            or not all(isinstance(v, str) and Path(v).is_absolute() for v in previous.values())):
        raise ValueError('previous session handoff requires absolute run_dir and workspace')
    evidence = origin.get(EVIDENCE)
    if evidence is not None:
        if (not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}
                or not isinstance(evidence['path'], str) or not Path(evidence['path']).is_absolute()):
            raise ValueError('seed evidence requires one absolute path/sha256 pin')
        from merlin.common.tree_hash import hash_tree
        if hash_tree(Path(evidence['path']))['sha256'] != evidence['sha256']:
            raise ValueError('reviewed seed evidence changed')
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
    if origin.get(EVIDENCE) is not None:
        result['seed_evidence'] = stage_evidence(origin[EVIDENCE], workspace)
        result['seed_evidence']['run_private'] = stage_run_evidence(origin[EVIDENCE], run_dir)
    (Path(run_dir) / 'seed_submission.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def stage_evidence(pin, workspace):
    """Copy the reviewed evidence entries to the workspace root; never replaces a staged file."""
    import shutil

    source, root = Path(pin['path']), Path(workspace)
    staged, skipped = [], []
    for entry in sorted(source.iterdir()):
        target = root / entry.name
        if (entry.is_symlink() or entry.name in ('submission', RUN_PRIVATE) or target.exists()
                or target.is_symlink()):
            skipped.append(entry.name)
            continue
        if entry.is_dir():
            shutil.copytree(entry, target, symlinks=False)
        else:
            shutil.copyfile(entry, target)
        if entry.name not in EVIDENCE_EDITABLE:
            # Files only: a read-only directory would make every later copy of the workspace undeletable.
            for path in ([target, *target.rglob('*')] if target.is_dir() else [target]):
                if path.is_file():
                    path.chmod(0o444)
        staged.append(entry.name)
    return {'source': str(source), 'sha256': pin['sha256'], 'staged': staged, 'skipped': skipped}


def stage_run_evidence(pin, run_dir):
    """Copy the evidence's ``RUN_PRIVATE/`` files into the run directory; never replaces a file there."""
    import shutil

    source, root = Path(pin['path']) / RUN_PRIVATE, Path(run_dir)
    staged, skipped = [], []
    if not source.is_dir() or source.is_symlink():
        return {'staged': staged, 'skipped': skipped}
    for entry in sorted(source.iterdir()):
        target = root / entry.name
        if entry.is_symlink() or target.exists() or target.is_symlink():
            skipped.append(entry.name)
            continue
        (shutil.copytree if entry.is_dir() else shutil.copyfile)(entry, target)
        staged.append(entry.name)
    return {'staged': staged, 'skipped': skipped}
