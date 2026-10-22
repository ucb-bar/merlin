"""Join independently passed native tiers without equating their harness ELFs.

This is a source/capsule obligation join, never an executable-cache alias. Each
engine keeps its own exact execution digest. Saved public reports are not an
input: observe() reads the operator's original completed products and producer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

_COMMON = {'capsule', 'target', 'required_tiers', 'compiler_sha256', 'submission_digest', 'toolchain_shas',
    'source_ir_sha256', 'raw_command_buffer_sha256', 'effective_command_buffer_sha256',
    'lowered_ir_sha256', 'kernel_object_sha256'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(path, *, tier):
    from .candidate_execution_products import reopen_operator_tier_products
    from .promotion import execution_digest, submission_tree_digest

    path = Path(path)
    raw = path.read_bytes()
    result = json.loads(raw)
    tiers, trace = result.get('tiers', {}), result.get('trace_check', {})
    if (result.get('candidate_native_execution') is not None
            or any(tiers.get(t, {}).get('status') != 'pass' for t in ('L0', 'L1', tier))
            or result.get('numeric', {}).get('status') != 'pass'
            or trace.get('status') != 'pass' or trace.get('violations')
            or trace.get('trace_requirement_status') != 'pass'
            or trace.get('unavailable_requirements')):
        raise ValueError('tier join requires original complete numerical and structural checks')
    failure = result.get('failure')
    other = 'L3' if tier == 'L2' else 'L2'
    if failure and not (failure.get('plane') == 'oracle_unavailable'
            and failure.get('category') == 'NOT_RUN_IS_NOT_PASS'
            and failure.get('tier') == other
            and tiers.get(other, {}).get('status') == 'unavailable'):
        raise ValueError('tier join cannot hide a measured failure')
    gen = path.parent / 'generated'
    products, _ = reopen_operator_tier_products(gen, result, tier=tier)
    if tiers[tier].get('engine') != tiers[tier].get('sim_provenance', {}).get('engine'):
        raise ValueError('tier engine differs from its reopened execution producer')
    manifest_raw = (path.parent / 'run_manifest.yaml').read_bytes()
    manifest = yaml.safe_load(manifest_raw)
    source = result['compiler_source']
    identity = execution_digest(path)
    if identity is None or manifest.get('benchmark') != result.get('capsule'):
        raise ValueError('tier join lacks an exact native execution/capsule identity')
    # Same effective buffer binds the original stimulus/numerical policy as well
    # as the raw compiler declaration. Neither buffer's contents are published.
    common = {
        'capsule': result['capsule'], 'target': manifest['target'],
        'required_tiers': manifest['metadata']['required_oracle_tiers'],
        'compiler_sha256': source['candidate_sha256'],
        'submission_digest': submission_tree_digest(gen / source['snapshot']),
        'toolchain_shas': result['toolchain_shas'],
        'source_ir_sha256': _sha(products['source']),
        'raw_command_buffer_sha256': _sha(products['compiler_command_buffer']),
        'effective_command_buffer_sha256': _sha(products['command_buffer']),
        'lowered_ir_sha256': _sha(products['emitted']),
        'kernel_object_sha256': _sha(products['object']),
    }
    evidence = {'schema': 'merlin.original_tier_boundary.v1', 'common': common,
        'tier': tier, 'engine': tiers[tier]['engine'], 'execution_digest': identity,
        'elf_sha256': _sha(products['elf']), 'harness_sha256': _sha(products['caller']),
        'result_sha256': hashlib.sha256(raw).hexdigest()}
    reopen_operator_tier_products(gen, result, tier=tier)
    if path.read_bytes() != raw or (path.parent / 'run_manifest.yaml').read_bytes() != manifest_raw:
        raise ValueError('original tier result changed while reopening')
    return evidence


def optional_observation(path, *, tier):
    """Unsupported roles remain unjoined and retain their original verdict."""
    try:
        return observe(path, tier=tier)
    except (ValueError, OSError, KeyError, TypeError):
        return None


def join(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError('both original tier observations are required')
    for value, tier in ((left, 'L2'), (right, 'L3')):
        if value.get('schema') != 'merlin.original_tier_boundary.v1' or value.get('tier') != tier:
            raise ValueError('tier observation role differs')
        for field in ('execution_digest', 'elf_sha256', 'harness_sha256', 'result_sha256'):
            pin = value.get(field)
            if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
                raise ValueError('tier observation lacks an exact artifact pin')
    common = left.get('common')
    if not isinstance(common, dict) or set(common) != _COMMON:
        raise ValueError('tier observation boundary is incomplete')
    if common != right.get('common'):
        raise ValueError('compiler, capsule, input, numerical or emitted-kernel boundary changed')
    required = common.get('required_tiers')
    if (not isinstance(required, list) or any(not isinstance(t, str) for t in required)
            or len(set(required)) != len(required) or not {'L2', 'L3'} <= set(required)):
        raise ValueError('tier boundary does not declare both execution obligations')
    if not left.get('engine') or not right.get('engine') or left['engine'] == right['engine']:
        raise ValueError('same-engine execution changes require the ordinary exact identity route')
    return {'schema': 'merlin.cross_engine_obligation_join.v1',
        'common': left['common'], 'L2': left, 'L3': right,
        'scope': 'same source/capsule boundary; separate native ELFs, no executable-cache alias'}


def replay(loop_result, cert_result):
    """Read-only operator replay of retained results; runs no compiler/simulator."""
    return join(observe(loop_result, tier='L2'), observe(cert_result, tier='L3'))


def resolve(state, name, requested_identity, row, *, source_identity_verified):
    """Resolve one queued boundary, retaining the actual cert ELF as its key."""
    from . import promotion as P
    if not source_identity_verified or row.get('pass') is not True or row.get('barrier_status') != 'pass':
        return None
    pending = P._slots_ro(state, name, 'L3').get(requested_identity)
    loop = P._slots_ro(state, name, 'L2').get(requested_identity)
    if (not isinstance(pending, dict) or pending.get('status') != 'pending'
            or not isinstance(loop, dict) or loop.get('status') != 'pass'
            or loop.get('execution_digest') != requested_identity):
        return None
    try:
        proof = join(pending.get('tier_boundary'), row.get('tier_boundary'))
    except ValueError:
        return None
    actual = row.get('execution_digest')
    if (proof['common']['capsule'] != name
            or proof['common']['submission_digest'] != pending.get('digest')
            or proof['common']['submission_digest'] != loop.get('digest')
            or loop.get('tier_boundary') != proof['L2']
            or proof['L2']['execution_digest'] != requested_identity
            or proof['L3']['execution_digest'] != actual):
        return None
    entry = dict(pending, execution_digest=actual, tier_boundary=proof['L3'], boundary_join=proof)
    # A link is NOT a pass for the Spike ELF, nor readable by tier_cache.lookup.
    old = dict(pending, status='joined', certified_execution_digest=actual)
    P._slots(state, name, 'L3')[requested_identity] = old
    return entry


def apply_retained(ws, loop_result, cert_result, *, instrument, completion):
    """Operator-only recovery through the ordinary recorder, with zero target runs.

    Uses the existing ledger transaction lock. Original result files are not edited.
    Existing pending source/engine records must already match these exact inputs.
    """
    from . import promotion as P
    proof = replay(loop_result, cert_result)
    return P._ledger_transaction(_apply_verified)(ws, proof, instrument=instrument, completion=completion)


def _completion_custody(ws, path, proof, instrument):
    """Reopen the original private broker's source/judge binding, never its public response."""
    from . import completions, snapshots

    path = Path(path)
    anchor = snapshots.promotion_snapshot_path(ws, '0' * 32)
    root = anchor.parent / '.completed'
    if (root.resolve().is_relative_to(Path(ws).resolve()) or path.parent != root
            or path.resolve(strict=True) != path or path.is_symlink() or not path.is_file()
            or path.stat().st_nlink != 1 or path.stat().st_size > 16 << 20):
        raise ValueError('retained join requires the original private completion owner')
    raw = path.read_bytes()
    record = json.loads(raw)
    job = record.get('job_id')
    if (not isinstance(job, str) or not job
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in job)
            or path.name != hashlib.sha256(job.encode()).hexdigest() + '.json'
            or record.get('schema') != 'private_promotion_completion_v1'
            or record.get('workspace') != str(Path(ws).resolve())
            or record.get('source_identity_verified') is not True
            or record.get('cert_tier') != 'L3' or record.get('instrument') != instrument
            or record.get('identity') != proof['L2']['execution_digest']
            or record.get('source_digest') != proof['common']['submission_digest']):
        raise ValueError('retained completion source, execution or original instrument differs')
    verdict = record.get('verdict')
    if (not isinstance(verdict, dict)
            or hashlib.sha256(completions._encoded(verdict)).hexdigest() != record.get('verdict_sha256')):
        raise ValueError('retained completion verdict changed')
    rows = [row for row in verdict.get('per_capsule', ())
            if isinstance(row, dict) and row.get('capsule') == proof['common']['capsule']]
    if (len(rows) != 1 or rows[0].get('pass') is not True or rows[0].get('barrier_tier') != 'L3'
            or rows[0].get('barrier_status') != 'pass'
            or rows[0].get('execution_digest') != proof['L3']['execution_digest']
            or path.read_bytes() != raw):
        raise ValueError('retained completion does not bind this passed physical execution')


def _apply_verified(ws, proof, *, instrument, completion):
    from . import promotion as P
    name, identity = proof['common']['capsule'], proof['L2']['execution_digest']
    state = P._tier_state(ws)
    _completion_custody(ws, completion, proof, instrument)
    pending = P._slots_ro(state, name, 'L3').get(identity)
    loop = P._slots_ro(state, name, 'L2').get(identity)
    if (not instrument or not isinstance(pending, dict) or pending.get('instrument') != instrument
            or pending.get('status') != 'pending' or not isinstance(loop, dict)
            or loop.get('status') != 'pass' or loop.get('digest') != pending.get('digest')
            or pending.get('digest') != proof['common']['submission_digest']
            or loop.get('execution_digest') != identity or pending.get('execution_digest') != identity
            or any(entry.get('tier_boundary') not in (None, proof['L2']) for entry in (loop, pending))):
        raise ValueError('retained join has no matching native pending/loop records')
    # Upgrade only the missing observation for these already recorded executions.
    # The original certificates and original result bytes keep their identities.
    for tier, entry in (('L2', loop), ('L3', pending)):
        entry = dict(entry, tier_boundary=proof['L2'])
        P._slots(state, name, tier)[identity] = entry
        mirror = state.get(name, {}).get(tier)
        if isinstance(mirror, dict) and P.record_identity(mirror) == identity:
            state[name][tier] = entry
    P._save_tier_state(ws, state)
    return P.record_cert.__wrapped__(ws, {'per_capsule': [{
        'capsule': name, 'pass': True, 'barrier_tier': 'L3', 'barrier_status': 'pass',
        'execution_digest': proof['L3']['execution_digest'], 'tier_boundary': proof['L3'],
    }]}, 'L3', identity=identity, source_identity_verified=True)
