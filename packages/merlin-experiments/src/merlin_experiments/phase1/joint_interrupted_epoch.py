"""Explicit operator recovery of interrupted custody; never a completion verdict.

Capture owners before termination. Recovery requires an empty dedicated campaign
cgroup and all captured process identities gone. This module sends no signals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import yaml

from merlin.common.tree_hash import hash_tree
from . import joint_calibration_owner as O

SCHEMA = "merlin.joint_interrupted_epoch.v1"


def _pin(path):
    from .joint_epochs import _read
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _read(path)[1]}


def _process(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "argv_sha256": hashlib.sha256((root / 'cmdline').read_bytes()).hexdigest()}
    except FileNotFoundError:
        return None


def _members(cgroup):
    root = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
    if root.is_symlink() or root != root.resolve() or not root.is_dir():
        raise ValueError('dedicated campaign cgroup missing or redirected')
    return sorted({int(line) for path in root.rglob('cgroup.procs')
                   for line in path.read_text().splitlines()})


def _capture(receipt_path):
    receipt = O.read(receipt_path)
    if (receipt.get('schema') != 'operator.owned_termination_result.v1'
            or receipt.get('applied') is not True or receipt.get('owned_terminal') is not True
            or receipt.get('completion_claimed') is not False
            or receipt.get('unfinished_evaluations') != 'UNKNOWN'):
        raise ValueError('explicit truthful owned termination receipt required')
    O.verify_pin(receipt['plan'])
    plan = O.read(receipt['plan']['path'])
    if plan.get('schema') != 'operator.owned_termination_plan.v1':
        raise ValueError('unknown owner termination plan')
    selected = O.read(plan['selection']['path'])
    O.verify_pin(plan['selection'])
    O.verify_pin(selected['configuration'])
    if O.read(selected['configuration']['path'])['reservation']['cgroup'] != plan['cgroup']:
        raise ValueError('termination plan differs from selected reservation')
    for key in ('native_owner', 'calibration_owner', 'author_summary'):
        O.verify_pin(plan[key])
    if receipt.get('attestation_kind') == 'independent_post_termination_census':
        O.verify_pin(receipt['operator_interruption_receipt'])
    native = O.read(plan['native_owner']['path'])
    calibration = O.read(plan['calibration_owner']['path'])
    argv = native['argv']
    if argv[:2] == ['/usr/bin/taskset', '-c']:
        if argv[2] != ','.join(map(str, selected['taskset_cpus'])):
            raise ValueError('native owner CPU selection changed')
        argv = argv[3:]
    if argv != selected['argv'] or plan['selection']['path'] not in calibration['argv']:
        raise ValueError('termination owners differ from selected campaign')
    if not {native['pid'], calibration['pid']} <= {row['pid'] for row in plan['processes']}:
        raise ValueError('termination plan omits native owners')
    return dict(plan, previous_run=selected['run_dir'], owner_pins=[plan['native_owner'], plan['calibration_owner']])


def _stopped(capture, *, check_cgroup=False):
    O.verify_pin(capture['selection'])
    for pin in capture['owner_pins']:
        O.verify_pin(pin)
        if O.process_identity(O.read(pin['path'])) is not None:
            raise ValueError('previous owner remains alive')
    for previous in capture['processes']:
        current = _process(previous['pid'])
        if current and current['start_ticks'] == previous['start_ticks']:
            raise ValueError('captured author/worker remains alive')
    if check_cgroup:
        allowed = set()
        for row in capture['preserved_residents']:
            current = _process(row['pid'])
            if current:
                if any(current[k] != row[k] for k in ('start_ticks', 'argv_sha256')):
                    raise ValueError('reservation resident identity changed')
                allowed.add(row['pid'])
        if {pid for pid in _members(capture['cgroup']) if _process(pid)} - allowed:
            raise ValueError('previous campaign still has owned or unknown processes')


def _tree(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError('source tree is absent or indirect')
    files = []
    size = 0
    for member in path.rglob('*'):
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise ValueError('source tree contains indirect members')
        if member.is_file():
            size += member.stat().st_size
            files.append(member)
    if len(files) > 10000 or size > 64 * 1024 * 1024:
        raise ValueError('source recovery exceeds bounded compiler inputs')
    return hash_tree(path)['sha256']


def _unclaimed_ledger(pin):
    """Do not transport ambiguous in-flight scheduling claims into a new owner."""
    if pin is None:
        return
    O.verify_pin(pin)
    pending = [O.read(pin['path'])]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get('status') == 'pending':
                raise ValueError('pending tier claims require separate proven reconciliation before interrupted recovery')
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def prepare(previous_run, *, owner_capture, contract_sha256, review_id, output,
            performance_selection_sha256=None):
    """Freeze stopped source, with unfinished work explicitly UNKNOWN."""
    from . import joint_epochs as E
    root = Path(previous_run).resolve(strict=True)
    capture_pin = _pin(owner_capture)
    capture_record = _capture(owner_capture)
    if capture_record['previous_run'] != str(root):
        raise ValueError('owner capture belongs to another campaign')
    _stopped(capture_record, check_cgroup=True)
    if (root / 'joint_campaign_summary.json').exists():
        raise ValueError('completed native runs must use the normal epoch path')
    if not E.stop_reason(root):
        raise ValueError('explicit cooperative epoch stop was not requested')
    if not review_id or len(contract_sha256) != 64 or any(c not in '0123456789abcdef' for c in contract_sha256):
        raise ValueError('explicit reviewed contract and reason required')
    environment = yaml.safe_load((root / 'environment.yaml').read_text())
    state = O.read(root / 'joint_campaign.json')
    if environment.get('backend_origin') != {'schema': 'merlin.empty_backend_origin.v1', 'initial_files': 0} or not environment.get('lineage_id'):
        raise ValueError('original empty compiler lineage missing')
    if state.get('started_at_epoch') is None:
        raise ValueError('original author clock missing')
    _unclaimed_ledger(E._tier_ledger(environment))
    workspace = Path(environment['workspace_path'])
    if not workspace.is_absolute() or workspace != workspace.resolve():
        raise ValueError('workspace is not canonical')
    source = workspace / 'submission'
    if not (source / 'manifest.yaml').is_file():
        raise ValueError('authored compiler manifest missing')
    destination = Path(output).resolve().parent / 'interrupted_source'
    digest = _tree(source)
    shutil.copytree(source, destination)
    if _tree(destination) != digest or _tree(source) != digest:
        raise ValueError('authored source changed during recovery')
    variants = []
    variant_root = workspace / 'variants'
    if variant_root.is_symlink():
        raise ValueError('variant root redirected')
    for variant in sorted(variant_root.iterdir()) if variant_root.exists() else []:
        before = _tree(variant)
        copied = destination.parent / 'interrupted_variants' / variant.name
        shutil.copytree(variant, copied)
        if _tree(copied) != before or _tree(variant) != before:
            raise ValueError('variant changed during recovery')
        variants.append({'name': variant.name, 'path': str(copied), 'sha256': before})
    evidence = []
    for directory in (workspace / 'qa/tier_instrument_history', root / 'rounds'):
        for path in sorted(directory.glob('*.json')):
            evidence.append(_pin(path))
    record = {'schema': SCHEMA, 'previous_run': str(root), 'review_id': review_id,
              'pins': {name: E._read(root / name)[1] for name in ('environment.yaml', 'joint_campaign.json', 'joint_epoch_stop.json')},
              'owner_capture': capture_pin, 'source': str(destination), 'compiler_sha256': digest,
              'new_contract_sha256': contract_sha256, 'lineage_id': environment['lineage_id'],
              'source_run_id': environment['run_id'], 'variants': variants, 'historical_evidence': evidence,
              'interruption': {'status': 'interrupted', 'unfinished_evaluations': 'UNKNOWN',
                               'completion_claimed': False, 'current_credit_granted': False},
              'tier_ledger': E._tier_ledger(environment), 'calibration_history': E._calibration_history(root)}
    if record['tier_ledger'] is None:
        record.pop('tier_ledger')
    if performance_selection_sha256 is not None:
        if len(performance_selection_sha256) != 64 or any(c not in '0123456789abcdef' for c in performance_selection_sha256):
            raise ValueError('invalid reviewed model pin')
        record['performance_selection_sha256'] = performance_selection_sha256
    verify(record)
    O.write(output, record, exclusive=True)
    return record


def verify(record):
    """Reopen stopped-owner/source/evidence custody without minting a verdict."""
    from . import joint_epochs as E
    if record.get('schema') != SCHEMA or not record.get('review_id'):
        raise ValueError('invalid interrupted epoch review')
    O.verify_pin(record['owner_capture'])
    capture_record = _capture(record['owner_capture']['path'])
    if capture_record['previous_run'] != record['previous_run']:
        raise ValueError('interrupted owner capture changed run')
    _stopped(capture_record)
    root = Path(record['previous_run'])
    for name in ('environment.yaml', 'joint_campaign.json', 'joint_epoch_stop.json'):
        if E._read(root / name)[1] != record['pins'][name]:
            raise ValueError('interrupted epoch input changed')
    environment = yaml.safe_load((root / 'environment.yaml').read_text())
    origin = environment.get('backend_origin')
    if origin != {'schema': 'merlin.empty_backend_origin.v1', 'initial_files': 0} or environment.get('lineage_id') != record['lineage_id']:
        raise ValueError('interrupted compiler lineage changed')
    if environment.get('run_id') != record.get('source_run_id'):
        raise ValueError('interrupted source run identity changed')
    if not (Path(record['source']) / 'manifest.yaml').is_file() or _tree(Path(record['source'])) != record['compiler_sha256']:
        raise ValueError('interrupted compiler changed')
    for variant in record['variants']:
        if _tree(Path(variant['path'])) != variant['sha256']:
            raise ValueError('interrupted variant changed')
    for pin in record['historical_evidence']:
        O.verify_pin(pin)
    if record.get('tier_ledger') != E._tier_ledger(environment) or record['calibration_history'] != E._calibration_history(root):
        raise ValueError('interrupted evidence changed')
    _unclaimed_ledger(record.get('tier_ledger'))
    if record.get('interruption') != {'status': 'interrupted', 'unfinished_evaluations': 'UNKNOWN', 'completion_claimed': False, 'current_credit_granted': False}:
        raise ValueError('interruption cannot assert completion or credit')
    state = O.read(root / 'joint_campaign.json')
    if state.get('started_at_epoch') is None:
        raise ValueError('original clock missing')
    return state, origin


def seed_variants(record, workspace):
    """Carry only the same author's exact alternatives, never external answers."""
    rows = []
    destination = Path(workspace) / 'variants'
    for row in record['variants']:
        if not row['name'] or Path(row['name']).name != row['name'] or row['name'] in {'.', '..'}:
            raise ValueError('invalid authored variant name')
        source = Path(row['path'])
        if _tree(source) != row['sha256']:
            raise ValueError('authored alternative changed before handoff')
        if destination.exists() and (destination.is_symlink() or destination != destination.resolve()):
            raise ValueError('authored variant destination redirected')
        copied = destination / row['name']
        shutil.copytree(source, copied)
        if _tree(copied) != row['sha256'] or _tree(source) != row['sha256']:
            raise ValueError('authored alternative changed during handoff')
        rows.append({**row, 'destination': str(copied), 'current_credit_granted': False})
    return rows
