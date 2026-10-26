"""Passive calibration intake from the original background completion callback.

No observer, simulator, fitting or certificate issuance occurs here. The same
selected public registry used by explicit selfchecks retains these products.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.runtime import _native_files as F

from .rolling_scalability import RESULT_MAX_BYTES


def register(result, *, result_path, compiler, capsules_root, contract):
    from .. import joint_control
    from . import joint_performance as performance
    from merlin.targetgen.capsule_common import load_capsule

    state = joint_control.selected()
    if state is None or not state['policy'].get('native_calibration'):
        return None
    from ..joint_calibration_eligibility import eligible
    if not eligible(result):
        return None
    tier = result.get('tiers', {}).get('L2', {})
    if (result.get('label') not in {'public', 'dev'}
            or result.get('numeric', {}).get('status') != 'pass'
            or tier.get('status') != 'pass' or tier.get('engine') != 'spike'):
        return None
    name = result.get('capsule')
    if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('completed calibration capsule identity is malformed')
    path = F.canonical(Path(result_path))
    if path.name != 'capsule_result.json' or path.parent.name != name:
        raise ValueError('completed calibration result has another native owner')
    from .declared_bounds import result_bytes

    bound = result_bytes(load_capsule(Path(capsules_root) / name, contract=contract), RESULT_MAX_BYTES)
    before = F.pin(path, max_bytes=bound)
    durable = json.loads(path.read_bytes())
    if durable != json.loads(json.dumps(result, allow_nan=False)):
        raise ValueError('completed calibration row differs from its durable result')
    source = result.get('compiler_source') or {}
    # Whole-model execution uses a different product ABI. Its existing explicit
    # intake remains separate until it supplies this original product identity.
    if not source:
        return None
    if (source.get('status') != 'stable'
            or source.get('candidate_sha256') != hash_tree(Path(compiler))['sha256']):
        raise ValueError('completed calibration compiler differs from its frozen source')
    capsule = load_capsule(Path(capsules_root) / name, contract=contract)
    if capsule.get('label') not in {'public', 'dev'} or capsule.get('name') != name:
        raise ValueError('completed calibration input is not its selected public capsule')
    choice = performance.selected()
    if choice is None:
        return None
    capsule = performance.selected_capsule(choice[1], capsule)
    if F.pin(path, max_bytes=bound) != before:
        raise ValueError('completed calibration result changed before intake')
    from .. import joint_calibration_service

    row = {}
    joint_calibration_service.register(row, path.parent / 'generated', result=durable,
                                      compiler=compiler, capsule=capsule)
    return row.get('performance', {})
