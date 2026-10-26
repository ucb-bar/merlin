"""Public workload search roles, independent of correctness and physical gates.

Roles allocate optimization attention. They never select compiler behavior,
waive a test, authorize gSIM, or assert that a timing prediction is supported.
Operator overrides bind the exact public declaration and carry a reason.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PRIMARY = frozenset(('large_kernel', 'connected', 'repeated_inference', 'whole_chain'))
ROLES = PRIMARY | {'correctness', 'calibration', 'diagnostic_control', 'unclassified'}


def classify(member, *, selection=None):
    """Use declared source volume or an explicit source-bound search selection.

    The volume threshold is a transparent experiment policy, not a hardware
    capacity, latency estimate, or universal definition of relevance. Small but
    important repeated regions can be selected explicitly by their source pin.
    """
    selection = selection or {}
    threshold = selection.get('minimum_input_elements', 65536)
    if type(threshold) is not int or threshold < 1:
        raise ValueError('workload role threshold must be a positive element count')
    cohort = member.get('cohort')
    role, basis = 'unclassified', 'no declared search role'
    if cohort == 'original_correctness':
        role, basis = 'correctness', 'original mandatory correctness roster'
    elif cohort == 'calibration':
        role, basis = 'calibration', 'selected model-mechanism evidence pool'
    elif cohort in ('scalability_performance', 'connected_programs'):
        shapes = [t.get('shape') for t in member.get('input_tensors', ())]
        known = bool(shapes) and all(isinstance(s, list) and all(type(d) is int and d > 0 for d in s)
                                     for s in shapes)
        volume = sum(math.prod(s) for s in shapes) if known else None
        if known:
            role = 'large_kernel' if volume >= threshold else 'diagnostic_control'
            basis = f'declared input elements {volume}; selected threshold {threshold}'
            if role == 'large_kernel' and (cohort == 'connected_programs'
                                          or member.get('declared_operation') == 'model'):
                role = 'connected'
    name = member.get('name')
    override = selection.get('members', {}).get(name)
    if override is not None:
        if (not isinstance(override, dict) or set(override) != {'source_sha256', 'role', 'reason'}
                or override['source_sha256'] != member.get('source_sha256')
                or override['role'] not in ROLES or not isinstance(override['reason'], str)
                or not override['reason'].strip()):
            raise ValueError('workload search override requires the exact public source and a reason')
        # Original physical/correctness and calibration roles cannot be moved
        # into optimization rankings through an advisory metadata override.
        if cohort in ('original_correctness', 'calibration') and override['role'] != role:
            raise ValueError('workload search override cannot replace original correctness/calibration roles')
        role, basis = override['role'], override['reason']
    return dict(workload_role=role, role_basis=basis,
                performance_priority='primary' if role in PRIMARY else 'diagnostic',
                optimization_goal=('reduce complete elapsed cycles after correct execution' if role in PRIMARY
                                   else 'pass and retain matching evidence; optimize only as a diagnostic'),
                engine_authorization_changed=False)


def filter_primary(cohorts, roles):
    """Retain diagnostic cases outside the primary optimization ranking."""
    result = {key: list(value) for key, value in cohorts.items()}
    controls = [name for name in result['primary'] if roles.get(name) not in PRIMARY]
    result['primary'] = [name for name in result['primary'] if roles.get(name) in PRIMARY]
    return result, controls


def workspace_roles(workspace):
    """Read only the staged public planning inventory, never result labels.

    Legacy consumers without a role-bearing inventory retain their diagnostic
    view. A prospective role-bearing inventory treats absent cases as unranked.
    """
    path = Path(workspace) / 'PUBLIC_CAPABILITIES.json'
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16 << 20:
        return None
    record = json.loads(path.read_bytes())
    if record.get('schema') != 'merlin.public_capability_inventory.v1':
        return None
    members = record.get('members') or {}
    if not any('workload_role' in m for m in members.values()):
        return None
    return {name: member.get('workload_role', 'unclassified') for name, member in members.items()}
