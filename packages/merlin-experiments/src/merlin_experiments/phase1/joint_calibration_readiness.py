"""Cheap declared-panel status, before native sensitivity or physical work.

Only the existing completed candidate registry can supply variants. This report
adds no witness, fit price, protocol qualification, or physical authority.
"""
from __future__ import annotations

from .feedback import joint_performance as PERF
from .feedback.joint_timing_selection import choose
from .joint_calibration_admission import assert_allowed
from ..phase2.contracts import document_sha256


def panel(profile, selection, rows, *, witnesses):
    """Return per-member readiness and admissible registered row IDs.

    Denial is evaluated even with zero candidate variants. Invalid/missing review
    evidence raises instead of being misclassified as a need for more samples.
    Actual native sensitivity, original ELF custody and budget gates remain in
    run_round. No physical labels are read here.
    """
    from .joint_calibration import _capsule
    declared = profile['members']
    if len(declared) > 256 or len(rows) > 2048:
        raise ValueError('calibration readiness exceeds bounded panel')
    by_name = {}
    for row in rows:
        cap = _capsule(row['capsule'])
        name = cap.get('name')
        member = selection.get('members', {}).get(name)
        if (name in declared and member and cap.get('kind') != 'model'
                and cap.get('label') in {'public', 'dev'}
                and PERF.capsule_identity(cap) == member['capsule_sha256']
                and row['source_group_sha256'] == declared[name]['source_group_sha256']):
            grade = PERF._read(row['result'])
            tier = grade.get('tiers', {}).get('L2', {})
            if (grade.get('numeric', {}).get('status') == 'pass'
                    and tier.get('status') == 'pass' and tier.get('engine') == 'spike'
                    and (grade.get('compiler_source') or grade.get('candidate_native_execution'))):
                by_name.setdefault(name, []).append(row)
    runtime_status = {}
    result, allowed = [], []
    for name, member in sorted(declared.items()):
        source = selection.get('members', {}).get(name)
        variants = by_name.get(name, [])
        compilers = sorted({r['compiler_sha256'] for r in variants})
        reason = None
        if source is None:
            status = 'not_in_current_source_selection'
        elif name not in witnesses:
            status = 'outside_physical_witness_roster'
        else:
            runtime = choose(selection, source['capsule_sha256'])['runtime']
            key = document_sha256(runtime)
            if key not in runtime_status:
                try:
                    assert_allowed(dict(runtime=runtime, parameters=profile['parameters']))
                    runtime_status[key] = None
                except ValueError as error:
                    if not str(error).startswith('blocked_structural_model_admission:'):
                        raise
                    runtime_status[key] = str(error)
            reason = runtime_status[key]
            if reason:
                status = 'blocked_structural_model'
            elif not variants:
                status = 'pending_passing_candidate_spike_product'
            elif len(compilers) < 2:
                status = 'pending_same_source_compiler_variant'
            else:
                status = 'ready_for_native_identifiability_review'
            if reason is None:
                allowed.extend(r['id'] for r in variants)
        result.append(dict(name=name, source_group_sha256=member['source_group_sha256'],
            status=status, reason=reason, registered_products=len(variants),
            compiler_sha256s=compilers, dispatch_authorized=False))
    return dict(schema='merlin.native_calibration_panel_readiness.v1', members=result,
        admissible_registry_ids=allowed,
        ordering='cheap source/numerical admission; source-disjoint variant pairs; native identifiability; train labels; held validation; reviewed epoch',
        candidate_custody='All variants are completed products of the one evolving author compiler, pinned by exact compiler hash; canonical ELF custody is reopened before measurement.',
        physical_measurements='existing automatic run_round only, within selected witness roster and cumulative budget; never a panel sweep',
        fit_authorized=False, dispatch_authorized=False)
