"""Advisory search order from public, matched observations; no grade authority."""
from __future__ import annotations

import math

_BINDING = ('capsule_sha256', 'facts_sha256', 'selection_sha256', 'model_sha256',
            'comparison_domain_sha256', 'scope_sha256')
_GROUP = ('facts_sha256', 'selection_sha256', 'model_sha256', 'comparison_domain_sha256', 'scope')


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def project(result, performance):
    """Describe headroom, keeping provisional ordering separate from qualified ranking.

    Invocation counts must come from a trusted public projection with the same
    source pin. Missing impact uses an explicit unit weight, not a model-wide
    invocation estimate. Confidence is categorical; no invented probability.
    """
    value = dict(status='UNKNOWN', reason='complete correctness required',
                 qualified_advisory_score=None, diagnostic_headroom=None, gap_ratio=None,
                 invocation_count=None, invocation_weight=1,
                 workload_impact='unknown; unit invocation weight',
                 confidence='unknown', promotion_authority=False,
                 additional_gsim_requested=False)
    if ((result.get('numeric') or {}).get('status') != 'pass'
            or ((result.get('tiers') or {}).get('L2') or {}).get('status') != 'pass'
            or (result.get('trace_check') or {}).get('status') != 'pass'
            or (result.get('formal_verification') or {}).get('status') in ('fail', 'failed')
            or result.get('failure')):
        value['status'] = 'CORRECTNESS_FIRST'
        return value
    value['reason'] = 'complete model and matched bound required'
    binding = performance.get('structural_comparison_binding') or {}
    if (any(not _pin(performance.get(k)) or performance[k] != binding.get(k) for k in _BINDING)
            or performance.get('complete_event_accounting') is not True
            or performance.get('status') not in ('PROVISIONAL', 'QUALIFIED')):
        return value
    cycles = performance.get('estimated_whole_soc_cycles')
    floor = (performance.get('structural_roofline') or {}).get('structural_compute_floor_cycles')
    if not _positive(cycles) or not _positive(floor) or cycles < floor:
        value['reason'] = 'missing, invalid or refuted structural lower bound'
        return value
    weight = performance.get('public_workload_weight') or {}
    count = weight.get('invocation_count')
    if (type(count) is int and 0 < count <= 2**53
            and weight.get('capsule_sha256') == performance['capsule_sha256']
            and _pin(weight.get('public_source_sha256'))):
        value.update(invocation_count=count, invocation_weight=count,
                     workload_impact='source-bound public invocation count')
    headroom = (cycles - floor) * value['invocation_weight']
    ratio = cycles / floor
    if not math.isfinite(headroom) or not math.isfinite(ratio):
        value['reason'] = 'non-finite headroom'
        return value
    qualified = (performance.get('status') == 'QUALIFIED'
                 and performance.get('timing_qualified') is True
                 and performance.get('timing_mode') == 'tuned')
    value.update(status='QUALIFIED_ADVISORY' if qualified else 'PROVISIONAL_DIAGNOSTIC',
                 confidence='qualified covered model' if qualified else 'provisional model; structural uncertainty',
                 reason='structural compute floor; attainable whole-system envelope not established here',
                 gap_ratio=ratio, estimated_cycles=cycles, structural_bound_cycles=floor,
                 diagnostic_headroom=headroom,
                 qualified_advisory_score=headroom if qualified else None,
                 score_basis='invocation weight * (estimated cycles - structural compute floor)',
                 confidence_weight='qualified and provisional observations ordered separately',
                 comparison_group={key: performance.get(key) for key in _GROUP})
    return value


def summarize(rows):
    """Order only comparable observations; changed/failed correctness stays first."""
    groups = {}
    correctness, unknown = [], []
    for row in rows:
        priority = (row.get('performance') or {}).get('search_priority') or {}
        name = row.get('capsule', row.get('name'))
        item = dict(capsule=name, **priority)
        status = priority.get('status')
        if status == 'CORRECTNESS_FIRST':
            correctness.append(item)
        elif status in ('QUALIFIED_ADVISORY', 'PROVISIONAL_DIAGNOSTIC'):
            group = priority['comparison_group']
            key = tuple(group.get(k) for k in _GROUP)
            output = groups.setdefault(key, dict(comparison_group=group, qualified=[], provisional_diagnostics=[]))
            lane = 'qualified' if status == 'QUALIFIED_ADVISORY' else 'provisional_diagnostics'
            output[lane].append(item)
        else:
            unknown.append(dict(capsule=name, reason=priority.get('reason', 'no complete timing feedback')))
    for group in groups.values():
        for lane in ('qualified', 'provisional_diagnostics'):
            group[lane].sort(key=lambda row: (-row['diagnostic_headroom'], -row['gap_ratio'], str(row['capsule'])))
    return dict(schema='merlin.joint_performance_priority.v1', advisory_only=True,
                correctness_first=correctness, comparison_groups=list(groups.values()), unranked=unknown,
                policy='Repair changed-case correctness first. Search larger weighted headroom within each '
                       'matched model group; provisional order is diagnostic, not certified performance.',
                missing_workload_coverage='This report covers observed rows only; omitted cases remain pending.',
                promotion_authority=False, additional_gsim_requested=False)
