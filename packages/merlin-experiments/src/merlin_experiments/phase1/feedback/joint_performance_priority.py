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


def _summarize(rows):
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


def summarize(rows, *, cohorts=None, roles=None):
    """Keep the selected large-workload ranking primary in joint campaigns.

    ``cohorts`` is produced by native scope selection from the verified public
    suite and operator policy. Per-result labels cannot select a cohort. Generic
    non-joint callers retain the original single-cohort projection.
    """
    if cohorts is None:
        return _summarize(rows)
    rows = list(rows)
    controls = []
    if roles is not None:
        from ..workload_roles import filter_primary
        cohorts, controls = filter_primary(cohorts, roles)
    lanes = ('primary', 'correctness_regression', 'calibration_diagnostics')
    membership = {}
    valid = isinstance(cohorts, dict) and set(cohorts) == set(lanes)
    for lane in lanes:
        names = cohorts.get(lane) if isinstance(cohorts, dict) else None
        if (not isinstance(names, (list, tuple)) or any(type(name) is not str or not name for name in names)
                or len(set(names)) != len(names)):
            valid = False
            names = ()
        membership[lane] = set(names)
    overlap = set()
    for index, lane in enumerate(lanes):
        for other in lanes[index + 1:]:
            overlap.update(membership[lane] & membership[other])
    # A malformed roster provides no primary rank. Overlapping names retain
    # their original observations below, explicitly unclassified.
    selected = {lane: [] for lane in lanes}
    unclassified = []
    for row in rows:
        name = row.get('capsule', row.get('name'))
        if name in controls:
            continue
        matches = [lane for lane in lanes if name in membership[lane]] if valid else []
        if len(matches) == 1 and name not in overlap:
            selected[matches[0]].append(row)
        else:
            priority = (row.get('performance') or {}).get('search_priority') or {}
            unclassified.append(dict(capsule=name, reason='missing, invalid or overlapping trusted cohort',
                                     observation_priority=dict(priority)))
    result = _summarize(selected['primary'])
    # Correctness failures across ALL cohorts remain visible before optimization.
    result['correctness_first'] = _summarize(rows)['correctness_first']
    result.update(
        primary_cohort='selected_representative_workloads' if roles is not None else 'selected_public_scalability',
        diagnostic_controls=_summarize([row for row in rows
            if row.get('capsule', row.get('name')) in controls]),
        cohort_selection_status='valid' if valid and not overlap else 'invalid_or_overlapping',
        correctness_regression=_summarize(selected['correctness_regression']),
        calibration_diagnostics=_summarize(selected['calibration_diagnostics']),
        unclassified=unclassified,
        observed_members={lane: sorted({row.get('capsule', row.get('name')) for row in selected[lane]})
                          for lane in lanes},
        policy='Primary performance ranking contains only the selected public scalability cohort. '
               'Source-bound workload roles exclude small diagnostic controls when supplied. '
               'Original capsules retain correctness/regression feedback; calibration cases supply diagnostics. '
               'Useful small-case improvements remain permitted. All original correctness gates are unchanged.',
    )
    return result
