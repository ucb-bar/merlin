"""Independent outcome axes and an actionable priority; grants no grade credit."""


def performance_signals(performance, runtime=None):
    """Project existing scoped counts and structural floors, never price or add them."""
    roofline = performance.get('structural_roofline') or {}
    signals = {
        'timing_status': performance.get('status', 'UNKNOWN'),
        'timing_mode': performance.get('timing_mode'),
        'timing_qualified': performance.get('timing_qualified', False),
        'model_sha256': performance.get('model_sha256'),
        'scope_sha256': performance.get('scope_sha256'),
        'comparison_domain_sha256': performance.get('comparison_domain_sha256'),
        'scope': performance.get('scope'), 'scope_excludes': performance.get('scope_excludes', []),
        'estimated_whole_soc_cycles': performance.get('estimated_whole_soc_cycles'),
        'roofline': {key: roofline.get(key) for key in
                    ('status', 'useful_macs', 'structural_compute_floor_cycles', 'convergence_eligible')},
        'structural_floor_to_estimate_ratio': performance.get('structural_floor_to_estimate_ratio'),
        'roofline_missing_terms': (roofline.get('missing_terms') or [])[:8],
        'scoped_counts_status': 'UNKNOWN', 'scoped_counts': {},
        'limits': ['Structural floors are not measured or attainable whole-call performance.',
                   'Counts and requested bytes are not cycles or physical bandwidth.',
                   'Whole-trace resource busy totals are excluded; no stall decomposition or overlap claim.'],
    }
    runtime = runtime or {}
    records = runtime.get('observations') or {}
    pinned = bool(performance.get('scope_sha256') and performance.get('model_sha256'))
    for name in ('cpu_sync', 'schedule_queues'):
        pins = (records.get(name) or {}).get('pins') or {}
        pinned = pinned and all(pins.get(key) == performance.get(key) for key in ('scope_sha256', 'model_sha256'))
    counts = performance.get('feature_counts') or {}
    emitted = (records.get('schedule_queues') or {}).get('emitted') or {}
    completed_counts = (type(emitted.get('commands_total')) is int
                        and emitted['commands_total'] == counts.get('target.commands'))
    if (runtime.get('selected_window') is not None and runtime.get('feature_counts_complete') is True
            and pinned and completed_counts):
        # Native producer has already checked the counter-window/ELF/model joins.
        allowed = ('target.commands', 'target.fences', 'target.class.config', 'target.class.load',
                   'target.class.preload', 'target.class.store', 'target.useful_shape_macs',
                   'target.nominal_padded_macs', 'device.requested_read_bytes', 'device.requested_write_bytes',
                   'cpu.integer', 'cpu.branch', 'cpu.load', 'cpu.store')
        signals['scoped_counts'] = {key: counts[key] for key in allowed
                                   if type(counts.get(key)) is int and counts[key] >= 0}
        signals['scoped_counts_status'] = 'observed' if signals['scoped_counts'] else 'UNKNOWN'
        signals['selected_window'] = runtime['selected_window']
    return signals


def project(result, performance):
    numeric = (result.get('numeric') or {}).get('status', 'UNKNOWN')
    trace = (result.get('trace_check') or {}).get('status', 'UNKNOWN')
    formal = (result.get('formal_verification') or {}).get('status', 'UNKNOWN')
    tiers = result.get('tiers') or {}
    rtl = {name: row.get('status', 'UNKNOWN') for name, row in tiers.items()
           if isinstance(row, dict) and row.get('derived_from_rtl') is True}
    missing_rtl = [name for name, status in rtl.items() if status in ('unavailable', 'skipped', 'UNKNOWN')]
    stage = performance.get('unavailability_stage')
    if numeric in ('fail', 'failed'):
        priority, owner = 'numerical_mismatch', 'compiler'
        action = 'Inspect the first mismatching output and the affected named source case.'
    elif numeric != 'pass' and result.get('failure'):
        priority, owner = 'compiler_or_runtime_failure', 'compiler_or_operator'
        action = 'Resolve the concrete emission or execution failure before performance comparison.'
    elif trace in ('fail', 'failed', 'invalid_binding', 'unavailable_fields', 'unavailable_record'):
        priority, owner = 'trace_or_structural_evidence', 'compiler_or_checker'
        action = 'Inspect the named trace binding or unresolved structural requirement; preserve passing numerics.'
    elif formal in ('fail', 'failed'):
        priority, owner = 'formal_obligation', 'compiler_or_checker'
        action = 'Resolve the reported formal obligation without changing the numerical contract.'
    elif stage:
        priority, owner = 'timing_infrastructure', 'operator'
        action = 'Repair the reported timing feedback stage; extra RTL simulations do not repair this failure.'
    elif performance.get('estimated_whole_soc_cycles') is None and numeric == 'pass':
        priority, owner = 'timing_model_or_observation', 'operator'
        action = 'Inspect the exact timing observation reason and unresolved model events before collecting labels.'
    elif numeric == 'pass' and performance.get('estimated_whole_soc_cycles') is not None:
        priority, owner = 'performance_search', 'compiler'
        action = 'Compare legal numerically passing variants under this same model and scope; keep estimates provisional.'
    else:
        priority, owner = 'numerical_evidence_missing', 'compiler_or_operator'
        action = 'Obtain the affected named case numerical result before drawing a performance conclusion.'
    return dict(numerical_status=numeric, trace_status=trace, formal_status=formal,
                rtl_tiers=rtl, unobserved_rtl_tiers=missing_rtl,
                timing_status=performance.get('status', 'UNKNOWN'), timing_failure_stage=stage,
                next_priority=priority, next_owner=owner, next_action=action,
                missing_rtl_is_numerical_failure=False, additional_gsim_requested=False,
                certification_unchanged=True,
                rtl_policy='Unobserved required RTL tiers remain uncertified. Original required gSIM checks remain mandatory; additional calibration gSIM needs an admitted mechanism question.')
