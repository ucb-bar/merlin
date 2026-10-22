"""Independent outcome axes and an actionable priority; grants no grade credit."""


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
                rtl_policy='Unobserved required RTL tiers remain uncertified. Use gSIM only for an admitted mechanism question, not to clear every development row.')
