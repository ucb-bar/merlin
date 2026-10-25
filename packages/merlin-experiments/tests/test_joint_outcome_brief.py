from merlin_experiments.phase1.feedback.joint_outcome_brief import project


def test_withheld_rtl_does_not_override_numeric_and_timing_failure():
    result={'numeric':{'status':'pass'},'tiers':{'L3':{'derived_from_rtl':True,'status':'unavailable'}},
            'trace_check':{'status':'pass'}}
    value=project(result,{'status':'UNKNOWN','unavailability_stage':'source_binding'})
    assert value['numerical_status']=='pass'
    assert value['rtl_tiers']=={'L3':'unavailable'}
    assert value['formal_status']=='UNKNOWN'
    assert value['next_priority']=='timing_infrastructure'
    assert not value['additional_gsim_requested']
    assert value['certification_unchanged']


def test_trace_failure_and_numerical_failure_remain_separate():
    result={'numeric':{'status':'pass'},'trace_check':{'status':'unavailable_fields'}}
    value=project(result,{'status':'UNKNOWN'})
    assert value['numerical_status']=='pass'
    assert value['next_priority']=='trace_or_structural_evidence'
    result['numeric']['status']='fail'
    assert project(result,{})['next_priority']=='numerical_mismatch'


def test_provisional_search_does_not_need_all_rtl_rows():
    result={'numeric':{'status':'pass'},'trace_check':{'status':'pass'},
            'tiers':{'L3':{'derived_from_rtl':True,'status':'unavailable'}}}
    value=project(result,{'status':'PROVISIONAL','estimated_whole_soc_cycles':123})
    assert value['next_priority']=='performance_search'
    assert value['unobserved_rtl_tiers']==['L3']


def test_unknown_or_refuted_model_does_not_become_search_score_or_authoring_barrier():
    result = {'numeric': {'status': 'pass'}, 'trace_check': {'status': 'pass'}}
    for status in ('UNKNOWN', 'REFUTED'):
        value = project(result, {'status': status, 'estimated_whole_soc_cycles': 123,
                                 'reason': 'cold output memory is unresolved',
                                 'unresolved_events': ['cold line']})
        assert value['next_priority'] == 'timing_model_or_observation'
        assert value['compiler_work_can_continue'] is True
        assert value['timing_reason'] == 'cold output memory is unresolved'
        assert value['unresolved_model_events'] == ['cold line']
        assert value['additional_gsim_requested'] is False


def test_compact_signals_keep_partial_roofline_and_only_proven_scoped_counts():
    from merlin_experiments.phase1.feedback.joint_outcome_brief import performance_signals
    perf = {'status': 'PROVISIONAL', 'timing_mode': 'untuned', 'timing_qualified': False,
            'estimated_whole_soc_cycles': 400, 'scope_sha256': 's', 'model_sha256': 'm',
            'structural_floor_to_estimate_ratio': 0.25,
            'structural_roofline': {'status': 'partial', 'structural_compute_floor_cycles': 100,
                                   'convergence_eligible': False, 'missing_terms': ['host costs']},
            'feature_counts': {'target.commands': 10, 'target.fences': 9, 'cpu.integer': 20},
            'modeled_resource_activity': {'scope': 'whole_trace', 'busy_unit_cycles': 9999}}
    assert performance_signals(perf)['scoped_counts_status'] == 'UNKNOWN'
    runtime = {'selected_window': 0, 'feature_counts_complete': True, 'observations': {
        name: {'pins': {'scope_sha256': 's', 'model_sha256': 'm'},
               'emitted': {'commands_total': 10} if name == 'schedule_queues' else {}}
        for name in ('cpu_sync', 'schedule_queues')}}
    value = performance_signals(perf, runtime)
    assert value['scoped_counts']['target.fences'] == 9
    assert value['roofline']['status'] == 'partial'
    assert value['timing_mode'] == 'untuned' and value['timing_qualified'] is False
    assert '9999' not in str(value)
    runtime['observations']['schedule_queues']['pins']['scope_sha256'] = 'different'
    assert performance_signals(perf, runtime)['scoped_counts_status'] == 'UNKNOWN'
    runtime['observations']['schedule_queues']['pins']['scope_sha256'] = 's'
    runtime['observations']['schedule_queues']['emitted'] = {}
    assert performance_signals(perf, runtime)['scoped_counts_status'] == 'UNKNOWN'


def test_ordinary_brief_exposes_shared_same_roi_projection_without_runtime_repricing():
    from merlin_experiments.phase1.feedback.joint_outcome_brief import performance_signals
    from merlin_timing.optimization_feedback import project as optimization_feedback
    from merlin_timing.scoped_resources import SCHEMA
    perf={'scope_sha256':'a'*64,'model_sha256':'b'*64,'elf_sha256':'c'*64,'estimated_whole_soc_cycles':10,
          'structural_roofline':{'useful_macs':1000,'structural_peak_macs_per_cycle':256},
          'scoped_resource_feedback':{'schema':SCHEMA,'status':'modeled','source_kind':'modeled',
              'scope_sha256':'a'*64,'model_sha256':'b'*64,'elf_sha256':'c'*64,
              'estimated_cycles_same_window':10,'compute':{'useful_macs':1000}},
          'modeled_resource_activity':{'scope':'whole_trace','busy_unit_cycles':10**12}}
    value=performance_signals(perf)
    assert value['same_roi_optimization']==optimization_feedback(perf)
    assert value['same_roi_optimization']['work']['compute_ceiling_attainment']==1000/2560
    assert value['same_roi_optimization']['physical']['array_occupancy'] is None
    assert str(10**12) not in str(value)
