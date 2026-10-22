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
