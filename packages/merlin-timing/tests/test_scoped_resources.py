"""Synthetic report arithmetic tests; no hardware/model qualification."""
import copy
import pytest
from merlin_timing import scoped_resources as S
from merlin_timing import contracts as C


def report():
    pins=dict(scope_sha256='a'*64,model_sha256='b'*64,elf_sha256='c'*64)
    def row(rid,name,capacity,busy,active):
        return dict(resource_id=rid,name=name,capacity=capacity,busy_unit_cycles=busy,active_cycles=active,
                    queued_entry_cycles=0,queue_active_cycles=0,peak_active_units=capacity,peak_queue_entries=0)
    return dict(schema='merlin.annotated_spike_report.v2',profile='phase1-public',**pins,
        estimated_cycles=10,instructions=100,execution_completed=True,timing_complete=True,unresolved=[],
        functional_status='PASS',raw_functional_exit_code=0,scope_binding_sha256='d'*64,
        scope_binding=dict(begin_pc=16,end_pc=32,occurrence=0),
        counter_windows=[dict(begin_pc=16,end_pc=32,instructions=100,estimated_cycles=10)],
        feature_observation=dict(schema='merlin.spike_feature_observation.v1',complete=True,scope_instructions=100,
            counted_cpu_instructions=100,scope_kind='counter_pc_pair',window_index=0,primary_cpu_features=['cpu.integer'],
            unresolved=[],qualification='not_established'),
        feature_counts={'cpu.integer':100,'target.useful_shape_macs':1000,'target.nominal_padded_macs':2000,
                        'device.requested_read_bytes':64,'device.requested_write_bytes':32},
        scoped_resource_activity=dict(schema=S.ACTIVITY_SCHEMA,source_kind='modeled',unit='model_cycles',complete=True,
            **pins,scope_binding_sha256='d'*64,begin_cycle=5,end_cycle=15,
            resources=[row(0,'compute.capacity',2,20,10),row(1,'load.capacity',1,5,5)],
            overlaps=[dict(left=0,right=1,active_cycles=5)],meaning='modeled reservations'))


def test_capacity_normalization_and_overlap_not_added_to_latency():
    source=report();before=copy.deepcopy(source)
    value=S.project(source,structural={'structural_compute_floor_cycles':5})
    assert value['estimated_cycles_same_window']==10
    assert value['resources'][0]['capacity_normalized_activity']==1
    assert value['resources'][1]['capacity_normalized_activity']==.5
    assert value['overlaps'][0]['fraction_of_window']==.5
    assert value['compute']['useful_fraction_of_padded_work']==.5
    assert value['compute']['structural_floor_to_estimate_ratio']==.5
    assert value['traffic'][0]['bytes_per_model_cycle']==6.4
    assert value['host']['preparation_cycles'] is None
    assert not value['resource_cycles_additive'] and not value['qualification_issued']
    assert source==before


def test_missing_interval_producer_keeps_structural_and_requested_rates():
    source=report();del source['scoped_resource_activity']
    source['cycle_breakdown']={'scope':'whole_trace','resources':[{'busy_unit_cycles':10**9}]}
    value=S.project(source)
    assert value['activity_status']=='UNKNOWN' and not value['resources']
    assert value['compute']['useful_fraction_of_padded_work']==.5
    assert not value['traffic'][0]['physical_bandwidth']


@pytest.mark.parametrize('change',['scope','window','capacity','duplicate','overlap','private','padding'])
def test_wrong_scope_or_impossible_activity_fails_closed(change):
    source=report();activity=source['scoped_resource_activity']
    if change=='scope':activity['scope_sha256']='f'*64
    elif change=='window':activity['end_cycle']=16
    elif change=='capacity':activity['resources'][0]['busy_unit_cycles']=21
    elif change=='duplicate':activity['resources'].append(dict(activity['resources'][0]))
    elif change=='overlap':activity['overlaps'][0]['active_cycles']=6
    elif change=='private':source['profile']='historical-private'
    else:source['feature_counts']['target.useful_shape_macs']=3000
    with pytest.raises(C.ContractError):S.project(source)


def test_incomplete_interval_capture_never_looks_zero_utilized():
    source=report();source['scoped_resource_activity']['complete']=False
    value=S.project(source)
    assert value['activity_status']=='UNKNOWN' and value['resources']==[]
    assert value['compute']['useful_fraction_of_padded_work']==.5


def test_compact_projection_is_bounded_and_keeps_source_pins():
    value=S.project(report())
    value['resources']=[dict(value['resources'][0],resource_id=i) for i in range(50)]
    small=S.compact(value)
    assert len(small['resources'])==16 and small['resource_rows_omitted']==34
    assert small['scope_sha256']==value['scope_sha256']


def test_missing_window_is_refused_instead_of_index_error():
    source=report();source['counter_windows']=[]
    with pytest.raises(C.ContractError):S.project(source)


def test_failed_functional_execution_cannot_offer_activity():
    source=report();source['functional_status']='FAIL'
    with pytest.raises(C.ContractError):S.project(source)
