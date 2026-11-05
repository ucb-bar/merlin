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


def test_command_projection_preserves_missing_and_aggregates_only_complete_classes():
    source=report()
    source['feature_counts'].update({'target.commands':4,'target.fences':1,
        'target.class.load':1,'target.class.compute_preloaded':1,
        'target.class.compute_accumulated':1,'target.class.pool_store':1})
    value=S.project(source)
    assert value['commands']['compute']==2 and value['commands']['store']==1
    assert value['commands']['fences']==1 and value['commands']['config'] is None
    source['feature_counts']['target.commands']=5
    value=S.project(source)
    assert value['commands']['compute'] is None and value['commands']['store'] is None


def test_scalar_fence_class_is_separate_from_accelerator_command_census():
    source=report()
    source['feature_counts'].update({'target.commands':7,'target.fences':3,
        'target.class.load':2,'target.class.preload':2,'target.class.compute_ws':2,
        'target.class.store':1,'target.class.fence':3})
    value=S.project(source)
    assert value['commands']['total']==7 and value['commands']['fences']==3
    assert value['commands']['compute']==2 and value['commands']['store']==1
    source['feature_counts']['target.class.fence']=4
    assert S.project(source)['commands']['compute'] is None
    source['feature_counts']['target.class.fence']=3
    source['feature_counts']['target.commands']=8
    assert S.project(source)['commands']['compute'] is None


def test_utilization_views_distinguish_time_capacity_padding_and_queue_units():
    source = report()
    r = source["scoped_resource_activity"]["resources"][1]
    r.update(queued_entry_cycles=12, queue_active_cycles=4, peak_queue_entries=3)
    value = S.project(source, structural=dict(useful_macs=800, structural_peak_macs_per_cycle=256))
    views = value["utilization_views"]
    assert views["compute"]["compute_ceiling_attainment"]["value"] == 800 / 2560
    assert views["compute"]["compute_ceiling_attainment"]["numerator"]["quantity"] == "source_useful_macs"
    assert views["compute"]["useful_fraction_of_padded_work"]["value"] == 0.5
    load = views["resources"][1]
    assert load["capacity_reservation_fraction"] == 0.5
    assert load["reservation_active_fraction"] == 0.5
    assert load["denominators"] == dict(capacity_reservation_unit_cycles=10, active_model_cycles=10)
    assert load["queue"]["average_waiting_entries"] == 1.2
    assert load["queue"]["active_fraction"] == 0.4
    assert load["queue"]["capacity_normalized_pressure"] is None
    assert load["queue"]["capacity_normalized_pressure_availability"] == "UNKNOWN"
    assert views["overlaps"][0]["value"] == 0.5
    assert views["traffic"][0]["value"] == 6.4
    assert views["traffic"][0]["boundary"] == "device_requested_read"
    assert views["traffic"][0]["denominator"]["value"] == 10
    assert views["traffic"][0]["bandwidth_capacity_fraction"] is None
    assert views["provenance"] == {key: source[key] for key in ("scope_sha256", "model_sha256", "elf_sha256")}
    assert not views["resource_times_additive"]


def test_utilization_missing_producers_stay_unknown_and_zero_queues_are_not_hardware_proof():
    source = report()
    source["cycle_breakdown"] = {"host_cycles": 9, "storage_byte_highwater": 1024}
    views = S.project(source)["utilization_views"]
    assert views["compute"]["compute_ceiling_attainment"]["availability"] == "UNKNOWN"
    assert views["resources"][0]["queue"]["average_waiting_entries"] == 0
    assert views["hardware_queue_pressure"]["availability"] == "UNKNOWN"
    assert views["physical_engine_occupancy"]["availability"] == "UNKNOWN"
    assert all(v["availability"] == "UNKNOWN" and v["value"] is None for v in views["host"].values())
    assert all(v["availability"] == "UNKNOWN" and v["value"] is None for v in views["storage"].values())
    absent = next(v for v in views["traffic"] if v["boundary"] == "modeled_dram_requests")
    assert absent["availability"] == "UNKNOWN" and absent["value"] is None


def test_utilization_incomplete_resources_do_not_discard_valid_work_or_invent_zero_activity():
    source = report()
    source["scoped_resource_activity"]["complete"] = False
    views = S.project(source, structural={"structural_peak_macs_per_cycle": 10})["utilization_views"]
    assert views["compute"]["compute_ceiling_attainment"]["value"] == 10  # Not capped to a fake utilization percentage.
    assert views["resource_availability"] == "UNKNOWN" and views["resources"] == []
    source["timing_complete"] = False
    views = S.project(source)["utilization_views"]
    assert views["availability"] == "UNKNOWN"
    assert views["denominator"]["value"] is None
    assert views["compute"]["useful_fraction_of_padded_work"]["value"] is None
    assert views["traffic"] == []


@pytest.mark.parametrize("peak", [0, -1, True, float("nan"), float("inf"), 1e308])
def test_utilization_invalid_structural_peak_is_refused(peak):
    with pytest.raises(C.ContractError):
        S.project(report(), structural={"structural_peak_macs_per_cycle": peak})


def test_compact_utilization_keeps_availability_provenance_and_denominators_bounded():
    value = S.project(report(), structural={"structural_peak_macs_per_cycle": 256})
    views = value["utilization_views"]
    views["resources"] *= 20
    views["overlaps"] *= 20
    brief = S.compact(value)["utilization_views"]
    assert len(brief["resources"]) == 16 and brief["resources_omitted"] == 24
    assert len(brief["overlaps"]) == 8 and brief["overlaps_omitted"] == 12
    assert brief["compute"] == views["compute"]
    assert brief["provenance"] == views["provenance"]
    assert brief["denominator"] == views["denominator"]
    assert brief["host"] == views["host"] and brief["storage"] == views["storage"]
    assert len(views["resources"]) == 40  # Compact does not mutate the full projection.


def test_compact_utilization_keeps_late_registered_busy_resources_visible():
    source = report()
    template = source['scoped_resource_activity']['resources'][1]
    source['scoped_resource_activity']['resources'] = [
        dict(template, resource_id=i, name=f'generic.resource.{i}',
             busy_unit_cycles=1, active_cycles=1) for i in range(20)
    ]
    source['scoped_resource_activity']['resources'][-1].update(
        busy_unit_cycles=10, active_cycles=10)
    source['scoped_resource_activity']['overlaps'] = []
    value = S.project(source)
    before = copy.deepcopy(value)
    brief = S.compact(value)
    raw_ids = [r['resource_id'] for r in brief['resources']]
    view_ids = [r['resource_id'] for r in brief['utilization_views']['resources']]
    assert raw_ids == view_ids and raw_ids[0] == 19
    assert brief['utilization_views']['resources'][0]['capacity_reservation_fraction'] == 1
    assert brief['estimated_cycles_same_window'] == 10
    assert not brief['qualification_issued']
    assert value == before
