"""Source report arithmetic/custody only; no physical/model qualification."""
import copy
import pytest
from merlin_timing import source_activity as S, contracts as C
from test_scoped_resources import report as original_report

def report():
    value=original_report()
    pins={k:value[k] for k in ("model_sha256","elf_sha256","scope_sha256","scope_binding_sha256")}
    value["scoped_source_actor_activity"]=dict(schema=S.SCHEMA,source_kind="modeled_source_actor",unit="model_cycles",complete=True,**pins,
        begin_cycle=5,end_cycle=15,actors=[dict(name="reader.slot_residency",meaning="finite source slot",availability="modeled",capacity=16,active_cycles=8,resident_unit_cycles=12,peak_units=2,starts_in_scope=3),
        dict(name="execute.feed",meaning="modeled service not physical PE busy",availability="modeled",capacity=1,active_cycles=6,resident_unit_cycles=6,peak_units=1,starts_in_scope=1)],
        overlaps=[dict(left="reader.slot_residency",right="execute.feed",active_cycles=4)])
    return value

def test_same_roi_actor_union_and_residency_are_separate():
    source=report();before=copy.deepcopy(source);value=S.project(source)
    assert source==before and value["status"]=="modeled"
    assert value["actors"][0]["active_fraction"]==.8
    assert value["actors"][0]["average_resident_units"]==1.2
    assert value["actors"][0]["capacity_normalized_residency"]==.075
    assert value["overlaps"][0]["fraction_of_window"]==.4
    assert not value["physical_occupancy"] and not value["timing_qualification_issued"]

@pytest.mark.parametrize("mutation",["elf","scope","window","overlap","capacity","duplicate","integral"])
def test_custody_and_impossible_source_activity_refused(mutation):
    source=report();raw=source["scoped_source_actor_activity"]
    if mutation=="elf":raw["elf_sha256"]="f"*64
    elif mutation=="scope":raw["scope_binding_sha256"]="e"*64
    elif mutation=="window":raw["end_cycle"]+=1
    elif mutation=="overlap":raw["overlaps"][0]["active_cycles"]=7
    elif mutation=="capacity":raw["actors"][0]["peak_units"]=17
    elif mutation=="duplicate":raw["actors"].append(dict(raw["actors"][0]))
    else:raw["actors"][0]["resident_unit_cycles"]=17
    with pytest.raises(C.ContractError):S.project(source)

def test_missing_and_incomplete_never_become_zero_activity():
    source=report();del source["scoped_source_actor_activity"]
    assert S.project(source)["actors"]==[]
    source=report();source["scoped_source_actor_activity"]["complete"]=False
    assert S.project(source)["status"]=="UNKNOWN" and not S.project(source)["actors"]
    source=report();row=source["scoped_source_actor_activity"]["actors"][0]
    row.update(availability="UNKNOWN",active_cycles=None,resident_unit_cycles=None,peak_units=None,starts_in_scope=None)
    source["scoped_source_actor_activity"]["overlaps"][0]["active_cycles"]=None
    assert S.project(source)["actors"][0]["availability"]=="UNKNOWN"


def test_common_feedback_preserves_source_intervals_in_compact_view():
    from merlin_timing import scoped_resources
    value = scoped_resources.compact(scoped_resources.project(report()))
    actors = value['source_actor_activity']
    assert actors['status'] == 'modeled'
    assert actors['actors'][0]['active_fraction'] == .8
    assert value['utilization_views']['source_actor_lifetimes'] == actors
    assert value['estimated_cycles_same_window'] == 10


def test_legacy_observer_without_source_capture_or_binding_remains_usable():
    from merlin_timing import scoped_resources
    source = original_report()
    del source['scope_binding_sha256']
    del source['scoped_resource_activity']
    value = scoped_resources.project(source)
    assert value['estimated_cycles_same_window'] == 10
    assert value['source_actor_activity']['status'] == 'UNKNOWN'
    assert value['source_actor_activity']['actors'] == []
