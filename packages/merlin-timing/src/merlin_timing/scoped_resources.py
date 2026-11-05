"""Same-window resource diagnostics; modeled reservations never become counters.

Consumes the opt-in native streaming sweep. No instruction-count conversion to
resource time, no sum of overlapping resources, and no qualification authority.
"""
from __future__ import annotations

from . import contracts as C
from . import observations as O

SCHEMA = 'merlin.scoped_performance_feedback.v1'
ACTIVITY_SCHEMA = 'merlin.scoped_resource_activity.v1'


def _count(value, name):
    if type(value) is not int or not 0 <= value < 2**64:
        raise C.ContractError(name+' must be an unsigned 64-bit integer')
    return value


def _project(report: dict, *, structural: dict | None = None) -> dict:
    """Project one original public native report; same function offline/in-loop."""
    if report.get('profile') != 'phase1-public':
        raise C.ContractError('public feedback requires the public observation profile')
    pins={key:C.sha(report.get(key),key) for key in ('scope_sha256','model_sha256','elf_sha256')}
    cycles=report.get('estimated_cycles')
    complete=(O.feature_complete(report) and report.get('execution_completed') is True
              and report.get('timing_complete') is True and report.get('unresolved') == []
              and report.get('functional_status') == 'PASS' and report.get('raw_functional_exit_code') == 0
              and type(cycles) is int and cycles > 0)
    counts=report.get('feature_counts') or {}
    observation=report.get('feature_observation') or {}
    index=observation.get('window_index')
    windows=report.get('counter_windows') or []
    window=windows[index] if type(index) is int and 0<=index<len(windows) else None
    complete=bool(complete and observation.get('scope_kind')=='counter_pc_pair' and window
                  and window.get('estimated_cycles')==cycles)
    result=dict(schema=SCHEMA,**pins,status='modeled' if complete else 'UNKNOWN',
        estimated_cycles_same_window=cycles if complete else None,source_kind='modeled',
        window=window,activity_status='UNKNOWN',resources=[],overlaps=[],traffic=[],commands={},
        compute=dict(useful_macs=None,nominal_padded_macs=None,useful_fraction_of_padded_work=None,
                     structural_floor_to_estimate_ratio=None,ratio_kind='structural bound / modeled cycles; not physical utilization'),
        host=dict(instruction_counts={},preparation_cycles=None,dispatch_cycles=None,
                  synchronization_wait_cycles=None,reason='No scoped preparation/dispatch/wait attribution producer'),
        unknown=['physical_compute_utilization','physical_bandwidth','reservation_station_occupancy',
                 'dependency_wait_cycles','host_preparation_cycles','host_dispatch_cycles'],
        suggestions=[],latency_is_objective=True,resource_cycles_additive=False,qualification_issued=False)
    if not complete:
        result['reason']='complete counts and modeled cycles in the exact counter-PC window required'
        return result
    for name,feature in [('total','target.commands'),('fences','target.fences'),
                         ('config','target.class.config'),('load','target.class.load'),
                         ('preload','target.class.preload'),('compute','target.class.compute'),
                         ('store','target.class.store')]:
        value=counts.get(feature)
        result['commands'][name]=_count(value,feature) if value is not None else None
    classes={key.removeprefix('target.class.'):value for key,value in counts.items()
             if key.startswith('target.class.')}
    # The observer counts target.commands at accelerator instruction issue.
    # A scalar fence is reported separately, even when its target drain is
    # classified as target.class.fence. It must not break the command census.
    command_classes={key:value for key,value in classes.items() if key!='fence'}
    if (classes and all(type(value) is int and value >= 0 for value in classes.values())
            and ('fence' not in classes or classes['fence']==result['commands']['fences'])
            and sum(command_classes.values())==result['commands']['total']):
        result['commands']['compute']=sum(value for key,value in command_classes.items()
                                          if key=='compute' or key.startswith('compute_'))
        result['commands']['store']=sum(value for key,value in command_classes.items()
                                        if key in {'store','pool_store'} or key.startswith(('store_','pool_store_')))
    for destination,source in [('useful_macs','target.useful_shape_macs'),('nominal_padded_macs','target.nominal_padded_macs')]:
        if counts.get(source) is not None:result['compute'][destination]=_count(counts[source],source)
    useful,padded=result['compute']['useful_macs'],result['compute']['nominal_padded_macs']
    if useful is not None and padded is not None:
        if useful > padded:raise C.ContractError('useful work exceeds declared nominal padded work')
        result['compute']['useful_fraction_of_padded_work']=useful/padded if padded else None
        if padded>useful:result['suggestions'].append('Compare tail handling and tiling alternatives that reduce padded work without increasing whole-call latency.')
    floor=(structural or {}).get('structural_compute_floor_cycles')
    if type(floor) in (int,float) and floor>=0:
        C.number(floor,'structural compute floor')
        result['compute']['structural_floor_to_estimate_ratio']=floor/cycles
    for boundary,feature in [('device_requested_read','device.requested_read_bytes'),
                             ('device_requested_write','device.requested_write_bytes'),
                             ('cpu_data_requested_read','cpu_data.requested_read_bytes'),
                             ('cpu_data_requested_write','cpu_data.requested_write_bytes'),
                             ('cpu_fetch_requested_read','cpu_fetch.requested_read_bytes'),
                             ('modeled_dram_requests','modeled.dram_request_bytes'),
                             ('modeled_cache_writebacks','modeled.writeback_bytes'),
                             ('modeled_coherence_writebacks','modeled.coherence_writeback_bytes')]:
        value=counts.get(feature)
        result['traffic'].append(dict(boundary=boundary,bytes=_count(value,feature) if value is not None else None,
            bytes_per_model_cycle=value/cycles if type(value) is int else None,
            source_kind=('modeled_boundary_bytes_over_modeled_cycles' if feature.startswith('modeled.')
                         else 'observed_requests_over_modeled_cycles'),physical_bandwidth=False))
    result['host']['instruction_counts']={k:_count(v,k) for k,v in counts.items() if k.startswith('cpu.') and type(v) is int}
    activity=report.get('scoped_resource_activity')
    if activity is None:
        result['reason']='native observer did not select scoped resource capture; whole-trace totals are excluded'
        return result
    if activity.get('schema')!=ACTIVITY_SCHEMA or activity.get('source_kind')!='modeled' or activity.get('unit')!='model_cycles':
        raise C.ContractError('unrecognized scoped resource provenance')
    for key,value in pins.items():
        if activity.get(key)!=value:raise C.ContractError('resource activity belongs to a different '+key)
    if not report.get('scope_binding_sha256') or activity.get('scope_binding_sha256')!=report['scope_binding_sha256']:
        raise C.ContractError('resource counter-boundary binding differs')
    begin,end=activity.get('begin_cycle'),activity.get('end_cycle')
    if activity.get('complete') is not True:
        result['reason']='scoped interval capture incomplete, unknown or bounded buffer exhausted'
        return result
    if _count(end,'resource end')-_count(begin,'resource begin')!=cycles:
        raise C.ContractError('resource interval differs from cycle-estimate window')
    rows=activity.get('resources')
    if type(rows) is not list or not 1<=len(rows)<=1024:raise C.ContractError('invalid scoped resource roster')
    indexed={}
    for row in rows:
        C.closed(row,{'resource_id','name','capacity','busy_unit_cycles','active_cycles','queued_entry_cycles',
                      'queue_active_cycles','peak_active_units','peak_queue_entries'},'scoped resource row')
        rid=_count(row['resource_id'],'resource ID');capacity=_count(row['capacity'],'resource capacity')
        if rid in indexed or capacity==0:raise C.ContractError('duplicate resource or zero capacity')
        C.text(row['name'],'resource name')
        for key in set(row)-{'resource_id','name'}:_count(row[key],key)
        if (row['active_cycles']>cycles or row['busy_unit_cycles']>capacity*row['active_cycles']
            or row['busy_unit_cycles']<row['active_cycles'] or row['peak_active_units']>capacity
            or row['queue_active_cycles']>cycles or row['queued_entry_cycles']<row['queue_active_cycles']
            or row['queued_entry_cycles']>row['peak_queue_entries']*row['queue_active_cycles']):
            raise C.ContractError('resource activity exceeds its window/capacity')
        indexed[rid]=row
        result['resources'].append(dict(row,capacity_normalized_activity=row['busy_unit_cycles']/(capacity*cycles),
            active_fraction=row['active_cycles']/cycles,average_modeled_waiting_entries=row['queued_entry_cycles']/cycles,
            meaning='modeled capacity reservation, not physical engine busy time',queue_meaning='modeled start queue, not accelerator ROB'))
    pairs=activity.get('overlaps');seen=set()
    if type(pairs) is not list or len(pairs)>32:raise C.ContractError('invalid overlap roster')
    for pair in pairs:
        C.closed(pair,{'left','right','active_cycles'},'resource overlap')
        key=tuple(sorted((pair['left'],pair['right'])))
        if key in seen or key[0]==key[1] or any(k not in indexed for k in key):raise C.ContractError('duplicate/unknown overlap resources')
        seen.add(key);duration=_count(pair['active_cycles'],'overlap')
        if duration>min(indexed[k]['active_cycles'] for k in key):raise C.ContractError('overlap exceeds resource activity')
        result['overlaps'].append(dict(pair,fraction_of_window=duration/cycles,source_kind='modeled'))
    result['activity_status']='modeled';result['activity_window']={'begin_cycle':begin,'end_cycle':end}
    queued=[r['name'] for r in rows if r['queued_entry_cycles']]
    if queued:result['suggestions'].append('Inspect modeled admission and resource contention for '+', '.join(queued[:4])+'. Compare legal ordering/overlap variants; queue time is not a measured hardware stall.')
    result['suggestions'].append('Rank candidates by complete same-window latency; activity and overlap are diagnostic, not objectives to maximize.')
    return result


def _measure(value, *, source, numerator, denominator, meaning, reason=None):
    """One diagnostic ratio with explicit availability and arithmetic custody."""
    return dict(
        availability="available" if value is not None else "UNKNOWN",
        value=value,
        source_kind=source,
        numerator=numerator,
        denominator=denominator,
        meaning=meaning,
        **({"reason": reason} if value is None else {}),
    )


def _utilization_views(value, structural):
    """Only already validated same-window quantities enter these views.

    Generic reservation queues have neither hardware queue capacity nor stall
    authority. No resource names are interpreted as physical engine counters.
    """
    cycles = value["estimated_cycles_same_window"]
    complete = value["status"] == "modeled" and cycles is not None
    provenance = {key: value[key] for key in ("scope_sha256", "model_sha256", "elf_sha256")}
    denominator = dict(quantity="same_window_modeled_cycles", value=cycles, unit="model_cycles")

    def unknown(reason):
        return _measure(None, source="unknown", numerator=None, denominator=None, meaning=reason, reason=reason)

    result = dict(
        schema="merlin.scoped_utilization_views.v1",
        provenance=provenance,
        availability="available" if complete else "UNKNOWN",
        denominator=denominator,
        resources=[],
        overlaps=[],
        traffic=[],
        host={
            name: unknown("No scoped host cycle-attribution producer")
            for name in ("preparation_cycle_fraction", "dispatch_cycle_fraction", "synchronization_wait_fraction")
        },
        storage=dict(
            capacity_occupancy=unknown("No scoped byte-capacity and lifetime occupancy producer"),
            byte_highwater=unknown("No scoped byte-capacity and lifetime occupancy producer"),
        ),
        physical_engine_occupancy=unknown(
            "No scoped actor-lifetime union producer: controller bookings may be dispatch-only; "
            "DMA/reader/response/compute lifetimes are not physical engine occupancy counters"
        ),
        hardware_queue_pressure=unknown(
            "Generic start queues do not measure hardware ROB occupancy or dependency stalls"
        ),
        resource_times_additive=False,
    )
    result["compute"] = dict(
        compute_ceiling_attainment=unknown(
            "Source/observed useful work, structural peak and complete modeled window required"
        ),
        useful_fraction_of_padded_work=unknown("Complete scoped useful and nominal padded MAC counts required"),
    )
    if not complete:
        result["reason"] = "Complete validated same-window modeled cycles are unavailable"
        return result
    compute = value["compute"]
    useful = compute["useful_macs"]
    padded = compute["nominal_padded_macs"]
    source_useful = structural.get("useful_macs")
    if source_useful is not None:
        source_useful = _count(source_useful, "source useful MACs")
    peak = structural.get("structural_peak_macs_per_cycle")
    if peak is not None:
        peak = C.number(peak, "structural peak MACs per cycle", positive=True)
    work = source_useful if source_useful is not None else useful
    if work is not None and peak is not None:
        peak_work = C.number(peak * cycles, "structural peak work over modeled window", positive=True)
        result["compute"]["compute_ceiling_attainment"] = _measure(
            work / peak_work,
            source="source_work_and_structural_peak_over_modeled_cycles"
            if source_useful is not None
            else "observed_shape_work_and_structural_peak_over_modeled_cycles",
            numerator=dict(
                quantity="source_useful_macs" if source_useful is not None else "observed_useful_shape_macs",
                value=work,
                unit="MACs",
            ),
            denominator=dict(
                quantity="structural_peak_times_modeled_window",
                value=peak_work,
                unit="MACs",
                peak_macs_per_cycle=peak,
                modeled_cycles=cycles,
            ),
            meaning="Compute ceiling attainment, not measured array occupancy; not capped at one",
        )
    result["compute"]["useful_fraction_of_padded_work"] = _measure(
        compute["useful_fraction_of_padded_work"],
        source="observed_same_window_work",
        numerator=dict(quantity="observed_useful_shape_macs", value=useful, unit="MACs"),
        denominator=dict(quantity="nominal_padded_macs", value=padded, unit="MACs"),
        meaning="Useful share of nominal padded arithmetic, independent of elapsed time",
        reason="Useful/padded counts absent or padded work is zero",
    )
    for row in value["resources"]:
        result["resources"].append(
            dict(
                resource_id=row["resource_id"],
                name=row["name"],
                availability="available",
                source_kind="modeled_capacity_reservations",
                capacity=dict(value=row["capacity"], unit="reservation_units"),
                capacity_reservation_fraction=row["capacity_normalized_activity"],
                reservation_active_fraction=row["active_fraction"],
                denominators=dict(
                    capacity_reservation_unit_cycles=row["capacity"] * cycles, active_model_cycles=cycles
                ),
                numerators=dict(reserved_unit_cycles=row["busy_unit_cycles"], active_model_cycles=row["active_cycles"]),
                queue=dict(
                    availability="available",
                    source_kind="modeled_generic_start_queue",
                    average_waiting_entries=row["average_modeled_waiting_entries"],
                    active_fraction=row["queue_active_cycles"] / cycles,
                    peak_entries=row["peak_queue_entries"],
                    numerators=dict(entry_cycles=row["queued_entry_cycles"], active_cycles=row["queue_active_cycles"]),
                    denominator_model_cycles=cycles,
                    capacity_normalized_pressure=None,
                    capacity_normalized_pressure_availability="UNKNOWN",
                    reason=("Hardware queue capacity/occupancy is not supplied; zero generic waiting "
                            "does not prove absent hardware pressure"),
                ),
                meaning=("Named modeled resource bookings may represent scalar dispatch only, "
                         "not DMA/engine busy time or stalls"),
            )
        )
    for pair in value["overlaps"]:
        result["overlaps"].append(
            dict(
                left=pair["left"],
                right=pair["right"],
                **_measure(
                    pair["fraction_of_window"],
                    source="modeled_reservation_intersection",
                    numerator=dict(
                        quantity="both_resources_active_cycles", value=pair["active_cycles"], unit="model_cycles"
                    ),
                    denominator=denominator,
                    meaning=("Named resource booking overlap; zero does not establish absent "
                             "load/compute/store execution overlap"),
                ),
            )
        )
    for row in value["traffic"]:
        result["traffic"].append(
            dict(
                boundary=row["boundary"],
                **_measure(
                    row["bytes_per_model_cycle"],
                    source=row["source_kind"],
                    numerator=dict(quantity="bytes_at_named_boundary", value=row["bytes"], unit="bytes"),
                    denominator=denominator,
                    meaning="Bytes per modeled cycle at this boundary; not physical bandwidth utilization",
                    reason="The selected scoped byte counter is absent",
                ),
                bandwidth_capacity_fraction=None,
                bandwidth_capacity_fraction_availability="UNKNOWN",
                capacity_reason="No independently validated service capacity for this traffic boundary",
            )
        )
    if value["activity_status"] != "modeled":
        result["resource_availability"] = "UNKNOWN"
        result["resource_reason"] = value.get("reason", "No validated scoped reservation producer")
    else:
        result["resource_availability"] = "available"
    return result


def project(report: dict, *, structural: dict | None = None) -> dict:
    """Project validated scoped diagnostics and common utilization views."""
    value = _project(report, structural=structural)
    value["utilization_views"] = _utilization_views(value, structural or {})
    from .source_activity import project as source_actor_feedback
    value["source_actor_activity"] = source_actor_feedback(report)
    value["utilization_views"]["source_actor_lifetimes"] = value["source_actor_activity"]
    return value


def compact(value):
    """Bounded common projection for terminal, automatic and offline feedback."""
    result = {key:value.get(key) for key in ('schema','status','source_kind','scope_sha256','model_sha256','elf_sha256','estimated_cycles_same_window',
        'activity_status','compute','traffic','commands','host','resources','overlaps','suggestions','unknown','reason',
        'latency_is_objective','resource_cycles_additive','qualification_issued','source_actor_activity')}

    views=value.get('utilization_views')
    if views is not None:
        result['utilization_views']=dict(views)
        for key,limit in (('resources',16),('overlaps',8)):
            items=views.get(key) or []
            if key == 'resources':
                # Resource IDs describe provider registration order, not
                # importance. Keep hot late-registered banks/writers visible
                # in the same bounded feedback as the reservation rows below.
                items=sorted(items,key=lambda r:(-r.get('capacity_reservation_fraction',0),
                                                  r.get('resource_id',0)))
            result['utilization_views'][key]=items[:limit]
            result['utilization_views'][key+'_omitted']=max(0,len(items)-limit)

    rows=value.get('resources') or []
    ranked=sorted(rows,key=lambda r:(-r.get('capacity_normalized_activity',0),r.get('resource_id',0)))
    result['resources']=ranked[:16]
    result['resource_rows_omitted']=max(0,len(rows)-16)
    host=dict(value.get('host') or {})
    counts=host.get('instruction_counts') or {}
    nonzero={key:count for key,count in counts.items() if count}
    host['instruction_counts']=dict(list(nonzero.items())[:16])
    host['instruction_count_rows_omitted']=max(0,len(nonzero)-16)
    result['host']=host
    result['overlaps']=(value.get('overlaps') or [])[:8]
    result['overlap_rows_omitted']=max(0,len(value.get('overlaps') or [])-8)
    return result
