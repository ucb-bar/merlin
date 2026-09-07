"""Four-arm route observations cannot launder static counts into equivalence."""
from copy import deepcopy
import pytest

from merlin.perf.task_instruction_evidence import digest
from merlin.perf.task_route_presence import compare_task_route_presence


def envelope(source, compiler, index, classes, *, roles=()):
    binding={"source_sha256":source*64,"lowered_sha256":"a"*64,"command_buffer_sha256":"b"*64,
        "compiler_sha256":compiler*64,"logical_dispatch_digest":source*64,"plan_digest":"c"*64,
        "target_facts_sha256":"d"*64,"host_verifier_policy_sha256":"e"*64}
    task={"task_index":3,"source_op_indices":[index],"declared_task_kind":"opaque",
        "instruction_indices":list(range(sum(classes.values()))),"class_counts":classes,
        "role_counts":{name:1 for name in roles},
        "unknown_instruction_indices":list(range(classes.get("UNKNOWN",0))),
        "instructions_without_target_roles":[] if roles else list(range(sum(classes.values()))),
        "owned_instruction_payload_sha256":digest(classes)}
    summary={"schema":"task_instruction_evidence_v1","status":"static_ownership_verified",
             "binding":binding,"tasks":[task]}
    return {"summary":summary,"summary_sha256":digest(summary),"expected_binding":deepcopy(binding)}


def scenario(*, second_family=False):
    before,after=("load_and_execute","descriptor_launch") if not second_family else ("vector_dispatch","fused_stream")
    full={"before":envelope("1","3",48,{before:80},roles=[before]),
          "after":envelope("1","4",48,{after:6},roles=[after])}
    short={"before":envelope("2","3",2,{before:2},roles=[before]),
           "after":envelope("2","4",2,{after:1},roles=[after])}
    extraction={"source_sha256":"1"*64,"probe_source_sha256":"2"*64,"source_op_index":48}
    return dict(full=full,short=short,extraction=extraction,
                extraction_sha256=digest(extraction),short_source_op_index=2)


@pytest.mark.parametrize("second_family",[False,True])
def test_supplied_target_classes_match_presence_without_geometry_or_timing(second_family):
    result=compare_task_route_presence(**scenario(second_family=second_family))
    assert result["status"]=="observed_known_class_presence_change_reproduced"
    assert result["known_class_presence_equal_by_arm"]=={"before":True,"after":True}
    assert result["all_static_classes_decoded"]
    assert result["descriptor_semantic_equivalence"]=="UNKNOWN"
    assert result["emitted_address_equivalence"]=="UNKNOWN"
    assert not result["timing_calibration_admissible"] and not result["global_cost_validated"]
    assert result["observations"]["full"]["before"]["static_instruction_count"]==80
    assert result["observations"]["short"]["before"]["static_instruction_count"]==2


def test_reduced_shape_losing_observed_route_is_reported_without_equivalence_claim():
    kwargs=scenario()
    kwargs["short"]["after"]=envelope("2","4",2,{"load_and_execute":2},roles=["load_and_execute"])
    result=compare_task_route_presence(**kwargs)
    assert result["status"]=="observed_known_class_presence_mismatch"
    assert result["known_class_presence_equal_by_arm"]=={"before":True,"after":False}
    assert result["class_presence_changes"]["short"]["appeared"]==[]
    assert not result["timing_calibration_admissible"]


def test_unknown_descriptor_sites_cannot_support_absence_claims():
    kwargs=scenario()
    kwargs["short"]["after"]=envelope("2","4",2,{"UNKNOWN":1})
    result=compare_task_route_presence(**kwargs)
    assert result["status"]=="observed_known_class_presence_mismatch"
    assert not result["all_static_classes_decoded"]
    assert result["observations"]["short"]["after"]["unknown_instruction_count"]==1
    assert result["descriptor_semantic_equivalence"]=="UNKNOWN"


@pytest.mark.parametrize("mutation",["source","compiler","policy","facts","source_index","multiowner","count","payload","extraction","missing"])
def test_stale_or_ambiguous_four_arm_join_refuses(mutation):
    kwargs=scenario()
    record=kwargs["short"]["after"]
    if mutation in {"source","compiler","policy","facts"}:
        key={"source":"source_sha256","compiler":"compiler_sha256","policy":"host_verifier_policy_sha256","facts":"target_facts_sha256"}[mutation]
        record["summary"]["binding"][key]="f"*64
        record["expected_binding"][key]="f"*64
    elif mutation=="source_index":record["summary"]["tasks"][0]["source_op_indices"]=[0]
    elif mutation=="multiowner":record["summary"]["tasks"][0]["source_op_indices"]=[2,5]
    elif mutation=="count":record["summary"]["tasks"][0]["class_counts"]={"descriptor_launch":99}
    elif mutation=="payload":record["summary"]["tasks"][0]["owned_instruction_payload_sha256"]="changed"
    elif mutation=="extraction":kwargs["extraction"]["source_op_index"]=5
    else:kwargs["short"].pop("before")
    if mutation!="payload":record["summary_sha256"]=digest(record["summary"])
    result=compare_task_route_presence(**kwargs)
    assert result["status"]=="UNKNOWN" and result.get("reason")
    assert not result["global_cost_validated"] and not result["timing_calibration_admissible"]


def test_static_count_decrease_alone_is_not_a_known_route_change():
    kwargs=scenario()
    for domain,source,index in (("full","1",48),("short","2",2)):
        kwargs[domain]["after"]=envelope(source,"4",index,{"load_and_execute":1},roles=["load_and_execute"])
    result=compare_task_route_presence(**kwargs)
    assert result["status"]=="no_known_class_presence_change"
    assert result["selected_owned_payload_changed"]["full"]
    assert result["descriptor_semantic_equivalence"]=="UNKNOWN"


@pytest.mark.parametrize("domain",["full","short"])
def test_initializer_exception_is_scoped_to_short_source_and_never_relabels_full_plan(domain):
    kwargs=scenario()
    record=kwargs[domain]["after"]
    binding=record["expected_binding"]
    admission={"schema":"short_initializer_execution_admission_v1","status":"source_bound_numerical_probe",
        **{key:binding[key] for key in ("source_sha256","lowered_sha256","command_buffer_sha256")},
        "candidate_sha256":binding["compiler_sha256"]}
    record["short_execution_admission"]=admission
    record["summary"].update(status="short_admitted_static_ownership",declared_source_plan_status="refused",
        implicit_initializer_source_op_indices=[0,1],short_execution_admission_sha256=digest(admission))
    record["summary_sha256"]=digest(record["summary"])
    result=compare_task_route_presence(**kwargs)
    if domain=="full":assert result["status"]=="UNKNOWN"
    else:
        assert result["status"]=="observed_known_class_presence_change_reproduced"
        assert result["observations"]["short"]["after"]["source_ownership_scope"]=="short_admitted_static_ownership"
        assert result["observations"]["short"]["after"]["implicit_initializer_source_op_indices"]==[0,1]
        admission["source_sha256"]="f"*64
        assert compare_task_route_presence(**kwargs)["status"]=="UNKNOWN"
