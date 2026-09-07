"""Static route presence is not loop semantics, numerical proof, or timing."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from merlin.perf.task_instruction_evidence import digest, summarize_task_instructions, _sha


def fixture():
    def op(name, owner):
        attr=SimpleNamespace(value=SimpleNamespace(data=owner))
        return SimpleNamespace(name=name,attributes={"merlin.global_task":attr})
    operations=[op("llvm.inline_asm",-1),op("llvm.add",0),op("llvm.inline_asm",0),
                op("llvm.inline_asm",0),op("llvm.store",1),op("llvm.inline_asm",1)]
    rows=[{"index":i,"class":klass,"funct":selector,"decoded":fields} for i,klass,selector,fields in
          [(0,"BARRIER",None,{}),(1,"COMPUTE",91,{"source":3}),
           (2,"UNKNOWN",92,{}),(3,"SEQUENCER",93,{})]]
    cb={"params":{"global_program_plan":{"tasks":[
        {"task_index":0,"kind":"contraction","source_op_indices":[2]},
        {"task_index":1,"kind":"contraction","source_op_indices":[5]}]}}}
    raw=json.dumps(cb)
    proof={"status":"verified","candidate_sha256":"a"*64,"logical_dispatch_digest":"b"*64,
        "source_sha256":_sha("source"),"candidate_lowered_sha256":_sha("llvm"),
        "candidate_command_buffer_sha256":_sha(raw),"plan_digest":digest(cb["params"]["global_program_plan"])}
    kwargs=dict(source_text="source",lowered_text="llvm",command_buffer_text=raw,
        command_buffer=cb,verified_plan=proof,parsed_module=SimpleNamespace(walk=lambda:iter(operations)),
        decoded_trace={"instructions":rows},target_facts={"isa":{"revision":"synthetic-a"},
            "roles_by_selector":{"91":["accumulate"],"93":["sequence"]}},
        host_policy_sha256="c"*64,decode_module=lambda _: {"instructions":rows})
    return kwargs,operations,rows


def test_actual_source_ownership_and_unknown_descriptor_coverage_remain_visible():
    kwargs,_,_=fixture()
    result=summarize_task_instructions(**kwargs)
    first,second=result["tasks"]
    assert result["status"]=="static_ownership_verified"
    assert first["source_op_indices"]==[2] and first["instruction_indices"]==[1,2]
    assert first["class_counts"]=={"COMPUTE":1,"UNKNOWN":1}
    assert first["role_counts"]=={"accumulate":1} and first["classification_coverage"]=="partial"
    assert second["static_operation_counts"]["llvm.store"]==1
    assert second["classification_coverage"]=="complete" and second["descriptor_semantics"]=="UNVERIFIED"
    assert second["instructions_without_decoded_fields"]==[3]
    assert result["wrapper_instruction_indices"]==[0]
    assert result["route_correspondence"]=="UNKNOWN" and not result["timing_calibration_admissible"]
    assert result["dynamic_instruction_counts"] is None


@pytest.mark.parametrize("change",["source","llvm","buffer","proof","policy","trace","owner","module"])
def test_stale_or_unowned_records_fail_closed(change):
    kwargs,ops,rows=fixture()
    if change=="source":kwargs["source_text"]+="changed"
    elif change=="llvm":kwargs["lowered_text"]+="changed"
    elif change=="buffer":kwargs["command_buffer"]["kernel_abi"]={"changed":True}
    elif change=="proof":kwargs["verified_plan"]["status"]="UNKNOWN"
    elif change=="policy":kwargs["host_policy_sha256"]=None
    elif change=="trace":kwargs["decoded_trace"]={"instructions":deepcopy(rows)};kwargs["decoded_trace"]["instructions"][1]["class"]="OTHER"
    elif change=="owner":ops[-1].attributes.clear()
    elif change=="module":kwargs["parsed_module"]=None
    with pytest.raises(ValueError):summarize_task_instructions(**kwargs)


def test_target_roles_are_data_and_change_evidence_identity():
    kwargs,_,_=fixture()
    first=summarize_task_instructions(**kwargs)
    kwargs["target_facts"]={"isa":{"revision":"synthetic-b"},"roles_by_selector":{"91":["multiply","accumulate"]}}
    second=summarize_task_instructions(**kwargs)
    assert first["binding"]["target_facts_sha256"]!=second["binding"]["target_facts_sha256"]
    assert second["tasks"][0]["role_counts"]=={"accumulate":1,"multiply":1}
    assert not second["tasks"][1]["role_counts"]
    assert second["tasks"][1]["instructions_without_target_roles"]==[3]


def test_exact_payload_change_changes_owned_digest_without_claiming_route_change():
    kwargs,_,rows=fixture()
    first=summarize_task_instructions(**kwargs)
    rows[1]["decoded"]["source"]=6
    second=summarize_task_instructions(**kwargs)
    assert first["tasks"][0]["owned_instruction_payload_sha256"]!=second["tasks"][0]["owned_instruction_payload_sha256"]
    assert first["tasks"][0]["class_counts"]==second["tasks"][0]["class_counts"]
    assert second["route_correspondence"]=="UNKNOWN"


@pytest.mark.parametrize("mutation",[None,"unrelated_omission","stale_llvm","stale_proof","unverified_cfg"])
def test_short_initializer_permission_is_labeled_separately_not_full_coverage(mutation):
    kwargs,_,_=fixture()
    proof=kwargs["verified_plan"]
    proof.update(status="refused",problems=["source operations are not fully covered: [0, 1]"],
                 control_flow={"status":"verified"})
    admission={"schema":"short_initializer_execution_admission_v1","status":"source_bound_numerical_probe",
        "source_sha256":_sha(kwargs["source_text"]),"lowered_sha256":_sha(kwargs["lowered_text"]),
        "command_buffer_sha256":digest(kwargs["command_buffer"]),"candidate_sha256":proof["candidate_sha256"],
        "declared_proof_sha256":digest(proof),"declared_plan_modified":False,"declared_plan_verified":False,
        "target_route_verified":False,"global_cost_calibration":False,
        "initializer_closure":{"status":"proved","source_sha256":_sha(kwargs["source_text"]),
            "command_buffer_sha256":digest(kwargs["command_buffer"]),"implicit_source_op_indices":[0,1]}}
    if mutation=="unrelated_omission":admission["initializer_closure"]["implicit_source_op_indices"]=[0,1,4]
    elif mutation=="stale_llvm":admission["lowered_sha256"]="f"*64
    elif mutation=="stale_proof":admission["declared_proof_sha256"]="f"*64
    elif mutation=="unverified_cfg":proof["control_flow"]["status"]="UNKNOWN";admission["declared_proof_sha256"]=digest(proof)
    kwargs["short_execution_admission"]=admission
    if mutation:
        with pytest.raises(ValueError):summarize_task_instructions(**kwargs)
    else:
        result=summarize_task_instructions(**kwargs)
        assert result["status"]=="short_admitted_static_ownership"
        assert result["declared_source_plan_status"]=="refused"
        assert result["implicit_initializer_source_op_indices"]==[0,1]
        assert proof["status"]=="refused" and not result["timing_calibration_admissible"]
