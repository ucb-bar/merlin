"""Synthetic boundary traces exercise comparison, never hardware qualification."""
import copy
import hashlib
import json

import pytest
from merlin_timing import contracts as C, memory_reference as M


def selection(tmp_path):
    result = {"schema": M.SELECTION, "configuration": "example.ExistingMemoryModel",
              "axi": dict(data_bits=64,address_bits=32,id_bits=4,max_read_bytes=64,max_write_bytes=64,
                          max_flight=8,id_reuse=2,address_base=0,address_size=4096), "sources": {}}
    for role in M.REQUIRED_SOURCES:
        path = tmp_path / role
        path.write_text("+mm_latency_0=7\n" if role == "runtime_settings" else role)
        result["sources"][role] = {"path": role, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def traces(tmp_path):
    header = dict(schema=M.TRACE, memory_reference_sha256=C.digest("memory"),
                  initial_state_sha256=C.digest("cold"), scope_sha256=C.digest("boundary"),
                  stimulus_sha256=C.digest("inputs"), engine_sha256=C.digest("reference"),
                  ports={"inputs": {"request": {"width": 1, "valid_when": None}},
                         "outputs": {"ready": {"width": 1, "valid_when": None},
                                     "valid": {"width": 1, "valid_when": None},
                                     "data": {"width": 8, "valid_when": "valid"}}})
    frames = [dict(cycle=i,inputs=dict(request=i%2),outputs=dict(ready=i%2,valid=int(i==2),data=42)) for i in range(4)]
    paths = [tmp_path/"reference.jsonl", tmp_path/"candidate.jsonl"]
    for i,path in enumerate(paths):
        value = copy.deepcopy(header);value["engine_sha256"] = C.digest(str(i))
        path.write_text("\n".join(json.dumps(x) for x in [value,*frames])+"\n")
    return paths


def rewrite(path, index, change):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows[index])
    path.write_text("\n".join(json.dumps(x) for x in rows)+"\n")


def test_preparation_verifies_source_bytes_without_claiming_equivalence(tmp_path):
    selected=selection(tmp_path)
    report=M.prepare(selected,source_root=tmp_path)
    assert report['settings']=={'+mm_latency_0':7}
    assert report['source_settings_verified'] and not report['native_adapter_qualified']
    (tmp_path/'timing_model').write_text('changed')
    with pytest.raises(C.ContractError,match='bytes changed'):
        M.prepare(selected,source_root=tmp_path)


@pytest.mark.parametrize('text',['','+mm_latency_0=1\n+mm_latency_0=1','+mm_latency_0=nan','+mm_useDefaults_0','+other=1'])
def test_runtime_settings_never_guess_missing_defaults(text):
    with pytest.raises(C.ContractError): M.runtime_settings(text)


def test_equal_cycles_pass_trace_only_and_invalid_payload_is_ignored(tmp_path):
    a,b=traces(tmp_path)
    rewrite(b,1,lambda x: x['outputs'].update(data=99))
    report=M.compare(a,b)
    assert report['equivalent_for_trace'] and report['target_cycles']==4
    assert not report['whole_system_qualified']


@pytest.mark.parametrize('field,value',[('ready',1),('valid',1)])
def test_admission_and_response_timing_mismatches_are_observed(tmp_path,field,value):
    a,b=traces(tmp_path)
    rewrite(b,1,lambda x: x['outputs'].update({field:value}))
    report=M.compare(a,b)
    assert not report['equivalent_for_trace'] and report['signal_mismatches']==1


def test_valid_response_data_cannot_differ(tmp_path):
    a,b=traces(tmp_path)
    rewrite(b,3,lambda x: x['outputs'].update(data=99))
    assert M.compare(a,b)['signal_mismatches']==1


@pytest.mark.parametrize('field',['initial_state_sha256','memory_reference_sha256','scope_sha256','stimulus_sha256'])
def test_distinct_execution_state_or_scope_cannot_be_compared(tmp_path,field):
    a,b=traces(tmp_path)
    rewrite(b,0,lambda x: x.update({field:C.digest('different')}))
    with pytest.raises(C.ContractError,match='differ'): M.compare(a,b)


def test_skipped_cycles_changed_inputs_and_excess_budget_refuse(tmp_path):
    a,b=traces(tmp_path)
    with pytest.raises(C.ContractError,match='budget'): M.compare(a,b,maximum_cycles=2)
    rewrite(b,1,lambda x: x['inputs'].update(request=1))
    with pytest.raises(C.ContractError,match='inputs'): M.compare(a,b)
    a,b=traces(tmp_path)
    rewrite(b,2,lambda x: x.update(cycle=9))
    with pytest.raises(C.ContractError,match='every target cycle'): M.compare(a,b)


def test_cli_exposes_preparation_and_comparison_without_running_tools(tmp_path):
    from merlin_timing import cli
    selected=selection(tmp_path)
    source=tmp_path/'selection.json';source.write_text(json.dumps(selected))
    output=tmp_path/'prepared.json'
    assert cli.main(['memory-reference-prepare','--selection',str(source),'--source-root',str(tmp_path),
                     '--output',str(output)])==0
    assert not json.loads(output.read_text())['native_adapter_qualified']
    a,b=traces(tmp_path)
    assert cli.main(['memory-boundary-compare','--reference',str(a),'--candidate',str(b),
                     '--output',str(tmp_path/'comparison.json')])==0
