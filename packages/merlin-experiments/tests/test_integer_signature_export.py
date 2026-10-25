"""Actual neutral output producers; no Spike, gSIM or guest ELF is executed."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F
from merlin_experiments.phase1 import joint_calibration as C
from merlin_experiments.phase2.integer_signature_export import IntegerSignatureExport, reopen_values
from test_retained_memory_observation import original


def pin(path):return F.pin(path,max_bytes=64<<20)


@pytest.fixture
def exported(original,tmp_path):
    raw,cb=original
    elf=tmp_path/'original.elf';elf.write_bytes(raw)
    commands=tmp_path/'commands.json';commands.write_text(json.dumps(cb))
    plan=IntegerSignatureExport.prepare(elf=elf,command_buffer=commands)
    region,expected,symbols=plan.materialize(elf)
    supplement=tmp_path/'symbols.elf';supplement.write_bytes(symbols)
    scope=tmp_path/'scope.json';scope.write_text('{}')
    processes=[]
    for index in range(2):
        signature=tmp_path/f'output-{index}.signature'
        # The neutral Python producer writes fixture bytes. The original ELF
        # remains an observed input, never an executed host or guest program.
        script='import pathlib,sys; p=next(a.split("=",1)[1]for a in sys.argv if a.startswith("+signature=")); pathlib.Path(p).write_text('+repr(''.join(f'{b:02x}\n'for b in expected))+'); print("DONE")'
        argv=plan.commands([sys.executable,'-I','-B','-c',script,str(elf)],elf=elf,supplement=supplement,signature=signature)
        I.run(argv,directory=tmp_path,stage='archive_screen_count',inputs=(elf,scope,supplement),outputs=(signature,),
            dependencies=(Path(__file__),),env={},capture_output=True,timeout=5)
        paths=[p for p in(tmp_path/'invocations').glob('*/invocation.json')if str(signature)in str(json.loads(p.read_bytes())['argv'])]
        path=next(p for p in paths if any(x['path']==str(signature)for x in json.loads(p.read_bytes())['outputs']))
        row=I.verify(path)
        processes.append(dict(path=str(path),sha256=pin(path)['sha256'],stdout_sha256=row['stdout']['sha256'],full_output_signature=pin(signature)))
    observation=dict(elf_sha256=pin(elf)['sha256'],scope_binding={},processes=processes,
        output_export=dict(plan=plan.record(),supplement=pin(supplement)))
    return SimpleNamespace(elf=elf,commands=commands,plan=plan,supplement=supplement,processes=processes,
        observation=observation,expected=expected,root=tmp_path)


def test_ordinary_symbol_export_reopens_two_actual_complete_output_producers(exported):
    f=exported
    assert [reopen_values(f.observation,p)for p in f.processes]==[{'Y':[[-2,3]]}]*2
    assert f.elf.read_bytes()==Path(f.plan.elf[0]).read_bytes()
    # Export contains only symbol metadata, never code or tensor payloads.
    import struct
    assert struct.unpack_from('<H',f.supplement.read_bytes(),56)[0]==0


@pytest.mark.parametrize('defect',['elf','command_buffer','supplement','signature','producer','ambiguous_options'])
def test_changed_export_or_missing_actual_producer_cannot_supply_values(exported,defect):
    f=exported;process=f.processes[0]
    if defect=='elf':f.elf.write_bytes(b'changed')
    elif defect=='command_buffer':f.commands.write_text('{}')
    elif defect=='supplement':f.supplement.write_bytes(b'changed')
    elif defect=='signature':Path(process['full_output_signature']['path']).write_bytes(b'00\n')
    else:
        path=Path(process['path']);row=json.loads(path.read_bytes())
        if defect=='producer':row['outputs']=[]
        else:row['argv'].insert(-1,'+signature='+process['full_output_signature']['path'])
        path.write_text(json.dumps(row));process['sha256']=pin(path)['sha256']
    with pytest.raises(ValueError):reopen_values(f.observation,process)


def test_normal_calibration_resume_reopens_exported_values_not_done_or_match_bits(exported):
    f=exported;model=f.root/'model.json';model.write_text('{}')
    report=f.root/'report.json';report.write_text(json.dumps(dict(elf_sha256=pin(f.elf)['sha256'],
        model_sha256=pin(model)['sha256'],scope_binding={},timing_complete=True,unresolved=[])))
    f.observation['report']=pin(report)
    record=dict(elf=str(f.elf),observation=f.observation)
    assert C._reopen_observation(record,SimpleNamespace(model_path=model))[0]==f.elf
    Path(f.processes[1]['full_output_signature']['path']).write_bytes(b'00\n')
    with pytest.raises(ValueError):C._reopen_observation(record,SimpleNamespace(model_path=model))


def test_export_refuses_an_unrelated_primary_elf_or_preexisting_flags(exported):
    f=exported
    for argv in (['spike','another.elf'],['spike','+signature=unrelated',str(f.elf)]):
        with pytest.raises(ValueError):f.plan.commands(argv,elf=f.elf,supplement=f.supplement,signature=f.root/'out')


@pytest.mark.parametrize('gap,mixed',[(False,False),(True,False),(False,True)])
def test_multifield_export_covers_only_adjacent_complete_typed_outputs(tmp_path,gap,mixed):
    import shutil,subprocess
    cc=shutil.which('cc')
    if not cc:pytest.skip('host compiler unavailable for ELF object-layout fixture')
    source=tmp_path/'objects.c';elf=tmp_path/'objects.elf'
    first='static signed char tensor_0[4]'if mixed else 'static int tensor_0[2]'
    second='static int tensor_1[1]'if mixed else 'static int tensor_1[2]'
    attributes=' __attribute__((used,section(".export_fixture"),aligned(4)));\n'
    source.write_text(first+attributes+('static int unrelated[2]'+attributes if gap else '')+
        second+attributes+'int main(void){return 0;}\n')
    subprocess.run([cc,'-O0','-no-pie',str(source),'-o',str(elf)],check=True,capture_output=True)
    values={'Y0':[[-2,3,4,5]]if mixed else[[-2,3]],'Y1':[[-200000]]if mixed else[[4,-5]]}
    cb={'target':'fixture','commands':[],'params':{},'tensors':{
        'Y0':{'role':'output','dtype':'i8'if mixed else'i32','shape':[1,4]if mixed else[1,2]},
        'Y1':{'role':'output','dtype':'i32','shape':[1,1]if mixed else[1,2]}},'_oracle_expected_outputs':values}
    path=tmp_path/'commands.json';path.write_text(json.dumps(cb))
    if gap:
        with pytest.raises(ValueError,match='gaps, overlaps'):
            IntegerSignatureExport.prepare(elf=elf,command_buffer=path)
        return
    plan=IntegerSignatureExport.prepare(elf=elf,command_buffer=path)
    region,expected,symbols=plan.materialize(elf)
    assert len(region['fields'])==2 and region['bytes']==sum(field['bytes']for field in region['fields'])
    signature=tmp_path/'outputs.signature'
    # One actual neutral producer supplies both output fields in one frame;
    # there is no per-output rerun or fabricated numerical execution.
    I.run([sys.executable,'-I','-B','-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('+repr(''.join(f'{b:02x}\n'for b in expected))+')',str(signature)],
        directory=tmp_path,stage='neutral_multifield_fixture',inputs=(elf,),outputs=(signature,),env={},capture_output=True)
    assert plan.values(signature)==values
