"""Original neutral process/input joins only; no ISA or physical timing claim."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F
from merlin_experiments.phase1 import joint_replay_input_equivalence as E


def pin(path):return F.pin(path,max_bytes=64<<20)


@pytest.fixture
def retained(tmp_path):
    owner=tmp_path/'replay';workspace=owner/'workspace';workspace.mkdir(parents=True)
    original=tmp_path/'missing-original.elf';original.write_bytes(b'opaque original fixture')
    original_pin=pin(original);copied=workspace/'program.elf';copied.write_bytes(original.read_bytes())
    original.unlink()
    model=tmp_path/'model.json';model.write_text(json.dumps({'domain':{'scope_sha256':'s'*64}}))
    scope=workspace/'scope.json';binding=dict(elf_sha256=original_pin['sha256'],input_sha256=original_pin['sha256'],
        scope_sha256='s'*64,begin_pc=4,end_pc=20,occurrence=0);scope.write_text(json.dumps(binding))
    report=workspace/'report.json'
    report_body=dict(elf_sha256=original_pin['sha256'],model_sha256=pin(model)['sha256'],scope_binding=binding,
        timing_complete=True,execution_completed=True,unresolved=[])
    runs=[]
    for index in range(2):
        folder=owner/f'run-{index}';folder.mkdir()
        script='print("DONE")'
        options=[];outputs=()
        if index:
            script='import pathlib; pathlib.Path('+repr(str(report))+').write_text('+repr(json.dumps(report_body))+'); print("DONE")'
            options=['--timing-model='+str(model),'--timing-report='+str(report),
                '--timing-scope-binding='+str(scope),'--timing-input-sha256='+original_pin['sha256']]
            outputs=(report,)
        I.run([sys.executable,'-I','-B','-c',script,*options,str(copied)],directory=folder,
            stage='archive_screen_count',inputs=(copied,scope),outputs=outputs,
            dependencies=(model,Path(__file__)),env={},capture_output=True,timeout=5)
        invocation=next((folder/'invocations').glob('*/invocation.json'))
        runs.append(dict(invocation=pin(invocation)))
    receipt=dict(schema='operator.retained_same_elf_observation.v1',transport='existing_htif_symbol_elf_signature',
        physical_runs=0,calibration_eligible=False,runtime_sha256='r'*64,model=pin(model),functional_console_equal=True,
        original={'elf':original_pin},report=pin(report),scope_binding=binding,runs=runs)
    receipt_path=owner/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    return SimpleNamespace(owner=owner,original=original,copied=copied,model=model,scope=scope,report=report,
        receipt=receipt,receipt_path=receipt_path,receipt_pin=pin(receipt_path))


def verify(f):return E.verify(f.receipt_pin,f.receipt,runtime_sha256='r'*64,model_pin=pin(f.model))


def test_only_original_pinned_replay_input_supplies_missing_elf_bytes(retained):
    f=retained;proof=verify(f)
    assert proof['kind']=='original_recorded_replay_input_equivalence'
    assert proof['original_path_present'] is False and proof['original']['path']==str(f.original)
    assert proof['input']==pin(f.copied) and not f.original.exists()
    assert 'no new observation'in proof['qualification']


@pytest.mark.parametrize('defect',['copied','arbitrary_copy','scope','model','receipt','missing_invocation','producer_stage',
                                   'model_dependency','report_output','annotated_options'])
def test_missing_or_substituted_original_input_closure_refuses(retained,defect):
    f=retained
    if defect=='copied':f.copied.write_bytes(b'changed')
    elif defect=='arbitrary_copy':
        elsewhere=f.owner/'unrelated.elf';shutil.copyfile(f.copied,elsewhere);f.copied.unlink()
    elif defect=='scope':f.scope.write_text('{}')
    elif defect=='model':f.model.write_text('{}')
    elif defect=='receipt':f.receipt_path.write_text('{}')
    elif defect=='missing_invocation':Path(f.receipt['runs'][0]['invocation']['path']).unlink()
    else:
        path=Path(f.receipt['runs'][1]['invocation']['path']);row=json.loads(path.read_bytes())
        if defect=='producer_stage':row['stage']='unrelated'
        elif defect=='model_dependency':row['dependencies']=[p for p in row['dependencies']if p['path']!=str(f.model)]
        elif defect=='report_output':row['outputs']=[]
        else:row['argv']=[arg for arg in row['argv']if not arg.startswith('--timing-scope-binding=')]
        path.write_text(json.dumps(row));f.receipt['runs'][1]['invocation']=pin(path)
        f.receipt_path.write_text(json.dumps(f.receipt));f.receipt_pin=pin(f.receipt_path)
    with pytest.raises((ValueError,OSError,KeyError)):verify(f)


def test_present_but_changed_original_never_uses_matching_copy_fallback(retained):
    f=retained;f.original.write_bytes(b'changed original')
    with pytest.raises(ValueError):verify(f)
    f.original.write_bytes(f.copied.read_bytes())
    assert verify(f)['kind']=='original_elf_present'
