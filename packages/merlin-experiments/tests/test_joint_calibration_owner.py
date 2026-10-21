import copy
import json
import os
from pathlib import Path
import pytest

from merlin_experiments.phase1 import joint_calibration_owner as O


def fixture(tmp_path,monkeypatch):
    from merlin_experiments.phase1 import joint_epochs as E
    run=tmp_path/'initial';run.mkdir()
    state={'policy_sha256':'policy','started_at_epoch':1000,'policy':{'duration_s':43200,'drain_s':1800}}
    (run/'joint_campaign.json').write_text(json.dumps(state))
    epoch=tmp_path/'epoch.json';epoch.write_text(json.dumps({'previous_run':str(run),'performance_selection_sha256':'new'}))
    perf=tmp_path/'performance.json';perf.write_text('{}')
    initial={'run_id':'initial','run_dir':str(run),'task':str(tmp_path),'pins':[],
             'environment':{'MERLIN_COMPUTE_CPU_SLOTS':'same-compute','MERLIN_GSIM_SLOT_ROOT':'same-gsim','OTHER':'unchanged'},
             'argv':['python','-B','-m','merlin_experiments.phase1','--fresh-backend','--run-id','initial','--codex-home-root','old-home','--joint-campaign','same-policy']}
    response={'status':'ready_for_native_epoch','joint_epoch':str(epoch),'performance_selection':{'path':str(perf),'sha256':'new'}}
    monkeypatch.setattr(E,'_verify',lambda record:(state,{}))
    monkeypatch.setattr(O.time,'time',lambda:2000)
    return initial,response


def test_continuation_changes_only_epoch_model_identity_and_keeps_pools(tmp_path,monkeypatch):
    initial,response=fixture(tmp_path,monkeypatch);original=copy.deepcopy(initial)
    new=O.prepare_continuation(initial,initial,response,1)
    assert initial==original
    assert new['run_id']=='initial-model-001'
    assert new['environment']['MERLIN_COMPUTE_CPU_SLOTS']=='same-compute'
    assert new['environment']['MERLIN_GSIM_SLOT_ROOT']=='same-gsim'
    assert new['environment']['OTHER']=='unchanged'
    assert new['argv'][new['argv'].index('--joint-epoch')+1]==response['joint_epoch']
    assert new['continuation']['original_started_at_epoch']==1000
    assert new['argv'][new['argv'].index('--joint-campaign')+1]=='same-policy'


def test_expired_original_clock_refuses_continuation(tmp_path,monkeypatch):
    initial,response=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(O.time,'time',lambda:50000)
    with pytest.raises(ValueError,match='clock expired'):O.prepare_continuation(initial,initial,response,1)


def test_other_run_cannot_be_adopted_as_epoch(tmp_path,monkeypatch):
    initial,response=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='another native run'):
        O.prepare_continuation(initial,dict(initial,run_dir='another'),response,1)


def test_live_pid_requires_exact_native_argv():
    args=[v.decode() for v in Path(f'/proc/{os.getpid()}/cmdline').read_bytes().split(b'\0') if v]
    assert O.process_identity({'pid':os.getpid(),'argv':args})['pid']==os.getpid()
    with pytest.raises(ValueError,match='no longer identifies'):
        O.process_identity({'pid':os.getpid(),'argv':['not-the-owner']})
