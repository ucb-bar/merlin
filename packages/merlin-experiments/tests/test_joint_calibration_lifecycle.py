"""Host lifecycle fixtures only; no simulator, author or fit is executed."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from merlin.runtime import _native_files as F
from merlin_experiments.phase1 import joint_calibration_lifecycle as L
from merlin_experiments.phase1 import joint_calibration_service as S


class Process:
    pid = 123456
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


@pytest.fixture
def owner(tmp_path, monkeypatch):
    profile = tmp_path / 'profile.json';profile.write_text('{}')
    selected = F.pin(profile, max_bytes=4096)
    state = tmp_path / 'joint_campaign.json'
    data = dict(started_at_epoch=1234, policy=dict(native_calibration=selected, gsim=dict(per_job_s=120)))
    state.write_text(json.dumps(data))
    descriptor = tmp_path / 'experiment.yaml';descriptor.write_text('target: fixture\n')
    pools = {}
    for key in ('MERLIN_COMPUTE_CPU_SLOTS', 'MERLIN_GSIM_SLOT_ROOT'):
        path = tmp_path / key;path.mkdir();pools[key] = str(path)
    calls = []
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()
    monkeypatch.setattr(L.subprocess, 'Popen', popen)
    def start(**overrides):
        args = dict(state_path=state, descriptor=descriptor, repo=tmp_path, python='/selected/python',
            environment={**pools, 'UNCHANGED': 'value'}, output=tmp_path/'calibration',
            directory=tmp_path/'watcher', cpus=sorted(os.sched_getaffinity(0)), request_epoch=False)
        args.update(overrides)
        return L.start(**args)
    return SimpleNamespace(start=start, calls=calls, state=state, data=data, profile=profile,
        selected=selected, pools=pools, root=tmp_path)


def test_exact_profile_existing_pools_and_clock_preserved(owner):
    f=owner;before=f.state.read_bytes();watcher=f.start()
    argv,kwargs=f.calls[0]
    assert argv[argv.index('--profile')+1]==str(f.profile)
    assert argv[argv.index('--profile-sha256')+1]==f.selected['sha256']
    assert argv[argv.index('--state')+1]==str(f.state)
    assert '--evidence-only' in argv
    assert {key:kwargs['env'][key] for key in f.pools}==f.pools
    assert kwargs['env']['UNCHANGED']=='value'
    assert kwargs['pass_fds']==(watcher.lease.fileno(),)
    assert f.state.read_bytes()==before
    watcher.stop()
    assert watcher.report['status']=='stopped'
    assert watcher.stop_file.is_file()


def test_single_owner_skips_duplicates_without_creating_pools(owner):
    f=owner;first=f.start();second=f.start(directory=f.root/'duplicate')
    assert second.process is None and second.report['status']=='skipped_existing_owner'
    assert len(f.calls)==1 and not (f.root/'duplicate').exists()
    first.stop()
    third=f.start(directory=f.root/'successor');third.stop()
    assert len(f.calls)==2


@pytest.mark.parametrize('damage',['profile','selection','clock','pools'])
def test_changed_or_incomplete_admission_cannot_spawn(owner,damage):
    f=owner;overrides={}
    if damage=='profile':f.profile.write_text('{"changed":true}')
    elif damage=='selection':overrides['profile']={**f.selected,'sha256':'x'*64}
    elif damage=='clock':
        f.data['started_at_epoch']=None;f.state.write_text(json.dumps(f.data))
    else:overrides['environment']={}
    with pytest.raises(ValueError):f.start(**overrides)
    assert not f.calls


def test_absent_profile_is_explicit_skip(owner):
    f=owner;f.data['policy'].pop('native_calibration');f.state.write_text(json.dumps(f.data))
    watcher=f.start()
    assert watcher.report['status']=='skipped_no_profile'
    assert not f.calls and watcher.stop()==watcher.report
    assert L.start(state_path=None,descriptor=None,repo=None,python=None,environment={},
        output=None,directory=None).report['status']=='skipped_no_campaign'


def test_unfinished_native_operation_retains_lease_and_reports_unknown(owner):
    f=owner;watcher=f.start()
    def timeout(**kwargs):raise subprocess.TimeoutExpired('fixture',kwargs['timeout'])
    watcher.process.wait=timeout
    with pytest.raises(RuntimeError,match='shutdown UNKNOWN'):watcher.stop()
    assert watcher.lease is not None
    assert f.start(directory=f.root/'duplicate').report['status']=='skipped_existing_owner'
    watcher.process.returncode=0;watcher.stop()
    assert watcher.lease is None


def test_stop_boundary_is_local_and_does_not_stop_campaign(tmp_path):
    stop=tmp_path/'stop.json'
    with L.stop_scope(stop):
        L.raise_if_stopping()
        stop.write_text('{}')
        with pytest.raises(L.StopRequested):L.raise_if_stopping()
    L.raise_if_stopping()


@pytest.mark.parametrize('status',['waiting_for_independent_variants','blocked_structural_model_admission'])
def test_service_keeps_waiting_or_denied_until_its_host_requests_stop(tmp_path,monkeypatch,status):
    from merlin_experiments.phase1 import context
    stop=tmp_path/'stop.json';output=tmp_path/'out';calls=[]
    monkeypatch.setattr(context,'load_context',lambda *a,**k:object())
    monkeypatch.setattr(S.joint_control,'stop_reason',lambda:None)
    def step(**kwargs):
        calls.append(kwargs);stop.write_text('{}')
        return dict(status=status)
    monkeypatch.setattr(S,'step',step)
    assert S.main(['--profile',str(tmp_path/'profile'),'--profile-sha256','a'*64,
        '--state',str(tmp_path/'state'),'--descriptor',str(tmp_path/'descriptor'),'--repo',str(tmp_path),
        '--output',str(output),'--stop-file',str(stop),'--evidence-only'])==0
    assert len(calls)==1 and calls[0]['request_epoch'] is False
    assert json.loads((output/'status.json').read_text())['status']=='watcher_host_stopped'


def test_ordinary_owner_uses_same_lifecycle_with_existing_epoch_policy(tmp_path,monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_owner as O
    process=Process();calls=[]
    def start(**kwargs):
        calls.append(kwargs)
        return L.Watcher(process,{'status':'watcher_started'})
    monkeypatch.setattr(L,'start',start)
    monkeypatch.setattr(O,'environment',lambda selection:{'EXACT':'environment'})
    selection=dict(run_dir=str(tmp_path),cwd=str(tmp_path),argv=['selected-python','--descriptor','descriptor'],taskset_cpus=[0])
    watcher=O.start_watcher(selection,{'path':'profile','sha256':'hash'},tmp_path/'out',tmp_path/'watcher')
    assert watcher.process is process
    assert calls[0]['environment']=={'EXACT':'environment'}
    assert 'request_epoch' not in calls[0]  # Existing ordinary owner default.


def test_construction_drains_watcher_before_broker_teardown(tmp_path,monkeypatch):
    from merlin_experiments.phase1 import construction_study as C, joint_control
    from merlin_experiments.phase1.feedback import lifecycle as FL
    from merlin_experiments.phase1.providers import execution as EX
    from merlin.targetgen.sandbox import bwrap as BW
    events=[];run=tmp_path/'run';run.mkdir()
    options=SimpleNamespace(sim_max_jobs=2,sim_per_capsule_timeout=120,max_wall_s=10)
    context=SimpleNamespace(repo=tmp_path,descriptor=tmp_path/'descriptor')
    request=SimpleNamespace(options=options,context=context,resolved_tools=lambda:[])
    prepared=SimpleNamespace(request=request,verify_inputs=lambda:None,workspace=tmp_path,run_dir=run,bundle={},
        environment={'run_config':{'construction_study':{'sha256':'plan'}}},public_root=tmp_path,policy_root=tmp_path,contract_root=tmp_path)
    monkeypatch.setattr(C,'validate',lambda options:None)
    monkeypatch.setattr(C.actions,'load',lambda path:({},'plan'))
    monkeypatch.setattr(joint_control,'start',lambda:events.append('clock'))
    monkeypatch.setattr(BW,'frozen_selected_rtl_facts',lambda *a,**k:None)
    monkeypatch.setattr(FL,'start_brokers',lambda *a,**k:events.append('broker_start'))
    monkeypatch.setattr(FL,'stop_brokers',lambda *a,**k:events.append('broker_stop'))
    monkeypatch.setattr(EX,'sandbox_command',lambda *a,**k:'fixture-author-not-executed')
    def start(**kwargs):
        events.append('watcher_start');assert kwargs['request_epoch'] is False
        return SimpleNamespace(report={'status':'watcher_started'},stop=lambda:events.append('watcher_stop'))
    monkeypatch.setattr(L,'start',start)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(C.os,'killpg',lambda *a:None)
    assert C.execute(prepared,timing_file=None)==0
    assert events==['clock','broker_start','watcher_start','watcher_stop','broker_stop']
