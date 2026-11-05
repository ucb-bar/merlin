"""No author, fitter, simulator or physical measurement is launched."""
import copy
import json
import os
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_model_activation as A
from merlin_experiments.phase1 import joint_control as JC


@pytest.fixture
def owner(tmp_path, monkeypatch):
    def write(name, value):
        return A._write(tmp_path / name, value)
    initial = write('selection0.json', {'schema': A.PERF.SCHEMA, 'runtime': {'path':'runtime0'}, 'members':{}})
    evidence = write('evidence.json', {'admitted':True})
    candidate = write('selection1.json', {'schema': A.PERF.SCHEMA, 'runtime': {'path':'runtime0'},
        'timing_epochs':[{'calibration_evidence':evidence}], 'members':{}})
    state = tmp_path / 'joint_campaign.json'
    state.write_text(json.dumps({'policy_sha256':'policy', 'started_at_epoch':1,
        'performance_selection':initial, 'policy':{}}))
    calls = []
    def epoch(candidate_pin, incumbent_pin, selected_state):
        calls.append((candidate_pin, incumbent_pin))
        return A.PERF._read(candidate_pin)['timing_epochs'][-1]
    monkeypatch.setattr(A, '_accepted_epoch', epoch)
    env = {JC.ENV:str(state), A.PERF.ENV:json.dumps(initial), 'KEEP':'yes'}
    return SimpleNamespace(root=tmp_path, state=state, initial=initial, candidate=candidate,
        evidence=evidence, env=env, calls=calls, write=write)


def test_acceptance_only_affects_new_admissions_and_preserves_inflight(owner):
    f = owner; original_state = f.state.read_bytes(); parent_env = dict(f.env)
    a = A.request_environment(f.env, channel='selfcheck', request_id='A', request_bytes=b'A')
    accepted = A.accept(state_path=f.state, selection_pin=f.candidate, incumbent_pin=f.initial, review_id='trusted')
    b = A.request_environment(f.env, channel='selfcheck', request_id='B', request_bytes=b'B')
    assert json.loads(a[A.PERF.ENV]) == f.initial
    assert json.loads(b[A.PERF.ENV]) == f.candidate
    assert A.request_environment(f.env, channel='selfcheck', request_id='A', request_bytes=b'A') == a
    assert A.request_binding(a)['selection'] == f.initial
    assert A.request_binding(b)['accepted_record'] == accepted['record']
    assert f.state.read_bytes() == original_state and f.env == parent_env
    assert not (f.root/'joint_epoch_stop.json').exists()
    assert A.accept(state_path=f.state, selection_pin=f.candidate, incumbent_pin=f.initial,
                    review_id='trusted')['status'] == 'already_accepted_for_future_requests'
    assert len(f.calls) == 1


def test_stale_incumbent_and_request_mutation_are_refused(owner):
    f = owner
    A.request_environment(f.env, channel='selfcheck', request_id='A', request_bytes=b'A')
    with pytest.raises(ValueError, match='identity changed'):
        A.request_environment(f.env, channel='selfcheck', request_id='A', request_bytes=b'changed')
    A.accept(state_path=f.state, selection_pin=f.candidate, incumbent_pin=f.initial, review_id='trusted')
    second = f.write('selection2.json', {'timing_epochs':[{'calibration_evidence':f.evidence}]})
    with pytest.raises(ValueError, match='superseded'):
        A.accept(state_path=f.state, selection_pin=second, incumbent_pin=f.initial, review_id='stale')
    assert A.current(f.state) == f.candidate


def test_bad_evidence_or_pointer_cannot_change_selection(owner, monkeypatch):
    f = owner
    monkeypatch.setattr(A, '_accepted_epoch', lambda *a: (_ for _ in ()).throw(ValueError('denied')))
    with pytest.raises(ValueError, match='denied'):
        A.accept(state_path=f.state, selection_pin=f.candidate, incumbent_pin=f.initial, review_id='trusted')
    assert A.current(f.state) == f.initial
    assert not (f.root/'joint_model_activation/accepted.json').exists()


def test_worker_pin_rejects_different_env_selection(owner):
    f=owner
    env=A.request_environment(f.env,channel='simjob',request_id='job',request_bytes=b'original')
    env[A.PERF.ENV]=json.dumps(f.candidate)
    with pytest.raises(ValueError,match='pins differ'):A.request_binding(env)


def test_corrupt_or_rolled_back_head_keeps_last_committed_selection(owner):
    f=owner
    accepted=A.accept(state_path=f.state,selection_pin=f.candidate,incumbent_pin=f.initial,review_id='trusted')
    head=f.root/'joint_model_activation/accepted.json'
    head.chmod(0o644);head.write_text('not json')
    env=A.request_environment(f.env,channel='selfcheck',request_id='after',request_bytes=b'after')
    assert A.request_binding(env)['selection']==f.candidate
    assert A.request_binding(env)['activation_warning']
    assert A.current(f.state)==f.candidate
    # An uncommitted orphan, even if the mutable pointer names it, is not accepted.
    orphan=A._write(f.root/'joint_model_activation/accepted-0002.json',{'selection':f.initial})
    head.write_text(json.dumps({'record':orphan}))
    assert A.current(f.state)==f.candidate


def test_entire_predecessor_chain_is_verified_and_last_valid_prefix_survives(owner):
    f=owner
    first=A.accept(state_path=f.state,selection_pin=f.candidate,incumbent_pin=f.initial,review_id='one')
    second=f.write('selection2.json',{'timing_epochs':[{'calibration_evidence':f.evidence}]})
    accepted=A.accept(state_path=f.state,selection_pin=second,incumbent_pin=f.candidate,review_id='two')
    assert A.current(f.state)==second
    head=f.root/'joint_model_activation/accepted.json';head.chmod(0o644)
    head.write_text(json.dumps({'record':first['record']}))
    assert A.current(f.state)==second  # rollback of the cache cannot rewind committed history
    path=f.root/'joint_model_activation/accepted-0002.json'
    data=json.loads(path.read_text());data['previous']=None;path.chmod(0o644);path.write_text(json.dumps(data))
    assert A.current(f.state)==f.candidate
    env=A.request_environment(f.env,channel='simjob',request_id='after',request_bytes=b'after')
    assert A.request_binding(env)['activation_warning']


def test_request_journal_cannot_be_reused_for_another_campaign_state(owner):
    f=owner
    A.request_environment(f.env,channel='selfcheck',request_id='one',request_bytes=b'one')
    other=f.root/'other-state.json';other.write_bytes(f.state.read_bytes())
    with pytest.raises(ValueError,match='identity changed'):
        A.request_environment({**f.env,JC.ENV:str(other)},channel='selfcheck',request_id='one',request_bytes=b'one')


@pytest.mark.parametrize('identity',['','a/b','x'*241,'nonascii-\u03b1'])
def test_request_identity_uses_bounded_structural_ascii(owner,identity):
    with pytest.raises(ValueError,match='bounded native'):
        A.request_environment(owner.env,channel='selfcheck',request_id=identity,request_bytes=b'one')


def test_background_grade_admission_and_callback_threads_keep_one_model(owner,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    f=owner;candidate=f.root/'candidate';candidate.mkdir()
    for key,value in f.env.items():monkeypatch.setenv(key,value)
    before=dict(os.environ);admitted=Event();release=Event()
    @A.grade_model_scope
    def grade(tick, wait=False):
        env=A.admit_grade(candidate=candidate,source_sha256='c'*64,output=f.root,
                          kind='test',identity={'tick':tick})
        if wait:
            admitted.set();assert release.wait(5)
        def callback():
            with A.PERF.environment_scope(env):
                return A.PERF.selected()[0],A.request_binding(A.PERF.environment())
        with ThreadPoolExecutor(1) as child:
            return child.submit(callback).result()
    with ThreadPoolExecutor(2) as pool:
        old=pool.submit(grade,1,True);assert admitted.wait(5)
        A.accept(state_path=f.state,selection_pin=f.candidate,incumbent_pin=f.initial,review_id='trusted')
        new=pool.submit(grade,2).result();release.set();old=old.result()
    assert old[0]==f.initial and new[0]==f.candidate
    assert old[1]['channel']==new[1]['channel']=='grade'
    assert old[1]['request_id']!=new[1]['request_id']
    assert dict(os.environ)==before and A.PERF.selected()[0]==f.initial


def test_nested_grade_requires_same_source_and_resets_context_after_failure(owner,monkeypatch):
    f=owner;candidate=f.root/'candidate';candidate.mkdir()
    for key,value in f.env.items():monkeypatch.setenv(key,value)
    @A.grade_model_scope
    def outer():
        original=A.admit_grade(candidate=candidate,source_sha256='a'*64,output=f.root,kind='outer',identity={})
        @A.grade_model_scope
        def nested():
            assert A.admit_grade(candidate=candidate,source_sha256='a'*64,output=f.root,kind='inner',identity={})==original
            A.admit_grade(candidate=candidate,source_sha256='b'*64,output=f.root,kind='bad',identity={})
        nested()
    with pytest.raises(ValueError,match='original frozen candidate'):outer()
    assert A._GRADE.get() is None
    assert A.PERF.selected()[0]==f.initial


def test_existing_tuned_coverage_and_unknown_rule():
    from merlin_experiments.phase1.feedback.joint_timing_selection import choose, within_coverage
    epoch=dict(epoch=1, mode='tuned',runtime={'path':'r1'},calibration_evidence={'path':'e'},
        coverage={'scope_sha256':'scope','memory_regime_sha256':'memory',
                  'capsule_sha256':['covered'], 'feature_ranges':{'compute':[1,4]}})
    selection=dict(runtime={'path':'r0'},timing_epochs=[epoch])
    assert choose(selection,'covered') == epoch
    assert choose(selection,'uncovered')['mode']=='untuned'
    assert not within_coverage(epoch,dict(scope_sha256='scope',domain={'memory_regime_sha256':'memory'},
                                           feature_counts={'compute':5}))


def test_producer_validation_refuses_changed_source_and_unadmitted_evidence(owner, monkeypatch):
    f=owner
    monkeypatch.undo()
    old=A.PERF._read(f.initial)
    candidate=copy.deepcopy(old)
    candidate['members']={'changed':{}}
    candidate['timing_epochs']=[{}]
    changed=f.write('changed.json',candidate)
    with pytest.raises(ValueError,match='unchanged source'):
        A._accepted_epoch(changed,f.initial,{'policy':{}})


@pytest.fixture
def accepted_producer(tmp_path,monkeypatch):
    from merlin_experiments.phase1 import joint_calibration as C, joint_runtime_revision as R
    from merlin_experiments.phase1 import joint_calibration_admission as admission
    from test_joint_calibration_eligibility import valid_grade
    write=lambda name,data:A._write(tmp_path/name,data)
    base_model={'prices':{'compute':1},'hardware':{'fixed':8}}
    new_model={'prices':{'compute':2},'hardware':{'fixed':8}}
    base_runtime=write('r0.json',{})
    tuned_runtime=write('r1.json',{})
    members={};cases=[];groups={}
    for group in range(4):
        cap=dict(name=f'public{group}',kind='layer',label='public',required_oracle_tiers=['L0','L1','L2','L3'])
        cap_pin=write(f'capsule{group}.json',cap)
        members[cap['name']]={'capsule_sha256':A.PERF.capsule_identity(cap)}
        groups[cap['name']]={'source_group_sha256':f'{group:064x}'}
        for variant in range(2):
            source=f'{variant+1:064x}'
            grade=write(f'grade{group}-{variant}.json',valid_grade(source=source))
            cases.append(dict(id=f'g{group}v{variant}',source_group_sha256=f'{group:064x}',
                split='train' if group<2 else 'held',capsule=cap_pin,result=grade,compiler_sha256=source))
    initial=write('selection0.json',dict(schema=A.PERF.SCHEMA,runtime=base_runtime,facts={},members=members))
    parameters=[dict(path=['prices','compute'],values=[1,2])]
    profile=write('profile.json',dict(performance_selection=initial,members=groups,parameters=parameters))
    plan=write('plan.json',dict(schema=C.SCHEMA,gates=C.GATES,selection=initial,runtime=base_runtime,
                              parameters=parameters,cases=cases))
    evidence=dict(status='limited_calibrated_provisional',admitted=True,design={'identifiable':True},
        selected_runtime=tuned_runtime,qualification_profile=C.GATES['qualification_profile'],
        all_declared_cases_evaluated=True,maximum_case_error=.02,p95=.02,rank_agreement=1,
        declared_held_cases=4,evaluated_held_cases=4,plan=plan)
    evidence_pin=write('outcome.json',evidence)
    epoch=dict(epoch=1,mode='tuned',qualification='limited_calibrated_provisional',runtime=tuned_runtime,
        calibration_evidence=evidence_pin,coverage=dict(scope_sha256='scope',memory_regime_sha256='memory',
            capsule_sha256=sorted(v['capsule_sha256'] for v in members.values()),feature_ranges={'compute':[1,4]}))
    selected=A.PERF._read(initial);selected['timing_epochs']=[epoch]
    candidate=write('selection1.json',selected)
    state=dict(started_at_epoch=1,policy_sha256='policy',performance_selection=initial,
        policy={'native_calibration':profile,'gsim':{'witness_capsules':list(members)}})
    state_path=tmp_path/'state.json';state_path.write_text(json.dumps(state))
    calls=[]
    monkeypatch.setattr(R,'validate',lambda *a:calls.append('ancestry'))
    monkeypatch.setattr(admission,'assert_allowed',lambda p:calls.append('fit-admission'))
    monkeypatch.setattr(A.PERF,'_runtime',lambda pin,**k:SimpleNamespace(
        model_json=json.dumps(new_model if pin==tuned_runtime else base_model),verify=lambda:calls.append('runtime')))
    return SimpleNamespace(state=state_path,initial=initial,candidate=candidate,epoch=epoch,
        new_model=new_model,calls=calls,evidence=evidence,evidence_pin=evidence_pin,write=write,selected=selected)


def test_actual_acceptance_reopens_ancestry_admission_numerics_and_reviewed_prices(accepted_producer):
    f=accepted_producer
    result=A.accept(state_path=f.state,selection_pin=f.candidate,incumbent_pin=f.initial,review_id='trusted')
    assert result['status']=='accepted_for_future_requests'
    assert f.calls==['ancestry','fit-admission','runtime']
    assert A.current(f.state)==f.candidate
    assert not (f.state.parent/'joint_epoch_stop.json').exists()


@pytest.mark.parametrize('change',['unreviewed-price','hardware','bad-held','nan-error'])
def test_actual_acceptance_cannot_broaden_prices_hardware_or_qualification(accepted_producer,change):
    f=accepted_producer
    if change=='unreviewed-price':f.new_model['prices']['compute']=3
    elif change=='hardware':f.new_model['hardware']['fixed']=99
    else:
        evidence=dict(f.evidence)
        if change=='bad-held':evidence['evaluated_held_cases']=3
        else:evidence['p95']=float('nan')
        # Deliberately malformed fixture tests refusal; production writer rejects NaN itself.
        path=f.state.parent/'mutated-outcome.json';path.write_text(json.dumps(evidence))
        from merlin.runtime import _native_files as F
        pin=F.pin(path,max_bytes=16<<20)
        selected=copy.deepcopy(f.selected);selected['timing_epochs'][-1]['calibration_evidence']=pin
        f.candidate=f.write('mutated-selection.json',selected)
    with pytest.raises(ValueError):
        A.accept(state_path=f.state,selection_pin=f.candidate,incumbent_pin=f.initial,review_id='trusted')
    assert A.current(f.state)==f.initial


def test_ordinary_launcher_uses_nonstop_mode_and_drains_before_brokers(tmp_path,monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_lifecycle as L
    from merlin_experiments.phase1.providers import execution as E
    from merlin_experiments.phase1.providers import codex_agent, agent_bridge
    from merlin_experiments.phase1 import conformance_sources
    events=[];(tmp_path/'TASK.md').write_text('sealed')
    context=SimpleNamespace(target='test',repo=tmp_path,descriptor=tmp_path/'descriptor')
    config=E.ExecutionConfig(context,E.ProviderConfig(driver='codex'),lambda:(),tmp_path/'timing')
    monkeypatch.setattr(E.FL,'start_brokers',lambda *a:events.append('brokers_start'))
    monkeypatch.setattr(E.FL,'stop_brokers',lambda *a,**k:events.append('brokers_stop'))
    monkeypatch.setattr(L,'start_for_author',lambda **k:SimpleNamespace(stop=lambda:events.append('watcher_stop')))
    monkeypatch.setattr(E,'codex_call_kwargs',lambda *a,**k:{})
    monkeypatch.setattr(E.author_subagents,'launch_kwargs',lambda *a:{})
    monkeypatch.setattr(E.codex_context,'launch_kwargs',lambda *a:{})
    monkeypatch.setattr(conformance_sources,'execution_context',lambda *a:None)
    monkeypatch.setattr(agent_bridge,'bridged_name',lambda *a:None)
    from merlin.targetgen import target_experiment
    monkeypatch.setattr(target_experiment,'load_target_experiment',lambda *a:None)
    monkeypatch.setattr(JC,'selected',lambda:None)
    monkeypatch.setattr(codex_agent,'run_round',lambda *a,**k:(events.append('author'),(0,tmp_path/'transcript'))[1])
    assert E.launch(tmp_path,tmp_path,'fixture','high','bwrap',{},0,10,config=config)[0]==0
    assert events==['brokers_start','author','watcher_stop','brokers_stop']
