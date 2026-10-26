import json
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import joint_curriculum as C
from merlin_experiments.phase1.feedback import joint_test_scope as S


def corpus(tmp_path):
    records = []
    for i in range(20):
        name = f'case{i:02}'
        root = tmp_path / name
        root.mkdir()
        (root/'capsule.yaml').write_text(json.dumps({'name':name,'label':'public',
            'inputs':[{'shape':[i+1]}], 'operation':{'op':'matmul'}}))
        (root/'input.bin').write_bytes(bytes([i]))
        records.append(('matmul', i+1, name, root))
    return records


def policy():
    return {'correctness_curriculum':{'original_capsules':[f'case{i:02}' for i in range(18)]},
        'gsim':{'witness_capsules':['case05','case19']},
        'operator_measurement_policy':{'witness_questions':{'case05':'missing transfer regime'}}}


def result(root, name):
    p = root/'runs'/'suite'/name/'capsule_result.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'capsule':name,'status':'pass','compiler_source':{'sha256':'a'*64},
        'numeric':{'status':'pass'},'tiers':{t:{'status':'pass','engine':'spike'} for t in ('L0','L1','L2')}}))
    return p


def test_reviewed_witness_priority_preserves_least_attempt_fairness(tmp_path):
    records=corpus(tmp_path)
    chosen,report=C.plan(records,policy=policy(),attempts={},passed={},hint=['case19'],limit=4)
    assert chosen[0][2]=='case05'
    assert chosen[1][2]=='case19'
    assert all(r[2]!='case18' for r in chosen)
    attempts={r[2]:1 for r in chosen}
    next_batch,_=C.plan(records,policy=policy(),attempts=attempts,passed={},hint=[r[2] for r in chosen],limit=4)
    assert not {r[2] for r in chosen}&{r[2] for r in next_batch}
    assert len(report['pending'])==20 and report['original_unresolved']==18
    assert not report['certified_complete']


def test_exact_native_receipt_reuse_and_each_invalidation(tmp_path):
    records=corpus(tmp_path)
    runs=tmp_path/'run';home=tmp_path/'progress'
    identity={'compiler':'first','runtime':'spike1','config':'device1','numeric_policy':'exact'}
    p=C.Progress(home,identity=identity,runs_root=runs)
    p.select(records,policy=policy(),hint=[])
    receipt=result(runs,'case05');p.completed({'capsule':'case05'})
    chosen,report=p.select(records,policy=policy(),hint=[])
    assert 'case05' in report['exact_correctness_reused']
    assert all(r[2]!='case05' for r in chosen)
    assert next(w for w in report['witness_prerequisites'] if w['capsule']=='case05')['dispatch_authorized'] is False
    for key in identity:
        changed=C.Progress(home,identity={**identity,key:'different'},runs_root=runs)
        assert not changed.select(records,policy=policy(),hint=[])[1]['exact_correctness_reused']
    (records[5][3]/'input.bin').write_bytes(b'changed numeric input')
    assert not p.select(records,policy=policy(),hint=[])[1]['exact_correctness_reused']
    (records[5][3]/'input.bin').write_bytes(bytes([5]))
    receipt.write_text(receipt.read_text()+'\n')
    assert not p.select(records,policy=policy(),hint=[])[1]['exact_correctness_reused']


def test_incomplete_and_gated_results_never_advance_correctness(tmp_path):
    records=corpus(tmp_path);runs=tmp_path/'run'
    p=C.Progress(tmp_path/'progress',identity={'compiler':'same'},runs_root=runs)
    p.select(records,policy=policy(),hint=[])
    receipt=result(runs,'case05');row=json.loads(receipt.read_text());row['numeric']['status']='unknown'
    receipt.write_text(json.dumps(row));p.completed({'capsule':'case05'})
    assert not p.report()['exact_correctness_reused']
    before=json.loads((p.root/'progress.json').read_text())
    p.completed({'capsule':'case05','background_deferred':True})
    assert json.loads((p.root/'progress.json').read_text())==before


def test_curriculum_hook_keeps_milestone_and_original_denominator(tmp_path,monkeypatch):
    records=corpus(tmp_path);candidate=tmp_path/'submission';candidate.mkdir()
    (candidate/'joint-test-scope.json').write_text('{"capsules":["case19"]}')
    monkeypatch.setattr(S.joint_control,'selected',lambda:{'policy':policy()})
    monkeypatch.delenv('MERLIN_JOINT_GRADE_MILESTONE',raising=False)
    progress=C.Progress(tmp_path/'progress',identity=None,runs_root=tmp_path/'runs')
    roots,scope=S.select(tmp_path,submission=candidate,curriculum=progress)
    assert len(roots)==20 and scope['required_development']==20 and len(scope['pending'])==20
    assert scope['scope']=='rolling' and 'case19' in scope['selected']
    monkeypatch.setenv('MERLIN_JOINT_GRADE_MILESTONE','1')
    roots,scope=S.select(tmp_path,submission=candidate,curriculum=progress)
    assert len(roots)==20 and scope['scope']=='milestone'


def test_invalid_operator_roster_cannot_drop_original_cases(tmp_path):
    with pytest.raises(ValueError,match='selected core'):
        C.plan(corpus(tmp_path),policy={'correctness_curriculum':{'original_capsules':['hidden']}},
               attempts={},passed={})


def test_joint_advisory_interval_uses_same_native_bounded_path(monkeypatch,tmp_path):
    from merlin_experiments.phase1.feedback import loop_grading as L
    calls=[]
    monkeypatch.setattr(L.JTS,'focused',lambda:True)
    monkeypatch.setattr(L,'fast_grade',lambda *a,**kw:calls.append((a,kw)) or {'bounded':True})
    result=L.grade(tmp_path,tmp_path,901,False,12,inputs=object(),advisory_background=True)
    assert result=={'bounded':True} and len(calls)==1
    assert calls[0][0][3]==12


def whole_program_result():
    return {'status':'pass','compiler_source':{'status':'stable'},
        'numeric':{'status':'pass','golden_source':'independent_operation_golden','mismatch_count':0,
            'missing_outputs':[],'per_output':{'Y':{'status':'pass','n_elements':12}}},
        'tiers':{'L0':{'status':'skipped','not_applicable':True,
                'reason':'command-buffer reference cannot interpret scalar regions of a whole-program ELF'},
            'L1':{'status':'skipped','not_applicable':True,
                'reason':'command-buffer simulate cannot execute scalar regions of a whole-program ELF'},
            'L2':{'status':'pass','engine':'spike'}}}


def test_whole_program_uses_actual_independent_spike_outputs_not_skipped_numeric():
    from copy import deepcopy
    row=whole_program_result()
    assert C._passed(row)=='spike_correctness'
    for field,value in [('status','skipped'),('status','unknown'),('per_output',{}),('missing_outputs',['Y'])]:
        bad=deepcopy(row);bad['numeric'][field]=value
        assert C._passed(bad) is None
    for field,value in [('not_applicable',False),('reason','not run yet'),('status','unavailable')]:
        bad=deepcopy(row);bad['tiers']['L0'][field]=value
        assert C._passed(bad) is None
    bad=deepcopy(row);bad['tiers']['L2']['status']='skipped'
    assert C._passed(bad) is None


def test_queue_preflight_and_actual_admission_have_distinct_fairness(tmp_path):
    records=corpus(tmp_path)
    p=C.Progress(tmp_path/'progress',identity=None,runs_root=tmp_path/'runs')
    first,_=p.select(records,policy=policy(),hint=[])
    for r in first:p.completed({'capsule':r[2],'status':'gated'})
    before=json.loads((p.root/'progress.json').read_text())
    report=p.report()
    assert json.loads((p.root/'progress.json').read_text())==before
    assert before['members']=={}  # Queuing/preflight are not execution attempts.
    assert len(report['pending'])==20 and not report['exact_correctness_reused']
    assert report['next_batch'][0]==first[0][2]
    assert report['admitted_this_snapshot']==[]
    for r in first[:18]:
        p.started({'name':r[2]})
        p.started({'name':r[2]})  # Duplicate callback cannot age a case twice.
    after=json.loads((p.root/'progress.json').read_text())
    assert all(v['admissions']==1 and v['attempts']==0 for v in after['members'].values())
    report=p.report()
    assert set(report['unadmitted_pending'])=={'case17','case18'}
    # Unadmitted originals and extra cases precede already-admitted witnesses.
    assert set(report['next_batch'][:2])=={'case17','case18'}


def test_frozen_inputs_reverified_only_before_actual_reuse(tmp_path):
    records=corpus(tmp_path);runs=tmp_path/'run';calls=[]
    p=C.Progress(tmp_path/'progress',identity={'compiler':'same'},runs_root=runs,
                 verify_reuse=lambda:calls.append('verified'))
    p.select(records,policy=policy(),hint=[])
    assert calls==[]
    result(runs,'case05');p.completed({'capsule':'case05'})
    assert p.report()['exact_correctness_reused']==['case05'] and calls==['verified']
    def refuse():raise ValueError('changed frozen inputs')
    p.verify_reuse=refuse
    assert p.report()['exact_correctness_reused']==[]


def test_rolling_admission_pins_deadline_scope_and_original_case_timeout(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from merlin.targetgen import capsule_scheduling as scheduler
    progress=C.Progress(tmp_path/'progress',identity=None,runs_root=tmp_path/'runs')
    progress.select(corpus(tmp_path),policy=policy(),hint=[])
    observed={}
    monkeypatch.setattr(C.time,'monotonic',lambda:10)
    def measured(history, **kwargs):
        observed['history']=(history,kwargs)
        return {'case05':20},{}
    @contextmanager
    def rolling(**kwargs):
        observed.update(kwargs)
        kwargs['on_start']({'name':'case05'})
        yield
    monkeypatch.setattr(scheduler,'measured_costs',measured)
    monkeypatch.setattr(scheduler,'rolling_admission',rolling,raising=False)
    with C.admission(progress,timeout=900,remaining_s=120,history=['same-run']) as timeout:
        assert timeout==900
    assert observed['admission_deadline_monotonic']==130
    assert observed['admission_window_s']==120
    assert observed['admission_allow_oversized_first'] is True
    assert observed['history']==(['same-run'],{'requested_tiers':{'L2'},'engine':'spike'})
    assert observed['admission_costs']=={'case05':20}
    assert progress.report()['admitted_this_snapshot']==['case05']
    assert json.loads((progress.root/'progress.json').read_text())['in_flight']=={}


def test_live_same_snapshot_ownership_excludes_only_actual_starts_and_releases(tmp_path):
    records=corpus(tmp_path)
    home=tmp_path/'progress'
    identity={'compiler':'frozen-a','runtime':'spike-a','numerics':'exact'}
    first=C.Progress(home,identity=identity,runs_root=tmp_path/'run1')
    second=C.Progress(home,identity=identity,runs_root=tmp_path/'run2')
    first.select(records,policy=policy(),hint=[])
    first.started({'name':'case05'})
    chosen,report=second.select(records,policy=policy(),hint=[])
    assert len(chosen)==19 and all(r[2]!='case05' for r in chosen)
    assert report['in_flight_pending']==['case05'] and 'case05' in report['pending']
    assert not report['exact_correctness_reused']
    # Even a caller with a stale queued list cannot dispatch a duplicate.
    with pytest.raises(RuntimeError,match='already owned'):
        second.started({'name':'case05'})
    ledger=json.loads((home/'progress.json').read_text())
    assert ledger['members']['case05']['admissions']==1
    assert set(ledger['members'])=={'case05'}  # 19 queued cases were not attempted.
    second.release()  # A different owner cannot retire the first owner's work.
    assert len(second.select(records,policy=policy(),hint=[])[0])==19
    # Native completion without a usable receipt releases ownership but gives no pass.
    first.completed({'capsule':'case05','status':'fail'})
    assert len(second.select(records,policy=policy(),hint=[])[0])==20
    assert not second.report()['exact_correctness_reused']
    second.started({'name':'case05'})
    second.release()  # Trusted full owner drain also retires unfinished callback claims.
    assert json.loads((home/'progress.json').read_text())['in_flight']=={}


def test_ownership_and_passes_cannot_cross_compiler_or_runtime_identity(tmp_path):
    records=corpus(tmp_path);home=tmp_path/'progress'
    old=C.Progress(home,identity={'compiler':'a','runtime':'one'},runs_root=tmp_path/'old')
    old.select(records,policy=policy(),hint=[]);old.started({'name':'case05'})
    for identity in ({'compiler':'b','runtime':'one'},{'compiler':'a','runtime':'two'}):
        new=C.Progress(home,identity=identity,runs_root=tmp_path/'new')
        chosen,report=new.select(records,policy=policy(),hint=[])
        assert len(chosen)==20 and report['in_flight_pending']==[]
    result(old.runs_root,'case05');old.completed({'capsule':'case05'})
    assert old.report()['exact_correctness_reused']==['case05']
    assert not new.report()['exact_correctness_reused']


def test_unqualified_execution_identity_deduplicates_without_receipt_reuse(tmp_path,monkeypatch):
    records=corpus(tmp_path);home=tmp_path/'progress'
    identity={'compiler':'same','frozen_runtime_selection':'one'}
    first=C.Progress(home,identity=None,ownership_identity=identity,runs_root=tmp_path/'a')
    second=C.Progress(home,identity=None,ownership_identity=identity,runs_root=tmp_path/'b')
    first.select(records,policy=policy(),hint=[]);first.started({'name':'case05'})
    monkeypatch.setattr(C.time,'time_ns',lambda:10**30)
    assert len(second.select(records,policy=policy(),hint=[])[0])==19  # No elapsed-time expiry.
    result(first.runs_root,'case05');first.completed({'capsule':'case05'})
    assert len(second.select(records,policy=policy(),hint=[])[0])==20
    assert not second.report()['exact_correctness_reused']


def test_drained_admission_scope_releases_claims_after_grade_exception(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from merlin.targetgen import capsule_scheduling as scheduler
    p=C.Progress(tmp_path/'progress',identity={'compiler':'same'},runs_root=tmp_path/'runs')
    p.select(corpus(tmp_path),policy=policy(),hint=[])
    monkeypatch.setattr(scheduler,'measured_costs',lambda *a,**kw:({},{}))
    @contextmanager
    def rolling(**kwargs):
        kwargs['on_start']({'name':'case05'})
        yield
    monkeypatch.setattr(scheduler,'rolling_admission',rolling)
    with pytest.raises(ValueError,match='grade failed after draining'):
        with C.admission(p,timeout=30,remaining_s=120,history=[]):
            raise ValueError('grade failed after draining')
    assert json.loads((p.root/'progress.json').read_text())['in_flight']=={}


def test_scheduler_priority_matches_witness_queue_without_physical_authority(tmp_path):
    records = corpus(tmp_path)
    progress = C.Progress(tmp_path/'progress', identity=None, runs_root=tmp_path/'runs')
    chosen, report = progress.select(records, policy=policy(), hint=['case18'])
    assert [r[2] for r in chosen[:2]] == ['case05', 'case19']
    assert progress.priority({'name': 'case19'}) < progress.priority({'name': 'case00'})
    assert report['original_development_required'] == 18
    assert set(report['pending']) == {r[2] for r in records}
    assert all(not w['dispatch_authorized'] for w in report['witness_prerequisites'])
    progress.started({'name': 'case19'})
    progress.release('case19')
    chosen, report = progress.select(records, policy=policy(), hint=['case19'])
    assert chosen[-1][2] == 'case19'  # Prior attempt overrides persistent hint/witness priority.
    assert not report['exact_correctness_reused']


@pytest.mark.parametrize('remaining', [0, -5])
def test_expired_clock_admits_nothing_new_and_does_not_fail_the_grade(tmp_path, monkeypatch, remaining):
    from merlin.targetgen import capsule_scheduling as scheduler
    progress = C.Progress(tmp_path/'progress', identity=None, runs_root=tmp_path/'runs')
    progress.select(corpus(tmp_path), policy=policy(), hint=[])
    monkeypatch.setattr(scheduler, 'measured_costs', lambda *a, **kw: ({}, {}))
    ran = []
    with C.admission(progress, timeout=30, remaining_s=remaining, history=[]):
        results = scheduler.run_pipelined(
            [{'name': 'case05'}, {'name': 'case06'}],
            lambda cap, workers: ran.append(cap['name']) or {'capsule': cap['name'], 'status': 'pass'},
            max_workers=2,
        )
    assert ran == []  # even the oversized-first allowance starts nothing after the deadline
    assert len(results) == 2 and all(r.get('status') != 'pass' for r in results)
    assert progress.admission_window_s is None
    assert json.loads((progress.root/'progress.json').read_text())['in_flight'] == {}
