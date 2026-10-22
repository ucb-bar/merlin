"""Native snapshot→certificate request→ledger closure, without a simulator."""
import io
import json
import pytest
from copy import deepcopy

from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1.feedback import promotion as P, joint_physical_correctness as Q
from merlin_experiments.phase1.brokers import simjob
from test_current_source_promotions import workspace


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(joint_control,'selected',lambda:{'policy':{'gsim_correctness':{'capsules':['A','B','refuse']}}})
    monkeypatch.setattr(P,'cert_sim',lambda *a,**k:'gsim')
    monkeypatch.setattr(P,'cert_instrument',lambda *a,**k:'a'*64)
    ws=workspace(tmp_path/'ws');(ws/'.qa_channel').mkdir()
    row={'capsule':'A','status':'pass','compiler_source':{'status':'stable'},'numeric':{'status':'pass'},
         'tiers':{t:{'status':'pass','engine':'spike'} for t in ('L0','L1','L2')}}
    return ws,row


def promote(ws,row):
    return P.promote(ws,ws/'.qa_channel',{'per_capsule':[{'capsule':row['capsule'],'pass':True}]},
        'L2','L3',set(),io.StringIO(),context=None,closed_loop_rows={row['capsule']:row})


def test_original_ready_queues_real_gsim_l3_once_and_waits_for_native_certificate(tmp_path,monkeypatch):
    ws,row=setup(tmp_path,monkeypatch)
    assert promote(ws,row)==['A']
    requests=list((ws/'.qa_channel').glob('simreq*.json'));assert len(requests)==1
    request=json.loads(requests[0].read_text())
    assert request['sim']=='gsim' and request['tiers']=='L3' and request['joint_physical_correctness']
    assert request['submission_snapshot'] and request['submission_digest']==P._submission_digest(ws)
    report=json.loads((ws/'qa/joint_physical_correctness.json').read_text())
    assert report['required']==3 and report['pending']==3 and not report['complete']
    assert promote(ws,row)==[]  # Same exact native identity already pending.
    # Source edits neither cancel the required queue nor mutate its frozen compiler.
    (ws/'submission/compiler.py').write_text('def main(): return 999\n')
    admitted=list(simjob._admitted_promotions(requests,ws,loop_tier='L2',cert_tier='L3',context=None))
    assert admitted==requests
    native={'per_capsule':[{'capsule':'A','pass':True,'tiers':{'L3':'pass'}}]}
    assert P.record_cert(ws,native,'L3',identity=request['identity'],source_identity_verified=True)==['A=pass']
    report=json.loads((ws/'qa/joint_physical_correctness.json').read_text())
    assert report['passed']==1 and report['pending']==2 and not report['complete']
    assert report['source_sha256']==request['submission_digest']!=P._submission_digest(ws)


def test_scale_incomplete_prerequisites_and_refusal_never_launch_gsim(tmp_path,monkeypatch):
    ws,row=setup(tmp_path,monkeypatch)
    for mutate in ('scale','numeric','structural'):
        bad=deepcopy(row)
        if mutate=='scale':bad['capsule']='scale77'
        elif mutate=='numeric':bad['numeric']['status']='fail'
        else:bad['tiers']['L1']['status']='unknown'
        assert promote(ws,bad)==[]
    refusal={'capsule':'refuse','status':'pass','declined':True,'expected_refusal':{'status':'met'}}
    assert promote(ws,refusal)==[]
    assert not list((ws/'.qa_channel').glob('simreq*.json'))
    report=json.loads((ws/'qa/joint_physical_correctness.json').read_text())
    assert report['expected_refusal_satisfied']==['refuse'] and report['passed']==0 and report['pending']==2


def test_closed_receipt_is_required_and_instrument_tampering_still_refuses(tmp_path,monkeypatch):
    ws,row=setup(tmp_path,monkeypatch)
    assert P.promote(ws,ws/'.qa_channel',{'per_capsule':[{'capsule':'A','pass':True}]},
        'L2','L3',None,io.StringIO(),context=None)==[]
    promote(ws,row)
    requests=list((ws/'.qa_channel').glob('simreq*.json'))
    monkeypatch.setattr(P,'cert_instrument',lambda *a,**k:'b'*64)
    assert list(simjob._admitted_promotions(requests,ws,loop_tier='L2',cert_tier='L3',context=None))==[]


@pytest.mark.parametrize("pending_l3", [False, True])
def test_native_completed_callback_reopens_receipt_and_enqueues_physical(tmp_path,monkeypatch,pending_l3):
    from types import SimpleNamespace
    from merlin.targetgen import capsule_common
    from merlin_experiments.phase1.feedback import loop_grading as L
    ws,row=setup(tmp_path,monkeypatch)
    if pending_l3:
        row = split_stage(ws, row)
    runs=tmp_path/'runs'
    result=runs/'runs'/'suite'/'A'/'capsule_result.json'
    result.parent.mkdir(parents=True);result.write_text(json.dumps(row))
    te=SimpleNamespace(target='test',effective_exclusions=lambda names:())
    monkeypatch.setattr(L,'load_target_experiment',lambda _:te)
    monkeypatch.setattr(L,'_suite_for',lambda _:'suite')
    monkeypatch.setattr(capsule_common,'discover_capsules',lambda *a,**k:[{'name':'A'}])
    context=SimpleNamespace(descriptor=tmp_path/'descriptor')
    inputs=SimpleNamespace(context=context,public_root=tmp_path/'capsules',contract=None,promotion_root=None)
    callback=L._completed_loop_promoter(ws,ws,runs,'L2','L3',{'A'},inputs=inputs)
    from merlin.targetgen.capsule_scheduling import BoundaryReadiness, run_pipelined
    results = run_pipelined([{'name': 'A'}], lambda cap, workers: row,
        max_workers=1, on_complete=callback, boundary_readiness=BoundaryReadiness({'A': 'route'}))
    assert results[0]['boundary_readiness']['authority'] == 'ordering_only'
    assert 'boundary_readiness' not in row
    request=json.loads(next((ws/'.qa_channel').glob('simreq*.json')).read_text())
    assert request['sim']=='gsim' and request['tiers']=='L3'
    assert request['submission_snapshot']
    verdict=Q.attach(ws/'qa/verdict.json',{'feedback_brief':{}})
    assert verdict['feedback_brief']['physical_correctness']['pending']==3
    assert verdict['joint_physical_correctness']['complete'] is False


def test_projection_reuses_only_matching_native_execution_and_instrument(tmp_path,monkeypatch):
    ws,_=setup(tmp_path,monkeypatch)
    state={}
    P._record(state,'A','L3',{'status':'pass','digest':'old','instrument':'a'*64,'execution_digest':'c'*64})
    def report():
        Q.project(ws,state,source='new',instrument='a'*64)
        return json.loads((ws/'qa/joint_physical_correctness.json').read_text())
    assert report()['passed']==0
    P._record(state,'A','L2',{'status':'pass','digest':'new','execution_digest':'d'*64})
    assert report()['passed']==0
    P._record(state,'A','L2',{'status':'pass','digest':'new','execution_digest':'c'*64})
    reused=report()
    assert reused['passed']==1
    assert reused['obligations']['A']['certificate_source_sha256']=='old'
    assert reused['obligations']['A']['reused_execution_identity']=='c'*64
    Q.project(ws,state,source='new',instrument='b'*64)
    assert json.loads((ws/'qa/joint_physical_correctness.json').read_text())['passed']==0


def split_stage(ws, row):
    from merlin.common.tree_hash import hash_tree
    row = deepcopy(row)
    reason = ("this capsule declares required oracle tier L3 (elaborated_rtl) "
              "but this phase supplies no adapter for it; reachable tiers here: ['L2']")
    row['status'] = 'incomplete'
    row['tiers']['L3'] = {'status': 'unavailable', 'mandatory': True, 'reason': reason}
    row['failure'] = {'plane': 'oracle_unavailable', 'category': 'NOT_RUN_IS_NOT_PASS',
                      'tier': 'L3', 'tier_status': 'unavailable', 'tier_reason': reason}
    row['trace_check'] = {'status': 'pass', 'trace_requirement_status': 'pass',
                          'violations': [], 'unavailable_requirements': []}
    row['numeric'].update(mismatch_count=0, missing_outputs=[],
                         per_output={'Y': {'status': 'pass', 'n_elements': 8}})
    row['compiler_source']['candidate_sha256'] = hash_tree(ws / 'submission')['sha256']
    return row


def test_split_l3_prerequisites_queue_but_do_not_grant_completeness(tmp_path, monkeypatch):
    ws, row = setup(tmp_path, monkeypatch)
    row = split_stage(ws, row)
    original = deepcopy(row)
    assert promote(ws, row) == ['A']
    request = json.loads(next((ws / '.qa_channel').glob('simreq*.json')).read_text())
    assert request['sim'] == 'gsim' and request['tiers'] == 'L3'
    report = json.loads((ws / 'qa/joint_physical_correctness.json').read_text())
    assert report['passed'] == 0 and report['pending'] == 3 and not report['complete']
    assert row == original
    assert Q._passed(row) is None  # Scheduling projection is not curriculum/final pass.
    assert promote(ws, row) == []
    native = {'per_capsule': [{'capsule': 'A', 'pass': True, 'tiers': {'L3': 'pass'}}]}
    assert P.record_cert(ws, native, 'L3', identity=request['identity'],
                         source_identity_verified=True) == ['A=pass']
    report = json.loads((ws / 'qa/joint_physical_correctness.json').read_text())
    assert report['passed'] == 1 and report['pending'] == 2 and not report['complete']


def test_split_l3_prerequisites_reject_other_failures_or_unbound_evidence(tmp_path, monkeypatch):
    ws, row = setup(tmp_path, monkeypatch)
    row = split_stage(ws, row)
    bad_rows = []
    def bad():
        item = deepcopy(row); bad_rows.append(item); return item
    bad()['tiers']['L3']['status'] = 'fail'
    bad()['failure']['tier'] = 'L1'
    bad()['tiers']['L1']['status'] = 'unavailable'
    bad()['tiers']['L4'] = {'mandatory': True, 'status': 'unavailable'}
    bad()['tiers']['L3']['reason'] = 'timed out'
    bad()['numeric']['per_output'] = {}
    bad()['numeric']['missing_outputs'] = ['Y']
    bad()['numeric']['status'] = 'unknown'
    bad()['numeric']['per_output']['Y']['status'] = 'fail'
    bad()['trace_check']['status'] = 'fail'
    bad()['trace_check']['violations'] = ['trace does not open with a FENCE']
    bad()['trace_check']['trace_requirement_status'] = 'unavailable_fields'
    bad()['trace_check'] = {}
    bad()['compiler_source']['status'] = 'source_changed'
    bad()['compiler_source']['candidate_sha256'] = 'f' * 64
    for item in bad_rows:
        assert promote(ws, item) == []
    assert not list((ws / '.qa_channel').glob('simreq*.json'))
