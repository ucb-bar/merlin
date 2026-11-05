"""Synthetic ancestry fixtures only; no native accuracy or execution credit."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import joint_scoped_revision as S
from merlin_experiments.phase1 import joint_runtime_revision as R


@pytest.fixture
def case(tmp_path):
    def put(name, value):
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    old_model=dict(domain=dict(feature_engine_sha256='a'*64,scope_sha256='s'*64),
        resources=[dict(id=0,capacity=1),dict(id=1,capacity=2)],libraries=['provider-pin'],
        target=dict(prices=dict(compute_ws=dict(latency_cycles=40)),scheduling=dict(overlap=1)),
        memory=dict(line_bytes=64))
    new_model=copy.deepcopy(old_model);new_model['domain']['feature_engine_sha256']='b'*64
    config=dict(enabled=True,pairs=[dict(left=0,right=1)])
    new_model['diagnostics']=dict(scoped_resources=config)
    old_mp,new_mp=put('old-model.json',old_model),put('new-model.json',new_model)
    old_observer,new_observer=put('old-spike',b'synthetic-old'),put('new-spike',b'synthetic-new')
    code=put('src/observer.cc',b'synthetic-code')
    old_source=put('old-source.json',dict(sha256='a'*64,provider_build={'sha256':'unchanged'},shared_provider_observer_headers=[]))
    build=put('build.json',dict(source_identity='b'*64,spike_sha256=new_observer['sha256'],source_sha256s={'observer.cc':code['sha256']}))
    new_source=put('new-source.json',dict(sha256='b'*64,provider_build={'sha256':'unchanged'},shared_provider_observer_headers=[],build=build,
        identity_payload=dict(base_source_manifest=old_source,base_selected_source_identity='a'*64,candidate_compiled_leaves=[code])))
    old_spec=dict(model=old_mp['path'],model_sha256=old_mp['sha256'],source_manifest=old_source['path'],observer=old_observer['path'],
        functional_argv=['unchanged-functional'],libraries=['same-provider'],dependencies=['same-lib'],timing_selection={'scope':'same'},completion_symbol='done')
    new_spec=dict(old_spec,model=new_mp['path'],model_sha256=new_mp['sha256'],source_manifest=new_source['path'],observer=new_observer['path'])
    old_pin,new_pin=put('old-runtime.json',old_spec),put('new-runtime.json',new_spec)
    parameters=[dict(path=['target','prices','compute_ws','latency_cycles'])]
    profile=put('profile.json',dict(parameters=parameters))
    old_row=dict(model_sha256=old_mp['sha256'],price_paths=[parameters[0]['path']],reason='structural error')
    prior=put('prior-denial.json',dict(schema='merlin.native_calibration_admission_review.v1',deny=[old_row]))
    new_row=dict(old_row,model_sha256=new_mp['sha256'],evidence=[build])
    denial=put('denial.json',dict(schema='merlin.native_calibration_admission_review.v1',deny=[old_row,new_row]))
    products={};rows=[]
    for name,mp in [('baseline',old_mp),('off',new_mp),('on',new_mp)]:
        report=put(name+'-report.json',dict(model_sha256=mp['sha256'],execution_completed=True,timing_complete=True,unresolved=[],functional_status='PASS',raw_functional_exit_code=0,
            estimated_cycles=12,feature_counts={'compute':1},scope_sha256='s',scope_binding_sha256='binding',elf_sha256='elf',input_sha256='input',counter_windows=[{'estimated_cycles':12}],scoped_resource_activity={'complete':True}))
        products[name]=dict(report=report,stdout=put(name+'.stdout',b'synthetic numerical output'))
        rows.append(dict(name=name,model_sha256=mp['sha256'],report_sha256=report['sha256']))
    neutrality=put('neutrality.json',dict(schema='merlin.scoped_ws_observer_acceptance.v1',old_off_on_equal=True,qualification_issued=False,coefficient_changes=False,new_physical_runs=0,results=rows))
    evidence=dict(schema=S.SCHEMA,status='diagnostics_only_provisional',previous_runtime=old_pin,runtime=new_pin,fit_authorized=False,promotion_authorized=False,
        previous_model=old_mp,model=new_mp,previous_observer=old_observer,observer=new_observer,diagnostics=config,source_manifests=dict(previous=old_source,current=new_source),
        neutrality=neutrality,neutrality_products=products,previous_denial=prior,continued_denial=denial,calibration_profile=profile)
    return SimpleNamespace(put=put,old_model=old_model,new_model=new_model,old_spec=old_spec,new_spec=new_spec,old_pin=old_pin,new_pin=new_pin,evidence=evidence)


def check(c):
    S.verify(SimpleNamespace(model_json=json.dumps(c.old_model)),SimpleNamespace(model_json=json.dumps(c.new_model)),c.evidence,previous=c.old_pin,current=c.new_pin)


def update_model(c):
    pin=c.put('new-model.json',c.new_model);c.evidence['model']=pin
    c.new_spec.update(model=pin['path'],model_sha256=pin['sha256']);update_spec(c)


def update_spec(c):
    c.new_pin=c.put('new-runtime.json',c.new_spec);c.evidence['runtime']=c.new_pin


def test_diagnostics_ancestry_accepts_only_reviewed_identity_and_capture(case,monkeypatch):
    check(case)
    old={'runtime':case.old_pin,'members':{},'timing_epochs':[]}
    new=copy.deepcopy(old);new['runtime']=case.new_pin
    review=case.put('review.json',case.evidence)
    new['runtime_revisions']=[dict(schema='merlin.runtime_implementation_revision.v1',kind='scoped_observer_diagnostics',previous=case.old_pin,runtime=case.new_pin,evidence=review,review_id='synthetic-operator')]
    monkeypatch.setattr(R.PERF,'_runtime',lambda p:SimpleNamespace(model_json=json.dumps(case.old_model if p==case.old_pin else case.new_model)))
    R.validate(old,new)


@pytest.mark.parametrize('field',['coefficient','capacity','model_provider','model_scope','ordering','memory','other_diagnostic'])
def test_scoped_revision_refuses_other_model_changes(case,field):
    m=case.new_model
    if field=='coefficient':m['target']['prices']['compute_ws']['latency_cycles']=1
    elif field=='capacity':m['resources'][0]['capacity']=3
    elif field=='model_provider':m['libraries']=['changed']
    elif field=='model_scope':m['domain']['scope_sha256']='x'*64
    elif field=='ordering':m['target']['scheduling']['overlap']=0
    elif field=='memory':m['memory']['line_bytes']=128
    else:m['diagnostics']['unreviewed']=True
    update_model(case)
    with pytest.raises(ValueError,match='coefficients|semantics'):check(case)


@pytest.mark.parametrize('field',['libraries','functional_argv','dependencies','timing_selection','completion_symbol'])
def test_scoped_revision_refuses_runtime_provider_and_boundary_changes(case,field):
    case.new_spec[field]='changed';update_spec(case)
    with pytest.raises(ValueError,match='provider, functional runtime or scope'):check(case)


@pytest.mark.parametrize('field',['denial','source','binary','neutrality','authority'])
def test_scoped_revision_requires_original_evidence_and_preserved_denial(case,field):
    if field=='denial':
        p=case.evidence['continued_denial'];doc=json.loads(Path(p['path']).read_text());doc['deny']=doc['deny'][1:]
        case.evidence['continued_denial']=case.put('denial.json',doc)
    elif field=='source':
        p=case.evidence['source_manifests']['current'];doc=json.loads(Path(p['path']).read_text());doc['identity_payload']['base_selected_source_identity']='foreign'
        case.evidence['source_manifests']['current']=case.put('new-source.json',doc)
    elif field=='binary':Path(case.evidence['observer']['path']).write_bytes(b'changed')
    elif field=='neutrality':
        p=case.evidence['neutrality_products']['on']['report'];doc=json.loads(Path(p['path']).read_text());doc['estimated_cycles']=11
        case.evidence['neutrality_products']['on']['report']=case.put('on-report.json',doc)
    else:case.evidence['fit_authorized']=True
    with pytest.raises((ValueError,RuntimeError)):check(case)


@pytest.mark.parametrize('source',['panel','round'])
def test_active_old_denial_cannot_authorize_new_diagnostic_model(case,monkeypatch,source):
    from merlin_experiments.phase1 import joint_calibration_admission as A
    review=case.put('review.json',case.evidence)
    selected=dict(runtime=case.new_pin,runtime_revisions=[dict(kind='scoped_observer_diagnostics',runtime=case.new_pin,evidence=review)])
    plan=dict(runtime=case.new_pin,parameters=[{'path':['target','prices','compute_ws','latency_cycles']}])
    if source=='panel':plan['selection_document']=selected
    else:plan['selection']=case.put('selection.json',selected)
    monkeypatch.setenv(A.ENV,json.dumps(case.evidence['previous_denial']))
    with pytest.raises(ValueError,match='active admission review differs'):A.assert_allowed(plan)
    monkeypatch.delenv(A.ENV)
    with pytest.raises(ValueError,match='active admission review differs'):A.assert_allowed(plan)
    monkeypatch.setenv(A.ENV,json.dumps(case.evidence['continued_denial']))
    with pytest.raises(ValueError,match='blocked_structural_model_admission'):A.assert_allowed(plan)
