import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_structural_revision as S


def fixture(monkeypatch):
    before=dict(cpu={'isa':'fixed'}, libraries=['functional','old-provider'],
        parameters={'memory.snoop_cycles':1}, domain={'feature_engine_sha256':'old-source','scope':'same'},
        resources=[dict(id=0,name='cpu.issue',capacity=1,queue_capacity=0)],
        target=dict(hardware={'rows':16},prices={'compute':40},scheduling=dict(version=4,
            require_fenced_phases=1,conservative_local_store_holds=1,
            serialized_execute_epochs=1,serialized_output_stationary_epochs=1)),
        memory=dict(caches={'l1d':{'line_bytes':64}},paths={'cpu_data':['l1d'],'device':[]}))
    after=copy.deepcopy(before);after['libraries'][-1]='new-provider'
    after['domain']['feature_engine_sha256']='new-source'
    after['target']['scheduling'].update(version=5,split_ws_feed_lifetimes=1)
    fields=dict(beat_bytes=16,lookup_cycles=2,dirty_enter_cycles=1,data_pipeline_cycles=2,metadata_cycles=1)
    resource=dict(id=1,name='cache.probe',capacity=1,queue_capacity=0)
    after['resources'].append(resource)
    after['memory']['transaction_accounting']='coherence_request_separate_v1'
    after['memory']['caches']['l1d']['coherence_probe_service']=dict(fields,
        schema='merlin.private_cache_probe_service.v1',resource='cache.probe')
    previous,current={'id':'old-runtime'},{'id':'new-runtime'}
    modelpin={'id':'new-model','sha256':'b'*64}
    proof=dict(schema='merlin.public_private_probe_protocol_facts.v1',source={'id':'rtl-probe'},
        selected_elaboration={'id':'firrtl'},line_bytes=64,
        field_bindings={k:{'value':v} for k,v in fields.items()})
    docs={'old-model':before,'new-model':after,
        'old-runtime':dict(libraries=['functional','old-provider'],source_manifest='old-manifest',scope='same'),
        'new-runtime':dict(libraries=['functional','new-provider'],source_manifest='new-manifest',scope='same'),
        'old-manifest':{'sha256':'old-source'},'new-manifest':{'sha256':'new-source'},
        'ws-facts':dict(schema='merlin.ws_feed_lifetime_facts.v1',source={'id':'rtl-ws'},
            selected_elaboration={'id':'firrtl'},selected_scheduling=dict(version=5,split_ws_feed_lifetimes=1)),
        'probe-facts':proof,'replay':dict(price_changes=False,physical_runs_added=0,rows=[{'runtime':current}]),
        'profile':dict(parameters=[{'path':['target','prices','compute']}]),
        'denial':dict(schema='merlin.native_calibration_admission_review.v1',deny=[
            dict(model_sha256='b'*64,price_paths=[['target','prices','compute']])])}
    evidence=dict(schema='merlin.structural_accounting_review.v1',status='provisional_untuned',
        previous_runtime=previous,runtime=current,previous_model={'id':'old-model'},model=modelpin,
        source_manifests={k:{'id':v,'path':v} for k,v in [('previous','old-manifest'),('current','new-manifest')]},
        source_audits={'ws_lifetimes':{'id':'ws-facts'},'private_probe':{'id':'probe-facts'}},
        private_cache='l1d',added_resource=resource,replay_evidence={'id':'replay'},
        calibration_profile={'id':'profile'},calibration_admission_review={'id':'denial'})
    monkeypatch.setattr(S.PERF,'_read',lambda pin:docs[pin['id']])
    monkeypatch.setattr(S,'_verify_file',lambda pin:None)
    def verify():S.verify(SimpleNamespace(model_json=json.dumps(before)),
        SimpleNamespace(model_json=json.dumps(after)),evidence,previous=previous,current=current)
    return before,after,evidence,docs,verify


def test_structural_review_allows_only_exact_selected_correction(monkeypatch):
    before,after,evidence,docs,verify=fixture(monkeypatch)
    verify()
    for path,value in [(('target','prices','compute'),1),(('target','hardware','rows'),32),
                        (('target','scheduling','require_fenced_phases'),0),
                        (('parameters','memory.snoop_cycles'),20),(('cpu','isa'),'changed')]:
        dest=after
        for key in path[:-1]:dest=dest[key]
        old=dest[path[-1]];dest[path[-1]]=value
        with pytest.raises(ValueError,match='unreviewed'):verify()
        dest[path[-1]]=old
    after['resources'][0]['capacity']=2
    with pytest.raises(ValueError,match='capacities'):verify()


def test_structural_review_rejects_unreviewed_facts_scope_and_qualification(monkeypatch):
    before,after,evidence,docs,verify=fixture(monkeypatch)
    docs['probe-facts']['field_bindings']['lookup_cycles']['value']=3
    with pytest.raises(ValueError,match='unreviewed'):verify()
    docs['probe-facts']['field_bindings']['lookup_cycles']['value']=2
    docs['new-runtime']['scope']='other'
    with pytest.raises(ValueError,match='execution contract'):verify()
    docs['new-runtime']['scope']='same';evidence['status']='qualified'
    with pytest.raises(ValueError,match='provisional'):verify()


def test_structural_review_requires_exact_model_and_continued_denial(monkeypatch):
    before,after,evidence,docs,verify=fixture(monkeypatch)
    docs['new-model']=dict(after,extra='not-actual')
    with pytest.raises(ValueError,match='exact model'):verify()
    docs['new-model']=after;docs['denial']['deny']=[]
    with pytest.raises(ValueError,match='denial'):verify()
