import copy
import json
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import joint_arrival_revision as A


def fixture(monkeypatch):
    before = dict(libraries=['functional', 'old-provider'], domain={'feature_engine_sha256':'old-engine'},
        cpu={'isa':'fixed'}, resources=[{'capacity':1}], memory={'caches':{'private':{'line_bytes':64}}},
        target=dict(hardware={'capacity':16}, prices={'load':10}, scheduling=dict(version=5,
            require_fenced_phases=1, conservative_local_store_holds=1)))
    facts = {k:1 for k in A._HARDWARE}
    fetch = dict(schema='merlin.provisional_fetch_refill.v1',
        assumptions='demand_only_immediate_replay_no_backpressure', **{k:1 for k in A._FETCH_FIELDS})
    after = copy.deepcopy(before)
    after['libraries'][-1] = 'new-provider'
    after['domain']['feature_engine_sha256'] = 'new-engine'
    after['target']['hardware'].update(facts)
    for path, value in A._SELECTORS.items():
        _, section, name = path.split('.')
        after['target'].setdefault(section, {})[name] = value
    after['memory']['fetch_refill_timeline'] = fetch
    def pin(name):return {'path':name, 'sha256':name}
    previous, current = pin('old-runtime'), pin('new-runtime')
    source_pins = [pin('source')]
    audit = dict(schema='merlin.public_arrival_mechanism_facts.v1', selected_elaboration=pin('elaboration'),
        source_pins=source_pins)
    specs = dict(functional_argv=['fixed'], entry_symbol='entry', completion_symbol='done')
    docs = {
        'old-model':before, 'new-model':after,
        'old-runtime':dict(specs, libraries=['functional','old-provider'], source_manifest='old-manifest'),
        'new-runtime':dict(specs, libraries=['functional','new-provider'], source_manifest='new-manifest',observer='observer'),
        'old-manifest':{'sha256':'old-engine'}, 'new-manifest':{'sha256':'new-engine'},
        'row':dict(audit,mechanism='row_requests',hardware=facts,selectors=A._SELECTORS),
        'fetch':dict(audit,mechanism='fetch_refill',timeline=fetch),
        'provider':dict(library='new-provider',library_sha256='new-provider'),
        'replays':dict(runtime=current,physical_runs_added=0,price_changes=False,rows=[pin('receipt')]),
        'receipt':dict(runtime_sha256='runtime-identity',model=pin('new-model'),report=pin('report'),
            timing_complete=True,unresolved=[],functional_console_equal=True,physical_runs=0,
            calibration_eligible=False,runs=[dict(complete_output_match=True,bytes=32)],
            original={'elf':pin('elf')}),
        'report':dict(timing_complete=True,execution_completed=True,model_sha256='new-model',elf_sha256='elf'),
        'profile':dict(parameters=[{'path':['target','prices','load']}]),
        'denial':dict(schema='merlin.native_calibration_admission_review.v1',
            deny=[dict(model_sha256='new-model',price_paths=[['target','prices','load']])]),
    }
    evidence=dict(schema='merlin.arrival_pipeline_review.v1',status='conditional_provisional_untuned',
        previous_runtime=previous,runtime=current,previous_model=pin('old-model'),model=pin('new-model'),
        source_manifests={'previous':pin('old-manifest'),'current':pin('new-manifest')},
        source_audits={'row_requests':pin('row'),'fetch_refill':pin('fetch')},provider_build=pin('provider'),
        observer=pin('observer'),replay_evidence=pin('replays'),calibration_profile=pin('profile'),
        continued_denial=pin('denial'))
    monkeypatch.setattr(A.PERF,'_read',lambda p:docs[p['path']])
    verified=[]
    monkeypatch.setattr(A,'_verify_file',lambda p:verified.append(p))
    def verify():
        A.verify(SimpleNamespace(model_json=json.dumps(before)),
            SimpleNamespace(model_json=json.dumps(after),sha256='runtime-identity'),
            evidence,previous=previous,current=current)
    return before,after,evidence,docs,verified,verify


def test_exact_pipeline_review_preserves_models_and_reopens_sources(monkeypatch):
    before,after,evidence,docs,verified,verify=fixture(monkeypatch)
    saved=copy.deepcopy((before,after));verify()
    assert (before,after)==saved
    assert {'path':'elaboration','sha256':'elaboration'} in verified
    assert {'path':'elf','sha256':'elf'} in verified


@pytest.mark.parametrize('path,value', [
    (('target','prices','load'),9), (('target','hardware','capacity'),32),
    (('target','scheduling','require_fenced_phases'),0),
    (('memory','caches','private','line_bytes'),128), (('cpu','isa'),'changed'),
])
def test_old_prices_geometry_dependencies_and_cpu_refuse(monkeypatch,path,value):
    before,after,evidence,docs,verified,verify=fixture(monkeypatch)
    node=after
    for key in path[:-1]:node=node[key]
    node[path[-1]]=value
    with pytest.raises(ValueError,match='unreviewed'):verify()


def test_missing_scope_source_and_complete_output_evidence_refuse(monkeypatch):
    before,after,evidence,docs,verified,verify=fixture(monkeypatch)
    docs['new-runtime']['completion_symbol']='other'
    with pytest.raises(ValueError,match='scope'):verify()
    docs['new-runtime']['completion_symbol']='done'
    docs['fetch']['selected_elaboration']={'path':'other','sha256':'other'}
    with pytest.raises(ValueError,match='hardware domains'):verify()
    docs['fetch']['selected_elaboration']=docs['row']['selected_elaboration']
    docs['receipt']['runs'][0]['complete_output_match']=False
    with pytest.raises(ValueError,match='numerical'):verify()


def test_no_qualification_or_denial_bypass(monkeypatch):
    before,after,evidence,docs,verified,verify=fixture(monkeypatch)
    evidence['status']='qualified'
    with pytest.raises(ValueError,match='provisional'):verify()
    evidence['status']='conditional_provisional_untuned'
    docs['denial']['deny']=[]
    with pytest.raises(ValueError,match='denial'):verify()


def test_missing_assumption_or_changed_prior_selector_refuses(monkeypatch):
    before,after,evidence,docs,verified,verify=fixture(monkeypatch)
    docs['row']['selectors']=dict(A._SELECTORS)
    docs['row']['selectors'].pop('target.assumptions.reader_a_ready_with_transaction_credit')
    with pytest.raises(ValueError,match='mechanism fields'):verify()
    docs['row']['selectors']=A._SELECTORS
    before['target']['scheduling']['scratchpad_row_loads']=0
    with pytest.raises(ValueError,match='overwrite'):verify()
