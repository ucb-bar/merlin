import copy
import json
from types import SimpleNamespace
import pytest
from merlin_experiments.phase1 import joint_runtime_revision as R


def test_runtime_revision_retains_prices_and_rejects_unreviewed_change(monkeypatch):
    model = dict(cpu={'isa':'old'}, libraries=['old'], target={'prices':{'compute':40}}, domain={'scope':'same'})
    newer = copy.deepcopy(model); newer['cpu']['isa']='corrected'; newer['libraries']=['new']
    models = {'old':model,'new':newer}
    monkeypatch.setattr(R.PERF,'_runtime',lambda p: SimpleNamespace(model_json=json.dumps(models[p['sha256']])))
    monkeypatch.setattr(R.PERF,'_read',lambda p: {'review':'pinned'})
    old={'runtime':{'sha256':'old'},'members':[1],'timing_epochs':[]}
    new=copy.deepcopy(old);new['runtime']={'sha256':'new'}
    with pytest.raises(ValueError,match='ancestry'):R.validate(old,new)
    new['runtime_revisions']=[dict(schema='merlin.runtime_implementation_revision.v1',previous=old['runtime'],runtime=new['runtime'],evidence={},review_id='operator')]
    R.validate(old,new)
    new['timing_epochs']=[{'mode':'tuned'}]; R.validate(old,new)
    newer['target']['prices']['compute']=12
    with pytest.raises(ValueError,match='prices'):R.validate(old,new)
    newer['target']['prices']['compute']=40
    new['members']=[2]
    with pytest.raises(ValueError,match='non-model'):R.validate(old,new)


def test_scope_extension_cannot_change_old_prices_or_capacity(monkeypatch):
    model=dict(cpu={'isa':'same'},libraries=['old'],domain={'scope':'same'},
               target={'hardware':{'rows':16},'scheduling':{'version':3},'prices':{'existing':{'latency':40}}})
    after=copy.deepcopy(model);after['target']['prices']['new']={'latency':94}
    after['target']['scheduling'].update(version=4,serial_new=1)
    runtime=lambda m:SimpleNamespace(model_json=json.dumps(m))
    evidence=dict(schema='merlin.model_scope_extension_review.v1',source_audit={},
                  added_primitive_prices={'new':{'latency':94}},
                  scheduling_changes=[dict(key='version',before=3,after=4),dict(key='serial_new',before=None,after=1)])
    monkeypatch.setattr(R.PERF,'_read',lambda p:{'pinned':'source'})
    R._verify_scope_extension(runtime(model),runtime(after),evidence)
    after['target']['hardware']['rows']=32
    with pytest.raises(ValueError,match='unreviewed'):R._verify_scope_extension(runtime(model),runtime(after),evidence)
    after['target']['hardware']['rows']=16;after['target']['prices']['existing']['latency']=1
    with pytest.raises(ValueError,match='existing prices'):R._verify_scope_extension(runtime(model),runtime(after),evidence)


def test_implementation_promotion_cannot_bypass_calibration(tmp_path,monkeypatch):
    old=dict(runtime={'sha256':'old'},timing_epochs=[],runtime_revisions=[])
    new=dict(runtime={'sha256':'new'},timing_epochs=[{'epoch':1}],runtime_revisions=[{'review':'operator'}])
    (tmp_path/'joint_campaign.json').write_text(json.dumps({'performance_selection':{'id':'old'}}))
    monkeypatch.setattr(R.PERF,'_read',lambda p:old if p.get('id')=='old' else new)
    with pytest.raises(ValueError,match='tuned epochs'):
        R.request_promotion({'id':'new'},previous_run=tmp_path,review_id='operator',output=tmp_path/'epoch.json')


def test_capability_restriction_preserves_prices_ordering_and_capacity(monkeypatch):
    model = dict(target=dict(hardware=dict(rows=16), scheduling=dict(version=4),
                             prices=dict(compute=dict(latency=40))))
    after = copy.deepcopy(model); after['target']['hardware']['optional_norm'] = 0
    runtime = lambda m: SimpleNamespace(model_json=json.dumps(m))
    evidence = dict(schema='merlin.target_capability_restriction_review.v1',
                    source_audit={}, disabled_optional_features=['optional_norm'])
    monkeypatch.setattr(R.PERF, '_read', lambda p: {'pinned': 'selected RTL'})
    R._verify_capability_restriction(runtime(model), runtime(after), evidence)
    for section, key, value in [('prices', 'compute', {'latency':12}),
                                ('scheduling', 'version', 5), ('hardware', 'rows', 32)]:
        changed = copy.deepcopy(after); changed['target'][section][key] = value
        with pytest.raises(ValueError, match='unreviewed'):
            R._verify_capability_restriction(runtime(model), runtime(changed), evidence)
    after['target']['hardware']['optional_norm'] = 1
    with pytest.raises(ValueError, match='disable'):
        R._verify_capability_restriction(runtime(model), runtime(after), evidence)
