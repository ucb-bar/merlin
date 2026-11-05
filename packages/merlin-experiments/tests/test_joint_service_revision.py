import copy
import json
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import joint_service_revision as S


@pytest.fixture
def case(monkeypatch):
    before = dict(libraries=['functional', 'provider1'], target=dict(
        hardware={'normalizations': 0}, scheduling={'version': 5},
        prices={'pool_store': {'latency_cycles': 8}}), memory={'fixed': True})
    after = copy.deepcopy(before)
    after['libraries'][-1] = 'provider2'
    after['target']['hardware']['relu_same_accumulator_scale_pipeline'] = 1
    after['target']['scheduling']['pool_relu_shared_scale_pipeline'] = 1
    old = dict(libraries=['functional.so', 'old.so'], model='old-model',
               model_sha256='old-sha', observer='same-spike', timing_selection='same')
    new = dict(old, libraries=['functional.so', 'new.so'], model='new-model',
               model_sha256='new-sha')
    price_path = ['target', 'prices', 'compute_ws', 'latency_cycles']
    audit = dict(schema='merlin.public_pool_relu_service_review.v1', runtime='new',
                 old_runtime='old', private_labels_used=False, physical_executions=0,
                 source='source', rtl='rtl', excerpts=['excerpt'], model='model',
                 continued_calibration_denial='denial')
    docs = dict(old=old, new=new, audit=audit, model=after,
                profile={'parameters': [{'path': price_path}]},
                denial=dict(schema='merlin.native_calibration_admission_review.v1',
                            deny=[dict(model_sha256='new-sha', price_paths=[price_path])]))
    evidence = dict(schema='merlin.pool_relu_service_revision.v1',
                    status='provisional_untuned', previous_runtime='old', runtime='new',
                    source_audit='audit', calibration_profile='profile')
    checked = []
    monkeypatch.setattr(S.PERF, '_read', docs.__getitem__)
    monkeypatch.setattr(S, '_verify_file', checked.append)

    def verify():
        S.verify(SimpleNamespace(model_json=json.dumps(before)),
                 SimpleNamespace(model_json=json.dumps(after)), evidence,
                 previous='old', current='new')
    return before, after, docs, evidence, checked, verify


def test_same_service_accepts_only_reviewed_fact_and_pins_sources(case):
    *_, checked, verify = case
    verify()
    assert checked == ['source', 'rtl', 'excerpt']


@pytest.mark.parametrize('mutation', ['price', 'memory', 'observer', 'functional'])
def test_service_review_rejects_unrelated_semantic_changes(case, mutation):
    _, after, docs, _, _, verify = case
    if mutation == 'price':
        after['target']['prices']['pool_store']['latency_cycles'] = 1
    elif mutation == 'memory':
        after['memory']['fixed'] = False
    elif mutation == 'observer':
        docs['new']['observer'] = 'different-spike'
    else:
        docs['new']['libraries'][0] = 'different-functional.so'
    with pytest.raises(ValueError):
        verify()


def test_service_review_cannot_remove_structural_denial(case):
    _, _, docs, _, _, verify = case
    docs['denial']['deny'].clear()
    with pytest.raises(ValueError, match='calibration denial'):
        verify()


def test_service_review_rejects_normalization_and_relabeling(case):
    before, _, _, evidence, _, verify = case
    before['target']['hardware']['normalizations'] = 1
    with pytest.raises(ValueError, match='original unpriced'):
        verify()
    before['target']['hardware']['normalizations'] = 0
    evidence['status'] = 'calibrated'
    with pytest.raises(ValueError, match='provisional'):
        verify()
