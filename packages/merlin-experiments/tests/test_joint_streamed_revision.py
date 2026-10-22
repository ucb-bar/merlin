import copy

import pytest

from merlin_experiments.phase1.joint_streamed_revision import _HARDWARE, _SELECTORS, _model_delta


def models():
    before = dict(libraries=['functional', 'old'], cpu={'latency':1}, memory={'capacity':32},
        target=dict(hardware={'normalizations':0}, prices={'compute':40},
                    scheduling=dict(version=5, split_ws_feed_lifetimes=1, pool_relu_shared_scale_pipeline=1)))
    facts = {name:{'value':1} for name in _HARDWARE}
    after = copy.deepcopy(before)
    after['libraries'][-1] = 'new'
    after['target']['hardware'].update({k:v['value'] for k,v in facts.items()})
    for path, value in _SELECTORS.items():
        _, section, name = path.split('.')
        after['target'].setdefault(section,{})[name] = value
    return before, after, facts


def test_named_additions_only_and_original_models_not_mutated():
    before, after, facts = models()
    old = copy.deepcopy(before)
    _model_delta(before, after, facts, _SELECTORS)
    assert before == old


@pytest.mark.parametrize('section,key,value', [('prices','compute',20), ('hardware','normalizations',1),
    ('scheduling','serialized_execute_epochs',0)])
def test_unreviewed_price_capability_or_dependency_change_refuses(section,key,value):
    before, after, facts = models()
    after['target'][section][key] = value
    with pytest.raises(ValueError, match='changes prices'):
        _model_delta(before, after, facts, _SELECTORS)


def test_missing_translation_assumption_or_existing_capacity_cannot_be_overwritten():
    before, after, facts = models()
    selectors = dict(_SELECTORS)
    selectors.pop('target.assumptions.accumulator_readout_translation_hits')
    with pytest.raises(ValueError, match='mechanism roster'):
        _model_delta(before, after, facts, selectors)
    before['target']['hardware']['writer_transaction_entries'] = 5
    with pytest.raises(ValueError, match='never replace capacities'):
        _model_delta(before, after, facts, _SELECTORS)
