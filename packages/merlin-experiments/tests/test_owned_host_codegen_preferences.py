"""Explicit owner choices rejoin actual codegen custody, not requested speedups."""

import json
from pathlib import Path

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower import impr_features
from merlin.runtime import host_rounding
from merlin.targetgen.source_composition_preferences import (
    host_codegen_evidence, host_preparation, require_host_preparation,
)
from merlin_experiments.phase1.feedback.composition_brief import build


def pin(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


@pytest.fixture
def products(tmp_path):
    # Synthetic command records exercise custody refusals; an independent tiny
    # real build in test_host_rounding_model proves assembly and pipeline emission.
    source = tmp_path / 'host_roundeven.S'
    source.write_text(host_rounding.roundeven_f32_assembly(isa='rv64gc', abi='lp64d', symbol='roundevenf'))
    obj = tmp_path / 'host_roundeven.o'
    obj.write_bytes(b'synthetic object identity')
    pipeline = 'func.func(linalg-generalize-named-ops),func.func(linalg-fuse-elementwise-ops)'
    report = tmp_path / 'host-elementwise-fusion.json'
    report.write_text(json.dumps({'schema': 'merlin.host_elementwise_fusion.v1',
        'feature': impr_features.FUSE_AFTER_GENERALIZE_NAME, 'effective_pipeline': pipeline,
        'rewritten_work': 'unmeasured'}))
    recipe = tmp_path / 'recipe.json'
    document = {'status': 'completed', 'preparation': {
        'host_roundeven_implementation': pin(Path(host_rounding.__file__)),
        'host_elementwise_fusion_implementation': pin(Path(impr_features.__file__)),
        'host_elementwise_fusion': pin(report)}, 'commands': [
            {'status': 'returned', 'argv': [], 'inputs': [pin(source)], 'output': pin(obj)},
            {'status': 'returned', 'argv': ['-DMERLIN_EXTERNAL_ROUNDEVEN_F32'], 'inputs': []},
            {'status': 'returned', 'argv': [], 'inputs': [pin(obj)]}]}
    receipt = {'inputs': {'host_roundeven': {'host_isa': 'rv64gc', 'abi': 'lp64d', 'symbol': 'roundevenf'},
                          'host_elementwise_fusion': True},
               'output': {'index_lowering': {'effective_pipeline': pipeline}}}
    def publish():
        recipe.write_text(json.dumps(document))
        receipt['output']['compilation_recipe'] = pin(recipe)
        return receipt
    return receipt, document, publish, source, obj, report


@pytest.mark.parametrize('name', ['host_roundeven', 'host_elementwise_fusion'])
def test_explicit_boolean_default_and_existing_kwargs_route(name):
    assert host_preparation({}, target='device') == ({}, {})
    for value in (False, True):
        options, selection = host_preparation({name: value}, target='device')
        assert options == {name: value}
        assert selection[name]['effect_grant'] is False
        assert selection[name]['historical_observer_permissions_imported'] is False
    for value in (1, None, 'true'):
        with pytest.raises(ValueError, match='explicit Boolean'):
            host_preparation({name: value}, target='device')
    require_host_preparation({'inputs': {}}, {name: False})
    with pytest.raises(ValueError, match='selected request'):
        require_host_preparation({'inputs': {}}, {name: True})


def test_selected_products_and_feedback_separate_pipeline_work_and_perf(products):
    receipt, _, publish, *_ = products
    publish()
    options = {'host_roundeven': True, 'host_elementwise_fusion': True}
    require_host_preparation(receipt, options)
    feedback = build({'source_composition_selection': {'preferences': options}, 'build': receipt})
    assert feedback['features']['host_roundeven']['status'] == 'linked_shared_implementation'
    fusion = feedback['features']['host_elementwise_fusion']
    assert fusion['status'] == 'pipeline_selected'
    assert fusion['rewritten_work'] == 'unmeasured'
    assert fusion['performance_qualified'] is False
    assert not feedback['performance_ranking_eligible']
    assert str(Path(host_rounding.__file__).parent) not in str(feedback)


@pytest.mark.parametrize('change', ['assembly', 'object', 'unlinked', 'macro', 'implementation',
                                   'pipeline', 'pipeline_join', 'uncompleted'])
def test_changed_or_self_consistent_unproved_codegen_is_refused(products, change):
    receipt, document, publish, source, obj, report = products
    if change == 'assembly':
        source.write_text('ret\n')
        document['commands'][0]['inputs'] = [pin(source)]  # Even repinning cannot invent helper equivalence.
    elif change == 'object':
        obj.write_bytes(b'changed after compile')
    elif change == 'unlinked':
        document['commands'][-1]['inputs'] = []
    elif change == 'macro':
        document['commands'][1]['argv'] = []
    elif change == 'implementation':
        document['preparation']['host_roundeven_implementation'] = pin(source)
    elif change == 'pipeline':
        data = json.loads(report.read_bytes())
        data['effective_pipeline'] = 'func.func(linalg-generalize-named-ops)'
        report.write_text(json.dumps(data))
        document['preparation']['host_elementwise_fusion'] = pin(report)
        receipt['output']['index_lowering']['effective_pipeline'] = data['effective_pipeline']
    elif change == 'pipeline_join':
        receipt['output']['index_lowering']['effective_pipeline'] = 'other pipeline'
    else:
        document['status'] = 'invoked'
    publish()
    with pytest.raises(ValueError):
        host_codegen_evidence(receipt, {'host_roundeven': True, 'host_elementwise_fusion': True})
    feedback = build({'source_composition_selection': {'preferences': {
        'host_roundeven': True, 'host_elementwise_fusion': True}}, 'build': receipt})
    feature = 'host_elementwise_fusion' if change.startswith('pipeline') else 'host_roundeven'
    assert feedback['features'][feature]['status'] == 'applied_evidence_unavailable'
