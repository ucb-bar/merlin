"""Release adaptation keeps the native schema's unique composition constraints.

Reproduced on a selected C8 window_movement pool case: chain_for_node already
states the pool obligations before _release_chain adds instance obligations.
"""
import copy

import pytest
from merlin_experiments.phase0.composition_generation import _release_chain
from merlin_experiments.phase0.fusion_cases import chain_for_node
from merlin.targetgen.software_spec import validate_compositions


@pytest.mark.parametrize('node_has_pool', [True, False])
def test_release_pool_constraints_are_unique_and_complete(node_has_pool):
    node = dict(form='window_movement', seed=False, residual=None,
                realization='pool', scheme='q_f32_tensor',
                readout=dict(exit='requant', stages=[dict(kind='pool')] if node_has_pool else []))
    original = copy.deepcopy(node)
    chain = _release_chain(node, 'not_applicable', dict(producer='movement', readout_stages=['pool']))
    for constraint in ({'max_requires_store_path_pooling': 'from_facts'},
                       'scratchpad_source_pooling_fenced',
                       {'padded_max_requires': 'producer_lower_bound_or_derived_pad_value'}):
        assert chain['constraints'].count(constraint) == 1
    validate_compositions(dict(schema='merlin.software_spec.compositions.v1', status='reviewed',
                               grading='placement_and_instruction_audit', chains=[chain]))
    assert node == original


def test_release_nonpool_constraints_are_unchanged():
    node = dict(form='contraction', seed=False, residual=None,
                realization='matmul', scheme='q_f32_tensor', readout=dict(exit='accumulator', stages=[]))
    assert _release_chain(node, 'WS', dict(producer='contraction', readout_stages=[])) == chain_for_node(node, 'WS')
