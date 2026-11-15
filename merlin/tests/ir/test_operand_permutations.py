"""Shared permutation ownership does not grant target indexing support."""
from dataclasses import replace

import pytest

from merlin.compile.operand_permutations import private_operand_permutations
from merlin.tests.ir.test_host_permutation_fold import HEAD, host_plan
from merlin.tests.runtime.test_bf16_linear_c import permuted_head_source


@pytest.mark.parametrize('m,k,n', [(1, 1, 1), (3, 5, 7), (32, 65, 33)])
def test_candidates_preserve_source_ports_and_proofs(m, k, n):
    source = permuted_head_source(m, k, n)
    plan = host_plan(source, HEAD)
    candidate, = private_operand_permutations(source, plan.selected)
    assert candidate.producer == plan.selected[0] and candidate.consumer == plan.selected[1]
    assert [p.buffer for p in candidate.inputs] == ['x', 'w', 'seed']
    assert candidate.region.outputs == source.outputs
    assert candidate.region.node('wt') == source.node('wt')
    assert source.record() == permuted_head_source(m, k, n).record()


def test_shared_public_broadcast_and_changed_representations_stay_materialized():
    for kwargs, groups in (({'extra_reader': True}, (*HEAD, ('other',))),
                           ({'published': True}, HEAD), ({'broadcast': True}, HEAD)):
        source = permuted_head_source(2, 5, 3, **kwargs)
        plan = host_plan(source, groups)
        assert not tuple(private_operand_permutations(source, plan.selected))
    source = permuted_head_source(2, 5, 3)
    plan = host_plan(source, HEAD)
    producer, consumer = plan.selected
    for kind in ('input', 'output', 'placement'):
        if kind == 'placement':
            changed = replace(producer, placement='other')
        else:
            ports = producer.inputs if kind == 'input' else producer.outputs
            field = 'inputs' if kind == 'input' else 'outputs'
            changed = replace(producer, **{field: tuple(replace(p,
                representation=replace(p.representation, encoding='other')) for p in ports)})
        assert not tuple(private_operand_permutations(source, (changed, consumer)))
