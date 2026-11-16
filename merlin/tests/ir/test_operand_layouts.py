"""Affine layout ownership is independent of consumer and target legality."""
from dataclasses import replace

from merlin.compile.operand_layouts import private_operand_layouts
from merlin.compile.operand_permutations import private_operand_permutations
from merlin.tests.ir.test_host_permutation_fold import HEAD, host_plan
from merlin.tests.runtime.test_bf16_linear_c import permuted_head_source


def test_private_broadcast_is_generic_but_not_a_permutation():
    source = permuted_head_source(2,5,3,broadcast=True)
    choices = host_plan(source,HEAD).selected
    candidate, = private_operand_layouts(source,choices)
    assert candidate.region.outputs==source.outputs
    assert not tuple(private_operand_permutations(source,choices))
    assert not tuple(private_operand_layouts(source,choices,admit=lambda *_: None))
    public = replace(source,outputs=(*source.outputs,'wt'),output_storages=('external',)*2)
    assert not tuple(private_operand_layouts(public,host_plan(public,HEAD).selected))


def test_shared_or_mismatched_boundaries_keep_materialization():
    source = permuted_head_source(2,5,3,broadcast=True,extra_reader=True)
    assert not tuple(private_operand_layouts(source,host_plan(source,(*HEAD,('other',))).selected))
    source = permuted_head_source(2,5,3,broadcast=True)
    producer,consumer = host_plan(source,HEAD).selected
    for changed in (replace(producer,placement='other'),
        replace(producer,outputs=tuple(replace(port,representation=replace(port.representation,encoding='other'))
            for port in producer.outputs))):
        assert not tuple(private_operand_layouts(source,(changed,consumer)))
