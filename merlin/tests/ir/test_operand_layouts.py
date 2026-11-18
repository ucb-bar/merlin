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


def full_read_case(kind='broadcast'):
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, IndexMap, pointwise_identity_maps
    shape = (3, 5)
    physical, read = {
        'broadcast': ((5,), IndexMap(2, ((0,1),), (0,))),
        'transpose': ((5,3), IndexMap(2, ((0,1),(1,0)), (0,0))),
        'reverse': (shape, IndexMap(2, ((1,0),(0,-1)), (0,4))),
        'slice': ((3,6), IndexMap(2, ((1,0),(0,1)), (0,0))),
    }[kind]
    base = SemanticNode('base', 'input', (), TensorType(physical, 'i32', 'raw-bit-pattern'), effect='input')
    view = SemanticNode('view', 'gather', ('base',), replace(base.type, shape=shape),
                        index_maps=(read, pointwise_identity_maps(2,1)[-1]))
    consumer = SemanticNode('out', 'consumer', ('view',), view.type,
                            index_maps=pointwise_identity_maps(2,1))
    return KernelRequest((base,view,consumer), ('out',), ('external',), (('base','external'),), 'synthetic')


def test_full_read_composition_matches_independent_enumeration():
    import numpy as np
    from merlin.compile.operand_layouts import fold_private_full_read
    from merlin.semantic_compiler.model import IndexMap

    for kind in ('broadcast','transpose','reverse'):
        original = full_read_case(kind)
        # A nonidentity consumer map exercises actual affine composition.
        consumer = original.nodes[-1]
        consumer = replace(consumer, index_maps=(IndexMap(2, ((-1,0),(0,1)), (2,0)), consumer.index_maps[-1]))
        original = replace(original, nodes=(*original.nodes[:-1], consumer))
        folded = fold_private_full_read(original, 'view', 'out')
        mapping = folded.nodes[-1].index_maps[0]
        values = np.arange(np.prod(original.nodes[0].type.shape)).reshape(original.nodes[0].type.shape)
        view = {'broadcast': lambda: np.broadcast_to(values, (3,5)),
                'transpose': lambda: values.T, 'reverse': lambda: values[:,::-1]}[kind]()
        for r in range(3):
            for c in range(5):
                coordinates = tuple(offset + row[0]*r + row[1]*c
                                    for row,offset in zip(mapping.coefficients,mapping.offsets,strict=True))
                assert values[coordinates] == view[2-r,c]
        assert folded.nodes[-1].inputs == ('base',)
        assert len(folded.nodes) == 2


def test_full_read_rejects_slices_shared_results_and_non_affine_consumers():
    import pytest
    from merlin.compile.operand_layouts import fold_private_full_read

    source = full_read_case()
    for request in (full_read_case('slice'),
                    replace(source, outputs=('view','out'), output_storages=('external',)*2),
                    replace(source, nodes=(*source.nodes, replace(source.nodes[-1], id='another'))),
                    replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], index_maps=())))):
        with pytest.raises(ValueError):
            fold_private_full_read(request, 'view', 'out')
