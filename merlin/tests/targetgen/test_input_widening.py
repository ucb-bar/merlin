"""Private input casts expose a source proof without target arithmetic facts."""
from dataclasses import replace

import pytest

from merlin.compile.input_widening import split_private_bf16_widening
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def source():
    dtype = TensorType((2, 35), 'bf16', 'raw-bit-pattern')
    operand = SemanticNode('x', 'input', (), dtype, effect='input')
    widened = SemanticNode('wide', 'bf16_to_f32', ('x',), replace(dtype, dtype='f32'),
        (('numeric_semantics', 'bf16-to-f32-exact-v1'),), index_maps=pointwise_identity_maps(2, 1))
    result = SemanticNode('y', 'opaque_consumer', ('wide',), widened.type)
    return KernelRequest((operand, widened, result), ('y',), ('external',), (('x', 'external'),), 'test-target')


def test_preserves_opaque_consumer_and_requires_separate_target_proof():
    request = source()
    result = split_private_bf16_widening(request)
    assert result.input_id == 'x' and result.widened_id == 'wide'
    assert result.consumer.nodes[-1] == request.nodes[-1]
    assert result.consumer.node('wide').effect == 'input'
    assert result.consumer.outputs == request.outputs
    assert result.consumer.node('wide').type.dtype == 'f32'


def test_refuses_changed_semantics_escaping_result_and_reused_operand():
    request = source()
    operand, widened, consumer = request.nodes
    changes = [replace(request, nodes=(operand, replace(widened, attrs=()), consumer)),
               replace(request, nodes=(operand, replace(widened, index_maps=()), consumer)),
               replace(request, outputs=('wide', 'y'), output_storages=('external', 'external')),
               replace(request, nodes=(operand, widened, replace(consumer, inputs=('wide', 'x')))),
               replace(request, physical_extents=(('x', 70),))]
    for changed in changes:
        with pytest.raises(ValueError):
            split_private_bf16_widening(changed)


def test_provenance_is_not_a_semantic_decision():
    request = source()
    widened = replace(request.nodes[1], attrs=(*request.nodes[1].attrs, ('prov.module', 'unrelated-name')))
    actual = split_private_bf16_widening(replace(request, nodes=(request.nodes[0], widened, request.nodes[2])))
    assert actual.consumer.nodes == split_private_bf16_widening(request).consumer.nodes
