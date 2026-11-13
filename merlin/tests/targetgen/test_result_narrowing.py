"""Private narrowing recognizes ownership without granting numeric equivalence."""
from dataclasses import replace

import pytest

from merlin.compile.result_narrowing import split_private_bf16_cast
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def source(shape=(2, 3, 35)):
    f32 = TensorType(shape, 'f32', 'raw-bit-pattern')
    value = SemanticNode('input', 'input', (), f32, effect='input')
    producer = SemanticNode('value', 'opaque_arithmetic', ('input',), f32)
    cast = SemanticNode('out', 'f32_to_bf16', ('value',), replace(f32, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((value, producer, cast), ('out',), ('external',),
                         (('input', 'external'),), 'test-device', 'diagnostic')


def test_preserves_producer_without_asserting_an_arithmetic_policy():
    request = source()
    producer = split_private_bf16_cast(request)
    assert producer.nodes == request.nodes[:-1]
    assert producer.outputs == ('value',)
    # The provider must still decide whether opaque_arithmetic can publish BF16.
    assert producer.nodes[-1].op == 'opaque_arithmetic'


def test_refuses_escape_unrelated_input_and_changed_cast_contract():
    request = source()
    with pytest.raises(ValueError, match='private'):
        split_private_bf16_cast(replace(request, outputs=('value', 'out'),
                                       output_storages=('external', 'external')))
    cast = request.nodes[-1]
    for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                    replace(cast, inputs=('input',)),
                    replace(cast, type=replace(cast.type, shape=(210,)))):
        with pytest.raises(ValueError):
            split_private_bf16_cast(replace(request, nodes=(*request.nodes[:-1], changed)))
    unrelated = replace(request.nodes[0], id='unused')
    with pytest.raises(ValueError, match='operand boundary'):
        split_private_bf16_cast(replace(request, nodes=(unrelated, *request.nodes),
                                       input_storages=(('unused', 'external'), *request.input_storages)))


def test_provenance_does_not_change_rounding_or_types():
    request = source((1,))
    cast = request.nodes[-1]
    cast = replace(cast, attrs=(*cast.attrs, ('prov.source', 'captured-result')))
    producer = split_private_bf16_cast(replace(request, nodes=(*request.nodes[:-1], cast)))
    assert producer.node('value').type == request.node('value').type
