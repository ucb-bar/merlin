"""Recognize a private exact input cast without selecting target arithmetic."""
from dataclasses import dataclass, replace

from merlin.semantic_compiler.model import KernelRequest, pointwise_identity_maps

from .semantic_regions import extract_semantic_region


@dataclass(frozen=True)
class PrivateInputWidening:
    consumer: KernelRequest
    input_id: str
    widened_id: str


def split_private_bf16_widening(source: KernelRequest) -> PrivateInputWidening:
    """Separate one exact BF16-to-f32 input cast from its unchanged consumers.

    This establishes source indexing, representation and private ownership.
    A provider must prove that consuming the original BF16 codes preserves its
    selected arithmetic and rejection behavior before omitting the cast.
    """
    pure = [node for node in source.nodes if node.effect == 'pure']
    if (len(pure) < 2 or source.physical_extents or source.boundary_layouts
            or source.output_layouts):
        raise ValueError('input widening needs an unallocated dense consumer region')
    cast = pure[0]
    if cast.op != 'bf16_to_f32' or len(cast.inputs) != 1 or cast.id in source.outputs:
        raise ValueError('input widening needs a private initial BF16-to-f32 cast')
    operand = source.node(cast.inputs[0])
    attrs = {key: value for key, value in cast.attrs if not key.startswith('prov.')}
    if (operand.effect != 'input' or operand.type.dtype != 'bf16'
            or operand.type.numerical_policy != 'raw-bit-pattern'
            or cast.type != replace(operand.type, dtype='f32')
            or attrs != {'numeric_semantics': 'bf16-to-f32-exact-v1'}
            or cast.index_maps != pointwise_identity_maps(len(cast.type.shape), 1)):
        raise ValueError('input widening cast boundary, shape, indexing or semantics differs')
    consumer = extract_semantic_region(source, tuple(node.id for node in pure[1:]))
    expected = {node.id: node for node in source.nodes if node.effect in ('input', 'constant')}
    del expected[operand.id]
    widened = consumer.node(cast.id)
    expected[cast.id] = widened
    actual = {node.id: node for node in consumer.nodes if node.effect in ('input', 'constant')}
    if (consumer.outputs != source.outputs or actual != expected or widened.effect != 'input'
            or widened.type != cast.type):
        raise ValueError('input widening must exclusively replace one consumer boundary')
    return PrivateInputWidening(consumer, operand.id, cast.id)
