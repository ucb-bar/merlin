"""Recognize a private exact result cast without selecting target arithmetic."""
from dataclasses import replace

from merlin.semantic_compiler.model import KernelRequest, pointwise_identity_maps

from .semantic_regions import extract_semantic_region


def split_private_bf16_cast(source: KernelRequest) -> KernelRequest:
    """Return the unchanged producer region of a private f32-to-BF16 cast.

    This proves only the dense source boundary, rounding and ownership. A
    provider must separately prove that its selected implementation already
    produces the required BF16 encoding, including exceptional-value behavior,
    before omitting the cast or changing the producer's publication type.
    """
    pure = [node for node in source.nodes if node.effect == 'pure']
    if (len(pure) < 2 or source.outputs != (pure[-1].id,)
            or source.physical_extents or source.boundary_layouts or source.output_layouts):
        raise ValueError('result narrowing needs one private dense cast result')
    cast = pure[-1]
    if cast.op != 'f32_to_bf16' or len(cast.inputs) != 1:
        raise ValueError('result narrowing needs a final f32-to-BF16 cast')
    result = source.node(cast.inputs[0])
    attrs = {key: value for key, value in cast.attrs if not key.startswith('prov.')}
    if (result.effect != 'pure' or result.type.dtype != 'f32'
            or result.type.numerical_policy != 'raw-bit-pattern'
            or cast.type != replace(result.type, dtype='bf16')
            or attrs != {'numeric_semantics': 'f32-to-bf16-rne-v1'}
            or cast.index_maps != pointwise_identity_maps(len(cast.type.shape), 1)):
        raise ValueError('result narrowing cast shape, indexing or rounding differs')
    producer = extract_semantic_region(source, tuple(node.id for node in pure[:-1]))
    if producer.outputs != (result.id,):
        raise ValueError('result narrowing producer has another escaping result')
    boundary = lambda request: tuple(node for node in request.nodes if node.effect in ('input', 'constant'))
    if boundary(producer) != boundary(source):
        raise ValueError('result narrowing changed the original operand boundary')
    return producer
