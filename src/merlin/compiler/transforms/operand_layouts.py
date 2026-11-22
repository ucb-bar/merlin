"""Offer private dense reads to consumers that can absorb their indexing."""

from dataclasses import dataclass

from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.partition.global_plan import BufferRepresentation, RegionAlternative

from merlin.compiler.lower.c.layout import bounded_dense_read_map
from merlin.compiler.partition.semantic_regions import extract_semantic_region


@dataclass(frozen=True)
class OperandLayout:
    producer: RegionAlternative
    consumer: RegionAlternative
    region: KernelRequest
    inputs: tuple[BufferRepresentation, ...]
    outputs: tuple[BufferRepresentation, ...]


def private_operand_layouts(source, choices, *, admit=bounded_dense_read_map):
    """Find bounded dense reads with no readers outside a consumer region.

    This only proves graph ownership and unchanged boundary representation.
    The caller must separately admit the consumer's indexing, arithmetic and
    resources. Publication must replay those target-specific obligations.
    """
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        return
    pure = [node for node in source.nodes if node.effect == "pure"]
    singles = {pure[row.nodes[0]].id: row for row in choices if len(row.nodes) == 1}
    readers = {}
    for node in source.nodes:
        for key in node.inputs:
            readers.setdefault(key, set()).add(node.id)
    for consumer in choices:
        inside = {pure[index].id for index in consumer.nodes}
        for port in consumer.inputs:
            producer = singles.get(port.buffer)
            if (producer is None or producer.placement != consumer.placement
                    or set(producer.nodes) & set(consumer.nodes)):
                continue
            view = source.node(port.buffer)
            if (len(view.inputs) != 1 or view.id in source.outputs
                    or not readers.get(view.id, set()) <= inside):
                continue
            base = source.node(view.inputs[0])
            if admit(base, view) is None:
                continue
            incoming = [row.representation for row in producer.inputs if row.buffer == base.id]
            outgoing = [row.representation for row in producer.outputs if row.buffer == view.id]
            if incoming != [port.representation] or outgoing != [port.representation]:
                continue
            ports = {row.buffer: row for row in (*producer.inputs, *consumer.inputs)}
            if base.id not in ports or ports[base.id].representation != port.representation:
                continue
            nodes = tuple(sorted((*producer.nodes, *consumer.nodes)))
            region = extract_semantic_region(source, tuple(pure[index].id for index in nodes))
            inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
            outputs = {row.buffer: row for row in consumer.outputs}
            if set(inputs) - ports.keys() or set(region.outputs) != outputs.keys():
                continue
            yield OperandLayout(producer, consumer, region,
                                     tuple(ports[key] for key in inputs),
                                     tuple(outputs[key] for key in region.outputs))


def fold_private_full_read(source, view_id, consumer_id):
    """Compose a private, full-coverage read into a consumer's affine operands.

    Full coverage matters when consumers validate every source element before
    execution: dropping a slice must not newly validate previously unread bytes.
    Admit permutations, reversals and broadcasts of complete source axes. This
    proves indexing and ownership only, not the consumer's numerical policy.
    """
    from dataclasses import replace
    from merlin.compiler.semantic.model import IndexMap

    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('operand folding requires dense logical boundaries')
    view, consumer = source.node(view_id), source.node(consumer_id)
    if (len(view.inputs) != 1 or view.id in source.outputs or consumer.effect != 'pure'
            or {node.id for node in source.nodes if view.id in node.inputs} != {consumer.id}
            or len(consumer.index_maps) != len(consumer.inputs) + 1):
        raise ValueError('operand read must be private to one affine consumer')
    base = source.node(view.inputs[0])
    mapping = bounded_dense_read_map(base, view)
    if (mapping is None or base.effect not in ('input', 'constant') or base.inputs
            or base.index_maps or base.attrs or base.op != base.effect):
        raise ValueError('operand folding requires a bounded external read')
    used_axes = set()
    for extent, row, offset in zip(base.type.shape, mapping.coefficients, mapping.offsets, strict=True):
        terms = [(axis, coefficient) for axis, coefficient in enumerate(row) if coefficient]
        if not terms and extent == 1 and offset == 0:
            continue
        if len(terms) != 1:
            raise ValueError('operand read must cover every complete source axis')
        axis, coefficient = terms[0]
        if (axis in used_axes or coefficient not in (-1, 1)
                or view.type.shape[axis] != extent
                or offset != (extent - 1 if coefficient == -1 else 0)):
            raise ValueError('operand read must cover every complete source axis')
        used_axes.add(axis)
    maps = list(consumer.index_maps)
    inputs = list(consumer.inputs)
    for index, key in enumerate(inputs):
        if key != view.id:
            continue
        inner = maps[index]
        if len(inner.coefficients) != mapping.loop_rank:
            raise ValueError('consumer map rank differs from its operand read')
        maps[index] = IndexMap(inner.loop_rank,
            tuple(tuple(sum(a * b[j] for a, b in zip(row, inner.coefficients, strict=True))
                        for j in range(inner.loop_rank)) for row in mapping.coefficients),
            tuple(offset + sum(a * b for a, b in zip(row, inner.offsets, strict=True))
                  for row, offset in zip(mapping.coefficients, mapping.offsets, strict=True)))
        inputs[index] = base.id
    replacement = replace(consumer, inputs=tuple(inputs), index_maps=tuple(maps))
    return replace(source, nodes=tuple(replacement if node.id == consumer.id else node
                                      for node in source.nodes if node.id != view.id))
