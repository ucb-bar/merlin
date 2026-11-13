"""Offer private operand permutations to consumers that can absorb indexing."""

from dataclasses import dataclass

from merlin.semantic_compiler.model import KernelRequest
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, RegionAlternative

from .layout_c import permutation_axes
from .semantic_regions import extract_semantic_region


@dataclass(frozen=True)
class OperandPermutation:
    producer: RegionAlternative
    consumer: RegionAlternative
    region: KernelRequest
    inputs: tuple[BufferRepresentation, ...]
    outputs: tuple[BufferRepresentation, ...]


def private_operand_permutations(source, choices):
    """Find complete permutations with no readers outside a consumer region.

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
            if permutation_axes(base, view) is None:
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
            yield OperandPermutation(producer, consumer, region,
                                     tuple(ports[key] for key in inputs),
                                     tuple(outputs[key] for key in region.outputs))
