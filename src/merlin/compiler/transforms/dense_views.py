"""Prove internal dense aliases and retain their storage through all readers.

Only unchanged dense element order can share storage. Public buffers keep their
caller-owned allocations, except that a view may read an immutable parameter in
place; the C binding requires that root to be a read-only input. Ordinary
regions keep disjoint input ports. Target kernels and nontrivial layout
transformations remain provider-owned.
"""

from dataclasses import replace
from math import prod

from merlin.compiler.partition.global_plan import CycleInterval

IMPLEMENTATION = "merlin.dense_buffer_view"


def _same_dense_bytes(parent, node):
    """Whether ``node`` holds ``parent``'s bytes in the same dense order: a reshape, a copy, an order-preserving
    affine layout of a computed parent, or an identity gather of an input."""
    if node.op in {"reshape", "copy"}:
        return not node.attrs and not node.index_maps
    from merlin.compiler.lower.c.layout import bounded_dense_read_map, dense_read_preserves_order

    if parent.effect == "pure":
        # A computed parent: any affine layout whose reads keep the dense byte order.
        return not node.attrs and dense_read_preserves_order(parent, node)
    read = bounded_dense_read_map(parent, node) if node.op == "gather" and parent.effect == "input" else None
    if read is None:
        return False
    # Every read is in bounds, so each result element reads source element sum(coordinate * stride).
    strides = [prod(parent.type.shape[axis + 1:]) for axis in range(len(parent.type.shape))]
    flat = [sum(row[loop] * stride for row, stride in zip(read.coefficients, strides)) for loop in range(read.loop_rank)]
    shape = node.type.shape
    return not sum(offset * stride for offset, stride in zip(read.offsets, strides)) and all(
        extent == 1 or step == prod(shape[axis + 1:]) for axis, (extent, step) in enumerate(zip(shape, flat))
    )


def _view_pair(source, alternative, immutable):
    if (
        source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
        or len(alternative.nodes) != 1
        or len(alternative.inputs) != 1
        or len(alternative.outputs) != 1
    ):
        return None
    computations = [node for node in source.nodes if node.effect == "pure"]
    node = computations[alternative.nodes[0]]
    if len(node.inputs) != 1:
        return None
    parent = source.node(node.inputs[0])
    if (
        not (parent.effect == "pure" or parent.effect == "input" and parent.id in immutable)
        or not _same_dense_bytes(parent, node)
        or parent.id in source.outputs
        or node.id in source.outputs
        or (parent.type.dtype, parent.type.numerical_policy) != (node.type.dtype, node.type.numerical_policy)
        or not 0 < prod(parent.type.shape) == prod(node.type.shape)
    ):
        return None
    left, right = alternative.inputs[0], alternative.outputs[0]
    representation = left.representation
    if (
        left.buffer != parent.id
        or right.buffer != node.id
        or representation != right.representation
        or representation.dtype != node.type.dtype
        or representation.layout != "dense"
        or representation.encoding != "plain"
        or representation.quantization != "none"
        or representation.attributes
    ):
        return None
    return node.id, parent.id, 0


def _distinct_inputs(plan, aliases):
    parents = {name: parent for name, parent, _ in aliases}

    def root(name):
        seen = set()
        while name in parents:
            if name in seen:
                raise ValueError("cyclic dense buffer view")
            seen.add(name)
            name = parents[name]
        return name

    for item in plan.selected:
        inputs = {port.buffer for port in item.inputs}
        # Existing region ABIs may reject overlapping *input* buffers too.
        # Sharing two ports requires a separate provider permission contract.
        if len({root(name) for name in inputs}) != len(inputs):
            return False
    return True


def select_dense_buffer_views(source, plan, parameters=None, reader_alignment=None):
    """Replace eligible materializations; leave every unproved case unchanged.

    ``parameters`` maps each read-only parameter to the byte alignment its bound pointer is guaranteed, and
    ``reader_alignment`` is the largest buffer alignment any region of ``plan`` may require. A view of a
    parameter hands its readers the parameter's own pointer instead of the copy's arena storage, so a
    parameter may be a view's root only when its guarantee covers that requirement; otherwise it keeps its copy.
    """
    immutable = frozenset(key for key, alignment in (parameters or {}).items()
                          if reader_alignment is not None and alignment % reader_alignment == 0)
    selected, aliases = [], []
    for item in plan.selected:
        pair = _view_pair(source, item, immutable)
        # A host copy of a parameter costs no device start and keeps its declared host route.
        if pair is not None and source.node(pair[1]).effect == "input" and item.placement != "accelerator":
            pair = None
        if pair is not None and _distinct_inputs(plan, (*aliases, pair)):
            item = replace(
                item,
                implementation=IMPLEMENTATION,
                placement="host",
                cycles=CycleInterval.unknown("storage view; no device execution"),
                demands=(),
                occupancy=(),
                required_capabilities=(),
                resident_inputs=(),
                resident_outputs=(),
                scratch_bytes=0,
                metadata=(("kind", "dense_buffer_view"),),
            )
            aliases.append(pair)
        selected.append(item)
    result = replace(plan, selected=tuple(selected))
    checked_view_aliases(source, result)
    return result


def checked_view_aliases(source, plan):
    """Recheck the semantic proof before granting physical alias permission.

    An input root is accepted here; the C binding refuses one that is not a read-only input.
    """
    inputs = {node.id for node in source.nodes if node.effect == "input"}
    aliases = []
    for item in plan.selected:
        if item.implementation != IMPLEMENTATION:
            continue
        pair = _view_pair(source, item, inputs)
        if pair is None or item.placement != "host" or item.metadata != (("kind", "dense_buffer_view"),):
            raise ValueError("dense buffer view differs from its source proof")
        aliases.append(pair)
    if not _distinct_inputs(plan, aliases):
        raise ValueError("dense buffer views would overlap distinct region input ports")
    return tuple(aliases)
