"""Prove internal dense aliases and retain their storage through all readers.

Only unchanged dense element order can share storage. Public buffers keep their
caller-owned allocations, and ordinary regions keep disjoint input ports.
Target kernels and nontrivial layout transformations remain provider-owned.
"""

from dataclasses import replace
from math import prod

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

IMPLEMENTATION = "merlin.dense_buffer_view"


def _view_pair(source, alternative):
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
    if node.op not in {"reshape", "copy"} or node.attrs or node.index_maps or len(node.inputs) != 1:
        return None
    parent = source.node(node.inputs[0])
    if (
        parent.effect != "pure"
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


def select_dense_buffer_views(source, plan):
    """Replace eligible materializations; leave every unproved case unchanged."""
    selected, aliases = [], []
    for item in plan.selected:
        pair = _view_pair(source, item)
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
    """Recheck the semantic proof before granting physical alias permission."""
    aliases = []
    for item in plan.selected:
        if item.implementation != IMPLEMENTATION:
            continue
        pair = _view_pair(source, item)
        if pair is None or item.placement != "host" or item.metadata != (("kind", "dense_buffer_view"),):
            raise ValueError("dense buffer view differs from its source proof")
        aliases.append(pair)
    if not _distinct_inputs(plan, aliases):
        raise ValueError("dense buffer views would overlap distinct region input ports")
    return tuple(aliases)


def compile_view(region, name):
    """Emit a checked no-op for an already proved and allocated storage view."""
    from .c_sources import c_source_pair
    from .semantic_application import CompiledCRegion

    nodes = [node for node in region.nodes if node.effect == "pure"]
    if len(nodes) != 1 or len(nodes[0].inputs) != 1 or region.outputs != (nodes[0].id,):
        raise ValueError("dense buffer view needs one source and one result")
    output = nodes[0]
    declaration = f"int {name}(void *,const void *const *,void *const *);"
    implementation = f"""int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  return !inputs || !outputs || !inputs[0] || inputs[0]!=outputs[0];
}}
"""
    files = c_source_pair(
        name, public_headers={}, private_headers={}, sources={"view.c": implementation}, declarations=declaration
    )
    return CompiledCRegion(
        name,
        name + ".h",
        output.inputs,
        region.outputs,
        {path: data.encode() for path, data in files.items()},
        (),
        (),
        {
            "kind": "dense_buffer_view",
            "launches_per_invocation": 0,
            "input": output.inputs[0],
            "output": output.id,
            "byte_offset": 0,
        },
    )
