"""Transport original sibling root identities onto source-bound compute work.

These records carry source identity, never an execution or coverage grant.
The caller still checks actual arithmetic, storage, epochs and physical work.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr

_KEYS = frozenset({"prov.region_id", "prov.source_node_ids"})


def projection_provenance_attributes(attributes):
    """Read the closed optional CB identity pair, without silently losing fields.

    The interface's existing flat string/list syntax cannot quote punctuation
    inside identities. Refuse unrepresentable strings instead of changing them.
    """
    if type(attributes) is not dict:
        raise ValueError("projection provenance requires a plain attribute dictionary")
    if not attributes:
        return {}
    if set(attributes) != _KEYS:
        raise ValueError("projection provenance needs exactly original region and node identities")

    def identity(value):
        if (type(value) is not str or not 0 < len(value) <= 4096
                or any(ord(c) < 32 or c in '\"\\,[]{}' for c in value)):
            raise ValueError("projection source identity cannot round-trip through the selected interface")
        return value

    region = identity(attributes["prov.region_id"])
    nodes = attributes["prov.source_node_ids"]
    if (type(nodes) is not list or not 1 <= len(nodes) <= 256
            or any(type(node) is not str for node in nodes)):
        raise ValueError("projection provenance requires a bounded original node list")
    nodes = [identity(node) for node in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError("projection provenance repeats an original node identity")
    return {"prov.region_id": region, "prov.source_node_ids": nodes}


def shared_projection_provenance(module, group):
    """Re-read identities from the exact current typed roots, in member order.

    Fully unannotated source remains unannotated. Partial/duplicate identity
    records refuse; a local normalized operation ordinal is never synthesized.
    """
    from xdsl.dialects import func

    from merlin.compile.scheduling.shared_input_groups import SharedInputGroup

    if type(group) is not SharedInputGroup:
        raise ValueError("projection provenance requires the checked typed source group")
    group.verify_current(module)
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    operations = tuple(function.body.block.ops)
    roots = tuple(operations[index] for index in group.contraction_ordinals)
    if not any(_KEYS.intersection(root.attributes) for root in roots):
        return ()
    members = []
    for root in roots:
        region = root.attributes.get("prov.region_id")
        nodes = root.attributes.get("prov.source_node_ids")
        if (type(region) is not StringAttr or type(nodes) is not ArrayAttr
                or any(type(node) is not StringAttr for node in nodes)):
            raise ValueError("every projection member needs its original typed region and nodes")
        members.append(projection_provenance_attributes({
            "prov.region_id": region.data, "prov.source_node_ids": [node.data for node in nodes],
        }))
    if len({member["prov.region_id"] for member in members}) != len(members):
        raise ValueError("projection members have ambiguous duplicate original regions")
    return tuple(members)


def projection_compute_member(boundaries, column_bounds, width_bounds):
    """Resolve a real compute interval to ONE original member, conservatively.

    Bounds come from the consumed counted/SCF indices, not receipt assertions.
    Unknown, negative, crossing or ambiguous intervals cannot be attributed.
    """
    if (type(boundaries) is not tuple or len(boundaries) < 3 or boundaries[0] != 0
            or any(type(v) is not int for v in boundaries)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("compute attribution needs the complete original column partition")
    for bounds in (column_bounds, width_bounds):
        if (type(bounds) is not tuple or len(bounds) != 2
                or any(type(v) is not int for v in bounds) or bounds[0] > bounds[1]):
            raise ValueError("compute attribution needs proved counted index bounds")
    if column_bounds[0] < 0 or width_bounds[0] <= 0:
        raise ValueError("compute attribution has a negative or empty interval")
    members = [i for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))
               if start <= column_bounds[0] and column_bounds[1] + width_bounds[1] <= stop]
    if len(members) != 1:
        raise ValueError("actual compute interval crosses original projection members")
    return members[0]


def projection_ir_attributes(attributes):
    """Build the exact LLVM/typed-target identity attributes, without a grant."""
    attributes = projection_provenance_attributes(attributes)
    if not attributes:
        return {}
    return {"prov.region_id": StringAttr(attributes["prov.region_id"]),
            "prov.source_node_ids": ArrayAttr(StringAttr(node) for node in attributes["prov.source_node_ids"])}
