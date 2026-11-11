"""Common identical pure tensor values with reversible source provenance.

The semantic graph's pure effect promises deterministic, effect-free values.
Only provenance attributes are ignored; numerical policies, operand order,
indexing and every other attribute remain part of the equality key. This is
logical value reuse, not permission to alias externally owned storage.
"""

import json
from dataclasses import replace

from .model import SemanticNode

SCHEMA = "merlin.common_pure_values.v1"


def semantic_attributes(node):
    """Provenance traces do not change logical values or dense element order."""
    return tuple((key, value) for key, value in node.attrs if not key.startswith("prov."))


def common_pure_values(source, *, groups=()):
    """Share pure values, preserving supplied atomic provider regions.

    Multi-result sibling regions are compared as complete ordered tuples.
    Regions containing internal dependencies remain intact and unshared.
    Untagged nodes are single-result regions; callers lowering bundled source
    operations must supply their provider's checked region groups.
    """
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("value commoning requires an unallocated logical source")
    nodes, available, aliases, changes = [], {}, {}, []
    public = set(source.outputs)
    positions = {node.id: index for index, node in enumerate(source.nodes)}
    membership, normalized = {}, []
    for group in groups:
        if (
            not isinstance(group, (tuple, list))
            or not group
            or any(not isinstance(key, str) or key not in positions or key in membership for key in group)
            or len(set(group)) != len(group)
            or any(source.node(key).effect != "pure" for key in group)
        ):
            raise ValueError("atomic commoning groups need distinct pure source nodes")
        ordered = tuple(sorted(group, key=positions.__getitem__))
        normalized.append(ordered)
        membership.update((key, ordered) for key in ordered)
    normalized.sort(key=lambda group: positions[group[0]])
    removals = {}
    for index, original in enumerate(source.nodes):
        node = replace(original, inputs=tuple(aliases.get(key, key) for key in original.inputs))
        if node.effect == "pure":
            group = membership.get(node.id, (node.id,))
            if node.id == group[0]:
                members = [source.node(key) for key in group]
                # Do not hoist a sibling whose inputs are defined after the
                # group's first member, or partially common a fused region.
                ready = all(positions[key] < index for item in members for key in item.inputs)
                if ready:
                    records = []
                    for item in members:
                        record = item.record()
                        del record["id"]
                        record["inputs"] = [aliases.get(key, key) for key in item.inputs]
                        record["attrs"] = dict(semantic_attributes(item))
                        records.append(record)
                    key = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    equivalent_group = available.get(key) if not public.intersection(group) else None
                    if equivalent_group is not None and any(positions[key] >= index for key in equivalent_group):
                        equivalent_group = None
                    if equivalent_group is None:
                        available.setdefault(key, group)
                    else:
                        removals.update(zip(group, equivalent_group, strict=True))
                        aliases.update(zip(group, equivalent_group, strict=True))
        equivalent = removals.get(node.id)
        if equivalent is None:
            nodes.append(node)
        if equivalent is not None or node != original:
            changes.append({"index": index, "original": original.record(), "replacement": equivalent})
    result = replace(source, nodes=tuple(nodes)) if changes else source
    return result, {
        "schema": SCHEMA,
        "before_digest": source.digest(),
        "after_digest": result.digest(),
        "changes": changes,
        "groups": [list(group) for group in normalized],
    }


def verify_common_pure_values(source, record):
    """Reconstruct every changed/removed node and independently replay admission."""
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "before_digest", "after_digest", "changes", "groups"}
        or record["schema"] != SCHEMA
        or record["after_digest"] != source.digest()
        or not isinstance(record["changes"], list)
        or not isinstance(record["groups"], list)
    ):
        raise ValueError("value commoning receipt has invalid fields or result identity")
    originals = {}
    removed = 0
    for row in record["changes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "original", "replacement"}
            or type(row["index"]) is not int
            or row["index"] < 0
            or row["index"] in originals
            or (row["replacement"] is not None and not isinstance(row["replacement"], str))
        ):
            raise ValueError("value commoning receipt has an invalid change")
        originals[row["index"]] = (SemanticNode.from_record(row["original"]), row["replacement"])
        removed += row["replacement"] is not None
    count = len(source.nodes) + removed
    if any(index >= count for index in originals):
        raise ValueError("value commoning receipt position exceeds the source")
    current, restored = iter(source.nodes), []
    for index in range(count):
        original, replacement = originals.get(index, (None, None))
        if replacement is None:
            node = next(current)
            if original is not None and node.id != original.id:
                raise ValueError("value commoning changed a surviving node identity")
        restored.append(original if original is not None else node)
    original = replace(source, nodes=tuple(restored))
    if original.digest() != record["before_digest"]:
        raise ValueError("value commoning original source changed")
    replayed, expected = common_pure_values(original, groups=record["groups"])
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("value commoning differs from exact replay")
    return original
