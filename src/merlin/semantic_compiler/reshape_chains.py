"""Compose dense logical reshapes while preserving every element's exact bits.

Only adjacent, unannotated reshape operations are transparent. This removes
redundant logical movement; it does not create physical aliases or authorize
in-place writes to a caller's input.
"""

from dataclasses import replace
from math import prod

from .model import SemanticNode

SCHEMA = "merlin.reshape_chains.v1"


def _plain_reshape(node, parent):
    return (
        node.op == "reshape"
        and node.effect == "pure"
        and len(node.inputs) == 1
        and not node.attrs
        and not node.index_maps
        and (node.type.dtype, node.type.numerical_policy) == (parent.type.dtype, parent.type.numerical_policy)
        and prod(node.type.shape) == prod(parent.type.shape)
    )


def compose_reshape_chains(source):
    """Bypass complete dense reshapes and erase only newly unused intermediates.

    Public results, other consumers and all source input/constant ports retain
    their identities. Transposes, gathers, casts, attributes and physical views
    are not admitted by this transform.
    """
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("reshape composition requires an unallocated logical source")
    seen, bypassed = {}, set()
    for original in source.nodes:
        node = original
        if len(node.inputs) == 1:
            parent = seen[node.inputs[0]]
            if _plain_reshape(node, parent) and len(parent.inputs) == 1:
                base = seen[parent.inputs[0]]
                if _plain_reshape(parent, base):
                    node = replace(node, inputs=(base.id,))
                    bypassed.add(parent.id)
        seen[node.id] = node
    used = set(source.outputs) | {key for node in seen.values() for key in node.inputs}
    removed = bypassed - used
    nodes, changes = [], []
    for index, original in enumerate(source.nodes):
        node = seen[original.id]
        if node.id not in removed:
            nodes.append(node)
        if node.id in removed or node != original:
            changes.append({"index": index, "original": original.record(), "removed": node.id in removed})
    result = replace(source, nodes=tuple(nodes)) if changes else source
    return result, {
        "schema": SCHEMA,
        "before_digest": source.digest(),
        "after_digest": result.digest(),
        "changes": changes,
    }


def verify_reshape_chains(source, record):
    """Restore the source, then replay the exact shape and liveness checks."""
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "before_digest", "after_digest", "changes"}
        or record["schema"] != SCHEMA
        or record["after_digest"] != source.digest()
        or not isinstance(record["changes"], list)
    ):
        raise ValueError("reshape receipt has invalid fields or result identity")
    originals, removed = {}, 0
    for row in record["changes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "original", "removed"}
            or type(row["index"]) is not int
            or row["index"] < 0
            or row["index"] in originals
            or type(row["removed"]) is not bool
        ):
            raise ValueError("reshape receipt has an invalid change")
        originals[row["index"]] = (SemanticNode.from_record(row["original"]), row["removed"])
        removed += row["removed"]
    count = len(source.nodes) + removed
    if any(index >= count for index in originals):
        raise ValueError("reshape receipt position exceeds the source")
    current, restored = iter(source.nodes), []
    for index in range(count):
        original, erased = originals.get(index, (None, False))
        if not erased:
            node = next(current)
            if original is not None and node.id != original.id:
                raise ValueError("reshape composition changed a surviving node identity")
        restored.append(original if original is not None else node)
    original = replace(source, nodes=tuple(restored))
    if original.digest() != record["before_digest"]:
        raise ValueError("reshape composition original source changed")
    replayed, expected = compose_reshape_chains(original)
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("reshape composition differs from exact replay")
    return original
