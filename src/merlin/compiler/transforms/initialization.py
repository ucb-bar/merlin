"""Exact tensor-splat destinations expressed as explicit destination fills.

The original scalar SSA is retained. No scalar value is decoded or arithmetic
reordered. This keeps upstream canonicalization from replacing an initialization
with an immutable dense buffer that must then be copied before a reduction.
The independent replay checker reconstructs the entire original module.
"""

from __future__ import annotations

from xdsl.dialects import tensor
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.ops import FillOp
from xdsl.rewriter import Rewriter

from merlin.compiler.lower.model.prov_cse import PROV_PREFIX


def _eligible(operation):
    if (
        type(operation) is not tensor.SplatOp
        or len(operation.operands) != 1
        or len(operation.results) != 1
        or operation.properties
        or operation.regions
        or operation.successors
        or any(not key.startswith(PROV_PREFIX) for key in operation.attributes)
    ):
        return False
    result = operation.results[0]
    typ = result.type
    if (
        type(typ) is not TensorType
        or any(d < 0 for d in typ.get_shape())
        or operation.operands[0].type != typ.element_type
    ):
        return False
    uses = tuple(result.uses)
    return bool(uses) and all(
        isinstance(use.operation, LinalgStructuredOperation)
        and len(use.operation.inputs) <= use.index < len(use.operation.inputs) + len(use.operation.outputs)
        and use.operation.operands[use.index] is result
        for use in uses
    )


def _metadata_only_empty(operation):
    return (
        type(operation) is tensor.EmptyOp
        and bool(operation.attributes)
        and all(key.startswith(PROV_PREFIX) for key in operation.attributes)
        and not operation.properties
        and not operation.operands
        and len(operation.results) == 1
        and type(operation.results[0].type) is TensorType
        and all(d >= 0 for d in operation.results[0].type.get_shape())
    )


def normalize_destinations(original):
    """Clone and normalize only complete static destination-only splat uses."""
    original.verify()
    transformed = original.clone()
    selected, metadata = [], []
    for ordinal, operation in enumerate(tuple(transformed.walk())):
        if _metadata_only_empty(operation):
            attributes = {key: str(value) for key, value in operation.attributes.items()}
            metadata.append((ordinal, operation, attributes))
            operation.attributes.clear()
        if not _eligible(operation):
            continue
        typ = operation.results[0].type
        empty = tensor.EmptyOp((), typ)
        fill = FillOp(inputs=[operation.operands[0]], outputs=[empty.results[0]], res=[typ])
        fill.attributes.update(operation.attributes)
        Rewriter.replace_op(operation, [empty, fill], fill.results)
        selected.append((ordinal, empty, fill))
    transformed.verify()
    ordinals = {operation: n for n, operation in enumerate(transformed.walk())}
    records = tuple(
        {"original_ordinal": ordinal, "empty_ordinal": ordinals[empty], "fill_ordinal": ordinals[fill]}
        for ordinal, empty, fill in selected
    )
    records += tuple(
        {"original_ordinal": ordinal, "empty_metadata_ordinal": ordinals[empty], "attributes": attrs}
        for ordinal, empty, attrs in metadata
    )
    verify_destinations(original, transformed, records)
    return transformed, records


def verify_destinations(original, transformed, records):
    """Independently replay all replacements and compare the whole original SSA."""
    original.verify()
    transformed.verify()
    original_ops = tuple(original.walk())
    splats = tuple(record for record in records if "fill_ordinal" in record)
    metadata = tuple(record for record in records if "empty_metadata_ordinal" in record)
    if len(splats) + len(metadata) != len(records):
        raise ValueError("destination initialization replay has unknown replacement kinds")
    eligible = tuple(n for n, op in enumerate(original_ops) if _eligible(op))
    if tuple(record["original_ordinal"] for record in splats) != eligible:
        raise ValueError("destination initialization did not cover exactly the admitted original splats")
    replay = transformed.clone()
    replay_ops = tuple(replay.walk())
    touched = set()
    if tuple(record["original_ordinal"] for record in metadata) != tuple(
        n for n, op in enumerate(original_ops) if _metadata_only_empty(op)
    ):
        raise ValueError("destination initialization changed its provenance-only empty inventory")
    for record in metadata:
        if set(record) != {"original_ordinal", "empty_metadata_ordinal", "attributes"}:
            raise ValueError("destination initialization metadata has unknown fields")
        position = record["empty_metadata_ordinal"]
        if type(position) is not int or not 0 <= position < len(replay_ops) or position in touched:
            raise ValueError("destination initialization metadata changed its original operation binding")
        touched.add(position)
        old, empty = original_ops[record["original_ordinal"]], replay_ops[position]
        if (
            type(empty) is not tensor.EmptyOp
            or empty.attributes
            or record["attributes"] != {key: str(value) for key, value in old.attributes.items()}
        ):
            raise ValueError("destination initialization changed detached provenance or semantic attributes")
        empty.attributes.update(old.attributes)
    for record in splats:
        if set(record) != {"original_ordinal", "empty_ordinal", "fill_ordinal"}:
            raise ValueError("destination initialization replay has unknown fields")
        positions = (record["empty_ordinal"], record["fill_ordinal"])
        if any(type(n) is not int or n < 0 or n >= len(replay_ops) or n in touched for n in positions):
            raise ValueError("destination initialization replay changed its operation inventory")
        touched.update(positions)
        empty, fill = (replay_ops[n] for n in positions)
        old = original_ops[record["original_ordinal"]]
        if (
            type(empty) is not tensor.EmptyOp
            or type(fill) is not FillOp
            or empty.next_op is not fill
            or empty.parent is not fill.parent
            or empty.attributes
            or empty.properties
            or empty.operands
            or len(empty.results) != 1
            or empty.results[0].type != old.results[0].type
            or len(tuple(empty.results[0].uses)) != 1
            or len(fill.inputs) != 1
            or tuple(fill.outputs) != (empty.results[0],)
        ):
            raise ValueError("destination initialization replay has a changed destination or scalar binding")
        expected = FillOp(inputs=list(fill.inputs), outputs=list(fill.outputs), res=[old.results[0].type])
        expected.attributes.update(old.attributes)
        equivalent = expected.is_structurally_equivalent(fill)
        expected.drop_all_references()
        if not equivalent:
            raise ValueError("destination initialization fill has changed arithmetic, effects or provenance")
        restored = tensor.SplatOp(fill.inputs[0], (), old.results[0].type)
        restored.attributes.update(old.attributes)
        Rewriter.replace_op(fill, [restored], restored.results)
        Rewriter.erase_op(empty)
    replay.verify()
    if not original.is_structurally_equivalent(replay):
        raise ValueError("destination initialization changed original scalar bits, uses or entry semantics")
