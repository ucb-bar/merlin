"""Preserve complete affine broadcasts through exact casts and pointwise uses.

These are logical indexing rewrites, not physical aliases. Complete source
coverage keeps the set of validated input elements unchanged. No arithmetic
rounding boundary, output identity or declared input is removed.
"""

from dataclasses import replace

from .model import SemanticNode, TensorType, pointwise_identity_maps

SCHEMA = "merlin.affine_view_fusion.v1"


def _projection(source, view):
    if (
        view.op != "gather"
        or view.effect != "pure"
        or len(view.inputs) != 1
        or view.attrs
        or not view.type.shape
        or len(view.index_maps) != 2
        or (source.type.dtype, source.type.numerical_policy) != (view.type.dtype, view.type.numerical_policy)
    ):
        return False
    read, write = view.index_maps
    if (
        write != pointwise_identity_maps(len(view.type.shape), 1)[-1]
        or read.loop_rank != len(view.type.shape)
        or len(read.coefficients) != len(source.type.shape)
        or any(read.offsets)
    ):
        return False
    axes = []
    for row, extent in zip(read.coefficients, source.type.shape, strict=True):
        if set(row) - {0, 1} or sum(row) != 1:
            return False
        axis = row.index(1)
        if view.type.shape[axis] != extent or axis in axes:
            return False
        axes.append(axis)
    return True


def fuse_affine_views(source):
    """Commute exact BF16 widening and absorb complete pointwise broadcasts.

    Only dimension projections with full source coverage are admitted. This
    deliberately excludes slices, reversals and general affine expressions.
    Other users and explicit outputs retain their original logical values.
    Dead computations may subsequently be removed by the ordinary exact pass.
    """
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("affine view fusion requires an unallocated logical source")
    nodes, seen, rewrites = [], {}, []
    occupied = {node.id for node in source.nodes}
    for original in source.nodes:
        node, generated = original, []
        if node.op == "bf16_to_f32" and node.effect == "pure" and len(node.inputs) == 1 and node.type.shape:
            view = seen[node.inputs[0]]
            base = seen[view.inputs[0]] if view.op == "gather" and len(view.inputs) == 1 else None
            if (
                base is not None
                and base.type.shape
                and _projection(base, view)
                and view.type == TensorType(node.type.shape, "bf16", "raw-bit-pattern")
                and node.type == TensorType(node.type.shape, "f32", "raw-bit-pattern")
                and node.attrs == (("numeric_semantics", "bf16-to-f32-exact-v1"),)
                and node.index_maps == pointwise_identity_maps(len(node.type.shape), 1)
            ):
                key, suffix = node.id + "_affine_source", 0
                while key in occupied:
                    suffix += 1
                    key = node.id + f"_affine_source_{suffix}"
                occupied.add(key)
                cast = replace(
                    node,
                    id=key,
                    inputs=(base.id,),
                    type=replace(node.type, shape=base.type.shape),
                    index_maps=pointwise_identity_maps(len(base.type.shape), 1),
                )
                nodes.append(cast)
                seen[key] = cast
                generated.append(key)
                node = replace(node, op="gather", inputs=(key,), attrs=(), index_maps=view.index_maps)
        if (
            node.effect == "pure"
            and node.op in {"f32_add", "f32_sub", "f32_mul", "f32_div"}
            and node.type.shape
            and node.type.numerical_policy == "raw-bit-pattern"
            and node.type.dtype == "f32"
            and len(node.inputs) == 2
            and node.attrs == (("numeric_semantics", f"ieee754-f32-{node.op[4:]}-rne-v1"),)
            and node.index_maps == pointwise_identity_maps(len(node.type.shape), 2)
        ):
            inputs, maps = list(node.inputs), list(node.index_maps)
            for i, key in enumerate(node.inputs):
                view = seen[key]
                base = seen[view.inputs[0]] if view.op == "gather" and len(view.inputs) == 1 else None
                if base is not None and view.type == node.type and _projection(base, view):
                    inputs[i], maps[i] = base.id, view.index_maps[0]
            node = replace(node, inputs=tuple(inputs), index_maps=tuple(maps))
        if node != original:
            rewrites.append({"original": original.record(), "generated": generated})
        nodes.append(node)
        seen[node.id] = node
    result = replace(source, nodes=tuple(nodes)) if rewrites else source
    return result, {
        "schema": SCHEMA,
        "before_digest": source.digest(),
        "after_digest": result.digest(),
        "rewrites": rewrites,
    }


def verify_affine_views(source, record):
    """Reconstruct original SSA and replay every indexing/cast precondition."""
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "before_digest", "after_digest", "rewrites"}
        or record["schema"] != SCHEMA
        or record["after_digest"] != source.digest()
        or not isinstance(record["rewrites"], list)
    ):
        raise ValueError("affine view receipt has invalid fields or result identity")
    originals, inserted = {}, set()
    for row in record["rewrites"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"original", "generated"}
            or not isinstance(row["generated"], list)
            or any(not isinstance(key, str) or key in inserted for key in row["generated"])
        ):
            raise ValueError("affine view receipt has an invalid rewrite")
        node = SemanticNode.from_record(row["original"])
        if node.id in originals or node.effect != "pure":
            raise ValueError("affine view receipt repeats a node or changes an effect")
        originals[node.id] = node
        inserted.update(row["generated"])
    if not (set(originals) | inserted) <= {node.id for node in source.nodes} or set(originals) & inserted:
        raise ValueError("affine view receipt refers to missing or overlapping nodes")
    original = replace(
        source, nodes=tuple(originals.get(node.id, node) for node in source.nodes if node.id not in inserted)
    )
    if original.digest() != record["before_digest"]:
        raise ValueError("affine view original source changed")
    replayed, expected = fuse_affine_views(original)
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("affine view derivation differs from replay")
    return original
