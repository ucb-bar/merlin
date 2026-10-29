"""Same-block CSE of static immutable tensor views with explicit source lineage.

This is a structural transform, not a physical alias/residency permission or a
numerical qualification. Ordinary pipeline verification remains caller-owned.
No arithmetic, quantization, operation order or tensor payload is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects import builtin, tensor
from xdsl.rewriter import Rewriter

from .certificates import ir_digest


@dataclass(frozen=True)
class TensorViewWitness:
    operation_index: int
    retained_index: int
    operation: str
    before_sha256: str
    result_type: str


def _key(op):
    allowed = {
        tensor.CollapseShapeOp: {"reassociation"},
        tensor.ExpandShapeOp: {"reassociation", "static_output_shape"},
        tensor.CastOp: set(),
    }
    if (type(op) not in allowed or len(op.operands) != 1 or len(op.results) != 1
            or op.regions or op.successors or set(op.properties) != allowed[type(op)]
            or any(not name.startswith("prov.") for name in op.attributes)):
        return None
    source, result = op.operands[0].type, op.results[0].type
    if any(not isinstance(t, builtin.TensorType) or t.encoding != builtin.NoneAttr()
           or any(d <= 0 for d in t.get_shape()) for t in (source, result)):
        return None
    if source.element_type != result.element_type or prod(source.get_shape()) != prod(result.get_shape()):
        return None
    # Static same-type casts and explicit reassociation views have immutable
    # tensor semantics. An encoded/dynamic/layout-dependent view is excluded.
    if type(op) is tensor.CastOp and source != result:
        return None
    return (type(op), tuple(op.operands), tuple(op.result_types),
            tuple(sorted(op.properties.items())))


def _lineage(op):
    previous = op.attributes.get("prov.view_cse_sources")
    if previous is not None:
        if not isinstance(previous, builtin.ArrayAttr) or any(
                not isinstance(item, builtin.DictionaryAttr) for item in previous):
            raise ValueError("tensor view CSE lineage must contain exact attribute dictionaries")
        return tuple(previous)
    return (builtin.DictionaryAttr(dict(op.attributes)),)


def canonicalize_tensor_views(module: builtin.ModuleOp) -> tuple[TensorViewWitness, ...]:
    """Reuse a dominating identical view, preserving every use and source record.

    Caches are local to each block; nothing moves across CFG/region boundaries.
    Unknown operations are retained, including side effects between two views.
    This is legal only for the closed immutable tensor-view domain above: it
    does not merge loads or mutable memref views. Arithmetic and visible outputs
    remain in their original order. Use the ordinary host verification boundary
    when consuming the transformed program; direct calls are structural planning.
    """
    module.verify()
    indices = {op: i for i, op in enumerate(module.walk())}
    witnesses = []
    blocks = [block for op in module.walk() for region in op.regions for block in region.blocks]
    for block in blocks:
        known = {}
        for op in tuple(block.ops):
            key = _key(op)
            if key is None:
                continue
            previous = known.get(key)
            if previous is None:
                known[key] = op
                continue
            witness = TensorViewWitness(indices[op], indices[previous], op.name,
                                        ir_digest(op), str(op.results[0].type))
            lineage = (*_lineage(previous), *_lineage(op))
            # Metadata is preserved, not used to justify view equivalence.
            previous.attributes["prov.view_cse_sources"] = builtin.ArrayAttr(lineage)
            for name in ("prov.source_node_ids", "prov.origin_node_ids"):
                entries = [record.data.get(name) for record in lineage]
                if all(value is None or isinstance(value, builtin.ArrayAttr) for value in entries):
                    union = tuple(dict.fromkeys(item for value in entries if value is not None for item in value))
                    if union:
                        previous.attributes[name] = builtin.ArrayAttr(union)
            Rewriter.replace_op(op, [], new_results=list(previous.results))
            witnesses.append(witness)
    module.verify()
    return tuple(witnesses)


def fold_complete_matrix_slice_concats(module: builtin.ModuleOp) -> int:
    """View a complete ordered matrix-slice concat through its original aggregate.

    A rank-three tensor ``[h,m,n]`` sliced at every ``h`` coordinate,
    collapsed to ``[m,n]`` and concatenated along dimension zero is exactly
    its row-major ``[h*m,n]`` collapse. Slice observers remain intact; only
    the concat is replaced. Static unencoded tensors and exact unit strides
    exclude partial, reordered, duplicated and layout-dependent alternatives.
    """
    module.verify()
    count = 0
    for concat in tuple(module.walk()):
        if (type(concat) is not tensor.ConcatOp or concat.dim.value.data != 0
                or set(concat.properties) != {"dim"}
                or any(not key.startswith("prov.") for key in concat.attributes)):
            continue
        aggregate = None
        for index, value in enumerate(concat.inputs):
            collapse = value.owner
            if (type(collapse) is not tensor.CollapseShapeOp or len(collapse.operands) != 1
                    or set(collapse.properties) != {"reassociation"}
                    or any(not key.startswith("prov.") for key in collapse.attributes)
                    or tuple(tuple(i.value.data for i in group) for group in collapse.reassociation)
                       != ((0, 1), (2,))):
                break
            view = collapse.operands[0].owner
            if (type(view) is not tensor.ExtractSliceOp or len(view.operands) != 1
                    or set(view.properties) != {
                        "static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
                    or any(not key.startswith("prov.") for key in view.attributes)):
                break
            typ = view.source.type
            if (not isinstance(typ, builtin.TensorType) or typ.encoding != builtin.NoneAttr()
                    or len(typ.get_shape()) != 3 or any(d <= 0 for d in typ.get_shape())):
                break
            h, m, n = typ.get_shape()
            if (h != len(concat.inputs)
                    or (aggregate is not None and view.source is not aggregate)
                    or tuple(view.static_offsets.get_values()) != (index, 0, 0)
                    or tuple(view.static_sizes.get_values()) != (1, m, n)
                    or tuple(view.static_strides.get_values()) != (1, 1, 1)
                    or view.result.type != builtin.TensorType(typ.element_type, (1, m, n))
                    or value.type != builtin.TensorType(typ.element_type, (m, n))
                    or concat.result.type != builtin.TensorType(typ.element_type, (h*m, n))):
                break
            aggregate = view.source
        else:
            if aggregate is None:
                continue
            replacement = tensor.CollapseShapeOp.create(
                operands=[aggregate], result_types=[concat.result.type],
                properties={"reassociation": collapse.reassociation},
                attributes=dict(concat.attributes),
            )
            Rewriter.replace_op(concat, replacement)
            count += 1
    module.verify()
    return count
