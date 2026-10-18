"""Exact destination passing for a pointwise producer inserted into a splat.

The scalar body is cloned unchanged. Static unit-stride slices, an unused output
block argument, a sole tensor-result use, and pure arithmetic are required.
A fresh fill gives the padded destination writable storage; an extract-slice is
then the producer's destination. Bufferization owns alias legality.

The explicit ``initialize_border_only`` option fills the disjoint rectangular
complement instead. The unread interior is completely overwritten by the same
scalar producer. A live original splat remains immutable and separate. This
option changes no arithmetic, input layout, call ABI, or default selection.
"""

from xdsl.dialects import tensor
from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap


def exterior_slabs(shape, offsets, sizes):
    """Disjoint rectangular complement of one in-bounds rectangular interior.

    Earlier dimensions are restricted to the interior; the current dimension
    selects its lower or upper exterior. Every exterior point therefore belongs
    to the slab for its first out-of-interior dimension, exactly once.
    """
    if not (len(shape) == len(offsets) == len(sizes)):
        raise ValueError("rank mismatch")
    if any(d <= 0 or n <= 0 or o < 0 or o + n > d for d, o, n in zip(shape, offsets, sizes)):
        raise ValueError("invalid static interior")
    result = []
    for axis, (dim, offset, size) in enumerate(zip(shape, offsets, sizes)):
        for start, extent in ((0, offset), (offset + size, dim - offset - size)):
            if not extent:
                continue
            slab_offsets = tuple(offsets[:axis]) + (start,) + (0,) * (len(shape) - axis - 1)
            slab_sizes = tuple(sizes[:axis]) + (extent,) + tuple(shape[axis + 1 :])
            result.append((slab_offsets, slab_sizes))
    return result


def _border_destination(empty, value, shape, offsets, sizes):
    current = empty.tensor
    operations = [empty]
    for slab_offsets, slab_sizes in exterior_slabs(shape, offsets, sizes):
        view = tensor.ExtractSliceOp.from_static_parameters(current, slab_offsets, slab_sizes, (1,) * len(shape))
        fill = FillOp([value], [view.result], [view.result.type])
        insert = tensor.InsertSliceOp.from_static_parameters(
            fill.results[0], current, slab_offsets, slab_sizes, (1,) * len(shape)
        )
        operations.extend((view, fill, insert))
        current = insert.result
    return operations, current


def _safe_source(producer, insert):
    """Moving scalar arithmetic may cross constants/storage constructors only."""
    from xdsl.dialects import arith, builtin, func

    from .quant_round_check import pure_scalar_body

    block = insert.parent_block()
    parent = block.parent_op() if block is not None else None
    if not isinstance(parent, func.FuncOp) or len(parent.body.blocks) != 1:
        return False
    ancestor = parent
    while ancestor is not None:
        allowed = {"llvm.emit_c_interface"} if isinstance(ancestor, func.FuncOp) else (
            {"llvm.data_layout", "llvm.target_triple"} if isinstance(ancestor, builtin.ModuleOp) else set()
        )
        if any(k not in allowed and not k.startswith("prov.") for k in ancestor.attributes):
            return False
        ancestor = ancestor.parent_op()
    if any(not k.startswith("prov.") for op in (insert, insert.dest.owner, producer.outputs[0].owner)
           for k in op.attributes):
        return False
    if (not pure_scalar_body(producer.body.block)
            or any(not k.startswith("prov.") for k in producer.attributes)):
        return False
    if set(producer.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        return False
    domain = producer.results[0].type.get_shape()
    for value, attr in zip(producer.inputs, producer.indexing_maps, strict=False):
        typ, mapping = value.type, attr.data
        if (not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr)
                or any(n <= 0 for n in typ.get_shape()) or mapping.num_symbols
                or mapping.num_dims != len(domain) or len(mapping.results) != len(typ.get_shape())):
            return False
        dimensions = []
        for extent, expression in zip(typ.get_shape(), mapping.results, strict=True):
            if isinstance(expression, AffineDimExpr):
                if not 0 <= expression.position < len(domain) or extent != domain[expression.position]:
                    return False
                dimensions.append(expression.position)
            elif not isinstance(expression, AffineConstantExpr) or expression.value != 0 or extent != 1:
                return False
        if len(set(dimensions)) != len(dimensions):
            return False
    for scalar in producer.body.block.ops:
        for operand in scalar.operands:
            owner = operand.owner
            if owner is not producer.body.block and (
                not isinstance(owner, Operation) or owner.parent_block() is not producer.body.block
            ):
                return False
    cursor = producer.next_op
    while cursor is not insert:
        if cursor is None or not isinstance(cursor, (arith.ConstantOp, tensor.EmptyOp, tensor.SplatOp)):
            return False
        if any(not k.startswith("prov.") for k in cursor.attributes):
            return False
        cursor = cursor.next_op
    return True


def rewrite_module(
    module, *, initialize_border_only=False, max_added_operations=None, max_metadata_items=None, index_bits=None,
):
    """Rewrite proved static regions, optionally bounded before constructing IR.

    These are IR growth budgets, not hardware dimensions or a cycle estimate.
    The standalone unbounded API remains available to explicit callers.
    """
    for budget in (max_added_operations, max_metadata_items):
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("destination IR budgets must be nonnegative integers or unknown")
    if index_bits is not None and (type(index_bits) is not int or not 2 <= index_bits <= 128):
        raise ValueError("destination index width must be an explicit integer within the 2..128-bit reader budget")
    accepted = []
    added_operations = metadata_items = 0
    for insert in list(module.walk()):
        if not isinstance(insert, tensor.InsertSliceOp):
            continue
        producer = insert.source.owner
        pad = insert.dest.owner
        if (
            not isinstance(producer, Operation)
            or producer.name != "linalg.generic"
            or not isinstance(pad, tensor.SplatOp)
        ):
            continue
        if (
            producer.parent_block() is not insert.parent_block()
            or len(producer.results) != 1
            or len(producer.outputs) != 1
        ):
            continue
        init_owner = producer.outputs[0].owner
        if (
            len(list(insert.source.uses)) != 1
            or not isinstance(init_owner, Operation)
            or init_owner.name != "tensor.empty"
        ):
            continue
        source_type = insert.source.type
        dest_type = insert.dest.type
        if any(
            not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr) or any(d <= 0 for d in t.get_shape())
            for t in (source_type, dest_type)
        ):
            continue
        rank = len(source_type.get_shape())
        if index_bits is not None:
            from math import prod

            from xdsl.dialects.builtin import IntegerType

            element = dest_type.get_element_type()
            width = element.width.data if isinstance(element, IntegerType) else getattr(element, "bitwidth", None)
            if (type(width) is not int or width <= 0 or width % 8
                    or prod(dest_type.get_shape()) * (width // 8) >= 2 ** (index_bits - 1)):
                continue
        if len(dest_type.get_shape()) != rank or source_type.get_element_type() != dest_type.get_element_type():
            continue
        if source_type.get_shape() == dest_type.get_shape():
            # A full overwrite folds to its source; adding a padded destination
            # cannot improve that canonical route.
            continue
        offsets = tuple(insert.static_offsets.get_values())
        sizes = tuple(insert.static_sizes.get_values())
        strides = tuple(insert.static_strides.get_values())
        if (
            len(offsets) != rank
            or sizes != source_type.get_shape()
            or strides != (1,) * rank
            or any(o < 0 or o + n > d for o, n, d in zip(offsets, sizes, dest_type.get_shape()))
        ):
            continue
        if insert.offsets or insert.sizes or insert.strides or pad.dynamicSizes:
            continue
        if producer.properties["indexing_maps"].data[-1].data != AffineMap.identity(rank):
            continue
        if any(x.data.value != "parallel" for x in producer.properties["iterator_types"].data):
            continue
        body = producer.body.block
        if body.args[-1].uses:
            continue
        if any(
            op.regions or not (op.name.startswith(("arith.", "math.")) or op.name == "linalg.yield") for op in body.ops
        ):
            continue
        if not _safe_source(producer, insert):
            continue
        slabs = sum((o > 0) + (o + n < d) for o, n, d in zip(offsets, sizes, dest_type.get_shape()))
        growth = 2 + (3 * slabs if initialize_border_only else 1)
        # Each emitted slice/insert carries rank-sized type and geometry data.
        metadata = (12 * slabs + 12) * rank
        if (max_added_operations is not None and added_operations + growth > max_added_operations
                or max_metadata_items is not None and metadata_items + metadata > max_metadata_items):
            continue
        empty = tensor.EmptyOp([], dest_type)
        if initialize_border_only:
            initialization, destination = _border_destination(empty, pad.input, dest_type.get_shape(), offsets, sizes)
        else:
            fill = FillOp([pad.input], [empty.tensor], [dest_type])
            initialization, destination = [empty, fill], fill.results[0]
        view = tensor.ExtractSliceOp.from_static_parameters(destination, offsets, sizes, strides)
        clone = producer.clone()
        clone.operands = [*producer.inputs, view.result]
        replacement = tensor.InsertSliceOp.from_static_parameters(
            clone.results[0], destination, offsets, sizes, strides
        )
        replacement.attributes.update(insert.attributes)
        block = insert.parent_block()
        block.insert_ops_before([*initialization, view, clone, replacement], insert)
        insert.result.replace_all_uses_with(replacement.result)
        block.erase_op(insert)
        block.erase_op(producer)
        added_operations += growth
        metadata_items += metadata
        accepted.append(
            {
                "source_shape": list(source_type.get_shape()),
                "destination_shape": list(dest_type.get_shape()),
                "offsets": list(offsets),
                "scalar_body": "unchanged clone",
                "storage": (
                    "fresh exterior slabs with fully overwritten interior"
                    if initialize_border_only
                    else "fresh fill with producer writing exact interior slice"
                ),
            }
        )
    module.verify()
    return accepted


def rewrite_prepared_file(source, work, *, initialize_border_only=False):
    import hashlib
    import json
    from pathlib import Path

    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects._common import text

    source, work = Path(source), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    module = parse_mlir_file(source)
    routes = rewrite_module(module, initialize_border_only=initialize_border_only)
    target = work / "model.mlir"
    target.write_text(text(module))
    parse_mlir_file(target).verify()
    report = {
        "schema": "exact_insert_slice_destination_v1",
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "rewritten_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "routes": routes,
        "scalar_arithmetic_changed": False,
        "call_abi_changed": False,
    }
    (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[destination_reuse] pointwise insert-slice rewrites={len(routes)}")
    return target
