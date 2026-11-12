"""Specialize disjoint static copies into contiguous suffixes and outer loops.

The element representation is unchanged. Fresh, distinct allocation roots prove
non-overlap; views of one allocation and unknown origins are refused.
The original lexicographic outer traversal is retained. Upstream lowers each
contiguous suffix copy to memcpy, whose runtime owns alignment and byte tails.
"""

FEATURE = "specialize_contiguous_copy"

RUNNER_PRELUDE = r"""
def _copy_allocation_root(value):
    seen = set()
    views = {"memref.subview", "memref.cast", "memref.collapse_shape",
             "memref.expand_shape", "memref.reinterpret_cast", "memref.view"}
    while value not in seen:
        seen.add(value)
        owner = value.owner
        if not hasattr(owner, "operation"):
            return None
        op = owner.operation
        if op.name == "memref.alloc":
            return op
        if op.name not in views or not op.operands:
            return None
        value = op.operands[0]
    return None


def _specialize_contiguous_copies(ctx, module):
    from torch_mlir import ir
    from torch_mlir.dialects import arith, memref, scf
    plans = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    walk(child.operation)
                    cp = child.operation
                    if cp.name != "memref.copy":
                        continue
                    src, dst = cp.operands
                    roots = (_copy_allocation_root(src), _copy_allocation_root(dst))
                    if roots[0] is None or roots[1] is None or roots[0] == roots[1]:
                        continue
                    try:
                        st, dt = ir.MemRefType(src.type), ir.MemRefType(dst.type)
                        ss, _ = st.get_strides_and_offset()
                        ds, _ = dt.get_strides_and_offset()
                    except (ValueError, TypeError):
                        continue
                    shape = list(st.shape)
                    if (len(shape) < 2 or shape != list(dt.shape)
                        or st.element_type != dt.element_type or st.memory_space != dt.memory_space
                        or any(d <= 0 for d in shape) or any(s <= 0 for s in [*ss, *ds])):
                        continue
                    axis, span = len(shape), 1
                    for i in reversed(range(len(shape))):
                        if shape[i] != 1 and (ss[i] != span or ds[i] != span):
                            break
                        span *= shape[i]
                        axis = i
                    if axis == 0 or axis == len(shape) or span <= 1 or span >= (1 << 63):
                        continue
                    total = span
                    for d in shape[:axis]:
                        total *= d
                    if total >= (1 << 63):
                        continue
                    plans.append((cp, st, dt, shape, list(ss), list(ds), axis))
    walk(module.operation)
    with ctx, ir.Location.unknown():
        for cp, st, dt, shape, ss, ds, axis in plans:
            source, dest = cp.operands
            rank = len(shape)
            dynamic = ir.ShapedType.get_dynamic_size()
            index = ir.IndexType.get()
            with ir.InsertionPoint(cp):
                zero = arith.ConstantOp(index, 0).result
                one = arith.ConstantOp(index, 1).result
                bounds = [arith.ConstantOp(index, n).result for n in shape[:axis]]
                def emit(level, offsets):
                    if level < axis:
                        if shape[level] == 1:
                            emit(level + 1, [*offsets, zero])
                            return
                        loop = scf.ForOp(zero, bounds[level], one)
                        with ir.InsertionPoint(loop.body):
                            emit(level + 1, [*offsets, loop.induction_variable])
                            scf.YieldOp([])
                        return
                    static_offsets = [dynamic] * axis + [0] * (rank - axis)
                    static_sizes = [1] * axis + shape[axis:]
                    rows = []
                    for value, ty, strides in [(source, st, ss), (dest, dt, ds)]:
                        layout = ir.StridedLayoutAttr.get(dynamic, strides[axis:])
                        rowty = ir.MemRefType.get(shape[axis:], ty.element_type, layout=layout, memory_space=ty.memory_space)
                        row = memref.SubViewOp(rowty, value, offsets, [], [], static_offsets, static_sizes, [1] * rank)
                        rows.append(row.result)
                    memref.CopyOp(rows[0], rows[1])
                emit(0, [])
            cp.erase()
    return len(plans)
"""

MID_STAGE_SRC = r"""
if len(sys.argv) > 18 and sys.argv[18] == "1":
    _MID_STAGES.insert(0, ("specialize_contiguous_copy", _specialize_contiguous_copies))
"""


def require_report(stdout, work):
    """A selected runner must report its exact application count."""
    import json
    from pathlib import Path

    prefix = "OK specialize_contiguous_copy "
    rows = [line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1 or not rows[0].isdigit():
        raise ValueError("selected contiguous-copy rewrite did not report exactly one count")
    result = {"schema": "merlin.contiguous_suffix_copy.v1", "copies_specialized": int(rows[0])}
    (Path(work) / "contiguous_suffix_copy.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
