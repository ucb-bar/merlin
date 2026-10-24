"""Declared tensor layouts for certified CPU scalar/vector storage forms.

Only metadata keys are copied. Payloads, expected results, operation names and
case identities cannot choose a path. Unknown compound layouts stay unproved.
"""


def input_layouts(report, arguments):
    result = {}

    def tensors(value):
        if isinstance(value, dict):
            if value.get("kind") == "tensor":
                return [{key: value[key] for key in ("shape", "stride") if key in value}]
            return [item for child in value.values() for item in tensors(child)]
        if isinstance(value, list):
            return [item for child in value for item in tensors(child)]
        return []

    for record in report["cases"]:
        if record.get("status") not in {None, "bundled"}:
            continue
        layouts = tensors((record.get("case") or {}).get("arguments", {}))
        indices = record.get("input_indices", [])
        if len(layouts) != len(indices):
            continue
        for index, layout in zip(indices, layouts):
            if index < len(arguments):
                result[arguments[index]] = layout
    return result


def profile(layout, *, block, grain):
    from math import prod

    shape, stride = layout.get("shape"), layout.get("stride")
    if shape is None or stride is None or len(shape) != len(stride) or any(s < 0 for s in (*shape, *stride)):
        return None
    total = prod(shape)
    if total > grain:
        return None  # Parallel range boundaries need their explicitly captured source profile.
    axes = sorted((i for i, n in enumerate(shape) if n > 1), key=lambda i: (stride[i] == 0, stride[i], -i))
    size = 1
    dense = True
    for axis in axes:
        if stride[axis] != size:
            dense = False
        size *= shape[axis]
    if dense:
        row, vector = total, True
    elif len(shape) <= 1 or all(s == 0 for s in stride):
        axes = list(reversed(range(len(shape))))
        row, vector = total, not stride or stride[0] in (0, 1)
    else:
        # TensorIterator coalesces adjacent dimensions exactly when the outer
        # stride equals inner size * stride. Gaps terminate a row; zero strides
        # coalesce only with zero strides. Storage offset changes no step.
        active = [axis for axis in axes if stride[axis]]
        span = 1
        for axis in active:
            if stride[axis] < span:
                return None  # Unproved overlapping non-broadcast layout.
            span += (shape[axis] - 1) * stride[axis]
        row = 1
        if axes:
            row = shape[axes[0]]
            for inner, outer in zip(axes, axes[1:]):
                if stride[outer] != stride[inner] * shape[inner]:
                    break
                row *= shape[outer]
        vector = not axes or stride[axes[0]] in (0, 1)
    return dict(
        shape=list(shape),
        stride=list(stride),
        axes=axes,
        row=row,
        vector_prefix=row // block * block if vector else 0,
        vector_width=block,
        dense=dense,
        theorem="static_tensor_iterator_rows_v2",
    )


def vector_predicate(record, emit):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType, IntegerAttr, i1
    from xdsl.dialects.linalg import ops as L

    from merlin.llvmlower._ir import constants

    if not record["vector_prefix"]:
        return emit(arith.ConstantOp(IntegerAttr(0, i1)))

    c = constants(emit, IndexType())
    position, factor = c(0), 1
    for axis in record["axes"]:
        index = emit(L.IndexOp(axis))
        position = emit(arith.AddiOp(position, emit(arith.MuliOp(index, c(factor)))))
        factor *= record["shape"][axis]
    local = emit(arith.RemUIOp(position, c(max(1, record["row"]))))
    return emit(arith.CmpiOp(local, c(record["vector_prefix"]), "ult"))


def joint_profile(layouts, shape, *, block, grain):
    """Closed functional TensorIterator ordering/coalescing, including broadcast.

    Output allocation follows the input permutation. Zero strides do not decide
    order, and each adjacent dimension must coalesce for *every* operand.
    """
    from math import prod

    if shape is None or not layouts or prod(shape) > grain or any(n <= 0 for n in shape):
        return None
    rank = len(shape)
    strides = []
    for layout in layouts:
        source, stride = layout.get("shape"), layout.get("stride")
        if source is None or stride is None or len(source) != len(stride) or len(source) > rank:
            return None
        if profile(layout, block=block, grain=grain) is None:
            return None
        source = [1] * (rank - len(source)) + list(source)
        stride = [0] * (rank - len(stride)) + list(stride)
        if any(a not in (1, b) for a, b in zip(source, shape)):
            return None
        strides.append([0 if n == 1 else s for n, s in zip(source, stride)])
    axes = [i for i in reversed(range(rank)) if shape[i] > 1]

    def compare(a, b):
        for stride in strides:
            if stride[a] == 0 or stride[b] == 0:
                continue
            if stride[a] != stride[b]:
                return -1 if stride[a] < stride[b] else 1
            if shape[a] > shape[b]:
                return 1
        return 0

    # TensorIterator's stable insertion ordering leaves ambiguous zeros in
    # logical reverse order until another operand supplies a strict comparison.
    for i in range(1, len(axes)):
        j = i
        for k in range(i - 1, -1, -1):
            order = compare(axes[k], axes[j])
            if order > 0:
                axes[k], axes[j] = axes[j], axes[k]
                j = k
            elif order < 0:
                break
    row = shape[axes[0]] if axes else 1
    for a, b in zip(axes, axes[1:]):
        if any(stride[b] != shape[a] * stride[a] for stride in strides):
            break
        row *= shape[b]
    vector = not axes or all(stride[axes[0]] in (0, 1) for stride in strides)
    # The loop vectorizes at most one broadcast operand; otherwise it uses
    # the scalar functor, including the all-scalar case.
    if axes and sum(stride[axes[0]] == 0 for stride in strides) > 1:
        vector = False
    return dict(
        shape=list(shape),
        stride=[],
        axes=axes,
        row=row,
        vector_prefix=row // block * block if vector else 0,
        vector_width=block,
        dense=False,
        theorem="static_joint_tensor_iterator_rows_v1",
    )
