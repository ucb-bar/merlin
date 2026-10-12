"""Exact typed public ATen shape relations for ordinary connected cuts.

Keep original scalar/dtype arguments and publication. Integer casts describe
source representations, never a new quantization permission or physical layout.
"""

from __future__ import annotations

import math

DTYPES = {
    "bool": "i1",
    "uint8": "u8",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
}
CONTRACTIONS = {"aten.linear.default", "aten.mm.default", "aten.matmul.default", "aten.bmm.default"}
CASTS = {"aten.to.dtype", "aten._to_copy.default"}
REDUCTIONS = {"aten.sum.dim_IntList", "aten.mean.dim", "aten.softmax.int", "aten._softmax.default"}
SUPPORTED = (
    CONTRACTIONS
    | CASTS
    | REDUCTIONS
    | {
        "aten.clone.default",
        "aten.relu.default",
        "aten.transpose.int",
        "aten.permute.default",
        "aten.add.Tensor",
        "aten.mul.Tensor",
        "aten.mul.Scalar",
        "aten.div.Tensor",
        "aten.div.Scalar",
        "aten.round.default",
        "aten.clamp.default",
    }
)


def layout_order(row):
    """Retain observed dense permutations; no arbitrary strided/alias proof."""
    stride = row.get("stride")
    if stride is None:
        return None
    shape = row["shape"]
    if type(stride) is not list or len(stride) != len(shape) or any(type(s) is not int or s < 1 for s in stride):
        raise ValueError("source input has unsupported broadcast/negative/opaque strides")
    order = sorted(range(len(shape)), key=lambda i: (-stride[i], i))
    running = 1
    for i in reversed(order):
        if shape[i] != 1 and stride[i] != running:
            raise ValueError("source input is not an observed dense permutation")
        running *= shape[i]
    return order


def dtype_literal(value):
    if type(value) is dict and set(value) == {"kind", "value"} and value["kind"] == "dtype":
        name = value["value"].removeprefix("torch.") if type(value["value"]) is str else None
        if name in DTYPES and value["value"] == "torch." + name:
            return name
    raise ValueError("original dtype argument is not a supported exact Torch literal")


def literal(value):
    """Render only closed source literals, never arbitrary expressions."""
    if type(value) is dict and set(value) == {"kind", "value"}:
        return "torch." + dtype_literal(value)
    if value is None or type(value) in {int, bool, str}:
        return repr(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    raise ValueError("original typed source literal is unsupported")


def shape_relations(node, values):
    from .public_boundary_semantics import references

    target, args, kwargs = node["target"], node["args"], node["kwargs"]
    if target not in SUPPORTED or len(node["results"]) != 1:
        raise ValueError("original typed call has no exact connected constructor")
    refs = list(references(args))
    result = node["results"][0]
    rows = [values[r] for r in refs]
    if not refs or any(
        r.get("kind") != "tensor"
        or r.get("dtype") not in DTYPES
        or type(r.get("shape")) is not list
        or any(type(d) is not int or d < 1 for d in r["shape"])
        for r in [*rows, result]
    ):
        raise ValueError("typed source requires supported static tensor shapes and storage")
    if type(args[0]) is not dict or args[0].get("value_id") != refs[0]:
        raise ValueError("original ordered Tensor operands are unavailable")
    source, output, out = rows[0]["shape"], result["shape"], result["id"]
    relations, units = [], []

    def same(a, b, pairs):
        for i, j in pairs:
            if values[a]["shape"][i] != values[b]["shape"][j]:
                raise ValueError("original tensor types disagree with exact shape semantics")
            relations.append(((a, i), (b, j)))

    def unchanged():
        if output != source:
            raise ValueError("original elementwise shape differs")
        same(refs[0], out, enumerate(range(len(source))))

    if target in CASTS:
        unchanged()
        if target == "aten.to.dtype":
            if not 2 <= len(args) <= 5 or set(kwargs) - {"non_blocking", "copy", "memory_format"}:
                raise ValueError("cast source flags are unsupported")
            destination = dtype_literal(args[1])
            flags = [*args[2:4], *[kwargs[k] for k in ("non_blocking", "copy") if k in kwargs]]
            if (
                any(type(v) is not bool for v in flags)
                or (len(args) > 4 and args[4] is not None)
                or kwargs.get("memory_format") is not None
            ):
                raise ValueError("cast source changes device or memory-format semantics")
        else:
            if len(args) != 1 or set(kwargs) - {
                "dtype",
                "non_blocking",
                "memory_format",
                "pin_memory",
                "device",
                "layout",
            }:
                raise ValueError("copy cast source flags are unsupported")
            destination = dtype_literal(kwargs.get("dtype"))
            if (
                any(kwargs.get(k) is not None for k in ("memory_format", "device", "layout"))
                or kwargs.get("pin_memory", False) is not False
                or kwargs.get("non_blocking", False) is not False
            ):
                raise ValueError("copy cast requires the original synchronous logical storage")
        if result["dtype"] != destination:
            raise ValueError("cast destination differs from its original literal")
    elif target in {"aten.sum.dim_IntList", "aten.mean.dim"}:
        if not 2 <= len(args) <= 3 or set(kwargs) - {"keepdim", "dtype"}:
            raise ValueError("reduction source flags are unsupported")
        dimensions = args[1]
        if dimensions is None or dimensions == []:
            dimensions = list(range(len(source)))
        if type(dimensions) is not list or any(
            type(d) is not int or not -len(source) <= d < len(source) for d in dimensions
        ):
            raise ValueError("reduction requires its original exact axes")
        axes = {d % len(source) for d in dimensions}
        keep = args[2] if len(args) == 3 else kwargs.get("keepdim", False)
        if type(keep) is not bool or (len(args) == 3 and "keepdim" in kwargs):
            raise ValueError("reduction keepdim is ambiguous")
        expected = [1 if i in axes else d for i, d in enumerate(source) if keep or i not in axes]
        if output != expected:
            raise ValueError("reduction result differs from original axes")
        requested = kwargs.get("dtype")
        expected_dtype = (
            dtype_literal(requested)
            if requested is not None
            else (
                "int64"
                if target == "aten.sum.dim_IntList" and rows[0]["dtype"] in {"bool", "uint8", "int8", "int16", "int32"}
                else rows[0]["dtype"]
            )
        )
        if result["dtype"] != expected_dtype or (
            target == "aten.mean.dim" and not expected_dtype.startswith(("float", "bfloat"))
        ):
            raise ValueError("reduction result dtype differs from original promotion")
        j = 0
        for i in range(len(source)):
            if i in axes:
                if keep:
                    units.append((out, j))
                    j += 1
            else:
                same(refs[0], out, [(i, j)])
                j += 1
    elif target in {"aten.softmax.int", "aten._softmax.default"}:
        unchanged()
        if (
            not len(source)
            or not 2 <= len(args) <= 3
            or type(args[1]) is not int
            or not -len(source) <= args[1] < len(source)
            or set(kwargs) - {"dtype"}
        ):
            raise ValueError("softmax requires an exact original axis/dtype")
        if target == "aten._softmax.default":
            if len(args) != 3 or args[2] is not False or kwargs:
                raise ValueError("half-to-float softmax has no admitted CPU reference")
            requested = None
        else:
            requested = args[2] if len(args) == 3 else kwargs.get("dtype")
        dtype = dtype_literal(requested) if requested is not None else rows[0]["dtype"]
        if result["dtype"] != dtype or not dtype.startswith(("float", "bfloat")):
            raise ValueError("softmax result differs from original floating storage")
    elif target in {"aten.transpose.int", "aten.permute.default"}:
        if kwargs:
            raise ValueError("permutation has unknown original keyword arguments")
        if target == "aten.transpose.int":
            if len(args) != 3 or any(type(d) is not int or not -len(source) <= d < len(source) for d in args[1:]):
                raise ValueError("transpose axes differ from the original static call")
            order = list(range(len(source)))
            a, b = (d % len(source) for d in args[1:])
            order[a], order[b] = order[b], order[a]
        else:
            if (
                len(args) != 2
                or type(args[1]) is not list
                or len(args[1]) != len(source)
                or any(type(d) is not int or not -len(source) <= d < len(source) for d in args[1])
            ):
                raise ValueError("permutation requires all original axes")
            order = [d % len(source) for d in args[1]]
            if len(set(order)) != len(source):
                raise ValueError("permutation repeats an axis")
        if result["dtype"] != rows[0]["dtype"] or output != [source[d] for d in order]:
            raise ValueError("original view shape or storage differs")
        same(refs[0], out, [(i, j) for j, i in enumerate(order)])
    elif target in CONTRACTIONS:
        linear = target == "aten.linear.default"
        if kwargs or len(args) not in ({2, 3} if linear else {2}) or len(rows) not in ({2, 3} if linear else {2}):
            raise ValueError("contraction operands are not the original complete Tensor roster")
        rhs = rows[1]["shape"]
        if any(r["dtype"] != rows[0]["dtype"] for r in [*rows, result]):
            raise ValueError("contraction storage differs from original dtype semantics")
        if linear:
            if not source or len(rhs) != 2 or output != [*source[:-1], rhs[0]]:
                raise ValueError("linear result shape differs")
            same(refs[0], refs[1], [(len(source) - 1, 1)])
            same(refs[0], out, [(i, i) for i in range(len(source) - 1)])
            same(refs[1], out, [(0, len(output) - 1)])
            if len(rows) == 3:
                if rows[2]["shape"] != [output[-1]]:
                    raise ValueError("linear bias extent differs")
                same(refs[2], out, [(0, len(output) - 1)])
        else:
            if (
                len(source) < 2
                or len(rhs) < 2
                or (target == "aten.mm.default" and (len(source) != 2 or len(rhs) != 2))
                or (target == "aten.bmm.default" and (len(source) != 3 or len(rhs) != 3 or source[0] != rhs[0]))
            ):
                raise ValueError("contraction ranks/batches have no exact supported constructor")
            batch = max(len(source), len(rhs)) - 2
            if len(output) != batch + 2:
                raise ValueError("batched result rank differs")
            expected = []
            for j in range(batch):
                pair = [(ref, len(values[ref]["shape"]) - 2 - batch + j) for ref in refs[:2]]
                extents = [values[ref]["shape"][i] if i >= 0 else 1 for ref, i in pair]
                if len(set(extents) - {1}) > 1:
                    raise ValueError("matmul batch broadcasting differs")
                expected.append(max(extents))
                for (ref, i), d in zip(pair, extents, strict=True):
                    if i >= 0 and d != 1:
                        same(ref, out, [(i, j)])
            if output != [*expected, source[-2], rhs[-1]]:
                raise ValueError("batched contraction result shape differs")
            same(refs[0], refs[1], [(len(source) - 1, len(rhs) - 2)])
            same(refs[0], out, [(len(source) - 2, len(output) - 2)])
            same(refs[1], out, [(len(rhs) - 1, len(output) - 1)])
    elif target in {"aten.add.Tensor", "aten.mul.Tensor", "aten.div.Tensor", "aten.mul.Scalar", "aten.div.Scalar"}:
        if len(args) != 2 or set(kwargs) - ({"alpha"} if target == "aten.add.Tensor" else set()):
            raise ValueError("binary operation does not preserve exact original defaults")
        rank = max(len(r["shape"]) for r in rows)
        if len(output) != rank:
            raise ValueError("binary result rank differs")
        expected = []
        for j in range(rank):
            pairs = [(ref, len(values[ref]["shape"]) - rank + j) for ref in refs]
            extents = [values[ref]["shape"][i] if i >= 0 else 1 for ref, i in pairs]
            if len(set(extents) - {1}) > 1:
                raise ValueError("binary broadcasting differs")
            expected.append(max(extents))
            for (ref, i), d in zip(pairs, extents, strict=True):
                if i >= 0 and d != 1:
                    same(ref, out, [(i, j)])
        if output != expected:
            raise ValueError("binary result shape differs")
    else:
        unchanged()
        allowed = (
            {"memory_format"}
            if target == "aten.clone.default"
            else {"min", "max"}
            if target == "aten.clamp.default"
            else set()
        )
        if (
            set(kwargs) - allowed
            or (target == "aten.clone.default" and kwargs.get("memory_format") is not None)
            or not 1 <= len(args) <= (3 if target == "aten.clamp.default" else 1)
            or result["dtype"] != rows[0]["dtype"]
        ):
            raise ValueError("elementwise source storage/defaults differ")
    return relations, units
