"""Read any int8 contraction as a GEMM, and rewrite it into the one layout a matrix-unit kernel takes.

A matrix-unit kernel has one calling convention -- ``A(b?, m, k) . B(b?, k, n) -> C(b?, m, n)``, int8 x
int8 -> int32 -- and a frontend can hand over the same arithmetic in many layouts: a transposed operand
(``A.B^T``, an attention score), several batch dims (``(batch, head, seq, dim)``), several free or
reduced dims (a 1x1 or an im2col'd convolution kept in NCHW), or a WINDOWED input map (a direct
convolution, strided or dilated). Each of those is the same contraction; refusing them all because
their maps are not the canonical three costs every MAC they carry.

This module derives the GEMM from the op's SEMANTICS -- its indexing maps and iterator types -- and
nothing else. Every iteration dim is classified by which operands index it:

========  =====================  ==========================================
role      indexed by              becomes
========  =====================  ==========================================
batch     A, B and C             the kernel's leading (looped) extent
m         A and C only           rows of the kernel's A and of C
n         B and C only           columns of the kernel's B and of C
k         A and B only           the reduction (its iterator must say so)
========  =====================  ==========================================

and the op is rewritten as: transpose each operand to ``(batch, free, k)`` / ``(batch, k, free)`` order,
collapse each role group to one dim, call the kernel, expand the result back and transpose it into the
op's output order. A transpose or collapse that would be the identity is not emitted, so a contraction
already in the kernel's layout is untouched. A windowed operand is first gathered (im2col) straight into
that order by an all-parallel ``linalg.generic`` that reads it through the op's own map -- the same
gather a convolution frontend emits -- after which its map is a plain projection like any other.

Refused, with the reason, exactly when the rewrite would not compute the op: a dim one operand reduces
alone (a sum, not a contraction), an output dim no input indexes (a broadcast), a role whose iterator
type disagrees with it, both inputs windowed, an output map that is not a permutation, an extent that
cannot be read, and an operand with no free dim at all. The last is the depthwise / per-channel form:
every output reads one element of that operand per reduction step, so the outer product the unit
computes would be one lane wide; it is refused by name rather than priced.

The role of each operand can also be swapped -- ``C^T = B^T . A^T`` -- which is the same arithmetic with
the result read out along the other dim; :mod:`matrix_route_cost` decides which, and this module emits
either. Numerics are unchanged by construction: only data movement is added, and the kernel's int32
accumulation is the op's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["AS_IS", "SWAPPED", "NormalForm", "analyze", "emit", "site_of"]

AS_IS, SWAPPED = "as_is", "swapped"

#: Named contraction ops and their default (row-major) maps as iteration-dim lists.
_NAMED_DEFAULT_MAPS: dict[str, tuple[tuple[int, ...], ...]] = {
    "linalg.matmul": ((0, 2), (2, 1), (0, 1)),
    "linalg.batch_matmul": ((0, 1, 3), (0, 3, 2), (0, 1, 2)),
}
_INT8_TRIPLE = ("i8", "i8", "i32")


@dataclass(frozen=True)
class NormalForm:
    """One contraction read as ``batch x (m . k) x (k . n)``, with how to lay it out for a kernel."""

    op: Any
    operand_dims: tuple[tuple[int, ...], tuple[int, ...]]  # iteration dims of A, B (gathered, if windowed)
    out_dims: tuple[int, ...]
    extents: dict[int, int]
    batch: tuple[int, ...]
    m: tuple[int, ...]
    n: tuple[int, ...]
    k: tuple[int, ...]
    #: Index (0 or 1) of the input read through a windowed (non-projection) map, or None.
    windowed: int | None = None
    window_map: Any = None
    window_dims: tuple[int, ...] = ()
    transforms: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def size(self, dims: tuple[int, ...]) -> int:
        out = 1
        for d in dims:
            out *= int(self.extents[d])
        return out

    @property
    def bmnk(self) -> tuple[int, int, int, int]:
        return self.size(self.batch), self.size(self.m), self.size(self.n), self.size(self.k)

    @property
    def macs(self) -> int:
        b, m, n, k = self.bmnk
        return b * m * n * k

    def kernel_layout(self, orientation: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int]:
        """``(kernel A dims, kernel B dims, kernel C dims, op input feeding kernel A, op input feeding B)``."""
        if orientation == AS_IS:
            return (
                (*self.batch, *self.m, *self.k),
                (*self.batch, *self.k, *self.n),
                (*self.batch, *self.m, *self.n),
                0,
                1,
            )
        if orientation == SWAPPED:
            return (
                (*self.batch, *self.n, *self.k),
                (*self.batch, *self.k, *self.m),
                (*self.batch, *self.n, *self.m),
                1,
                0,
            )
        raise ValueError(f"unknown orientation {orientation!r}")

    def steps(self, orientation: str) -> tuple[str, ...]:
        """The data movement ``emit`` would add, named, for a report (empty: the op is already canonical)."""
        ka, kb, kc, ia, ib = self.kernel_layout(orientation)
        out: list[str] = []
        for want, src, name in ((ka, ia, "a"), (kb, ib, "b")):
            if self.windowed == src:
                out.append(f"im2col_{name}")
            elif tuple(self.operand_dims[src]) != tuple(want):
                out.append(f"transpose_{name}")
        if any(len(g) > 1 for g in (self.batch, self.m, self.n, self.k)):
            out.append("collapse")
        if tuple(kc) != tuple(self.out_dims):
            out.append("transpose_c")
        return tuple(out)

    def layout_elements(self, orientation: str) -> int:
        """Elements the layout steps of ``orientation`` move: each transposed or gathered operand, and the
        result when it has to be transposed back. A collapse or expand is a view and moves nothing."""
        ka, kb, kc, ia, ib = self.kernel_layout(orientation)
        total = 0
        for want, src in ((ka, ia), (kb, ib)):
            if self.windowed == src or tuple(self.operand_dims[src]) != tuple(want):
                total += self.size(tuple(want))
        if tuple(kc) != tuple(self.out_dims):
            total += self.size(tuple(kc))
        return total

    def shape(self, orientation: str = AS_IS):
        """The kernel's view as a :class:`ContractionShape`: ``(b?, m, n)`` parallel, ``(k,)`` reduction."""
        from ..kernels.microkernel import ContractionShape

        b, m, n, k = self.bmnk
        if orientation == SWAPPED:
            m, n = n, m
        parallel = (b, m, n) if self.batch else (m, n)
        op = "linalg.batch_matmul" if self.batch else "linalg.matmul"
        return ContractionShape(op=op, parallel=parallel, reduction=(k,), dtypes=_INT8_TRIPLE)


def _dim_of(expr) -> int | None:
    from xdsl.ir.affine import AffineDimExpr

    return int(expr.position) if isinstance(expr, AffineDimExpr) else None


def _maps(op) -> list[list[Any]] | None:
    """Result expressions per operand (A, B, C). Named ops without explicit maps get their defaults."""
    from xdsl.ir.affine import AffineDimExpr

    from ..kernels.shapes import indexing_maps

    maps = indexing_maps(op)
    if maps is not None:
        return maps
    default = _NAMED_DEFAULT_MAPS.get(getattr(op, "name", ""))
    if default is None:
        return None
    return [[AffineDimExpr(d) for d in dims] for dims in default]


def _iterators(op) -> list[str] | None:
    from ..kernels.shapes import _iterator_types

    its = _iterator_types(op)
    if its is not None:
        return [("reduction" if "reduction" in it else "parallel") for it in its]
    default = _NAMED_DEFAULT_MAPS.get(getattr(op, "name", ""))
    if default is None:
        return None
    n = len(default[2]) + 1
    return ["parallel"] * (n - 1) + ["reduction"]


def _body_is_int_mac(op) -> bool:
    """The body is ``acc + ext(a) * ext(b)`` in integers: only sign extensions, one multiply, one add."""
    if getattr(op, "name", "") in _NAMED_DEFAULT_MAPS:
        return True
    regions = getattr(op, "regions", ())
    if not regions or not regions[0].blocks:
        return False
    names = [o.name for o in regions[0].blocks[0].ops]
    allowed = {"arith.extsi", "arith.muli", "arith.addi", "linalg.yield"}
    return set(names) <= allowed and names.count("arith.muli") == 1 and names.count("arith.addi") == 1


def analyze(op) -> NormalForm | str:
    """The op's GEMM normal form, or the reason it has none. Never mutates."""
    from ..kernels.shapes import _shaped

    name = getattr(op, "name", "")
    if name not in ("linalg.generic", *_NAMED_DEFAULT_MAPS):
        return f"{name} is not a contraction op"
    operands = list(op.operands)
    if len(operands) != 3 or len(getattr(op, "results", ())) != 1:
        return f"expected 2 inputs, 1 init and 1 result, got {len(operands)} operands"
    shaped = [_shaped(v) for v in operands]
    if any(s is None for s in shaped):
        return "operand types are not readable"
    dtypes = tuple(s[1] for s in shaped)
    if dtypes != _INT8_TRIPLE:
        return f"dtypes {dtypes} are not {_INT8_TRIPLE}"
    if not _body_is_int_mac(op):
        return "body is not an integer multiply-accumulate"
    maps, its = _maps(op), _iterators(op)
    if maps is None or its is None or len(maps) != 3:
        return "indexing maps are not readable"
    n_dims = len(its)
    pure = [[_dim_of(e) for e in m] for m in maps]
    if any(d is None for d in pure[2]) or len(set(pure[2])) != len(pure[2]):
        return f"output map {pure[2]} is not a permutation of iteration dims"
    windowed = [i for i in (0, 1) if any(d is None for d in pure[i])]
    if len(windowed) > 1:
        return "both inputs are read through windowed maps"
    for i in (0, 1):
        dims = [d for d in pure[i] if d is not None]
        if len(set(dims)) != len(dims):
            return f"input {i} indexes one dim twice (a diagonal), not a contraction operand"
    # Extents from every PURE occurrence; they must agree, and every dim must have one.
    extents: dict[int, int] = {}
    for (shape, _dt), dims in zip(shaped, pure):
        for axis, d in enumerate(dims):
            if d is None:
                continue
            if extents.setdefault(d, int(shape[axis])) != int(shape[axis]):
                return f"dim d{d} has two extents ({extents[d]} and {shape[axis]})"
    if set(extents) != set(range(n_dims)):
        return f"extents of {sorted(set(range(n_dims)) - set(extents))} are not readable from a projected operand"
    window_dims: tuple[int, ...] = ()
    window_map = None
    operand_dims: list[tuple[int, ...]] = []
    for i in (0, 1):
        if windowed and windowed[0] == i:
            seen: list[int] = []
            for expr in maps[i]:
                for d in sorted(expr.used_dims()):
                    if d not in seen:
                        seen.append(d)
            window_dims, window_map = tuple(seen), maps[i]
            operand_dims.append(window_dims)
        else:
            operand_dims.append(tuple(pure[i]))
    in_a, in_b, in_c = set(operand_dims[0]), set(operand_dims[1]), set(pure[2])
    for d in range(n_dims):
        role = (d in in_a, d in in_b, d in in_c)
        if role == (True, False, False) or role == (False, True, False):
            return f"d{d} is reduced by one operand alone (a sum folded into the contraction)"
        if role == (False, False, True):
            return f"d{d} indexes only the output (a broadcast)"
        if role == (False, False, False):
            return f"d{d} indexes no operand"
        is_red = its[d] == "reduction"
        if role == (True, True, False) and not is_red:
            return f"d{d} is shared by both inputs and absent from the output but iterates in parallel"
        if role != (True, True, False) and is_red:
            return f"d{d} is a reduction iterator but appears in the output"
    out = tuple(pure[2])
    batch = tuple(d for d in out if d in in_a and d in in_b)
    m = tuple(d for d in out if d in in_a and d not in in_b)
    n = tuple(d for d in out if d in in_b and d not in in_a)
    k = tuple(d for d in operand_dims[0] if d in in_b and d not in in_c)
    if not k:
        return "no reduction dim is shared by both inputs"
    if not m or not n:
        side = "first" if not m else "second"
        return (
            f"the {side} input has no free dim (a depthwise / per-channel form): each output reads one of "
            "its elements per reduction step, so the unit's outer product would be one lane wide"
        )
    return NormalForm(
        op=op,
        operand_dims=(operand_dims[0], operand_dims[1]),
        out_dims=out,
        extents=extents,
        batch=batch,
        m=m,
        n=n,
        k=k,
        windowed=(windowed[0] if windowed else None),
        window_map=window_map,
        window_dims=window_dims,
    )


def site_of(op) -> str:
    """Where in a model a contraction sits -- ``conv`` / ``attention`` / ``linear`` -- read from provenance.

    REPORT ONLY: this labels a routing report so a reader can see which layers the unit takes. It never
    selects anything; when provenance is absent the answer is ``unknown``.
    """
    from xdsl.dialects.builtin import StringAttr

    attrs = getattr(op, "attributes", {}) or {}

    def text(key: str) -> str:
        v = attrs.get(key)
        return v.data if isinstance(v, StringAttr) else ""

    aten, pop = text("prov.aten"), text("prov.op")
    if "prov.conv_path" in attrs or aten.startswith("aten.convolution") or pop.startswith("conv"):
        return "conv"
    if any(t in aten for t in ("bmm", "baddbmm", "scaled_dot_product", "einsum")) or pop in ("batch_matmul", "bmm"):
        return "attention"
    if any(t in aten for t in ("aten.mm", "addmm", "linear", "aten.matmul")) or pop in ("matmul", "addmm", "linear"):
        return "linear"
    if aten.startswith("aten._fft") or pop.startswith("fft"):
        return "spectral"
    return "unknown"


# ---------------------------------------------------------------------------------------------- emission


def _i64_array(values):
    from xdsl.dialects.builtin import DenseArrayBase, i64

    return DenseArrayBase.from_list(i64, [int(v) for v in values])


def _reassociation(groups: list[list[int]]):
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64

    return ArrayAttr([ArrayAttr([IntegerAttr(int(i), i64) for i in g]) for g in groups])


def _groups(sizes: list[int]) -> list[list[int]]:
    out, at = [], 0
    for s in sizes:
        out.append(list(range(at, at + s)))
        at += s
    return out


def emit(nf: NormalForm, orientation: str, call_factory) -> list[Any]:
    """Rewrite ``nf.op`` through ``call_factory(a, b, c_init, result_type) -> call op``; return the new ops.

    The op's result is replaced and the op erased. ``call_factory`` receives kernel-layout operands, so
    the callee's signature is the kernel's (``(b?, m, k) x (b?, k, n) -> (b?, m, n)``) whatever the op's.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    op = nf.op
    i8, i32 = IntegerType(8), IntegerType(32)
    ka, kb, kc, ia, ib = nf.kernel_layout(orientation)
    groups_a = [nf.batch, *((nf.m, nf.k) if orientation == AS_IS else (nf.n, nf.k))]
    groups_b = [nf.batch, *((nf.k, nf.n) if orientation == AS_IS else (nf.k, nf.m))]
    groups_c = [nf.batch, *((nf.m, nf.n) if orientation == AS_IS else (nf.n, nf.m))]
    new: list[Any] = []

    def shape_of(dims) -> list[int]:
        return [int(nf.extents[d]) for d in dims]

    def to_kernel(src_index: int, want: tuple[int, ...], groups) -> Any:
        value = op.operands[src_index]
        have = nf.operand_dims[src_index]
        if nf.windowed == src_index:
            # The im2col gather, straight into the kernel order: all-parallel, reading the operand
            # through the op's own (windowed) map with each dim renumbered to its slot in `want`.
            slot = {d: AffineDimExpr(i) for i, d in enumerate(want)}
            exprs = tuple(
                e.replace_dims_and_symbols([slot.get(j, AffineDimExpr(0)) for j in range(len(nf.extents))], [])
                for e in nf.window_map
            )
            out_t = TensorType(i8, shape_of(want))
            empty = tensor.EmptyOp((), out_t)
            body = Block(arg_types=[i8, i8])
            body.add_op(L.YieldOp(body.args[0]))
            gather = L.GenericOp(
                inputs=(value,),
                outputs=(empty.results[0],),
                body=Region(body),
                indexing_maps=ArrayAttr(
                    [AffineMapAttr(AffineMap(len(want), 0, exprs)), AffineMapAttr(AffineMap.identity(len(want)))]
                ),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(want)),
                result_types=(out_t,),
            )
            new.extend([empty, gather])
            value = gather.results[0]
        elif tuple(have) != tuple(want):
            perm = [have.index(d) for d in want]
            out_t = TensorType(i8, shape_of(want))
            empty = tensor.EmptyOp((), out_t)
            tr = L.TransposeOp(value, empty.results[0], _i64_array(perm), out_t)
            new.extend([empty, tr])
            value = tr.results[0]
        return collapse(value, groups, i8)

    def collapse(value, groups, elem) -> Any:
        sizes = [len(g) for g in groups if g]
        if all(s == 1 for s in sizes):
            return value
        out_t = TensorType(elem, [nf.size(tuple(g)) for g in groups if g])
        col = tensor.CollapseShapeOp(
            operands=[value], result_types=[out_t], properties={"reassociation": _reassociation(_groups(sizes))}
        )
        new.append(col)
        return col.results[0]

    a = to_kernel(ia, tuple(ka), groups_a)
    b = to_kernel(ib, tuple(kb), groups_b)
    c_t = TensorType(i32, [nf.size(tuple(g)) for g in groups_c if g])
    c_empty = tensor.EmptyOp((), c_t)
    zero = arith.ConstantOp.from_int_and_width(0, 32)
    c_init = L.FillOp(inputs=[zero.results[0]], outputs=[c_empty.results[0]], res=[c_t])
    new.extend([c_empty, zero, c_init])
    call = call_factory(a, b, c_init.results[0], c_t)
    new.append(call)
    value = call.results[0]
    sizes = [len(g) for g in groups_c if g]
    if not all(s == 1 for s in sizes):
        full_t = TensorType(i32, shape_of(kc))
        exp = tensor.ExpandShapeOp(value, (), _reassociation(_groups(sizes)), shape_of(kc), full_t)
        new.append(exp)
        value = exp.results[0]
    if tuple(kc) != tuple(nf.out_dims):
        perm = [list(kc).index(d) for d in nf.out_dims]
        out_t = TensorType(i32, shape_of(nf.out_dims))
        empty = tensor.EmptyOp((), out_t)
        tr = L.TransposeOp(value, empty.results[0], _i64_array(perm), out_t)
        new.extend([empty, tr])
        value = tr.results[0]
    parent = op.parent
    for o in new:
        parent.insert_op_before(o, op)
    op.results[0].replace_all_uses_with(value)
    op.detach()
    op.erase()
    return new
