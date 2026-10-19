"""Emit the im2col column matrix ALREADY PANEL-PACKED, so the B operand walks contiguously.

WHAT IS WRONG WITH THE CODE WE EMIT TODAY
-----------------------------------------
model2MLIR rewrites every convolution into ``im2col gather + matmul`` (``prov.conv_path =
"im2col_matmul"``), and it lays the column matrix out as ``[K][M]``: K is the contraction extent,
M = N*Oh*Ow is the parallel extent the schedule vectorizes. The per-op schedule then tiles
``[MR, NR, 0]`` and tiles K by 1, so **K is the innermost loop and the B-operand load walks with row
pitch M**.

MEASURED off the LINKED ELF (``llvm-objdump -d`` of ``merlin_k1``, deepjscc int8, the shipping
feature set), five independent K-loops, each of the form::

    vsetvli zero, zero, e16, m1
    vle8.v  v16, (a4)                  # 16 bytes of B
    lb a3, ...; lb a5, ...; ...        # the MR scalar A loads
    vwmacc.vx v8,  a3, v17
    ...
    addi    a4, a4, 0x100              # <-- B pointer += M bytes
    addi    a0, a0, 0x1
    bne     a4, a2, <top>

with the increments ``0x1000, 0x400, 0x100, 0x100, 0x100`` -- exactly M bytes for M =
4096, 1024, 256, 256, 256. Every K step therefore pulls a fresh 64-byte cache line and consumes
NR = 16 bytes of it: **4.0x cache-line amplification** on the operand the loop streams. This is the
cost XNNPACK's packed-panel GEMM does not pay.

WHAT THIS DOES
--------------
Nothing about the arithmetic changes and no extra pass over memory is added -- the SAME gather writes
the SAME number of elements, in a different order::

    col[k][m]                 ->    col_p[mo][k][mi]        m = mo*NR + mi

so the innermost (K) loop advances the B pointer by NR ELEMENTS, contiguously, and the NR vector
lanes are the NR consecutive ``mi`` that were already the vectorized axis. The gather's own write is
still a linear walk of its output.

Concretely, for a 2-D conv with output ``(N, Oh, Ow)`` and ``Ow = OWO * NR`` the 6-D gather

    col6[c, kh, kw, n, oh, ow]  <- in[n, c, oh*sh + kh*dh, ow*sw + kw*dw]
    collapse-all -> expand [K, M]

is re-emitted as a 7-D gather whose iteration space splits ``ow`` into ``(owo, owi)`` and reorders so
the panel index leads::

    col7[n, oh, owo, c, kh, kw, owi] <- in[n, c, oh*sh + kh*dh, (owo*NR + owi)*sw + kw*dw]
    collapse [[0,1,2],[3,4,5],[6]] -> [MO, K, NR]        MO = N*Oh*OWO,  K = C*kh*kw

and the contraction becomes a 4-dim ``linalg.generic`` (f, mo, mi, k) with

    A   (f, mo, mi, k) -> (f, k)
    Bp  (f, mo, mi, k) -> (mo, k, mi)
    C   (f, mo, mi, k) -> (f, mo, mi)

whose result collapses ``[[0],[1,2]]`` back to ``[F, M]``. Both reshapes are pure row-major
collapses -- views, not copies -- and the accumulator is the SAME ``linalg.fill`` result seen through
an ``expand_shape``, so the zero-initialization is not re-derived (a contraction reading an unfilled
accumulator is a defect this repo has already shipped once).

The matching schedule arm tiles ``[MR, 1, 0, 0]`` then ``[0, 0, 0, 1]`` and vectorizes
``[MR, 1, NR, 1]``: the same MR x NR register block over the same K-by-1 reduction as the matmul arm,
just with the panel index as its own loop. See ``perop_blocks.schedule_text``.

When ``Ow`` is not divisible by ``NR``, complete panels keep that same path.  A second direct gather
and ordinary matmul cover the row remainder at its true width, and both loops insert directly into
the original ``[F, M]`` accumulator.  There is no padded activation, padded whole-result copy, or
out-of-bounds masked load.  The only new index cost is quotient/remainder once per complete panel;
it is outside the gather element loop and contraction K loop.  This is deliberately a narrow tail
rather than padded compute: ResNet-50's 23 formerly refused sites have tails 8 or 12 at NR=16, so
padding them would execute 2x or 1.33x as many tail MACs for values that are immediately discarded.

WHERE NR COMES FROM
-------------------
NR is NOT chosen here and is never a literal. The caller passes the per-op block table
``perop_blocks.block_table`` already derived for this model -- N tile widened per contraction for the
board's VLEN and that contraction's own narrowest element width (``nr_cap_for_dtypes``:
``vlen // narrowest_elem_bits``) and clipped by the lowering predicate. This pass reads the NR that
table assigned to THIS contraction's geometry and packs to it. A geometry the table did not price has
no NR, so it is refused, not guessed.

WHAT IT REFUSES TO DO (fail closed, and every refusal is counted)
-----------------------------------------------------------------
* a contraction whose B operand is not a single-use ``expand_shape <- collapse_shape <- 6-D
  all-parallel copy generic`` chain (``refused_not_im2col``);
* any operand of that chain with more than one use -- rewriting it would leave the old gather alive
  and pay for both (``refused_shared_value``);
* a gather whose input map is not ``(n, c, sh*oh + dh*kh, sw*ow + dw*kw)`` with non-negative integer
  coefficients, read STRUCTURALLY off the affine expressions (``refused_unreadable_gather_map``);
* a geometry with no entry in the block table (``refused_unpriced``);
* a chosen panel width below two, which cannot improve contiguity
  (``refused_panel_width_too_narrow``);
* an accumulator that is not a ``linalg.fill`` result (``refused_accumulator_not_filled``);
* the grouped (rank-7) im2col form (``refused_not_im2col`` -- its contraction is 4-dim already).

Structure-keyed throughout: no op name beyond the linalg/tensor ops it rewrites, no model name, no
target name, no shape constant, no ``prov.*`` read as a classifier. Default OFF -- with the feature
absent nothing here runs, the block table, the tags and the schedule are byte-identical, and so is
the emitted ``.ll``.

NOT MEASURED ON HARDWARE. The layout change is verified on the emitted code (the B-load pointer
advance in the linked ELF's K loops) and the numerics are verified by output digest; no wall-clock
claim is made here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The default-OFF request that turns this on. A REQUEST consumed by
#: ``runtime.backends.zephyr_model.prepare_for_lowering``, exactly like ``perop_blocks``'
#: ``CONV_ARM_FEATURE``: it edits no pipeline and no schedule by itself.
FEATURE = "im2col_panel_pack"

#: MERLIN class name for the packed contraction, in the same namespace as ``perop_blocks.CONV_CLASS``
#: -- spelled like an op name so it flows through ``shape_key`` / ``distinct_blocks`` / ``coverage``
#: with the other contraction classes. There is no MLIR op with this name; the packed contraction is a
#: 4-dim ``linalg.generic``.
PACKED_CLASS = "linalg.matmul_packed_b"

#: Set on the rewritten contraction and its gather, carrying the panel width. Diagnostic: it makes a
#: packed op identifiable in a dumped module without re-deriving the map signature.
PANEL_ATTR = "merlin.b_panel"


@dataclass
class PackReport:
    """What the pass did, and what it refused. Every refusal is named and counted."""

    packed: int = 0
    #: ``[(shape_key, MR, NR)]`` for the ordinary contractions produced. A row-tail rewrite produces
    #: one full-panel entry and one narrow-tail entry.
    entries: list[tuple[str, int, int]] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "packed": self.packed,
            "entries": [[k, mr, nr] for k, mr, nr in self.entries],
            "refusals": dict(sorted(self.refusals.items())),
        }


# --------------------------------------------------------------------------------------------------
# affine-expression reading. Structural: no text, no regex, no assumed spelling.
# --------------------------------------------------------------------------------------------------


def _dim_pos(expr) -> int | None:
    from xdsl.ir.affine import AffineDimExpr

    return int(expr.position) if isinstance(expr, AffineDimExpr) else None


def _affine_term(expr) -> "tuple[int, int] | None":
    """``(coefficient, dim)`` for ``d`` or ``d * c`` / ``c * d`` with a positive integer ``c``."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr

    p = _dim_pos(expr)
    if p is not None:
        return 1, p
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind == AffineBinaryOpKind.Mul:
        for a, b in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            p = _dim_pos(a)
            if p is not None and isinstance(b, AffineConstantExpr) and int(b.value) > 0:
                return int(b.value), p
    return None


def _window_term(expr, stride_dim: int, kernel_dim: int) -> "tuple[int, int] | None":
    """``(stride, dilation)`` for ``s*d<stride_dim> + t*d<kernel_dim>``, else None.

    This is the whole geometry read: a conv window is exactly this expression, and reading the two
    coefficients off it is what keeps the pass from assuming a stride or a dilation.
    """
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind

    if not isinstance(expr, AffineBinaryOpExpr) or expr.kind != AffineBinaryOpKind.Add:
        return None
    lhs, rhs = _affine_term(expr.lhs), _affine_term(expr.rhs)
    if lhs is None or rhs is None:
        return None
    by_dim = {lhs[1]: lhs[0], rhs[1]: rhs[0]}
    if set(by_dim) != {stride_dim, kernel_dim}:
        return None
    return by_dim[stride_dim], by_dim[kernel_dim]


def _static_shape(value) -> "list[int] | None":
    from xdsl.dialects.builtin import TensorType

    t = value.type
    if not isinstance(t, TensorType):
        return None
    shape = [int(d) for d in t.get_shape()]
    return None if any(d < 0 for d in shape) else shape


def _single_use(value) -> bool:
    return len(list(value.uses)) == 1


def _is_copy_body(op) -> bool:
    """The gather's body is a pure element copy: one ``linalg.yield`` of the input block argument."""
    from xdsl.dialects.linalg.ops import YieldOp

    block = op.body.block
    ops = list(block.ops)
    if len(ops) != 1 or not isinstance(ops[0], YieldOp):
        return False
    return len(ops[0].operands) == 1 and ops[0].operands[0] is block.args[0]


def _identity(map_, rank: int) -> bool:
    return (
        map_.num_dims == rank
        and len(map_.results) == rank
        and all(_dim_pos(r) == i for i, r in enumerate(map_.results))
    )


def _reassoc(groups: "list[list[int]]"):
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64

    return ArrayAttr([ArrayAttr([IntegerAttr(j, i64) for j in g]) for g in groups])


# --------------------------------------------------------------------------------------------------
# the match
# --------------------------------------------------------------------------------------------------


@dataclass
class _Match:
    contraction: Any
    gather: Any
    collapse: Any
    expand: Any
    n: int
    oh: int
    ow: int
    channels: int
    kh: int
    kw: int
    sh: int
    sw: int
    dh: int
    dw: int
    f: int
    k: int
    m: int


def _match_im2col(op, report: PackReport) -> "_Match | None":
    """The ``gather -> collapse -> expand -> contraction`` chain, or None with a counted refusal."""
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp

    maps = [a.data for a in op.indexing_maps]
    if len(maps) != 3:
        return None
    # (d0, d2) x (d2, d1) -> (d0, d1): the plain M x K x N contraction m2m emits for an im2col matmul.
    want = [[(0,), (2,)], [(2,), (1,)], [(0,), (1,)]]
    got = [[(_dim_pos(r),) for r in mp.results] for mp in maps]
    if got != want or any(mp.num_dims != 3 for mp in maps):
        return None
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return None

    b = op.inputs[1]
    expand = b.owner
    if not isinstance(expand, ExpandShapeOp):
        report.refuse("refused_not_im2col")
        return None
    if not _single_use(b):
        report.refuse("refused_shared_value")
        return None
    flat = expand.operands[0]
    collapse = flat.owner
    if not isinstance(collapse, CollapseShapeOp):
        report.refuse("refused_not_im2col")
        return None
    if not _single_use(flat):
        report.refuse("refused_shared_value")
        return None
    col6 = collapse.operands[0]
    gather = col6.owner
    if not isinstance(gather, GenericOp):
        report.refuse("refused_not_im2col")
        return None
    if not _single_use(col6):
        report.refuse("refused_shared_value")
        return None

    gshape = _static_shape(col6)
    if gshape is None or len(gshape) != 6:
        report.refuse("refused_not_im2col")  # grouped im2col is rank 7 and lands here
        return None
    if len(gather.inputs) != 1 or len(gather.outputs) != 1:
        report.refuse("refused_not_im2col")
        return None
    if [str(i) for i in gather.get_iterator_types()].count("reduction"):
        report.refuse("refused_not_im2col")
        return None
    if not _is_copy_body(gather):
        report.refuse("refused_not_im2col")
        return None
    gmaps = [a.data for a in gather.indexing_maps]
    if len(gmaps) != 2 or not _identity(gmaps[1], 6) or gmaps[0].num_dims != 6:
        report.refuse("refused_unreadable_gather_map")
        return None

    # dims (c=0, kh=1, kw=2, n=3, oh=4, ow=5); input map -> (n, c, sh*oh + dh*kh, sw*ow + dw*kw)
    res = gmaps[0].results
    if len(res) != 4 or _dim_pos(res[0]) != 3 or _dim_pos(res[1]) != 0:
        report.refuse("refused_unreadable_gather_map")
        return None
    hterm, wterm = _window_term(res[2], 4, 1), _window_term(res[3], 5, 2)
    if hterm is None or wterm is None:
        report.refuse("refused_unreadable_gather_map")
        return None

    ashape, cshape = _static_shape(op.inputs[0]), _static_shape(op.outputs[0])
    inshape = _static_shape(gather.inputs[0])
    bshape = _static_shape(b)
    if None in (ashape, cshape, inshape, bshape):
        report.refuse("refused_not_im2col")
        return None
    if len(ashape) != 2 or len(cshape) != 2 or len(inshape) != 4 or len(bshape) != 2:
        report.refuse("refused_not_im2col")
        return None
    channels, kh, kw, n, oh, ow = gshape
    k, m = bshape
    if channels * kh * kw != k or n * oh * ow != m:
        report.refuse("refused_not_im2col")
        return None
    if inshape[0] != n or inshape[1] != channels:
        report.refuse("refused_not_im2col")
        return None
    if ashape[1] != k or cshape != [ashape[0], m]:
        report.refuse("refused_not_im2col")
        return None
    return _Match(
        contraction=op,
        gather=gather,
        collapse=collapse,
        expand=expand,
        n=n,
        oh=oh,
        ow=ow,
        channels=channels,
        kh=kh,
        kw=kw,
        sh=hterm[0],
        dh=hterm[1],
        sw=wterm[0],
        dw=wterm[1],
        f=ashape[0],
        k=k,
        m=m,
    )


# --------------------------------------------------------------------------------------------------
# the rewrite
# --------------------------------------------------------------------------------------------------


def _matmul_maps():
    """The ordinary ``(m, n, k)`` contraction maps -- the same triple m2m emits for an im2col matmul.

    The panel-loop body is a PLAIN contraction of exactly this shape, which is the point: it is priced,
    tagged and scheduled by the per-op block machinery that already exists, with no packed-specific
    schedule arm, tagger key or op class. A 4-dim ``(f, mo, mi, k)`` generic was tried first and does
    NOT lower -- vectorizing it yields a ``vector.contract`` with two N-like parallel dims (``mo``
    appears in B and C but not in A, so it is neither a batch dim nor an N dim), which no contraction
    lowering strategy handles; it reached LLVM translation as a live
    ``builtin.unrealized_conversion_cast`` and failed the build.
    """
    from xdsl.ir.affine import AffineExpr, AffineMap

    d = AffineExpr.dimension
    return [AffineMap(3, 0, (d(0), d(2))), AffineMap(3, 0, (d(2), d(1))), AffineMap(3, 0, (d(0), d(1)))]


def _dyn_slice_props(rank: int, dyn_dim: int, sizes: "list[int]"):
    """``static_offsets/sizes/strides`` for a slice with ONE dynamic offset at ``dyn_dim``."""
    from xdsl.dialects.builtin import DenseArrayBase, i64
    from xdsl.dialects.tensor import ExpandShapeOp

    offsets = [0] * rank
    # MLIR's ShapedType::kDynamic sentinel, taken from the xDSL op that names it rather than written
    # out as a literal here (one spelling of the constant, in the library that has to agree with MLIR).
    offsets[dyn_dim] = ExpandShapeOp.DYNAMIC_INDEX
    return {
        "static_offsets": DenseArrayBase.from_list(i64, offsets),
        "static_sizes": DenseArrayBase.from_list(i64, sizes),
        "static_strides": DenseArrayBase.from_list(i64, [1] * rank),
    }


def _rewrite_one(mt: _Match, nr: int, *, parallel_panels: bool = False) -> None:
    """Replace the matched chain in place with the panel-packed one."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.scf import ForOp
    from xdsl.dialects.scf import YieldOp as ScfYieldOp
    from xdsl.dialects.tensor import CollapseShapeOp, EmptyOp, ExpandShapeOp, ExtractSliceOp, InsertSliceOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap
    from xdsl.rewriter import InsertPoint, Rewriter

    par = IteratorTypeAttr(IteratorType.PARALLEL)
    red = IteratorTypeAttr(IteratorType.REDUCTION)
    d = AffineExpr.dimension
    idx = IndexType()
    owo = mt.ow // nr
    mo = mt.n * mt.oh * owo
    col_elem = mt.contraction.inputs[1].type.get_element_type()
    acc_elem = mt.contraction.outputs[0].type.get_element_type()

    # --- the packed gather. dims: n=0, oh=1, owo=2, c=3, kh=4, kw=5, owi=6 -----------------------
    packed_col_t = TensorType(col_elem, [mt.n, mt.oh, owo, mt.channels, mt.kh, mt.kw, nr])
    g_empty = EmptyOp([], packed_col_t)
    gblk = Block(arg_types=[col_elem, col_elem])
    gblk.add_op(YieldOp(gblk.args[0]))
    in_map = AffineMap(
        7,
        0,
        (
            d(0),  # n
            d(3),  # c
            d(1) * mt.sh + d(4) * mt.dh,  # oh*sh + kh*dh
            d(2) * (mt.sw * nr) + d(6) * mt.sw + d(5) * mt.dw,  # (owo*NR + owi)*sw + kw*dw
        ),
    )
    new_gather = GenericOp(
        inputs=[mt.gather.inputs[0]],
        outputs=[g_empty.results[0]],
        body=Region(gblk),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(AffineMap(7, 0, tuple(d(i) for i in range(7))))],
        iterator_types=[par] * 7,
        result_types=[packed_col_t],
    )
    for key, val in mt.gather.attributes.items():
        new_gather.attributes[key] = val
    new_gather.attributes[PANEL_ATTR] = IntegerAttr(nr, i64)

    # [N, Oh, OWO, C, kh, kw, NR] -> [MO, K, NR]. A pure row-major collapse: no element moves.
    packed_col = CollapseShapeOp(
        operands=[new_gather.results[0]],
        result_types=[TensorType(col_elem, [mo, mt.k, nr])],
        properties={"reassociation": _reassoc([[0, 1, 2], [3, 4, 5], [6]])},
    )

    Rewriter.insert_op([g_empty, new_gather, packed_col], InsertPoint.before(mt.gather))

    # --- the panel loop ---------------------------------------------------------------------------
    # The accumulator is the EXISTING zero-filled [F, M] tensor seen as [F, MO, NR] -- a view, so the
    # zero init is reused rather than re-derived (a contraction reading an unfilled accumulator is a
    # defect this repo has already shipped once).
    acc_t = TensorType(acc_elem, [mt.f, mo, nr])
    acc = ExpandShapeOp(mt.contraction.outputs[0], [], _reassoc([[0], [1, 2]]), [mt.f, mo, nr], acc_t)
    lb = ConstantOp(IntegerAttr(0, idx), idx)
    ub = ConstantOp(IntegerAttr(mo, idx), idx)
    step = ConstantOp(IntegerAttr(1, idx), idx)

    body = Block(arg_types=[idx, acc_t])
    ivar, carried = body.args
    panel_t = TensorType(col_elem, [mt.k, nr])
    tile_t = TensorType(acc_elem, [mt.f, nr])
    panel = ExtractSliceOp.build(
        operands=[packed_col.results[0], [ivar], [], []],
        result_types=[panel_t],
        properties=_dyn_slice_props(3, 0, [1, mt.k, nr]),
    )
    tile = ExtractSliceOp.build(
        operands=[carried, [ivar], [], []], result_types=[tile_t], properties=_dyn_slice_props(3, 1, [mt.f, 1, nr])
    )
    inner = GenericOp(
        inputs=[mt.contraction.inputs[0], panel.results[0]],
        outputs=[tile.results[0]],
        body=mt.contraction.body.clone(),
        indexing_maps=[AffineMapAttr(mp) for mp in _matmul_maps()],
        iterator_types=[par, par, red],
        result_types=[tile_t],
    )
    for key, val in mt.contraction.attributes.items():
        inner.attributes[key] = val
    inner.attributes[PANEL_ATTR] = IntegerAttr(nr, i64)
    put = InsertSliceOp.build(
        operands=[inner.results[0], carried, [ivar], [], []],
        result_types=[acc_t],
        properties=_dyn_slice_props(3, 1, [mt.f, 1, nr]),
    )
    from .panel_parallel import marker_ops

    markers = marker_ops(parallel_panels, mo)
    body.add_ops([*markers, panel, tile, inner, put, ScfYieldOp(put.results[0])])
    loop = ForOp(lb.results[0], ub.results[0], step.results[0], [acc.results[0]], Region(body))

    out = CollapseShapeOp(
        operands=[loop.results[0]],
        result_types=[TensorType(acc_elem, [mt.f, mt.m])],
        properties={"reassociation": _reassoc([[0], [1, 2]])},
    )
    Rewriter.insert_op([acc, lb, ub, step, loop, out], InsertPoint.before(mt.contraction))

    mt.contraction.results[0].replace_all_uses_with(out.results[0])
    for dead in (mt.contraction, mt.expand, mt.collapse, mt.gather):
        Rewriter.erase_op(dead)


def _spatial_panel_gather(mt: _Match, panel_count: int, width: int, first_ow: int):
    """Build a direct gather for ``panel_count`` panels of ``width`` columns in every output row.

    ``first_ow`` is the first output column covered.  Keeping the output-row dimensions explicit is
    important: it avoids a floordiv/mod pair for every im2col element.  Tail support pays index
    arithmetic once per contraction panel instead (in :func:`_spatial_panel_loop`).
    """
    from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import CollapseShapeOp, EmptyOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    par = IteratorTypeAttr(IteratorType.PARALLEL)
    d = AffineExpr.dimension
    col_elem = mt.contraction.inputs[1].type.get_element_type()
    # dims: n=0, oh=1, panel=2, c=3, kh=4, kw=5, wi=6
    shape = [mt.n, mt.oh, panel_count, mt.channels, mt.kh, mt.kw, width]
    gather_t = TensorType(col_elem, shape)
    empty = EmptyOp([], gather_t)
    block = Block(arg_types=[col_elem, col_elem])
    block.add_op(YieldOp(block.args[0]))
    in_map = AffineMap(
        7,
        0,
        (
            d(0),
            d(3),
            d(1) * mt.sh + d(4) * mt.dh,
            (first_ow + d(2) * width + d(6)) * mt.sw + d(5) * mt.dw,
        ),
    )
    gather = GenericOp(
        inputs=[mt.gather.inputs[0]],
        outputs=[empty.results[0]],
        body=Region(block),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(AffineMap(7, 0, tuple(d(i) for i in range(7))))],
        iterator_types=[par] * 7,
        result_types=[gather_t],
    )
    for key, val in mt.gather.attributes.items():
        gather.attributes[key] = val
    gather.attributes[PANEL_ATTR] = IntegerAttr(width, i64)
    panels = mt.n * mt.oh * panel_count
    collapsed = CollapseShapeOp(
        operands=[gather.results[0]],
        result_types=[TensorType(col_elem, [panels, mt.k, width])],
        properties={"reassociation": _reassoc([[0, 1, 2], [3, 4, 5], [6]])},
    )
    return [empty, gather, collapsed], collapsed.results[0]


def _spatial_panel_loop(
    mt: _Match,
    packed,
    initial_acc,
    *,
    panels: int,
    panels_per_row: int,
    width: int,
    first_ow: int,
    parallel_panels: bool,
):
    """Contract packed panels into their unpadded positions in the original ``[F, M]`` result.

    The quotient/remainder is evaluated once per panel, never in the K loop or the gather element
    loop.  Consequently a non-divisible output width retains the full-width kernel for its complete
    panels and uses one narrower kernel only for each row's remainder.
    """
    from xdsl.dialects.arith import AddiOp, ConstantOp, DivUIOp, MuliOp, RemUIOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.dialects.scf import ForOp
    from xdsl.dialects.scf import YieldOp as ScfYieldOp
    from xdsl.dialects.tensor import ExtractSliceOp, InsertSliceOp
    from xdsl.ir import Block, Region

    par = IteratorTypeAttr(IteratorType.PARALLEL)
    red = IteratorTypeAttr(IteratorType.REDUCTION)
    idx = IndexType()
    col_elem = mt.contraction.inputs[1].type.get_element_type()
    acc_elem = mt.contraction.outputs[0].type.get_element_type()
    acc_t = TensorType(acc_elem, [mt.f, mt.m])
    lb = ConstantOp.from_int_and_width(0, idx)
    ub = ConstantOp.from_int_and_width(panels, idx)
    step = ConstantOp.from_int_and_width(1, idx)

    body = Block(arg_types=[idx, acc_t])
    ivar, carried = body.args
    offset_ops = []
    if panels_per_row == 1:
        row = ivar
        panel_col = None
    else:
        per_row = ConstantOp.from_int_and_width(panels_per_row, idx)
        row_op = DivUIOp(ivar, per_row)
        panel_op = RemUIOp(ivar, per_row)
        offset_ops.extend([per_row, row_op, panel_op])
        row, panel_col = row_op.results[0], panel_op.results[0]
    ow = ConstantOp.from_int_and_width(mt.ow, idx)
    row_base = MuliOp(row, ow)
    offset_ops.extend([ow, row_base])
    offset = row_base.results[0]
    if panel_col is not None:
        panel_width = ConstantOp.from_int_and_width(width, idx)
        panel_base = MuliOp(panel_col, panel_width)
        add_panel = AddiOp(offset, panel_base)
        offset_ops.extend([panel_width, panel_base, add_panel])
        offset = add_panel.results[0]
    if first_ow:
        first = ConstantOp.from_int_and_width(first_ow, idx)
        add_first = AddiOp(offset, first)
        offset_ops.extend([first, add_first])
        offset = add_first.results[0]

    panel_t = TensorType(col_elem, [mt.k, width])
    tile_t = TensorType(acc_elem, [mt.f, width])
    panel = ExtractSliceOp.build(
        operands=[packed, [ivar], [], []], result_types=[panel_t], properties=_dyn_slice_props(3, 0, [1, mt.k, width])
    )
    tile = ExtractSliceOp.build(
        operands=[carried, [offset], [], []], result_types=[tile_t], properties=_dyn_slice_props(2, 1, [mt.f, width])
    )
    inner = GenericOp(
        inputs=[mt.contraction.inputs[0], panel.results[0]],
        outputs=[tile.results[0]],
        body=mt.contraction.body.clone(),
        indexing_maps=[AffineMapAttr(mp) for mp in _matmul_maps()],
        iterator_types=[par, par, red],
        result_types=[tile_t],
    )
    for key, val in mt.contraction.attributes.items():
        inner.attributes[key] = val
    inner.attributes[PANEL_ATTR] = IntegerAttr(width, i64)
    put = InsertSliceOp.build(
        operands=[inner.results[0], carried, [offset], [], []],
        result_types=[acc_t],
        properties=_dyn_slice_props(2, 1, [mt.f, width]),
    )
    from .panel_parallel import marker_ops

    markers = marker_ops(parallel_panels, panels)
    body.add_ops([*offset_ops, *markers, panel, tile, inner, put, ScfYieldOp(put.results[0])])
    loop = ForOp(lb.results[0], ub.results[0], step.results[0], [initial_acc], Region(body))
    return [lb, ub, step, loop], loop.results[0]


def _rewrite_one_with_tail(mt: _Match, nr: int, *, parallel_panels: bool = False) -> int:
    """Pack full-width panels and a row tail without padding or copying the whole activation/result.

    Returns the tail width, which is also the N extent of the second ordinary contraction geometry.
    """
    from xdsl.rewriter import InsertPoint, Rewriter

    full = mt.ow // nr
    tail = mt.ow % nr
    assert tail and nr >= 2

    tail_ops, tail_value = _spatial_panel_gather(mt, 1, tail, full * nr)
    full_value = None
    gather_ops = []
    if full:
        full_ops, full_value = _spatial_panel_gather(mt, full, nr, 0)
        gather_ops.extend(full_ops)
    gather_ops.extend(tail_ops)
    Rewriter.insert_op(gather_ops, InsertPoint.before(mt.gather))

    acc = mt.contraction.outputs[0]
    if full:
        assert full_value is not None
        loop_ops, acc = _spatial_panel_loop(
            mt,
            full_value,
            acc,
            panels=mt.n * mt.oh * full,
            panels_per_row=full,
            width=nr,
            first_ow=0,
            parallel_panels=parallel_panels,
        )
        Rewriter.insert_op(loop_ops, InsertPoint.before(mt.contraction))
    tail_loop_ops, acc = _spatial_panel_loop(
        mt,
        tail_value,
        acc,
        panels=mt.n * mt.oh,
        panels_per_row=1,
        width=tail,
        first_ow=full * nr,
        parallel_panels=parallel_panels,
    )
    Rewriter.insert_op(tail_loop_ops, InsertPoint.before(mt.contraction))

    mt.contraction.results[0].replace_all_uses_with(acc)
    for dead in (mt.contraction, mt.expand, mt.collapse, mt.gather):
        Rewriter.erase_op(dead)
    return tail


def rewrite_module(module, table: "dict[str, tuple[int, int]]", *, parallel_panels: bool = False) -> PackReport:
    """Pack every eligible im2col contraction in ``module`` (mutated in place).

    ``table`` is the per-op block table already derived for THIS model
    (``perop_blocks.block_table``); the NR it assigned to a contraction's geometry is the panel width
    used for that contraction. A geometry the table did not price is refused.
    """
    from ..common import mlir_query as mq
    from .perop_blocks import shape_key

    report = PackReport()
    candidates: list[tuple[Any, _Match]] = []
    for op in mq.walk(module, "linalg.generic"):
        mt = _match_im2col(op, report)
        if mt is not None:
            candidates.append((op, mt))
    for _, mt in candidates:
        key = shape_key("linalg.matmul", (mt.f, mt.m), (mt.k,))
        blk = table.get(key)
        if blk is None:
            report.refuse("refused_unpriced")
            continue
        mr, nr = int(blk[0]), int(blk[1])
        if nr < 2:
            report.refuse("refused_panel_width_too_narrow")
            continue
        if not _accumulator_is_filled(mt.contraction):
            report.refuse("refused_accumulator_not_filled")
            continue
        tail = mt.ow % nr
        if tail:
            _rewrite_one_with_tail(mt, nr, parallel_panels=parallel_panels)
        else:
            _rewrite_one(mt, nr, parallel_panels=parallel_panels)
        report.packed += 1
        # The geometry the panel loop leaves behind: an ORDINARY [F, NR] x K contraction, which the
        # caller's SECOND block_table pass observes, prices and tags with no packed-specific
        # machinery. Recorded so a caller can check that the block it gets is the one that was packed
        # for -- a panel packed at NR whose contraction is then tiled at a narrower N would be a
        # silent half-application.
        if not tail or mt.ow // nr:
            report.entries.append((shape_key("linalg.matmul", (mt.f, nr), (mt.k,)), mr, nr))
        if tail:
            # The remainder is a separate ordinary matmul at its true width.  Recording it is what
            # makes the caller's second block-table/tagging pass prove that this arm is schedulable.
            report.entries.append((shape_key("linalg.matmul", (mt.f, tail), (mt.k,)), mr, tail))
    if parallel_panels and report.packed:
        from .panel_parallel import ensure_marker_declaration

        ensure_marker_declaration(module)
    return report


def _accumulator_is_filled(op) -> bool:
    """The contraction's ``outs`` must be a ``linalg.fill`` result.

    A contraction accumulates into its output, so an ``outs`` that is a bare ``tensor.empty`` reads
    undefined memory. Today m2m always emits the fill; requiring it here means the reshaped
    accumulator this pass builds cannot silently become the un-zeroed one.
    """
    from xdsl.dialects.linalg.ops import FillOp

    return isinstance(op.outputs[0].owner, FillOp)


def rewrite_prepared_file(
    prepared: "str | Path",
    table: "dict[str, tuple[int, int]]",
    work: "str | Path | None" = None,
    *,
    parallel_panels: bool = False,
) -> "tuple[Path, PackReport]":
    """Pack ``prepared`` and write ``model.bpacked.mlir``; returns ``(path, report)``.

    Runs in merlin's own interpreter over xDSL -- the same library that BUILT these ops in
    model2MLIR -- and BEFORE ``perop_blocks.tag_prepared_mlir``, which round-trips the module through
    the MLIR printer. Nothing is written when nothing was packed: the caller keeps the original path,
    so a run where every candidate was refused is byte-identical to the baseline rather than
    re-serialized.
    """
    from ..common import mlir_query as mq

    prepared = Path(prepared)
    module = mq.parse(prepared.read_text(encoding="utf-8"))
    report = rewrite_module(module, table, parallel_panels=parallel_panels)
    if not report.packed:
        return prepared, report
    out = Path(work) / "model.bpacked.mlir" if work is not None else prepared.with_name("model.bpacked.mlir")
    out.write_text(str(module), encoding="utf-8")
    return out, report


def ensure_registered() -> str:
    """Register the default-off :data:`FEATURE` request. Idempotent.

    Registered from HERE and called by the preparation step, ``llvmlower.lower`` and
    ``pipeline.lower_to_llvm_ir``, for the reason ``perop_blocks.ensure_registered`` records: the
    lowering runs in a child process that re-imports the feature registry, and a name registered only
    in the parent fails to resolve in the child. It also has to be registered before
    ``wholemodel_proposer._composes`` is asked about it -- that helper swallows the ``KeyError`` for an
    unregistered name and answers False, so an unregistered lever is never proposed and never
    complains.
    """
    from .impr_features import ImprFeature, known, register

    if FEATURE in known():
        return FEATURE
    # NO `implies={PEROP_BLOCK_NAME}`, deliberately, even though this lever REQUIRES per-op blocking.
    # `perop_register_block` is a sentinel the preparation step CONSUMES and replaces with a concrete
    # per-table feature; `implies` is closed over at `normalize` time, which happens again inside the
    # lowering -- so implying it would re-materialize the consumed sentinel after preparation and trip
    # `_perop_sentinel_unresolved`. The requirement is enforced where it can actually be checked, in
    # `zephyr_model.prepare_for_lowering`, which refuses the combination outright.
    register(
        ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Emit the im2col column matrix panel-packed as [M/NR][K][NR] instead of [K][M], and "
                "contract each panel with an ordinary [F,NR]xK matmul inside a panel loop. Non-divisible "
                "output rows retain full NR panels and use one ordinary narrow matmul for the remainder, "
                "inserting both directly into [F,M] without padding/copying the whole result. Today the "
                "column matrix is [K][M] and K is the "
                "innermost loop, so the B-operand pointer advances by M bytes per K step: measured on the "
                "LINKED ELF (deepjscc int8, five K loops) the advances are exactly M bytes, so each 64-byte "
                "line delivers NR=16 used bytes -- 4.0x cache-line amplification on the streamed operand. "
                "Packing makes that walk contiguous, which is what a packed-panel GEMM does. NR is read "
                "from the per-op block table (VLEN- and dtype-derived), never chosen here. A REQUEST "
                "consumed by runtime.backends.zephyr_model.prepare_for_lowering; with it absent the block "
                "table, the tags, the schedule and the emitted .ll are byte-identical. NOT MEASURED ON "
                "HARDWARE -- emitted-code and output-digest evidence only."
            ),
        )
    )
    return FEATURE


@dataclass(frozen=True)
class BoundedIm2ColSource:
    """Original typed source witness, frozen before the old island is erased."""

    geometry: tuple[int, ...]
    source_ir: str
    input: Any
    weight: Any
    initializer: Any
    input_type: str
    gather_type: str

    @property
    def f(self):
        return self.geometry[10]

    @property
    def k(self):
        return self.geometry[11]

    @property
    def m(self):
        return self.geometry[12]

    @property
    def identity(self):
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(
                (self.geometry, self.source_ir, self.input_type, self.gather_type), separators=(",", ":")
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class BoundedIm2ColArm:
    width: int
    first_ow: int
    panels_per_row: int
    loop: Any
    packing_empty: Any
    patch: Any
    gather: Any
    contraction: Any
    output_empty: Any
    output_fill: Any
    insert: Any


@dataclass(frozen=True)
class BoundedIm2ColRewrite:
    source: BoundedIm2ColSource
    source_identity: str
    width: int
    operations: tuple[Any, ...]
    arms: tuple[BoundedIm2ColArm, ...]
    result: Any


def _bounded_plain(op, properties=()):
    return not op.attributes and set(op.properties) <= set(properties) and not op.successors


def match_bounded_im2col(operation, *, selected_contracts, max_elements=64 << 20):
    """Read a sole-use static integer island without granting device gathers.

    The ordinary dense reader owns the scalar recurrence and exact zero seed.
    This extra admission checks the actual window/reshape maps and motion scope.
    Unsupported source retains its full materialization route.
    """
    from xdsl.dialects import builtin, tensor
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, read_source_dense

    if type(operation) is not GenericOp:
        return None
    try:
        read_source_dense(operation, selected_contracts=selected_contracts, max_source_elements=max_elements)
    except DenseSourceUnavailable:
        return None
    mt = _match_im2col(operation, PackReport())
    if mt is None:
        return None
    values = (
        mt.gather.inputs[0],
        mt.gather.outputs[0],
        mt.gather.results[0],
        mt.collapse.results[0],
        mt.expand.results[0],
    )
    if any(
        not isinstance(v.type, builtin.TensorType)
        or not isinstance(v.type.encoding, builtin.NoneAttr)
        or any(d <= 0 for d in v.type.get_shape())
        for v in values
    ):
        return None
    allowed = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if (
        not _bounded_plain(mt.gather, allowed)
        or len(mt.gather.body.blocks) != 1
        or tuple(t.data for t in mt.gather.get_iterator_types()) != (IteratorType.PARALLEL,) * 6
        or not _bounded_plain(mt.gather.body.block.last_op)
        or not _bounded_plain(mt.collapse, ("reassociation",))
        or not _bounded_plain(mt.expand, ("reassociation", "static_output_shape"))
        or mt.collapse.properties.get("reassociation") != _reassoc([list(range(6))])
        or mt.expand.properties.get("reassociation") != _reassoc([[0, 1]])
        or len(mt.collapse.operands) != 1
        or len(mt.expand.operands) != 1
        or type(mt.gather.outputs[0].owner) is not tensor.EmptyOp
        or not _bounded_plain(mt.gather.outputs[0].owner)
        or mt.gather.outputs[0].owner.operands
        or not _single_use(mt.gather.outputs[0])
    ):
        return None
    shape = tuple(mt.gather.inputs[0].type.get_shape())
    if (mt.oh - 1) * mt.sh + (mt.kh - 1) * mt.dh >= shape[2] or (mt.ow - 1) * mt.sw + (mt.kw - 1) * mt.dw >= shape[3]:
        return None
    # Moving materialization to its consumer may not cross an unknown effect.
    block = operation.parent_block()
    if block is None or any(op.parent_block() is not block for op in (mt.gather, mt.collapse, mt.expand)):
        return None
    cursor = mt.gather
    island = {mt.gather, mt.collapse, mt.expand}
    initializer = operation.outputs[0].owner
    allowed_between = {initializer, initializer.inputs[0].owner, initializer.outputs[0].owner}
    while cursor is not operation:
        if cursor is None or cursor not in island | allowed_between:
            return None
        cursor = cursor.next_op
    mt.bounded_source = _snapshot_bounded_source(mt)
    return mt


def _snapshot_bounded_source(mt):
    from xdsl.dialects import builtin, func
    from xdsl.ir import Block, Region

    from merlin.transforms.certificates import ir_text

    op = mt.contraction
    block = Block(arg_types=[v.type for v in op.inputs])
    values = dict(zip(op.inputs, block.args, strict=True))
    fill = op.outputs[0].owner
    for source in (fill.inputs[0].owner, fill.outputs[0].owner, fill, op):
        block.add_op(source.clone(value_mapper=values))
    block.add_op(func.ReturnOp(values[op.results[0]]))
    fn = func.FuncOp.from_region("source", [v.type for v in op.inputs], [op.results[0].type], Region(block))
    window = Block(arg_types=[mt.gather.inputs[0].type])
    mapping = {mt.gather.inputs[0]: window.args[0]}
    for original in (mt.gather.outputs[0].owner, mt.gather, mt.collapse, mt.expand):
        window.add_op(original.clone(value_mapper=mapping))
    window.add_op(func.ReturnOp(mapping[mt.expand.results[0]]))
    window_fn = func.FuncOp.from_region(
        "window", [mt.gather.inputs[0].type], [mt.expand.results[0].type], Region(window)
    )
    geometry = tuple(
        getattr(mt, name) for name in ("n", "oh", "ow", "channels", "kh", "kw", "sh", "sw", "dh", "dw", "f", "k", "m")
    )
    return BoundedIm2ColSource(
        geometry,
        ir_text(builtin.ModuleOp([fn, window_fn])),
        mt.gather.inputs[0],
        op.inputs[0],
        op.outputs[0],
        str(mt.gather.inputs[0].type),
        str(mt.gather.results[0].type),
    )


def bounded_source_identity(match):
    """Identity for the original analysis stamp, before any source mutation."""
    source = getattr(match, "bounded_source", None)
    if type(source) is not BoundedIm2ColSource:
        raise ValueError("bounded im2col requires the original admitted source witness")
    if (
        match.gather.inputs[0] is not source.input
        or match.contraction.inputs[0] is not source.weight
        or match.contraction.outputs[0] is not source.initializer
        or _snapshot_bounded_source(match).identity != source.identity
    ):
        raise ValueError("bounded im2col source changed after admission")
    return source.identity


def _bounded_slice_props(offsets, sizes):
    from xdsl.dialects.builtin import DenseArrayBase, i64
    from xdsl.dialects.tensor import ExpandShapeOp

    return {
        "static_offsets": DenseArrayBase.from_list(
            i64, [v if type(v) is int else ExpandShapeOp.DYNAMIC_INDEX for v in offsets]
        ),
        "static_sizes": DenseArrayBase.from_list(i64, sizes),
        "static_strides": DenseArrayBase.from_list(i64, [1] * len(sizes)),
    }


def _bounded_arm(mt, source, initial, width, first_ow, panels_per_row):
    from xdsl.dialects import arith, builtin, scf, tensor
    from xdsl.dialects.linalg.ops import FillOp, GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    idx, elem, acc_elem = builtin.IndexType(), mt.gather.results[0].type.element_type, initial.type.element_type
    packing_t = builtin.TensorType(elem, (mt.channels, mt.kh, mt.kw, 1, 1, width))
    packing = tensor.EmptyOp([], packing_t)
    lb = arith.ConstantOp.from_int_and_width(0, idx)
    ub = arith.ConstantOp.from_int_and_width(mt.n * mt.oh * panels_per_row, idx)
    step = arith.ConstantOp.from_int_and_width(1, idx)
    body = Block(arg_types=(idx, initial.type, packing_t))
    iv, carried, destination = body.args
    prefix = []

    def constant(value):
        op = arith.ConstantOp.from_int_and_width(value, idx)
        prefix.append(op)
        return op.results[0]

    def binary(cls, left, right):
        op = cls(left, right)
        prefix.append(op)
        return op.results[0]

    row = binary(arith.DivUIOp, iv, constant(panels_per_row))
    panel = binary(arith.RemUIOp, iv, constant(panels_per_row))
    batch = binary(arith.DivUIOp, row, constant(mt.oh))
    oh = binary(arith.RemUIOp, row, constant(mt.oh))
    ow = binary(arith.AddiOp, binary(arith.MuliOp, panel, constant(width)), constant(first_ow))
    ih = binary(arith.MuliOp, oh, constant(mt.sh))
    iw = binary(arith.MuliOp, ow, constant(mt.sw))
    offset = binary(arith.AddiOp, binary(arith.MuliOp, row, constant(mt.ow)), ow)
    patch_shape = (1, mt.channels, (mt.kh - 1) * mt.dh + 1, (width - 1) * mt.sw + (mt.kw - 1) * mt.dw + 1)
    offsets = (batch, 0, ih, iw)
    patch = tensor.ExtractSliceOp.build(
        operands=[source.input, [v for v in offsets if type(v) is not int], [], []],
        result_types=[builtin.TensorType(elem, patch_shape)],
        properties=_bounded_slice_props(offsets, patch_shape),
    )
    d, par, red = (
        AffineExpr.dimension,
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.REDUCTION),
    )
    block = Block(arg_types=(elem, elem))
    block.add_op(YieldOp(block.args[0]))
    gather = GenericOp(
        inputs=[patch.results[0]],
        outputs=[destination],
        body=Region(block),
        indexing_maps=[
            builtin.AffineMapAttr(AffineMap(6, 0, (d(3), d(0), d(1) * mt.dh, d(5) * mt.sw + d(2) * mt.dw))),
            builtin.AffineMapAttr(AffineMap.identity(6)),
        ],
        iterator_types=[par] * 6,
        result_types=[packing_t],
    )
    panel_t = builtin.TensorType(elem, (mt.k, width))
    matrix = tensor.CollapseShapeOp(
        operands=[gather.results[0]],
        result_types=[panel_t],
        properties={"reassociation": _reassoc([[0, 1, 2], [3, 4, 5]])},
    )
    output_t = builtin.TensorType(acc_elem, (mt.f, width))
    empty = tensor.EmptyOp([], output_t)
    zero = mt.contraction.outputs[0].owner.inputs[0].owner.clone()
    fill = FillOp(inputs=[zero.results[0]], outputs=[empty.results[0]], res=[output_t])
    inner = GenericOp(
        inputs=[source.weight, matrix.results[0]],
        outputs=[fill.results[0]],
        body=mt.contraction.body.clone(),
        indexing_maps=[builtin.AffineMapAttr(mp) for mp in _matmul_maps()],
        iterator_types=[par, par, red],
        result_types=[output_t],
    )
    inner.attributes.update(mt.contraction.attributes)
    insert = tensor.InsertSliceOp.build(
        operands=[inner.results[0], carried, [offset], [], []],
        result_types=[initial.type],
        properties=_bounded_slice_props((0, offset), (mt.f, width)),
    )
    body.add_ops(
        [
            *prefix,
            patch,
            gather,
            matrix,
            empty,
            zero,
            fill,
            inner,
            insert,
            scf.YieldOp(insert.results[0], gather.results[0]),
        ]
    )
    loop = scf.ForOp(lb.results[0], ub.results[0], step.results[0], [initial, packing.results[0]], Region(body))
    return (packing, lb, ub, step, loop), BoundedIm2ColArm(
        width, first_ow, panels_per_row, loop, packing, patch, gather, inner, empty, fill, insert
    )


def rewrite_bounded_match(match, width):
    """Reuse a fixed activation panel in each serial main/tail loop.

    The host owns the explicit sum-of-panel payload budget. This transformation
    grants neither a device gather nor asynchronous operand retention.
    """
    from xdsl.rewriter import InsertPoint, Rewriter

    identity = bounded_source_identity(match)
    if type(width) is not int or not 1 <= width <= match.ow:
        raise ValueError("bounded im2col width must fit a positive output row")
    source = match.bounded_source
    original_packing = match.gather.outputs[0].owner
    operations, arms, result = [], [], source.initializer
    full, tail = divmod(match.ow, width)
    for arm_width, first, panels in ((width, 0, full), (tail, full * width, 1)):
        if not arm_width or not panels:
            continue
        ops, arm = _bounded_arm(match, source, result, arm_width, first, panels)
        Rewriter.insert_op(ops, InsertPoint.before(match.contraction))
        operations.extend(ops)
        arms.append(arm)
        result = arm.loop.results[0]
    match.contraction.results[0].replace_all_uses_with(result)
    for dead in (match.contraction, match.expand, match.collapse, match.gather):
        Rewriter.erase_op(dead)
    if not original_packing.results[0].uses:
        Rewriter.erase_op(original_packing)
    return BoundedIm2ColRewrite(source, identity, width, tuple(operations), tuple(arms), result)


def verify_bounded_match(rewrite, *, expected_source=None):
    """Reconstruct the realized affine windows, complete output coverage and reuse.

    An original pass analysis may supply expected_source to bind the witness to
    its immutable current-source context. No allocation or timing is inferred
    from these source-level tensor destinations.
    """
    from xdsl.dialects import arith, builtin, scf, tensor
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineExpr, AffineMap

    from merlin.common import mlir_query as mq

    def require(claim, reason):
        if not claim:
            raise ValueError("bounded im2col realization: " + reason)

    require(
        type(rewrite) is BoundedIm2ColRewrite and type(rewrite.source) is BoundedIm2ColSource,
        "requires the original typed source witness",
    )
    source = rewrite.source
    require(
        source.identity == rewrite.source_identity and (expected_source is None or expected_source == source.identity),
        "original source witness changed",
    )
    original = mq.parse(source.source_ir)
    original.verify()
    reference = next(op for op in original.walk() if type(op) is GenericOp)
    ref_fill = reference.outputs[0].owner
    n, oh, ow, channels, kh, kw, sh, sw, dh, dw, f, k, m = source.geometry
    require(
        all(type(v) is int and v > 0 for v in source.geometry) and k == channels * kh * kw and m == n * oh * ow,
        "inconsistent original geometry",
    )
    require(
        str(source.input.type) == source.input_type
        and tuple(reference.results[0].type.get_shape()) == (f, m)
        and tuple(reference.inputs[0].type.get_shape()) == (f, k)
        and tuple(reference.inputs[1].type.get_shape()) == (k, m)
        and source.weight.type == reference.inputs[0].type
        and source.initializer.type == reference.results[0].type,
        "original operand types changed",
    )
    elem = reference.inputs[1].type.element_type
    require(type(elem) is builtin.IntegerType and elem.width.data in (8, 16, 32), "unknown packing element format")
    width_bytes = elem.width.data // 8
    input_shape = tuple(source.input.type.get_shape())
    require(
        len(input_shape) == 4
        and input_shape[:2] == (n, channels)
        and (oh - 1) * sh + (kh - 1) * dh < input_shape[2]
        and (ow - 1) * sw + (kw - 1) * dw < input_shape[3]
        and source.gather_type == str(builtin.TensorType(elem, (channels, kh, kw, n, oh, ow))),
        "original window exceeds its typed input or changes element order",
    )
    require(type(rewrite.width) is int and 1 <= rewrite.width <= ow, "invalid selected panel width")
    full, tail = divmod(ow, rewrite.width)
    expected_arms = [(rewrite.width, 0, full)] + ([(tail, full * rewrite.width, 1)] if tail else [])
    require(len(rewrite.arms) == len(expected_arms), "missing or duplicated main/tail output domain")
    initial, top_operations, arm_records = source.initializer, [], []
    for arm, (width, first, per_row) in zip(rewrite.arms, expected_arms, strict=True):
        require(
            type(arm) is BoundedIm2ColArm and (arm.width, arm.first_ow, arm.panels_per_row) == (width, first, per_row),
            "changed main/tail coverage",
        )
        loop, packing = arm.loop, arm.packing_empty
        require(
            type(loop) is scf.ForOp
            and _bounded_plain(loop)
            and type(packing) is tensor.EmptyOp
            and _bounded_plain(packing)
            and not packing.operands
            and loop.parent_block() is packing.parent_block(),
            "panel destination is not outside its serial loop",
        )
        packing_type = builtin.TensorType(elem, (channels, kh, kw, 1, 1, width))
        require(
            len(loop.results) == 2
            and packing.results[0].type == packing_type
            and tuple(loop.iter_args) == (initial, packing.results[0]),
            "panel must be an explicit loop-carried destination",
        )
        bounds = tuple(v.owner for v in (loop.lb, loop.ub, loop.step))
        for op, value in zip(bounds, (0, n * oh * per_row, 1), strict=True):
            require(
                type(op) is arith.ConstantOp
                and _bounded_plain(op, ("value",))
                and isinstance(op.value, builtin.IntegerAttr)
                and op.value.value.data == value
                and type(op.results[0].type) is builtin.IndexType,
                "incorrect serial panel bounds",
            )
        top_operations.extend((packing, *bounds, loop))
        block = loop.body.block
        require(len(loop.body.blocks) == 1 and len(block.args) == 3, "unexpected loop ownership boundary")
        iv, carried, destination = block.args
        require(destination.type == packing_type and carried.type == initial.type, "changed loop-carried formats")
        matrix = arm.contraction.inputs[1].owner
        zero = arm.output_fill.inputs[0].owner
        terminator = block.last_op
        suffix = (
            arm.patch,
            arm.gather,
            matrix,
            arm.output_empty,
            zero,
            arm.output_fill,
            arm.contraction,
            arm.insert,
            terminator,
        )
        body = tuple(block.ops)
        require(body[-len(suffix) :] == suffix, "unexpected operations in panel body")
        prefix = body[: -len(suffix)]
        require(len(prefix) <= 32, "unbounded panel index work")
        visited = set()

        def expression(value):
            if value is iv:
                return ("iv",)
            op = value.owner
            require(
                op in prefix
                and op.parent_block() is block
                and not op.attributes
                and not op.regions
                and not op.successors
                and len(op.results) == 1
                and type(value.type) is builtin.IndexType,
                "unknown panel index expression",
            )
            visited.add(op)
            if type(op) is arith.ConstantOp:
                require(
                    set(op.properties) == {"value"} and isinstance(op.value, builtin.IntegerAttr),
                    "unknown index constant",
                )
                return op.value.value.data
            names = {arith.AddiOp: "add", arith.MuliOp: "mul", arith.DivUIOp: "div", arith.RemUIOp: "rem"}
            require(
                type(op) in names
                and len(op.operands) == 2
                and set(op.properties) <= {"overflowFlags"}
                and all(str(v) == "#arith.overflow<none>" for v in op.properties.values()),
                "unknown index arithmetic",
            )
            return (names[type(op)], *(expression(v) for v in op.operands))

        row, panel = ("div", ("iv",), per_row), ("rem", ("iv",), per_row)
        column = ("add", ("mul", panel, width), first)
        expected_offsets = (("div", row, oh), 0, ("mul", ("rem", row, oh), sh), ("mul", column, sw))

        def slice_check(op, offsets, sizes, source_value):
            require(
                type(op) in (tensor.ExtractSliceOp, tensor.InsertSliceOp)
                and not op.attributes
                and set(op.properties) == {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
                and not op.regions
                and not op.successors,
                "unknown slice effects",
            )
            actual = iter(op.offsets)
            observed = tuple(
                v if v != tensor.ExpandShapeOp.DYNAMIC_INDEX else expression(next(actual))
                for v in op.static_offsets.get_values()
            )
            require(
                observed == offsets
                and tuple(op.static_sizes.get_values()) == sizes
                and tuple(op.static_strides.get_values()) == (1,) * len(sizes)
                and op.operands[0] is source_value,
                "incorrect source window or output position",
            )
            require(next(actual, None) is None and not op.sizes and not op.strides, "unexpected dynamic slice extent")

        patch_shape = (1, channels, (kh - 1) * dh + 1, (width - 1) * sw + (kw - 1) * dw + 1)
        slice_check(arm.patch, expected_offsets, patch_shape, source.input)
        require(arm.patch.results[0].type == builtin.TensorType(elem, patch_shape), "changed patch shape")
        gather = arm.gather
        d = AffineExpr.dimension
        maps = (AffineMap(6, 0, (d(3), d(0), d(1) * dh, d(5) * sw + d(2) * dw)), AffineMap.identity(6))
        require(
            type(gather) is GenericOp
            and _bounded_plain(gather, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
            and tuple(gather.inputs) == (arm.patch.results[0],)
            and tuple(gather.outputs) == (destination,)
            and gather.results[0].type == packing_type
            and tuple(a.data for a in gather.indexing_maps) == maps
            and tuple(t.data for t in gather.get_iterator_types()) == (IteratorType.PARALLEL,) * 6
            and _is_copy_body(gather)
            and _bounded_plain(gather.body.block.last_op),
            "gather changed copied values, coordinates or reused destination",
        )
        require(
            type(matrix) is tensor.CollapseShapeOp
            and _bounded_plain(matrix, ("reassociation",))
            and tuple(matrix.operands) == (gather.results[0],)
            and matrix.results[0].type == builtin.TensorType(elem, (k, width))
            and matrix.reassociation == _reassoc([[0, 1, 2], [3, 4, 5]]),
            "changed complete K order or matrix view",
        )
        output_type = builtin.TensorType(reference.results[0].type.element_type, (f, width))
        require(
            type(arm.output_empty) is tensor.EmptyOp
            and _bounded_plain(arm.output_empty)
            and not arm.output_empty.operands
            and arm.output_empty.results[0].type == output_type
            and type(zero) is arith.ConstantOp
            and zero.attributes == ref_fill.inputs[0].owner.attributes
            and zero.properties == ref_fill.inputs[0].owner.properties
            and zero.results[0].type == ref_fill.inputs[0].type
            and _bounded_plain(arm.output_fill, ("operandSegmentSizes",))
            and tuple(arm.output_fill.inputs) == (zero.results[0],)
            and tuple(arm.output_fill.outputs) == (arm.output_empty.results[0],)
            and arm.output_fill.body.is_structurally_equivalent(ref_fill.body),
            "changed fresh zero initialization",
        )
        inner = arm.contraction
        require(
            type(inner) is GenericOp
            and inner.attributes == reference.attributes
            and inner.properties == reference.properties
            and tuple(inner.inputs) == (source.weight, matrix.results[0])
            and tuple(inner.outputs) == (arm.output_fill.results[0],)
            and inner.results[0].type == output_type
            and inner.body.is_structurally_equivalent(reference.body),
            "changed scalar recurrence or reduction order",
        )
        slice_check(arm.insert, (0, ("add", ("mul", row, ow), column)), (f, width), inner.results[0])
        require(
            arm.insert.dest is carried
            and type(terminator) is scf.YieldOp
            and _bounded_plain(terminator)
            and tuple(terminator.operands) == (arm.insert.results[0], gather.results[0])
            and not tuple(loop.results[1].uses)
            and tuple(use.operation for use in packing.results[0].uses) == (loop,)
            and tuple(use.operation for use in destination.uses) == (gather,)
            and {use.operation for use in gather.results[0].uses} == {matrix, terminator}
            and visited == set(prefix),
            "panel ownership or completed result escaped",
        )
        initial = loop.results[0]
        arm_records.append(
            dict(
                width=width,
                first_ow=first,
                panels_per_row=per_row,
                iterations=n * oh * per_row,
                packing_shape=list(packing_type.get_shape()),
                packing_bytes=k * width * width_bytes,
                output_tile_bytes=f * width * reference.results[0].type.element_type.width.data // 8,
            )
        )
    require(
        tuple(top_operations) == rewrite.operations and rewrite.result is initial, "changed actual emitted membership"
    )
    cursor = top_operations[0]
    for op in top_operations:
        require(cursor is op, "unreported effect between panel operations")
        cursor = cursor.next_op
    return dict(
        schema="merlin.bounded_im2col.v1",
        source_identity=source.identity,
        original_packing_bytes=k * m * width_bytes,
        resident_packing_bytes=sum(a["packing_bytes"] for a in arm_records),
        gathered_elements=k * m,
        requested_gather_read_bytes=k * m * width_bytes,
        requested_gather_write_bytes=k * m * width_bytes,
        output_shape=[f, m],
        panel_invocations=sum(a["iterations"] for a in arm_records),
        arms=arm_records,
        scope="checked source windows, K order, coverage and loop-carried destinations; "
        "actual allocation unestablished",
    )
