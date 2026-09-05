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
* ``NR`` that does not divide ``Ow`` -- the split would need a mod/floordiv in the gather's map and a
  masked panel in the schedule (``refused_nr_does_not_divide_ow``);
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
    #: ``[(shape_key, MR, NR)]`` for each contraction rewritten -- the caller's block-table entries.
    entries: list[tuple[str, int, int]] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"packed": self.packed,
                "entries": [[k, mr, nr] for k, mr, nr in self.entries],
                "refusals": dict(sorted(self.refusals.items()))}


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
    return (map_.num_dims == rank and len(map_.results) == rank
            and all(_dim_pos(r) == i for i, r in enumerate(map_.results)))


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
        report.refuse("refused_not_im2col")           # grouped im2col is rank 7 and lands here
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
    return _Match(contraction=op, gather=gather, collapse=collapse, expand=expand,
                  n=n, oh=oh, ow=ow, channels=channels, kh=kh, kw=kw,
                  sh=hterm[0], dh=hterm[1], sw=wterm[0], dw=wterm[1],
                  f=ashape[0], k=k, m=m)


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
    return [AffineMap(3, 0, (d(0), d(2))),
            AffineMap(3, 0, (d(2), d(1))),
            AffineMap(3, 0, (d(0), d(1)))]


def _dyn_slice_props(rank: int, dyn_dim: int, sizes: "list[int]"):
    """``static_offsets/sizes/strides`` for a slice with ONE dynamic offset at ``dyn_dim``."""
    from xdsl.dialects.builtin import DenseArrayBase, i64
    from xdsl.dialects.tensor import ExpandShapeOp

    offsets = [0] * rank
    # MLIR's ShapedType::kDynamic sentinel, taken from the xDSL op that names it rather than written
    # out as a literal here (one spelling of the constant, in the library that has to agree with MLIR).
    offsets[dyn_dim] = ExpandShapeOp.DYNAMIC_INDEX
    return {"static_offsets": DenseArrayBase.from_list(i64, offsets),
            "static_sizes": DenseArrayBase.from_list(i64, sizes),
            "static_strides": DenseArrayBase.from_list(i64, [1] * rank)}


def _rewrite_one(mt: _Match, nr: int) -> None:
    """Replace the matched chain in place with the panel-packed one."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import (AffineMapAttr, IndexType, IntegerAttr, TensorType, i64)
    from xdsl.dialects.linalg.ops import (GenericOp, IteratorType, IteratorTypeAttr, YieldOp)
    from xdsl.dialects.scf import ForOp, YieldOp as ScfYieldOp
    from xdsl.dialects.tensor import (CollapseShapeOp, EmptyOp, ExpandShapeOp, ExtractSliceOp,
                                      InsertSliceOp)
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
    in_map = AffineMap(7, 0, (
        d(0),                                              # n
        d(3),                                              # c
        d(1) * mt.sh + d(4) * mt.dh,                        # oh*sh + kh*dh
        d(2) * (mt.sw * nr) + d(6) * mt.sw + d(5) * mt.dw,  # (owo*NR + owi)*sw + kw*dw
    ))
    new_gather = GenericOp(
        inputs=[mt.gather.inputs[0]], outputs=[g_empty.results[0]], body=Region(gblk),
        indexing_maps=[AffineMapAttr(in_map),
                       AffineMapAttr(AffineMap(7, 0, tuple(d(i) for i in range(7))))],
        iterator_types=[par] * 7, result_types=[packed_col_t])
    for key, val in mt.gather.attributes.items():
        new_gather.attributes[key] = val
    new_gather.attributes[PANEL_ATTR] = IntegerAttr(nr, i64)

    # [N, Oh, OWO, C, kh, kw, NR] -> [MO, K, NR]. A pure row-major collapse: no element moves.
    packed_col = CollapseShapeOp(
        operands=[new_gather.results[0]],
        result_types=[TensorType(col_elem, [mo, mt.k, nr])],
        properties={"reassociation": _reassoc([[0, 1, 2], [3, 4, 5], [6]])})

    Rewriter.insert_op([g_empty, new_gather, packed_col], InsertPoint.before(mt.gather))

    # --- the panel loop ---------------------------------------------------------------------------
    # The accumulator is the EXISTING zero-filled [F, M] tensor seen as [F, MO, NR] -- a view, so the
    # zero init is reused rather than re-derived (a contraction reading an unfilled accumulator is a
    # defect this repo has already shipped once).
    acc_t = TensorType(acc_elem, [mt.f, mo, nr])
    acc = ExpandShapeOp(mt.contraction.outputs[0], [], _reassoc([[0], [1, 2]]),
                        [mt.f, mo, nr], acc_t)
    lb = ConstantOp(IntegerAttr(0, idx), idx)
    ub = ConstantOp(IntegerAttr(mo, idx), idx)
    step = ConstantOp(IntegerAttr(1, idx), idx)

    body = Block(arg_types=[idx, acc_t])
    ivar, carried = body.args
    panel_t = TensorType(col_elem, [mt.k, nr])
    tile_t = TensorType(acc_elem, [mt.f, nr])
    panel = ExtractSliceOp.build(
        operands=[packed_col.results[0], [ivar], [], []], result_types=[panel_t],
        properties=_dyn_slice_props(3, 0, [1, mt.k, nr]))
    tile = ExtractSliceOp.build(
        operands=[carried, [ivar], [], []], result_types=[tile_t],
        properties=_dyn_slice_props(3, 1, [mt.f, 1, nr]))
    inner = GenericOp(
        inputs=[mt.contraction.inputs[0], panel.results[0]], outputs=[tile.results[0]],
        body=mt.contraction.body.clone(),
        indexing_maps=[AffineMapAttr(mp) for mp in _matmul_maps()],
        iterator_types=[par, par, red], result_types=[tile_t])
    for key, val in mt.contraction.attributes.items():
        inner.attributes[key] = val
    inner.attributes[PANEL_ATTR] = IntegerAttr(nr, i64)
    put = InsertSliceOp.build(
        operands=[inner.results[0], carried, [ivar], [], []], result_types=[acc_t],
        properties=_dyn_slice_props(3, 1, [mt.f, 1, nr]))
    body.add_ops([panel, tile, inner, put, ScfYieldOp(put.results[0])])
    loop = ForOp(lb.results[0], ub.results[0], step.results[0], [acc.results[0]], Region(body))

    out = CollapseShapeOp(
        operands=[loop.results[0]],
        result_types=[TensorType(acc_elem, [mt.f, mt.m])],
        properties={"reassociation": _reassoc([[0], [1, 2]])})
    Rewriter.insert_op([acc, lb, ub, step, loop, out], InsertPoint.before(mt.contraction))

    mt.contraction.results[0].replace_all_uses_with(out.results[0])
    for dead in (mt.contraction, mt.expand, mt.collapse, mt.gather):
        Rewriter.erase_op(dead)


def rewrite_module(module, table: "dict[str, tuple[int, int]]") -> PackReport:
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
        if nr < 2 or mt.ow % nr:
            report.refuse("refused_nr_does_not_divide_ow")
            continue
        if not _accumulator_is_filled(mt.contraction):
            report.refuse("refused_accumulator_not_filled")
            continue
        _rewrite_one(mt, nr)
        report.packed += 1
        # The geometry the panel loop leaves behind: an ORDINARY [F, NR] x K contraction, which the
        # caller's SECOND block_table pass observes, prices and tags with no packed-specific
        # machinery. Recorded so a caller can check that the block it gets is the one that was packed
        # for -- a panel packed at NR whose contraction is then tiled at a narrower N would be a
        # silent half-application.
        report.entries.append((shape_key("linalg.matmul", (mt.f, nr), (mt.k,)), mr, nr))
    return report


def _accumulator_is_filled(op) -> bool:
    """The contraction's ``outs`` must be a ``linalg.fill`` result.

    A contraction accumulates into its output, so an ``outs`` that is a bare ``tensor.empty`` reads
    undefined memory. Today m2m always emits the fill; requiring it here means the reshaped
    accumulator this pass builds cannot silently become the un-zeroed one.
    """
    from xdsl.dialects.linalg.ops import FillOp

    return isinstance(op.outputs[0].owner, FillOp)


def rewrite_prepared_file(prepared: "str | Path", table: "dict[str, tuple[int, int]]",
                          work: "str | Path | None" = None) -> "tuple[Path, PackReport]":
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
    report = rewrite_module(module, table)
    if not report.packed:
        return prepared, report
    out = Path(work) / "model.bpacked.mlir" if work is not None else \
        prepared.with_name("model.bpacked.mlir")
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
    register(ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "Emit the im2col column matrix panel-packed as [M/NR][K][NR] instead of [K][M], and "
            "contract it with a 4-dim (f, mo, mi, k) generic whose schedule arm tiles [MR,1,0,0] + "
            "K-by-1 and vectorizes [MR,1,NR,1]. Today the column matrix is [K][M] and K is the "
            "innermost loop, so the B-operand pointer advances by M bytes per K step: measured on the "
            "LINKED ELF (deepjscc int8, five K loops) the advances are exactly M bytes, so each 64-byte "
            "line delivers NR=16 used bytes -- 4.0x cache-line amplification on the streamed operand. "
            "Packing makes that walk contiguous, which is what a packed-panel GEMM does. NR is read "
            "from the per-op block table (VLEN- and dtype-derived), never chosen here. A REQUEST "
            "consumed by runtime.backends.zephyr_model.prepare_for_lowering; with it absent the block "
            "table, the tags, the schedule and the emitted .ll are byte-identical. NOT MEASURED ON "
            "HARDWARE -- emitted-code and output-digest evidence only."),
    ))
    return FEATURE
