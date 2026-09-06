"""Store each weight in NR-wide PANELS at build time, so the B operand walks contiguously.

WHAT IS WRONG WITH THE CODE WE EMIT TODAY
-----------------------------------------
Measured off the LINKED ELF of the binaries this repo benchmarks (not a package default), the int8
micro-kernel is 14 instructions per k-step for 4 ``vwmacc`` covering 64 MAC lanes, its accumulators
are register-resident, there are zero vector stores in the loop and zero ``@memrefCopy`` in
``forward``. The loop body's static issue floor is 5-10% of measured cycles: **the loop body is not
the defect.** The B stream is. In every int8 loop NR contiguous bytes of B are loaded and the pointer
then advances by N bytes -- always the contraction's own N -- because B is stored ``[K][N]`` and the
schedule makes K the innermost loop. So each k-step pulls a fresh cache line and consumes NR of its
bytes, and the line is not touched again until a reuse distance of K lines.

    model        line touches   bytes used      compulsory   amplification
    lstmnetvit      73.6 MB     17.2 MB (23.4%)    5.9 MB       12.6x
    tiny_llama    8,284 MB    2,070 MB (25.0%)  1,035 MB        8.0x

WHAT THIS DOES
--------------
Stores the weight as ``[N/NR][K][NR]`` instead of ``[K][N]``, so the NR bytes the kernel wants per
k-step are contiguous and one line carries ``line_bytes / (NR * elem_bytes)`` useful k-steps instead
of one. Nothing about the arithmetic changes -- it is a permutation of the weight's elements, applied
ONCE at build time rather than never.

The contraction is re-emitted in exactly the shape :mod:`merlin.llvmlower.im2col_pack` uses for the
same job on the im2col column matrix, and for the reason recorded there: an ``scf.for`` panel loop
whose body is a PLAIN ``(m, n, k)`` contraction of extent ``[M, NR] x K``::

    acc  = expand_shape  fill([M, N])        -> [M, N/NR, NR]      (a view; the zero init is reused)
    for no in [0, N/NR):
        panel = extract_slice Bp[no, :, :]   -> [K, NR]            (contiguous by construction)
        tile  = extract_slice acc[:, no, :]  -> [M, NR]
        tile  = matmul(A, panel) into tile
        acc   = insert_slice tile
    out  = collapse_shape acc                -> [M, N]             (a view: n = no*NR + ni)

A 4-dim ``(m, no, ni, k)`` generic is NOT used, because ``im2col_pack._matmul_maps`` records that it
does not lower: vectorizing it yields a ``vector.contract`` with two N-like parallel dims that no
contraction strategy handles. The panel-loop body is an ordinary contraction the existing per-op
block machinery prices, tags and schedules with no packed-specific arm.

THE TWO CONSUMERS, which is the whole plumbing problem.
:func:`merlin.llvmlower.c_runtime.generate` is handed a **bundle directory**, not the prepared IR: it
re-parses ``model.mlir`` for the argument table (rank + dims + element size, from which the C runtime
builds the memref descriptor) and copies ``weights.safetensors``' payload verbatim into
``weights.bin``. The compiled object follows the PREPARED module; the ABI table and the blob follow
the BUNDLE. Retyping an argument in one and not the other leaves the object indexing a ``[N/NR][K][NR]``
weight that the blob stores as ``[K][N]`` -- a build that links, runs, and computes nonsense.

So this module produces BOTH halves from ONE plan and then CHECKS they agree:
:func:`rewrite_prepared_file` retypes the arguments in the prepared IR, and :func:`packed_bundle`
materializes a bundle whose ``weights.safetensors`` bytes are physically packed, whose manifest shapes
are packed, and whose ``@forward`` signature is packed -- and :func:`assert_abi_agrees` compares the
two signatures argument by argument and refuses if they differ. That comparison is strictly stronger
than "the same directory was handed to both": handing one directory to both is what the transpose
hoist does, and it never checks that the types actually match.

The packed bundle's BODY is kept valid rather than left as a lie: the inverse of the pack (and of the
layout chain the pass elided) is emitted at the top of ``@forward`` and every use of the argument is
forwarded to it, so the module still describes the same function. Nothing lowers that body -- it
exists so a reader, or a future consumer, is not handed a module whose types do not check.

WHERE NR COMES FROM
-------------------
NR is never a literal and is never assumed. The caller passes the per-op block table already derived
for THIS model (``perop_blocks.block_table``), whose N tile is widened for the board's VLEN and the
contraction's own narrowest element width and clipped by the lowering predicate. This pass reads the
NR that table assigned to THIS contraction's geometry. A geometry the table did not price has no NR,
so it is refused, not guessed. A weight feeding two contractions the table gave DIFFERENT NR is
refused; an N the assigned NR does not divide is refused. Never a pad, never a guess.

WHAT IT REFUSES TO DO (fail closed, every refusal named and counted)
--------------------------------------------------------------------
``refused_b_not_from_argument``   B does not trace to a ``@forward`` argument through layout-only ops.
``refused_shared_value``          an op on that chain has more than one use, so erasing it would keep
                                  the old layout alive and pay for both.
``refused_arg_has_other_readers`` the argument is read somewhere else; packing changes what it sees.
``refused_arg_not_2d``            the argument is not 2-D, so its pack is not unambiguous. Non-2-D
                                  weights are DECLINED here, not padded and not reshaped: the packed
                                  layout is defined against a ``[K][N]`` B operand, and a rank-3+
                                  argument reaching one has a layout chain whose inverse this pass
                                  would have to synthesize blind.
``refused_dynamic_shape``         a dynamic extent anywhere on the chain or the contraction.
``refused_unpriced``              no block-table entry for this contraction's geometry.
``refused_nr_below_2``            the table declined this geometry to a scalar N tile.
``refused_nr_does_not_divide_n``  the panel split would need a mask; never padded.
``refused_conflicting_nr``        one weight feeds two contractions at different NR.
``refused_accumulator_undefined``  the ``outs`` is a bare ``tensor.empty``, so the accumulator this
                                  pass reshapes would be reading undefined memory.
``refused_no_manifest_weight``    the argument names no weight in the bundle manifest.
``refused_stub_weight``           the weight is STUBBED (no bytes in the blob), so packing the
                                  manifest shape would describe a permutation nobody performed.
``refused_aliased_offsets``       two packed arguments share a ``data_offsets`` range. ASSERTED, not
                                  assumed: ``mining/section_build`` dedups weights by NAME, so two
                                  arguments CAN index one tensor, and packing it for one packs it for
                                  the other underneath.
``refused_unknown_dtype``         a safetensors dtype with no numpy spelling here.

NO SPEED CLAIM. This module reports line-touch arithmetic, instruction counts and output digests as
EVIDENCE (:func:`line_touch_model`). The verdict needs the board.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The default-OFF request that turns this on. A REQUEST consumed by
#: ``runtime.backends.zephyr_model.prepare_for_lowering``, exactly like ``im2col_pack.FEATURE``: it
#: edits no pipeline and no schedule by itself, so a build naming no feature is byte-identical.
FEATURE = "prepack_weight_panels"

#: bump when the packed BYTES would change, so a cached bundle from an older pack is not reused
PACK_VERSION = "1"

#: Set on the rewritten contraction, carrying the panel width. Diagnostic: it makes a packed op
#: identifiable in a dumped module without re-deriving the loop structure.
PANEL_ATTR = "merlin.w_panel"

#: The plan the preparation step leaves in ``work/`` for the ABI half to pick up. Named here because
#: the writer and the reader must agree and a silent disagreement is the failure this module exists
#: to prevent.
PLAN_FILE = "weight_panel_plan.json"

#: safetensors dtype spelling -> numpy. Shared spelling with ``baselines.bundle_rewrite``; an unknown
#: dtype is a REFUSAL, never a guess, because guessing reinterprets the weight's bytes.
_NP = {"I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64",
       "F16": "float16", "F32": "float32", "F64": "float64", "BF16": "uint16"}

#: MLIR element-type spelling -> bytes. Only what a weight argument can carry; an unknown spelling
#: yields None and the caller refuses rather than pricing the traffic at a guessed width.
_ELEM_BYTES = {"i8": 1, "u8": 1, "i16": 2, "f16": 2, "bf16": 2,
               "i32": 4, "f32": 4, "i64": 8, "f64": 8}


class PanelPackRefused(RuntimeError):
    """This model's weight panels cannot be packed soundly. Raised, never downgraded to a warning: a
    build that quietly kept the stock layout would report the lever as applied and measure the
    baseline."""


# --------------------------------------------------------------------------------------------------
# the layout chain: argument -> B operand
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutStep:
    """One layout-only op between a ``@forward`` argument and a contraction's B operand.

    Layout-only means: it moves elements, it does not compute. Each step is recorded in a form that
    can be REPLAYED on the stored bytes (numpy) and INVERTED back into MLIR, so the byte packer and
    the packed bundle's body are derived from the same record rather than restated.
    """

    kind: str                       # "transpose" | "collapse" | "expand"
    #: transpose: the permutation. collapse/expand: the reassociation groups, flattened per group.
    perm: tuple[int, ...] = ()
    groups: tuple[tuple[int, ...], ...] = ()
    in_shape: tuple[int, ...] = ()
    out_shape: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "perm": list(self.perm),
                "groups": [list(g) for g in self.groups],
                "in_shape": list(self.in_shape), "out_shape": list(self.out_shape)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "LayoutStep":
        return LayoutStep(kind=d["kind"], perm=tuple(d["perm"]),
                          groups=tuple(tuple(g) for g in d["groups"]),
                          in_shape=tuple(d["in_shape"]), out_shape=tuple(d["out_shape"]))


def replay(arr, steps: "tuple[LayoutStep, ...]"):
    """Apply `steps` to a numpy array, oldest first. The bytes half of the plan.

    ``linalg.transpose`` is ``np.transpose``; ``tensor.collapse_shape`` / ``tensor.expand_shape`` are
    row-major reshapes, which is what those ops are -- MLIR requires the reassociation to name
    contiguous groups, and both dialects are row-major. So the replay is exact, not approximate.
    """
    import numpy as np

    for st in steps:
        if tuple(arr.shape) != st.in_shape:
            raise PanelPackRefused(
                f"layout replay: step {st.kind} expected shape {st.in_shape}, got {tuple(arr.shape)}")
        if st.kind == "transpose":
            arr = np.transpose(arr, st.perm)
        elif st.kind in ("collapse", "expand"):
            arr = np.ascontiguousarray(arr).reshape(st.out_shape)
        else:
            raise PanelPackRefused(f"layout replay: unknown step kind {st.kind!r}")
    return arr


def invert(steps: "tuple[LayoutStep, ...]") -> "tuple[LayoutStep, ...]":
    """The chain that undoes `steps`, so the packed bundle's body can reconstruct the argument's
    original value from the packed one. Every step here is its own kind's inverse."""
    out: list[LayoutStep] = []
    for st in reversed(steps):
        if st.kind == "transpose":
            inv = [0] * len(st.perm)
            for i, p in enumerate(st.perm):
                inv[p] = i
            out.append(LayoutStep("transpose", perm=tuple(inv),
                                  in_shape=st.out_shape, out_shape=st.in_shape))
        elif st.kind == "collapse":
            out.append(LayoutStep("expand", groups=st.groups,
                                  in_shape=st.out_shape, out_shape=st.in_shape))
        elif st.kind == "expand":
            out.append(LayoutStep("collapse", groups=st.groups,
                                  in_shape=st.out_shape, out_shape=st.in_shape))
        else:
            raise PanelPackRefused(f"cannot invert layout step {st.kind!r}")
    return tuple(out)


# --------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------


@dataclass
class PackedArg:
    """One ``@forward`` argument whose weight this build stores as NR-wide panels."""

    arg: int
    orig_shape: tuple[int, ...]
    elem: str
    #: the layout-only ops between the argument and the contraction's B operand, oldest first
    steps: tuple[LayoutStep, ...]
    m: int
    k: int
    n: int
    mr: int
    nr: int

    @property
    def panels(self) -> int:
        return self.n // self.nr

    @property
    def packed_shape(self) -> tuple[int, int, int]:
        return (self.panels, self.k, self.nr)

    @property
    def elem_bytes(self) -> int:
        b = _ELEM_BYTES.get(self.elem)
        if b is None:                            # never priced at a guessed width
            raise PanelPackRefused(f"arg {self.arg}: unknown element width for {self.elem!r}")
        return b

    def to_json(self) -> dict[str, Any]:
        return {"arg": self.arg, "orig_shape": list(self.orig_shape), "elem": self.elem,
                "steps": [s.to_json() for s in self.steps],
                "m": self.m, "k": self.k, "n": self.n, "mr": self.mr, "nr": self.nr,
                "packed_shape": list(self.packed_shape)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "PackedArg":
        return PackedArg(arg=int(d["arg"]), orig_shape=tuple(d["orig_shape"]), elem=d["elem"],
                         steps=tuple(LayoutStep.from_json(s) for s in d["steps"]),
                         m=int(d["m"]), k=int(d["k"]), n=int(d["n"]),
                         mr=int(d["mr"]), nr=int(d["nr"]))


@dataclass
class PanelReport:
    """What the pass did and what it refused. Every refusal is named and counted -- a lever that
    silently declines is indistinguishable from one that did nothing."""

    packed: int = 0
    args: list[PackedArg] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)
    #: dead ops erased off the packed weights' layout chains (see :func:`rewrite_module`)
    dead_ops_erased: int = 0
    #: the geometry the panel loop leaves behind, so the caller can check the SECOND block-table pass
    #: prices it at the SAME NR the pack was made for. A panel packed at NR whose contraction is then
    #: tiled at a narrower N is a silent half-application.
    entries: list[tuple[str, int, int]] = field(default_factory=list)

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def to_json(self) -> dict[str, Any]:
        return {"packed": self.packed,
                "dead_ops_erased": self.dead_ops_erased,
                "args": [a.to_json() for a in self.args],
                "entries": [[k, mr, nr] for k, mr, nr in self.entries],
                "refusals": dict(sorted(self.refusals.items()))}


# --------------------------------------------------------------------------------------------------
# tracing the B operand back to a @forward argument
# --------------------------------------------------------------------------------------------------


def _static_shape(value) -> "tuple[int, ...] | None":
    from xdsl.dialects.builtin import TensorType

    t = value.type
    if not isinstance(t, TensorType):
        return None
    shape = tuple(int(d) for d in t.get_shape())
    return None if any(d < 0 for d in shape) else shape


def _reassoc_groups(op) -> "tuple[tuple[int, ...], ...]":
    """The reassociation of a collapse/expand, as plain ints."""
    return tuple(tuple(int(i.value.data) for i in g) for g in op.reassociation)


def _elem_token(value) -> str:
    return str(value.type.get_element_type())


def _trace_to_argument(value, block_args, live) -> "tuple[int, tuple[LayoutStep, ...]] | str":
    """``(arg_index, steps)`` for `value`, or a REFUSAL REASON string.

    Walks back through layout-only ops -- ``linalg.transpose``, ``tensor.collapse_shape``,
    ``tensor.expand_shape`` -- composing what each one did. Structural: an op is admitted because of
    what it IS, never because of a name, a ``prov.*`` tag or a spelling.

    Every op on the chain must be SINGLE-USE. Otherwise erasing it to read the argument directly
    would leave the old layout alive and the build would pay for both -- and, worse, the other reader
    would keep seeing a value this pass has decided nobody needs.
    """
    from xdsl.dialects.linalg.ops import TransposeOp
    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp

    by_value = {a: i for i, a in enumerate(block_args)}
    steps: list[LayoutStep] = []
    cur = value
    for _ in range(16):                          # a layout chain deeper than this is not one
        if cur in by_value:
            return by_value[cur], tuple(reversed(steps))
        owner = cur.owner
        if not isinstance(owner, (TransposeOp, CollapseShapeOp, ExpandShapeOp)):
            return "refused_b_not_from_argument"
        if len(_live_uses(cur, live)) != 1:
            return "refused_shared_value"
        out_shape = _static_shape(cur)
        if out_shape is None:
            return "refused_dynamic_shape"
        src = owner.operands[0]
        in_shape = _static_shape(src)
        if in_shape is None:
            return "refused_dynamic_shape"
        if isinstance(owner, TransposeOp):
            perm = tuple(int(p) for p in owner.permutation.get_values())
            steps.append(LayoutStep("transpose", perm=perm,
                                    in_shape=in_shape, out_shape=out_shape))
        else:
            kind = "collapse" if isinstance(owner, CollapseShapeOp) else "expand"
            steps.append(LayoutStep(kind, groups=_reassoc_groups(owner),
                                    in_shape=in_shape, out_shape=out_shape))
        cur = src
    return "refused_b_not_from_argument"


def _matmul_generic_shapes(op) -> "tuple[int, int, int] | None":
    """``(M, N, K)`` when `op` is the plain ``(d0,d2) x (d2,d1) -> (d0,d1)`` contraction, else None."""
    from xdsl.ir.affine import AffineDimExpr

    def pos(e):
        return int(e.position) if isinstance(e, AffineDimExpr) else None

    maps = [a.data for a in op.indexing_maps]
    if len(maps) != 3 or any(mp.num_dims != 3 for mp in maps):
        return None
    if [[pos(r) for r in mp.results] for mp in maps] != [[0, 2], [2, 1], [0, 1]]:
        return None
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return None
    a, b, c = _static_shape(op.inputs[0]), _static_shape(op.inputs[1]), _static_shape(op.outputs[0])
    if a is None or b is None or c is None:
        return None
    if len(a) != 2 or len(b) != 2 or len(c) != 2:
        return None
    if a[1] != b[0] or c != (a[0], b[1]):
        return None
    return c[0], c[1], a[1]


def _accumulator_is_defined(op) -> bool:
    """The contraction's ``outs`` must hold a DEFINED value, i.e. not be a bare ``tensor.empty``.

    A contraction accumulates into its output, so an uninitialized ``outs`` reads undefined memory,
    and the reshaped accumulator this pass builds must not silently become that one -- a defect this
    repo has shipped once already. Anything else is admitted, and the reason is exactly why the test
    is not "is it a ``linalg.fill``": the panel loop sees the SAME value through an ``expand_shape``,
    a view, so whatever it holds is carried through unchanged and every element is still accumulated
    into exactly once. Requiring the fill SPELLING refused 22 of 22 weights on both W8A8 captures,
    whose quant pass zeroes the accumulator with a ``tensor.splat`` -- an inapplicability that was a
    spelling and not a fact.
    """
    from xdsl.dialects.tensor import EmptyOp

    return not isinstance(op.outputs[0].owner, EmptyOp)


# --------------------------------------------------------------------------------------------------
# the rewrite
# --------------------------------------------------------------------------------------------------


def _reassoc(groups):
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64

    return ArrayAttr([ArrayAttr([IntegerAttr(j, i64) for j in g]) for g in groups])


def _rewrite_one(contraction, argval, steps, m: int, n: int, k: int, nr: int, live,
                 *, parallel_panels: bool = False) -> int:
    """Replace `contraction` with the panel loop over the (now packed) argument, in place.

    The shape is ``im2col_pack._rewrite_one``'s, deliberately: the body is a PLAIN ``[M, NR] x K``
    contraction that the existing per-op block machinery prices, tags and schedules with no
    packed-specific arm. A 4-dim ``(m, no, ni, k)`` generic is not used because it does not lower --
    see ``im2col_pack._matmul_maps``.
    """
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.dialects.scf import ForOp
    from xdsl.dialects.scf import YieldOp as ScfYieldOp
    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp, ExtractSliceOp, InsertSliceOp
    from xdsl.ir import Block, Region
    from xdsl.rewriter import InsertPoint, Rewriter

    from .im2col_pack import _dyn_slice_props, _matmul_maps

    # The dead cone is collected BEFORE anything is built. Collecting it afterwards catches the ops
    # this function is in the middle of inserting -- they read the argument too, and they are not in
    # the liveness set, which was computed before they existed. That bug emitted an `scf.for` holding
    # one `extract_slice` and no yield: it printed, it re-parsed, and the panel contraction had simply
    # vanished, so the second block-table pass priced nothing and every packed weight would have
    # lowered to scalar loops.
    cone = _dead_cone(argval, live, {id(o) for o in _chain_ops(argval, len(steps), live)})

    par = IteratorTypeAttr(IteratorType.PARALLEL)
    red = IteratorTypeAttr(IteratorType.REDUCTION)
    idx = IndexType()
    no = n // nr
    b_elem = contraction.inputs[1].type.get_element_type()
    acc_elem = contraction.outputs[0].type.get_element_type()

    # The accumulator is the EXISTING zero-filled [M, N] tensor seen as [M, NO, NR] -- a view, so the
    # zero init is reused rather than re-derived, and `n = no*NR + ni` is exactly the row-major
    # collapse that puts it back.
    acc_t = TensorType(acc_elem, [m, no, nr])
    acc = ExpandShapeOp(contraction.outputs[0], [], _reassoc([[0], [1, 2]]), [m, no, nr], acc_t)
    lb = ConstantOp(IntegerAttr(0, idx), idx)
    ub = ConstantOp(IntegerAttr(no, idx), idx)
    step = ConstantOp(IntegerAttr(1, idx), idx)

    body = Block(arg_types=[idx, acc_t])
    ivar, carried = body.args
    panel_t = TensorType(b_elem, [k, nr])
    tile_t = TensorType(acc_elem, [m, nr])
    panel = ExtractSliceOp.build(
        operands=[argval, [ivar], [], []], result_types=[panel_t],
        properties=_dyn_slice_props(3, 0, [1, k, nr]))
    tile = ExtractSliceOp.build(
        operands=[carried, [ivar], [], []], result_types=[tile_t],
        properties=_dyn_slice_props(3, 1, [m, 1, nr]))
    inner = GenericOp(
        inputs=[contraction.inputs[0], panel.results[0]], outputs=[tile.results[0]],
        body=contraction.body.clone(),
        indexing_maps=[AffineMapAttr(mp) for mp in _matmul_maps()],
        iterator_types=[par, par, red], result_types=[tile_t])
    for key, val in contraction.attributes.items():
        inner.attributes[key] = val
    inner.attributes[PANEL_ATTR] = IntegerAttr(nr, i64)
    put = InsertSliceOp.build(
        operands=[inner.results[0], carried, [ivar], [], []], result_types=[acc_t],
        properties=_dyn_slice_props(3, 1, [m, 1, nr]))
    markers = []
    if parallel_panels:
        from .panel_parallel import marker_call
        markers.append(marker_call())
    body.add_ops([*markers, panel, tile, inner, put, ScfYieldOp(put.results[0])])
    loop = ForOp(lb.results[0], ub.results[0], step.results[0], [acc.results[0]], Region(body))

    out = CollapseShapeOp(
        operands=[loop.results[0]], result_types=[TensorType(acc_elem, [m, n])],
        properties={"reassociation": _reassoc([[0], [1, 2]])})
    Rewriter.insert_op([acc, lb, ub, step, loop, out], InsertPoint.before(contraction))

    contraction.results[0].replace_all_uses_with(out.results[0])
    Rewriter.erase_op(contraction)
    # The layout chain the argument used to go through is now dead. Erased HERE rather than left to
    # a later canonicalization: a live transpose of a PACKED argument is not the transpose it used to
    # be, and leaving one would be a wrong model that still lowers and still links.
    return _erase_cone(cone)


def _live_ops(fn) -> set:
    """The ids of every operation in `fn` whose result the function's RETURN transitively depends on.

    Computed backwards from the terminator rather than forwards from the arguments, because that is
    what "observable" means: an op nothing returns cannot change the model's output, whatever it looks
    like. Values a nested region captures from its enclosing scope are followed too, so an op feeding
    a loop body counts as live. Membership is by IDENTITY (``id``): two structurally equal ops are two
    different ops, and comparing them by value would mark one live because the other is.
    """
    from xdsl.ir import BlockArgument

    live: set[int] = set()
    work = [op for op in fn.walk() if not op.results]        # terminators, `func.return` among them
    while work:
        op = work.pop()
        if id(op) in live:
            continue
        live.add(id(op))
        for inner in op.walk():
            for operand in inner.operands:
                if isinstance(operand, BlockArgument):
                    continue
                owner = operand.owner
                if owner is not None and id(owner) not in live:
                    work.append(owner)
    return live


def _live_uses(value, live: set) -> list:
    """The uses of `value` that the function's output actually depends on."""
    return [u for u in value.uses if id(u.operation) in live]


def _chain_ops(argval, depth: int, live: set) -> list:
    """The `depth` layout ops between the argument and the contraction's B operand, in order."""
    ops = []
    cur = argval
    for _ in range(depth):
        uses = _live_uses(cur, live)
        if len(uses) != 1:
            break
        op = uses[0].operation
        ops.append(op)
        if not op.results:
            break
        cur = op.results[0]
    return ops


def _dead_cone(argval, live: set, doomed: set) -> list:
    """Everything downstream of `argval` that the function's output will not depend on.

    Two kinds of op end up here and both must: the f32 fallback consumers the int8 rewrite left
    standing (already dead, and what holds the layout chain alive), and the layout chain itself
    (still live now, DOOMED once the contraction reading it has been replaced). Collected BEFORE the
    panel loop is built -- collect it afterwards and it catches the ops being inserted, which read the
    argument too and are not in a liveness set computed before they existed. That bug emitted an
    ``scf.for`` holding one ``extract_slice`` and no yield: it printed, it re-parsed, and the panel
    contraction had simply vanished, so the second block-table pass priced nothing and every packed
    weight would have lowered to scalar loops.
    """
    cone: list = []
    seen: set[int] = set()
    work = [u.operation for u in argval.uses]
    while work:
        op = work.pop()
        if id(op) in seen:
            continue
        if id(op) in live and id(op) not in doomed:
            continue
        seen.add(id(op))
        cone.append(op)
        for res in op.results:
            work.extend(u.operation for u in res.uses)
    return cone


def _erase_cone(cone: list) -> int:
    """Erase `cone`, uses first, by fixpoint on "has no remaining uses" -- which needs no topological
    order and cannot erase anything a live op still reads. Returns the count."""
    from xdsl.rewriter import Rewriter

    cone = list(cone)
    erased = 0
    changed = True
    while changed:
        changed = False
        for op in list(cone):
            if any(len(list(r.uses)) for r in op.results):
                continue
            Rewriter.erase_op(op)
            cone.remove(op)
            erased += 1
            changed = True
    if cone:                                     # fail closed: a chain we cannot erase must not be
        raise PanelPackRefused(                   # left reading a retyped argument
            f"{len(cone)} op(s) downstream of the packed argument could not be erased "
            f"({[type(o).__name__ for o in cone[:4]]}); refusing to retype an argument they read")
    return erased


def _forward(module, func_name: str):
    from ..common import mlir_query as mq

    for fn in mq.walk(module, "func.func"):
        name = fn.properties.get("sym_name") or fn.attributes.get("sym_name")
        if name is None or func_name in str(name):
            return fn
    return None


def storage_gate(bundle: "str | Path | None"):
    """``arg -> refusal reason or None`` for whether that argument's weight can be PACKED IN STORAGE.

    The IR analysis proves the contraction reads the argument through a layout chain. That is a fact
    about the graph and says nothing about how the argument's bytes are stored -- or whether it has
    any. Checking it during the PLAN, rather than only when the bundle is written, is what turns "this
    model refuses" into "these 2 of 24 arguments refuse": on the W8A8 captures the quant pass lifts
    quantized-subclass inner tensors to `@forward` arguments that live in ``extra.npz`` and appear in
    no manifest entry, and gating the whole bundle on them cost the other 22 weights their pack.

    `bundle` None -> every argument passes, which is what a caller with no bundle (a unit test over a
    module) gets. The whole-bundle checks that cannot be made per-argument -- aliased byte ranges, two
    arguments naming one tensor -- stay in :func:`pack_problems`, which still runs before any byte is
    written.
    """
    if bundle is None:
        return lambda arg, shape: None
    bundle = Path(bundle)
    try:
        man = json.loads((bundle / "weights.safetensors.manifest.json").read_text())
        header, _ = _read_header(bundle / "weights.safetensors")
    except (OSError, ValueError):
        return lambda arg, shape: None           # no bundle to check against; `pack_problems` still will

    def gate(arg: int, shape) -> "str | None":
        entry = man.get(str(arg))
        if entry is None or "weight" not in entry:
            return "refused_no_manifest_weight"
        spec = header.get(entry["weight"])
        if spec is None:
            return "refused_stub_weight"
        if _NP.get(spec.get("dtype")) is None:
            return "refused_unknown_dtype"
        if tuple(int(d) for d in spec.get("shape", ())) != tuple(shape):
            return "refused_stored_shape_disagrees"
        return None

    return gate


def rewrite_module(module, table: "dict[str, tuple[int, int]]",
                   func_name: str = "forward", bundle: "str | Path | None" = None,
                   *, parallel_panels: bool = False) -> PanelReport:
    """Panel-pack every eligible weight in `module` (mutated in place).

    `table` is the per-op block table already derived for THIS model
    (``perop_blocks.block_table``); the NR it assigned to a contraction's geometry is the panel width
    for that contraction's weight. A geometry the table did not price has no NR, so it is refused.
    """
    from xdsl.rewriter import Rewriter

    from ..common import mlir_query as mq
    from .perop_blocks import shape_key

    report = PanelReport()
    fn = _forward(module, func_name)
    if fn is None:
        report.refuse("refused_no_forward_function")
        return report

    # LIVENESS FIRST, and it is not housekeeping. The int8 datapath rewrite leaves the f32 fallback
    # chain it replaced STANDING: measured on small_llama int8, every one of the 15 weight transposes
    # has a SECOND consumer -- a `dequantize` generic whose result nothing reads. Those uses are dead
    # and the compiler's own pipeline drops them later, but to a use-count test they are
    # indistinguishable from a live second reader, and this pass refused all 15 weights as
    # `refused_shared_value`. A refusal caused by a consumer that does not exist at run time reads as
    # a fact about the model and is not one. `xdsl`'s own `dce` does not reach them either (a
    # `linalg.generic` carries a region, so it is not trivially dead), so liveness is computed here:
    # what the function RETURNS, transitively. Nothing outside that set is observable.
    live = _live_ops(fn)
    block_args = list(fn.body.blocks[0].args)
    gate = storage_gate(bundle)

    # --- collect, without mutating: a decision made mid-walk cannot see the conflicts below --------
    cands: list[tuple[Any, int, tuple[LayoutStep, ...], int, int, int, int, int]] = []
    for op in mq.walk(module, "linalg.generic"):
        mnk = _matmul_generic_shapes(op)
        if mnk is None:
            continue
        m, n, k = mnk
        traced = _trace_to_argument(op.inputs[1], block_args, live)
        if isinstance(traced, str):
            report.refuse(traced)
            continue
        arg, steps = traced
        if len(_live_uses(block_args[arg], live)) != 1:
            report.refuse("refused_arg_has_other_readers")
            continue
        if len(_static_shape(block_args[arg]) or ()) != 2:
            report.refuse("refused_arg_not_2d")
            continue
        blk = table.get(shape_key("linalg.matmul", (m, n), (k,)))
        if blk is None:
            report.refuse("refused_unpriced")
            continue
        mr, nr = int(blk[0]), int(blk[1])
        if nr < 2:
            report.refuse("refused_nr_below_2")
            continue
        if n % nr:
            report.refuse("refused_nr_does_not_divide_n")
            continue
        if not _accumulator_is_defined(op):
            report.refuse("refused_accumulator_undefined")
            continue
        if _ELEM_BYTES.get(_elem_token(block_args[arg])) is None:
            report.refuse("refused_unknown_dtype")
            continue
        stored = gate(arg, _static_shape(block_args[arg]))
        if stored is not None:
            report.refuse(stored)
            continue
        cands.append((op, arg, steps, m, n, k, mr, nr))

    # --- one weight, one NR. A weight feeding two contractions the table priced at different N tiles
    # has no single panel width; PADDING or picking one would silently mis-tile the other, so both are
    # refused and counted. (`refused_arg_has_other_readers` already covers the two-consumer case for
    # the argument itself; this stays as the check that must not be assumed away.)
    seen: dict[int, int] = {}
    for _op, arg, _s, _m, _n, _k, _mr, nr in cands:
        if seen.setdefault(arg, nr) != nr:
            seen[arg] = -1
    kept = []
    for c in cands:
        if seen.get(c[1]) == -1:
            report.refuse("refused_conflicting_nr")
            continue
        kept.append(c)

    rewriter = Rewriter()
    if parallel_panels and kept:
        from .panel_parallel import ensure_marker_declaration
        ensure_marker_declaration(module)
    for op, arg, steps, m, n, k, mr, nr in kept:
        argval = block_args[arg]
        orig_shape = _static_shape(argval)
        elem = _elem_token(argval)
        report.dead_ops_erased += _rewrite_one(
            op, argval, steps, m, n, k, nr, live, parallel_panels=parallel_panels)
        packed = PackedArg(arg=arg, orig_shape=orig_shape, elem=elem, steps=steps,
                           m=m, k=k, n=n, mr=mr, nr=nr)
        fn.replace_argument_type(arg, _packed_type(elem, packed.packed_shape), rewriter)
        report.args.append(packed)
        report.packed += 1
        report.entries.append((shape_key("linalg.matmul", (m, nr), (k,)), mr, nr))
    return report


def _packed_type(elem: str, shape):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, TensorType
    from xdsl.parser import Parser

    ctx = Context()
    ctx.load_dialect(Builtin)
    ty = Parser(ctx, elem).parse_type()
    return TensorType(ty, list(shape))


def rewrite_prepared_file(prepared: "str | Path", table: "dict[str, tuple[int, int]]",
                          work: "str | Path | None" = None,
                          bundle: "str | Path | None" = None, *,
                          parallel_panels: bool = False) -> "tuple[Path, PanelReport]":
    """Pack `prepared` and write ``model.wpacked.mlir``; returns ``(path, report)``.

    Nothing is written when nothing was packed, so a run where every candidate was refused keeps the
    original path and stays byte-identical to the baseline rather than being re-serialized.
    """
    from ..common import mlir_query as mq

    prepared = Path(prepared)
    module = mq.parse(prepared.read_text(encoding="utf-8"))
    report = rewrite_module(module, table, bundle=bundle, parallel_panels=parallel_panels)
    if not report.packed:
        return prepared, report
    out = Path(work) / "model.wpacked.mlir" if work is not None else \
        prepared.with_name("model.wpacked.mlir")
    out.write_text(str(module), encoding="utf-8")
    return out, report


# --------------------------------------------------------------------------------------------------
# the plan sidecar: how the IR half reaches the ABI half
# --------------------------------------------------------------------------------------------------


def write_plan(work: "str | Path", report: PanelReport, prepared: "str | Path") -> Path:
    """Leave `report` where the ABI half will find it. Written by the preparation step, read by
    :func:`abi_bundle`; named in :data:`PLAN_FILE` because the two must agree and a silent
    disagreement is exactly the failure this module exists to prevent."""
    p = Path(work) / PLAN_FILE
    p.write_text(json.dumps({"feature": FEATURE, "version": PACK_VERSION,
                             "prepared": str(prepared), **report.to_json()}, indent=2) + "\n")
    return p


def read_plan(work: "str | Path") -> "tuple[list[PackedArg], dict] | None":
    """`(packed args, raw plan)` left by the preparation step, or None if this build packed nothing."""
    p = Path(work) / PLAN_FILE
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    return [PackedArg.from_json(a) for a in raw.get("args", [])], raw


# --------------------------------------------------------------------------------------------------
# the packed bundle: bytes, manifest, signature
# --------------------------------------------------------------------------------------------------


def _read_header(safetensors: Path) -> tuple[dict, int]:
    with open(safetensors, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def pack_problems(src: "str | Path", args: "list[PackedArg]") -> list[str]:
    """Reasons the panels of `args` CANNOT be stored in `src`. Empty means safe.

    Every check is here because its ABSENCE is silent: the IR analysis proves the contraction reads
    the argument through a layout chain, which says NOTHING about how the argument's bytes are
    stored. Each of these would produce a bundle that builds, links and grades while reading the wrong
    bytes.
    """
    src = Path(src)
    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    header, _ = _read_header(src / "weights.safetensors")
    problems: list[str] = []
    names: dict[str, int] = {}

    for a in args:
        entry = man.get(str(a.arg))
        if entry is None or "weight" not in entry:
            problems.append(f"arg {a.arg}: packed in the IR but names no weight in the manifest")
            continue
        name = entry["weight"]
        spec = header.get(name)
        if spec is None:
            stub = bool(entry.get("stub"))
            problems.append(
                f"arg {a.arg}: weight {name!r} has no bytes in weights.safetensors "
                f"({'manifest marks it stub=true' if stub else 'dangling manifest entry'}); its "
                "panels cannot be stored, and packing only the manifest shape would describe a "
                "permutation nobody performed")
            continue
        if _NP.get(spec.get("dtype")) is None:
            problems.append(f"arg {a.arg}: weight {name!r} has safetensors dtype "
                            f"{spec.get('dtype')!r}, which has no numpy spelling here")
        if tuple(int(d) for d in spec.get("shape", ())) != tuple(a.orig_shape):
            problems.append(f"arg {a.arg}: weight {name!r} is stored {spec.get('shape')} but the IR "
                            f"argument is {list(a.orig_shape)}; refusing to pack bytes whose shape "
                            "the module and the blob do not agree on")
        if name in names:
            problems.append(f"weight {name!r} is named by BOTH arg {names[name]} and arg {a.arg}; "
                            "packing it for one packs it for the other underneath")
        names[name] = a.arg

    # ASSERTED, not assumed. `mining/section_build` dedups weights by NAME, so two arguments CAN
    # index one `data_offsets` range, and two distinct names can index overlapping bytes (a tied
    # head, an aliased view). Packing one would rewrite the other's data underneath it.
    ranges = {n: tuple(s["data_offsets"]) for n, s in header.items() if "data_offsets" in s}
    packed_names = sorted(names)
    for i, name in enumerate(packed_names):
        mine = ranges.get(name)
        if mine is None:
            continue
        for other in packed_names[i + 1:]:
            theirs = ranges.get(other)
            if theirs is not None and theirs[0] < mine[1] and mine[0] < theirs[1]:
                problems.append(f"packed weights {name!r} and {other!r} share bytes "
                                f"[{max(mine[0], theirs[0])}, {min(mine[1], theirs[1])})")
        for other, theirs in ranges.items():
            if other != name and other not in names and theirs[0] < mine[1] and mine[0] < theirs[1]:
                problems.append(f"packed weight {name!r} shares bytes with UNPACKED {other!r}; "
                                "packing it would rewrite that tensor's data underneath it")
    return problems


def pack_bytes(arr, a: PackedArg):
    """`arr` (the weight as STORED) in ``[N/NR][K][NR]``. The bytes half of the plan.

    The layout chain the pass elided is REPLAYED here rather than restated, so the permutation the
    blob gets is derived from the same record the IR rewrite was derived from. Then the single line
    that is the whole point: ``B[k][no*NR + ni] -> Bp[no][k][ni]``.
    """
    import numpy as np

    b = replay(arr, a.steps)
    if tuple(b.shape) != (a.k, a.n):
        raise PanelPackRefused(
            f"arg {a.arg}: the layout chain yields {tuple(b.shape)}, not the ({a.k}, {a.n}) B operand "
            "the IR rewrite was derived from")
    packed = np.ascontiguousarray(np.ascontiguousarray(b).reshape(a.k, a.panels, a.nr)
                                  .transpose(1, 0, 2))
    # the assertions that make a silent layout error impossible: shape, size, and a spot-check that
    # unpacking gives back exactly the operand the contraction used to read
    assert packed.shape == a.packed_shape, a.arg
    assert packed.nbytes == arr.nbytes, a.arg
    assert np.array_equal(packed.transpose(1, 0, 2).reshape(a.k, a.n), b), \
        f"arg {a.arg}: panel pack round-trip failed"
    return packed


def _pack_safetensors(src: Path, dst: Path, by_name: "dict[str, PackedArg]") -> int:
    """Copy the blob, packing exactly the tensors in `by_name`, keeping order and tight packing."""
    import numpy as np

    with open(src / "weights.safetensors", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        meta = header.pop("__metadata__", None)

        new_header: dict[str, Any] = {}
        order: list[tuple[str, int, int, dict]] = []
        off = 0
        for name, spec in header.items():
            s, e = spec["data_offsets"]
            shape = list(spec["shape"])
            if name in by_name:
                shape = list(by_name[name].packed_shape)
            new_header[name] = {"dtype": spec["dtype"], "shape": shape,
                                "data_offsets": [off, off + (e - s)]}
            order.append((name, s, e, spec))
            off += e - s
        if meta is not None:
            new_header["__metadata__"] = meta
        blob = json.dumps(new_header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)         # safetensors wants 8-byte aligned data

        done = 0
        with open(dst / "weights.safetensors", "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for name, s, e, spec in order:
                f.seek(data_start + s)
                if name not in by_name:
                    left = e - s
                    while left:
                        chunk = f.read(min(left, 64 << 20))
                        out.write(chunk)
                        left -= len(chunk)
                    continue
                dt = _NP.get(spec["dtype"])
                if dt is None:
                    raise PanelPackRefused(f"{name}: unknown safetensors dtype {spec['dtype']!r}")
                arr = np.frombuffer(f.read(e - s), dtype=dt).reshape(spec["shape"])
                out.write(pack_bytes(arr, by_name[name]).tobytes())
                done += 1
    return done


def _type_str(shape, elem: str) -> str:
    return "tensor<" + "".join(f"{int(d)}x" for d in shape) + elem + ">"


def _groups_str(groups) -> str:
    return "[" + ", ".join("[" + ", ".join(str(int(j)) for j in g) + "]" for g in groups) + "]"


def _step_text(st: LayoutStep, src: str, dst: str, empty: str, elem: str) -> list[str]:
    """One layout step as MLIR, in the spelling the captures themselves use."""
    in_t, out_t = _type_str(st.in_shape, elem), _type_str(st.out_shape, elem)
    if st.kind == "transpose":
        perm = "[" + ", ".join(str(int(p)) for p in st.perm) + "]"
        return [f"    {empty} = tensor.empty() : {out_t}",
                f"    {dst} = linalg.transpose ins({src}:{in_t}) outs({empty}:{out_t}) "
                f"permutation = {perm}"]
    if st.kind == "collapse":
        return [f"    {dst} = tensor.collapse_shape {src} {_groups_str(st.groups)} : "
                f"{in_t} into {out_t}"]
    if st.kind == "expand":
        sizes = "[" + ", ".join(str(int(d)) for d in st.out_shape) + "]"
        return [f"    {dst} = tensor.expand_shape {src} {_groups_str(st.groups)} "
                f"output_shape {sizes} : {in_t} into {out_t}"]
    raise PanelPackRefused(f"cannot emit layout step {st.kind!r}")


def unpack_steps(a: PackedArg) -> "tuple[LayoutStep, ...]":
    """The chain that turns the PACKED argument back into the value the stock module's argument held.

    Two halves, and both are derived: undo the panel pack (``[NO][K][NR] -> [K][NO][NR] -> [K][N]``),
    then undo the layout chain the IR rewrite elided (:func:`invert`). The result is emitted into the
    packed bundle's ``@forward`` so that module still describes the same function rather than being a
    signature with a body that no longer type-checks.
    """
    no, k, nr = a.packed_shape
    return (LayoutStep("transpose", perm=(1, 0, 2),
                       in_shape=(no, k, nr), out_shape=(k, no, nr)),
            LayoutStep("collapse", groups=((0,), (1, 2)),
                       in_shape=(k, no, nr), out_shape=(k, a.n))) + invert(a.steps)


def rewrite_bundle_mlir(text: str, args: "list[PackedArg]") -> tuple[str, int]:
    """Retype the packed arguments in a bundle's ``model.mlir`` and keep its body valid.

    `(new_text, n_retyped)`. Structural line handling, no regex, per the repo's rule: the signature is
    found by the `func.func @forward` prefix and an argument by its exact `%N: <type>` spelling; uses
    are substituted with the token-boundary-aware helper the transpose hoist already uses (`%15285`
    is not `%152850`).
    """
    from ..baselines.bundle_rewrite import _replace_ssa

    by_arg = {a.arg: a for a in args}
    lines = text.splitlines()
    sig_i = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("func.func @forward"):
            sig_i = i
            break
    if sig_i is None:
        raise PanelPackRefused("no `func.func @forward` line in the bundle module")

    inserted: list[str] = []
    subst: dict[str, str] = {}
    retyped = 0
    sig = lines[sig_i]
    for arg in sorted(by_arg):
        a = by_arg[arg]
        old = f"%{arg}: {_type_str(a.orig_shape, a.elem)}"
        new = f"%{arg}: {_type_str(a.packed_shape, a.elem)}"
        if old not in sig:
            raise PanelPackRefused(
                f"arg {arg} is packed in the prepared IR but the bundle signature does not declare it "
                f"as {_type_str(a.orig_shape, a.elem)}; refusing to retype an argument the two "
                "modules do not agree about")
        sig = sig.replace(old, new, 1)
        retyped += 1
        cur = f"%{arg}"
        for j, st in enumerate(unpack_steps(a)):
            dst = f"%wp{arg}_{j}"
            inserted.extend(_step_text(st, cur, dst, f"%wpe{arg}_{j}", a.elem))
            cur = dst
        subst[f"%{arg}"] = cur
    lines[sig_i] = sig

    out = lines[:sig_i + 1] + inserted
    for line in lines[sig_i + 1:]:
        for old, new in subst.items():
            line = _replace_ssa(line, old, new)
        out.append(line)
    return "\n".join(out) + "\n", retyped


def _cache_key(src: Path, args: "list[PackedArg]") -> str:
    """Identity of (this bundle's inputs, this plan, this packer). Content-addressed by size + mtime
    rather than by digest, for the reason ``weight_prepack.cache_key`` records: the blob is gigabytes
    on the models this matters most for. The PLAN is hashed in full -- NR is a function of the feature
    set and the board's VLEN, so two plans over one bundle are two different packings."""
    parts = [FEATURE, PACK_VERSION, str(src.resolve())]
    for name in ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json"):
        st = (src / name).stat()
        parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    parts.append(json.dumps([a.to_json() for a in args], sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def packed_bundle(src: "str | Path", args: "list[PackedArg]", *,
                  cache_root: "str | Path | None" = None) -> "tuple[Path, dict]":
    """`(bundle_dir, effect)` for a bundle storing `args`' weights as NR-wide panels.

    NEVER mutates `src`: the recapture tree is shared by every other session and every other
    measurement made from it. The result is a separate, cached directory published by an atomic
    rename, so two concurrent builds cannot observe a half-written bundle (the loser reuses the
    winner's).
    """
    from ..baselines.bundle_rewrite import (REWRITES_FILE, RewriteRecord, _carry_sidecars,
                                            read_rewrites, record_rewrite, retarget_weights_file)

    src = Path(src).resolve()
    if not args:
        raise PanelPackRefused(f"{src.name}: nothing to pack")
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{src.name}__{_cache_key(src, args)}"
    if dst.is_dir():
        recs = [r for r in read_rewrites(dst) if r.name == "pack_weight_panels"]
        if recs:
            return dst, {"cached": True, **recs[-1].effect}
        shutil.rmtree(dst)                       # a directory without its record is not a result

    problems = pack_problems(src, args)
    if problems:
        raise PanelPackRefused(f"cannot store the weight panels of {src.name}: "
                               + "; ".join(problems))

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    by_name = {man[str(a.arg)]["weight"]: a for a in args}
    tmp = root / f".tmp-{src.name}-{os.getpid()}-{_cache_key(src, args)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        done = _pack_safetensors(src, tmp, by_name)
        if done != len(by_name):                 # a silent undercount is the failure, not the count
            raise PanelPackRefused(
                f"{src.name}: packed {done} of {len(by_name)} weights; refusing to write a bundle "
                "whose manifest claims a layout its bytes do not have")
        for a in args:
            man[str(a.arg)]["shape"] = list(a.packed_shape)
        (tmp / "weights.safetensors.manifest.json").write_text(json.dumps(man, indent=2))

        text, retyped = rewrite_bundle_mlir((src / "model.mlir").read_text(), args)
        text, retargeted = retarget_weights_file(text, (dst / "weights.safetensors").resolve())
        (tmp / "model.mlir").write_text(text)
        skipped = _carry_sidecars(src, tmp, {"model.mlir", "weights.safetensors",
                                             "weights.safetensors.manifest.json", REWRITES_FILE})
        if (src / REWRITES_FILE).is_file():      # carry the chain forward, do not start a new one
            shutil.copy2(src / REWRITES_FILE, tmp / REWRITES_FILE)
        rec = RewriteRecord(
            name="pack_weight_panels",
            source_bundle=src.name,
            soundness=("each packed argument's only reader is the contraction whose B operand it "
                       "feeds, through a chain of layout-only ops; the pack is a permutation of that "
                       "argument's elements, replayed on the stored bytes from the SAME recorded "
                       "chain the IR rewrite used, and asserted per weight to unpack back to the "
                       "operand the contraction used to read"),
            effect={"weights_packed": done, "args_retyped": retyped,
                    "panel_widths": sorted({a.nr for a in args}),
                    "args": [a.to_json() for a in args],
                    "weights_file_retargeted": retargeted,
                    "sidecars_not_carried": skipped},
            caveats=[
                "the @forward body of THIS bundle reconstructs each argument's original value and is "
                "kept only so the module still type-checks; the build that produced it lowers the "
                "PREPARED module, in which the pack is already applied. Preparing this bundle from "
                "scratch would re-materialize the unpack at run time.",
            ] + ([f"stale, NOT carried over from the source bundle: {skipped}"] if skipped else []),
        )
        record_rewrite(tmp, rec)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.replace(tmp, dst)
    except OSError:                              # another build published first -- use theirs
        shutil.rmtree(tmp, ignore_errors=True)
        if not dst.is_dir():
            raise
        return dst, {"cached": True, **rec.effect}
    return dst, {"cached": False, **rec.effect}


def _default_cache_root() -> Path:
    from ..common.artifacts import cache_dir

    return cache_dir("weight_panel")


def assert_abi_agrees(bundle_dir: "str | Path", prepared: "str | Path") -> int:
    """Refuse unless the bundle's ``@forward`` signature is the prepared module's, argument by
    argument. Returns the argument count.

    THIS is the check that makes the two halves safe. ``c_runtime.generate`` builds the C runtime's
    memref descriptors from the BUNDLE's signature while the compiled object follows the PREPARED
    module, so a rank or an extent that differs by one argument is a build that links, runs, and
    computes nonsense -- silently, because nothing downstream compares them. Handing one directory to
    both consumers (what the transpose hoist does) does not check this; it only makes it likely.
    """
    from .model_runner import parse_forward_signature

    a = parse_forward_signature(Path(bundle_dir) / "model.mlir")
    b = parse_forward_signature(Path(prepared))
    if len(a) != len(b):
        raise PanelPackRefused(
            f"ABI skew: the bundle declares {len(a)} @forward arguments, the prepared module {len(b)}")
    bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if list(x[0]) != list(y[0]) or x[1] != y[1]]
    if bad:
        raise PanelPackRefused(
            "ABI skew between the packed bundle and the prepared module at "
            f"{len(bad)} argument(s): " + "; ".join(
                f"arg {i}: bundle {x[0]}{x[1]} vs prepared {y[0]}{y[1]}" for i, x, y in bad[:4]))
    return len(a)


def abi_bundle(model_dir: "str | Path", work: "str | Path", prepared: "str | Path",
               *, cache_root: "str | Path | None" = None) -> "tuple[Path, dict | None]":
    """The bundle ``c_runtime.generate`` must be handed for THIS build, and why.

    Returns `(model_dir, None)` unchanged when the preparation step packed nothing -- so a caller can
    route through this unconditionally and a build with the feature off is byte-identical. When it did
    pack, this materializes the packed bundle and CHECKS its signature against the prepared module
    before handing it back.
    """
    plan = read_plan(work)
    if plan is None or not plan[0]:
        return Path(model_dir), None
    args, raw = plan
    dst, effect = packed_bundle(model_dir, args, cache_root=cache_root)
    effect["abi_args_checked"] = assert_abi_agrees(dst, prepared)
    return dst, effect


# --------------------------------------------------------------------------------------------------
# the arithmetic. EVIDENCE, not a verdict -- see the module docstring.
# --------------------------------------------------------------------------------------------------


def line_touch_model(args: "list[PackedArg]", *, line_bytes: int) -> dict[str, Any]:
    """B-operand line traffic before and after the pack, per weight and in total.

    `line_bytes` is a REQUIRED parameter and is never defaulted: a cache-line width is a fact about a
    target, and this library does not know which target it is compiling for. The caller passes the one
    it derived.

    THE MODEL, stated so it can be argued with. Per weight, with M x N x K the contraction's extents,
    MR x NR its register block and `eb` the weight's element width:

        m_tiles   = ceil(M / MR)                              the kernel sweeps B once per M tile
        used      = m_tiles * K * N * eb                      bytes the kernel actually consumes
        compulsory= K * N * eb                                the weight, read once
        before    = m_tiles * (N / NR) * K * ceil(NR*eb / line_bytes)      lines touched today
        after     = m_tiles * (N / NR) * ceil(K * NR * eb / line_bytes)    lines touched packed

    ``before`` counts a fresh line for EVERY k-step. That is exact when the reuse distance (K lines,
    128 KB at K=2048) exceeds the cache, and an UPPER BOUND otherwise -- it is the measured behaviour
    of the shipping loops, where the B pointer advances by N bytes per k-step, but a small enough K
    would let a line survive to the next n-tile and this would over-count. ``after`` is the compulsory
    traffic of a contiguous panel and is a lower bound on nothing: the panel IS contiguous, so its
    lines are all fully consumed.
    """
    line_bytes = int(line_bytes)
    if line_bytes <= 0:
        raise ValueError("line_bytes must be positive; it is a derived target fact, not a default")
    rows: list[dict[str, Any]] = []
    for a in args:
        eb = a.elem_bytes
        m_tiles = -(-a.m // a.mr)
        per_step = -(-(a.nr * eb) // line_bytes)
        before = m_tiles * a.panels * a.k * per_step
        after = m_tiles * a.panels * -(-(a.k * a.nr * eb) // line_bytes)
        used = m_tiles * a.k * a.n * eb
        rows.append({"arg": a.arg, "m": a.m, "n": a.n, "k": a.k, "mr": a.mr, "nr": a.nr,
                     "elem_bytes": eb, "m_tiles": m_tiles,
                     "compulsory_bytes": a.k * a.n * eb, "used_bytes": used,
                     "line_touches_before": before, "touched_bytes_before": before * line_bytes,
                     "line_touches_after": after, "touched_bytes_after": after * line_bytes})
    tot = {k: sum(r[k] for r in rows) for k in
           ("compulsory_bytes", "used_bytes", "line_touches_before", "touched_bytes_before",
            "line_touches_after", "touched_bytes_after")}
    tot["line_bytes"] = line_bytes
    tot["weights"] = len(rows)
    for tag in ("before", "after"):
        t = tot[f"touched_bytes_{tag}"]
        tot[f"useful_fraction_{tag}"] = (tot["used_bytes"] / t) if t else None
        tot[f"amplification_vs_compulsory_{tag}"] = (t / tot["compulsory_bytes"]
                                                     if tot["compulsory_bytes"] else None)
    return {"per_weight": rows, "total": tot}


# --------------------------------------------------------------------------------------------------
# the feature
# --------------------------------------------------------------------------------------------------


def _feature():
    from .impr_features import ImprFeature

    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "store each weight as NR-wide panels ([N/NR][K][NR] instead of [K][N]) at BUILD time, so "
            "the micro-kernel's B operand walks contiguously. Measured off the linked ELF of the "
            "shipping int8 binaries: the loop body is 14 instructions per k-step for 4 vwmacc over 64 "
            "MAC lanes with register-resident accumulators, zero vector stores and zero @memrefCopy -- "
            "its static issue floor is 5-10% of measured cycles, so the body is not the defect. The B "
            "stream is: NR bytes are loaded and the pointer then advances by the contraction's own N, "
            "so one fresh cache line per k-step delivers NR useful bytes and is not touched again "
            "until a reuse distance of K lines. Packing makes those NR bytes contiguous. The "
            "contraction is re-emitted as im2col_panel_pack's panel loop -- an scf.for over N/NR whose "
            "body is a PLAIN [M, NR] x K contraction the existing per-op block machinery prices and "
            "tags -- and the argument is retyped, so the layout change reaches the INDEXING MAPS and "
            "not only the bytes. NR is read from the per-op block table, never assumed; an N it does "
            "not divide, a weight two contractions price differently, a non-2-D argument, a stubbed "
            "weight and an aliased byte range are all REFUSED and counted. Carries no pipeline, "
            "schedule or cflags hook: it changes the input weights and the prepared IR, not the "
            "compiler, so with the feature off the build is byte-identical. NO SPEED CLAIM is made "
            "from static evidence."
        ),
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, and necessary in EVERY process that
    normalizes a feature set: the lowering runs in a child process that re-imports the registry, and
    ``wholemodel_proposer._composes`` swallows the ``KeyError`` for an unregistered name and answers
    False -- so an unregistered lever is silently unproposable rather than reported."""
    from .impr_features import known, register

    if FEATURE not in known():
        register(_feature())
    return FEATURE


def guard_planned_pack(model_dir: "str | Path", out_dir: "str | Path") -> None:
    """Refuse if THIS build panel-packed its weights but is being handed the stock bundle.

    Called from :func:`merlin.llvmlower.c_runtime.generate`, whose ``out_dir`` is a subdirectory of
    the same ``work`` the preparation step wrote its plan into. Inert -- and a no-op read of a file
    that does not exist -- when nothing was packed, so every build that names no feature is
    byte-identical.

    This exists because the failure it catches is SILENT. The argument table is built from the
    bundle's ``model.mlir`` while the compiled object follows the prepared module; a caller that
    lowers the packed module and passes the stock directory gets an object indexing ``[N/NR][K][NR]``
    weights against ``[K][N]`` bytes. It links. It runs. Its numbers are wrong, and nothing between
    here and the result says so.
    """
    plan = read_plan(Path(out_dir).parent)
    if plan is None:
        return
    args, _raw = plan
    if not args:
        return
    man_path = Path(model_dir) / "weights.safetensors.manifest.json"
    if not man_path.is_file():
        raise PanelPackRefused(
            f"{FEATURE}: this build packed {len(args)} weight(s) but {model_dir} has no manifest")
    man = json.loads(man_path.read_text())
    bad = [a.arg for a in args
           if [int(d) for d in man.get(str(a.arg), {}).get("shape", ())] != list(a.packed_shape)]
    if bad:
        raise PanelPackRefused(
            f"{FEATURE}: this build panel-packed {len(args)} weight argument(s), but the bundle it is "
            f"about to build the ABI table from ({model_dir}) still stores {len(bad)} of them "
            f"unpacked (args {bad[:6]}). Route the bundle through `weight_panel.abi_bundle` -- the "
            "compiled object would index a packed weight against unpacked bytes.")
