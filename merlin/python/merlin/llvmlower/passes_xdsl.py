"""Merlin-authored xDSL passes that make the model module pure-upstream MLIR.

These run before the upstream pipeline:

- :func:`lower_quant_ext` — rewrite model2MLIR's ``quant_ext.dequantize_per_channel``
  (i8 weights + per-channel f32 scales + i32 zero points) into ``linalg.generic``:
  ``out[i,j] = (sitofp(w[i,j]) - sitofp(zp[j])) * scale[j]`` (axis-broadcast). After
  this, the module contains only upstream linalg/arith/tensor/scf/func ops.
- :func:`add_c_interface` — attach ``llvm.emit_c_interface`` so each public func gets
  a `_mlir_ciface_<name>` wrapper taking one pointer per memref argument.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

from ..frontends.linalg_mlir import make_context, parse_mlir_text  # noqa: F401


def carry_provenance(new_op, src_op, applied: str) -> None:
    """Carry model-layer provenance from a rewritten op onto its replacement + record WHAT ran.

    Three threads, so a lowering that rewrites an op does not sever it from its source layer:
      1. copy every ``prov.*`` attribute (region_id/fqn/op/... — the join key the compare + slicer use);
      2. propagate the MLIR ``Location`` (the in-IR "which original file/line" carrier — a no-op until
         model2MLIR emits locations, but then it survives passes for free);
      3. APPEND ``applied`` to an ordered ``prov.transforms`` breadcrumb (comma-joined) so the trace can
         see the transformation sequence a region actually went through ("what was actually applied").
    """
    from xdsl.dialects.builtin import StringAttr

    chain: list[str] = []
    for key, val in src_op.attributes.items():
        if key.startswith("prov."):
            new_op.attributes[key] = val
            if key == "prov.transforms" and isinstance(val, StringAttr):
                chain = [t for t in val.data.split(",") if t]
    chain.append(applied)
    new_op.attributes["prov.transforms"] = StringAttr(",".join(chain))
    loc = getattr(src_op, "location", None)
    if loc is not None:
        try:
            new_op.location = loc
        except Exception:  # noqa: BLE001 - location is best-effort in-IR provenance, never load-bearing
            pass


def collapse_overrank_matmul(module) -> int:
    """Fix `linalg.matmul` ops with a >2-D LHS + 2-D RHS (model2MLIR emits `aten.linear` on a
    3-D activation as a 2-D-map matmul over 3-D operands, which is invalid linalg). Rebuild
    each as a `linalg.generic` batched matmul with a broadcast RHS, using proper-rank maps:

        out[b.., n] = sum_k lhs[b.., k] * rhs[k, n]

    Iteration dims = (R-1 leading parallel) + N(parallel) + K(reduction). Returns count.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       IntegerType, TensorType)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    targets = [op for op in module.walk()
               if op.name == "linalg.matmul" and isinstance(op.results[0].type, TensorType)
               and len(op.results[0].type.get_shape()) > 2
               and len(op.inputs[1].type.get_shape()) == 2]
    for op in targets:
        lhs, rhs = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        et = out_t.element_type
        R = len(out_t.get_shape())                 # out rank (== lhs rank)
        ndim = R + 1                                # leading(R-1) + N + K
        idm = AffineMap.identity(ndim)
        dims = idm.results
        lead = dims[:R - 1]
        N, K = dims[R - 1], dims[R]
        lhs_map = AffineMap(ndim, 0, (*lead, K))
        rhs_map = AffineMap(ndim, 0, (K, N))
        out_map = AffineMap(ndim, 0, (*lead, N))
        iters = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1)
                          + [L.IteratorTypeAttr(par), L.IteratorTypeAttr(red)])
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        zero = arith.ConstantOp(IntegerAttr(0, et) if isinstance(et, IntegerType)
                                else FloatAttr(0.0, et))
        fill = L.FillOp(inputs=[zero.results[0]], outputs=[empty.results[0]], res=[out_t])
        body = Block(arg_types=[et, et, et])
        a, b, acc = body.args
        prod = arith.MulfOp(a, b)
        summ = arith.AddfOp(acc, prod.result)
        body.add_ops([prod, summ, L.YieldOp(summ.result)])
        gen = L.GenericOp(
            inputs=(lhs, rhs), outputs=(fill.results[0],), body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(lhs_map), AffineMapAttr(rhs_map),
                                     AffineMapAttr(out_map)]),
            iterator_types=iters, result_types=(out_t,))
        carry_provenance(gen, op, "collapse_overrank_matmul")
        for new_op in (empty, zero, fill, gen):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(gen.results[0])
        block.detach_op(op)
    return len(targets)


# quant_ext.dequantize granularities the generic (target-agnostic) dequant lowering handles. Each
# differs only in how the scale/zero-point operands index into the output — the dequant body
# (sitofp(w) - sitofp(zp)) * scale is identical. This keeps the lowering FORMAT-GENERAL (any affine
# granularity), not per_channel-overfit, and target-agnostic (pure linalg+arith → f32, runs anywhere).
_DEQUANT_KINDS = {
    "quant_ext.dequantize_per_tensor": "per_tensor",
    "quant_ext.dequantize_per_channel": "per_channel",
    "quant_ext.dequantize_per_group": "per_group",
}


def lower_quant_ext(module) -> int:
    """Rewrite all quant_ext.dequantize ops (per_tensor/per_channel/per_group); returns the count.

    Generic dequant → f32 (or bf16) via a linalg.generic; the scale/zp indexing map is derived from
    the granularity: scalar-broadcast (per_tensor), axis-projection (per_channel), or axis-floordiv by
    group_size (per_group). No target-specific datapath — runs on any backend.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, IntegerType,
                                       StringAttr, TensorType)
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.dialects.builtin import f32
    from xdsl.ir import Attribute, Block, Region
    from xdsl.utils.hints import isa
    from xdsl.ir.affine import AffineMap

    rewrites = []
    for op in module.walk():
        # quant_ext is not a registered dialect; ops parse as builtin.unregistered.
        name = getattr(op, "op_name", None)
        if op.name == "builtin.unregistered" and name is not None and name.data in _DEQUANT_KINDS:
            rewrites.append((op, _DEQUANT_KINDS[name.data]))

    for op, kind in rewrites:
        w, scale, zp = op.operands
        out_t = op.results[0].type
        rank = len(out_t.get_shape())
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        identity = AffineMap.identity(rank)
        if kind == "per_tensor":
            # scalar scale/zp broadcast to every element: affine_map<(d...) -> ()>
            sz_map = AffineMap(rank, 0, ())
        elif kind == "per_channel":
            axis = int(op.properties["axis"].value.data)
            sz_map = AffineMap(rank, 0, (identity.results[axis],))
        else:  # per_group: scale/zp indexed by the group of the quantized axis (dim floordiv group_size)
            axis = int(op.properties["axis"].value.data) if "axis" in op.properties else rank - 1
            gsz = int(op.properties["group_size"].value.data)
            results = list(identity.results)
            results[axis] = identity.results[axis] // gsz    # xDSL AffineExpr floordiv
            sz_map = AffineMap(rank, 0, tuple(results))
        maps = ArrayAttr([AffineMapAttr(identity), AffineMapAttr(sz_map),
                          AffineMapAttr(sz_map), AffineMapAttr(identity)])
        iters = ArrayAttr([linalg_ops.IteratorTypeAttr(linalg_ops.IteratorType.PARALLEL)
                           for _ in range(rank)])

        elem = out_t.element_type  # f32 or bf16 — arithmetic in the output type
        body = Block(arg_types=[w.type.element_type, scale.type.element_type,
                                zp.type.element_type, elem])
        wv, sv, zv, _ = body.args
        wf = arith.SIToFPOp(wv, elem)
        zf = arith.SIToFPOp(zv, elem)
        sub = arith.SubfOp(wf.result, zf.result)
        mul = arith.MulfOp(sub.result, sv)
        body.add_ops([wf, zf, sub, mul, linalg_ops.YieldOp(mul.result)])

        generic = linalg_ops.GenericOp(
            inputs=(w, scale, zp),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=maps,
            iterator_types=iters,
            result_types=(out_t,),
        )
        carry_provenance(generic, op, "dequant_per_channel")

        block.insert_op_before(empty, op)
        block.insert_op_before(generic, op)
        op.results[0].replace_all_uses_with(generic.results[0])
        block.detach_op(op)
    return len(rewrites)


def lower_bf16_matmul_f32acc(module) -> int:
    """Rewrite every 16-bit-float (bf16 OR fp16) ``linalg.matmul`` to accumulate in f32; returns count.

    A half-precision (bf16/fp16) ``linalg.matmul`` accumulates in its 16-bit output type, rounding
    every partial sum to the small mantissa — lossy over long contractions, and for fp16 (5-bit
    exponent, max ~65504) the accumulator also OVERFLOWS to inf on real-LLM-sized reductions.
    Hardware/torch instead accumulate in f32 and round only the final result. This rewrites

        C_bf16 = matmul(A_bf16, B_bf16)              # bf16 accumulate

    into a ``linalg.generic`` matmul whose body extends each operand to f32, multiplies and
    accumulates in f32, followed by a ``truncf`` back to bf16 — matching the reference
    numerics. (``extf``/``truncf`` here are scalar ops inside the generic body, so the
    earlier "extf is tensor-rank-polymorphic" limitation does not apply.)

    Also handles bf16 ``linalg.generic`` contractions (e.g. ``batch_matmul``, ``aten.bmm``)
    whose body is the standard ``mulf``/``addf`` reduction; these otherwise accumulate in
    bf16 too (attention scores/context), the dominant precision loss on VLAs. Their own
    indexing maps + iterator types are reused; only the accumulation dtype is promoted.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.arith import FastMathFlag, FastMathFlagsAttr
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, BFloat16Type, Float16Type,
                                       FloatAttr, TensorType, f32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    # FP-CONTRACTION LICENSE on the accumulate. The mul/add below are a multiply-accumulate; without
    # `contract` LLVM must keep them as a separate fmul + fadd and CANNOT form an FMA, so the RVV
    # backend emits vfwmul.vf + vfadd.vv instead of the single fused vfwmacc.vf -- measured: zero
    # vfwmacc in the emitted fp16 kernel until this flag was set. The f32 path never needed it
    # because linalg.matmul -> vector.contract -> vector.fma already lowers to llvm.fmuladd; a
    # MIXED-precision contraction (16-bit operands, f32 accumulator) cannot use vector.contract
    # (which is same-type only) and so arrives here as explicit mulf/addf.
    #
    # This is a strict-accuracy WIN, not a fast-math relaxation: contraction removes the
    # intermediate rounding of the product, which is exactly the single-rounding semantics the
    # hardware FMA and the f32 reference path already have. No other fast-math flag is set (no
    # reassoc/nnan/ninf), so the reduction order is unchanged and NaN/Inf behavior is preserved.
    _CONTRACT = FastMathFlagsAttr([FastMathFlag.ALLOW_CONTRACT])

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def is_half_tensor(t):
        return isinstance(t, TensorType) and isinstance(t.element_type, (BFloat16Type, Float16Type))

    def is_mul_add_contraction(op):
        """A 2-input linalg.generic with a reduction whose body is mulf+addf+yield."""
        if op.name != "linalg.generic" or len(op.inputs) != 2:
            return False
        iters = [a for a in op.iterator_types]
        if not any(getattr(a, "data", a) == L.IteratorType.REDUCTION for a in iters):
            return False
        names = [b.name for b in op.body.blocks[0].ops]
        return names == ["arith.mulf", "arith.addf", "linalg.yield"]

    targets = []
    for op in module.walk():
        if op.name == "linalg.matmul" and is_half_tensor(op.results[0].type):
            targets.append(("matmul", op))
        elif is_mul_add_contraction(op) and is_half_tensor(op.results[0].type) \
                and all(is_half_tensor(i.type) for i in op.inputs):
            targets.append(("generic", op))
    if not targets:
        return 0

    idm = AffineMap.identity(3)
    d0, d1, d2 = idm.results
    mm_maps = ArrayAttr([AffineMapAttr(AffineMap(3, 0, (d0, d2))),
                         AffineMapAttr(AffineMap(3, 0, (d2, d1))),
                         AffineMapAttr(AffineMap(3, 0, (d0, d1)))])
    mm_iters = ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(par),
                          L.IteratorTypeAttr(red)])

    for kind, op in targets:
        a, b = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        f32_t = TensorType(f32, out_t.get_shape())
        block = op.parent_block()
        maps = mm_maps if kind == "matmul" else op.indexing_maps
        iters = mm_iters if kind == "matmul" else op.iterator_types

        # Preserve each operand's own 16-bit type (bf16 or f16); truncate back to the result's.
        a_e, b_e, out_e = a.type.element_type, b.type.element_type, out_t.element_type
        empty_f = tensor.EmptyOp((), f32_t)
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        fill_f = L.FillOp(inputs=[zero.results[0]], outputs=[empty_f.results[0]], res=[f32_t])
        body = Block(arg_types=[a_e, b_e, f32])
        x, y, acc = body.args
        xf, yf = arith.ExtFOp(x, f32), arith.ExtFOp(y, f32)
        prod = arith.MulfOp(xf.result, yf.result, _CONTRACT)
        summ = arith.AddfOp(acc, prod.result, _CONTRACT)
        body.add_ops([xf, yf, prod, summ, L.YieldOp(summ.result)])
        acc_f32 = L.GenericOp(inputs=(a, b), outputs=(fill_f.results[0],),
                              body=Region(body), indexing_maps=maps, iterator_types=iters,
                              result_types=(f32_t,))

        # truncf the f32 accumulator back to the result's 16-bit type (identity map of its rank)
        rank = len(out_t.get_shape())
        idr = AffineMap.identity(rank)
        empty_b = tensor.EmptyOp((), out_t)
        body2 = Block(arg_types=[f32, out_e])
        trf = arith.TruncFOp(body2.args[0], out_e)
        body2.add_ops([trf, L.YieldOp(trf.result)])
        trunc = L.GenericOp(
            inputs=(acc_f32.results[0],), outputs=(empty_b.results[0],), body=Region(body2),
            indexing_maps=ArrayAttr([AffineMapAttr(idr), AffineMapAttr(idr)]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * rank),
            result_types=(out_t,))

        # Tag by the ACTUAL operand width -- this pass covers bf16 and fp16, and a breadcrumb
        # that says "bf16" on an fp16 kernel sends the next reader to the wrong datapath.
        carry_provenance(acc_f32, op,
                         "fp16_f32acc" if isinstance(a_e, Float16Type) else "bf16_f32acc")
        for new_op in (empty_f, zero, fill_f, acc_f32, empty_b, trunc):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(trunc.results[0])
        block.detach_op(op)
    return len(targets)


def fix_bool_sitofp(module) -> int:
    """Rewrite ``arith.sitofp`` of an ``i1`` (a bool) to ``arith.uitofp``; returns the count.

    PyTorch casts a bool tensor to float as ``True -> 1.0`` (unsigned). model2MLIR, however,
    emits a *signed* ``arith.sitofp %b : i1 to fN`` for these casts. ``sitofp`` interprets the
    i1 as a signed 1-bit integer, whose ``true`` bit pattern is the value ``-1`` (two's
    complement), so the compiled kernel produces ``-1.0`` instead of ``+1.0``. This is silent
    on the driver's scalar interpreter (Python ``float(True) == 1.0``) but wrong in every
    *compiled* kernel (host clang and RVV alike).

    The canonical place it bites: the eager-attention causal mask, built as
    ``mask_f32 * sitofp(bool)``; the sign flip turns the ``-FLT_MAX`` fill into ``+FLT_MAX``,
    so softmax attends to *future* tokens (molmoact decoder cos 0.10). Rewriting to
    ``uitofp`` (``true -> 1.0``) matches torch. ``module.walk()`` recurses into linalg.generic
    bodies, so casts inside outlined kernels are covered. Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name != "arith.sitofp":
            continue
        src = op.operands[0]
        st = src.type
        if not (isinstance(st, IntegerType) and st.width.data == 1):
            continue
        new = arith.UIToFPOp(src, op.results[0].type)
        carry_provenance(new, op, "bool_sitofp_uitofp")
        op.parent_block().insert_op_before(new, op)
        op.results[0].replace_all_uses_with(new.results[0])
        op.parent_block().detach_op(op)
        n += 1
    return n


def fix_bool_fptosi(module) -> int:
    """Rewrite a float->``i1`` ``arith.fptosi``/``arith.fptoui`` into ``x != 0``; returns the count.

    The mirror image of :func:`fix_bool_sitofp`, and far worse than a sign flip. model2MLIR emits
    a cast to a bool tensor (``aten._to_copy`` with ``prov.orig_dtype = "bool"``) as
    ``arith.fptosi %x : f32 to i1``. Signed ``i1`` holds only ``{-1, 0}``, so *every* float whose
    truncation is not one of those two values — ``1.0`` included — is **poison** in LLVM, not a
    wrong number. mlir-translate emits ``fptosi float 1.0 to i1``, instcombine folds it to
    ``poison``, and the poison propagates out of the bool tensor into whatever consumes it.

    Where that lands is not local. In smolvla the poisoned mask feeds a masked-select whose result
    count sizes a ``malloc`` and bounds a data-dependent loop, so ``br i1 poison`` becomes a
    self-branch, ``simplifycfg`` deletes every block after it, and ``forward`` compiles to 3,654
    bytes with a call set of ``malloc``/``memset``/``roundevenf`` — a whole 500M-parameter model
    erased while the link succeeds and the build reports success.

    PyTorch's bool cast is ``x != 0`` (NaN included, which is why the predicate is the *unordered*
    ``une``), so that is what replaces it. ``module.walk()`` recurses into linalg.generic bodies.
    Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import AnyFloat, FloatAttr, IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name not in ("arith.fptosi", "arith.fptoui"):
            continue
        res_t = op.results[0].type
        if not (isinstance(res_t, IntegerType) and res_t.width.data == 1):
            continue
        src = op.operands[0]
        if not isinstance(src.type, AnyFloat):
            continue
        zero = arith.ConstantOp(FloatAttr(0.0, src.type))
        # `une` = unordered-or-not-equal: NaN compares TRUE, matching torch's `bool(nan) is True`.
        cmp = arith.CmpfOp(src, zero.results[0], "une")
        carry_provenance(cmp, op, "bool_fptosi_cmpf_une")
        block = op.parent_block()
        block.insert_op_before(zero, op)
        block.insert_op_before(cmp, op)
        op.results[0].replace_all_uses_with(cmp.results[0])
        block.detach_op(op)
        n += 1
    return n


def add_c_interface(module) -> int:
    """Mark public funcs with llvm.emit_c_interface; returns count marked."""
    from xdsl.dialects.builtin import UnitAttr

    n = 0
    for op in module.walk():
        if op.name == "func.func":
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            n += 1
    return n


def preprocess_text(mlir_text: str) -> tuple[str, dict]:
    """Run all Merlin xDSL passes on linalg-on-tensors MLIR text -> upstream text."""
    from ..frontends.linalg_mlir import parse_mlir_text
    from ..xdsl_dialects._common import text as module_to_text

    module = parse_mlir_text(mlir_text)
    stats = {
        "dequantize_lowered": lower_quant_ext(module),
        "c_interface_funcs": add_c_interface(module),
    }
    return module_to_text(module), stats


# --- textual variant -----------------------------------------------------------
#
# This path exists because a whole-model artifact needs repairs BEFORE anything can parse it,
# and because xDSL 0.65 re-printed rank-reducing tensor.extract_slice with truncated
# static_sizes, so a round-trip of the full module produced invalid IR. The model artifacts keep
# each `quant_ext.dequantize_per_channel` on a single line, so the rewrite is done textually.
#
# The re-print bug is GONE on the pinned xDSL (0.68 round-trips `static_sizes = array<i64:
# 1, 1, 1, 32>` intact, and parses a 2.5 MB gemma2 recapture in ~3.5 s), so this whole variant
# could eventually be replaced by the structural :func:`preprocess_text` — but only once the two
# model2MLIR repairs below (rank-reduced sizes, over-long slice_scatter strides) are ported to
# xDSL rewrites and whole-model lowering is re-validated end to end. Until then it stays.

# All three repairs below scan on FIXED LITERALS (`"quant_ext.dequantize_per_channel"(`,
# `static_sizes = array<i64: `, `func.func @`, …) and index arithmetic. They have to be textual:
# the input is INVALID IR at this point, so there is nothing to parse yet. Every one of them
# leaves text it does not recognize byte-for-byte alone, so an unrecognized spelling reaches the
# MLIR parser and fails there loudly instead of being half-rewritten here.

_IDENT_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_DEQUANT_HEAD = ' = "quant_ext.dequantize_per_channel"('
_AXIS_HEAD = ") <{axis = "
_AXIS_TAIL = " : i64"
_PROPS_CLOSE = "}>"
_TYPES_SEP = " : ("
_RESULT_SEP = ") -> "


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _split_operand_types(line: str, at: int) -> tuple[str, str, str, str] | None:
    """Parse ``wty, sty, zty) -> outty`` starting at ``at``; ``outty`` runs to end of line.

    The first two operand types are delimited by ``, `` and the third by ``)`` — exact for
    the ranked tensor types this op carries, none of which contains a comma or a paren.
    ``None`` when the tail is not that shape, so the caller can try the next candidate.
    """
    fields = []
    for delimiter in (", ", ", "):
        end = line.find(",", at)
        if end <= at or not line.startswith(delimiter, end):
            return None
        fields.append(line[at:end])
        at = end + len(delimiter)
    end = line.find(")", at)
    if end <= at or not line.startswith(_RESULT_SEP, end):
        return None
    fields.append(line[at:end])
    outty = line[end + len(_RESULT_SEP):]
    return (*fields, outty) if outty else None


@dataclass
class _Dequant:
    """One ``quant_ext.dequantize_per_channel`` line, split into its fields.

    The real line (gemma2 2b int8 recapture, ``out/artifacts/recaptures/*/model.mlir``)::

        %1121 = "quant_ext.dequantize_per_channel"(%1115, %2, %1120) <{axis = 1 : i64,
          input_dtype = "i8"}> {prov.op = "dequantize", …} :
          (tensor<2304x2048xi8>, tensor<2048xf32>, tensor<2048xi32>) -> tensor<2304x2048xf32>

    (wrapped here; it is ONE line in the artifact — which is exactly why the rewrite can be
    done line by line without a parser).
    """

    indent: str
    result: str
    operands: list[str]          # weights, scales, zero points
    axis: int
    weight_type: str
    scale_type: str
    zero_type: str
    result_type: str


def _parse_dequant(line: str) -> "_Dequant | None":
    """Split a dequantize line into :class:`_Dequant`, or ``None`` if it is not one."""
    body = line.lstrip()
    indent = line[: len(line) - len(body)]
    head = body.find(_DEQUANT_HEAD)
    if head < 1:
        return None
    result = body[:head]
    if len(result) < 2 or not result.startswith("%") or any(c.isspace() for c in result):
        return None
    args_at = head + len(_DEQUANT_HEAD)
    axis_at = body.find(_AXIS_HEAD, args_at)
    if axis_at < 0:
        return None
    operands = body[args_at:axis_at].split(", ")
    if len(operands) != 3 or not all(
            o.startswith("%") and len(o) > 1 and not any(c.isspace() for c in o) for o in operands):
        return None
    digits_at = axis_at + len(_AXIS_HEAD)
    digits_end = digits_at
    while digits_end < len(body) and body[digits_end].isdigit():
        digits_end += 1
    if digits_end == digits_at or not body.startswith(_AXIS_TAIL, digits_end):
        return None
    props_end = body.find("}", digits_end + len(_AXIS_TAIL))
    if props_end < 0 or not body.startswith(_PROPS_CLOSE, props_end):
        return None
    # The type signature is the LAST ` : (` on the line that parses as one — anything before it
    # is the (discarded) attribute dictionary, which may itself contain a colon.
    probe = len(body)
    while True:
        types_at = body.rfind(_TYPES_SEP, props_end, probe)
        if types_at < 0:
            return None
        probe = types_at
        parsed = _split_operand_types(body, types_at + len(_TYPES_SEP))
        if parsed is not None:
            return _Dequant(indent, result, operands, int(body[digits_at:digits_end]), *parsed)


def _dequant_to_generic(op: "_Dequant") -> str:
    ind, res = op.indent, op.result
    init = f"%dq_init_{res[1:]}"   # %123 -> %dq_init_123 (suffixing %123 is invalid)
    wty, sty, zty, outty = (t.strip() for t in
                            (op.weight_type, op.scale_type, op.zero_type, op.result_type))
    axis = op.axis
    shape = outty[len("tensor<"):-1].split("x")
    elem = shape[-1]
    rank = len(shape) - 1
    dims = ", ".join(f"d{i}" for i in range(rank))
    ident = f"affine_map<({dims}) -> ({dims})>"
    proj = f"affine_map<({dims}) -> (d{axis})>"
    iters = ", ".join(['"parallel"'] * rank)
    welem = wty[len("tensor<"):-1].split("x")[-1]
    zelem = zty[len("tensor<"):-1].split("x")[-1]
    w, s, z = op.operands
    return (
        f"{ind}{init} = tensor.empty() : {outty}\n"
        f"{ind}{res} = linalg.generic {{indexing_maps = [{ident}, {proj}, {proj}, "
        f"{ident}], iterator_types = [{iters}]}} "
        f"ins({w}, {s}, {z} : {wty}, {sty}, {zty}) "
        f"outs({init} : {outty}) {{\n"
        f"{ind}^bb0(%w_el: {welem}, %s_el: {elem}, %z_el: {zelem}, %o_el: {elem}):\n"
        f"{ind}  %dq_wf = arith.sitofp %w_el : {welem} to {elem}\n"
        f"{ind}  %dq_zf = arith.sitofp %z_el : {zelem} to {elem}\n"
        f"{ind}  %dq_df = arith.subf %dq_wf, %dq_zf : {elem}\n"
        f"{ind}  %dq_r = arith.mulf %dq_df, %s_el : {elem}\n"
        f"{ind}  linalg.yield %dq_r : {elem}\n"
        f"{ind}}} -> {outty}")


_FUNC_HEAD = "func.func @"


def _attach_c_interface(text: str) -> tuple[str, int]:
    """Attach ``attributes {llvm.emit_c_interface}`` to the FIRST ``func.func`` in ``text``.

    Matches the signature by structure: ``func.func @name(`` … ``)`` (its first ``)``, which
    is the argument list's, since a ranked tensor type carries no parenthesis), an optional
    ``-> <results>``, then the body ``{``. Returns (text, number of funcs annotated).
    """
    at = 0
    while True:
        start = text.find(_FUNC_HEAD, at)
        if start < 0:
            return text, 0
        at = start + 1
        name_at = start + len(_FUNC_HEAD)
        name_end = name_at
        while name_end < len(text) and text[name_end] in _IDENT_CHARS:
            name_end += 1
        if name_end == name_at or not text.startswith("(", name_end):
            continue
        args_close = text.find(")", name_end)
        brace = text.find("{", name_end)
        if args_close < 0 or brace < 0 or brace < args_close:
            continue                       # a `{` inside the arg list: not a signature we know
        after_args = _skip_space(text, args_close + 1)
        results = ""
        if text.startswith("->", after_args):
            body_at = text.find("{", after_args)
            if body_at < 0:
                continue
            trimmed = body_at
            while trimmed > after_args and text[trimmed - 1].isspace():
                trimmed -= 1
            results = text[after_args:trimmed]
        else:
            body_at = after_args
            if not text.startswith("{", body_at):
                continue
        signature = text[start:args_close + 1]
        return (f"{text[:start]}{signature} {results} attributes {{llvm.emit_c_interface}} {{"
                f"{text[body_at + 1:]}"), 1


_EXTRACT_HEAD = '"tensor.extract_slice"('
_INSERT_HEAD = '"tensor.insert_slice"('
_PROPS_OPEN = ") <{"
_SIZES_FIELD = "static_sizes = array<i64: "
_STRIDES_FIELD = "static_strides = array<i64: "
_OFFSETS_FIELD = "static_offsets = array<i64: "
_OPERAND_TYPES = ": (tensor<"


def _array_field(props: str, field: str, *, last: bool) -> tuple[int, int] | None:
    """(value start, value end) of ``field = array<i64: …>`` inside a properties dict."""
    at = props.rfind(field) if last else props.find(field)
    if at < 0:
        return None
    value_at = at + len(field)
    value_end = props.find(">", value_at)
    return None if value_end <= value_at else (value_at, value_end)


def _slice_op_span(line: str, head_at: int, head: str, n_types: int):
    """Split a generic-form slice op into (args, props, attrs, types, end-of-op).

    ``head_at`` indexes the op's ``"tensor.<x>_slice"(``. ``n_types`` is how many operand
    types its signature carries (1 for extract_slice, 2 for insert_slice). ``None`` when the
    text at ``head_at`` is not that op spelled the way model2MLIR spells it.
    """
    args_at = head_at + len(head)
    args_end = line.find(")", args_at)
    if args_end <= args_at or not line.startswith(_PROPS_OPEN, args_end):
        return None
    props_at = args_end + len(_PROPS_OPEN)
    props_end = line.find("}", props_at)
    if props_end < 0 or not line.startswith(_PROPS_CLOSE, props_end):
        return None
    attrs_at = props_end + len(_PROPS_CLOSE)
    colon = line.find(":", attrs_at)
    if colon < 0 or not line.startswith(_OPERAND_TYPES, colon):
        return None
    types: list[str] = []
    at = colon + len(_OPERAND_TYPES) - len("tensor<")
    for index in range(n_types):
        if not line.startswith("tensor<", at):
            return None
        dims_at = at + len("tensor<")
        dims_end = line.find(">", dims_at)
        closer = ">)" if index == n_types - 1 else ">, "
        if dims_end <= dims_at or not line.startswith(closer, dims_end):
            return None
        types.append(line[dims_at:dims_end])
        at = dims_end + len(closer)
    return (line[args_at:args_end], line[props_at:props_end],
            line[attrs_at:colon], types, at)


def _fix_extract_slices(line: str) -> str:
    """Left-pad rank-reduced ``tensor.extract_slice`` static_sizes to the SOURCE rank.

    model2MLIR emits the sizes with only the result dims (real op from
    ``out/artifacts/recaptures/…/model.mlir``: ``static_sizes = array<i64: 32>`` against a
    ``tensor<1x1x32x32xf32>`` source), while upstream MLIR requires offsets/sizes/strides all
    at the source rank. Sizes already at (or above) the source rank are left alone.
    """
    out: list[str] = []
    i = 0
    while True:
        at = line.find(_EXTRACT_HEAD, i)
        if at < 0:
            out.append(line[i:])
            return "".join(out)
        span = _slice_op_span(line, at, _EXTRACT_HEAD, 1)
        if span is None:
            out.append(line[i:at + 1])
            i = at + 1
            continue
        args, props, attrs, (src,), end = span
        field = _array_field(props, _SIZES_FIELD, last=True)
        if field is None:
            out.append(line[i:end])
            i = end
            continue
        raw_sizes = props[field[0]:field[1]]
        src_dims = src.split("x")[:-1]
        sizes = [s.strip() for s in raw_sizes.split(",")]
        if len(sizes) >= len(src_dims):
            out.append(line[i:end])
            i = end
            continue
        padded = ["1"] * (len(src_dims) - len(sizes)) + sizes
        fixed = props.replace(f"{_SIZES_FIELD}{raw_sizes}>",
                              _SIZES_FIELD + ", ".join(padded) + ">")
        out.append(line[i:at])
        out.append(f'{_EXTRACT_HEAD}{args}{_PROPS_OPEN}{fixed}{_PROPS_CLOSE}'
                   f'{attrs}{_OPERAND_TYPES}{src}>)')
        i = end


def _fix_insert_slices(line: str) -> str:
    """Reset ``tensor.insert_slice`` strides that overrun the destination back to 1.

    Pre-fixes model2MLIR artifacts that predate the slice_scatter step fix (m2m's
    ``ir/decompositions.py`` read ``step`` from the ``end`` argument slot, so insert_slice got
    ``strides[dim] = end``). ``step`` is almost always 1; only a stride the slice cannot FIT under
    is reset. New captures do not hit this, but the recaptures already on disk do, so the repair
    stays.

    "Fits" is ``offset + (size - 1) * stride < extent`` -- the LAST written index, not one stride
    past it. The earlier ``size * stride > extent`` counted the step past the final element as a
    written element and so rejected a scatter that is exactly full: a ``ConvTranspose2d(stride=2)``
    upsample writes ``size=16`` elements at 0, 2, ..., 30 of a 31-wide destination, and ``16 * 2 =
    32 > 31`` reset it to 1, turning the scatter into a dense corner copy. Silently: the module
    stays verifier-clean, so it surfaced only as a wrong number (measured on deepjscc, whose two
    transposed convolutions are the only non-unit-stride inserts in any tracked recapture --
    whole-model ``fp32_cos 0.885366`` against the capture's own golden, ``1.000000`` with this
    predicate). ``merlin/tests/ir/test_insert_slice_strides.py`` pins both directions.
    """
    out: list[str] = []
    i = 0
    while True:
        at = line.find(_INSERT_HEAD, i)
        if at < 0:
            out.append(line[i:])
            return "".join(out)
        span = _slice_op_span(line, at, _INSERT_HEAD, 2)
        if span is None:
            out.append(line[i:at + 1])
            i = at + 1
            continue
        args, props, attrs, (src, dst), end = span
        sizes_field = _array_field(props, _SIZES_FIELD, last=True)
        strides_field = (None if sizes_field is None
                         else _array_field(props[sizes_field[1]:], _STRIDES_FIELD, last=False))
        offsets_field = _array_field(props, _OFFSETS_FIELD, last=False)
        if sizes_field is None or strides_field is None:
            out.append(line[i:end])
            i = end
            continue
        raw_strides = props[sizes_field[1] + strides_field[0]:sizes_field[1] + strides_field[1]]
        sizes = [int(s) for s in props[sizes_field[0]:sizes_field[1]].split(",")]
        strides = [int(s) for s in raw_strides.split(",")]
        dst_dims = [int(d) for d in dst.split("x")[:-1]]
        # The offset is part of where the last element lands; an absent field means all-zero.
        offsets = ([int(s) for s in props[offsets_field[0]:offsets_field[1]].split(",")]
                   if offsets_field is not None else [0] * len(sizes))
        repaired = [1 if off + (sz - 1) * st >= extent else st
                    for off, sz, st, extent in zip(offsets, sizes, strides, dst_dims)]
        if repaired == strides:
            out.append(line[i:end])
            i = end
            continue
        fixed = props.replace(f"{_STRIDES_FIELD}{raw_strides}>",
                              _STRIDES_FIELD + ", ".join(str(s) for s in repaired) + ">", 1)
        out.append(line[i:at])
        out.append(f'{_INSERT_HEAD}{args}{_PROPS_OPEN}{fixed}{_PROPS_CLOSE}'
                   f'{attrs}{_OPERAND_TYPES}{src}>, tensor<{dst}>)')
        i = end


def preprocess_text_textual(mlir_text: str) -> tuple[str, dict]:
    """Whole-model variant of :func:`preprocess_text` (pure text; no xDSL roundtrip)."""
    out_lines = []
    n = 0
    for line in mlir_text.splitlines():
        op = _parse_dequant(line)
        if op is not None:
            out_lines.append(_dequant_to_generic(op))
            n += 1
        else:
            out_lines.append(_fix_insert_slices(_fix_extract_slices(line)))
    text, n_funcs = _attach_c_interface("\n".join(out_lines))
    return text, {"dequantize_lowered": n, "c_interface_funcs": n_funcs}
