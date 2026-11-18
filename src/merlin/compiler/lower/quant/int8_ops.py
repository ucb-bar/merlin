"""Integer fast reciprocal-square-root lowering used by quantized normalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from merlin.compiler.lower.options.quantization_scope import prequantized_source

# ---- passes_quant_rsqrt ------------------------------------------------------------------------

# Fast inverse square root magic constant (Quake): y0 = bitcast(0x5f3759df - (bits(v)>>1)).
_RSQRT_MAGIC = 0x5F3759DF


def lower_rsqrt_int(module, *, select=None) -> int:
    """Replace RMSNorm/LayerNorm's transcendental ``math.rsqrt`` with the fast inverse square
    root: a bit-hack initial guess (an integer subtract + shift on the f32 bit pattern) refined
    by Newton steps in f32 (``y = y*(1.5 - 0.5*v*y*y)``). No libm call, no integer per-lane
    divide — only ``arith`` integer ops + f32 ``vfmul``/``vfsub`` (all RVV-vectorizable, the
    plan's sanctioned f32-normalization path). Operates in-place on each ``math.rsqrt`` op inside
    its host generic body. Validated vs 1/sqrt: rmsnorm cos ~ 1.0 (4 Newton iters). Returns count."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i32

    NEWTON = 4
    rsqrts = [op for op in module.walk() if op.name == "math.rsqrt" and op.operands[0].type == f32]
    rsqrts = _select_targets(rsqrts, select)
    n = 0
    for op in rsqrts:
        v = op.operands[0]
        block = op.parent_block()
        new = []
        # y0 = bitcast(magic - (bitcast(v) >> 1))
        bi = arith.BitcastOp(v, i32)
        one = arith.ConstantOp(IntegerAttr(1, i32))
        sh = arith.ShRSIOp(bi.result, one.results[0])
        magic = arith.ConstantOp(IntegerAttr(_RSQRT_MAGIC, i32))
        sub = arith.SubiOp(magic.results[0], sh.result)
        y = arith.BitcastOp(sub.result, f32)
        new += [bi, one, sh, magic, sub, y]
        half = arith.ConstantOp(FloatAttr(0.5, f32))
        threehalf = arith.ConstantOp(FloatAttr(1.5, f32))
        new += [half, threehalf]
        cur = y.result
        for _ in range(NEWTON):
            yy = arith.MulfOp(cur, cur)
            vyy = arith.MulfOp(v, yy.result)
            hvyy = arith.MulfOp(half.results[0], vyy.result)
            s = arith.SubfOp(threehalf.results[0], hvyy.result)
            ny = arith.MulfOp(cur, s.result)
            new += [yy, vyy, hvyy, s, ny]
            cur = ny.result
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(cur)
        block.detach_op(op)
        n += 1
    return n


# ---- passes_quant_int --------------------------------------------------------------------------
# Integer (W8A8) int8 compute datapath.
#
# Today the int8 capture is WEIGHT-ONLY: ``quant_ext.dequantize_per_channel(i8, scale, zp)``
# produces an f32 weight that feeds an f32 contraction — the matmul/attention runs in float.
# This module rewrites every contraction (``linalg.matmul`` and the batched ``linalg.generic``
# matmuls/attention from ``collapse_overrank_matmul``) into a real integer contraction:
#
#   * each f32 activation operand is dynamically quantized to i8 (symmetric, per output-row:
#     ``s = max(max|x|/127, 1e-5)`` reduced over the operand's contraction dim, then
#     ``roundeven(x * (1/s))``, zero-point 0), matching TorchAO's captured arithmetic;
#   * a ``dequantize_per_channel`` weight operand is used directly as i8 (its per-channel scale
#     carried forward), the dequant dropped;
#   * the contraction runs ``i8×i8→i32`` (clang ``-march=rv64gcv`` lowers it to widening
#     ``vwmacc``), reusing the original indexing maps / iterator types;
#   * the i32 accumulator is requantized by ``acc * prod(operand scales)`` back to f32.
#
# The output stays f32 so downstream (nonlinears) is unchanged — this is the contraction stage
# (M1/M2). Weight zero-point must be 0 (symmetric per-channel). Only contractions whose output
# map is an identity projection of the parallel iterator dims are converted (matmul + the m2m
# attention/batched generics); anything else is left for the f32 fallback (``lower_quant_ext``).


def lower_conv_int8(module, *, select=None, report_out: "dict | None" = None) -> int:
    """Rewrite an f32 conv (``linalg.generic`` whose input map carries stride/dilation affine
    expressions, ``prov.op = conv2d``) into an i8×i8→i32 conv: the activation is dynamically
    quantized **per-tensor** (a conv pixel feeds many outputs, so a per-output-row act scale is
    ill-defined — one scalar ``s_a = max|x|/127`` over the whole activation), the weight is used
    as i8 with its per-output-channel scale, the contraction keeps the **exact** original
    indexing maps + iterator types (so the RVV schedule can tile/vectorize it), and the i32
    accumulator is requantized by ``acc * s_a * s_w[out_channel]``. Returns count.

    **THIS PASS IS UNREACHABLE ON EVERY CONVOLUTIONAL MODEL IN THIS FLEET, AND SAYS SO.** Its
    predicate wants a 2-input ``linalg.generic`` whose indexing maps carry a compound affine term.
    model2MLIR never produces one: it rewrites every conv into ``im2col gather + linalg.matmul``
    ahead of merlin (``prov.conv_path = "im2col_matmul"``; measured 190 such ops in deepjscc int8,
    175 in lstmnetvit int8, and ZERO ops tagged ``prov.op = "conv2d"`` in either). The compound term
    that this pass keys on lives in the GATHER, which has one input and an empty body, so nothing
    matches and the pass has never fired on any conv here. ``contraction_view``'s comment that convs
    "are handled by lower_conv_int8" describes an intent, not a fact.

    It is kept rather than deleted because its ARITHMETIC is the correct scheme and is the thing
    ``quantize_before_gather`` reuses (a per-tensor activation scale computed before the expansion);
    a frontend that hands us a fused conv would put it straight back in reach. But a pass that cannot
    fire must not be indistinguishable from one that had nothing to do, so it now COUNTS what it
    scanned into ``report_out`` (``generics_scanned``, ``compound_map_generics``, ``conv_prov_ops``,
    ``lowered``) and prints one ``INERT`` line when a module full of convolution provenance yields
    zero candidates. ``tests/ir/test_quantize_before_gather.py`` asserts that on a real bundle,
    so the deadness is a gated fact instead of a comment."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, TensorType, f32, i8, i32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.compiler.lower.quant.int8_contraction import (
        _affine_terms,
        _carry_prov,
        _constant_int,
        _emit_torchao_reciprocal_tensor,
        _emit_torchao_symmetric_scale,
        _is_dequant,
        _is_dequant_per_tensor,
        _select_targets,
    )

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def static_activation(act):
        """Recover PT2E's calibrated i8 activation, commuting through zero-padding.

        Returns ``(i8_value, scale, ops_to_insert)``. Refuses any non-zero
        zero-point: padding q=0 is exact only for the symmetric W8A8 contract.
        """
        from xdsl.dialects.tensor import InsertSliceOp

        owner = getattr(act, "owner", None)
        pad = None
        if _is_dequant_per_tensor(owner):
            deq = owner
        elif getattr(owner, "name", None) == "tensor.insert_slice":
            source_owner = getattr(owner.source, "owner", None)
            if not _is_dequant_per_tensor(source_owner):
                return None
            deq, pad = source_owner, owner
        else:
            return None
        if len(deq.operands) < 3 or _constant_int(deq.operands[2]) != 0:
            return None
        qvalue, scale = deq.operands[0], deq.operands[1]
        if not (isinstance(qvalue.type, TensorType) and qvalue.type.element_type == i8):
            return None
        if pad is None:
            return qvalue, scale, []

        # The frontend pads real zero using splat+insert_slice. For symmetric
        # quantization zero maps exactly to i8 zero, so form the same view in i8
        # instead of scanning and dynamically requantizing the padded f32 tensor.
        padded_t = TensorType(i8, list(act.type.get_shape()))
        zero = arith.ConstantOp.from_int_and_width(0, 8)
        base = tensor.SplatOp(zero.results[0], [], padded_t)
        inserted = InsertSliceOp.from_static_parameters(
            qvalue,
            base.results[0],
            list(pad.static_offsets.iter_values()),
            list(pad.static_sizes.iter_values()),
            list(pad.static_strides.iter_values()),
        )
        return inserted.results[0], scale, [zero, base, inserted]

    targets = []
    n_generics = n_compound = n_windowed = n_conv_prov = 0
    for op in module.walk():
        # Counted only on ops the FRONTEND tagged. `prov.role` is added exclusively by this file's
        # own rewrites, so excluding it keeps the diagnostic reporting the module as captured rather
        # than inflating with the ops an earlier quant pass just minted (measured: 190 -> 228 on
        # deepjscc once `quantize_before_gather` had run).
        if "prov.role" not in op.attributes and (
            "prov.conv_path" in op.attributes or getattr(op.attributes.get("prov.op"), "data", "").startswith("conv")
        ):
            n_conv_prov += 1
        if op.name != "linalg.generic" or len(op.inputs) != 2 or not op.body.blocks:
            continue
        n_generics += 1
        maps = list(op.indexing_maps)
        if not any(not isinstance(r, AffineDimExpr) for m in maps for r in m.data.results):
            continue  # not a conv (no compound map term)
        n_compound += 1
        # "compound" is not the same as "windowed". The predicate above also admits a BROADCAST,
        # whose non-dim map result is a bare constant index (`(d0, d1, 0, 0)`) with no stride in it
        # -- on deepjscc int8 all four of its candidates are exactly that (elementwise mul/add on
        # `*.res_list.1`), and each is then dropped for having a one-op body. Counting them as
        # near-misses would make this pass look like it merely missed by a hair on a model where it
        # has no candidate at all, so the window count is kept separately.
        # Fails OPEN on the diagnostic side: an expression `_affine_terms` cannot decompose (mod,
        # floordiv) counts as windowed, so an unrecognised window can never make this pass claim it
        # was inert when it was merely unlucky.
        if any(not isinstance(r, AffineDimExpr) and _affine_terms(r)[0] != [] for m in maps for r in m.data.results):
            n_windowed += 1
        out_t = op.results[0].type
        if not (isinstance(out_t, TensorType) and out_t.element_type == f32):
            continue
        bn = [b.name for b in op.body.blocks[0].ops]
        if not (("arith.mulf" in bn and "arith.addf" in bn) or ("arith.muli" in bn and "arith.addi" in bn)):
            continue
        # operand 0 = activation (compound map); operand 1 = weight (dequant i8, or plain f32 —
        # torchao leaves conv weights f32, so we dynamically per-output-channel quantize them).
        act, wt = op.inputs[0], op.inputs[1]
        if not (isinstance(act.type, TensorType) and act.type.element_type == f32):
            continue
        if not (_is_dequant(wt.owner) or (isinstance(wt.type, TensorType) and wt.type.element_type == f32)):
            continue
        targets.append(op)
    targets = _select_targets(targets, select)

    n = 0
    n_static_activation = 0
    for op in targets:
        block = op.parent_block()
        maps = list(op.indexing_maps)
        iters = [getattr(a, "data", a) for a in op.iterator_types]
        ndim = maps[0].data.num_dims
        red_flags = [it == red for it in iters]
        act, wt = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        ash = list(act.type.get_shape())
        r = len(ash)
        pre = []

        # --- weight -> i8 + per-output-channel scale s_w ---
        wt_exprs = list(maps[1].data.results)  # iterator expression per weight axis
        # A grouped direct conv indexes weight axis 0 as ``g * (F/G) + fg``.  That compound
        # expression is still entirely parallel and therefore still a per-output-channel axis;
        # requiring a bare AffineDimExpr here made the integer rewrite crash precisely on the
        # grouped-direct form that removes im2col.  Classify an axis by every dimension it uses.
        wt_keep = []
        for ax, expr in enumerate(wt_exprs):
            terms, _const = _affine_terms(expr)
            if terms and all(0 <= d < len(red_flags) and not red_flags[d] for d, _coeff in terms):
                wt_keep.append(ax)
        if _is_dequant(wt.owner):
            deq = wt.owner
            wt_i8, s_w = deq.operands[0], deq.operands[1]  # already i8 + per-channel scale
        else:
            # dynamically per-output-channel quantize the f32 weight (reduce over its kernel axes)
            wsh = list(wt.type.get_shape())
            wr = len(wsh)
            ws_shape = [wsh[ax] for ax in wt_keep]
            ws_t = TensorType(f32, ws_shape)
            id_w = AffineMap.identity(wr).results
            keep_res = [id_w[ax] for ax in wt_keep]
            w_amx_e = tensor.EmptyOp((), ws_t)
            wzero = arith.ConstantOp(FloatAttr(0.0, f32))
            w_amx_f = L.FillOp(inputs=[wzero.results[0]], outputs=[w_amx_e.results[0]], res=[ws_t])
            wrb = Block(arg_types=[f32, f32])
            wa, wacc = wrb.args
            wab = mathd.AbsFOp(wa)
            wmx = arith.MaximumfOp(wab.result, wacc)
            wrb.add_ops([wab, wmx, L.YieldOp(wmx.result)])
            w_amx = L.GenericOp(
                inputs=(wt,),
                outputs=(w_amx_f.results[0],),
                body=Region(wrb),
                indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res)]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(par if ax in wt_keep else red) for ax in range(wr)]),
                result_types=(ws_t,),
            )
            ws_e = tensor.EmptyOp((), ws_t)
            wc127 = arith.ConstantOp(FloatAttr(127.0, f32))
            id_k = AffineMap.identity(len(ws_shape)).results
            wsb = Block(arg_types=[f32, f32])
            ws_in, _ = wsb.args
            safe_wscale = _emit_torchao_symmetric_scale(wsb, ws_in, wc127.results[0])
            wsb.add_op(L.YieldOp(safe_wscale))
            s_w_g = L.GenericOp(
                inputs=(w_amx.results[0],),
                outputs=(ws_e.results[0],),
                body=Region(wsb),
                indexing_maps=ArrayAttr([amap(len(ws_shape), id_k), amap(len(ws_shape), id_k)]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(ws_shape)),
                result_types=(ws_t,),
            )
            s_w = s_w_g.results[0]
            reciprocal_ops, reciprocal = _emit_torchao_reciprocal_tensor(s_w)
            wi8_t = TensorType(i8, wsh)
            wq_e = tensor.EmptyOp((), wi8_t)
            wc127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            wqb = Block(arg_types=[f32, f32, i8])
            wxv, inv_wsv, _ = wqb.args
            wq1 = arith.MulfOp(wxv, inv_wsv)
            wq2 = mathd.RoundEvenOp(wq1.result)
            wq3 = arith.MinimumfOp(wq2.result, wc127.results[0])
            wq4 = arith.MaximumfOp(wq3.result, wc127n.results[0])
            wq5 = arith.FPToSIOp(wq4.result, i8)
            wqb.add_ops([wq1, wq2, wq3, wq4, wq5, L.YieldOp(wq5.result)])
            wq = L.GenericOp(
                inputs=(wt, reciprocal),
                outputs=(wq_e.results[0],),
                body=Region(wqb),
                indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res), amap(wr, id_w)]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * wr),
                result_types=(wi8_t,),
            )
            wt_i8 = wq.results[0]
            pre += [w_amx_e, wzero, w_amx_f, w_amx, ws_e, wc127, s_w_g, *reciprocal_ops, wq_e, wc127n, wq]

        recovered = static_activation(act)
        if recovered is not None:
            q_value, s_a_value, static_ops = recovered
            pre += static_ops
            n_static_activation += 1
        else:
            # Dynamic fallback for legacy weight-only captures. A PT2E static
            # capture must reuse the calibrated branch above for every conv.
            sc_t = TensorType(f32, [])
            ident_r = AffineMap.identity(r).results
            amx_e = tensor.EmptyOp((), sc_t)
            zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
            amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
            rb = Block(arg_types=[f32, f32])
            a_in, acc_in = rb.args
            ab = mathd.AbsFOp(a_in)
            mx = arith.MaximumfOp(ab.result, acc_in)
            rb.add_ops([ab, mx, L.YieldOp(mx.result)])
            amx = L.GenericOp(
                inputs=(act,),
                outputs=(amx_f.results[0],),
                body=Region(rb),
                indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, [])]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(red)] * r),
                result_types=(sc_t,),
            )
            sc_e = tensor.EmptyOp((), sc_t)
            c127 = arith.ConstantOp(FloatAttr(127.0, f32))
            sb = Block(arg_types=[f32, f32])
            s_in, _ = sb.args
            safe_scale = _emit_torchao_symmetric_scale(sb, s_in, c127.results[0])
            sb.add_op(L.YieldOp(safe_scale))
            s_a = L.GenericOp(
                inputs=(amx.results[0],),
                outputs=(sc_e.results[0],),
                body=Region(sb),
                indexing_maps=ArrayAttr([amap(0, []), amap(0, [])]),
                iterator_types=ArrayAttr([]),
                result_types=(sc_t,),
            )
            reciprocal_ops, reciprocal = _emit_torchao_reciprocal_tensor(s_a.results[0])
            i8_t = TensorType(i8, ash)
            q_e = tensor.EmptyOp((), i8_t)
            c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            qb = Block(arg_types=[f32, f32, i8])
            xv, inv_sv, _ = qb.args
            q1 = arith.MulfOp(xv, inv_sv)
            q2 = mathd.RoundEvenOp(q1.result)
            q3 = arith.MinimumfOp(q2.result, c127.results[0])
            q4 = arith.MaximumfOp(q3.result, c127n.results[0])
            q5 = arith.FPToSIOp(q4.result, i8)
            qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
            q = L.GenericOp(
                inputs=(act, reciprocal),
                outputs=(q_e.results[0],),
                body=Region(qb),
                indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, []), amap(r, ident_r)]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                result_types=(i8_t,),
            )
            pre += [amx_e, zero_f, amx_f, amx, sc_e, c127, s_a, *reciprocal_ops, q_e, c127n, q]
            q_value, s_a_value = q.results[0], s_a.results[0]
        # --- i8×i8→i32 conv: EXACT original maps + iterators preserved ---
        acc_t = TensorType(i32, list(out_t.get_shape()))
        acc_e = tensor.EmptyOp((), acc_t)
        zi = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zi.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        mb = Block(arg_types=[i8, i8, i32])
        av, bv, acc = mb.args
        ea = arith.ExtSIOp(av, i32)
        eb = arith.ExtSIOp(bv, i32)
        pm = arith.MuliOp(ea.result, eb.result)
        pa = arith.AddiOp(pm.result, acc)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        i8cv = L.GenericOp(
            inputs=(q_value, wt_i8),
            outputs=(acc_f.results[0],),
            body=Region(mb),
            indexing_maps=ArrayAttr(maps),
            iterator_types=op.iterator_types,
            result_types=(acc_t,),
        )

        # --- requant: out_f32[..] = sitofp(acc) * s_a * s_w[out_channel] ---
        # output map is a plain identity projection of the parallel dims.
        out_dims = [rr.position for rr in maps[-1].data.results]
        P = len(out_dims)
        d_par = AffineMap.identity(P).results
        # weight's (single) parallel iterator dim -> output channel position
        scale_exprs = [wt_exprs[ax] for ax in wt_keep]
        if not scale_exprs:
            continue
        # The output map is an identity projection of the parallel iterator prefix for every
        # admitted direct conv. Reusing the weight's parallel expression therefore maps a rank-1
        # scale as ``d1`` for ordinary conv and ``d1 * (F/G) + d2`` for grouped conv.
        if out_dims != list(range(P)):
            continue
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32, f32, f32, f32])
        accv, sav, swv, _ = wb.args
        cur = arith.SIToFPOp(accv, f32)
        m1 = arith.MulfOp(cur.result, sav)
        m2 = arith.MulfOp(m1.result, swv)
        wb.add_ops([cur, m1, m2, L.YieldOp(m2.result)])
        requant = L.GenericOp(
            inputs=(i8cv.results[0], s_a_value, s_w),
            outputs=(out_e.results[0],),
            body=Region(wb),
            indexing_maps=ArrayAttr([amap(P, d_par), amap(P, []), amap(P, scale_exprs), amap(P, d_par)]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * P),
            result_types=(out_t,),
        )
        _carry_prov(op, contraction=i8cv, requant=requant)

        for new in pre + [acc_e, zi, acc_f, i8cv, out_e, requant]:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(op)
        n += 1
    if report_out is not None:
        report_out.update(
            {
                "generics_scanned": n_generics,
                "compound_map_generics": n_compound,
                "windowed_map_generics": n_windowed,
                "conv_prov_ops": n_conv_prov,
                "lowered": n,
                "static_activation_reused": n_static_activation,
                "dynamic_activation_quantized": n - n_static_activation,
            }
        )
    # SAY SO when a module is full of convolutions and this pass found none of them. Silence here is
    # exactly how a dead pass passes for a live one: "0 lowered" reads as "nothing to do" whether the
    # module had no convs or had 190 the predicate cannot see.
    if n == 0 and n_windowed == 0 and n_conv_prov:
        print(
            f"[quant] conv_int8 INERT: {n_conv_prov} ops carry convolution provenance but 0 "
            f"linalg.generic of {n_generics} carries a WINDOWED (stride/dilation) indexing map "
            f"({n_compound} carry a constant-index one, which is a broadcast, not a conv) -- the "
            f"frontend already expanded every conv into im2col + matmul, so this pass cannot "
            f"reach one"
        )
    return n


_IEXP_A, _IEXP_B, _IEXP_C = 0.35815147, 1.35330989, 0.34401959


_IEXP_SH = 30  # fixed-point fraction bits for the poly


_IEXP_K = 8  # exponent-quantization step is ln2 / 2**K  (see _IEXP_S below)


_IEXP_CLAMP = -30.0  # exp(-30) ~ 1e-13: the value a masked/saturated input decays to


_IEXP_S = 0.6931471805599453 / (1 << _IEXP_K)  # ln2 / 2**K


_IEXP_BQ = int(round(_IEXP_B / _IEXP_S))  # b in exponent-grid units


_IEXP_AQ = int(round(_IEXP_A * (1 << _IEXP_SH) * _IEXP_S * _IEXP_S))


_IEXP_CQ = int(round(_IEXP_C * (1 << _IEXP_SH)))


_IEXP_QMAX = int(round(-_IEXP_CLAMP / _IEXP_S))  # |q| ceiling => z <= QMAX >> K


def _iexp_body(arith, i32, i64, q):
    """The integer I-BERT exp on the fixed exponent grid, as a flat op list.

    ``q`` is the exponent in grid units (i32, ``<= 0``, ``|q| <= _IEXP_QMAX``), i.e. ``x ~ q * S``.
    Returns the ops in emission order; the LAST one yields ``exp(x)`` as f32. Softmax and the SiLU
    logistic both call this, so the two integer exps cannot drift apart — they used to be two
    copies of the same twenty lines, and a fix to one would have silently missed the other.
    """
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32

    zc = arith.ConstantOp.from_int_and_width(0, 32)
    nq = arith.SubiOp(zc.result, q)  # -q >= 0
    ck = arith.ConstantOp(IntegerAttr(_IEXP_K, i32))
    z = arith.ShRSIOp(nq.result, ck.results[0])  # z = floor(-q / 2**K)
    zs = arith.ShLIOp(z.result, ck.results[0])
    r = arith.AddiOp(q, zs.result)  # r = q + z*2**K, in (-2**K, 0]
    cb = arith.ConstantOp(IntegerAttr(_IEXP_BQ, i32))
    t = arith.AddiOp(r.result, cb.results[0])  # t = r + b, all in grid units
    t64 = arith.ExtSIOp(t.result, i64)
    tt = arith.MuliOp(t64.result, t64.result)
    ca = arith.ConstantOp(IntegerAttr(_IEXP_AQ, i64))
    att = arith.MuliOp(ca.results[0], tt.result)
    cc = arith.ConstantOp(IntegerAttr(_IEXP_CQ, i64))
    ep = arith.AddiOp(att.result, cc.results[0])  # A*t^2 + C, at 2**SH fixed point
    z64 = arith.ExtSIOp(z.result, i64)
    e = arith.ShRSIOp(ep.result, z64.result)  # ... >> z  (the 2**-z range reduction)
    ef = arith.SIToFPOp(e.result, f32)
    inv = arith.ConstantOp(FloatAttr(1.0 / (1 << _IEXP_SH), f32))
    exf = arith.MulfOp(ef.result, inv.results[0])
    return [zc, nq, ck, z, zs, r, cb, t, t64, tt, ca, att, cc, ep, z64, e, ef, inv, exf]


def lower_softmax_int(module, *, select=None) -> int:
    """Replace softmax's ``math.exp`` with an integer (I-BERT) exp: a fixed-point 2nd-order
    polynomial + power-of-two shift evaluated in integer arithmetic on the FIXED exponent grid
    ``_IEXP_S``, producing the same f32 exp values the downstream sum/divide consume. The
    transcendental ``math.exp`` is gone. Anchors on each exp generic (no fragile region
    detection). Returns count.

    The exponent grid is deliberately NOT derived from the row (see ``_IEXP_S``): softmax's input
    carries the attention mask's -inf sentinels, so a per-row dynamic scale is set by the mask
    rather than by the scores. The rewrite is therefore purely element-wise — one generic, no row
    reduction — and every row is quantized identically whether it is masked or not.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, TensorType, f32, i32, i64
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    from merlin.compiler.lower.quant.int8_contraction import _carry_rewrite_prov, _select_targets

    par = L.IteratorType.PARALLEL

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def _is_softmax_exp(op):
        # a bare exp generic whose input is the max-subtraction (S - rowmax) — the softmax
        # signature. Guards against rewriting non-softmax exps (whose input may be > 0, which
        # the <=0 i-exp + [-30,0] clamp would corrupt).
        if not (
            op.name == "linalg.generic"
            and len(op.inputs) == 1
            and op.body.blocks
            and [b.name for b in op.body.blocks[0].ops] == ["math.exp", "linalg.yield"]
        ):
            return False
        src = op.inputs[0].owner
        return (
            getattr(src, "name", None) == "linalg.generic"
            and src.body.blocks
            and any(b.name == "arith.subf" for b in src.body.blocks[0].ops)
        )

    targets = [op for op in module.walk() if _is_softmax_exp(op)]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        xs = op.inputs[0]  # x = scores - rowmax, f32 [.., L], <= 0
        st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape())
        R = len(shp)
        if R < 1:
            continue
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR)
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        new = []

        # exp(x) for x <= 0, integer I-BERT i-exp on the FIXED exponent grid (_IEXP_S). Element-
        # wise and self-contained: no row reduction, no per-row constants, no reciprocal.
        #   xc = max(x, -30)                 masked positions are -inf; exp(-30) ~ 1e-13 ~ 0, and
        #                                    an unclamped -inf would make q = -inf -> poison
        #   q  = roundeven(xc / S)           exponent in grid units, in [-QMAX, 0]
        #   z  = (-q) >> K ; r = q + (z << K)    x = -z*ln2 + r*S with r*S in (-ln2, 0]
        #   e  = (A*(r+b)^2 + C) >> z        the poly, then the power-of-two range reduction
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32])
        xv, _ = eb.args
        cclamp = arith.ConstantOp(FloatAttr(_IEXP_CLAMP, f32))
        xc = arith.MaximumfOp(xv, cclamp.results[0])
        cs = arith.ConstantOp(FloatAttr(_IEXP_S, f32))
        qd = arith.DivfOp(xc.result, cs.results[0])
        qr = mathd.RoundEvenOp(qd.result)
        qi = arith.FPToSIOp(qr.result, i32)
        ops_e = [cclamp, xc, cs, qd, qr, qi]
        ops_e += _iexp_body(arith, i32, i64, qi.result)
        exf = ops_e[-1]
        eb.add_ops(ops_e + [L.YieldOp(exf.result)])
        eg = L.GenericOp(
            inputs=(xs,),
            outputs=(ee.results[0],),
            body=Region(eb),
            indexing_maps=ArrayAttr([full, full]),
            iterator_types=all_par,
            result_types=(st,),
        )
        new += [ee, eg]
        _carry_rewrite_prov(op, eg, "softmax_int")

        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(eg.results[0])
        block.detach_op(op)
        n += 1
    return n


_IGELU_A, _IGELU_B = -0.2888, -1.769


def lower_gelu_int(module, *, select=None) -> int:
    """Replace GELU's ``math.erf`` with an integer (I-BERT) i-GELU: per-row dynamic scale, the
    erf approximated by a fixed-point 2nd-order polynomial in integer arithmetic, then
    ``0.5*x*(1+erf)``. The transcendental ``math.erf`` is gone. Anchors on each erf generic.
    Numerically validated vs exact GELU: cos > 0.9999. Returns count."""
    import math as _m

    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, TensorType, f32, i32, i64
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    from merlin.compiler.lower.quant.int8_contraction import _carry_rewrite_prov, _select_targets

    par = L.IteratorType.PARALLEL
    red = L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    sqrt2 = _m.sqrt(2.0)
    SH = 24

    targets = [
        op
        for op in module.walk()
        if op.name == "linalg.generic"
        and len(op.inputs) == 1
        and op.body.blocks
        and any(b.name == "math.erf" for b in op.body.blocks[0].ops)
    ]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        x = op.inputs[0]
        st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape())
        R = len(shp)
        if R < 1:
            continue
        Rt = TensorType(f32, shp[:-1])
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR)
        row = amap(R, idR[: R - 1])
        idp = AffineMap.identity(R - 1).results
        red_it = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(red)])
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        par_rowit = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1))
        new = []

        # sx[..] = max|x| over last / 127
        ae = tensor.EmptyOp((), Rt)
        z0 = arith.ConstantOp(FloatAttr(0.0, f32))
        af = L.FillOp(inputs=[z0.results[0]], outputs=[ae.results[0]], res=[Rt])
        rb = Block(arg_types=[f32, f32])
        a_in, acc = rb.args
        ab = mathd.AbsFOp(a_in)
        mx = arith.MaximumfOp(ab.result, acc)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amax = L.GenericOp(
            inputs=(x,),
            outputs=(af.results[0],),
            body=Region(rb),
            indexing_maps=ArrayAttr([full, row]),
            iterator_types=red_it,
            result_types=(Rt,),
        )
        sre = tensor.EmptyOp((), Rt)
        c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32])
        nm, _ = sb.args
        sd = arith.DivfOp(nm, c127.results[0])
        eps = arith.ConstantOp(FloatAttr(1e-12, f32))
        sfl = arith.MaximumfOp(sd.result, eps.results[0])
        sz = arith.ConstantOp(FloatAttr(0.0, f32))
        so = arith.ConstantOp(FloatAttr(1.0, f32))
        is_zero = arith.CmpfOp(nm, sz.results[0], "oeq")
        safe = arith.SelectOp(is_zero.result, so.results[0], sfl.result)
        sb.add_ops([sd, eps, sfl, sz, so, is_zero, safe, L.YieldOp(safe.result)])
        sx = L.GenericOp(
            inputs=(amax.results[0],),
            outputs=(sre.results[0],),
            body=Region(sb),
            indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
            iterator_types=par_rowit,
            result_types=(Rt,),
        )
        new += [ae, z0, af, amax, sre, c127, sx]

        def per_row(fn, elem):
            ot = TensorType(elem, shp[:-1])
            e = tensor.EmptyOp((), ot)
            bb = Block(arg_types=[f32, elem])
            s_in, _ = bb.args
            cops, res = fn(s_in)
            bb.add_ops(cops + [L.YieldOp(res)])
            g = L.GenericOp(
                inputs=(sx.results[0],),
                outputs=(e.results[0],),
                body=Region(bb),
                indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                iterator_types=par_rowit,
                result_types=(ot,),
            )
            new.extend([e, g])
            return g.results[0]

        # qb = round(-B*sqrt2 / sx) ; Aq = round(A * sx^2/2 * 2^SH)
        def _qb(s):
            c = arith.ConstantOp(FloatAttr(-_IGELU_B * sqrt2, f32))
            d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result)
            ci = arith.FPToSIOp(r.result, i32)
            return [c, d, r, ci], ci.result

        def _A(s):
            c = arith.ConstantOp(FloatAttr(_IGELU_A * 0.5 * (1 << SH), f32))
            s2 = arith.MulfOp(s, s)
            m2 = arith.MulfOp(c.results[0], s2.result)
            r = mathd.RoundEvenOp(m2.result)
            ci = arith.FPToSIOp(r.result, i32)
            return [c, s2, m2, r, ci], ci.result

        qb = per_row(_qb, i32)
        Aq = per_row(_A, i32)

        # per-element integer i-GELU
        ge = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32, i32, i32, f32])  # x, sx, qb, Aq, out
        xv, sv, qbv, av, _ = eb.args
        c127f = arith.ConstantOp(FloatAttr(127.0, f32))
        c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qd = arith.DivfOp(xv, sv)
        qr = mathd.RoundEvenOp(qd.result)
        qcl = arith.MinimumfOp(qr.result, c127f.results[0])
        qcl2 = arith.MaximumfOp(qcl.result, c127n.results[0])
        q = arith.FPToSIOp(qcl2.result, i32)
        # |q| via arith (math.absi has no LLVM lowering in this pipeline): maxsi(q, -q)
        zi0 = arith.ConstantOp.from_int_and_width(0, 32)
        nq = arith.SubiOp(zi0.result, q.result)
        aq = arith.MaxSIOp(q.result, nq.result)
        qc = arith.MinSIOp(aq.result, qbv)
        t = arith.SubiOp(qc.result, qbv)
        t64 = arith.ExtSIOp(t.result, i64)
        a64 = arith.ExtSIOp(av, i64)
        tt = arith.MuliOp(t64.result, t64.result)
        att = arith.MuliOp(a64.result, tt.result)
        Ci = arith.ConstantOp(IntegerAttr(1 << SH, i64))
        poly = arith.AddiOp(att.result, Ci.results[0])
        polyf = arith.SIToFPOp(poly.result, f32)
        inv = arith.ConstantOp(FloatAttr(1.0 / (1 << SH), f32))
        pf = arith.MulfOp(polyf.result, inv.results[0])
        # sgn(x): x<0 -> -1 else +1
        zc = arith.ConstantOp(FloatAttr(0.0, f32))
        lt = arith.CmpfOp(xv, zc.results[0], "olt")
        nf = arith.ConstantOp(FloatAttr(-1.0, f32))
        pf1 = arith.ConstantOp(FloatAttr(1.0, f32))
        sgn = arith.SelectOp(lt.result, nf.results[0], pf1.results[0])
        erf = arith.MulfOp(sgn.result, pf.result)
        one = arith.ConstantOp(FloatAttr(1.0, f32))
        ope = arith.AddfOp(one.results[0], erf.result)
        half = arith.ConstantOp(FloatAttr(0.5, f32))
        hx = arith.MulfOp(half.results[0], xv)
        g = arith.MulfOp(hx.result, ope.result)
        eb.add_ops(
            [
                c127f,
                c127n,
                qd,
                qr,
                qcl,
                qcl2,
                q,
                zi0,
                nq,
                aq,
                qc,
                t,
                t64,
                a64,
                tt,
                att,
                Ci,
                poly,
                polyf,
                inv,
                pf,
                zc,
                lt,
                nf,
                pf1,
                sgn,
                erf,
                one,
                ope,
                half,
                hx,
                g,
                L.YieldOp(g.result),
            ]
        )
        gelu = L.GenericOp(
            inputs=(x, sx.results[0], qb, Aq),
            outputs=(ge.results[0],),
            body=Region(eb),
            indexing_maps=ArrayAttr([full, row, row, row, full]),
            iterator_types=all_par,
            result_types=(st,),
        )
        new += [ge, gelu]
        _carry_rewrite_prov(op, gelu, "gelu_int")
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(gelu.results[0])
        block.detach_op(op)
        n += 1
    return n


def lower_silu_int(module, *, select=None) -> int:
    """Replace the logistic ``sigmoid`` generic (``1/(1+exp(-x))``, the SiLU/swish nonlinear)
    with an integer (I-BERT) version: the shared integer-exp (``_iexp_body``, the same fixed-point
    poly + power-of-two shift softmax uses, on the same fixed exponent grid) evaluated on
    ``-|x| <= 0``, then the stable logistic ``sig = (x>=0 ? 1 : e) / (1 + e)`` in f32 (the divide
    is RVV ``vfdiv`` — no integer per-lane divide). The ``math.exp`` is gone. Anchors on each
    ``prov.op = sigmoid`` generic; the downstream ``x * sigmoid`` multiply stays. Returns count."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, TensorType, f32, i32, i64
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    from merlin.compiler.lower.quant.int8_contraction import _carry_rewrite_prov, _select_targets

    par = L.IteratorType.PARALLEL

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def _is_sigmoid(op):
        if not (op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks):
            return False
        hint = op.attributes.get("prov.op")
        return getattr(hint, "data", None) == "sigmoid" and any(b.name == "math.exp" for b in op.body.blocks[0].ops)

    targets = [op for op in module.walk() if _is_sigmoid(op)]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        x = op.inputs[0]
        st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape())
        R = len(shp)
        if R < 1:
            continue
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR)
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        new = []

        # sigmoid body: e = iexp(-|x|); sig = (x>=0 ? 1 : e)/(1+e). Element-wise on the SAME fixed
        # exponent grid softmax uses (_iexp_body) — the scale used to be a per-row max|x|/127,
        # which made the logistic's resolution depend on the largest activation sharing its row.
        #   q = -min(roundeven(|x| / S), QMAX)     QMAX caps z so the >> z below stays in range
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32])
        xv, _ = eb.args
        axv = mathd.AbsFOp(xv)
        cs = arith.ConstantOp(FloatAttr(_IEXP_S, f32))
        qd = arith.DivfOp(axv.result, cs.results[0])
        qr = mathd.RoundEvenOp(qd.result)
        cqm = arith.ConstantOp(FloatAttr(float(_IEXP_QMAX), f32))
        qcl = arith.MinimumfOp(qr.result, cqm.results[0])
        qa = arith.FPToSIOp(qcl.result, i32)
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        qv = arith.SubiOp(c0.result, qa.result)
        ops_s = [axv, cs, qd, qr, cqm, qcl, qa, c0, qv]
        ops_s += _iexp_body(arith, i32, i64, qv.result)
        exf = ops_s[-1]
        # sig = num/denom, num = (x>=0 ? 1 : e), denom = 1+e
        one = arith.ConstantOp(FloatAttr(1.0, f32))
        denom = arith.AddfOp(one.results[0], exf.result)
        zc = arith.ConstantOp(FloatAttr(0.0, f32))
        ge0 = arith.CmpfOp(xv, zc.results[0], "oge")
        num = arith.SelectOp(ge0.result, one.results[0], exf.result)
        sig = arith.DivfOp(num.result, denom.result)
        eb.add_ops(ops_s + [one, denom, zc, ge0, num, sig, L.YieldOp(sig.result)])
        sg = L.GenericOp(
            inputs=(x,),
            outputs=(ee.results[0],),
            body=Region(eb),
            indexing_maps=ArrayAttr([full, full]),
            iterator_types=all_par,
            result_types=(st,),
        )
        new += [ee, sg]
        _carry_rewrite_prov(op, sg, "silu_int")
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(sg.results[0])
        block.detach_op(op)
        n += 1
    return n


# ---- quant_passes ------------------------------------------------------------------------------
# Registry over the integer (W8A8) quantization passes — the ``quantization`` region's registrable
# edit-point.
#
# Mirrors ``impr_features``: the six ``lower_*_int`` passes in ``passes_quant_int`` become NAMED, ordered,
# toggleable entries instead of a hardcoded call sequence, so "change how we apply quantization / real-quant
# vs QDQ-fake" is an obvious edit-point (register a pass, toggle/reorder the set) rather than editing a
# runtime call site. ``apply_quant()`` with the default set runs EXACTLY the historical sequence in the same
# order, so the int8 datapath stays byte-identical — the registry is a seam, not a behavior change.

# Canonical order (the historical sequence in dispatch_runtime / zephyr_model). Static — no import.
_ORDER = ("contraction_int8", "conv_int8", "softmax_int", "gelu_int", "silu_int", "rsqrt_int")
# W8A8 compute selects integer contractions, not an unrelated host numerical policy.
# The four nonlinear approximations remain available through an explicit pass list.
_COMPUTE_DEFAULT = ("contraction_int8", "conv_int8")


@dataclass(frozen=True)
class QuantPass:
    name: str  # short registry name
    op_class: str  # contraction | conv | softmax | gelu | silu | rsqrt
    fn: Callable  # (module) -> int  (count of ops lowered; mutates module in place)
    description: str


def registry() -> dict[str, QuantPass]:
    """The registered int8 quant passes wired to the real ``passes_quant_int`` functions (lazy import —
    that module needs xDSL). Rebuilt per call so a test can monkeypatch the underlying functions."""

    passes = [
        QuantPass(
            "contraction_int8", "contraction", lower_contraction_int8, "matmul/attention -> i8xi8->i32 + requant"
        ),
        QuantPass("conv_int8", "conv", lower_conv_int8, "conv2d -> i8xi8->i32 (per-tensor act-quant) + requant"),
        QuantPass("softmax_int", "softmax", lower_softmax_int, "softmax exp -> integer I-BERT i-exp (no math.exp)"),
        QuantPass("gelu_int", "gelu", lower_gelu_int, "GELU erf -> integer I-BERT i-GELU (no math.erf)"),
        QuantPass("silu_int", "silu", lower_silu_int, "SiLU sigmoid -> integer i-sigmoid"),
        QuantPass("rsqrt_int", "rsqrt", lower_rsqrt_int, "rsqrt -> integer fast-rsqrt"),
    ]
    return {p.name: p for p in passes}


def known() -> tuple[str, ...]:
    """The registered pass names in canonical order (import-free)."""
    return _ORDER


def _explicit_passes(passes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate an already selected numerical pass set in canonical order."""
    if not isinstance(passes, (list, tuple)) or any(type(name) is not str for name in passes):
        raise ValueError("quantization passes must be an explicit list of registered names")
    if len(passes) != len(set(passes)) or set(passes) - set(_ORDER):
        raise ValueError("quantization passes contain a duplicate or unregistered name")
    return tuple(name for name in _ORDER if name in passes)


def compute_passes(passes: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Resolve a W8A8 caller's exact numerical pass selection.

    ``None`` starts with contraction/conv only. An explicitly selected optional
    quantization pass may alter that set; callers bind this returned effective
    policy before invoking ``apply_quant`` with an explicit pass list. The
    low-level ``apply_quant(None)`` historical all-six behavior is separate.
    """
    from merlin.compiler.lower.options.optional_passes import selected_quant_passes

    requested = _COMPUTE_DEFAULT if passes is None else _explicit_passes(passes)
    selected = selected_quant_passes(requested)
    return tuple(name for name in _ORDER if name in selected)


def apply_quant(
    module: Any,
    passes: list[str] | None = None,
    *,
    named_contraction: bool = False,
    prequant_gather: bool = False,
    report_out: dict[str, dict] | None = None,
    select: Callable[[Any], bool] | None = None,
) -> dict[str, int]:
    """Run the selected int8 quant passes IN CANONICAL ORDER (mutating ``module``). ``passes=None`` starts
    with all six historical passes and applies the optional-pass selection. An explicit list is
    authoritative, ordinarily from ``compute_passes``. Returns per-pass lowered-op counts.

    ``select`` is an ``(op) -> bool`` predicate restricting WHICH ops the selected passes may
    rewrite (default None = every op the pass recognizes, i.e. the shipped datapath). It makes the
    pass set's REACH measurable independently of its arithmetic: running only the contractions that
    descend from one framework construct answers "how much of the gap against a reference that
    quantizes fewer operations is policy rather than error". The predicate is the caller's policy —
    this module never assumes one.

    ``named_contraction`` asks the contraction pass to emit a mixed-type ``linalg.matmul`` for the
    canonical 2-D case instead of a ``linalg.generic``. Default False keeps the datapath
    byte-identical; True is what makes the 39 named-op transform-schedule levers reachable on int8
    at all (they match on the op NAME, and this pass otherwise leaves zero of them in the module).

    ``prequant_gather`` asks the contraction pass to quantize an activation BEFORE the pure gather
    that expands it (the ``quantize_before_gather`` feature). Default False keeps the datapath
    byte-identical; True is a genuine NUMERIC change (per-tensor instead of per-parallel-row
    activation scale) and is gated on the accuracy check like any other, never assumed bit-exact.

    ``report_out``, when given, collects each pass's own counters under its registry name — how many
    ops it rewrote, and for the two passes that can find nothing to do, WHY. A pass reporting only
    "0" cannot be told apart from a pass that cannot fire at all.
    """
    from merlin.compiler.lower.options.optional_passes import selected_quant_passes

    reg = registry()
    # The historical low-level default remains all six, with optional-pass
    # selection applied at this edge. An explicit list is already the caller's
    # selected and recorded numerical policy; do not widen it a second time.
    want = set(selected_quant_passes(_ORDER)) if passes is None else set(_explicit_passes(passes))
    out: dict[str, int] = {}
    for n in _ORDER:
        if n not in want:
            continue
        fn = reg[n].fn
        kw: dict[str, Any] = {}
        if select is not None:
            kw["select"] = select  # every pass honors the reach restriction
        # Only the contraction pass takes the flag; passing it to the others would couple every
        # quant pass to a decision that is not theirs to make.
        if named_contraction and n == "contraction_int8":
            kw["named_contraction"] = True
        if prequant_gather and n == "contraction_int8":
            kw["prequant_gather"] = True
        if report_out is not None and n in ("contraction_int8", "conv_int8"):
            kw["report_out"] = report_out.setdefault(n, {})
        out[n] = fn(module, **kw)
    return out
