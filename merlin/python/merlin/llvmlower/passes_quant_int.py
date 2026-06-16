"""Integer (W8A8) int8 compute datapath.

Today the int8 capture is WEIGHT-ONLY: ``quant_ext.dequantize_per_channel(i8, scale, zp)``
produces an f32 weight that feeds an f32 contraction — the matmul/attention runs in float.
This module rewrites every contraction (``linalg.matmul`` and the batched ``linalg.generic``
matmuls/attention from ``collapse_overrank_matmul``) into a real integer contraction:

  * each f32 activation operand is dynamically quantized to i8 (symmetric, per output-row:
    ``s = max|x|/127`` reduced over the operand's contraction dim, zero-point 0);
  * a ``dequantize_per_channel`` weight operand is used directly as i8 (its per-channel scale
    carried forward), the dequant dropped;
  * the contraction runs ``i8×i8→i32`` (clang ``-march=rv64gcv`` lowers it to widening
    ``vwmacc``), reusing the original indexing maps / iterator types;
  * the i32 accumulator is requantized by ``acc * prod(operand scales)`` back to f32.

The output stays f32 so downstream (nonlinears) is unchanged — this is the contraction stage
(M1/M2). Weight zero-point must be 0 (symmetric per-channel). Only contractions whose output
map is an identity projection of the parallel iterator dims are converted (matmul + the m2m
attention/batched generics); anything else is left for the f32 fallback (``lower_quant_ext``).
"""
from __future__ import annotations


def _is_dequant(op) -> bool:
    # ``op`` may be a Block (a block-argument owner) — guard with getattr.
    name = getattr(op, "op_name", None)
    return (getattr(op, "name", None) == "builtin.unregistered" and name is not None
            and name.data == "quant_ext.dequantize_per_channel")


def lower_contraction_int8(module) -> int:
    """Rewrite f32 contractions into i8×i8→i32 + dynamic act-quant + requant. Returns count."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, TensorType,
                                       f32, i8, i32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap, AffineDimExpr

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def contraction_view(op):
        """(ndim, in_maps[list of dim-index lists], out_dims, iters[bool reduction])
        for a linalg.matmul or a mul(f|i)+add(f|i)-bodied reduction generic; else None."""
        if op.name == "linalg.matmul":
            # standard 2-D: (m,n,k); lhs (m,k), rhs (k,n), out (m,n)
            return 3, [[0, 2], [2, 1]], [0, 1], [False, False, True]
        if op.name != "linalg.generic" or len(op.inputs) != 2:
            return None
        # conv / non-projection contractions (input map carries a stride/dilation affine
        # expression like (d2*16+d5)) are handled by lower_conv_int8, which preserves the
        # exact maps — the matmul rebuild below would drop the compound terms.
        if any(not isinstance(r, AffineDimExpr)
               for m in op.indexing_maps for r in m.data.results):
            return None
        iters = [getattr(a, "data", a) for a in op.iterator_types]
        if not any(i == red for i in iters):
            return None
        bn = [b.name for b in op.body.blocks[0].ops]
        if not (("arith.mulf" in bn and "arith.addf" in bn)
                or ("arith.muli" in bn and "arith.addi" in bn)):
            return None
        maps = list(op.indexing_maps)
        ndim = maps[0].data.num_dims
        def dimlist(m):
            return [r.position for r in m.data.results if isinstance(r, AffineDimExpr)]
        in_maps = [dimlist(maps[0]), dimlist(maps[1])]
        out_dims = dimlist(maps[-1])
        red_flags = [it == red for it in iters]
        # only identity-projection outputs (out = the parallel dims, in order)
        par_dims = [d for d in range(ndim) if not red_flags[d]]
        if out_dims != par_dims:
            return None
        return ndim, in_maps, out_dims, red_flags

    targets = []
    for op in module.walk():
        view = contraction_view(op)
        if view is None:
            continue
        out_t = op.results[0].type
        if not (isinstance(out_t, TensorType) and out_t.element_type == f32):
            continue
        if not all(isinstance(i.type, TensorType) for i in op.inputs):
            continue
        # at least one f32 operand to quantize (else nothing to do)
        if not any(i.type.element_type == f32 or _is_dequant(i.owner) for i in op.inputs):
            continue
        targets.append((op, view))

    n = 0
    for op, (ndim, in_maps, out_dims, red_flags) in targets:
        block = op.parent_block()
        out_t = op.results[0].type
        P = len(out_dims)                                # output / parallel rank
        d_par = AffineMap.identity(P).results
        # output-position of each iterator dim (parallel dims only)
        pos_of = {d: out_dims.index(d) for d in out_dims}
        pre = []                                         # ops to insert before `op`

        i8_inputs = []
        scale_vals = []                                  # (ssa, [output-positions]) per operand
        for oi, operand in enumerate(op.inputs):
            dims = in_maps[oi]                            # iterator dims this operand indexes
            red_pos = [p for p, d in enumerate(dims) if red_flags[d]]   # positions in operand
            par_pos = [p for p, d in enumerate(dims) if not red_flags[d]]
            par_outpos = [pos_of[dims[p]] for p in par_pos]             # -> requant out dims
            shp = list(operand.type.get_shape())
            sc_shape = [shp[p] for p in par_pos]
            sc_t = TensorType(f32, sc_shape)
            r = len(shp)

            if _is_dequant(operand.owner):
                # weight: use i8 source directly; per-channel scale (already over par dims)
                deq = operand.owner
                i8_inputs.append(deq.operands[0])
                scale_vals.append((deq.operands[1], par_outpos))
                continue
            if operand.type.element_type != f32:
                i8_inputs.append(operand)                # already integer
                scale_vals.append((None, par_outpos))
                continue

            # dynamic per-(parallel-row) quant of an f32 activation
            ident_r = AffineMap.identity(r).results
            red_iters = ArrayAttr([L.IteratorTypeAttr(red if p in red_pos else par)
                                   for p in range(r)])
            sc_map_in = amap(r, ident_r)
            sc_map_out = amap(r, [ident_r[p] for p in par_pos])
            # abs-max
            amx_e = tensor.EmptyOp((), sc_t)
            zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
            amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
            rb = Block(arg_types=[f32, f32]); a_in, acc_in = rb.args
            ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc_in)
            rb.add_ops([ab, mx, L.YieldOp(mx.result)])
            amx = L.GenericOp(inputs=(operand,), outputs=(amx_f.results[0],), body=Region(rb),
                              indexing_maps=ArrayAttr([sc_map_in, sc_map_out]),
                              iterator_types=red_iters, result_types=(sc_t,))
            # s = amax / 127
            sc_e = tensor.EmptyOp((), sc_t); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
            ident_p = AffineMap.identity(len(sc_shape)).results
            sb = Block(arg_types=[f32, f32]); s_in, _ = sb.args
            sd = arith.DivfOp(s_in, c127.results[0]); sb.add_ops([sd, L.YieldOp(sd.result)])
            sc = L.GenericOp(inputs=(amx.results[0],), outputs=(sc_e.results[0],), body=Region(sb),
                             indexing_maps=ArrayAttr([amap(len(sc_shape), ident_p),
                                                      amap(len(sc_shape), ident_p)]),
                             iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(sc_shape)),
                             result_types=(sc_t,))
            # quantize: q = fptosi(clamp(roundeven(x/s), -127, 127))
            i8_t = TensorType(i8, shp)
            q_e = tensor.EmptyOp((), i8_t); c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            qb = Block(arg_types=[f32, f32, i8]); xv, sv, _ = qb.args
            q1 = arith.DivfOp(xv, sv); q2 = mathd.RoundEvenOp(q1.result)
            q3 = arith.MinimumfOp(q2.result, c127.results[0])
            q4 = arith.MaximumfOp(q3.result, c127n.results[0]); q5 = arith.FPToSIOp(q4.result, i8)
            qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
            q = L.GenericOp(inputs=(operand, sc.results[0]), outputs=(q_e.results[0],),
                            body=Region(qb),
                            indexing_maps=ArrayAttr([amap(r, ident_r), sc_map_out, amap(r, ident_r)]),
                            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                            result_types=(i8_t,))
            pre += [amx_e, zero_f, amx_f, amx, sc_e, c127, sc, q_e, c127n, q]
            i8_inputs.append(q.results[0])
            scale_vals.append((sc.results[0], par_outpos))

        # i8×i8→i32 contraction (reuse maps/iters)
        acc_t = TensorType(i32, list(out_t.get_shape()))
        acc_e = tensor.EmptyOp((), acc_t); zi = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zi.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        mm_maps = ArrayAttr([amap(ndim, [AffineDimExpr(d) for d in in_maps[0]]),
                             amap(ndim, [AffineDimExpr(d) for d in in_maps[1]]),
                             amap(ndim, [AffineDimExpr(d) for d in out_dims])])
        mm_iters = ArrayAttr([L.IteratorTypeAttr(red if red_flags[d] else par) for d in range(ndim)])
        mb = Block(arg_types=[i8, i8, i32]); av, bv, acc = mb.args
        ea = arith.ExtSIOp(av, i32); eb = arith.ExtSIOp(bv, i32)
        pm = arith.MuliOp(ea.result, eb.result); pa = arith.AddiOp(pm.result, acc)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        i8mm = L.GenericOp(inputs=tuple(i8_inputs), outputs=(acc_f.results[0],), body=Region(mb),
                           indexing_maps=mm_maps, iterator_types=mm_iters, result_types=(acc_t,))

        # requant: out_f32 = sitofp(acc) * scale_lhs * scale_rhs
        sc_inputs, sc_maps = [], [amap(P, d_par)]
        for ssa, outpos in scale_vals:
            if ssa is None:
                continue
            sc_inputs.append(ssa)
            sc_maps.append(amap(P, [d_par[p] for p in outpos]))
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32] + [f32] * len(sc_inputs) + [f32])
        accv = wb.args[0]; svs = wb.args[1:1 + len(sc_inputs)]
        cur = arith.SIToFPOp(accv, f32); ops_w = [cur]
        for sv in svs:
            m = arith.MulfOp(cur.result, sv); ops_w.append(m); cur = m
        wb.add_ops(ops_w + [L.YieldOp(cur.result)])
        requant = L.GenericOp(inputs=(i8mm.results[0], *sc_inputs), outputs=(out_e.results[0],),
                              body=Region(wb),
                              indexing_maps=ArrayAttr(sc_maps + [amap(P, d_par)]),
                              iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * P),
                              result_types=(out_t,))
        for k, v in op.attributes.items():
            if k.startswith("prov."):
                requant.attributes[k] = v

        for new in pre + [acc_e, zi, acc_f, i8mm, out_e, requant]:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(op)
        n += 1
    return n


# Back-compat alias (M1 name).
lower_matmul_int8 = lower_contraction_int8


def lower_conv_int8(module) -> int:
    """Rewrite an f32 conv (``linalg.generic`` whose input map carries stride/dilation affine
    expressions, ``prov.op = conv2d``) into an i8×i8→i32 conv: the activation is dynamically
    quantized **per-tensor** (a conv pixel feeds many outputs, so a per-output-row act scale is
    ill-defined — one scalar ``s_a = max|x|/127`` over the whole activation), the weight is used
    as i8 with its per-output-channel scale, the contraction keeps the **exact** original
    indexing maps + iterator types (so the RVV schedule can tile/vectorize it), and the i32
    accumulator is requantized by ``acc * s_a * s_w[out_channel]``. Returns count."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, TensorType,
                                       f32, i8, i32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap, AffineDimExpr

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    targets = []
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.inputs) != 2 or not op.body.blocks:
            continue
        maps = list(op.indexing_maps)
        if not any(not isinstance(r, AffineDimExpr)
                   for m in maps for r in m.data.results):
            continue                                     # not a conv (no compound map term)
        out_t = op.results[0].type
        if not (isinstance(out_t, TensorType) and out_t.element_type == f32):
            continue
        bn = [b.name for b in op.body.blocks[0].ops]
        if not (("arith.mulf" in bn and "arith.addf" in bn)
                or ("arith.muli" in bn and "arith.addi" in bn)):
            continue
        # operand 0 = activation (compound map); operand 1 = weight (dequant i8, or plain f32 —
        # torchao leaves conv weights f32, so we dynamically per-output-channel quantize them).
        act, wt = op.inputs[0], op.inputs[1]
        if not (isinstance(act.type, TensorType) and act.type.element_type == f32):
            continue
        if not (_is_dequant(wt.owner)
                or (isinstance(wt.type, TensorType) and wt.type.element_type == f32)):
            continue
        targets.append(op)

    n = 0
    for op in targets:
        block = op.parent_block()
        maps = list(op.indexing_maps)
        iters = [getattr(a, "data", a) for a in op.iterator_types]
        ndim = maps[0].data.num_dims
        red_flags = [it == red for it in iters]
        act, wt = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        ash = list(act.type.get_shape()); r = len(ash)
        pre = []

        # --- weight -> i8 + per-output-channel scale s_w ---
        wt_dims = [rr.position for rr in maps[1].data.results]   # iterator dim per weight axis
        wt_keep = [ax for ax, d in enumerate(wt_dims) if not red_flags[d]]   # parallel (out-ch) axes
        if _is_dequant(wt.owner):
            deq = wt.owner
            wt_i8, s_w = deq.operands[0], deq.operands[1]        # already i8 + per-channel scale
        else:
            # dynamically per-output-channel quantize the f32 weight (reduce over its kernel axes)
            wsh = list(wt.type.get_shape()); wr = len(wsh)
            ws_shape = [wsh[ax] for ax in wt_keep]
            ws_t = TensorType(f32, ws_shape)
            id_w = AffineMap.identity(wr).results
            keep_res = [id_w[ax] for ax in wt_keep]
            w_amx_e = tensor.EmptyOp((), ws_t); wzero = arith.ConstantOp(FloatAttr(0.0, f32))
            w_amx_f = L.FillOp(inputs=[wzero.results[0]], outputs=[w_amx_e.results[0]], res=[ws_t])
            wrb = Block(arg_types=[f32, f32]); wa, wacc = wrb.args
            wab = mathd.AbsFOp(wa); wmx = arith.MaximumfOp(wab.result, wacc)
            wrb.add_ops([wab, wmx, L.YieldOp(wmx.result)])
            w_amx = L.GenericOp(inputs=(wt,), outputs=(w_amx_f.results[0],), body=Region(wrb),
                                indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res)]),
                                iterator_types=ArrayAttr(
                                    [L.IteratorTypeAttr(par if ax in wt_keep else red)
                                     for ax in range(wr)]),
                                result_types=(ws_t,))
            ws_e = tensor.EmptyOp((), ws_t); wc127 = arith.ConstantOp(FloatAttr(127.0, f32))
            id_k = AffineMap.identity(len(ws_shape)).results
            wsb = Block(arg_types=[f32, f32]); ws_in, _ = wsb.args
            wsd = arith.DivfOp(ws_in, wc127.results[0]); wsb.add_ops([wsd, L.YieldOp(wsd.result)])
            s_w_g = L.GenericOp(inputs=(w_amx.results[0],), outputs=(ws_e.results[0],),
                                body=Region(wsb),
                                indexing_maps=ArrayAttr([amap(len(ws_shape), id_k),
                                                         amap(len(ws_shape), id_k)]),
                                iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(ws_shape)),
                                result_types=(ws_t,))
            s_w = s_w_g.results[0]
            wi8_t = TensorType(i8, wsh); wq_e = tensor.EmptyOp((), wi8_t)
            wc127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            wqb = Block(arg_types=[f32, f32, i8]); wxv, wsv, _ = wqb.args
            wq1 = arith.DivfOp(wxv, wsv); wq2 = mathd.RoundEvenOp(wq1.result)
            wq3 = arith.MinimumfOp(wq2.result, wc127.results[0])
            wq4 = arith.MaximumfOp(wq3.result, wc127n.results[0]); wq5 = arith.FPToSIOp(wq4.result, i8)
            wqb.add_ops([wq1, wq2, wq3, wq4, wq5, L.YieldOp(wq5.result)])
            wq = L.GenericOp(inputs=(wt, s_w_g.results[0]), outputs=(wq_e.results[0],),
                             body=Region(wqb),
                             indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res),
                                                      amap(wr, id_w)]),
                             iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * wr),
                             result_types=(wi8_t,))
            wt_i8 = wq.results[0]
            pre += [w_amx_e, wzero, w_amx_f, w_amx, ws_e, wc127, s_w_g, wq_e, wc127n, wq]

        # --- per-tensor activation scale: s_a = max|act| / 127 (rank-0 tensor<f32>) ---
        sc_t = TensorType(f32, [])
        ident_r = AffineMap.identity(r).results
        amx_e = tensor.EmptyOp((), sc_t); zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
        amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
        rb = Block(arg_types=[f32, f32]); a_in, acc_in = rb.args
        ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc_in)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amx = L.GenericOp(inputs=(act,), outputs=(amx_f.results[0],), body=Region(rb),
                          indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, [])]),
                          iterator_types=ArrayAttr([L.IteratorTypeAttr(red)] * r),
                          result_types=(sc_t,))
        sc_e = tensor.EmptyOp((), sc_t); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); s_in, _ = sb.args
        sd = arith.DivfOp(s_in, c127.results[0]); sb.add_ops([sd, L.YieldOp(sd.result)])
        s_a = L.GenericOp(inputs=(amx.results[0],), outputs=(sc_e.results[0],), body=Region(sb),
                          indexing_maps=ArrayAttr([amap(0, []), amap(0, [])]),
                          iterator_types=ArrayAttr([]), result_types=(sc_t,))
        # --- quantize activation: q = fptosi(clamp(roundeven(x/s_a), ±127)) ---
        i8_t = TensorType(i8, ash); q_e = tensor.EmptyOp((), i8_t)
        c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qb = Block(arg_types=[f32, f32, i8]); xv, sv, _ = qb.args
        q1 = arith.DivfOp(xv, sv); q2 = mathd.RoundEvenOp(q1.result)
        q3 = arith.MinimumfOp(q2.result, c127.results[0])
        q4 = arith.MaximumfOp(q3.result, c127n.results[0]); q5 = arith.FPToSIOp(q4.result, i8)
        qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
        q = L.GenericOp(inputs=(act, s_a.results[0]), outputs=(q_e.results[0],), body=Region(qb),
                        indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, []), amap(r, ident_r)]),
                        iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                        result_types=(i8_t,))
        pre += [amx_e, zero_f, amx_f, amx, sc_e, c127, s_a, q_e, c127n, q]

        # --- i8×i8→i32 conv: EXACT original maps + iterators preserved ---
        acc_t = TensorType(i32, list(out_t.get_shape()))
        acc_e = tensor.EmptyOp((), acc_t); zi = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zi.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        mb = Block(arg_types=[i8, i8, i32]); av, bv, acc = mb.args
        ea = arith.ExtSIOp(av, i32); eb = arith.ExtSIOp(bv, i32)
        pm = arith.MuliOp(ea.result, eb.result); pa = arith.AddiOp(pm.result, acc)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        i8cv = L.GenericOp(inputs=(q.results[0], wt_i8), outputs=(acc_f.results[0],),
                           body=Region(mb), indexing_maps=ArrayAttr(maps),
                           iterator_types=op.iterator_types, result_types=(acc_t,))

        # --- requant: out_f32[..] = sitofp(acc) * s_a * s_w[out_channel] ---
        # output map is a plain identity projection of the parallel dims.
        out_dims = [rr.position for rr in maps[-1].data.results]
        P = len(out_dims); d_par = AffineMap.identity(P).results
        # weight's (single) parallel iterator dim -> output channel position
        wt_dims = [rr.position for rr in maps[1].data.results]
        wt_par = [d for d in wt_dims if not red_flags[d]]
        wpos = out_dims.index(wt_par[0])
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32, f32, f32, f32]); accv, sav, swv, _ = wb.args
        cur = arith.SIToFPOp(accv, f32); m1 = arith.MulfOp(cur.result, sav)
        m2 = arith.MulfOp(m1.result, swv)
        wb.add_ops([cur, m1, m2, L.YieldOp(m2.result)])
        requant = L.GenericOp(inputs=(i8cv.results[0], s_a.results[0], s_w),
                              outputs=(out_e.results[0],), body=Region(wb),
                              indexing_maps=ArrayAttr([amap(P, d_par), amap(P, []),
                                                       amap(P, [d_par[wpos]]), amap(P, d_par)]),
                              iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * P),
                              result_types=(out_t,))
        for k, v in op.attributes.items():
            if k.startswith("prov."):
                requant.attributes[k] = v

        for new in pre + [acc_e, zi, acc_f, i8cv, out_e, requant]:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(op)
        n += 1
    return n


# I-BERT integer-exp constants (2nd-order poly exp(p) ~ a(p+b)^2 + c on p in [-ln2,0]).
_IEXP_A, _IEXP_B, _IEXP_C = 0.35815147, 1.35330989, 0.34401959
_IEXP_SH = 24      # fixed-point fraction bits for the poly
_IEXP_RSH = 20     # fraction bits for the 1/qln2 reciprocal (avoids a per-element integer divide)


def lower_softmax_int(module) -> int:
    """Replace softmax's ``math.exp`` with an integer (I-BERT) exp: a per-row dynamic scale,
    then a fixed-point 2nd-order polynomial + power-of-two shift evaluated in integer arithmetic,
    producing the same f32 exp values the downstream sum/divide consume. The transcendental
    ``math.exp`` is gone. Anchors on each exp generic (no fragile region detection). Returns count.

    Numerically validated vs f32 softmax: cos > 0.995 (mean 0.999) over L=8..968, scale 0.5..8.
    """
    import math as _m
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))
    ln2 = _m.log(2.0)

    def _is_softmax_exp(op):
        # a bare exp generic whose input is the max-subtraction (S - rowmax) — the softmax
        # signature. Guards against rewriting non-softmax exps (whose input may be > 0, which
        # the <=0 i-exp + [-30,0] clamp would corrupt).
        if not (op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks
                and [b.name for b in op.body.blocks[0].ops] == ["math.exp", "linalg.yield"]):
            return False
        src = op.inputs[0].owner
        return (getattr(src, "name", None) == "linalg.generic" and src.body.blocks
                and any(b.name == "arith.subf" for b in src.body.blocks[0].ops))

    targets = [op for op in module.walk() if _is_softmax_exp(op)]
    n = 0
    for op in targets:
        xs = op.inputs[0]                          # x = scores - rowmax, f32 [.., L], <= 0
        st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        Rt = TensorType(f32, shp[:-1]); It = TensorType(i32, shp)
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR); row = amap(R, idR[:R - 1])
        idp = AffineMap.identity(R - 1).results
        red_it = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(red)])
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        par_rowit = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1))
        new = []

        # Clamp xs to [-30, 0]: attention masks make xs = -inf at masked positions, which would
        # blow srow up to +inf and make q = -inf/inf = NaN. exp(-30) ~ 1e-13 ~ 0, matching the
        # masked-out intent, and real (unmasked) logits-minus-max rarely fall below -30.
        xc_e = tensor.EmptyOp((), st); c30n = arith.ConstantOp(FloatAttr(-30.0, f32))
        cb = Block(arg_types=[f32, f32]); xa, _ = cb.args
        cm = arith.MaximumfOp(xa, c30n.results[0]); cb.add_ops([cm, L.YieldOp(cm.result)])
        xclamp = L.GenericOp(inputs=(xs,), outputs=(xc_e.results[0],), body=Region(cb),
                             indexing_maps=ArrayAttr([full, full]), iterator_types=all_par,
                             result_types=(st,))
        new += [xc_e, c30n, xclamp]
        xs = xclamp.results[0]

        # srow[..] = max(-xs over last dim) / 127   (xs <= 0, so this is max|xs|/127)
        se = tensor.EmptyOp((), Rt); z0 = arith.ConstantOp(FloatAttr(0.0, f32))
        sf = L.FillOp(inputs=[z0.results[0]], outputs=[se.results[0]], res=[Rt])
        rb = Block(arg_types=[f32, f32]); a_in, acc = rb.args
        ng = arith.NegfOp(a_in); mx = arith.MaximumfOp(ng.result, acc)
        rb.add_ops([ng, mx, L.YieldOp(mx.result)])
        nmax = L.GenericOp(inputs=(xs,), outputs=(sf.results[0],), body=Region(rb),
                           indexing_maps=ArrayAttr([full, row]), iterator_types=red_it,
                           result_types=(Rt,))
        sre = tensor.EmptyOp((), Rt); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); nm, _ = sb.args
        sd = arith.DivfOp(nm, c127.results[0])
        eps = arith.ConstantOp(FloatAttr(1e-6, f32))      # floor: all-equal rows (xs==0) -> no div0
        sfl = arith.MaximumfOp(sd.result, eps.results[0])
        sb.add_ops([sd, eps, sfl, L.YieldOp(sfl.result)])
        srow = L.GenericOp(inputs=(nmax.results[0],), outputs=(sre.results[0],), body=Region(sb),
                           indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                           iterator_types=par_rowit, result_types=(Rt,))
        new += [se, z0, sf, nmax, sre, c127, srow]

        def per_row(fn, elem):
            ot = TensorType(elem, shp[:-1]); e = tensor.EmptyOp((), ot)
            bb = Block(arg_types=[f32, elem]); s_in, _ = bb.args
            cops, res = fn(s_in); bb.add_ops(cops + [L.YieldOp(res)])
            g = L.GenericOp(inputs=(srow.results[0],), outputs=(e.results[0],), body=Region(bb),
                            indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                            iterator_types=par_rowit, result_types=(ot,))
            new.extend([e, g]); return g.results[0]

        def _qln2(s):
            c = arith.ConstantOp(FloatAttr(ln2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); o = arith.ConstantOp(FloatAttr(1.0, f32))
            mx2 = arith.MaximumfOp(r.result, o.results[0]); ci = arith.FPToSIOp(mx2.result, i32)
            return [c, d, r, o, mx2, ci], ci.result
        def _bq(s):
            c = arith.ConstantOp(FloatAttr(_IEXP_B, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, d, r, ci], ci.result
        def _A(s):
            c = arith.ConstantOp(FloatAttr(_IEXP_A * (1 << _IEXP_SH), f32))
            s2 = arith.MulfOp(s, s); m2 = arith.MulfOp(c.results[0], s2.result)
            r = mathd.RoundEvenOp(m2.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, s2, m2, r, ci], ci.result
        def _recip(s):
            c = arith.ConstantOp(FloatAttr(ln2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); o = arith.ConstantOp(FloatAttr(1.0, f32))
            ql = arith.MaximumfOp(r.result, o.results[0])
            cr = arith.ConstantOp(FloatAttr(float(1 << _IEXP_RSH), f32))
            rc = arith.DivfOp(cr.results[0], ql.result); rr = mathd.RoundEvenOp(rc.result)
            ci = arith.FPToSIOp(rr.result, i32)
            return [c, d, r, o, ql, cr, rc, rr, ci], ci.result
        qln2 = per_row(_qln2, i32); bq = per_row(_bq, i32)
        A = per_row(_A, i32); recip = per_row(_recip, i32)

        # q = round(xs / srow) -> i32  (xs/srow in [-127, 0])
        qe = tensor.EmptyOp((), It)
        qb = Block(arg_types=[f32, f32, i32]); xv, sv, _ = qb.args
        qd = arith.DivfOp(xv, sv); qr = mathd.RoundEvenOp(qd.result); qc = arith.FPToSIOp(qr.result, i32)
        qb.add_ops([qd, qr, qc, L.YieldOp(qc.result)])
        q = L.GenericOp(inputs=(xs, srow.results[0]), outputs=(qe.results[0],), body=Region(qb),
                        indexing_maps=ArrayAttr([full, row, full]), iterator_types=all_par,
                        result_types=(It,))
        new += [qe, q]

        # integer exp poly: z=((-q)*recip)>>RSH; r=q+z*qln2; t=r+bq; e=(A*t^2+C)>>z; exp=sitofp(e)/2^SH
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[i32, i32, i32, i32, i32, f32]); qv, ql2, bqv, av, rcp, _ = eb.args
        c0 = arith.ConstantOp.from_int_and_width(0, 32); nq = arith.SubiOp(c0.result, qv)
        zr = arith.MuliOp(nq.result, rcp); rs = arith.ConstantOp(IntegerAttr(_IEXP_RSH, i32))
        z = arith.ShRSIOp(zr.result, rs.results[0])
        zql = arith.MuliOp(z.result, ql2); rr = arith.AddiOp(qv, zql.result); t = arith.AddiOp(rr.result, bqv)
        t64 = arith.ExtSIOp(t.result, i64); a64 = arith.ExtSIOp(av, i64)
        tt = arith.MuliOp(t64.result, t64.result); att = arith.MuliOp(a64.result, tt.result)
        Ci = arith.ConstantOp(IntegerAttr(int(round(_IEXP_C * (1 << _IEXP_SH))), i64))
        ep = arith.AddiOp(att.result, Ci.results[0])
        z64 = arith.ExtSIOp(z.result, i64); e = arith.ShRSIOp(ep.result, z64.result)
        ef = arith.SIToFPOp(e.result, f32); inv = arith.ConstantOp(FloatAttr(1.0 / (1 << _IEXP_SH), f32))
        exf = arith.MulfOp(ef.result, inv.results[0])
        eb.add_ops([c0, nq, zr, rs, z, zql, rr, t, t64, a64, tt, att, Ci, ep, z64, e, ef, inv, exf,
                    L.YieldOp(exf.result)])
        eg = L.GenericOp(inputs=(q.results[0], qln2, bq, A, recip), outputs=(ee.results[0],),
                         body=Region(eb),
                         indexing_maps=ArrayAttr([full, row, row, row, row, full]),
                         iterator_types=all_par, result_types=(st,))
        new += [ee, eg]

        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(eg.results[0])
        block.detach_op(op)
        n += 1
    return n


# I-BERT i-GELU constants: erf(z) ~ sgn(z)[a(min(|z|,-b)+b)^2 + 1], z = x/sqrt(2).
_IGELU_A, _IGELU_B = -0.2888, -1.769


def lower_gelu_int(module) -> int:
    """Replace GELU's ``math.erf`` with an integer (I-BERT) i-GELU: per-row dynamic scale, the
    erf approximated by a fixed-point 2nd-order polynomial in integer arithmetic, then
    ``0.5*x*(1+erf)``. The transcendental ``math.erf`` is gone. Anchors on each erf generic.
    Numerically validated vs exact GELU: cos > 0.9999. Returns count."""
    import math as _m
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par = L.IteratorType.PARALLEL; red = L.IteratorType.REDUCTION
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))
    sqrt2 = _m.sqrt(2.0)
    SH = 24

    targets = [op for op in module.walk()
               if op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks
               and any(b.name == "math.erf" for b in op.body.blocks[0].ops)]
    n = 0
    for op in targets:
        x = op.inputs[0]; st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        Rt = TensorType(f32, shp[:-1])
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR); row = amap(R, idR[:R - 1]); idp = AffineMap.identity(R - 1).results
        red_it = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(red)])
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        par_rowit = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1))
        new = []

        # sx[..] = max|x| over last / 127
        ae = tensor.EmptyOp((), Rt); z0 = arith.ConstantOp(FloatAttr(0.0, f32))
        af = L.FillOp(inputs=[z0.results[0]], outputs=[ae.results[0]], res=[Rt])
        rb = Block(arg_types=[f32, f32]); a_in, acc = rb.args
        ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amax = L.GenericOp(inputs=(x,), outputs=(af.results[0],), body=Region(rb),
                           indexing_maps=ArrayAttr([full, row]), iterator_types=red_it, result_types=(Rt,))
        sre = tensor.EmptyOp((), Rt); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); nm, _ = sb.args
        sd = arith.DivfOp(nm, c127.results[0]); eps = arith.ConstantOp(FloatAttr(1e-12, f32))
        sfl = arith.MaximumfOp(sd.result, eps.results[0]); sb.add_ops([sd, eps, sfl, L.YieldOp(sfl.result)])
        sx = L.GenericOp(inputs=(amax.results[0],), outputs=(sre.results[0],), body=Region(sb),
                         indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                         iterator_types=par_rowit, result_types=(Rt,))
        new += [ae, z0, af, amax, sre, c127, sx]

        def per_row(fn, elem):
            ot = TensorType(elem, shp[:-1]); e = tensor.EmptyOp((), ot)
            bb = Block(arg_types=[f32, elem]); s_in, _ = bb.args
            cops, res = fn(s_in); bb.add_ops(cops + [L.YieldOp(res)])
            g = L.GenericOp(inputs=(sx.results[0],), outputs=(e.results[0],), body=Region(bb),
                            indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                            iterator_types=par_rowit, result_types=(ot,))
            new.extend([e, g]); return g.results[0]

        # qb = round(-B*sqrt2 / sx) ; Aq = round(A * sx^2/2 * 2^SH)
        def _qb(s):
            c = arith.ConstantOp(FloatAttr(-_IGELU_B * sqrt2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, d, r, ci], ci.result
        def _A(s):
            c = arith.ConstantOp(FloatAttr(_IGELU_A * 0.5 * (1 << SH), f32))
            s2 = arith.MulfOp(s, s); m2 = arith.MulfOp(c.results[0], s2.result)
            r = mathd.RoundEvenOp(m2.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, s2, m2, r, ci], ci.result
        qb = per_row(_qb, i32); Aq = per_row(_A, i32)

        # per-element integer i-GELU
        ge = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32, i32, i32, f32])   # x, sx, qb, Aq, out
        xv, sv, qbv, av, _ = eb.args
        c127f = arith.ConstantOp(FloatAttr(127.0, f32)); c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qd = arith.DivfOp(xv, sv); qr = mathd.RoundEvenOp(qd.result)
        qcl = arith.MinimumfOp(qr.result, c127f.results[0]); qcl2 = arith.MaximumfOp(qcl.result, c127n.results[0])
        q = arith.FPToSIOp(qcl2.result, i32)
        # |q| via arith (math.absi has no LLVM lowering in this pipeline): maxsi(q, -q)
        zi0 = arith.ConstantOp.from_int_and_width(0, 32); nq = arith.SubiOp(zi0.result, q.result)
        aq = arith.MaxSIOp(q.result, nq.result)
        qc = arith.MinSIOp(aq.result, qbv); t = arith.SubiOp(qc.result, qbv)
        t64 = arith.ExtSIOp(t.result, i64); a64 = arith.ExtSIOp(av, i64)
        tt = arith.MuliOp(t64.result, t64.result); att = arith.MuliOp(a64.result, tt.result)
        Ci = arith.ConstantOp(IntegerAttr(1 << SH, i64)); poly = arith.AddiOp(att.result, Ci.results[0])
        polyf = arith.SIToFPOp(poly.result, f32)
        inv = arith.ConstantOp(FloatAttr(1.0 / (1 << SH), f32)); pf = arith.MulfOp(polyf.result, inv.results[0])
        # sgn(x): x<0 -> -1 else +1
        zc = arith.ConstantOp(FloatAttr(0.0, f32)); lt = arith.CmpfOp(xv, zc.results[0], "olt")
        nf = arith.ConstantOp(FloatAttr(-1.0, f32)); pf1 = arith.ConstantOp(FloatAttr(1.0, f32))
        sgn = arith.SelectOp(lt.result, nf.results[0], pf1.results[0])
        erf = arith.MulfOp(sgn.result, pf.result)
        one = arith.ConstantOp(FloatAttr(1.0, f32)); ope = arith.AddfOp(one.results[0], erf.result)
        half = arith.ConstantOp(FloatAttr(0.5, f32)); hx = arith.MulfOp(half.results[0], xv)
        g = arith.MulfOp(hx.result, ope.result)
        eb.add_ops([c127f, c127n, qd, qr, qcl, qcl2, q, zi0, nq, aq, qc, t, t64, a64, tt, att, Ci, poly,
                    polyf, inv, pf, zc, lt, nf, pf1, sgn, erf, one, ope, half, hx, g, L.YieldOp(g.result)])
        gelu = L.GenericOp(inputs=(x, sx.results[0], qb, Aq), outputs=(ge.results[0],), body=Region(eb),
                           indexing_maps=ArrayAttr([full, row, row, row, full]),
                           iterator_types=all_par, result_types=(st,))
        new += [ge, gelu]
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(gelu.results[0])
        block.detach_op(op)
        n += 1
    return n


def lower_silu_int(module) -> int:
    """Replace the logistic ``sigmoid`` generic (``1/(1+exp(-x))``, the SiLU/swish nonlinear)
    with an integer (I-BERT) version: a per-row dynamic scale, the I-BERT integer-exp evaluated
    on ``-|x| <= 0`` (the same fixed-point poly + power-of-two shift as softmax), then the
    stable logistic ``sig = (x>=0 ? 1 : e) / (1 + e)`` in f32 (the divide is RVV ``vfdiv`` — no
    integer per-lane divide). The ``math.exp`` is gone. Anchors on each ``prov.op = sigmoid``
    generic; the downstream ``x * sigmoid`` multiply stays. Validated vs f32 SiLU: cos ~ 1.0.
    Returns count."""
    import math as _m
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))
    ln2 = _m.log(2.0)

    def _is_sigmoid(op):
        if not (op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks):
            return False
        hint = op.attributes.get("prov.op")
        return (getattr(hint, "data", None) == "sigmoid"
                and any(b.name == "math.exp" for b in op.body.blocks[0].ops))

    targets = [op for op in module.walk() if _is_sigmoid(op)]
    n = 0
    for op in targets:
        x = op.inputs[0]; st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        Rt = TensorType(f32, shp[:-1]); It = TensorType(i32, shp)
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR); row = amap(R, idR[:R - 1]); idp = AffineMap.identity(R - 1).results
        red_it = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(red)])
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        par_rowit = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1))
        new = []

        # srow[..] = max|x| over last dim / 127
        se = tensor.EmptyOp((), Rt); z0 = arith.ConstantOp(FloatAttr(0.0, f32))
        sf = L.FillOp(inputs=[z0.results[0]], outputs=[se.results[0]], res=[Rt])
        rb = Block(arg_types=[f32, f32]); a_in, acc = rb.args
        ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amax = L.GenericOp(inputs=(x,), outputs=(sf.results[0],), body=Region(rb),
                           indexing_maps=ArrayAttr([full, row]), iterator_types=red_it,
                           result_types=(Rt,))
        sre = tensor.EmptyOp((), Rt); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); nm, _ = sb.args
        sd = arith.DivfOp(nm, c127.results[0]); eps = arith.ConstantOp(FloatAttr(1e-6, f32))
        sfl = arith.MaximumfOp(sd.result, eps.results[0]); sb.add_ops([sd, eps, sfl, L.YieldOp(sfl.result)])
        srow = L.GenericOp(inputs=(amax.results[0],), outputs=(sre.results[0],), body=Region(sb),
                           indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                           iterator_types=par_rowit, result_types=(Rt,))
        new += [se, z0, sf, amax, sre, c127, srow]

        def per_row(fn, elem):
            ot = TensorType(elem, shp[:-1]); e = tensor.EmptyOp((), ot)
            bb = Block(arg_types=[f32, elem]); s_in, _ = bb.args
            cops, res = fn(s_in); bb.add_ops(cops + [L.YieldOp(res)])
            g = L.GenericOp(inputs=(srow.results[0],), outputs=(e.results[0],), body=Region(bb),
                            indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                            iterator_types=par_rowit, result_types=(ot,))
            new.extend([e, g]); return g.results[0]

        def _qln2(s):
            c = arith.ConstantOp(FloatAttr(ln2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); o = arith.ConstantOp(FloatAttr(1.0, f32))
            mx2 = arith.MaximumfOp(r.result, o.results[0]); ci = arith.FPToSIOp(mx2.result, i32)
            return [c, d, r, o, mx2, ci], ci.result
        def _bq(s):
            c = arith.ConstantOp(FloatAttr(_IEXP_B, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, d, r, ci], ci.result
        def _A(s):
            c = arith.ConstantOp(FloatAttr(_IEXP_A * (1 << _IEXP_SH), f32))
            s2 = arith.MulfOp(s, s); m2 = arith.MulfOp(c.results[0], s2.result)
            r = mathd.RoundEvenOp(m2.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, s2, m2, r, ci], ci.result
        def _recip(s):
            c = arith.ConstantOp(FloatAttr(ln2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); o = arith.ConstantOp(FloatAttr(1.0, f32))
            ql = arith.MaximumfOp(r.result, o.results[0])
            cr = arith.ConstantOp(FloatAttr(float(1 << _IEXP_RSH), f32))
            rc = arith.DivfOp(cr.results[0], ql.result); rr = mathd.RoundEvenOp(rc.result)
            ci = arith.FPToSIOp(rr.result, i32)
            return [c, d, r, o, ql, cr, rc, rr, ci], ci.result
        qln2 = per_row(_qln2, i32); bq = per_row(_bq, i32)
        A = per_row(_A, i32); recip = per_row(_recip, i32)

        # sigmoid body: e = iexp(-|x|); sig = (x>=0 ? 1 : e)/(1+e)
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32, i32, i32, i32, i32, f32])
        xv, sv, ql2, bqv, av, rcp, _ = eb.args
        # q = -round(|x|/s)  (<= 0, the i-exp domain)
        axv = mathd.AbsFOp(xv); qd = arith.DivfOp(axv.result, sv); qr = mathd.RoundEvenOp(qd.result)
        c127f = arith.ConstantOp(FloatAttr(127.0, f32)); qcl = arith.MinimumfOp(qr.result, c127f.results[0])
        qa = arith.FPToSIOp(qcl.result, i32)
        c0 = arith.ConstantOp.from_int_and_width(0, 32); qv = arith.SubiOp(c0.result, qa.result)
        # i-exp poly (identical to softmax): nq=-qv; z=(nq*recip)>>RSH; rr=qv+z*qln2; t=rr+bq;
        #   e=(A*t^2 + C)>>z ; ef = sitofp(e)/2^SH
        nq = arith.SubiOp(c0.result, qv.result)
        zr = arith.MuliOp(nq.result, rcp); rs = arith.ConstantOp(IntegerAttr(_IEXP_RSH, i32))
        z = arith.ShRSIOp(zr.result, rs.results[0])
        zql = arith.MuliOp(z.result, ql2); rrp = arith.AddiOp(qv.result, zql.result); t = arith.AddiOp(rrp.result, bqv)
        t64 = arith.ExtSIOp(t.result, i64); a64 = arith.ExtSIOp(av, i64)
        tt = arith.MuliOp(t64.result, t64.result); att = arith.MuliOp(a64.result, tt.result)
        Ci = arith.ConstantOp(IntegerAttr(int(round(_IEXP_C * (1 << _IEXP_SH))), i64))
        ep = arith.AddiOp(att.result, Ci.results[0])
        z64 = arith.ExtSIOp(z.result, i64); e = arith.ShRSIOp(ep.result, z64.result)
        ef = arith.SIToFPOp(e.result, f32); inv = arith.ConstantOp(FloatAttr(1.0 / (1 << _IEXP_SH), f32))
        exf = arith.MulfOp(ef.result, inv.results[0])
        # sig = num/denom, num = (x>=0 ? 1 : e), denom = 1+e
        one = arith.ConstantOp(FloatAttr(1.0, f32)); denom = arith.AddfOp(one.results[0], exf.result)
        zc = arith.ConstantOp(FloatAttr(0.0, f32)); ge0 = arith.CmpfOp(xv, zc.results[0], "oge")
        num = arith.SelectOp(ge0.result, one.results[0], exf.result)
        sig = arith.DivfOp(num.result, denom.result)
        eb.add_ops([axv, qd, qr, c127f, qcl, qa, c0, qv, nq, zr, rs, z, zql, rrp, t, t64, a64, tt,
                    att, Ci, ep, z64, e, ef, inv, exf, one, denom, zc, ge0, num, sig, L.YieldOp(sig.result)])
        sg = L.GenericOp(inputs=(x, srow.results[0], qln2, bq, A, recip), outputs=(ee.results[0],),
                         body=Region(eb),
                         indexing_maps=ArrayAttr([full, row, row, row, row, row, full]),
                         iterator_types=all_par, result_types=(st,))
        new += [ee, sg]
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(sg.results[0])
        block.detach_op(op)
        n += 1
    return n


# Fast inverse square root magic constant (Quake): y0 = bitcast(0x5f3759df - (bits(v)>>1)).
_RSQRT_MAGIC = 0x5f3759df


def lower_rsqrt_int(module) -> int:
    """Replace RMSNorm/LayerNorm's transcendental ``math.rsqrt`` with the fast inverse square
    root: a bit-hack initial guess (an integer subtract + shift on the f32 bit pattern) refined
    by Newton steps in f32 (``y = y*(1.5 - 0.5*v*y*y)``). No libm call, no integer per-lane
    divide — only ``arith`` integer ops + f32 ``vfmul``/``vfsub`` (all RVV-vectorizable, the
    plan's sanctioned f32-normalization path). Operates in-place on each ``math.rsqrt`` op inside
    its host generic body. Validated vs 1/sqrt: rmsnorm cos ~ 1.0 (4 Newton iters). Returns count."""
    from xdsl.dialects import arith
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i32

    NEWTON = 4
    rsqrts = [op for op in module.walk() if op.name == "math.rsqrt" and op.operands[0].type == f32]
    n = 0
    for op in rsqrts:
        v = op.operands[0]
        block = op.parent_block()
        new = []
        # y0 = bitcast(magic - (bitcast(v) >> 1))
        bi = arith.BitcastOp(v, i32)
        one = arith.ConstantOp(IntegerAttr(1, i32)); sh = arith.ShRSIOp(bi.result, one.results[0])
        magic = arith.ConstantOp(IntegerAttr(_RSQRT_MAGIC, i32)); sub = arith.SubiOp(magic.results[0], sh.result)
        y = arith.BitcastOp(sub.result, f32)
        new += [bi, one, sh, magic, sub, y]
        half = arith.ConstantOp(FloatAttr(0.5, f32)); threehalf = arith.ConstantOp(FloatAttr(1.5, f32))
        new += [half, threehalf]
        cur = y.result
        for _ in range(NEWTON):
            yy = arith.MulfOp(cur, cur); vyy = arith.MulfOp(v, yy.result)
            hvyy = arith.MulfOp(half.results[0], vyy.result); s = arith.SubfOp(threehalf.results[0], hvyy.result)
            ny = arith.MulfOp(cur, s.result)
            new += [yy, vyy, hvyy, s, ny]; cur = ny.result
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(cur)
        block.detach_op(op)
        n += 1
    return n
