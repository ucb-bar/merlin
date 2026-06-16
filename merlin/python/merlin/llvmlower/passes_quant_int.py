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
