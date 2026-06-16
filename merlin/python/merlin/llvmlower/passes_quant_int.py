"""Integer (W8A8) int8 compute datapath.

Today the int8 capture is WEIGHT-ONLY: ``quant_ext.dequantize_per_channel(i8, scale, zp)``
produces an f32 weight that feeds an f32 ``linalg.matmul`` — the contraction runs in float.
This module rewrites that into a real integer contraction:

    act_f32 ─quantize(per-row)→ act_i8 ┐
                                       ├─ linalg.matmul i8×i8→i32 (RVV vwmacc) ─requant→ out_f32
    dequant(w_i8,sw) ──(drop)── w_i8 ──┘

The matmul COMPUTE is i8×i8 with i32 accumulation (clang ``-march=rv64gcv`` lowers the named
``linalg.matmul`` to a widening ``vwmacc``). Activations are dynamically quantized per row
(``sa = max|a|/127``, symmetric, zero-point 0) and the i32 accumulator is requantized by
``acc * sa[m] * sw[n]`` — the proven W8A8 numerics (``tests/test_int8_compute.py``). The output
stays f32, so downstream (nonlinears) is unchanged: this is the matmul-only stage (M1). Weight
zero-point must be 0 (symmetric per-channel weight quant) — nonzero-zp matmuls are left alone for
the f32 dequant fallback (``lower_quant_ext``).
"""
from __future__ import annotations


def _is_dequant(op) -> bool:
    name = getattr(op, "op_name", None)
    return (op.name == "builtin.unregistered" and name is not None
            and name.data == "quant_ext.dequantize_per_channel")


def lower_matmul_int8(module) -> int:
    """Rewrite ``dequant(weight) → f32 matmul`` into ``quant(act) → i8 matmul → requant``.

    Returns the number of matmuls converted. Only 2-D ``linalg.matmul`` whose RHS is a
    per-channel weight dequant with zero-point 0 is converted (the common Linear case).
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerType,
                                       TensorType, f32, i8, i32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.dialects import math as mathd
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, exprs):
        return AffineMapAttr(AffineMap(n, 0, tuple(exprs)))

    # collect (matmul, dequant) pairs first (mutating while walking is unsafe)
    targets = []
    for op in module.walk():
        if op.name != "linalg.matmul":
            continue
        if len(op.operands) < 2:
            continue
        rhs = op.inputs[1]
        deq = rhs.owner
        if not _is_dequant(deq):
            continue
        out_t = op.results[0].type
        lhs_t = op.inputs[0].type
        if not (isinstance(out_t, TensorType) and len(out_t.get_shape()) == 2):
            continue
        if not (isinstance(lhs_t, TensorType) and lhs_t.element_type == f32):
            continue
        targets.append((op, deq))

    n = 0
    for mm, deq in targets:
        w_i8, w_scale, w_zp = deq.operands           # i8 [K,N], f32 [N], i32 [N]
        act = mm.inputs[0]                            # f32 [M,K]
        out_t = mm.results[0].type                    # f32 [M,N]
        M, K = act.type.get_shape()
        N = out_t.get_shape()[1]
        block = mm.parent_block()
        ins_at = deq                                  # insert everything before the dequant

        act_i8_t = TensorType(i8, [M, K])
        sa_t = TensorType(f32, [M])                   # per-row activation scale
        acc_t = TensorType(i32, [M, N])

        d2 = AffineMap.identity(2).results            # (d0,d1)
        # --- 1. per-row abs-max of the activation -> sa[M] = max|a|/127 ---
        amax_e = tensor.EmptyOp((), sa_t)
        zero_f = arith.ConstantOp(FloatAttr(0.0, f32))   # |a| >= 0, so 0.0 is the correct max init
        amax_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amax_e.results[0]], res=[sa_t])
        rb = Block(arg_types=[f32, f32])              # (a, acc)
        a_in, acc_in = rb.args
        ab = mathd.AbsFOp(a_in)
        mx = arith.MaximumfOp(ab.result, acc_in)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amax = L.GenericOp(
            inputs=(act,), outputs=(amax_f.results[0],), body=Region(rb),
            indexing_maps=ArrayAttr([amap(2, [d2[0], d2[1]]), amap(2, [d2[0]])]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(red)]),
            result_types=(sa_t,))
        # sa = amax / 127
        sa_e = tensor.EmptyOp((), sa_t)
        c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32])
        sa_in, _ = sb.args
        sd = arith.DivfOp(sa_in, c127.results[0])
        sb.add_ops([sd, L.YieldOp(sd.result)])
        d1 = AffineMap.identity(1).results
        sa = L.GenericOp(inputs=(amax.results[0],), outputs=(sa_e.results[0],), body=Region(sb),
                         indexing_maps=ArrayAttr([amap(1, [d1[0]]), amap(1, [d1[0]])]),
                         iterator_types=ArrayAttr([L.IteratorTypeAttr(par)]), result_types=(sa_t,))

        # --- 2. quantize activation: aq = clamp(roundeven(a/sa), -127, 127) -> i8 ---
        aq_e = tensor.EmptyOp((), act_i8_t)
        c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qb = Block(arg_types=[f32, f32, i8])          # (a, sa[row], out)
        av, sav, _ = qb.args
        q1 = arith.DivfOp(av, sav)
        q2 = mathd.RoundEvenOp(q1.result)
        q3 = arith.MinimumfOp(q2.result, c127.results[0])
        q4 = arith.MaximumfOp(q3.result, c127n.results[0])
        q5 = arith.FPToSIOp(q4.result, i8)
        qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
        aq = L.GenericOp(
            inputs=(act, sa.results[0]), outputs=(aq_e.results[0],), body=Region(qb),
            indexing_maps=ArrayAttr([amap(2, [d2[0], d2[1]]), amap(2, [d2[0]]), amap(2, [d2[0], d2[1]])]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(par)]),
            result_types=(act_i8_t,))

        # --- 3. i8×i8→i32 matmul as a generic (extsi→muli→addi i32; RVV integer MAC) ---
        acc_e = tensor.EmptyOp((), acc_t)
        zero_i = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zero_i.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        d3 = AffineMap.identity(3).results            # (d0=M, d1=N, d2=K)
        mb = Block(arg_types=[i8, i8, i32])           # (a, b, acc)
        av_i, bv_i, acc_i = mb.args
        ea = arith.ExtSIOp(av_i, i32)
        eb = arith.ExtSIOp(bv_i, i32)
        pm = arith.MuliOp(ea.result, eb.result)
        pa = arith.AddiOp(pm.result, acc_i)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        i8mm = L.GenericOp(
            inputs=(aq.results[0], w_i8), outputs=(acc_f.results[0],), body=Region(mb),
            indexing_maps=ArrayAttr([amap(3, [d3[0], d3[2]]), amap(3, [d3[2], d3[1]]),
                                     amap(3, [d3[0], d3[1]])]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(par),
                                      L.IteratorTypeAttr(red)]),
            result_types=(acc_t,))

        # --- 4. requant: out_f32[m,n] = sitofp(acc) * sa[m] * sw[n] ---
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32, f32, f32, f32])    # (acc, sa[m], sw[n], out)
        accv, sav2, swv, _ = wb.args
        r1 = arith.SIToFPOp(accv, f32)
        r2 = arith.MulfOp(r1.result, sav2)
        r3 = arith.MulfOp(r2.result, swv)
        wb.add_ops([r1, r2, r3, L.YieldOp(r3.result)])
        requant = L.GenericOp(
            inputs=(i8mm.results[0], sa.results[0], w_scale), outputs=(out_e.results[0],),
            body=Region(wb),
            indexing_maps=ArrayAttr([amap(2, [d2[0], d2[1]]), amap(2, [d2[0]]),
                                     amap(2, [d2[1]]), amap(2, [d2[0], d2[1]])]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(par)]),
            result_types=(out_t,))
        for k, v in mm.attributes.items():
            if k.startswith("prov."):
                requant.attributes[k] = v

        # insert in order before the matmul
        for new in (amax_e, zero_f, amax_f, amax, sa_e, c127, sa, aq_e, c127n, aq,
                    acc_e, zero_i, acc_f, i8mm, out_e, requant):
            block.insert_op_before(new, mm)
        mm.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(mm)
        # the dequant is now dead (weight used directly as i8); drop it if unused
        if not deq.results[0].uses:
            deq.parent_block().detach_op(deq)
        n += 1
    return n
