"""Integer fast reciprocal-square-root lowering used by quantized normalization."""

from __future__ import annotations

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

    from .passes_quant_int import _select_targets

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
