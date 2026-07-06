"""EXO RVV kernels for the whole-model glue's elementwise ops (f32, VLEN=256).

Moves the glue's **residual add** (``h += x``) and the **SwiGLU element-wise product** (``silu(g)*u``)
from scalar C to EXO-scheduled 8-wide RVV (``vfadd.vv`` / ``vfmul.vv``), so they register as vector
in ``rvv_audit`` and lift ``rvv_coverage_overall``.

Honest scope: in the LLM workloads these ops are <1% of wall time (the GEMM and the int8
quant/transpose dominate — see the region profile), so vectorising them is coverage-positive but
~latency-neutral; the material speedup is the GEMM autotune. SiLU's sigmoid keeps its ``expf`` on
the scalar path (no RVV transcendental in this EXO port), so the glue computes ``silu(g)`` scalar
then this kernel does the vector ``silu*u`` product — the exp stays labeled scalar, the product is
RVV.
"""
from __future__ import annotations

from exo import proc, DRAM
from exo.stdlib.scheduling import (
    divide_loop,
    stage_mem,
    set_memory,
    simplify,
    replace_all,
)

from merlin.baselines.exo_kernels.rvv256 import (
    RVV256,
    rvv256_vld,
    rvv256_vst,
    rvv256_vfadd,
    rvv256_vfmul,
)


@proc
def residual_add_ref(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    # pragma: no cover — out = a + b (element-wise, fully vectorisable)
    assert N % 8 == 0
    for i in seq(0, N):
        out[i] = a[i] + b[i]


@proc
def ewise_mul_ref(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    # pragma: no cover — out = a * b (element-wise; the SwiGLU silu(g)*u product)
    assert N % 8 == 0
    for i in seq(0, N):
        out[i] = a[i] * b[i]


def _sched_binop(p, op_instr):
    """Vectorise a length-N element-wise binary op into 8-wide RVV load/op/store."""
    p = divide_loop(p, "i", 8, ["io", "ii"], perfect=True)
    # stage a,b into vector regs, compute into out reg, store.
    p = stage_mem(p, "for ii in _: _", "a[8*io:8*io+8]", "a_reg")
    p = set_memory(p, "a_reg", RVV256)
    p = stage_mem(p, "for ii in _: _", "b[8*io:8*io+8]", "b_reg")
    p = set_memory(p, "b_reg", RVV256)
    p = stage_mem(p, "for ii in _: out[_] = _", "out[8*io:8*io+8]", "o_reg")
    p = set_memory(p, "o_reg", RVV256)
    p = simplify(p)
    p = replace_all(p, [rvv256_vld, op_instr, rvv256_vst])
    return simplify(p)


residual_add_rvv = _sched_binop(residual_add_ref, rvv256_vfadd)
ewise_mul_rvv = _sched_binop(ewise_mul_ref, rvv256_vfmul)
