"""Emit a Muon SIMT kernel as an LLVM-dialect MLIR module from a Merlin command buffer (fp32).

This is the MLIR analogue of :mod:`muon_codegen` (which emits C++) and the Muon analogue of
:mod:`merlin.targets.gemmini.backend.gemmini_codegen_mlir` (which emits LLVM-dialect MLIR with RoCC
``.insn``). It is the reference for the THESIS path: the agent emits a COMPILER LOWERING (LLVM-dialect
MLIR), the runner compiles it FORK-FREE (:func:`merlin.runtime.backends.muon.compile_mlir_forkfree` —
stock LLVM rv32 + the RTL-derived Muon re-encode), and grades it. No C++, no vendor fork.

The emitted module defines ``llvm.func @{target}_kernel(<ptr args>)`` whose argument order is the generic
``kernel_abi`` — ``[weight] ++ [lhs in command order] ++ [outputs in command order]`` — the SAME order the
runner-owned harness (:func:`muon_harness.args_from_cb`) feeds. The kernel is plain scalar compute over the
pointer operands (loads → multiply-accumulate → stores); the SIMT warps/barriers are the runtime BSP's, so
the kernel carries no scheduling. fp32 epilogues supported: ``relu`` and ``bias_add``.
"""
from __future__ import annotations

from typing import Any

from .muon_codegen import _plan
from merlin.runtime.commandbuffer import materialize_inputs


class MuonMlirCodegenError(RuntimeError):
    pass


def _matmul_loop_nest(w: str, l: str, o: str, m: int, k: int, n: int, epi: list, bias: str | None) -> str:
    """LLVM-dialect triple-loop matmul ``O[m,n] = sum_k L[m,k]*W[k,n]`` (row-major), with the fp32 epilogue
    applied to each accumulator before the store. Loop induction vars + the k-accumulator are carried as
    block arguments (the llvm-dialect phi form). All indices/consts are SSA ops (no nested exprs)."""
    epi_lines: list[str] = []
    for stage in (epi or []):
        if stage == "relu":
            epi_lines.append('    %__z = llvm.mlir.constant(0.000000e+00 : f32) : f32')
            epi_lines.append('    %__rc = llvm.fcmp "ogt" %acc_k, %__z : f32')
            epi_lines.append('    %acc_e = llvm.select %__rc, %acc_k, %__z : i1, f32')
        elif stage in ("bias_add", "bias") and bias is not None:
            epi_lines.append(f'    %__bp = llvm.getelementptr %{bias}[%ni] : (!llvm.ptr, i64) -> !llvm.ptr, f32')
            epi_lines.append('    %__bv = llvm.load %__bp : !llvm.ptr -> f32')
            epi_lines.append('    %acc_e = llvm.fadd %acc_k, %__bv : f32')
    acc_final = "%acc_e" if epi_lines else "%acc_k"
    epi_block = ("\n".join(epi_lines) + "\n") if epi_lines else ""
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cM = llvm.mlir.constant({m} : i64) : i64
    %cK = llvm.mlir.constant({k} : i64) : i64
    %cN = llvm.mlir.constant({n} : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cM : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mK = llvm.mul %mi, %cK : i64
    %mN = llvm.mul %mi, %cN : i64
    llvm.br ^n(%c0 : i64)
  ^n(%ni: i64):
    %nc = llvm.icmp "slt" %ni, %cN : i64
    llvm.cond_br %nc, ^nbody, ^mnext
  ^nbody:
    llvm.br ^k(%c0, %zero : i64, f32)
  ^k(%ki: i64, %acc: f32):
    %kc = llvm.icmp "slt" %ki, %cK : i64
    llvm.cond_br %kc, ^kbody, ^store
  ^kbody:
    %lidx = llvm.add %mK, %ki : i64
    %lp = llvm.getelementptr %{l}[%lidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lv = llvm.load %lp : !llvm.ptr -> f32
    %kN = llvm.mul %ki, %cN : i64
    %widx = llvm.add %kN, %ni : i64
    %wp = llvm.getelementptr %{w}[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wv = llvm.load %wp : !llvm.ptr -> f32
    %prod = llvm.fmul %lv, %wv : f32
    %acc2 = llvm.fadd %acc, %prod : f32
    %ki2 = llvm.add %ki, %c1 : i64
    llvm.br ^k(%ki2, %acc2 : i64, f32)
  ^store:
    %acc_k = llvm.fadd %acc, %zero : f32
{epi_block}    %oidx = llvm.add %mN, %ni : i64
    %op = llvm.getelementptr %{o}[%oidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store {acc_final}, %op : f32, !llvm.ptr
    %ni2 = llvm.add %ni, %c1 : i64
    llvm.br ^n(%ni2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return"""


def _matmul_stage(p: str, ll: str, rr: str, oo: str, m: int, k: int, n: int, done_br: str) -> str:
    """One row-major matmul loop ``O[i,j] = sum_p L[i,p]*R[p,j]`` with every SSA name / block label
    prefixed by ``p`` (so several stages compose in one function). When the m-loop finishes it falls into
    ``^{p}done`` and runs ``done_br`` (a full ``llvm.br`` the caller supplies, so it can pass the next
    stage's loop-header block argument). Shared constants ``%c0``/``%c1``/``%zero`` come from the prelude;
    the caller branches into ``^{p}m`` with the initial index (no leading branch here, so stages chain
    without an orphan terminator)."""
    return f"""  ^{p}m(%{p}mi: i64):
    %{p}mc = llvm.icmp "slt" %{p}mi, %{p}cM : i64
    llvm.cond_br %{p}mc, ^{p}mb, ^{p}done
  ^{p}mb:
    %{p}mK = llvm.mul %{p}mi, %{p}cK : i64
    %{p}mN = llvm.mul %{p}mi, %{p}cN : i64
    llvm.br ^{p}n(%c0 : i64)
  ^{p}n(%{p}ni: i64):
    %{p}nc = llvm.icmp "slt" %{p}ni, %{p}cN : i64
    llvm.cond_br %{p}nc, ^{p}nb, ^{p}mnext
  ^{p}nb:
    llvm.br ^{p}k(%c0, %zero : i64, f32)
  ^{p}k(%{p}ki: i64, %{p}acc: f32):
    %{p}kc = llvm.icmp "slt" %{p}ki, %{p}cK : i64
    llvm.cond_br %{p}kc, ^{p}kb, ^{p}st
  ^{p}kb:
    %{p}lidx = llvm.add %{p}mK, %{p}ki : i64
    %{p}lp = llvm.getelementptr %{ll}[%{p}lidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %{p}lv = llvm.load %{p}lp : !llvm.ptr -> f32
    %{p}kN = llvm.mul %{p}ki, %{p}cN : i64
    %{p}widx = llvm.add %{p}kN, %{p}ni : i64
    %{p}wp = llvm.getelementptr %{rr}[%{p}widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %{p}wv = llvm.load %{p}wp : !llvm.ptr -> f32
    %{p}prod = llvm.fmul %{p}lv, %{p}wv : f32
    %{p}acc2 = llvm.fadd %{p}acc, %{p}prod : f32
    %{p}ki2 = llvm.add %{p}ki, %c1 : i64
    llvm.br ^{p}k(%{p}ki2, %{p}acc2 : i64, f32)
  ^{p}st:
    %{p}oidx = llvm.add %{p}mN, %{p}ni : i64
    %{p}op = llvm.getelementptr %{oo}[%{p}oidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %{p}acc, %{p}op : f32, !llvm.ptr
    %{p}ni2 = llvm.add %{p}ni, %c1 : i64
    llvm.br ^{p}n(%{p}ni2 : i64)
  ^{p}mnext:
    %{p}mi2 = llvm.add %{p}mi, %c1 : i64
    llvm.br ^{p}m(%{p}mi2 : i64)
  ^{p}done:
    {done_br}"""


def _chained_matmul_loop_nest(a: str, w1: str, w2: str, y: str, m: int, k: int, k2: int, n: int) -> str:
    """Two chained matmuls ``H = A@W1`` (m,k,k2) then ``Y = H@W2`` (m,k2,n), with the intermediate ``H``
    in an ``llvm.alloca`` (m*k2 fp32) — never a kernel argument. Stage A writes H, stage B reads it."""
    hsz = m * k2
    prelude = f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %acM = llvm.mlir.constant({m} : i64) : i64
    %acK = llvm.mlir.constant({k} : i64) : i64
    %acN = llvm.mlir.constant({k2} : i64) : i64
    %bcM = llvm.mlir.constant({m} : i64) : i64
    %bcK = llvm.mlir.constant({k2} : i64) : i64
    %bcN = llvm.mlir.constant({n} : i64) : i64
    %hsz = llvm.mlir.constant({hsz} : i64) : i64
    %H = llvm.alloca %hsz x f32 : (i64) -> !llvm.ptr
    llvm.br ^am(%c0 : i64)"""
    stage_a = _matmul_stage("a", a, w1, "H", m, k, k2, "llvm.br ^bm(%c0 : i64)")
    stage_b = _matmul_stage("b", "H", w2, y, m, k2, n, "llvm.br ^end")
    return f"{prelude}\n{stage_a}\n{stage_b}\n  ^end:\n    llvm.return"


def _attention_qk_loop_nest(q: str, k: str, o: str, m: int, d: int, n: int) -> str:
    """LLVM-dialect nest for attention scores ``O[m,n] = sum_d Q[m,d]*K[n,d]`` — i.e. ``Q @ K^T`` (row-major).
    Same phi-form skeleton as the matmul nest but the K operand is indexed by ROW ``[n,d]`` (the transpose)
    rather than ``[d,n]``, and there is no weight/epilogue. fp32 loads/accumulate, one fp32 store per (m,n)."""
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cM = llvm.mlir.constant({m} : i64) : i64
    %cD = llvm.mlir.constant({d} : i64) : i64
    %cN = llvm.mlir.constant({n} : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cM : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mD = llvm.mul %mi, %cD : i64
    %mN = llvm.mul %mi, %cN : i64
    llvm.br ^n(%c0 : i64)
  ^n(%ni: i64):
    %nc = llvm.icmp "slt" %ni, %cN : i64
    llvm.cond_br %nc, ^nbody, ^mnext
  ^nbody:
    %nD = llvm.mul %ni, %cD : i64
    llvm.br ^k(%c0, %zero : i64, f32)
  ^k(%ki: i64, %acc: f32):
    %kc = llvm.icmp "slt" %ki, %cD : i64
    llvm.cond_br %kc, ^kbody, ^store
  ^kbody:
    %qidx = llvm.add %mD, %ki : i64
    %qp = llvm.getelementptr %{q}[%qidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %qv = llvm.load %qp : !llvm.ptr -> f32
    %kidx = llvm.add %nD, %ki : i64
    %kp = llvm.getelementptr %{k}[%kidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %kv = llvm.load %kp : !llvm.ptr -> f32
    %prod = llvm.fmul %qv, %kv : f32
    %acc2 = llvm.fadd %acc, %prod : f32
    %ki2 = llvm.add %ki, %c1 : i64
    llvm.br ^k(%ki2, %acc2 : i64, f32)
  ^store:
    %oidx = llvm.add %mN, %ni : i64
    %op = llvm.getelementptr %{o}[%oidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %acc, %op : f32, !llvm.ptr
    %ni2 = llvm.add %ni, %c1 : i64
    llvm.br ^n(%ni2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return"""


def _rmsnorm_loop_nest(g: str, x: str, o: str, r: int, c: int, eps: float) -> str:
    """LLVM-dialect nest for row RMSNorm ``O[i,j] = X[i,j] * rsqrt(mean_j(X[i,:]^2) + eps) * G[j]`` (row-major,
    fp32). Per row: a reduce loop accumulates sum-of-squares, then ``ms = ss/C``, ``inv = 1/sqrt(ms+eps)``
    (``llvm.intr.sqrt`` -> hardware ``fsqrt.s``, no libcall), then a write loop scales each element by
    ``inv * G[j]``. ``G`` is the length-C gamma row (weight); its pointer is arg-0 (weight-first ABI)."""
    cf = f"{float(c):.6e}"
    ef = f"{float(eps):.6e}"
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cR = llvm.mlir.constant({r} : i64) : i64
    %cC = llvm.mlir.constant({c} : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %one = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %cCf = llvm.mlir.constant({cf} : f32) : f32
    %eps = llvm.mlir.constant({ef} : f32) : f32
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cR : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mC = llvm.mul %mi, %cC : i64
    llvm.br ^r(%c0, %zero : i64, f32)
  ^r(%ri: i64, %ss: f32):
    %rc = llvm.icmp "slt" %ri, %cC : i64
    llvm.cond_br %rc, ^rbody, ^rdone
  ^rbody:
    %xidx = llvm.add %mC, %ri : i64
    %xp = llvm.getelementptr %{x}[%xidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %xv = llvm.load %xp : !llvm.ptr -> f32
    %sq = llvm.fmul %xv, %xv : f32
    %ss2 = llvm.fadd %ss, %sq : f32
    %ri2 = llvm.add %ri, %c1 : i64
    llvm.br ^r(%ri2, %ss2 : i64, f32)
  ^rdone:
    %ms = llvm.fdiv %ss, %cCf : f32
    %mse = llvm.fadd %ms, %eps : f32
    %rt = llvm.intr.sqrt(%mse) : (f32) -> f32
    %inv = llvm.fdiv %one, %rt : f32
    llvm.br ^w(%c0 : i64)
  ^w(%wi: i64):
    %wc = llvm.icmp "slt" %wi, %cC : i64
    llvm.cond_br %wc, ^wbody, ^mnext
  ^wbody:
    %widx = llvm.add %mC, %wi : i64
    %wxp = llvm.getelementptr %{x}[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wxv = llvm.load %wxp : !llvm.ptr -> f32
    %gp = llvm.getelementptr %{g}[%wi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %gv = llvm.load %gp : !llvm.ptr -> f32
    %xn = llvm.fmul %wxv, %inv : f32
    %yv = llvm.fmul %xn, %gv : f32
    %wop = llvm.getelementptr %{o}[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %yv, %wop : f32, !llvm.ptr
    %wi2 = llvm.add %wi, %c1 : i64
    llvm.br ^w(%wi2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return"""


def _elementwise_loop_nest(a: str, b: str, o: str, n: int, combine: str) -> str:
    """LLVM-dialect single flat loop for an equal-shape elementwise map ``O[i] = A[i] <op> B[i]`` (fp32),
    ``op`` = fadd (combine ``add``) or fmul (combine ``mul``) — the transcendental-free VECTOR_MAP core
    (elementwise residual-add / scale). ``n`` is the total element count (rows*cols, flattened)."""
    fop = {"add": "llvm.fadd", "mul": "llvm.fmul"}[combine]
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cN = llvm.mlir.constant({n} : i64) : i64
    llvm.br ^l(%c0 : i64)
  ^l(%i: i64):
    %ic = llvm.icmp "slt" %i, %cN : i64
    llvm.cond_br %ic, ^body, ^end
  ^body:
    %ap = llvm.getelementptr %{a}[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %av = llvm.load %ap : !llvm.ptr -> f32
    %bp = llvm.getelementptr %{b}[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bv = llvm.load %bp : !llvm.ptr -> f32
    %rv = {fop} %av, %bv : f32
    %op = llvm.getelementptr %{o}[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %rv, %op : f32, !llvm.ptr
    %i2 = llvm.add %i, %c1 : i64
    llvm.br ^l(%i2 : i64)
  ^end:
    llvm.return"""


def _scalar_op_loop_nest(a: str, o: str, n: int, scalar: float, combine: str) -> str:
    """LLVM-dialect single flat loop for a compile-time scalar map ``O[i] = A[i] <op> c`` (fp32), ``op`` =
    fadd/fmul, ``c`` baked as a constant. The per-tensor-scale (embed-scale) / scalar-bias core; the scalar
    lives in the kernel, not as a runtime operand."""
    fop = {"add": "llvm.fadd", "mul": "llvm.fmul"}[combine]
    cf = f"{float(scalar):.8e}"
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cN = llvm.mlir.constant({n} : i64) : i64
    %cf = llvm.mlir.constant({cf} : f32) : f32
    llvm.br ^l(%c0 : i64)
  ^l(%i: i64):
    %ic = llvm.icmp "slt" %i, %cN : i64
    llvm.cond_br %ic, ^body, ^end
  ^body:
    %ap = llvm.getelementptr %{a}[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %av = llvm.load %ap : !llvm.ptr -> f32
    %rv = {fop} %av, %cf : f32
    %op = llvm.getelementptr %{o}[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %rv, %op : f32, !llvm.ptr
    %i2 = llvm.add %i, %c1 : i64
    llvm.br ^l(%i2 : i64)
  ^end:
    llvm.return"""


def _broadcast_row_loop_nest(a: str, b: str, o: str, m: int, n: int, combine: str) -> str:
    """LLVM-dialect nest for a row-broadcast elementwise map ``O[i,j] = A[i,j] <op> B[j]`` (fp32), where
    ``B`` is a length-``n`` row broadcast over the ``m`` rows of ``A`` (standalone bias-add / per-channel
    scale). ``op`` = fadd (``add``) or fmul (``mul``). Two nested loops; the ``B`` element is reloaded per
    (i,j) at column index ``j``."""
    fop = {"add": "llvm.fadd", "mul": "llvm.fmul"}[combine]
    return f"""    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cM = llvm.mlir.constant({m} : i64) : i64
    %cN = llvm.mlir.constant({n} : i64) : i64
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cM : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mN = llvm.mul %mi, %cN : i64
    llvm.br ^n(%c0 : i64)
  ^n(%ni: i64):
    %nc = llvm.icmp "slt" %ni, %cN : i64
    llvm.cond_br %nc, ^nbody, ^mnext
  ^nbody:
    %idx = llvm.add %mN, %ni : i64
    %ap = llvm.getelementptr %{a}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %av = llvm.load %ap : !llvm.ptr -> f32
    %bp = llvm.getelementptr %{b}[%ni] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bv = llvm.load %bp : !llvm.ptr -> f32
    %rv = {fop} %av, %bv : f32
    %op = llvm.getelementptr %{o}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %rv, %op : f32, !llvm.ptr
    %ni2 = llvm.add %ni, %c1 : i64
    llvm.br ^n(%ni2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return"""


def _shape2(env: dict, name: str) -> tuple[int, int]:
    t = env.get(name)
    if t is None or len(t.shape) != 2:
        raise MuonMlirCodegenError(f"operand {name!r} is not a materialized 2-D tensor")
    return t.shape[0], t.shape[1]


def emit_kernel_mlir(cb: dict[str, Any], *, target: str | None = None) -> str:
    """Emit the LLVM-dialect MLIR kernel module for ``cb``. Dispatches on the command-buffer op: a single fp32
    matmul commit (the SIMT gemm corpus, optional relu/bias epilogue), attention scores ``Q@K^T``
    (``ATTENTION_QK``), or row RMSNorm (``RMSNORM``); raises on an unsupported shape (chained matmuls / mx),
    like :func:`muon_codegen.emit_kernel_cpp`. The kernel symbol is ``{target}_kernel`` — ``target`` is
    taken from the arg else the command buffer's own ``target`` field (never a baked default). Argument order
    is the generic kernel_abi ``[weight] ++ [inputs in command order] ++ [outputs]``, matching the harness
    (:func:`merlin.runtime.backends.muon_harness.args_from_cb`)."""
    target = target or cb.get("target")
    if not target:
        raise MuonMlirCodegenError("emit_kernel_mlir needs a target (arg or cb['target'])")
    env = materialize_inputs(cb)
    sym = f"{target}_kernel"

    # ---- non-matmul SIMT ops (attention scores, rmsnorm) -----------------------------------------
    by_op: dict[str, list] = {}
    for cmd in cb.get("commands", []):
        by_op.setdefault((cmd.get("opcode") or "").upper(), []).append(cmd)

    # A whole-op mnemonic (RMSNORM / ATTENTION_QK) is a SINGLE-op kernel here; if the buffer also
    # carries a matmul (a fused op class like rmsnorm+matmul), fail LOUD rather than silently emitting
    # only one half and mis-grading — fused emission is not yet supported by this reference emitter.
    _WHOLE_OPS = {"RMSNORM", "ATTENTION_QK"}
    _MATMUL_OPS = {"RES_PACK", "MATMUL", "MATMUL_RESIDENT", "COMMIT"}
    if (_WHOLE_OPS & by_op.keys()) and (_MATMUL_OPS & by_op.keys()):
        raise MuonMlirCodegenError(
            f"reference emitter does not support a fused op class "
            f"({sorted(_WHOLE_OPS & by_op.keys())} + matmul); single-op or single matmul commit only")

    if "ATTENTION_QK" in by_op:
        o = by_op["ATTENTION_QK"][0].get("operands", {})
        q, k, dst = o.get("q"), o.get("k"), o.get("dst")
        if not (q and k and dst):
            raise MuonMlirCodegenError("ATTENTION_QK needs operands q/k/dst")
        m, d = _shape2(env, q)
        n, d2 = _shape2(env, k)
        if d != d2:
            raise MuonMlirCodegenError(f"attention head-dim mismatch: {q}{(m, d)} vs {k}{(n, d2)}")
        arg_decl = ", ".join(f"%{a}: !llvm.ptr" for a in (q, k, dst))
        nest = _attention_qk_loop_nest(q, k, dst, m, d, n)
        return f"module {{\n  llvm.func @{sym}({arg_decl}) {{\n{nest}\n  }}\n}}\n"

    if "VECTOR_MAP" in by_op and len(by_op) == 1 and len(by_op["VECTOR_MAP"]) == 1:
        cmd = by_op["VECTOR_MAP"][0]
        o = cmd.get("operands", {})
        attrs = cmd.get("attributes", {}) or {}
        combine = attrs.get("combine", "add")
        if combine not in ("add", "mul"):
            raise MuonMlirCodegenError(f"VECTOR_MAP combine {combine!r} not supported by the reference "
                                       f"emitter (transcendental-free add/mul only)")
        a, dst = o.get("lhs"), o.get("dst")
        if "scalar" in attrs and o.get("rhs") is None:            # compile-time scalar map A <op> c
            if not (a and dst):
                raise MuonMlirCodegenError("scalar VECTOR_MAP needs operands lhs/dst")
            ta = env.get(a)
            if ta is None:
                raise MuonMlirCodegenError(f"VECTOR_MAP operand {a} not materialized")
            n = 1
            for d in ta.shape:
                n *= d
            arg_decl = ", ".join(f"%{x}: !llvm.ptr" for x in (a, dst))
            nest = _scalar_op_loop_nest(a, dst, n, float(attrs["scalar"]), combine)
            return f"module {{\n  llvm.func @{sym}({arg_decl}) {{\n{nest}\n  }}\n}}\n"
        b = o.get("rhs")
        if not (a and b and dst):
            raise MuonMlirCodegenError("VECTOR_MAP needs operands lhs/rhs/dst")
        ta, tb = env.get(a), env.get(b)
        if ta is None or tb is None:
            raise MuonMlirCodegenError(f"VECTOR_MAP operands {a}/{b} not materialized")
        arg_decl = ", ".join(f"%{x}: !llvm.ptr" for x in (a, b, dst))
        if ta.shape == tb.shape:                                     # equal-shape elementwise
            n = 1
            for d in ta.shape:
                n *= d
            nest = _elementwise_loop_nest(a, b, dst, n, combine)
        elif len(ta.shape) == 2 and tb.shape == (ta.shape[1],):      # row broadcast B[n] over A[m,n]
            m, n = ta.shape
            nest = _broadcast_row_loop_nest(a, b, dst, m, n, combine)
        else:
            raise MuonMlirCodegenError(
                f"VECTOR_MAP supports equal-shape or a row-broadcast rhs B[n] over A[m,n]; "
                f"got {a}{tuple(ta.shape)} / {b}{tuple(tb.shape)}")
        return f"module {{\n  llvm.func @{sym}({arg_decl}) {{\n{nest}\n  }}\n}}\n"

    if "RMSNORM" in by_op:
        cmd = by_op["RMSNORM"][0]
        o = cmd.get("operands", {})
        attrs = cmd.get("attributes", {}) or {}
        x, gamma, dst = o.get("src"), o.get("gamma"), o.get("dst")
        if not (x and gamma and dst):
            raise MuonMlirCodegenError("RMSNORM needs operands src/gamma/dst")
        r, c = _shape2(env, x)
        eps = float(attrs.get("eps", 1e-5))
        # weight-first ABI: [gamma] ++ [src] ++ [out]
        arg_decl = ", ".join(f"%{a}: !llvm.ptr" for a in (gamma, x, dst))
        nest = _rmsnorm_loop_nest(gamma, x, dst, r, c, eps)
        return f"module {{\n  llvm.func @{sym}({arg_decl}) {{\n{nest}\n  }}\n}}\n"

    # ---- matmul (gemm) path -----------------------------------------------------------------------
    resident_source, matmul_for, commits = _plan(cb)

    # chained matmul: TWO commits where the second matmul consumes the first commit's output (A@W1@W2).
    if len(commits) == 2:
        c0, c1 = commits
        mm0, mm1 = matmul_for.get(c0["operands"]["src"]), matmul_for.get(c1["operands"]["src"])
        if mm0 is None or mm1 is None:
            raise MuonMlirCodegenError("chained matmul: a commit has no source matmul")
        if mm1["operands"]["lhs"] != c0["operands"]["dst"]:
            raise MuonMlirCodegenError("reference MLIR emitter supports a single matmul commit or a "
                                       "2-matmul chain (second consuming the first); got two unrelated commits")
        a_nm = mm0["operands"]["lhs"]
        w1 = resident_source.get(mm0["operands"]["rhs"], mm0["operands"]["rhs"])
        w2 = resident_source.get(mm1["operands"]["rhs"], mm1["operands"]["rhs"])
        y = c1["operands"]["dst"]                           # the output is produced, not materialized
        for nm in (a_nm, w1, w2):
            if nm not in env or len(env[nm].shape) != 2:
                raise MuonMlirCodegenError(f"chained matmul operand {nm!r} is not a 2-D materialized leaf")
        m, k = env[a_nm].shape
        _, k2 = env[w1].shape
        k2b, n = env[w2].shape
        if env[w1].shape[0] != k or k2b != k2:
            raise MuonMlirCodegenError("chained matmul inner dimensions do not agree")
        arg_names = [w1, w2, a_nm, y]                       # weights first, then input, then output
        arg_decl = ", ".join(f"%{a}: !llvm.ptr" for a in arg_names)
        nest = _chained_matmul_loop_nest(a_nm, w1, w2, y, m, k, k2, n)
        return f"module {{\n  llvm.func @{sym}({arg_decl}) {{\n{nest}\n  }}\n}}\n"

    if len(commits) != 1:
        raise MuonMlirCodegenError(f"reference MLIR emitter supports a single matmul commit, got {len(commits)}")
    # Only the 2-D leaves (matmul operands + output) index into the m/k/n loop math; a 1-D leaf (a
    # length-n bias vector) is consumed by the bias_add epilogue, indexed by column — do NOT reject it.
    shapes: dict[str, tuple[int, int]] = {name: (t.shape[0], t.shape[1])
                                          for name, t in env.items() if len(t.shape) == 2}

    commit = commits[0]
    ops = commit.get("operands", {})
    attrs = commit.get("attributes", {})
    mm = matmul_for.get(ops["src"])
    if mm is None:
        raise MuonMlirCodegenError(f"commit {ops['dst']!r} has no source matmul")
    mops = mm.get("operands", {})
    lhs, rhs = mops["lhs"], resident_source.get(mops["rhs"], mops["rhs"])
    if lhs not in shapes or rhs not in shapes:
        raise MuonMlirCodegenError(f"matmul operands {lhs!r}/{rhs!r} are not 2-D materialized leaves")
    m, k = shapes[lhs]
    k2, n = shapes[rhs]
    if k != k2:
        raise MuonMlirCodegenError(f"matmul K mismatch: {lhs}{shapes[lhs]} @ {rhs}{shapes[rhs]}")
    dst = ops["dst"]
    epi = attrs.get("epilogue", []) or []
    bias = ops.get("bias")
    if bias is not None and bias not in env:
        raise MuonMlirCodegenError(f"bias {bias!r} not materialized")

    # Arg order = the generic kernel_abi [weight] ++ [lhs] ++ [outputs] (+ bias if present, after weight).
    arg_names = [rhs, lhs, dst] + ([bias] if bias is not None else [])
    arg_decl = ", ".join(f"%{a}: !llvm.ptr" for a in arg_names)
    nest = _matmul_loop_nest(rhs, lhs, dst, m, k, n, epi, bias)
    return f"module {{\n  llvm.func @{target}_kernel({arg_decl}) {{\n{nest}\n  }}\n}}\n"
