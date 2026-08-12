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

    # ---- matmul (gemm) path (unchanged) ----------------------------------------------------------
    resident_source, matmul_for, commits = _plan(cb)
    if len(commits) != 1:
        raise MuonMlirCodegenError(f"reference MLIR emitter supports a single matmul commit, got {len(commits)}")
    shapes: dict[str, tuple[int, int]] = {}
    for name, t in env.items():
        if len(t.shape) != 2:
            raise MuonMlirCodegenError(f"leaf {name!r} is rank {len(t.shape)}; expected 2D")
        shapes[name] = (t.shape[0], t.shape[1])

    commit = commits[0]
    ops = commit.get("operands", {})
    attrs = commit.get("attributes", {})
    mm = matmul_for.get(ops["src"])
    if mm is None:
        raise MuonMlirCodegenError(f"commit {ops['dst']!r} has no source matmul")
    mops = mm.get("operands", {})
    lhs, rhs = mops["lhs"], resident_source.get(mops["rhs"], mops["rhs"])
    if lhs not in shapes or rhs not in shapes:
        raise MuonMlirCodegenError(f"matmul operands {lhs!r}/{rhs!r} not materialized")
    m, k = shapes[lhs]
    k2, n = shapes[rhs]
    if k != k2:
        raise MuonMlirCodegenError(f"matmul K mismatch: {lhs}{shapes[lhs]} @ {rhs}{shapes[rhs]}")
    dst = ops["dst"]
    epi = attrs.get("epilogue", []) or []
    bias = ops.get("bias")
    if bias is not None and bias not in shapes:
        raise MuonMlirCodegenError(f"bias {bias!r} not materialized")

    # Arg order = the generic kernel_abi [weight] ++ [lhs] ++ [outputs] (+ bias if present, after weight).
    arg_names = [rhs, lhs, dst] + ([bias] if bias is not None else [])
    arg_decl = ", ".join(f"%{a}: !llvm.ptr" for a in arg_names)
    nest = _matmul_loop_nest(rhs, lhs, dst, m, k, n, epi, bias)
    return f"module {{\n  llvm.func @{target}_kernel({arg_decl}) {{\n{nest}\n  }}\n}}\n"
