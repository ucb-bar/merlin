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
from ..commandbuffer import materialize_inputs


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


def emit_kernel_mlir(cb: dict[str, Any], *, target: str | None = None) -> str:
    """Emit the LLVM-dialect MLIR kernel module for ``cb``. Supports a single fp32 matmul commit (the SIMT
    gemm corpus) with optional relu/bias epilogue; raises on an unsupported shape (chained matmuls / mx),
    like :func:`muon_codegen.emit_kernel_cpp`. The kernel symbol is ``{target}_kernel`` — ``target`` is
    taken from the arg else the command buffer's own ``target`` field (never a baked default)."""
    target = target or cb.get("target")
    if not target:
        raise MuonMlirCodegenError("emit_kernel_mlir needs a target (arg or cb['target'])")
    env = materialize_inputs(cb)
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
