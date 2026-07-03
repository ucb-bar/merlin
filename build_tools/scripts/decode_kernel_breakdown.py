#!/usr/bin/env python
"""THROWAWAY decode harness (task: rtl-derived-checks kernel breakdown).

Statically DECODES the emitted RVV asm for our whole-model-safe GEMM kernels and the two
expert ceilings (XNNPACK / OpenBLAS) and attributes the openvla/rdt2 ~half-gap across
NR(lane-width) / packing / accumulator-residency. HOST + SPIKE-toolchain only, no board.

Reuses the EXISTING decode infra (no rebuild):
  - merlin.kernels.decode.objdump  (tokenize -> RawInsn, section = enclosing symbol)
  - merlin.kernels.decode.rvv      (vtype state machine, loop_spans, count_in)
  - merlin.kernels.cca.lift_asm    (NR/LMUL/residency/register_block/nr_is_vsetvlmax)

ours forks: lowered to model.ll/model.o exactly as scripts/k1_cross_framework.measure_ours_k1
does (transform schedule + features), but stops at model.o (no scp/run). experts: the
bare-metal ELF run_expert_gemm builds, decoded at the ukernel symbol.

Output: a list of per-kernel decode dicts (printed JSON); the .md is written by hand from these.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.kernels.ceiling_drivers import run_expert_gemm as expert
from merlin.kernels.decode import rvv
from merlin.kernels.decode.objdump import tokenize
from merlin.kernels import cca

REPO = Path(repo_root())
VLEN = 256  # the mined K1 target VLEN (board is VLEN=256); spike harness ISA is VLEN=128.

# representative throughput-bound shapes (M,N,K). cube + real openvla/rdt2 matmul dims
# (extracted from output/{openvla,rdt2}_fp32_consistent/model.mlir linalg.matmul ins/outs).
SHAPES = {
    "cube_64": (64, 64, 64),
    "openvla_proj_17x192x576": (17, 576, 192),    # 17x192 * 192x576  (attn out / mlp)
    "openvla_mlp_20x128x512": (20, 512, 128),     # 20x128 * 128x512  (action-head MLP up)
    "rdt2_attn_28x1024x1024": (28, 1024, 1024),   # workhorse 28x1024 * 1024x1024
    "rdt2_mlp_28x1024x2816": (28, 2816, 1024),    # MLP up 28x1024 * 1024x2816
}

OURS_FORKS = (
    ("ours_wholemodel", ["accumulator_resident_wholemodel"]),
    ("ours_vfmacc_tiled", ["fused_vfmacc_tiled"]),
    ("ours_v3", ["accumulator_resident_microkernel_v3"]),
)


def _lower_ours_to_obj(features, M, N, K, work: Path):
    """Lower a single f32 matmul (M,N,K) with `features` to model.o, return (obj_path, blocker)."""
    from merlin.rvvgen import workloads
    from merlin.rvvgen.registry import load_rvv_package
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.runtime.backends import zephyr_model as zm

    bundle = workloads.gen_matmul_f32(work / "wl", M=M, N=N, K=K)
    hb = load_rvv_package(REPO / "artifacts/targets" / "rvv" / "hand_v0")
    pkg = replace(hb, run_id="decode", compiler_features=list(features))
    md = Path(bundle)
    prepared = zm._prepare_model_mlir(md / "model.mlir", work, int8_compute=pkg.is_int8)
    feats = frozenset(pkg.compiler_features or []) or None
    try:
        res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                               vectorize=True, transform_schedule=pkg.schedule_text,
                               hoist_static_allocs=False, features=feats)
    except PipelineError as e:
        return None, f"vectorized lowering raised: {str(e)[:200]}"
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    try:
        subprocess.run([str(clang23), "--target=riscv64-unknown-linux-gnu",
                        "-march=rv64gcv", "-mabi=lp64d", "-O2", "-Wno-override-module",
                        "-c", str(res.ll_path), "-o", str(model_o)],
                       capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        return None, f"model.o compile failed: {e.stderr[-300:] if e.stderr else e}"
    return model_o, None


# compiler-internal local labels objdump emits MID-function (e.g. `.Lpcrel_hi0`) split the
# function body into separate "sections". The compute body of `forward` therefore spans `forward`
# + any following `.L*` local-label blocks, UP TO the next real exported symbol
# (`_mlir_ciface_forward`). Scope by that contiguous run so the K-loop isn't truncated.
def _scope_raws(raws, symbol_substr: str):
    out = []
    inside = False
    for r in raws:
        sec = r.section
        is_local = "<.L" in sec or ".Lpcrel" in sec
        if symbol_substr in sec:
            inside = True
        elif inside and not is_local:
            # a new NON-local symbol begins -> end of the function body
            break
        if inside:
            out.append(r)
    return out


def _decode_symbol(obj_path: Path, symbol_substr: str | None):
    """Decode the object; if symbol_substr given, scope the InsnStream to that symbol's insns.
    Returns (stream, n_insns_scoped)."""
    raws = tokenize(obj_path)
    if symbol_substr is not None:
        raws = _scope_raws(raws, symbol_substr)
    # rebuild a scoped stream by re-running the vtype machine over the scoped raws
    cur = rvv.VType()
    insns = []
    for r in raws:
        if r.mnemonic in rvv._VSET:
            cur = rvv._parse_vtype(r.operands)
            insns.append(rvv.VInsn(raw=r, is_vector=True, vtype=cur))
            continue
        is_vec = r.mnemonic.startswith("v")
        insns.append(rvv.VInsn(raw=r, is_vector=is_vec, vtype=cur if is_vec else None))
    return rvv.InsnStream(insns=insns), len(raws)


def _resolve_nr(sew, lmul):
    if sew and lmul:
        return int(VLEN // sew * lmul)
    return None


def _fma_loop_vtype(stream, sp):
    """SEW/LMUL of the vfmacc INSIDE the fma loop (the matmul K-loop), not the function-global
    dominant vtype (which a greedy symbol scope pollutes with the e8/e64 memref-copy epilogue)."""
    if sp is None:
        return None, None
    for i in stream.insns_in(sp):
        if i.raw.mnemonic.startswith("vfmacc") and i.vtype and i.vtype.sew:
            return i.vtype.sew, i.vtype.lmul
    return None, None


def _analyze(stream, *, op, source):
    c = cca.lift_asm(stream, op=op, source=source)
    comp, vec = c.compute, c.vector
    sp = cca._fma_loop(stream)
    # vfmacc breakdown in the fma loop
    vf = vv = spills = 0
    if sp is not None:
        vf = stream.count_in(sp, "vfmacc.vf", "vmacc.vx")
        vv = stream.count_in(sp, "vfmacc.vv", "vfmacc.vvm")
        spills = stream.count_in(sp, *cca._ACC_SPILL_STORE) + stream.count_in(sp, *cca._ACC_SPILL_LOAD)
    total_vfmacc = stream.count("vfmacc", "vmacc")
    # SEW/LMUL/NR from the K-loop's vfmacc (robust to the greedy scope); fall back to global.
    lsew, llmul = _fma_loop_vtype(stream, sp)
    sew = lsew if lsew is not None else vec.sew
    lmul = llmul if llmul is not None else vec.lmul
    vec = replace(vec, sew=sew, lmul=lmul)
    nr = _resolve_nr(vec.sew, vec.lmul)
    mr = comp.register_block[0] if comp.register_block else None
    return {
        "sew": vec.sew, "lmul": vec.lmul,
        "nr_lanes_vlen256": nr,
        "vl_strategy": vec.vl_strategy,
        "nr_is_vsetvlmax": comp.nr_is_vsetvlmax,
        "MR": mr,
        "accumulator_resident": comp.accumulator_resident,
        "contraction_form": comp.contraction_form,
        "fma_loop_vfmacc_vf": vf,
        "fma_loop_vfmacc_vv": vv,
        "fma_loop_acc_spills": spills,
        "total_vfmacc": total_vfmacc,
        "has_loop": stream.has_loop(),
        "vtype_hist": stream.vtype_histogram(),
    }


def main():
    rows = []

    # ---- experts (shape-independent ukernel; decode once) ----
    for src, sym in (("xnnpack", "xnn_f32_gemm_ukernel_1x4v__rvv"),
                     ("openblas", "openblas_sgemm_kernel")):
        spec = expert._experts()[src]
        tmp = Path(tempfile.mkdtemp(prefix="dec_exp_"))
        elf = tmp / f"{src}.riscv"
        err = expert._build(spec["driver"], spec["incs"], elf)
        if err:
            rows.append({"kernel": src, "shape": "ukernel", "blocker": err[:200]})
            continue
        stream, n = _decode_symbol(elf, sym)
        a = _analyze(stream, op="matmul", source=f"{src}:{sym}")
        rows.append({"kernel": src, "shape": "ukernel(shape-indep)", "symbol": sym,
                     "n_insns": n, "packed": True, **a})

    # ---- ours forks x shapes ----
    for run_id, feats in OURS_FORKS:
        for sname, (M, N, K) in SHAPES.items():
            tmp = Path(tempfile.mkdtemp(prefix="dec_ours_"))
            obj, blk = _lower_ours_to_obj(feats, M, N, K, tmp)
            if blk:
                rows.append({"kernel": run_id, "shape": sname, "MNK": (M, N, K),
                             "blocker": blk})
                continue
            # ours model.o: the compute lives in `forward`/`_mlir_ciface_forward`. Scope to it.
            raws = tokenize(obj)
            secs = {r.section for r in raws}
            fsym = next((s for s in secs if "forward" in s), None)
            stream, n = _decode_symbol(obj, fsym)
            a = _analyze(stream, op="matmul", source=f"{run_id}:{sname}")
            rows.append({"kernel": run_id, "shape": sname, "MNK": (M, N, K),
                         "symbol": fsym, "n_insns": n, "packed": False, **a})

    out = REPO / "artifacts" / "ceiling" / "kernel_breakdown_decode.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
