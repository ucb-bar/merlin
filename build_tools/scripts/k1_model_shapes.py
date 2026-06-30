#!/usr/bin/env python
"""Task 2 — MODEL-SHAPED op benchmarks on the REAL K1 board.

The published ceiling matrix uses synthetic cubes (32/64/128). Models do NOT use cubes — they use
very RECTANGULAR GEMMs (small seq-len M, wide hidden N/K). This script pulls the ACTUAL [M,N,K] GEMM
shapes that appear in the lowered model.mlir of the captured models (smolvla / bitvla / small_llama /
tiny_llama) and runs the cross-framework comparison (OpenBLAS / XNNPACK + ours-intrinsic + ours-tiled)
on a representative handful, INNER-COMPUTE scope, bit-exact, N=3 min rdtime.

Shapes were extracted by parsing `linalg.matmul ins(tensor<MxK>, tensor<KxN>)` from each model.mlir
(see the SHAPES table below; counts = how many times that op appears in the model).

HONESTY on kernel constraints: the OpenBLAS 8x8 driver requires M%8==0 and N%8==0; the ours-intrinsic
MR=4 driver requires M%4==0; the XNNPACK 1x4v driver handles any M (mr=1 loop) but tiles N by
vsetvlmax. For a shape that violates a driver's divisibility the driver would mis-handle the tail, so
we GATE on VERIFY PASS and record honest not_run (blocker = "shape constraint: M%8 != 0" etc.) rather
than a wrong/fabricated number. ours-tiled goes through the compiler and handles arbitrary shapes.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merlin.common.paths import repo_root
import k1_large_shape_packing as L

# (label, model, M, N, K, count_in_model)
SHAPES = [
    ("small_llama_proj",  "small_llama", 8,    128,   128,  8),
    ("tiny_llama_attn",   "tiny_llama",  8,    2048,  2048, 4),
    ("tiny_llama_mlp_up", "tiny_llama",  8,    5632,  2048, 4),
    ("tiny_llama_lmhead", "tiny_llama",  8,    32000, 2048, 1),
    ("bitvla_proj",       "bitvla",      32,   512,   256,  4),
    ("smolvla_vlm_attn",  "smolvla",     1024, 768,   768,  48),
    ("smolvla_vlm_mlp",   "smolvla",     1024, 3072,  768,  12),
    ("smolvla_act_ffn",   "smolvla",     113,  2560,  960,  32),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/gemm/model_shape_matrix_k1.jsonl")
    a = ap.parse_args()

    rows = []
    for label, model, M, N, K, count in SHAPES:
        meta = {"label": label, "model": model, "op_count_in_model": count}
        print(f"\n##### {label} ({model}) [M={M} N={N} K={K}] x{count} #####", flush=True)
        # experts (inner-compute) + ours-intrinsic (inner-compute) + ours-tiled (compiler, fused),
        # all with explicit rectangular [M,N,K] (the measure_* in Task-1 are square-only).
        for src in ("openblas", "xnnpack"):
            r = _expert_rect(src, M=M, N=N, K=K, reps=a.reps)
            r.update(meta)
            print(f"  {src:9s}: {r.get('status')} {r.get('ticks')} {r.get('blocker','')}", flush=True)
            rows.append(r)
        r = _intrinsic_rect(M=M, N=N, K=K, reps=a.reps); r.update(meta)
        print(f"  intrinsic: {r.get('status')} {r.get('ticks')} {r.get('blocker','')}", flush=True)
        rows.append(r)
        r = _ours_tiled_mnk(M=M, N=N, K=K, reps=a.reps); r.update(meta)
        print(f"  ours-tiled: {r.get('status')} {r.get('ticks')} {r.get('blocker','')}", flush=True)
        rows.append(r)
        outp = Path(repo_root()) / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as fh:
            for rr in rows:
                fh.write(json.dumps(rr) + "\n")
        print(f"  (wrote {len(rows)} rows -> {outp})", flush=True)
    print(f"\nDONE: {len(rows)} rows")


def _expert_rect(source, *, M, N, K, reps):
    import re, subprocess, tempfile
    from merlin.kernels.ceiling_drivers import run_expert_gemm as expert
    from merlin.rvvgen import k1
    spec = expert._experts()[source]
    base = {"op": "matmul", "dtype": spec["dtype"], "M": M, "N": N, "K": K, "source": source,
            "target": "k1", "mode": "inner_compute", "scope": "inner_compute", "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "kernel_file": spec["kernel_file"], "measure_method": "standalone_linux_inner_compute"}
    # constraint guards (honest not_run instead of a wrong number)
    if source == "openblas" and (M % 8 or N % 8):
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"shape constraint: OpenBLAS 8x8 driver requires M%8==0 and N%8==0 (M={M},N={N})"}
    if source == "xnnpack" and (N % 8):
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"shape constraint: XNNPACK 1x4v driver expects N%NR==0 (N={N})"}
    cc = L._cc()
    incs = [L.K1H, L.HERE] + [p for p in spec["incs"] if p != L.HERE]
    inc_flags = []
    for d in incs:
        inc_flags += ["-I", str(d)]
    shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
    with tempfile.TemporaryDirectory(prefix="k1_expert_") as tmp:
        binp = Path(tmp) / f"{source}_gemm"
        cmd = [str(cc), *inc_flags, *L._K1_CFLAGS, *shape, "-static", "-o", str(binp),
               str(spec["driver"]), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"], capture_output=True,
                                    text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}
        runs, detail = L._min_ticks(binp, f"{source}_{M}_{N}_{K}", reps, timeout=900)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": f"K1 rdtime; inner-compute; bit-exact; min of {reps}"}


def _intrinsic_rect(*, M, N, K, reps):
    import subprocess, tempfile
    from merlin.rvvgen import k1
    base = {"op": "matmul", "dtype": "f32", "M": M, "N": N, "K": K, "source": "ours-intrinsic",
            "target": "k1", "mode": "inner_compute", "scope": "inner_compute", "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "kernel_file": str(L.INTRINSIC_DRIVER.relative_to(L.REPO)),
            "measure_method": "standalone_linux_inner_compute"}
    if M % 4:
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"shape constraint: ours-intrinsic MR=4 driver requires M%4==0 (M={M})"}
    cc = L._cc()
    inc_flags = ["-I", str(L.K1H), "-I", str(L.HERE)]
    shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
    with tempfile.TemporaryDirectory(prefix="k1_intrinsic_") as tmp:
        binp = Path(tmp) / "ours_intrinsic_gemm"
        cmd = [str(cc), *inc_flags, *L._K1_CFLAGS, *shape, "-static", "-o", str(binp),
               str(L.INTRINSIC_DRIVER), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"], capture_output=True,
                                    text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}
        runs, detail = L._min_ticks(binp, f"intrinsic_{M}_{N}_{K}", reps, timeout=900)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": f"K1 rdtime; inner-compute; bit-exact; min of {reps}"}


def _ours_tiled_mnk(*, M, N, K, reps):
    """Compiler-emitted tiled-vfmacc on an explicit rectangular [M,N,K]."""
    import subprocess, tempfile
    from dataclasses import replace
    from merlin.rvvgen import k1, workloads
    from merlin.rvvgen.registry import load_rvv_package
    from merlin.llvmlower import c_runtime, toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.runtime.backends import zephyr_model as zm

    base = {"op": "matmul", "dtype": "f32", "M": M, "N": N, "K": K, "source": "ours_tiled",
            "target": "k1", "mode": "compiler_fused_pack", "scope": "inner==full", "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "compiler_features": ["fused_vfmacc_tiled"],
            "measure_method": "standalone_linux_compiler_fused"}
    bundle = workloads.gen_matmul_f32(L.REPO / "artifacts" / "cache" / "rvv_workloads", M=M, N=N, K=K)
    hb = load_rvv_package(L.REPO / "generated_targets" / "rvv" / "hand_v0")
    pkg = replace(hb, run_id="ours_tiled", compiler_features=["fused_vfmacc_tiled"])
    cc = L._cc()
    with tempfile.TemporaryDirectory(prefix="k1_ours_") as tmp:
        work = Path(tmp) / "work"; work.mkdir(parents=True, exist_ok=True)
        md = Path(bundle)
        prepared = zm._prepare_model_mlir(md / "model.mlir", work, int8_compute=pkg.is_int8)
        feats = frozenset(pkg.compiler_features or []) or None
        try:
            res = lower_model_file(prepared, work / "lower", targets=(), textual=True, vectorize=True,
                                   transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                                   features=feats)
        except PipelineError as e:
            return {**base, "ticks": None, "status": "not_run",
                    "blocker": f"vectorized lowering raised (feature shape-unsafe): {str(e)[:200]}"}
        clang23 = toolchain.clang()
        model_o = work / "model.o"
        try:
            subprocess.run([str(clang23), "--target=riscv64-unknown-linux-gnu", "-march=rv64gcv",
                            "-mabi=lp64d", "-O2", "-Wno-override-module", "-c", str(res.ll_path),
                            "-o", str(model_o)], capture_output=True, text=True, timeout=400, check=True)
        except subprocess.CalledProcessError as e:
            return {**base, "ticks": None, "status": "not_run",
                    "blocker": f"model.o compile failed: {e.stderr[-300:] if e.stderr else e}"}
        cgen = work / "cgen"; c_runtime.generate(md, cgen, md / "inputs.npz")
        rt = L.REPO / "merlin/runtime/c"; abi = L.REPO / "merlin/runtime/abi"
        binp = Path(tmp) / "ours_gemm"
        inc_flags = []
        for d in [L.K1H, L.HERE, cgen, rt]:
            inc_flags += ["-I", str(d)]
        shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
        srcs = [str(L.HERE / "ours_gemm_driver.c"), str(cgen / "model_call.c"),
                str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
        cmd = [str(cc), *inc_flags, *L._K1_CFLAGS, *shape, "-static", "-o", str(binp), *srcs,
               "-lm", "-lpthread"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"], capture_output=True,
                                    text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"link failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}
        runs, detail = L._min_ticks(binp, f"ours_tiled_{M}_{N}_{K}", reps, timeout=1200)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": f"K1 rdtime; compiler tiled-vfmacc (pack fused); bit-exact; min of {reps}"}


if __name__ == "__main__":
    main()
