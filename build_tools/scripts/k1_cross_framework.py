#!/usr/bin/env python
"""PART 2 — cross-framework fp32 GEMM ceiling matrix on the REAL K1 board.

Reproduces out/artifacts/ceiling/cross_framework_matrix.md (which was measured on the FUNCTIONAL
spike, a cycle proxy) on the SpacemiT K1 silicon. Same five columns, same inner-compute scope
(operand pack / descriptor build hoisted OUT of the timed region), bit-exact verified, but timed
with the board's delegated `rdtime` (24 MHz) instead of spike's mcycle proxy.

The EXPERT columns (OpenBLAS sgemm_kernel_8x8_zvl128b, XNNPACK xnn_f32_gemm_ukernel_1x4v__rvv)
reuse the SAME driver C sources the spike matrix used (ceiling_drivers/{openblas,xnnpack}_gemm_
driver.c) — compiled UNCHANGED, with the k1_harness/util.h shim placed first on the include path
so `read_csr(mcycle)` maps to rdtime and printf comes from glibc. Compiled for riscv64-linux with
the SpacemiT clang (-march=rv64gcv -mabi=lp64d), scp'd to the board, run, parsed for VERIFY/CYCLES.

OURS columns (baseline + fused_vfmacc_contraction + fused_vfmacc_tiled) reuse ours_gemm_driver.c
linked against the COMPILER-EMITTED model.o for each (fork, shape) — the exact lowering the runner
uses — also rdtime-timed inner-compute. honest not_run with the exact blocker for anything that
won't build/run on K1; never a fabricated number.
"""
from __future__ import annotations

import argparse, json, subprocess, tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.common.driver_output import int_after
from merlin.kernels.ceiling_drivers import run_expert_gemm as expert
from merlin.kernels import bench_ceiling
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package

HERE = Path(repo_root()) / "merlin/python/merlin/kernels/ceiling_drivers"
K1H = HERE / "k1_harness"
REPO = Path(repo_root())

SHAPES = (32, 64)  # K1; 64 + 32 per the task. (128 optional — add via --shapes)
OURS_FORKS = (
    ("ours_baseline", []),
    ("ours_vfmacc_contraction", ["fused_vfmacc_contraction"]),
    ("ours_vfmacc_tiled", ["fused_vfmacc_tiled"]),
    ("ours_v3", ["accumulator_resident_microkernel_v3"]),  # the current compiler kernel (real K1 ticks)
)

# K1 Linux compile flags (glibc hosted; NOT medany/nostdlib). riscv_vector.h + the kernel
# intrinsics come from the SpacemiT clang. -ffast-math to match the spike experts' -O3 -ffast-math.
_K1_CFLAGS = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
              "-O3", "-ffast-math", "-DNDEBUG", "-std=gnu99", "-Wno-implicit-function-declaration"]


def _cc() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    return cc


def _deploy_run(binary: Path, tag: str, *, timeout: int = 300) -> tuple[str | None, str]:
    """scp the binary to the board, run it, return (stdout-or-None, detail)."""
    remote = f"/tmp/k1ceil_{tag}"
    try:
        subprocess.run(["scp", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no", str(binary), f"{k1.K1_HOST}:{remote}"],
                       capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as e:
        return None, f"scp failed: {e.stderr[-200:] if e.stderr else e}"
    try:
        k1._ssh(f"chmod +x {remote}", timeout=30)
        p = k1._ssh(remote, timeout=timeout)
    finally:
        try:
            k1._ssh(f"rm -f {remote}", timeout=30)
        except Exception:  # noqa: BLE001
            pass
    if p.returncode != 0:
        return None, f"run rc={p.returncode}; stderr: {p.stderr.strip()[-200:]}; stdout: {p.stdout.strip()[-200:]}"
    return p.stdout, "ok"


def _parse(base: dict, console: str | None, detail: str) -> dict:
    if console is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    if "VERIFY PASS" not in console:
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"verify did not pass; console tail: {console.strip()[-300:]}"}
    ticks = int_after(console, "CYCLES")  # driver prints CYCLES = read_csr(mcycle) delta = rdtime ticks
    if ticks is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": "no CYCLES/ticks line"}
    return {**base, "ticks": ticks, "status": "pass",
            "wall_ns_est": int(ticks * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": "K1 real-silicon rdtime ticks; inner-compute; bit-exact verified"}


def measure_expert_k1(source: str, *, M: int, N: int, K: int) -> dict:
    spec = expert._experts()[source]
    base = {"op": "matmul", "dtype": spec["dtype"], "M": M, "N": N, "K": K,
            "source": source, "target": "k1", "mode": "inner_compute",
            "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "kernel_file": spec["kernel_file"], "measure_method": "standalone_linux_inner_compute"}
    cc = _cc()
    incs = [K1H, HERE] + [p for p in spec["incs"] if p != HERE]  # k1_harness/util.h FIRST
    inc_flags = []
    for d in incs:
        inc_flags += ["-I", str(d)]
    shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
    with tempfile.TemporaryDirectory(prefix="k1_expert_") as tmp:
        binp = Path(tmp) / f"{source}_gemm"
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *shape, "-static", "-o", str(binp),
               str(spec["driver"]), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            # retry dynamic if static link fails
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"],
                                    capture_output=True, text=True, timeout=300)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-700:]}"}
        console, detail = _deploy_run(binp, f"{source}_{M}_{N}_{K}")
    return _parse(base, console, detail)


def measure_ours_k1(run_id: str, features: list[str], *, M: int, N: int, K: int,
                    timeout: int = 600) -> dict:
    from merlin.rvvgen.apply import apply_rvv_package
    from merlin.rvvgen import workloads

    base = {"op": "matmul", "dtype": "f32", "M": M, "N": N, "K": K,
            "source": run_id, "target": "k1", "mode": "inner_compute",
            "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "compiler_features": features,
            "kernel_file": f"merlin RVV codegen fork (features={features or 'baseline'})",
            "measure_method": "standalone_linux_inner_compute"}
    bundle = workloads.gen_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", M=M, N=N, K=K)
    hb = load_rvv_package(REPO / "out/artifacts/targets" / "rvv" / "hand_v0")
    pkg = replace(hb, run_id=run_id, compiler_features=list(features))
    cc = _cc()

    with tempfile.TemporaryDirectory(prefix="k1_ours_") as tmp:
        work = Path(tmp) / "work"
        work.mkdir(parents=True, exist_ok=True)
        # Reuse the K1 build path's lowering: model.mlir -> (vectorize, schedule, features) -> model.ll
        # -> model.o, plus the data-driven C runtime artifacts. We replicate build_k1_binary's
        # model.o + cgen steps but link OUR ceiling driver (inner-compute timed) instead of main_linux.
        from merlin.llvmlower import c_runtime, toolchain
        from merlin.llvmlower.lower import lower_model_file
        from merlin.llvmlower.pipeline import PipelineError
        from merlin.runtime.backends import zephyr_model as zm

        md = Path(bundle)
        prepared = zm._prepare_model_mlir(md / "model.mlir", work, int8_compute=pkg.is_int8)
        feats = frozenset(pkg.compiler_features or []) or None
        lowered_path = "vectorized"
        try:
            res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                                   vectorize=True, transform_schedule=pkg.schedule_text,
                                   hoist_static_allocs=False, features=feats)
        except PipelineError as e:
            return {**base, "ticks": None, "status": "not_run",
                    "blocker": f"vectorized lowering raised (feature whole-shape unsafe): {str(e)[:200]}"}
        clang23 = toolchain.clang()
        model_o = work / "model.o"
        try:
            subprocess.run([str(clang23), "--target=riscv64-unknown-linux-gnu",
                            "-march=rv64gcv", "-mabi=lp64d", "-O2", "-Wno-override-module",
                            "-c", str(res.ll_path), "-o", str(model_o)],
                           capture_output=True, text=True, timeout=300, check=True)
        except subprocess.CalledProcessError as e:
            return {**base, "ticks": None, "status": "not_run",
                    "blocker": f"model.o compile failed: {e.stderr[-400:] if e.stderr else e}"}
        cgen = work / "cgen"
        c_runtime.generate(md, cgen, md / "inputs.npz")

        rt = REPO / "merlin/runtime/c"
        abi = REPO / "merlin/runtime/abi"
        binp = Path(tmp) / "ours_gemm"
        incs = [K1H, HERE, cgen, rt]
        inc_flags = []
        for d in incs:
            inc_flags += ["-I", str(d)]
        shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
        srcs = [str(HERE / "ours_gemm_driver.c"), str(cgen / "model_call.c"),
                str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *shape, "-static", "-o", str(binp), *srcs, "-lm", "-lpthread"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"],
                                    capture_output=True, text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"link failed rc={p.returncode}: {p.stderr.strip()[-700:]}"}
        console, detail = _deploy_run(binp, f"{run_id}_{M}_{N}_{K}", timeout=timeout)
    r = _parse(base, console, detail)
    r["lowering_path"] = lowered_path
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="32,64")
    ap.add_argument("--out", default="out/artifacts/measurements/k1_spacemit/gemm/cross_framework_matrix_k1.jsonl")
    a = ap.parse_args()
    shapes = [int(s) for s in a.shapes.split(",")]

    rows = []
    for S in shapes:
        for src in ("openblas", "xnnpack"):
            print(f"--- expert {src} @ {S}^3 ---")
            r = measure_expert_k1(src, M=S, N=S, K=S)
            print("  ", r.get("status"), r.get("ticks"), r.get("blocker", ""))
            rows.append(r)
        for run_id, feats in OURS_FORKS:
            print(f"--- ours {run_id} @ {S}^3 ---")
            r = measure_ours_k1(run_id, feats, M=S, N=S, K=S)
            print("  ", r.get("status"), r.get("ticks"), r.get("blocker", ""))
            rows.append(r)

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
