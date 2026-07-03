#!/usr/bin/env python
"""Task 1 — LARGE-SHAPE, PACKING-INCLUDED cross-framework GEMM matrix on the REAL K1 board.

The existing K1 matrix (output/kernels/ceiling/cross_framework_matrix_k1.md) is INNER-COMPUTE
(operand pack hoisted OUT of the timed region) at small cubes 32/64/128, where the specialized
ours-intrinsic kernel wins. The open question this script answers: at LARGE shapes with PACKING
INSIDE the timed region (the realistic end-use cost), where do the experts (OpenBLAS / XNNPACK)
re-take the lead?

For each shape in {32,64,128,256,512} and each source in
{openblas, xnnpack, ours-intrinsic, ours-baseline, ours-tiled} we measure TWO timing scopes:
  (a) inner_compute  — pack EXCLUDED (current protocol; head-to-head kernel-only).
  (b) full           — pack INCLUDED (pack + compute both timed; -DPACK_INCLUDED on the C drivers).

The expert / intrinsic drivers gained a `-DPACK_INCLUDED` compile-time guard that moves the
`read_csr(mcycle)` start BEFORE the pack loops; with the guard absent the timed region is exactly
the prior inner-compute scope (bit-identical to the published matrix). ours-baseline / ours-tiled
go through the COMPILER (model.o); their pack is fused INTO the compute kernel (no hoistable
pre-pack step), so for those two columns inner==full by construction — we measure once and label it.

All cells: SpacemiT clang riscv64-linux -march=rv64gcv -mabi=lp64d -O3 -ffast-math, k1_harness/util.h
FIRST (read_csr(mcycle)->rdtime 24 MHz), bit-exact VERIFY gate, min of N=3 reps. Honest not_run with
the blocker for anything that won't build / run / verify; never a fabricated tick. cycle_accurate=false
(rdtime is a real-silicon wall proxy; spike/FireSim remain the cycle-accurate authorities).
"""
from __future__ import annotations

import argparse, json, re, subprocess, tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.kernels.ceiling_drivers import run_expert_gemm as expert
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package

HERE = Path(repo_root()) / "merlin/python/merlin/kernels/ceiling_drivers"
K1H = HERE / "k1_harness"
REPO = Path(repo_root())
INTRINSIC_DRIVER = HERE / "ours_intrinsic_gemm_driver.c"

_K1_CFLAGS = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
              "-O3", "-ffast-math", "-DNDEBUG", "-std=gnu99", "-Wno-implicit-function-declaration"]

OURS_FORKS = (
    ("ours_baseline", []),
    ("ours_tiled", ["fused_vfmacc_tiled"]),
)


def _cc() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    return cc


def _deploy_run(binary: Path, tag: str, *, timeout: int = 600) -> tuple[str | None, str]:
    remote = f"/tmp/k1pack_{tag}"
    try:
        subprocess.run(["scp", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no", str(binary), f"{k1.K1_HOST}:{remote}"],
                       capture_output=True, text=True, timeout=180, check=True)
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


def _min_ticks(binp: Path, tag: str, reps: int, timeout: int) -> tuple[list[int] | None, str]:
    """Run a built binary `reps` times; return (sorted-ticks-list, detail) or (None, blocker)."""
    runs = []
    for _ in range(reps):
        console, detail = _deploy_run(binp, tag, timeout=timeout)
        if console is None:
            return None, detail
        if "VERIFY PASS" not in console:
            return None, f"verify did not pass; tail: {console.strip()[-300:]}"
        m = re.search(r"CYCLES\s+(\d+)", console)
        if not m:
            return None, "no CYCLES line"
        runs.append(int(m.group(1)))
    return runs, "ok"


def _scope_flags(pack_included: bool) -> list[str]:
    return ["-DPACK_INCLUDED"] if pack_included else []


def measure_expert(source: str, *, S: int, pack_included: bool, reps: int) -> dict:
    spec = expert._experts()[source]
    scope = "full" if pack_included else "inner_compute"
    base = {"op": "matmul", "dtype": spec["dtype"], "M": S, "N": S, "K": S,
            "source": source, "target": "k1", "mode": scope, "scope": scope,
            "pack_included": pack_included, "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "kernel_file": spec["kernel_file"], "measure_method": f"standalone_linux_{scope}"}
    cc = _cc()
    incs = [K1H, HERE] + [p for p in spec["incs"] if p != HERE]
    inc_flags = []
    for d in incs:
        inc_flags += ["-I", str(d)]
    shape = [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"]
    with tempfile.TemporaryDirectory(prefix="k1_expert_") as tmp:
        binp = Path(tmp) / f"{source}_gemm"
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *_scope_flags(pack_included), *shape,
               "-static", "-o", str(binp), str(spec["driver"]), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"],
                                    capture_output=True, text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-700:]}"}
        runs, detail = _min_ticks(binp, f"{source}_{scope}_{S}", reps, timeout=600)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": f"K1 real-silicon rdtime ticks; {scope}; bit-exact; min of {reps} reps"}


def measure_intrinsic(*, S: int, pack_included: bool, reps: int) -> dict:
    scope = "full" if pack_included else "inner_compute"
    base = {"op": "matmul", "dtype": "f32", "M": S, "N": S, "K": S,
            "source": "ours-intrinsic", "target": "k1", "mode": scope, "scope": scope,
            "pack_included": pack_included, "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "kernel_file": str(INTRINSIC_DRIVER.relative_to(REPO)),
            "kernel_desc": "register-blocked MR=4 accumulator-resident K-streaming riscv_vector.h LMUL=4",
            "measure_method": f"standalone_linux_{scope}"}
    cc = _cc()
    inc_flags = ["-I", str(K1H), "-I", str(HERE)]
    shape = [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"]
    with tempfile.TemporaryDirectory(prefix="k1_intrinsic_") as tmp:
        binp = Path(tmp) / "ours_intrinsic_gemm"
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *_scope_flags(pack_included), *shape,
               "-static", "-o", str(binp), str(INTRINSIC_DRIVER), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            try:
                p2 = subprocess.run([c for c in cmd if c != "-static"],
                                    capture_output=True, text=True, timeout=400)
            except (subprocess.TimeoutExpired, OSError) as e:
                return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
            if p2.returncode != 0 or not binp.is_file():
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-700:]}"}
        runs, detail = _min_ticks(binp, f"intrinsic_{scope}_{S}", reps, timeout=600)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": f"K1 real-silicon rdtime ticks; {scope}; bit-exact; min of {reps} reps"}


def measure_ours_compiler(run_id: str, features: list[str], *, S: int, reps: int) -> dict:
    """ours-baseline / ours-tiled go through the COMPILER (model.o). Their pack is fused into
    the kernel (no hoistable pre-pack), so inner==full by construction. We measure the
    inner-compute driver (ours_gemm_driver.c) and label both scopes from it."""
    from merlin.rvvgen import workloads
    from merlin.llvmlower import c_runtime, toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.runtime.backends import zephyr_model as zm

    base = {"op": "matmul", "dtype": "f32", "M": S, "N": S, "K": S, "source": run_id,
            "target": "k1", "mode": "compiler_fused_pack", "scope": "inner==full",
            "pack_included": "fused", "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "cycle_accurate": False, "compiler_features": features,
            "kernel_file": f"merlin RVV codegen fork (features={features or 'baseline'})",
            "measure_method": "standalone_linux_compiler_fused"}
    bundle = workloads.gen_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", M=S, N=S, K=S)
    hb = load_rvv_package(REPO / "artifacts/targets" / "rvv" / "hand_v0")
    pkg = replace(hb, run_id=run_id, compiler_features=list(features))
    cc = _cc()
    with tempfile.TemporaryDirectory(prefix="k1_ours_") as tmp:
        work = Path(tmp) / "work"
        work.mkdir(parents=True, exist_ok=True)
        md = Path(bundle)
        prepared = zm._prepare_model_mlir(md / "model.mlir", work, int8_compute=pkg.is_int8)
        feats = frozenset(pkg.compiler_features or []) or None
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
                           capture_output=True, text=True, timeout=400, check=True)
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
        shape = [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"]
        srcs = [str(HERE / "ours_gemm_driver.c"), str(cgen / "model_call.c"),
                str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *shape, "-static", "-o", str(binp), *srcs,
               "-lm", "-lpthread"]
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
        runs, detail = _min_ticks(binp, f"{run_id}_{S}", reps, timeout=900)
    if runs is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    return {**base, "ticks": min(runs), "ticks_runs": runs, "status": "pass", "reps": reps,
            "wall_ns_est": int(min(runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": "K1 real-silicon rdtime; compiler-emitted (pack fused into kernel; inner==full); "
                    f"bit-exact; min of {reps} reps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="32,64,128,256,512")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/gemm/large_shape_packing_k1.jsonl")
    ap.add_argument("--skip-ours-compiler", action="store_true",
                    help="skip ours-baseline/tiled compiler columns (they are ~10x slow + may OOM at 512)")
    a = ap.parse_args()
    shapes = [int(s) for s in a.shapes.split(",")]

    rows = []
    for S in shapes:
        for pack in (False, True):
            scope = "full(pack-incl)" if pack else "inner(pack-excl)"
            for src in ("openblas", "xnnpack"):
                print(f"--- expert {src} @ {S}^3 [{scope}] ---", flush=True)
                r = measure_expert(src, S=S, pack_included=pack, reps=a.reps)
                print("  ", r.get("status"), r.get("ticks"), r.get("ticks_runs", ""), r.get("blocker", ""), flush=True)
                rows.append(r)
            print(f"--- ours-intrinsic @ {S}^3 [{scope}] ---", flush=True)
            r = measure_intrinsic(S=S, pack_included=pack, reps=a.reps)
            print("  ", r.get("status"), r.get("ticks"), r.get("ticks_runs", ""), r.get("blocker", ""), flush=True)
            rows.append(r)
        if not a.skip_ours_compiler:
            for run_id, feats in OURS_FORKS:
                print(f"--- {run_id} (compiler, inner==full) @ {S}^3 ---", flush=True)
                r = measure_ours_compiler(run_id, feats, S=S, reps=a.reps)
                print("  ", r.get("status"), r.get("ticks"), r.get("ticks_runs", ""), r.get("blocker", ""), flush=True)
                rows.append(r)
        # incremental write after every shape so a late OOM doesn't lose earlier rows
        outp = Path(repo_root()) / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as fh:
            for rr in rows:
                fh.write(json.dumps(rr) + "\n")
        print(f"  (wrote {len(rows)} rows so far -> {outp})", flush=True)

    print(f"\nDONE: {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
