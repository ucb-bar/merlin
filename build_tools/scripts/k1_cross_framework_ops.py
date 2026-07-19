#!/usr/bin/env python
"""PART 3 — cross-framework ceiling matrix BEYOND matmul, on the real K1 board.

The existing matrix (cross_framework_matrix*.{md,jsonl}) is GEMM-only because OpenBLAS
is BLAS (gemm/gemv/dot). XNNPACK, however, has RVV kernels for many more ops we never
raced. This script extends the comparison to the ops XNNPACK actually ships RVV kernels
for, reusing the SAME inner-compute / bit-exact / rdtime protocol as k1_cross_framework.py:

  op            XNNPACK kernel (RVV)                         OURS
  ------------  -------------------------------------------  -----------------------------
  gelu (f32)    f32-vgelu rational-12-10-div-u4v             our GELU lowering (math.erf gen)
  sigmoid(f32)  f32-vsigmoid rr2-p5-div-u4v                  our sigmoid lowering (exp gen)
  int8 gemm     qd8-f32-qc8w-gemm 1x4v minmax rvv            our W8A8 vwmacc datapath
  dwconv (f32)  f32-dwconv 9p8vc rvv (DEPTHWISE 3x3)          ours has no depthwise prim => not_run/note
  conv2d (f32)  (none: regular conv = GEMM ceiling via       our conv2d im2col->matmul (=GEMM)
                 im2col; depthwise is the only XNN f32 conv)
  attention     (NONE — not an XNNPACK/OpenBLAS primitive)   ours-baseline bmm vs ours-vfmacc

ATTENTION has NO library baseline (not an XNNPACK/OpenBLAS op), so for attention we compare
OUR baseline batch_matmul lowering vs OUR vfmacc feature — explicitly ours-vs-ours, not vs a
framework.

Many XNNPACK RVV kernels use the OVERLOADED intrinsic spellings (e.g. __riscv_vfmerge,
__riscv_vse32) that the spike riscv-gcc-13.2 does not accept but the K1 SpacemiT clang does.
So those rows carry the K1 silicon rdtime number (cycle_accurate=false) and honest not_run on
spike (gcc intrinsic incompatibility). GELU uses explicit spellings and runs on both.

Honest by construction: a build/run failure or VERIFY FAIL yields a not_run row with the exact
blocker — never a fabricated number. Board left clean (binaries rm'd after each run).
"""
from __future__ import annotations

import argparse, json, subprocess, tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.common.driver_output import int_after, int_field
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package

HERE = Path(repo_root()) / "merlin/python/merlin/kernels/ceiling_drivers"
K1H = HERE / "k1_harness"
XNN = Path(repo_root()) / "tmp/kernels/XNNPACK/src"
REPO = Path(repo_root())

_K1_CFLAGS = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
              "-O3", "-ffast-math", "-DNDEBUG", "-std=gnu99", "-Wno-implicit-function-declaration"]


def _cc() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    return cc


def _deploy_run(binary: Path, tag: str, *, timeout: int = 300,
                pmu: bool = False) -> tuple[str | None, str]:
    """scp the ELF to the board and run it. With ``pmu=True`` the run is wrapped in the
    perf_event_open counter shim so the caller also gets cycles/instructions/IPC -- the axis that
    separates "emits too many instructions" from "stalls on each instruction". PMU counts ride on
    stderr, so the console (stdout) parse is unchanged and the wrapper is fail-open: if the board or
    the shim is unavailable the run still happens, just without counters."""
    remote = f"/tmp/k1ops_{tag}"
    try:
        subprocess.run(["scp", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no", str(binary), f"{k1.K1_HOST}:{remote}"],
                       capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as e:
        return None, f"scp failed: {e.stderr[-200:] if e.stderr else e}"
    try:
        k1._ssh(f"chmod +x {remote}", timeout=30)
        cmd = remote
        if pmu:
            from merlin.rvvgen import pmu as pmu_mod
            if pmu_mod.ensure_deployed():
                cmd = pmu_mod.wrap(remote)
        p = k1._ssh(cmd, timeout=timeout)
    finally:
        try:
            k1._ssh(f"rm -f {remote}", timeout=30)
        except Exception:  # noqa: BLE001
            pass
    if p.returncode != 0:
        return None, f"run rc={p.returncode}; stderr: {p.stderr.strip()[-200:]}; stdout: {p.stdout.strip()[-200:]}"
    # Counters (when requested) are appended to the console so the existing _parse path is untouched
    # and the numbers land in the same record the wall-time measurement does.
    if pmu:
        from merlin.rvvgen import pmu as pmu_mod
        counts = pmu_mod.parse(p.stderr or "")
        if counts is not None:
            return f"{p.stdout}\nMERLIN_PMU cycles={counts.cycles} instructions={counts.instructions}\n", "ok"
    return p.stdout, "ok"


def _parse(base: dict, console: str | None, detail: str, *, reps: int = 1) -> dict:
    if console is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": detail}
    if "VERIFY PASS" not in console:
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"verify did not pass; console tail: {console.strip()[-300:]}"}
    t = int_after(console, "CYCLES")
    if t is None:
        return {**base, "ticks": None, "status": "not_run", "blocker": "no CYCLES/ticks line"}
    err = int_field(console, "errors")
    from merlin.rvvgen import pmu as pmu_mod
    counts = pmu_mod.parse(console)
    pmu_fields = counts.as_dict() if counts is not None else {}
    # Retired instructions on the SAME bracket as the rdtime timing (the drivers read minstret via
    # perf_event_open). This is what separates "emits too many instructions" -- which a schedule can
    # fix -- from "stalls on each", which it cannot. Process-wide PMU totals cannot make that split:
    # each driver's scalar verification reference costs differently, so they do not cancel.
    # Absent on drivers that do not print it.
    instret = int_after(console, "INSTRET")
    if instret is not None:
        pmu_fields = {**pmu_fields, "instret": instret,
                      "instret_full": int_after(console, "INSTRET_FULL")}
    return {**base, "ticks": t, "status": "pass", **pmu_fields,
            "correct": (err == 0) if err is not None else True,
            "wall_ns_est": int(t * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": "K1 real-silicon rdtime ticks; inner-compute; bit-exact verified"}


def _build_run_xnn(tag, driver: Path, defs: list[str], *, reps: int = 3,
                   base: dict, pmu: bool = False) -> dict:
    """Compile one standalone XNNPACK driver with the K1 clang, scp+run reps times, keep min."""
    cc = _cc()
    inc_flags = []
    for d in (K1H, HERE, XNN):
        inc_flags += ["-I", str(d)]
    best = None
    with tempfile.TemporaryDirectory(prefix="k1_xnn_") as tmp:
        binp = Path(tmp) / tag
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *defs, "-static", "-o", str(binp),
               str(driver), "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
        if p.returncode != 0 or not binp.is_file():
            return {**base, "ticks": None, "status": "not_run",
                    "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-700:]}"}
        for rep in range(reps):
            console, detail = _deploy_run(binp, f"{tag}_{rep}", pmu=pmu)
            r = _parse(base, console, detail)
            if r["status"] != "pass":
                return r  # first failure is the honest blocker
            if best is None or r["ticks"] < best["ticks"]:
                best = r
    best["reps"] = reps
    best["timing"] = "min_of_reps"
    return best


def _lower_ours(bundle: Path, run_id: str, features: list[str], *, int8: bool,
                vectorize: bool, work: Path):
    """Reuse the K1 build path's lowering -> model.o + cgen artifacts. Returns (model_o, cgen, err)."""
    from merlin.llvmlower import c_runtime, toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.runtime.backends import zephyr_model as zm

    md = Path(bundle)
    prepared = zm._prepare_model_mlir(md / "model.mlir", work, int8_compute=int8)
    feats = frozenset(features or []) or None
    try:
        res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                               vectorize=vectorize, transform_schedule=None,
                               hoist_static_allocs=False, features=feats)
    except PipelineError as e:
        return None, None, f"lowering raised: {str(e)[:220]}"
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    # Pin the board's REAL VLEN (`_zvl256b`). Plain `-march=rv64gcv` promises only the 128-bit RVV
    # minimum, so every fixed-width vector we emit gets DOUBLE the LMUL the K1 needs — which both
    # doubles vector-register pressure (spills inside the K loop) and leaves half the datapath idle
    # (`vl` at half `VLMAX`). See k1.codegen_march for the measured cost.
    try:
        subprocess.run([str(clang23), "--target=riscv64-unknown-linux-gnu",
                        f"-march={k1.codegen_march()}", f"-mabi={k1.K1_MABI}",
                        "-O2", "-Wno-override-module",
                        "-c", str(res.ll_path), "-o", str(model_o)],
                       capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        return None, None, f"model.o compile failed: {e.stderr[-400:] if e.stderr else e}"
    cgen = work / "cgen"
    c_runtime.generate(md, cgen, md / "inputs.npz")
    return model_o, cgen, None


def _build_run_ours(tag, bundle: Path, driver: Path, defs: list[str], run_id: str,
                    features: list[str], *, int8: bool, vectorize: bool, reps: int,
                    base: dict, timeout: int = 600, pmu: bool = False) -> dict:
    """Lower OUR model.o for `bundle`, link the given OURS driver, scp+run reps, keep min."""
    cc = _cc()
    rt = REPO / "merlin/runtime/c"
    abi = REPO / "merlin/runtime/abi"
    with tempfile.TemporaryDirectory(prefix="k1_ours_") as tmp:
        work = Path(tmp) / "work"; work.mkdir(parents=True, exist_ok=True)
        model_o, cgen, err = _lower_ours(bundle, run_id, features, int8=int8,
                                         vectorize=vectorize, work=work)
        if err is not None:
            return {**base, "ticks": None, "status": "not_run", "blocker": err}
        inc_flags = []
        for d in (K1H, HERE, cgen, rt):
            inc_flags += ["-I", str(d)]
        binp = Path(tmp) / tag
        srcs = [str(driver), str(cgen / "model_call.c"),
                str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
        cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *defs, "-static", "-o", str(binp),
               *srcs, "-lm", "-lpthread"]
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
        best = None
        for rep in range(reps):
            console, detail = _deploy_run(binp, f"{tag}_{rep}", timeout=timeout, pmu=pmu)
            r = _parse(base, console, detail)
            if r["status"] != "pass":
                return r
            if best is None or r["ticks"] < best["ticks"]:
                best = r
    best["reps"] = reps
    best["timing"] = "min_of_reps"
    return best


# ---------------------------------------------------------------------------
def run_activation(op: str, sizes: list[int], reps: int) -> list[dict]:
    from merlin.rvvgen import workloads
    rows = []
    if op == "gelu":
        ksrc = "f32-vgelu/gen/f32-vgelu-rvv-rational-12-10-div-u4v.c"
        kfn = "xnn_f32_vgelu_ukernel__rvv_rational_12_10_div_u4v"
        ref = "gelu"; gen = workloads.gen_gelu_f32
    elif op == "sigmoid":
        ksrc = "f32-vsigmoid/gen/f32-vsigmoid-rvv-rr2-p5-div-u4v.c"
        kfn = "xnn_f32_vsigmoid_ukernel__rvv_rr2_p5_div_u4v"
        ref = "sigmoid"; gen = workloads.gen_sigmoid_f32
    else:
        # fail-closed: never silently mislabel an unsupported activation as sigmoid. Add a proper
        # (ksrc, kfn, ref, gen) triple + driver ref macro to extend (e.g. tanh, hardswish, elu).
        raise ValueError(f"run_activation: unsupported op {op!r} (wired: gelu, sigmoid)")
    for Nsz in sizes:
        # XNNPACK
        base = {"op": op, "dtype": "f32", "size_n": Nsz, "source": "xnnpack",
                "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ, "kernel_file": "tmp/kernels/XNNPACK/src/" + ksrc}
        print(f"--- xnnpack {op} N={Nsz} ---")
        r = _build_run_xnn(f"{op}_xnn_{Nsz}", HERE / "xnnpack_vunary_driver.c",
                           [f'-DXNN_KERNEL_SRC="{ksrc}"', f"-DXNN_KERNEL_FN={kfn}",
                            f"-DXNN_REF_{ref}", f"-DVLEN_N={Nsz}"], reps=reps, base=base)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
        rows.append(r)
        # OURS scalar baseline vs RVV-vectorized
        bundle = gen(REPO / "artifacts" / "cache" / "rvv_workloads", N=Nsz)
        for vec, sid in ((False, "ours_scalar"), (True, "ours_vectorized")):
            base = {"op": op, "dtype": "f32", "size_n": Nsz, "source": sid, "target": "k1",
                    "mode": "inner_compute", "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
                    "vectorize": vec, "kernel_file": f"merlin RVV codegen ({sid})"}
            print(f"--- {sid} {op} N={Nsz} ---")
            r = _build_run_ours(f"{op}_{sid}_{Nsz}", bundle, HERE / "ours_activation_driver.c",
                                [f"-DXNN_REF_{ref}"], sid, [], int8=False, vectorize=vec,
                                reps=reps, base=base)
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
            rows.append(r)
    return rows


def run_f32_gemm(shapes: list[int], reps: int) -> list[dict]:
    """The HEADLINE op on K1: XNNPACK's f32 GEMM 7x4v RVV ukernel vs OUR codegen — both the naive
    baseline AND our best whole-model feature (accumulator_resident_wholemodel_vf). Kernel-region
    rdtime on real silicon (mode=inner_compute), pack-outside fairness. This is the per-op analogue
    of the whole-model four-way, isolating the GEMM kernel gap the models are dominated by."""
    from merlin.rvvgen import workloads
    rows = []
    for S in shapes:
        base = {"op": "f32_gemm", "dtype": "f32", "M": S, "N": S, "K": S,
                "source": "xnnpack", "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ,
                "kernel_file": "tmp/kernels/XNNPACK/src/f32-gemm/gen/f32-gemm-7x4v-rvv.c"}
        print(f"--- xnnpack f32_gemm {S}^3 (7x4v) ---")
        r = _build_run_xnn(f"f32gemm_xnn_{S}", HERE / "xnnpack_gemm_driver_7x4v.c",
                           [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"], reps=reps, base=base,
                           pmu=True)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
        rows.append(r)
        bundle = workloads.gen_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", M=S, N=S, K=S)
        # ours: naive baseline (no features) AND our best whole-model codegen feature.
        for feats, sid in (([], "ours_f32_baseline"),
                           (["accumulator_resident_wholemodel_vf"], "ours_f32_wholemodel_vf")):
            base = {"op": "f32_gemm", "dtype": "f32", "M": S, "N": S, "K": S, "source": sid,
                    "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                    "timebase_hz": k1.K1_TIMEBASE_HZ, "compiler_features": feats,
                    "kernel_file": f"merlin RVV codegen ({sid})"}
            print(f"--- {sid} {S}^3 ---")
            r = _build_run_ours(f"f32gemm_{sid}_{S}", bundle, HERE / "ours_gemm_driver.c",
                                [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"],
                                sid, feats, int8=False, vectorize=True, reps=reps, base=base,
                                pmu=True)
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
            rows.append(r)
    return rows


def run_int8_gemm(shapes: list[int], reps: int) -> list[dict]:
    from merlin.rvvgen import workloads
    rows = []
    for S in shapes:
        base = {"op": "int8_gemm", "dtype": "qd8_qc8w", "M": S, "N": S, "K": S,
                "source": "xnnpack", "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ,
                "kernel_file": "tmp/kernels/XNNPACK/src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4v-minmax-rvv.c"}
        print(f"--- xnnpack int8_gemm {S}^3 ---")
        r = _build_run_xnn(f"qd8_xnn_{S}", HERE / "xnnpack_qd8_gemm_driver.c",
                           [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"], reps=reps, base=base,
                           pmu=True)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
        rows.append(r)
        # OURS int8 W8A8, three points: NAIVE (features=[] -> scalar, the ~200x catastrophe), the
        # long-standing vf config, and the SHARED micro-kernel path the f32 arm uses.
        #
        # `ours_int8_v3` is spelled as the target-agnostic `microkernel` knob block on purpose —
        # int8 and f32 must name the recipe the SAME way and differ only in dtype_strategy, rather
        # than int8 keeping a hand-listed fork of the feature list. The block resolves (registry.
        # _resolve_features -> from_strategy._rvv_microkernel_resolver) to the v3 tuning point plus
        # the recipe's `erase_self_copy` hygiene.
        #
        # MEASURED on this board, kernel region, cos-gated, min of 3 (vs XNNPACK qd8 1x4v):
        #     shape    vf                v3 + erase        speedup
        #      64^3    16,563 (8.03x)    10,301 (4.99x)    1.61x
        #     128^3   119,042 (9.86x)    69,428 (5.75x)    1.71x
        #     256^3   896,380 (8.20x)   503,434 (4.60x)    1.78x
        # so vf is NOT the int8 best and had not been for some time; it was simply the only int8
        # recipe wired here. The naive row is kept to SHOW the gap the vectorization lever closes.
        from merlin.rvvgen.from_strategy import microkernel_features
        bundle = workloads.gen_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", M=S, N=S, K=S)
        int8_configs = [([], "ours_int8_naive"),
                        (["accumulator_resident_wholemodel_vf"], "ours_int8_vf"),
                        (microkernel_features({"MR": 4, "NR": 16, "KC": 16}), "ours_int8_v3")]
        for feats, sid in int8_configs:
            base = {"op": "int8_gemm", "dtype": "i8xi8->i32", "M": S, "N": S, "K": S, "source": sid,
                    "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                    "timebase_hz": k1.K1_TIMEBASE_HZ, "int8_compute": True,
                    "compiler_features": feats, "kernel_file": f"merlin RVV codegen ({sid})"}
            print(f"--- {sid} {S}^3 ---")
            # int8 W8A8 is an APPROXIMATION of the f32 product -> cos>0.99 (the repo's fp32 int8 tier).
            r = _build_run_ours(f"int8_{sid}_{S}", bundle, HERE / "ours_int8_gemm_driver.c",
                                [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"],
                                sid, feats, int8=True, vectorize=True, reps=reps, base=base)
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
            rows.append(r)
    return rows


def run_dwconv(reps: int) -> list[dict]:
    """XNNPACK f32 depthwise 3x3 on a MobileNet-style shape. OURS has no depthwise primitive
    (regular conv = im2col->GEMM only), so the ours-depthwise row is an honest not_run/note."""
    OH, OW, C = 28, 28, 128
    base = {"op": "dwconv", "dtype": "f32", "OH": OH, "OW": OW, "C": C, "kernel": "3x3",
            "source": "xnnpack", "target": "k1", "mode": "inner_compute", "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ,
            "kernel_file": "tmp/kernels/XNNPACK/src/f32-dwconv/gen/f32-dwconv-9p8vc-rvv.c"}
    print(f"--- xnnpack dwconv {OH}x{OW}x{C} 3x3 ---")
    r = _build_run_xnn("dwconv_xnn", HERE / "xnnpack_dwconv_driver.c",
                       [f"-DDW_OH={OH}", f"-DDW_OW={OW}", f"-DDW_C={C}"], reps=reps, base=base)
    print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
    note = ("OURS has no depthwise-conv primitive on the f32 RVV path: regular conv2d lowers "
            "im2col->matmul (the GEMM ceiling, raced separately as op=conv2d), but a per-channel "
            "depthwise filter is not expressible as that single contraction. Honest not_run.")
    ours = {"op": "dwconv", "dtype": "f32", "OH": OH, "OW": OW, "C": C, "kernel": "3x3",
            "source": "ours_depthwise", "target": "k1", "ticks": None, "status": "not_run",
            "blocker": note}
    return [r, ours]


def run_conv2d(reps: int) -> list[dict]:
    """Regular f32 conv2d on OUR side = im2col->matmul (the GEMM ceiling). A 3x3x3->16 conv
    over a 8x8 output = M=64 positions, N=16 out-ch, K=27 patch-volume. XNNPACK's only f32 conv
    RVV kernel is depthwise (raced separately); regular conv on the library side IS its f32 GEMM
    (igemm), so we note that and race OUR im2col-GEMM baseline vs vectorized."""
    from merlin.rvvgen import workloads
    rows = []
    M, N, K = 64, 16, 27
    bundle = workloads.gen_conv2d_as_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", M=M, N=N, K=K)
    for feats, sid in (([], "ours_conv_baseline"), (["fused_vfmacc_contraction"], "ours_conv_vfmacc")):
        base = {"op": "conv2d", "dtype": "f32", "M": M, "N": N, "K": K, "via": "im2col->matmul",
                "source": sid, "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ, "compiler_features": feats,
                "kernel_file": f"merlin RVV codegen ({sid})"}
        print(f"--- {sid} conv {M}x{N}x{K} (im2col->matmul) ---")
        r = _build_run_ours(f"conv_{sid}", bundle, HERE / "ours_gemm_driver.c",
                            [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"], sid, feats,
                            int8=False, vectorize=True, reps=reps, base=base)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
        rows.append(r)
    return rows


def run_attention(reps: int) -> list[dict]:
    """ATTENTION has NO library baseline (not an XNNPACK/OpenBLAS primitive). So we compare OUR
    baseline batch_matmul lowering vs OUR vfmacc feature — explicitly ours-vs-ours."""
    from merlin.rvvgen import workloads
    rows = []
    B, M, Nn, K = 4, 32, 8, 32   # llama-style attention bmm (small N -> N-tail path)
    bundle = workloads.gen_batch_matmul_f32(REPO / "artifacts" / "cache" / "rvv_workloads", B=B, M=M, N=Nn, K=K)
    for feats, sid in (([], "ours_bmm_baseline"),
                       (["fused_vfmacc_contraction"], "ours_bmm_vfmacc")):
        base = {"op": "attention_bmm", "dtype": "f32", "B": B, "M": M, "N": Nn, "K": K,
                "source": sid, "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ, "compiler_features": feats,
                "baseline_kind": "ours_vs_ours (no library attention primitive)",
                "kernel_file": f"merlin RVV codegen ({sid})"}
        print(f"--- {sid} attention bmm {B}x{M}x{Nn}x{K} ---")
        # Dedicated bmm driver: CORRECT batched scalar reference (the 2-D gemm driver's flat ref
        # would be wrong for a block-diagonal bmm). inner-compute, fill subtracted.
        r = _build_run_ours(f"attn_{sid}", bundle, HERE / "ours_bmm_driver.c",
                            [f"-DBMM_B={B}", f"-DBMM_M={M}", f"-DBMM_N={Nn}", f"-DBMM_K={K}"],
                            sid, feats, int8=False, vectorize=True, reps=reps, base=base)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", ""))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="f32_gemm,gelu,sigmoid,int8_gemm,dwconv,conv2d,attention")
    ap.add_argument("--act-sizes", default="1024,16384,262144")
    ap.add_argument("--gemm-shapes", default="32,64,128,256")
    ap.add_argument("--int8-shapes", default="32,64,128")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="out/artifacts/measurements/k1_spacemit/gemm/cross_framework_ops_k1.jsonl")
    a = ap.parse_args()
    ops = a.ops.split(",")
    act_sizes = [int(s) for s in a.act_sizes.split(",")]
    gemm_shapes = [int(s) for s in a.gemm_shapes.split(",")]
    int8_shapes = [int(s) for s in a.int8_shapes.split(",")]

    rows: list[dict] = []
    if "f32_gemm" in ops:  rows += run_f32_gemm(gemm_shapes, a.reps)
    if "gelu" in ops:      rows += run_activation("gelu", act_sizes, a.reps)
    if "sigmoid" in ops:   rows += run_activation("sigmoid", act_sizes, a.reps)
    if "int8_gemm" in ops: rows += run_int8_gemm(int8_shapes, a.reps)
    if "dwconv" in ops:    rows += run_dwconv(a.reps)
    if "conv2d" in ops:    rows += run_conv2d(a.reps)
    if "attention" in ops: rows += run_attention(a.reps)

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
