#!/usr/bin/env python
"""First real fp16 GEMM numbers on the SpacemiT K1, and the emitted-code proof that the
mixed-precision (f16 operand, f32 accumulator) contraction now reaches a FUSED widening MAC.

Per point this records:
  * ticks   -- rdtime on the compiled fill+matmul (min of reps), fill-only subtracted -> matmul-only;
  * instret -- retired instructions on the SAME bracket (perf_event_open);
  * the PER-ELEMENT correctness gate result parsed from the driver (cos > 0.9999 AND rel-L2 < 1e-2
    AND max-rel < 0.05, computed on the board against the f64-exact reference -- NOT the aggregate
    int8 gate, which accepts a broken f16-accumulating kernel);
  * the EMITTED inner-loop facts of `model.o`: vfwmacc / vfmacc / vfwcvt counts and the effective
    vtype from the decoder's vsetvli state machine (evidence the fusion is real, read from the
    instruction stream, not the schedule text).

Fail-closed: a build failure, a run failure, or a missing `VERIFY PASS` yields a `not_run` row
carrying the exact blocker -- never a timing.

Takes the host-wide board flock around the runs (the board is shared with other agents; an unlocked
run measures contention). Does NOT use k1.run_on_k1, so there is no double-lock. Unique remote-tag
prefix so concurrent campaigns do not collide.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merlin.common.artifacts import cache_dir
from merlin.common.driver_output import int_after, int_field
from merlin.common.paths import repo_root
from merlin.rvvgen import k1

import k1_cross_framework_ops as X

REPO = Path(repo_root())


def _emitted_facts(model_o: Path, keep: Path) -> dict:
    """Decode `model.o` and report the fused-MAC evidence: vfwmacc/vfmacc/vfwcvt counts across the
    whole kernel and the effective vtype histogram. Read from the decoded stream, never guessed."""
    from merlin.kernels.decode import rvv as _rvv
    from merlin.kernels.decode.objdump import objdump_bin
    try:
        p = subprocess.run([objdump_bin(), "-d", "--mattr=+v", str(model_o)],
                           capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as e:
        return {"emitted": None, "emitted_blocker": f"objdump failed: {e}"}
    if p.returncode != 0:
        return {"emitted": None, "emitted_blocker": f"objdump rc={p.returncode}"}
    (keep / "objdump.txt").write_text(p.stdout, encoding="utf-8")
    stream = _rvv.decode_text(p.stdout)
    counts: dict[str, int] = {}
    for i in stream.insns:
        m = i.raw.mnemonic
        for k in ("vfwmacc", "vfmacc", "vfwmul", "vfmul", "vfadd", "vfwcvt", "vfncvt"):
            if m.startswith(k):
                counts[m] = counts.get(m, 0) + 1
    return {"emitted": {"counts": counts, "vtype_histogram": stream.vtype_histogram()}}


def _run_ours(S: int, reps: int, tag: str, keep: Path, feats: list[str]) -> dict:
    base = {"op": "f16_gemm", "dtype": "f16", "M": S, "N": S, "K": S, "target": "k1",
            "mode": "inner_compute", "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "source": "ours_fp16_f32acc", "compiler_features": feats,
            "gate": "cos>0.9999 AND rel_l2<1e-2 AND max_rel<0.05 (per-element, vs f64-exact)"}
    from merlin.rvvgen import workloads
    bundle = workloads.gen_matmul_f16(cache_dir("rvv_workloads"), M=S, N=S, K=S)
    work = keep / "work"
    work.mkdir(parents=True, exist_ok=True)
    model_o, cgen, err = X._lower_ours(bundle, tag, feats, int8=False, vectorize=True, work=work)
    if err is not None:
        return {**base, "ticks": None, "status": "not_run", "blocker": err}
    base.update(_emitted_facts(model_o, keep))

    cc = X._cc()
    rt = REPO / "merlin/runtime/c"
    abi = REPO / "merlin/runtime/abi"
    inc = []
    for d in (X.K1H, X.HERE, cgen, rt):
        inc += ["-I", str(d)]
    binp = keep / f"{tag}.elf"
    srcs = [str(X.HERE / "ours_gemm_f16_driver.c"), str(cgen / "model_call.c"),
            str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
    cmd = [str(cc), *inc, *X._K1_CFLAGS, f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}",
           "-static", "-o", str(binp), *srcs, "-lm", "-lpthread"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
    if p.returncode != 0 or not binp.is_file():
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"link failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}

    best = None
    ticks_all = []
    with k1.board_lock():
        for rep in range(reps):
            console, detail = X._deploy_run(binp, f"{tag}_{rep}", timeout=900)
            r = X._parse(base, console, detail)
            if r["status"] != "pass":
                return r
            r["cos"] = int_after(console, "COS")
            r["rel_l2"] = int_after(console, "REL_L2")
            r["max_rel"] = int_after(console, "MAX_REL")
            r["instret"] = int_after(console, "INSTRET")
            ticks_all.append(r["ticks"])
            if best is None or r["ticks"] < best["ticks"]:
                best = r
    best["reps"] = reps
    best["timing"] = "min_of_reps"
    best["ticks_all"] = ticks_all
    if len(ticks_all) > 1:
        spread = (max(ticks_all) - min(ticks_all)) / min(ticks_all)
        best["run_spread_pct"] = round(spread * 100, 3)
    return best


def _run_xnn_f16(S: int, reps: int, tag: str, keep: Path) -> dict:
    """XNNPACK RVV fp16 GEMM (native f16 accumulate) -- the head-to-head, with its accuracy caveat.

    Built standalone with a MINIMAL shim for `src/xnnpack/gemm.h` (the real header pulls
    xnnpack.h -> pthreadpool.h, not vendored here; the 7x4v kernel needs only xnn_float16 + the
    minmax params struct + two branch-hint macros). Compiled with an f16-capable march
    (`+zfh,+zvfh`) since clang-19 does not turn those on from plain `rv64gcv`. VERIFY here is
    against XNN's OWN fp16-accumulate reference; COS/MAX_REL vs f64-exact carry the accuracy caveat.
    """
    base = {"op": "f16_gemm", "dtype": "f16", "M": S, "N": S, "K": S, "target": "k1",
            "mode": "inner_compute", "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "source": "xnnpack_f16_native_accumulate",
            "kernel_file": "tmp/kernels/XNNPACK/src/f16-gemm/gen/f16-gemm-7x4v-minmax-rvvfp16arith.c",
            "accuracy_caveat": "accumulates in fp16 (would FAIL our per-element gate); ours is f32-acc"}
    cc = X._cc()
    shim = REPO / "out/artifacts/cache/xnn_shim"
    xnnroot = REPO / "tmp/kernels/XNNPACK"
    inc = ["-I", str(shim)]
    for d in (X.K1H, X.HERE, xnnroot, xnnroot / "src"):
        inc += ["-I", str(d)]
    binp = keep / f"{tag}.elf"
    cflags = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d",
              "-O3", "-ffast-math", "-DNDEBUG", "-std=gnu99", "-Wno-implicit-function-declaration"]
    cmd = [str(cc), *inc, *cflags, f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}",
           "-static", "-o", str(binp), str(X.HERE / "xnnpack_gemm_f16_driver_7x4v.c"), "-lm"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {**base, "ticks": None, "status": "not_run", "blocker": f"build exec failed: {e}"}
    if p.returncode != 0 or not binp.is_file():
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"build failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}
    best = None
    ticks_all = []
    with k1.board_lock():
        for rep in range(reps):
            console, detail = X._deploy_run(binp, f"{tag}_{rep}", timeout=900)
            r = X._parse(base, console, detail)
            if r["status"] != "pass":
                return r
            r["cos"] = int_after(console, "COS")
            r["max_rel"] = int_after(console, "MAX_REL")
            r["instret"] = int_after(console, "INSTRET")
            ticks_all.append(r["ticks"])
            if best is None or r["ticks"] < best["ticks"]:
                best = r
    best["reps"] = reps
    best["timing"] = "min_of_reps"
    best["ticks_all"] = ticks_all
    if len(ticks_all) > 1:
        best["run_spread_pct"] = round((max(ticks_all) - min(ticks_all)) / min(ticks_all) * 100, 3)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="128")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag-prefix", default="f16b_")
    ap.add_argument("--features", default="erase_self_copy,accumulator_resident_microkernel_v3")
    ap.add_argument("--control", action="store_true",
                    help="also run the SAME binary a second batch for the noise-floor control")
    ap.add_argument("--xnn", action="store_true", help="also race the XNNPACK fp16 kernel")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    feats = [f for f in a.features.split(",") if f]
    shapes = [int(s) for s in a.shapes.split(",")]
    root = cache_dir("fp16_gemm")
    rows = []
    for S in shapes:
        tag = f"{a.tag_prefix}ours_{S}"
        keep = root / tag
        keep.mkdir(parents=True, exist_ok=True)
        print(f"--- ours fp16 {S}^3 ---", flush=True)
        r = _run_ours(S, a.reps, tag, keep, feats)
        r["artifact_dir"] = str(keep)
        em = (r.get("emitted") or {}).get("counts", {})
        print("   ", r["status"], "ticks=", r.get("ticks"), "instret=", r.get("instret"),
              "cos(x1e7)=", r.get("cos"), "max_rel(x1e7)=", r.get("max_rel"),
              "vfwmacc=", em.get("vfwmacc.vf", 0), "vfmacc=", em.get("vfmacc.vv", 0),
              "spread%=", r.get("run_spread_pct"), r.get("blocker", ""), flush=True)
        rows.append(r)
        if a.xnn:
            print(f"--- xnnpack fp16 (native f16-accumulate) {S}^3 ---", flush=True)
            keepx = root / f"{a.tag_prefix}xnn_{S}"
            keepx.mkdir(parents=True, exist_ok=True)
            rx = _run_xnn_f16(S, a.reps, f"{a.tag_prefix}xnn_{S}", keepx)
            rx["artifact_dir"] = str(keepx)
            print("   ", rx["status"], "ticks=", rx.get("ticks"), "instret=", rx.get("instret"),
                  "cos(x1e7)=", rx.get("cos"), "max_rel(x1e7)=", rx.get("max_rel"),
                  "spread%=", rx.get("run_spread_pct"), rx.get("blocker", ""), flush=True)
            rows.append(rx)
        if a.control and r["status"] == "pass":
            print(f"--- CONTROL (identical binary) fp16 {S}^3 ---", flush=True)
            rc = _run_ours(S, a.reps, f"{a.tag_prefix}ctrl_{S}", root / f"{a.tag_prefix}ctrl_{S}", feats)
            rc["control_of"] = tag
            print("   ", rc["status"], "ticks=", rc.get("ticks"),
                  "spread%=", rc.get("run_spread_pct"), flush=True)
            rows.append(rc)

    outp = Path(a.out) if a.out else (root / "fp16_gemm.jsonl")
    with outp.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
