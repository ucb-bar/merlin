#!/usr/bin/env python
"""K1 silicon measurement of the compiler-emitted RVV intrinsic micro-kernel.

Ports `ceiling_drivers/ours_intrinsic_gemm_driver.c` (the "scalable-gap winner": register-blocked
MR=4, accumulator-resident, K-streaming, riscv_vector.h, 0 inner-loop spills) onto the REAL
SpacemiT K1 board, on the SAME footing as the existing K1 expert columns
(`scripts/k1_cross_framework.py::measure_expert_k1`):

  * SpacemiT clang, riscv64-linux, -march=rv64gcv -mabi=lp64d -O3 -ffast-math
  * k1_harness/util.h FIRST on the include path (read_csr(mcycle)->rdtime, printf->glibc)
  * inner-compute scope (operand pack hoisted OUT of the timed region — the driver already does)
  * bit-exact verified vs the driver's own scalar reference (VERIFY PASS gate)
  * timed with the board's delegated rdtime (24 MHz), N repeats, min ticks reported

The driver uses VL-agnostic riscv_vector.h (vsetvl + LMUL=4), so it adapts to the board's VLEN=256
at run time. We objdump the emitted vsetvli to confirm it actually uses the wider VL and record the
effective NR (= vsetvlmax_e32m4 = VLEN/32 * LMUL = 32 lanes @ VLEN=256).

Honest by construction: a build/run failure or a VERIFY FAIL yields a not_run row with the blocker,
never a fabricated tick number. cycle_accurate=false (rdtime is a real-silicon wall proxy; spike is
the cycle/functional authority).
"""
from __future__ import annotations

import argparse, json, subprocess, tempfile
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.common.driver_output import int_after, int_field
from merlin.mining import k1

HERE = Path(repo_root()) / "merlin/python/merlin/kernels/ceiling_drivers"
K1H = HERE / "k1_harness"
DRIVER = HERE / "ours_intrinsic_gemm_driver.c"

# IDENTICAL flags to scripts/k1_cross_framework.py::_K1_CFLAGS (the expert columns).
_K1_CFLAGS = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
              "-O3", "-ffast-math", "-DNDEBUG", "-std=gnu99", "-Wno-implicit-function-declaration"]


def _cc() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    return cc


def _deploy_run(binary: Path, tag: str, *, timeout: int = 300) -> tuple[str | None, str]:
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


def _build(binp: Path, *, M: int, N: int, K: int) -> tuple[bool, str]:
    cc = _cc()
    inc_flags = ["-I", str(K1H), "-I", str(HERE)]  # k1_harness/util.h FIRST
    shape = [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
    cmd = [str(cc), *inc_flags, *_K1_CFLAGS, *shape, "-static", "-o", str(binp), str(DRIVER), "-lm"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"build exec failed: {e}"
    if p.returncode != 0 or not binp.is_file():
        try:
            p2 = subprocess.run([c for c in cmd if c != "-static"], capture_output=True,
                                text=True, timeout=300)
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"build exec failed: {e}"
        if p2.returncode != 0 or not binp.is_file():
            return False, f"build failed rc={p.returncode}: {p.stderr.strip()[-700:]}"
    return True, "ok"


def _objdump_vsetvli(binp: Path) -> dict:
    """Confirm the emitted asm uses LMUL=4 e32 vsetvli (the VL-agnostic wide-VL path)."""
    root = k1._toolchain_root()
    objdump = None
    if root is not None:
        for name in ("riscv64-unknown-linux-gnu-objdump", "llvm-objdump"):
            cand = root / "bin" / name
            if cand.is_file():
                objdump = cand
                break
    if objdump is None:
        return {"vsetvli_found": None, "note": "objdump not located"}
    try:
        p = subprocess.run([str(objdump), "-d", str(binp)], capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"vsetvli_found": None, "note": f"objdump failed: {e}"}
    setvli = sorted({ln.strip() for ln in p.stdout.splitlines()
                     if "vsetvl" in ln and "e32" in ln and "m4" in ln})
    vfmacc = p.stdout.count("vfmacc.vf")
    return {"vsetvli_e32m4_variants": setvli[:6], "vfmacc_vf_count": vfmacc}


def measure_intrinsic_k1(*, M: int, N: int, K: int, reps: int = 3) -> dict:
    base = {"op": "matmul", "dtype": "f32", "M": M, "N": N, "K": K,
            "source": "ours-intrinsic", "target": "k1", "mode": "inner_compute",
            "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ, "cycle_accurate": False,
            "kernel_file": "merlin/python/merlin/kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c",
            "kernel_desc": "register-blocked MR=4, accumulator-resident, K-streaming, riscv_vector.h LMUL=4",
            "measure_method": "standalone_linux_inner_compute"}
    with tempfile.TemporaryDirectory(prefix="k1_intrinsic_") as tmp:
        binp = Path(tmp) / "ours_intrinsic_gemm"
        ok, detail = _build(binp, M=M, N=N, K=K)
        if not ok:
            return {**base, "ticks": None, "status": "not_run", "blocker": detail}
        asm = _objdump_vsetvli(binp)
        ticks_runs, console_last, mr = [], None, None
        for _ in range(reps):
            console, detail = _deploy_run(binp, f"intrinsic_{M}_{N}_{K}")
            console_last = console
            if console is None:
                return {**base, "ticks": None, "status": "not_run", "blocker": detail, "asm": asm}
            if "VERIFY PASS" not in console:
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": f"verify did not pass; console tail: {console.strip()[-300:]}",
                        "asm": asm}
            cyc = int_after(console, "CYCLES")
            if cyc is None:
                return {**base, "ticks": None, "status": "not_run",
                        "blocker": "no CYCLES/ticks line", "asm": asm}
            ticks_runs.append(cyc)
            mr_val = int_field(console, "MR")
            if mr_val is not None:
                mr = mr_val
    return {**base, "ticks": min(ticks_runs), "ticks_runs": ticks_runs, "status": "pass",
            "reps": reps, "MR": mr, "asm": asm,
            "wall_ns_est": int(min(ticks_runs) * 1e9 / k1.K1_TIMEBASE_HZ),
            "note": "K1 real-silicon rdtime ticks; inner-compute; bit-exact verified; min of N reps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="32,64,128")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="out/artifacts/measurements/k1_spacemit/gemm/intrinsic_microkernel_k1.jsonl")
    a = ap.parse_args()
    shapes = [int(s) for s in a.shapes.split(",")]

    rows = []
    for S in shapes:
        print(f"--- ours-intrinsic @ {S}^3 ---")
        r = measure_intrinsic_k1(M=S, N=S, K=S, reps=a.reps)
        print("  ", r.get("status"), r.get("ticks"), r.get("ticks_runs", ""), r.get("blocker", ""))
        if r.get("asm"):
            print("   asm:", r["asm"])
        rows.append(r)

    outp = Path(repo_root()) / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
