#!/usr/bin/env python
"""Is the multicore RVV lowering actually FAST, or just correct? — measured on the K1 board.

Correctness is gated elsewhere (bit-exactness, 1 hart vs N). This answers the separate and
easily-fudged question: does fanning the model across cores actually buy time, and if not, why.

WHY THE K1 AND NOT A SIMULATOR. spike cannot answer this at all — it simulates every hart at
full speed, so a spinning or idle core costs simulated cycles exactly like a working one and
"speedup" there is meaningless. RTL simulation is cycle-accurate but ~10^4 cycles/s, which is
several orders of magnitude short of a whole inference. The K1 is real RVV silicon (VLEN=256,
8 cores) running a real OpenMP runtime (the cross-built libomp), so it is the only oracle that
can time a whole model on many cores.

METHOD. Build ONE binary with the composed vector+OpenMP lowering, then vary OMP_NUM_THREADS at
run time. Same code, same schedule, same weights — the only variable is how many cores execute
the parallel regions, which is what makes the resulting curve attributable.

WHAT IT REPORTS, and what would indict the implementation:
  * speedup(T) vs T=1, and parallel efficiency speedup/T. Efficiency collapsing as T grows is
    the signature of per-region synchronization cost dominating the region's work.
  * the implied Amdahl serial fraction, derived from the measured speedup. Merlin's lowering
    leaves every REDUCTION serial (softmax/norms), so a nonzero floor is expected and the
    question is whether it is that floor or something worse.
  * n_parallel_regions, counted from the emitted IR. A whole transformer lowers to hundreds of
    parallel regions, i.e. hundreds of fork/join barriers per inference; combined with the
    per-region cost implied by the curve this says whether the granularity is the problem.

Fail-closed: a thread count whose output does not pass the accuracy gate carries NO timing.

Usage:
    build_tools/scripts/k1_multicore_scaling.py --model tiny_llama --dtype int8 \
        --threads 1,2,4,8 -n 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from merlin.baselines import bundle as _bundle
from merlin.common.artifacts import artifacts_dir
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm


def count_parallel_regions(ll_path: Path) -> dict[str, int]:
    """Fork/join sites and outlined regions in the emitted IR — the granularity evidence."""
    forks = outlined = 0
    for line in ll_path.read_text().splitlines():
        if "@__kmpc_fork_call(" in line and not line.startswith("declare"):
            forks += 1
        elif line.startswith("define") and "omp_par" in line:
            outlined += 1
    return {"fork_sites": forks, "outlined_regions": outlined}


def _refs(mdir: Path) -> dict:
    refs = {"fp32": np.load(mdir / "golden.npy")}
    w8 = mdir / "golden_w8a8.npy"
    if w8.is_file():
        refs["w8a8"] = np.load(w8)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny_llama")
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("-n", type=int, default=3, help="repeats per thread count (median reported)")
    ap.add_argument("--package", default=None, help="rvv package dir (default: certified champion)")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    threads = [int(t) for t in a.threads.split(",") if t]
    if not k1.available():
        print("k1 board unavailable — this measurement needs real silicon; "
              "spike cannot answer it (it simulates idle harts at full speed)")
        return 1

    from merlin.compile_cli import default_package
    pkg_dir = a.package or default_package(a.dtype)
    pkg = load_rvv_package(pkg_dir)
    b = _bundle.resolve(a.model, a.dtype)
    mdir = b.root
    refs = _refs(mdir)
    out_path = Path(a.out) if a.out else (artifacts_dir() / "perf-bench" / "rvv" /
                                          f"multicore_scaling_{a.model}_{a.dtype}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ONE binary, lowered for the largest thread count; OMP_NUM_THREADS then selects how many
    # cores execute those chunks. Same code under every point on the curve.
    work = Path(tempfile.mkdtemp(prefix=f"k1mc_{a.model}_"))
    print(f"building composed vector+omp binary (parallel_harts={max(threads)}) "
          f"pkg={Path(pkg_dir).name} …", flush=True)
    t0 = time.time()
    binary = k1.build_k1_binary(mdir, work, pkg, inputs_npz=mdir / "inputs.npz",
                                parallel_harts=max(threads))
    build_s = time.time() - t0
    ll = work / "lower_vecomp" / "model.ll"
    regions = count_parallel_regions(ll) if ll.is_file() else {}
    print(f"built in {build_s:.0f}s; {regions}", flush=True)

    rows = []
    base_med = None
    for t in threads:
        walls, gate = [], None
        for i in range(a.n):
            try:
                res = k1.run_binary_on_k1(mdir, work, pkg, binary, timeout=a.timeout,
                                          env={"OMP_NUM_THREADS": str(t),
                                               "OMP_PROC_BIND": "spread"})
                gate = zm._gate(res["prefix"], refs)
                w = res.get("metrics", {}).get("wall_ns")
                if w:
                    walls.append(int(w))
                print(f"  T={t} run {i}: wall_ns={w} gate_ok={gate.get('ok')} "
                      f"cos={gate.get('cos')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  T={t} run {i}: BLOCKED — {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:200]}", flush=True)
                break
        ok = bool(gate and gate.get("ok")) and bool(walls)
        med = statistics.median(walls) if ok else None
        if t == threads[0] and med:
            base_med = med
        rec = {"model": a.model, "dtype": a.dtype, "package": Path(pkg_dir).name,
               "threads": t, "n": len(walls), "board": "k1_spacemit", "vlen": k1.VLEN,
               "median_wall_ns": med, "walls_ns": sorted(walls) if ok else [],
               "gate_ok": bool(gate and gate.get("ok")), "cos": (gate or {}).get("cos"),
               **regions,
               "status": "pass" if ok else "not_run",
               "blocker": None if ok else "gate did not pass / run blocked"}
        if med and base_med:
            rec["speedup_vs_1"] = round(base_med / med, 3)
            rec["efficiency"] = round((base_med / med) / t, 3) if t else None
            if t > 1 and med != base_med:
                # Amdahl: S = 1/(f + (1-f)/T)  ->  f = (T/S - 1)/(T - 1)
                s = base_med / med
                rec["implied_serial_fraction"] = round((t / s - 1) / (t - 1), 3)
        rows.append(rec)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**rec, "t": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}) + "\n")

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {out_path}")
    print(f"{'T':>3} {'median_ns':>14} {'speedup':>8} {'eff':>6} {'serial_f':>9}  gate")
    for r in rows:
        print(f"{r['threads']:>3} {str(r['median_wall_ns']):>14} "
              f"{str(r.get('speedup_vs_1','-')):>8} {str(r.get('efficiency','-')):>6} "
              f"{str(r.get('implied_serial_fraction','-')):>9}  {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
