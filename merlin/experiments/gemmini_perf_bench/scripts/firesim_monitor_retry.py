#!/usr/bin/env python3
"""Persistent FireSim monitor + auto-retry for the 4-backend profiling bundle.

Waits for the in-flight bundle to write firesim_arm_results.json, then loops: find every (kernel, arm)
cell that has NO cycle count and re-batches just those via run_firesim_bundle.py --skip-existing (one
FPGA queue job per attempt, which re-runs ONLY the missing cells). Keeps going until every expected cell
has cycles, or MAX_ATTEMPTS is hit. Fully fault-tolerant: a crashed attempt is logged and retried.

Target = all 4 agentic arms × the 12 shared kernels (all matmul/attention/model-slice kernels the
backends are functionally correct on). Logs to runs/<run>/firesim_monitor.log.
"""
from __future__ import annotations
import json, subprocess, sys, time, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID = "perf_4backends_0001"
RUNS = HERE.parent / "runs"
RUN = RUNS / RUN_ID
RESULTS = RUN / "firesim_arm_results.json"
LOG = RUN / "firesim_monitor.log"
import sys as _sys
_sys.path.insert(0, str(HERE.parents[3] / "merlin" / "python"))
from merlin.common.paths import ext_path, repo_root  # noqa: E402

# This repo's own venv and the external chipyard/toolchain, resolved rather than spelled out.
PY = str(repo_root() / ".venv" / "bin" / "python")
BUNDLE = str(HERE / "run_firesim_bundle.py")
ARMS = ["agentic_raw_cpp", "agentic_scaffold_cpp", "agentic_python", "agentic_circt"]
KERNELS = ["G01_multitile_sq_64x64x64", "G06_acc_scale_i8_64x64x64", "G07_relu_i8_64x64x64",
           "G08_large_sq_128x128x128", "K_attn_pv_64x64x64", "K_attn_qk_128x64x128",
           "K_attn_qk_64x64x64", "M00_smolvla_model_16x32x960_i8", "M01_smolvla_model_64x720x32_i8",
           "M02_smolvla_model_64x32x720_i8", "M03_openvla_vla_32x256x128_i8", "M04_openvla_vla_32x128x256_i8"]
EXPECTED = [(k, a) for k in KERNELS for a in ARMS]   # 48 cells
MAX_ATTEMPTS = 14


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def missing_cells():
    if not RESULTS.is_file():
        return list(EXPECTED)
    try:
        d = json.loads(RESULTS.read_text())
    except Exception:
        return list(EXPECTED)
    miss = []
    for k, a in EXPECTED:
        cell = (d.get(k) or {}).get(a) or {}
        if cell.get("cycles") is None:
            miss.append((k, a))
    return miss


def env():
    e = dict(os.environ)
    cl = str(repo_root() / ".compat_lib")
    conda = str(ext_path("chipyard") / ".conda-env")
    e["LD_LIBRARY_PATH"] = f"{cl}:{conda}/lib:{conda}/riscv-tools/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["PATH"] = f"{conda}/bin:{conda}/riscv-tools/bin:" + e.get("PATH", "")
    return e


def main():
    log(f"monitor start — target {len(EXPECTED)} cells (4 arms × {len(KERNELS)} kernels)")
    # Phase 1: wait for the in-flight bundle to finish (results file appears AND stops growing)
    t0 = time.time()
    last_n = -1
    while True:
        miss = missing_cells()
        have = len(EXPECTED) - len(miss)
        if RESULTS.is_file() and have == last_n and have > 0:
            log(f"in-flight bundle settled: {have}/{len(EXPECTED)} cells have cycles")
            break
        last_n = have
        if time.time() - t0 > 4 * 3600:   # safety: don't wait forever for phase 1
            log("phase-1 wait exceeded 4h; proceeding to retry loop anyway")
            break
        log(f"waiting for in-flight bundle… {have}/{len(EXPECTED)} cells so far")
        time.sleep(180)

    # Phase 2: retry loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        miss = missing_cells()
        if not miss:
            log(f"ALL {len(EXPECTED)} cells have cycles — SUCCESS. Done.")
            return 0
        log(f"attempt {attempt}/{MAX_ATTEMPTS}: {len(miss)} cells missing cycles -> re-batching")
        for k, a in miss:
            log(f"    missing: {k} / {a}")
        try:
            cp = subprocess.run(
                [PY, BUNDLE, "--run-id", RUN_ID, "--arms", ",".join(ARMS),
                 "--kernels", "all", "--skip-existing", "--tag", f"retry{attempt}", "--timeout", "2400"],
                cwd=str(HERE), env=env(), capture_output=True, text=True, timeout=5 * 3600)
            log(f"    re-batch rc={cp.returncode}; tail: {(cp.stdout or '')[-300:].strip()}")
        except subprocess.TimeoutExpired:
            log("    re-batch TIMED OUT (5h) — will retry")
        except Exception as e:
            log(f"    re-batch crashed: {type(e).__name__}: {str(e)[:200]} — will retry")
        time.sleep(20)

    miss = missing_cells()
    if miss:
        log(f"MAX_ATTEMPTS reached; still missing {len(miss)}: {miss}")
        return 1
    log("ALL cells have cycles — SUCCESS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
