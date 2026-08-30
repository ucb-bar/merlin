#!/usr/bin/env python3
"""L5 FireSim backfill: cycle-accurate cycles for the kernels verilator can't reach.

Verilator simulates the full SoC at ~kHz, so anything above ~32K MACs blows the wall-clock budget.
FireSim runs the SAME RTL on the Alveo U250 FPGA at ~MHz (built bitstream
`alveo_u250_firesim_shuttle_gemmini_opu`), finishing in seconds. We reuse the EXACT bare-metal ELFs
already built for spike/verilator (no rebuild) and run them through the shared-FPGA queue
(`firesim-queue runworkload-full --stage-from`), which owns kill→infrasetup→runworkload→kill under the
FPGA lock. Cycles come from the ELF's own `METRIC cycles` (rdcycle delta — a real CSR on the FPGA, no
zicntr gating); correctness from comparing `OUT Y0` to the shared capsule golden.

Representative arms: the 3 MLIR backends emit identical RoCC for matmul (verified identical on verilator),
so we run golden + merlin_targetgen and note the other two match. conv runs merlin_targetgen only
(the only backend that compiles conv).

Writes <run>/firesim_arm_results.json: {kernel: {arm: {cycles, util_pct, correct, wall_s, job_id}}}.
Merge with merge_firesim_arm.py. Usage: run_firesim_arm.py [--kernels ...] [--arms golden,merlin_targetgen]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

import _pbcommon as PB

import sys
sys.path.insert(0, str(PB.REPO / "merlin" / "python"))
from merlin.common.paths import ext_path  # noqa: E402
from merlin.targetgen import capsule_golden as CG  # noqa: E402

# Machine-specific locations come from .env (`MERLIN_EXT_CHIPYARD`, `MERLIN_EXT_FIRESIM_QUEUE`) —
# never a literal, so a fresh clone configures once and every FireSim script agrees. `ext_path` raises
# a named KeyError when a key is unset, which is the honest failure: a placeholder path would fail
# later, inside the queue, looking like an FPGA problem.
CHIPYARD = str(ext_path("chipyard"))
QUEUE = str(ext_path("firesim_queue") / "bin" / "firesim-queue")
WORKLOAD = "merlin-perfbench"
BOOTBINARY = "merlin-perfbench.elf"
RESULTS_ROOT = Path(CHIPYARD) / "sims/firesim/deploy/results-workload"

_JOB_RE = re.compile(r"job_id=(\d+)")
_CYC_RE = re.compile(r"METRIC cycles (\d+)")
_OUT_RE = re.compile(r"^OUT (\S+) (\d+) (\d+) (.*)$", re.M)


def locate_elf(run: Path, kernel: str, arm: str) -> Path | None:
    if arm == "golden":
        p = run / "_work" / kernel / f"golden_{kernel}.elf"
        return p if p.is_file() else None
    # MLIR arms: capsule_runner package ELF
    p = (run / "_capsule_runs" / "runs" / "gemmini-capsule-bench"
         / f"{arm}_{kernel}" / "generated" / "package_kernel.elf")
    return p if p.is_file() else None


def submit(elf: Path, timeout: int) -> tuple[int | None, str]:
    """Submit one ELF through the FPGA queue; return (job_id, uartlog_text)."""
    # The queue prints its [firesim-queue] phase/terminal lines to STDERR, so merge it into stdout
    # (else "terminal state=DONE" is missed and a successful run looks like a failure).
    out = subprocess.run(
        [QUEUE, "runworkload-full", "--chipyard", CHIPYARD, "--workload", WORKLOAD,
         "--bootbinary", BOOTBINARY, "--stage-from", str(elf),
         "--priority", "5", "--project", "merlin-perfbench", "--timeout", str(timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout + 300).stdout
    if "terminal state=DONE" not in out:
        raise RuntimeError(f"queue job not DONE: {out[-400:]}")
    m = _JOB_RE.search(out)
    job_id = int(m.group(1)) if m else None
    # locate the per-run uartlog by the q<job_id> suffix
    uart = ""
    if job_id is not None:
        for d in RESULTS_ROOT.glob(f"*-{WORKLOAD}-q{job_id}"):
            u = d / f"{WORKLOAD}0" / "uartlog"
            if u.is_file():
                uart = u.read_text(errors="replace")
                break
    return job_id, uart


def parse(uart: str) -> tuple[int | None, dict]:
    cyc = int(m.group(1)) if (m := _CYC_RE.search(uart)) else None
    outs = {}
    for name, r, c, vals in _OUT_RE.findall(uart):
        outs[name] = [int(x) for x in vals.split()]
    return cyc, outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    ap.add_argument("--arms", default="golden,merlin_targetgen")
    ap.add_argument("--kernels", default="infeasible",
                    help="'infeasible' (no verilator cycles), 'all', or a comma list")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]

    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = {k["id"]: k for sec in doc if isinstance(doc[sec], list) for k in doc[sec]}
    pr = json.loads((run / "perf_results.json").read_text())
    has_veri = {r["kernel"] for r in pr
                if any((v.get("per_sim") or {}).get("verilator", {}).get("cycles")
                       for v in r["approaches"].values())}
    if a.kernels == "infeasible":
        kernels = [r["kernel"] for r in pr if r["kernel"] not in has_veri]
    elif a.kernels == "all":
        kernels = [r["kernel"] for r in pr]
    else:
        kernels = a.kernels.split(",")

    res_path = run / "firesim_arm_results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else {}

    for kid in kernels:
        k = corpus.get(kid)
        if not k:
            continue
        cap = yaml.safe_load((PB.KERNELS / kid / "capsule.yaml").read_text())
        try:
            gold = CG.golden(cap).get("Y0")
            gold = [int(x) for x in (gold.flatten().tolist() if hasattr(gold, "flatten") else gold)]
        except Exception:
            gold = None
        results.setdefault(kid, {})
        for arm in arms:
            elf = locate_elf(run, kid, arm)
            if elf is None:
                results[kid][arm] = {"error": "no ELF (arm did not build for this kernel)"}
                print(f"[{kid:32s} {arm:16s}] no ELF", flush=True)
                res_path.write_text(json.dumps(results, indent=2)); continue
            t0 = time.time()
            try:
                job_id, uart = submit(elf, a.timeout)
                cyc, outs = parse(uart)
                got = outs.get("Y0")
                correct = (gold is not None and got is not None and got == gold)
                results[kid][arm] = {"cycles": cyc, "job_id": job_id,
                                     "correct": correct,
                                     "util_pct": PB.utilization_pct(k["macs"], cyc),
                                     "wall_s": round(time.time() - t0, 1)}
                print(f"[{kid:32s} {arm:16s}] cyc={cyc} correct={correct} "
                      f"util={results[kid][arm]['util_pct']}% ({results[kid][arm]['wall_s']}s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                results[kid][arm] = {"error": str(e)[-200:], "wall_s": round(time.time() - t0, 1)}
                print(f"[{kid:32s} {arm:16s}] ERR {str(e)[-120:]}", flush=True)
            res_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
