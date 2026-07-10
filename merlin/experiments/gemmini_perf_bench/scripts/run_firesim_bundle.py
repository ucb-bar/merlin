#!/usr/bin/env python3
"""Launch a BUNDLED FireSim backfill: one queue job flashes the FPGA once and runs every ELF in the
batch back-to-back (firesim_bundle.sh), then parse each per-ELF uartlog for cycle-accurate cycles +
correctness. ~6min flash amortized across the whole batch instead of paid per ELF.

Reuses the exact spike/verilator ELFs (golden + a representative MLIR arm — the 3 MLIR backends emit
identical RoCC for matmul). Fail-open: ELFs that produce no METRIC cycles are recorded as errors so
they can be fixed and re-batched.

Writes <run>/firesim_arm_results.json: {kernel: {arm: {cycles, util_pct, correct, ...}}}.
Usage: run_firesim_bundle.py [--kernels infeasible|all|id,..] [--arms golden,merlin_targetgen] [--timeout 300]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

import _pbcommon as PB

sys.path.insert(0, str(PB.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_golden as CG  # noqa: E402

QUEUE = "/path/to/firesim_queue/bin/firesim-queue"
BUNDLE_SH = str(Path(__file__).resolve().parent / "firesim_bundle.sh")
_CYC_RE = re.compile(r"METRIC cycles (\d+)")
_OUT_RE = re.compile(r"^OUT (\S+) (\d+) (\d+) (.*)$", re.M)


def locate_elf(run: Path, kernel: str, arm: str) -> Path | None:
    if arm == "golden":
        p = run / "_work" / kernel / f"golden_{kernel}.elf"
    elif arm == "iree_dialect":
        p = run / "_iree_elfs" / f"{kernel}.elf"   # built by build_iree_elfs.py
    else:  # baseline / merlin_targetgen / merlin_native — capsule_runner package ELF
        p = (run / "_capsule_runs" / "runs" / "gemmini-capsule-bench"
             / f"{arm}_{kernel}" / "generated" / "package_kernel.elf")
    return p if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    ap.add_argument("--arms", default="golden,merlin_targetgen")
    ap.add_argument("--kernels", default="infeasible")
    ap.add_argument("--timeout", type=int, default=600, help="per-ELF runworkload timeout (s)")
    ap.add_argument("--tag", default="bundle", help="outdir/label tag for this batch")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (kernel,arm) cells that already have FireSim cycles (don't re-run them)")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    existing = {}
    if a.skip_existing and (run / "firesim_arm_results.json").is_file():
        existing = json.loads((run / "firesim_arm_results.json").read_text())

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

    # Build manifest of (label, elf). label = "<kernel>__<arm>".
    outdir = run / f"_firesim_{a.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.tsv"
    rows, skipped = [], []
    for kid in kernels:
        for arm in arms:
            if existing.get(kid, {}).get(arm, {}).get("cycles") is not None:
                skipped.append(f"{kid}/{arm}=have")
                continue
            elf = locate_elf(run, kid, arm)
            if elf is None:
                skipped.append(f"{kid}/{arm}")
                continue
            rows.append((f"{kid}__{arm}", str(elf), kid, arm))
    manifest.write_text("".join(f"{lbl}\t{elf}\n" for lbl, elf, _, _ in rows))
    print(f"manifest: {len(rows)} ELFs, {len(skipped)} skipped (no ELF): {', '.join(skipped) or '-'}",
          flush=True)
    if not rows:
        print("nothing to run"); return 1

    # Per-bundle config_runtime.yaml: copy the shared one but point workload_name at merlin-perfbench
    # (bare `firesim runworkload` otherwise reads the shared config, which targets a different workload).
    shared_cfg = Path("/path/to/chipyard/sims/firesim/deploy/config_runtime.yaml")
    cfg = outdir / "config_runtime.yaml"
    cfg_text = re.sub(r"workload_name:\s*\S+", "workload_name: merlin-perfbench.json",
                      shared_cfg.read_text())
    cfg_text = re.sub(r"suffix_tag:\s*\S+", "suffix_tag: null", cfg_text)
    cfg.write_text(cfg_text)

    # One queue job: flash once + run all. submit blocks until the daemon finishes the bundle.
    print(f"submitting bundle ({len(rows)} ELFs) to FPGA queue ...", flush=True)
    t0 = time.time()
    sub = subprocess.run(
        [QUEUE, "submit", "--priority", "5", "--",
         "bash", BUNDLE_SH, str(manifest), str(outdir), str(a.timeout), str(cfg)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (outdir / "submit.log").write_text(sub.stdout)
    print(f"bundle job finished in {round(time.time()-t0)}s (rc={sub.returncode}); "
          f"submit log -> {outdir/'submit.log'}", flush=True)

    # Parse each per-ELF uartlog: cycles + OUT vs shared capsule golden.
    res_path = run / "firesim_arm_results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else {}
    gold_cache: dict[str, list | None] = {}
    for lbl, _elf, kid, arm in rows:
        k = corpus[kid]
        if kid not in gold_cache:
            try:
                import numpy as np
                cap = yaml.safe_load((PB.KERNELS / kid / "capsule.yaml").read_text())
                g = CG.golden(cap).get("Y0")  # may be a nested list of rows or an ndarray
                gold_cache[kid] = np.asarray(g).flatten().astype(int).tolist()
            except Exception:
                gold_cache[kid] = None
        u = outdir / f"{lbl}.uartlog"
        results.setdefault(kid, {})
        if not u.is_file():
            results[kid][arm] = {"error": "no uartlog (ELF not run)"}
            continue
        text = u.read_text(errors="replace")
        cyc = int(m.group(1)) if (m := _CYC_RE.search(text)) else None
        outs = {n: [int(x) for x in v.split()] for n, _r, _c, v in _OUT_RE.findall(text)}
        got = outs.get("Y0")
        gold = gold_cache[kid]
        if cyc is None:
            results[kid][arm] = {"error": "no METRIC cycles (run failed/hung — fix + re-batch)"}
        else:
            results[kid][arm] = {"cycles": cyc,
                                 "correct": bool(gold is not None and got == gold),
                                 "util_pct": PB.utilization_pct(k["macs"], cyc)}
    res_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for kid in results for arm in results[kid]
             if results[kid][arm].get("cycles") is not None)
    print(f"\nwrote {res_path} ({ok} cells with cycles)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
