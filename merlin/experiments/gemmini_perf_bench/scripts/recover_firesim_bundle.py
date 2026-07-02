#!/usr/bin/env python3
"""Recover firesim_arm_results.json from already-written per-ELF uartlogs WITHOUT touching the FPGA.

The bundle worker (firesim_bundle.sh) saves <outdir>/<kernel>__<arm>.uartlog as each ELF finishes; the
launcher (run_firesim_bundle.py) only writes firesim_arm_results.json at the very end. If the launcher
dies mid-batch (e.g. a host reboot), the uartlogs survive but the results file is lost. This script
re-runs the *parse-only* portion of run_firesim_bundle over every uartlog in one or more bundle outdirs
and rebuilds firesim_arm_results.json. Same parse/golden/util logic — no re-run.

Usage: recover_firesim_bundle.py [--run-id perf_full_0001] [--outdirs _firesim_full,_firesim_test]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

import _pbcommon as PB

sys.path.insert(0, str(PB.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_golden as CG  # noqa: E402

_CYC_RE = re.compile(r"METRIC cycles (\d+)")
_OUT_RE = re.compile(r"^OUT (\S+) (\d+) (\d+) (.*)$", re.M)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    ap.add_argument("--outdirs", default="_firesim_full,_firesim_test",
                    help="comma-separated bundle outdirs under the run (later dirs win on duplicate)")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = {k["id"]: k for sec in doc if isinstance(doc[sec], list) for k in doc[sec]}

    res_path = run / "firesim_arm_results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else {}
    gold_cache: dict[str, list | None] = {}
    recovered = failed = 0

    for od_name in [s.strip() for s in a.outdirs.split(",") if s.strip()]:
        outdir = run / od_name
        man = outdir / "manifest.tsv"
        if not man.is_file():
            print(f"  (skip {od_name}: no manifest)"); continue
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            lbl = line.split("\t", 1)[0]
            kid, _, arm = lbl.partition("__")
            if kid not in corpus:
                continue
            k = corpus[kid]
            u = outdir / f"{lbl}.uartlog"
            if not u.is_file():
                continue
            if kid not in gold_cache:
                try:
                    cap = yaml.safe_load((PB.KERNELS / kid / "capsule.yaml").read_text())
                    g = CG.golden(cap).get("Y0")
                    gold_cache[kid] = np.asarray(g).flatten().astype(int).tolist()
                except Exception:
                    gold_cache[kid] = None
            text = u.read_text(errors="replace")
            cyc = int(m.group(1)) if (m := _CYC_RE.search(text)) else None
            outs = {n: [int(x) for x in v.split()] for n, _r, _c, v in _OUT_RE.findall(text)}
            got = outs.get("Y0")
            gold = gold_cache[kid]
            results.setdefault(kid, {})
            if cyc is None:
                # don't overwrite a good earlier cell with a failed later one
                if results[kid].get(arm, {}).get("cycles") is None:
                    results[kid][arm] = {"error": "no METRIC cycles (run failed/hung — re-batch)"}
                failed += 1
            else:
                results[kid][arm] = {"cycles": cyc,
                                     "correct": bool(gold is not None and got == gold),
                                     "util_pct": PB.utilization_pct(k["macs"], cyc)}
                recovered += 1

    res_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for kid in results for arm in results[kid]
             if results[kid][arm].get("cycles") is not None)
    print(f"recovered {recovered} cells with cycles ({failed} without); "
          f"{ok} total cells in {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
