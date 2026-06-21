#!/usr/bin/env python3
"""Assemble a run's perf_results.json from the per-kernel <id>.json files (the source of truth).

run_perf_bench writes perf_results.json scoped to ONLY the kernels of its invocation, so a sweep done
in several batches (spike-only resume, etc.) would clobber it. Each kernel's <id>.json is written
incrementally and complete, so we rebuild perf_results.json by collecting them in corpus order. Run
this, then merge_iree_arm.py, then gen_perf_report.py.

Usage: assemble_results.py [--run-id perf_full_0001]
"""
from __future__ import annotations

import argparse
import json

import yaml

import _pbcommon as PB


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    order = [k["id"] for sec in ("golden_kernels", "model_kernels", "attention_kernels",
                                 "conv_kernels", "movement_kernels")
             for k in (doc.get(sec) or [])]
    rows, missing = [], []
    for kid in order:
        f = run / f"{kid}.json"
        if f.is_file():
            rows.append(json.loads(f.read_text()))
        else:
            missing.append(kid)
    (run / "perf_results.json").write_text(json.dumps(rows, indent=2))
    print(f"assembled {len(rows)} kernels into {run}/perf_results.json")
    if missing:
        print(f"  not yet run ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
