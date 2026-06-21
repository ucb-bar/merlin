#!/usr/bin/env python3
"""Build + SAVE the per-shape IREE Gemmini-dialect ELFs so they can be run on FireSim (L5).

run_iree_arm.py builds each shape's ELF in-place (overwriting the same path) and runs it on spike. For
the FireSim backfill we need each shape's ELF kept around to stage into the bundle. This reuses the same
build path (cmake -DGEMMINI_SPIKE_MATMUL_SHAPE → ninja) and copies the resulting bare-metal htif ELF to
runs/<run>/_iree_elfs/<kernel>.elf. IREE only covers matmul/attention.

Usage: build_iree_elfs.py [--kernels infeasible|id,..]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

import _pbcommon as PB
from run_iree_arm import BUILD_ENV, BUILD, ELF, ensure_fixture, build_shape  # reuse the build path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    ap.add_argument("--kernels", default="infeasible")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = {k["id"]: k for sec in doc if isinstance(doc[sec], list) for k in doc[sec]}
    pr = json.loads((run / "perf_results.json").read_text())
    has_veri = {r["kernel"] for r in pr
                if any((v.get("per_sim") or {}).get("verilator", {}).get("cycles")
                       for v in r["approaches"].values())}
    if a.kernels == "infeasible":
        want = [r["kernel"] for r in pr if r["kernel"] not in has_veri]
    else:
        want = a.kernels.split(",")
    # IREE only covers matmul/attention (op==matmul). Skip giants without dims handled by run_iree_arm.
    want = [k for k in want if corpus.get(k, {}).get("op") == "matmul"]

    outdir = run / "_iree_elfs"
    outdir.mkdir(parents=True, exist_ok=True)
    built, failed = [], []
    for kid in want:
        k = corpus[kid]
        if k.get("M") is not None:
            M, K, N = int(k["M"]), int(k["K"]), int(k["N"])
        else:
            M, K, N = (int(x) for x in str(k["shape"]).split("x"))
        if k["macs"] > 3_000_000:
            print(f"[{kid}] SKIP giant ({k['macs']:,} macs)", flush=True); continue
        try:
            shape, _ = ensure_fixture(M, N, K)
            build_shape(shape, outdir / f"{kid}.build.log")
            dst = outdir / f"{kid}.elf"
            shutil.copyfile(ELF, dst)
            dst.chmod(0o755)
            built.append(kid)
            print(f"[{kid}] built -> {dst.name}", flush=True)
        except subprocess.CalledProcessError:
            failed.append(kid)
            print(f"[{kid}] BUILD FAIL", flush=True)
    print(f"\nbuilt {len(built)} IREE ELFs in {outdir}; failed: {failed or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
