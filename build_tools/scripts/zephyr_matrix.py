#!/usr/bin/env python
"""Run captured bundles through the Zephyr whole-model path on spike (-p2), report cos.

Validates that runtime.backends.zephyr_model generalizes across models before any of them
are pushed to FireSim. Usage:
    ZMX_BACKEND=scalar .venv/bin/python build_tools/scripts/zephyr_matrix.py BUNDLE...
(default backend scalar = the FireSim-proven path; set ZMX_BACKEND=rvv for the vector tile.
default bundle list = a small/medium set). For scalar the worker pins to hart 0, for rvv
hart 1 (the FireSim Saturn tile) — the exact image that would run on FireSim.
"""
import os
import sys
import time

import numpy as np

from merlin.common.artifacts import recaptures_dir
from merlin.runtime.backends import zephyr_model as zm

DEFAULT = ["small_consistent", "tiny_consistent", "rdt2_fp32_consistent",
           "groot_n1d7_fp32_consistent", "molmoact_fp32_consistent"]

backend = os.environ.get("ZMX_BACKEND", "scalar")
hart = 0 if backend == "scalar" else 1
# ZMX_HARTS>1 builds the MULTICORE image (OpenMP over pinned harts, rvv backend only);
# ZMX_ITERS/ZMX_WARMUP drive the sustained-inference report.
n_harts = int(os.environ.get("ZMX_HARTS", "1"))
iters = int(os.environ.get("ZMX_ITERS", "1"))
warmup = int(os.environ.get("ZMX_WARMUP", "0"))
bundles = sys.argv[1:] or DEFAULT
print(f"available={zm.available()}  backend={backend}  n_harts={n_harts}  "
      f"iters={iters}  bundles={bundles}", flush=True)
for name in bundles:
    b = recaptures_dir() / name
    if not (b / "model.mlir").is_file():
        print(f"SKIP {name} (not captured)", flush=True)
        continue
    gpath = b / "golden.npy"
    golden = np.load(gpath) if gpath.is_file() else None
    t0 = time.time()
    try:
        res = zm.build_and_run(b, f"/tmp/zmx_{name}", board="spike_riscv64",
                               backend=backend, rvv_hart=hart,
                               harts=max(2, n_harts), arena_mb=192,
                               n_harts=n_harts, iters=iters, warmup=warmup,
                               reference=golden, timeout=3600)
        dt = time.time() - t0
        sus = res.get("sustained")
        print(f"ZRESULT {name} cos={res.get('cos',-1):.7f} rel={res.get('rel',-1):.3e} "
              f"ok={res.get('ok')} cyc={res.get('metrics',{}).get('cycles')} "
              f"{'sustained=' + str(sus) + ' ' if sus else ''}{dt:.0f}s",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ZRESULT {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
