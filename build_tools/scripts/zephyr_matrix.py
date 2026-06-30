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
from pathlib import Path

import numpy as np

from merlin.runtime.backends import zephyr_model as zm

REPO = Path(__file__).resolve().parents[2]
DEFAULT = ["small_consistent", "tiny_consistent", "rdt2_fp32_consistent",
           "groot_n1d7_fp32_consistent", "molmoact_fp32_consistent"]

backend = os.environ.get("ZMX_BACKEND", "scalar")
hart = 0 if backend == "scalar" else 1
bundles = sys.argv[1:] or DEFAULT
print(f"available={zm.available()}  backend={backend}  bundles={bundles}", flush=True)
for name in bundles:
    b = REPO / "artifacts" / "recaptures" / name
    if not (b / "model.mlir").is_file():
        print(f"SKIP {name} (not captured)", flush=True)
        continue
    gpath = b / "golden.npy"
    golden = np.load(gpath) if gpath.is_file() else None
    t0 = time.time()
    try:
        res = zm.build_and_run(b, f"/tmp/zmx_{name}", board="spike_riscv64",
                               backend=backend, rvv_hart=hart, harts=2, arena_mb=192,
                               reference=golden, timeout=3600)
        dt = time.time() - t0
        print(f"ZRESULT {name} cos={res.get('cos',-1):.7f} rel={res.get('rel',-1):.3e} "
              f"ok={res.get('ok')} cyc={res.get('metrics',{}).get('cycles')} {dt:.0f}s",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ZRESULT {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
