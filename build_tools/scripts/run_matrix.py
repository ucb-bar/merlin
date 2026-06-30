#!/usr/bin/env python3
"""Dispatch-run a set of consistent bundles (host==torch) and print the gate per bundle.

Usage: run_matrix.py <bundle_dir> [<bundle_dir> ...]
Each <bundle_dir> is under oscar-merlin/output/. Prints one RESULT line per bundle so a
monitor can stream them; tolerant of per-bundle failures (reports them, keeps going).
"""
import sys
import tempfile
import traceback
from pathlib import Path

from merlin.runtime.dispatch_runtime import run_model

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "recaptures"


def main(names):
    for name in names:
        b = OUT / name
        if not (b / "model.mlir").is_file():
            print(f"RESULT {name} SKIP(no model.mlir)", flush=True)
            continue
        try:
            r = run_model(b, Path(tempfile.mkdtemp(prefix=f"{name}_")),
                          cache_dir=OUT / f".kc_{name}")
            cos, rel, ok, nk = r.get("cos"), r.get("rel"), r.get("ok"), r["n_kernels"]
            cs = f"{cos:.7f}" if cos == cos else "nan"        # nan-safe (mask-like tensors)
            print(f"RESULT {name} kernels={nk} cos={cs} rel={rel:.2e} ok={ok}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(f"RESULT {name} ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            traceback.print_exc()
    print("__ALL_RUNS_FINISHED__", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
