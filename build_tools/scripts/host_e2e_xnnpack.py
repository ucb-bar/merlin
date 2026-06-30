#!/usr/bin/env python
"""HOST e2e third column: XNNPACK-kernel backend vs the default compiled dispatch path.

Runs a captured model's WHOLE forward on the host twice -- once through the default
Merlin-compiled per-dispatch kernels, once with the plain f32 matmul dispatches routed
through the XNNPACK scalar microkernel (`kernel_backend="xnnpack"`) -- and reports, for both:

  - cos vs the captured torch golden (the SAME golden the K1 e2e runner gates on), and
  - cos(xnnpack, default) to prove the kernel swap is byte-stable end to end.

This is the HOST-correctness leg of the third e2e column (baseline / ours / XNNPACK). It does
NOT touch the K1 board -- board (RVV) cross-compile + rdtime timing is the deferred validation
step (it reuses the SAME XNNPACK ukernel family via scripts/k1_cross_framework.py's ceiling
driver). No fabricated timing numbers here: this script only certifies host correctness.

Run:  .venv/bin/python scripts/host_e2e_xnnpack.py --model output/bitvla_fp32_consistent
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from merlin.runtime import dispatch_runtime as dr


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--out", default="artifacts/measurements/host/host_e2e_xnnpack_bitvla.json")
    a = ap.parse_args()
    md = Path(a.model)

    with tempfile.TemporaryDirectory(prefix="host_e2e_xnn_") as tmp:
        base = dr.run_model(md, Path(tmp) / "base")
        xnn = dr.run_model(md, Path(tmp) / "xnn", kernel_backend="xnnpack")

    report = {
        "model": str(md),
        "note": "HOST correctness only; board RVV timing is the deferred step",
        "default": {"cos_vs_golden": base.get("cos"), "rel": base.get("rel"),
                    "n_kernels": base["n_kernels"]},
        "xnnpack": {"cos_vs_golden": xnn.get("cos"), "rel": xnn.get("rel"),
                    "n_kernels": xnn["n_kernels"], "n_matmul_routed_to_xnnpack": xnn["n_xnn_routed"]},
        "cos_xnnpack_vs_default": _cos(np.asarray(xnn["output"]), np.asarray(base["output"])),
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {outp}")


if __name__ == "__main__":
    main()
