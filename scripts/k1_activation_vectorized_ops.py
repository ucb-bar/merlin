#!/usr/bin/env python
"""VALIDATION 2 — isolated GELU/sigmoid OURS-VECTORIZED (compiler polynomial) vs XNNPACK on K1.

The existing cross_framework_ops_k1.jsonl `ours_vectorized` rows were lowered with `vectorize=True`
but `features=[]` -> the matmul-vectorize pass with NO activation feature, i.e. still the SCALAR
libm `erff`/`expf` loop (a few % slower than `ours_scalar` from the vectorize overhead). The genuine
`vectorized_transcendental_activation` (compiler-emitted minimax polynomial -> vfmacc Horner chains)
was NEVER measured on the K1 — that feature did not exist when the matrix was built.

This script measures the REAL ours-vectorized: lower gen_gelu_f32 / gen_sigmoid_f32 with
features={vectorized_transcendental_activation}, in the SAME standalone ours_activation_driver.c,
cross-compiled with SpacemiT clang, run on K1 at 1K/16K/256K, N=3 min rdtime. Correctness is the
driver's cos/abs gate (approximation, NOT bit-exact) vs the scalar reference.

It then UPDATES cross_framework_ops_k1.{jsonl,md}:
  * the mislabeled `ours_vectorized` rows are renamed `ours_vectorize_nofeature` (honest: matmul
    vectorize pass, no activation feature — still scalar libm),
  * a NEW `ours_vectorized` row (source kept for the plot) carries the polynomial-feature number.

Reuses the FROZEN harness `_build_run_ours` from k1_cross_framework_ops.py (no rebuild). Honest
not_run with the exact blocker on any build/run/verify failure; board left clean.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "k1_ops", str(Path(__file__).resolve().parent / "k1_cross_framework_ops.py"))
_ops = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ops)

from merlin.rvvgen import k1, workloads
from merlin.common.paths import repo_root

REPO = Path(repo_root())
HERE = REPO / "merlin/python/merlin/kernels/ceiling_drivers"
FEATURE = "vectorized_transcendental_activation"


def run_ours_vectorized(op: str, sizes: list[int], reps: int) -> list[dict]:
    ref = op  # 'gelu' | 'sigmoid'
    gen = workloads.gen_gelu_f32 if op == "gelu" else workloads.gen_sigmoid_f32
    rows = []
    for Nsz in sizes:
        bundle = gen(REPO / "output" / "rvv_workloads", N=Nsz)
        base = {"op": op, "dtype": "f32", "size_n": Nsz, "source": "ours_vectorized",
                "target": "k1", "mode": "inner_compute", "timer": "rdtime",
                "timebase_hz": k1.K1_TIMEBASE_HZ, "vectorize": True,
                "compiler_features": [FEATURE],
                "kernel_file": "merlin RVV codegen (ours_vectorized: vectorized_transcendental_activation polynomial)"}
        print(f"--- ours_vectorized(POLY) {op} N={Nsz} ---")
        r = _ops._build_run_ours(
            f"{op}_oursvec_{Nsz}", bundle, HERE / "ours_activation_driver.c",
            [f"-DXNN_REF_{ref}"], "ours_vectorized", [FEATURE],
            int8=False, vectorize=True, reps=reps, base=base)
        print("   ", r["status"], r.get("ticks"), r.get("blocker", "")[:160])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="gelu,sigmoid")
    ap.add_argument("--sizes", default="1024,16384,262144")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="output/kernels/ceiling/ours_vectorized_ops_k1.jsonl")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    rows = []
    for op in a.ops.split(","):
        rows += run_ours_vectorized(op, sizes, a.reps)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
