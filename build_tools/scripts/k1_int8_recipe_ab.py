#!/usr/bin/env python
"""A/B the int8 GEMM code-generation RECIPE on the real K1 board.

int8 (W8A8) historically went through ``accumulator_resident_wholemodel_vf`` while f32 went through
the v3 micro-kernel tuning point (``accumulator_resident_microkernel_v3`` = MR4/NR16/KC16) plus the
``erase_self_copy`` lowering-hygiene feature. That is a PARALLEL SILO, not a shared capability: the
same register-blocking + self-copy-erase codegen that pays for f32 was simply never applied to int8.

This script measures the three int8 arms head-to-head against XNNPACK's qd8-f32-qc8w RVV ukernel, on
the same kernel-region rdtime bracket, with retired instructions (INSTRET) reported on the SAME
bracket so a win can be attributed to "retires fewer instructions" rather than guessed at.

Fail-closed: an arm whose driver does not print ``VERIFY PASS`` is ``not_run`` and carries no timing.

Usage:
    build_tools/scripts/k1_int8_recipe_ab.py --shapes 128,256 --reps 3 --tag-prefix i8_
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.common.driver_output import int_after
from merlin.rvvgen import k1

sys.path.insert(0, str(Path(repo_root()) / "build_tools" / "scripts"))
from k1_cross_framework_ops import HERE, REPO, _build_run_ours, _build_run_xnn  # noqa: E402

#: The int8 recipes under test. ``ours_int8_vf`` is the incumbent default; the v3 arms are the
#: f32 path's own best recipe applied to int8 unchanged -- shared capability, not an int8 fork.
ARMS = [
    (["accumulator_resident_wholemodel_vf"], "ours_int8_vf"),
    (["accumulator_resident_microkernel_v3"], "ours_int8_v3"),
    (["accumulator_resident_microkernel_v3", "erase_self_copy"], "ours_int8_v3_esc"),
]


def _instret(row: dict) -> int | None:
    """Retired instructions on the timing bracket, when the driver printed them."""
    return row.get("instret")


def run(shapes: list[int], reps: int, prefix: str, dtype: str) -> list[dict]:
    from merlin.rvvgen import workloads
    rows: list[dict] = []
    for S in shapes:
        bundle = workloads.gen_matmul_f32(REPO / "out" / "artifacts" / "cache" / "rvv_workloads",
                                          M=S, N=S, K=S)
        defs = [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"]
        common = {"M": S, "N": S, "K": S, "target": "k1", "mode": "inner_compute",
                  "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ}
        if dtype == "int8":
            base = {**common, "op": "int8_gemm", "dtype": "qd8_qc8w", "source": "xnnpack",
                    "kernel_file": "XNNPACK qd8-f32-qc8w-gemm-1x4v-minmax-rvv"}
            print(f"--- xnnpack int8_gemm {S}^3 ---", flush=True)
            r = _build_run_xnn(f"{prefix}qd8_xnn_{S}", HERE / "xnnpack_qd8_gemm_driver.c",
                               defs, reps=reps, base=base)
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""), flush=True)
            rows.append(r)
            arms, driver, int8 = ARMS, HERE / "ours_int8_gemm_driver.c", True
        else:
            base = {**common, "op": "f32_gemm", "dtype": "f32", "source": "xnnpack",
                    "kernel_file": "XNNPACK f32-gemm-7x4v-rvv"}
            print(f"--- xnnpack f32_gemm {S}^3 ---", flush=True)
            r = _build_run_xnn(f"{prefix}f32_xnn_{S}", HERE / "xnnpack_gemm_driver_7x4v.c",
                               defs, reps=reps, base=base)
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""), flush=True)
            rows.append(r)
            arms = [(f, s.replace("int8", "f32")) for f, s in ARMS]
            driver, int8 = HERE / "ours_gemm_driver.c", False
        for feats, sid in arms:
            base = {**common, "op": f"{dtype}_gemm", "dtype": ("i8xi8->i32" if int8 else "f32"),
                    "source": sid, "int8_compute": int8, "compiler_features": feats,
                    "kernel_file": f"merlin RVV codegen ({sid})"}
            print(f"--- {sid} {S}^3 {feats} ---", flush=True)
            r = _build_run_ours(f"{prefix}{sid}_{S}", bundle, driver, defs, sid, feats,
                                int8=int8, vectorize=True, reps=reps, base=base)
            print("   ", r["status"], r.get("ticks"), r.get("instret"),
                  r.get("blocker", ""), flush=True)
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="128", help="comma-separated square GEMM sizes")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag-prefix", default="i8_", help="unique remote /tmp tag prefix")
    ap.add_argument("--dtype", default="int8", choices=("int8", "f32"))
    ap.add_argument("--out", default=None, help="jsonl path (default: under out/artifacts)")
    a = ap.parse_args()
    shapes = [int(s) for s in a.shapes.split(",") if s]
    with k1.board_lock():
        rows = run(shapes, a.reps, a.tag_prefix, a.dtype)
    out = Path(a.out) if a.out else (artifacts_dir() / "perf-bench" / "rvv" /
                                     f"int8_recipe_ab_{a.dtype}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")
    for r in rows:
        print(f"  {r['source']:24s} {r.get('M')}^3  ticks={r.get('ticks')}  "
              f"instret={r.get('instret')}  {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
