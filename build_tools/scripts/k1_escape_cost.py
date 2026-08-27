#!/usr/bin/env python
"""Quantify a RUNTIME ESCAPE's cost on the real K1, by how its instruction count SCALES.

The escape audit (``merlin.mining.escape_sweep``) screens for calls the compiler emitted inside a
loop. Screening cannot say what one costs -- an in-loop call could be a cheap epilogue or, as it was
for the f32 GEMM, 77% of everything retired. This script measures that, using the scaling argument
that localized the original defect:

    instret(N) = a*N^3 + b*N^2

for a square NxNxN GEMM. ``a`` is per-MAC (real compute: a GEMM does N^3 multiply-accumulates) and
``b`` is per-OUTPUT-ELEMENT (there are N^2 outputs, so per-tile/per-element overhead lands here). A
large ``b`` beside a small ``a`` means the kernel is dominated by work that has nothing to do with
the arithmetic -- which is what a per-tile runtime call looks like from the outside. On the original
defect this fit gave a=0.197 ins/MAC and b=79 ins/output-element and predicted a held-out size to
0.4%, which is what turned "we are slow" into a located bug.

The fit is over-determined (>=3 sizes for 2 unknowns) and is REPORTED WITH ITS HELD-OUT ERROR: a fit
that cannot predict a size it did not see is not evidence, so the largest N is excluded from the fit
and used as the check.

Measurement is INSTRET on the same bracket as the timing (``ours_gemm_driver.c`` /
``ours_int8_gemm_driver.c``), gated on ``VERIFY PASS`` -- no pass, no number.

Run:  .venv/bin/python build_tools/scripts/k1_escape_cost.py --dtype f32 --sizes 64,96,128,160
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k1_large_shape_packing as L  # noqa: E402
from merlin.common.driver_output import int_after  # noqa: E402
from merlin.common.paths import repo_root  # noqa: E402
from merlin.mining import k1  # noqa: E402

TAG = "esc"          # unique remote-tag prefix: other agents measure on this board concurrently


def build_gemm(*, M: int, N: int, K: int, dtype: str, features: list[str],
               tmp: Path) -> tuple[Path | None, str]:
    """Build the standalone K1 GEMM driver around OUR compiler-emitted kernel. (binary, detail)."""
    from merlin.llvmlower import c_runtime, toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.mining import workloads
    from merlin.mining.registry import load_rvv_package
    from merlin.runtime.backends import zephyr_model as zm

    pkg_dir = "hand_v0_int8" if dtype == "int8" else "hand_v0"
    pkg = load_rvv_package(Path(repo_root()) / "out/artifacts/targets/rvv" / pkg_dir)
    pkg = replace(pkg, run_id=f"esc_{dtype}", compiler_features=list(features))

    bundle = Path(workloads.gen_matmul_f32(
        Path(repo_root()) / "out/artifacts/cache/escape-cost", M=M, N=N, K=K))
    work = tmp / "work"
    work.mkdir(parents=True, exist_ok=True)
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", work, int8_compute=pkg.is_int8)
    feats = frozenset(pkg.compiler_features or []) or None
    try:
        res = lower_model_file(prepared, work / "lower", targets=(), textual=True, vectorize=True,
                               transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                               features=feats)
    except PipelineError as e:
        return None, f"lowering failed: {str(e)[:200]}"

    model_o = work / "model.o"
    try:
        subprocess.run([str(toolchain.clang()), "--target=riscv64-unknown-linux-gnu",
                        "-march=rv64gcv", "-mabi=lp64d", "-O2", "-Wno-override-module",
                        "-c", str(res.ll_path), "-o", str(model_o)],
                       capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        return None, f"model.o compile failed: {(e.stderr or '')[-300:]}"

    cgen = work / "cgen"
    c_runtime.generate(bundle, cgen, bundle / "inputs.npz")
    rt = Path(repo_root()) / "merlin/runtime/c"
    abi = Path(repo_root()) / "merlin/runtime/abi"
    driver = L.HERE / ("ours_int8_gemm_driver.c" if dtype == "int8" else "ours_gemm_driver.c")
    binp = tmp / f"esc_gemm_{dtype}_{M}"
    inc_flags: list[str] = []
    for d in [L.K1H, L.HERE, cgen, rt]:
        inc_flags += ["-I", str(d)]
    cmd = [str(L._cc()), *inc_flags, *L._K1_CFLAGS,
           f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}", "-static", "-o", str(binp),
           str(driver), str(cgen / "model_call.c"), str(rt / "merlin_model.c"),
           str(abi / "mlir_runtime.c"), str(model_o), "-lm", "-lpthread"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not binp.is_file():
        return None, f"link failed rc={p.returncode}: {p.stderr.strip()[-400:]}"
    return binp, "ok"


def measure(binp: Path, tag: str, reps: int) -> dict:
    """Run on the board `reps` times; return the min-ticks row with INSTRET, or an honest not_run.

    Gated on VERIFY PASS: a run that did not verify yields no number at all.
    """
    ticks: list[int] = []
    instret: list[int] = []
    with k1.board_lock():
        for _ in range(reps):
            console, detail = L._deploy_run(binp, tag, timeout=1200)
            if console is None:
                return {"status": "not_run", "blocker": detail}
            if "VERIFY PASS" not in console:
                return {"status": "not_run",
                        "blocker": f"verify did not pass; tail: {console.strip()[-300:]}"}
            c, i = int_after(console, "CYCLES"), int_after(console, "INSTRET")
            if c is None or i is None:
                return {"status": "not_run", "blocker": "missing CYCLES/INSTRET line"}
            ticks.append(c)
            instret.append(i)
    return {"status": "pass", "ticks": min(ticks), "instret": min(instret),
            "ticks_runs": ticks, "instret_runs": instret, "reps": reps}


def fit_cubic_quadratic(points: list[tuple[int, int]]) -> dict:
    """Least-squares fit instret = a*N^3 + b*N^2 over (N, instret), holding out the largest N.

    Returns the coefficients plus the held-out prediction error -- the part that makes the fit
    evidence rather than curve-drawing.
    """
    import numpy as np

    pts = sorted(points)
    if len(pts) < 3:
        return {"status": "insufficient", "note": "need >=3 sizes (2 unknowns + a held-out check)"}
    train, (hn, hy) = pts[:-1], pts[-1]
    A = np.array([[n ** 3, n ** 2] for n, _ in train], dtype=float)
    y = np.array([v for _, v in train], dtype=float)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * hn ** 3 + b * hn ** 2
    return {"status": "ok", "a_per_mac": float(a), "b_per_output_elem": float(b),
            "heldout_N": hn, "heldout_actual": hy, "heldout_pred": float(pred),
            "heldout_err_pct": float(abs(pred - hy) / hy * 100.0),
            "train_sizes": [n for n, _ in train]}


def arm(dtype: str, features: list[str], sizes: list[int], reps: int) -> dict:
    """Measure one arm (dtype x feature set) across sizes and fit its scaling."""
    label = f"{dtype}:{'+'.join(features) if features else 'baseline'}"
    rows: list[dict] = []
    for N in sizes:
        with tempfile.TemporaryDirectory(prefix=f"esc_cost_{dtype}_") as td:
            binp, detail = build_gemm(M=N, N=N, K=N, dtype=dtype, features=features, tmp=Path(td))
            if binp is None:
                rows.append({"N": N, "status": "not_run", "blocker": detail})
                print(f"  N={N:<5} not_run: {detail[:120]}", flush=True)
                continue
            r = measure(binp, f"{TAG}_{dtype}_{'f' if features else 'b'}_{N}", reps)
        r["N"] = N
        rows.append(r)
        print(f"  N={N:<5} {r.get('status')} ticks={r.get('ticks')} instret={r.get('instret')}",
              flush=True)
    fit = fit_cubic_quadratic([(r["N"], r["instret"]) for r in rows if r.get("status") == "pass"])
    return {"arm": label, "dtype": dtype, "features": features, "rows": rows, "fit": fit}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="f32", choices=["f32", "int8"])
    ap.add_argument("--sizes", default="64,96,128,160")
    ap.add_argument("--features", default="erase_self_copy",
                    help="comma-separated feature set for the treatment arm ('' = control only)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    sizes = [int(s) for s in a.sizes.split(",") if s]
    feats = [f for f in a.features.split(",") if f]
    arms = [arm(a.dtype, [], sizes, a.reps)]
    if feats:
        arms.append(arm(a.dtype, feats, sizes, a.reps))

    report = {"board": "k1_spacemit", "dtype": a.dtype, "sizes": sizes, "reps": a.reps,
              "arms": arms}
    # Speedup per size, control vs treatment, only where BOTH verified.
    if len(arms) == 2:
        ctl = {r["N"]: r for r in arms[0]["rows"] if r.get("status") == "pass"}
        trt = {r["N"]: r for r in arms[1]["rows"] if r.get("status") == "pass"}
        report["delta"] = [
            {"N": n,
             "instret_ctl": ctl[n]["instret"], "instret_trt": trt[n]["instret"],
             "instret_ratio": ctl[n]["instret"] / trt[n]["instret"],
             "ticks_ctl": ctl[n]["ticks"], "ticks_trt": trt[n]["ticks"],
             "ticks_speedup": ctl[n]["ticks"] / trt[n]["ticks"]}
            for n in sorted(set(ctl) & set(trt))
        ]
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for ar in arms:
        print(f"\n{ar['arm']}: fit {json.dumps(ar['fit'])}")
    if a.out:
        dest = Path(a.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
