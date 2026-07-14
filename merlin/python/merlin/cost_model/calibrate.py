"""Calibrate the Gemmini cost model against the cycle-exact Verilator sim, then validate.

Flow:
  1. Build + run the calibration microbenchmarks (calib/gemmini_costmodel_calib.c) under
     Verilator; each isolates one command class, rdcycle measures the region only.
  2. Least-squares fit  cycles = const + Σ coeff[e]·n_e  over the analytic event counts.
  3. Validate on the Stage-F slate harnesses (separate from the fit set): predict region
     cycles from each variant's events, run the same harness under Verilator, report error
     AND confirm the cost model preserves each insight's act/park decision.

Env: MERLIN_CHIPYARD (default /path/to/chipyard). Verilator binary must be built
for a Gemmini config (GemminiAndOPUShuttleConfig). Slow: ~tens of seconds per RTL run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from merlin.cost_model.gemmini import EVENTS, GemminiCostModel

HERE = Path(__file__).resolve().parent
CALIB_SRC = HERE / "calib" / "gemmini_costmodel_calib.c"
# The cost-calibration ablation kernels are a LIBRARY-consumed benchmark input (this module compiles
# them), so they live under merlin/benchmarks/, not experiments/. HERE=cost_model -> parents[2]=merlin/.
STAGEF = HERE.parents[2] / "benchmarks" / "cost_calib"
from merlin.common.driver_output import int_after as _int_after

KIND = {"mvin": 0, "mvin2": 1, "compute": 2, "mvout": 3, "config": 4, "fence": 5, "matmul": 6}


def paths() -> dict:
    cy = Path(os.environ.get("MERLIN_CHIPYARD", "/path/to/chipyard"))
    t = cy / ".conda-env" / "riscv-tools"
    r = cy / "generators" / "gemmini" / "software" / "gemmini-rocc-tests"
    sims = list((cy / "sims" / "verilator").glob("simulator-*Gemmini*"))
    return {"gcc": t / "bin" / "riscv64-unknown-elf-gcc", "rocc": r,
            "bc": r / "riscv-tests" / "benchmarks" / "common",
            "sim": sims[0] if sims else None, "simdir": cy / "sims" / "verilator"}


def _cflags(p: dict, defs: list[str]) -> list[str]:
    bc, r = p["bc"], p["rocc"]
    return ["-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
            "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
            "-march=rv64gc", "-Wa,-march=rv64gc", "-nostdlib", "-nostartfiles", "-static",
            "-T", str(bc / "test.ld"), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1",
            *defs, f"-I{r}", f"-I{r}/riscv-tests", f"-I{r}/riscv-tests/env", f"-I{bc}"]


def build(p: dict, src: Path, defs: list[str], out: Path) -> Path:
    bc = p["bc"]
    cmd = [str(p["gcc"]), *_cflags(p, defs), str(src), str(bc / "crt.S"),
           str(bc / "syscalls.c"), "-lm", "-lgcc", "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def run_rtl(p: dict, binary: Path, label: str, timeout: int = 600) -> int:
    proc = subprocess.run([str(p["sim"]), "+permissive", "+permissive-off", str(binary)],
                          cwd=str(p["simdir"]), capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    cyc = _int_after(out, label)   # driver prints "... CYCLES <n>" / "REGION_CYCLES <n>"
    if cyc is None:
        raise RuntimeError(f"no {label} line from {binary.name}:\n{out[-400:]}")
    return cyc


def calib_events(kind: str, count: int) -> dict[str, float]:
    """Analytic region events between the two rdcycle reads, per microbenchmark KIND."""
    e = {k: 0.0 for k in EVENTS}
    if kind == "mvin":
        e["mvin_A"], e["fence"] = count, 1
    elif kind == "mvin2":
        e["mvin2_B"], e["fence"] = count, 1
    elif kind == "compute":
        e["mvin_A"], e["mvin2_B"], e["compute"], e["fence"] = 1, 1, count, 1
    elif kind == "mvout":
        e["mvin_A"], e["mvin2_B"], e["compute"], e["mvout"], e["fence"] = 1, 1, 1, count, 1
    elif kind == "config":
        e["config"], e["fence"] = count, 1
    elif kind == "fence":
        e["mvin_A"], e["fence"] = count, count
    elif kind == "matmul":
        e["mvin2_B"], e["mvin_A"], e["compute"], e["mvout"], e["fence"] = 1, count, count, count, 1
    return e


def calibrate(p: dict, counts=(4, 16), tmp=Path("/tmp/cmcalib")) -> tuple[GemminiCostModel, list]:
    tmp.mkdir(parents=True, exist_ok=True)
    rows, X, y = [], [], []
    for kind, kid in KIND.items():
        for c in counts:
            b = build(p, CALIB_SRC, [f"-DKIND={kid}", f"-DCOUNT={c}"], tmp / f"k{kid}_c{c}")
            cyc = run_rtl(p, b, "CYCLES")
            ev = calib_events(kind, c)
            rows.append({"kind": kind, "count": c, "cycles": cyc, "events": ev})
            X.append([1.0] + [ev[e] for e in EVENTS])
            y.append(cyc)
            print(f"  calib {kind:<8} count={c:>2} -> {cyc} cyc")
    return fit_model(rows, p["sim"].name if p["sim"] else "?", counts)


def fit_model(rows: list[dict], sim: str = "?", counts=(4, 16)) -> tuple[GemminiCostModel, list]:
    """Relative-error-weighted least squares over calibration rows.

    Weighting by 1/cycles minimizes MAPE (what we act on), so tiny config/fence runs are not
    swamped by the large matmul runs — unweighted abs-residual lstsq otherwise drives the
    near-free ``config`` coefficient to ~0 and inflates the intercept.
    """
    A = np.array([[1.0] + [r["events"][e] for e in EVENTS] for r in rows])
    b = np.array([r["cycles"] for r in rows], dtype=float)
    w = 1.0 / np.maximum(b, 1.0)
    coef, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    ape = np.abs(A @ coef - b) / np.maximum(b, 1)
    model = GemminiCostModel(
        const=float(coef[0]),
        coeff={e: float(coef[i + 1]) for i, e in enumerate(EVENTS)},
        error={"mape": float(ape.mean()), "max_abs_pct": float(ape.max()),
               "n_points": len(b)},
        meta={"fidelity": "L2.5 calibrated (linear, serial; no overlap)",
              "fit": "relative-error-weighted lstsq", "sim": sim, "counts": list(counts)})
    return model, rows


# Stage-F validation harnesses: (source, variant defs, analytic events, expected sign of decision).
def slate_events_resident(variant: str, reps: int) -> dict:
    e = {k: 0.0 for k in EVENTS}
    nb = reps if variant == "baseline" else 1
    e.update(mvin2_B=nb, mvin_A=reps, compute=reps, mvout=reps, fence=1)
    return e


def slate_events_dispatch(variant: str, tiles: int) -> dict:
    e = {k: 0.0 for k in EVENTS}
    if variant == "baseline":
        e.update(config=tiles * 4, mvin2_B=tiles, mvin_A=tiles, compute=tiles,
                 mvout=tiles, fence=tiles)
    else:
        e.update(config=4, mvin2_B=1, mvin_A=tiles, compute=tiles, mvout=tiles, fence=1)
    return e


def validate(model: GemminiCostModel, p: dict, tmp=Path("/tmp/cmcalib")) -> list:
    """Predict vs measured cycles on slate harnesses NOT used for the fit."""
    checks = []
    cases = [
        ("resident_rhs", STAGEF / "resident_rhs_ablation.c", "REPS", 16,
         [("baseline", slate_events_resident("baseline", 16)),
          ("hoisted", slate_events_resident("hoisted", 16))]),
        ("dispatch_batching", STAGEF / "dispatch_batching_ablation.c", "TILES", 16,
         [("baseline", slate_events_dispatch("baseline", 16)),
          ("batched", slate_events_dispatch("batched", 16))]),
    ]
    for name, src, knob, n, variants in cases:
        measured, predicted = {}, {}
        for variant, ev in variants:
            # add a rdcycle-bracketed copy: reuse the slate harness but it lacks the print;
            # instead predict from events and measure end-to-end gemmini-region via a wrapper.
            predicted[variant] = model.predict(ev)
            b = build(p, src, [f"-DVARIANT_{variant.upper()}", f"-D{knob}={n}",
                               "-DCOSTMODEL_TIME=1"], tmp / f"{name}_{variant}")
            measured[variant] = run_rtl(p, b, "REGION_CYCLES")
        for variant in measured:
            m_, pr = measured[variant], predicted[variant]
            checks.append({"harness": name, "variant": variant, "measured": m_,
                           "predicted": round(pr, 1),
                           "abs_pct_err": round(abs(pr - m_) / max(m_, 1) * 100, 1)})
        # decision preserved? ratio of (baseline / other) must agree in sign with measured
        keys = [v for v, _ in variants]
        mr = measured[keys[0]] / max(measured[keys[1]], 1)
        prr = predicted[keys[0]] / max(predicted[keys[1]], 1)
        checks.append({"harness": name, "decision_ratio_measured": round(mr, 2),
                       "decision_ratio_predicted": round(prr, 2),
                       "decision_preserved": (mr > 1) == (prr > 1)})
    return checks


def _print_validation(checks: list, val_mape: float) -> None:
    print("\nvalidation (predicted vs measured cycles, held out from the fit):")
    for c in checks:
        if "abs_pct_err" in c:
            print(f"  {c['harness']:<18} {c['variant']:<9} "
                  f"meas={c['measured']:>6} pred={c['predicted']:>8} err={c['abs_pct_err']}%")
        elif "decision_preserved" in c:
            ok = "OK" if c["decision_preserved"] else "BROKEN"
            print(f"  {c['harness']:<18} decision ratio meas={c['decision_ratio_measured']} "
                  f"pred={c['decision_ratio_predicted']} -> {ok}")
    print(f"  validation MAPE (operative band on realistic kernels) = {val_mape*100:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "gemmini_cost_coeffs.json"))
    ap.add_argument("--report", default="out/artifacts/cache/cost_model/calibration.json")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--refit", default=None,
                    help="refit offline from a saved calibration.json (no RTL); reuses its "
                         "calibration_rows + validation measured cycles")
    args = ap.parse_args()

    if args.refit:
        saved = json.loads(Path(args.refit).read_text(encoding="utf-8"))
        model, rows = fit_model(saved["calibration_rows"])
        # recompute validation predictions from the saved MEASURED cycles (no RTL)
        checks, errs = [], []
        for c in saved.get("validation", []):
            if "measured" not in c:
                continue
            ev = (slate_events_resident(c["variant"], 16) if c["harness"] == "resident_rhs"
                  else slate_events_dispatch(c["variant"], 16))
            pr = model.predict(ev)
            err = abs(pr - c["measured"]) / max(c["measured"], 1)
            errs.append(err)
            checks.append({"harness": c["harness"], "variant": c["variant"],
                           "measured": c["measured"], "predicted": round(pr, 1),
                           "abs_pct_err": round(err * 100, 1)})
        val_mape = float(np.mean(errs)) if errs else 0.0
        model.error["fit_mape"] = model.error["mape"]
        model.error["validation_mape"] = val_mape
        model.error["mape"] = val_mape  # operative band = realistic-kernel error, not degenerate fit
        model.save(args.out)
        print(f"refit (weighted) from {args.refit}")
        print("coeffs: const=%.0f " % model.const
              + " ".join(f"{e}={model.coeff[e]:.1f}" for e in EVENTS))
        _print_validation(checks, val_mape)
        Path(args.report).write_text(json.dumps(
            {"coeffs": model.coeff, "const": model.const, "error": model.error,
             "calibration_rows": rows, "validation": checks}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}\nwrote {args.report}")
        return 0

    p = paths()
    for k in ("gcc", "sim"):
        if not p[k] or not Path(p[k]).exists():
            sys.exit(f"missing {k}: {p[k]} (set MERLIN_CHIPYARD / build the Gemmini sim)")
    print("calibrating against", p["sim"].name)
    model, rows = calibrate(p)
    model.save(args.out)  # save before validation so coefficients survive a validation error
    print("\ncoeffs (cycles/command): const=%.0f " % model.const
          + " ".join(f"{e}={model.coeff[e]:.1f}" for e in EVENTS))
    print(f"fit MAPE={model.error['mape']*100:.1f}%  max={model.error['max_abs_pct']*100:.1f}%")
    report = {"coeffs": model.coeff, "const": model.const, "error": model.error,
              "calibration_rows": rows}
    if not args.no_validate:
        checks = validate(model, p)
        report["validation"] = checks
        val_errs = [c["abs_pct_err"] / 100 for c in checks if "abs_pct_err" in c]
        val_mape = float(np.mean(val_errs)) if val_errs else 0.0
        model.error["fit_mape"] = model.error["mape"]
        model.error["validation_mape"] = val_mape
        model.error["mape"] = val_mape  # operative band
        model.save(args.out)
        _print_validation(checks, val_mape)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
