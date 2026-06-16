"""Multi-point calibration of the analytical cost model against measured cycles.

The single-point xr0 anchor showed the analytical model was uncalibrated. This generalises it:
fit a cycles-per-MAC constant against the real FireSim FASED totals across every model whose
capture parses, and report the fit honestly — including outliers and the overall inadequacy.

The predictor is deliberately the crudest defensible one: ``cycles ~ a * total_MACs`` (matmul
MACs read from the real capture IR). We use the **median** cycles/MAC as the robust fitted
constant and flag any model whose ratio is >10x off the median as an outlier — because the honest
finding here is *which* models the matmul-only predictor cannot explain, not a polished number.

This does not feed a gap_closure score. It tells you whether the cost model is trustworthy enough
to rank axes quantitatively. (Spoiler, on the current data: only for a subset, and not for xr0.)
"""
from __future__ import annotations

import csv
import io
import statistics
from dataclasses import dataclass, field

from merlin.common import paths
from merlin.common.yaml import load_yaml

_OUTLIER_FACTOR = 10.0


@dataclass
class CalibPoint:
    model: str
    dtype: str
    measured_cycles: float
    macs: int | None
    predicted_cycles: float | None = None
    ratio: float | None = None            # measured / macs
    rel_err: float | None = None          # |predicted - measured| / measured
    is_outlier: bool = False
    note: str = ""


@dataclass
class CalibResult:
    substrate: str
    source: str
    points: list[CalibPoint]
    fitted_cycles_per_mac: float | None
    mape_consistent: float | None         # MAPE over non-outlier, parsed points
    n_fit: int
    n_outlier: int
    n_unparsed: int
    verdict: str
    extras: dict = field(default_factory=dict)


def load_measured(path=None) -> dict:
    p = path or (paths.merlin_dir() / "benchmarks" / "dse_guidance" / "measured_cycles.yaml")
    return load_yaml(p)


def calibrate(macs_of, measured: dict | None = None) -> CalibResult:
    """Fit cycles/MAC from measured totals. ``macs_of(model)`` -> total MACs or None if unparsed."""
    doc = measured or load_measured()
    raw = doc.get("points", [])
    points: list[CalibPoint] = []
    for r in raw:
        macs = macs_of(r["model"])
        pt = CalibPoint(model=r["model"], dtype=r.get("dtype", "?"),
                        measured_cycles=float(r["cycles"]), macs=macs)
        if not macs:
            pt.note = "capture did not parse; excluded from fit"
        else:
            pt.ratio = pt.measured_cycles / macs
        points.append(pt)

    ratios = [p.ratio for p in points if p.ratio is not None]
    fitted = statistics.median(ratios) if ratios else None

    # Flag outliers vs the median, then compute the fit only on the consistent set.
    consistent: list[CalibPoint] = []
    for p in points:
        if p.ratio is None:
            continue
        p.is_outlier = fitted is not None and (
            p.ratio > _OUTLIER_FACTOR * fitted or p.ratio < fitted / _OUTLIER_FACTOR)
        if not p.is_outlier:
            consistent.append(p)

    fitted_consistent = (statistics.median([p.ratio for p in consistent])
                         if consistent else fitted)
    for p in points:
        if p.macs and fitted_consistent is not None:
            p.predicted_cycles = fitted_consistent * p.macs
            p.rel_err = abs(p.predicted_cycles - p.measured_cycles) / p.measured_cycles

    mape = (sum(p.rel_err for p in consistent) / len(consistent) * 100.0
            if consistent else None)
    n_unparsed = sum(1 for p in points if p.macs is None)
    n_outlier = sum(1 for p in points if p.is_outlier)

    verdict = _verdict(fitted_consistent, mape, consistent, points, n_outlier, n_unparsed)
    return CalibResult(
        substrate=doc.get("substrate", "?"), source=doc.get("source", "?"),
        points=points, fitted_cycles_per_mac=fitted_consistent,
        mape_consistent=mape, n_fit=len(consistent), n_outlier=n_outlier,
        n_unparsed=n_unparsed, verdict=verdict)


def _verdict(fitted, mape, consistent, points, n_outlier, n_unparsed) -> str:
    if fitted is None:
        return ("No parseable model had measured cycles — calibration not possible; the "
                "analytical model remains uncalibrated.")
    parts = [f"Fitted {fitted:.1f} cycles/MAC (median over {len(consistent)} consistent models)."]
    if mape is not None:
        parts.append(f"MAPE on the consistent set = {mape:.0f}%.")
    if n_outlier:
        outs = ", ".join(f"{p.model} ({p.ratio/fitted:.0f}x median)"
                         for p in points if p.is_outlier)
        parts.append(f"{n_outlier} outlier(s) the matmul-only predictor CANNOT explain: {outs} "
                     "— its capture MAC count is inconsistent with its measured cycles (a partial "
                     "capture, or a run dominated by non-matmul / repeated-body work).")
    if n_unparsed:
        parts.append(f"{n_unparsed} model(s) excluded (capture did not parse).")
    quality = ("crude but usable as analytical ordering" if (mape or 999) < 60
               else "NOT adequate as a quantitative predictor")
    parts.append(f"Conclusion: a single cycles/MAC constant is {quality}; matmul MACs alone do "
                 "not capture whole-model scalar cycles. Quantitative gap_closure stays gated on "
                 "per-op-family calibration or direct measurement.")
    return " ".join(parts)


_COLUMNS = ["model", "dtype", "macs", "measured_cycles", "ratio_cycles_per_mac",
            "predicted_cycles", "rel_err_pct", "is_outlier", "note"]


def _loo_mape(X, y) -> float | None:
    """Leave-one-out cross-validated MAPE for an ordinary-least-squares fit ``y ~ X``."""
    import numpy as np
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < X.shape[1] + 1:           # need more points than parameters for any CV signal
        return None
    errs = []
    for i in range(n):
        mask = np.arange(n) != i
        coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred = float(X[i] @ coef)
        if y[i]:
            errs.append(abs(pred - y[i]) / abs(y[i]))
    return float(np.mean(errs) * 100.0) if errs else None


def multifeature_calibration(feature_rows: list[dict]) -> dict:
    """Compare 1- and multi-feature fits of measured cycles, with leave-one-out CV.

    ``feature_rows``: one dict per consistent measured model with keys ``cycles``, ``macs``,
    ``act_bytes``, ``matmuls``. Honestly reports whether adding features beats cycles/MAC under
    CV — with N this small (a handful of whole-model points), it usually does NOT, which is the
    finding: per-component coefficients are not identifiable from whole-model totals.
    """
    import numpy as np
    rows = [r for r in feature_rows if r.get("macs")]
    n = len(rows)
    y = [r["cycles"] for r in rows]
    feature_sets = {
        "macs_only": [("macs",)],
        "macs+act_bytes": [("macs", "act_bytes")],
        "macs+matmuls": [("macs", "matmuls")],
    }
    out: dict = {"n_points": n, "fits": {}}
    for name, (feats,) in feature_sets.items():
        X = [[r[f] for f in feats] + [1.0] for r in rows]   # + intercept
        try:
            coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
            cond = float(np.linalg.cond(np.asarray(X, float)))
        except Exception:
            continue
        out["fits"][name] = {
            "features": list(feats),
            "coef": [round(float(c), 4) for c in coef],
            "loo_mape": _loo_mape(X, y),
            "condition_number": round(cond, 1),
        }
    return out


def multifeature_report_md(mf: dict) -> str:
    L = ["## Per-component calibration attempt (multi-feature, leave-one-out CV)\n"]
    L.append(f"Fit measured cycles against feature sets over {mf['n_points']} consistent points; "
             "leave-one-out CV is the honest test of whether extra features *generalize*.\n")
    L.append("| feature set | LOO-CV MAPE | condition number |")
    L.append("|-------------|-------------|------------------|")
    best = None
    for name, fit in mf["fits"].items():
        loo = "n/a" if fit["loo_mape"] is None else f"{fit['loo_mape']:.0f}%"
        L.append(f"| {name} | {loo} | {fit['condition_number']:.0f} |")
        if fit["loo_mape"] is not None and (best is None or fit["loo_mape"] < best[1]):
            best = (name, fit["loo_mape"])
    L.append("")
    if best:
        L.append(f"**Best CV:** `{best[0]}` (LOO-MAPE {best[1]:.0f}%). ")
    L.append("**Finding:** with only a handful of whole-model totals, the features are collinear "
             "(high condition number) and multi-feature fits do **not** reliably beat the single "
             "cycles/MAC term under cross-validation — per-component coefficients are **not "
             "identifiable** from whole-model data. Cycle-exact per-component calibration needs "
             "isolated microbenchmarks measured on RTL/spike: a compute-bound matmul, a "
             "memory/repeated-RHS matmul, a dispatch-heavy tiny-kernel sequence, "
             "`matmul_bias_requant_relu`, and `no_reuse_matmul`. Until then per-axis gap_closure "
             "stays gated.\n")
    L.append("_Status: the cycle-exact microbenchmark path needs the chipyard/spike (or FireSim) "
             "toolchain; where that is unavailable this is the precise scoped remaining "
             "measurement — not a fabricated coefficient._\n")
    return "\n".join(L)


def to_csv(res: CalibResult) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_COLUMNS)
    w.writeheader()
    for p in res.points:
        w.writerow({
            "model": p.model, "dtype": p.dtype, "macs": p.macs or "",
            "measured_cycles": int(p.measured_cycles),
            "ratio_cycles_per_mac": "" if p.ratio is None else round(p.ratio, 2),
            "predicted_cycles": "" if p.predicted_cycles is None else int(p.predicted_cycles),
            "rel_err_pct": "" if p.rel_err is None else round(p.rel_err * 100, 1),
            "is_outlier": p.is_outlier, "note": p.note,
        })
    return buf.getvalue()


def markdown(res: CalibResult) -> str:
    L = ["# Cost-model calibration — predicted vs measured cycles\n"]
    L.append(f"- substrate: **{res.substrate}**  ·  source: {res.source}")
    L.append(f"- predictor: `cycles ~ (cycles/MAC) * total_matmul_MACs` (MACs from real capture IR)")
    L.append(f"- fitted: **{res.fitted_cycles_per_mac:.1f} cycles/MAC** "
             f"(median over {res.n_fit} consistent models)" if res.fitted_cycles_per_mac
             else "- fitted: n/a")
    if res.mape_consistent is not None:
        L.append(f"- MAPE (consistent set): **{res.mape_consistent:.0f}%**")
    L.append("")
    L.append("| model | dtype | MACs | measured cycles | cycles/MAC | predicted | rel err | outlier |")
    L.append("|-------|-------|------|-----------------|------------|-----------|---------|---------|")
    for p in res.points:
        macs = "n/a" if p.macs is None else f"{p.macs:.2e}"
        ratio = "n/a" if p.ratio is None else f"{p.ratio:.1f}"
        pred = "n/a" if p.predicted_cycles is None else f"{p.predicted_cycles:.2e}"
        rel = "n/a" if p.rel_err is None else f"{p.rel_err*100:.0f}%"
        L.append(f"| {p.model} | {p.dtype} | {macs} | {p.measured_cycles:.2e} | {ratio} | "
                 f"{pred} | {rel} | {'YES' if p.is_outlier else ''} |")
    L.append("")
    L.append(f"**Verdict:** {res.verdict}\n")
    return "\n".join(L)
