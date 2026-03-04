#!/usr/bin/env python3
"""
Plot sweep results produced by run_remote_sweep.py.

- Reads the CSV
- Filters successful runs (exit_code==0 and csv_parse_ok==1)
- Aggregates repeats (mean/std) per sweep value
- Produces PNG plots and aggregated CSV tables per sweep_param
- Summarizes min_all/max_all and renders 2D grid heatmaps (if present)
- NEW: Automatically plots ALL aggregated metrics (anything ending in *_mean)

Requires:
  pip install matplotlib

Usage:
  python3 plot_sweep_results.py --in_csv results/sweep.csv --out_dir results/plots

Optional:
  --no_plot_all_metrics   Disable auto-plotting of every aggregated metric
  --verbose               Print each generated PNG/CSV path
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, str) and x.startswith(("0x", "0X")):
            return int(x, 16)
        return int(x)
    except Exception:
        return default


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def stddev(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def popcount_u64(x: int) -> int:
    x &= (1 << 64) - 1
    return bin(x).count("1")


def read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def safe_metric(row: Dict[str, str], key: str) -> float:
    return to_float(row.get(key, ""))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in rows:
            w.writerow({k: rr.get(k, "") for k in fieldnames})


def effective_seconds(row: Dict[str, str], fallback_duration: float = 10.0) -> float:
    """
    Many totals are accumulated only after warmup.
    Use effective = duration_s - warmup_s (per row).
    """
    dur = to_float(row.get("duration_s", ""), fallback_duration)
    warm = to_float(row.get("warmup_s", ""), 0.0)
    eff = dur - warm
    if not math.isfinite(eff) or eff <= 0:
        eff = dur if (math.isfinite(dur) and dur > 0) else fallback_duration
    return eff


def safe_div(n: float, d: float) -> float:
    if not (math.isfinite(n) and math.isfinite(d)) or d == 0.0:
        return float("nan")
    return n / d


def sanitize_filename(s: str) -> str:
    # Keep it simple/portable.
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def plot_line_with_err(
    x: List[float],
    y: List[float],
    yerr: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    baseline_y: Optional[float] = None,
    refline: Optional[str] = None,  # "y=x"
) -> None:
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=4)
    if baseline_y is not None and math.isfinite(baseline_y):
        plt.axhline(baseline_y, linestyle="--", linewidth=1.0)
    if refline == "y=x" and x:
        xmin, xmax = min(x), max(x)
        plt.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_scatter(
    x: List[float],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    refline: Optional[str] = None,  # "y=x"
) -> None:
    plt.figure()
    plt.scatter(x, y, s=18)
    if refline == "y=x" and x and y:
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_heatmap(
    xs: List[float],
    ys: List[float],
    z: List[List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
) -> None:
    plt.figure()
    plt.imshow(z, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(xs)), [str(x) for x in xs], rotation=45, ha="right")
    plt.yticks(range(len(ys)), [str(y) for y in ys])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def baseline_for_metric(mean_key: str, baseline_vals: Dict[str, float], baseline_row: Optional[Dict[str, str]]) -> Optional[float]:
    """
    Try to find a sensible baseline horizontal line for an aggregated metric.
    - If we stored it in baseline_vals, use that.
    - Else, if it's a raw metric like d_lat_p99_ms_mean -> look up d_lat_p99_ms in baseline row.
    """
    if mean_key in baseline_vals and math.isfinite(baseline_vals[mean_key]):
        return baseline_vals[mean_key]

    if baseline_row is None:
        return None

    if mean_key.endswith("_mean"):
        raw = mean_key[:-5]  # strip "_mean"
        v = to_float(baseline_row.get(raw, ""))
        if math.isfinite(v):
            return v

    return None


def auto_plot_all_metrics(
    sp: str,
    xs: List[float],
    table_rows: List[Dict[str, Any]],
    out_dir: Path,
    xlabel: str,
    baseline_vals: Dict[str, float],
    baseline_row: Optional[Dict[str, str]],
    already_plotted: Set[str],
    verbose: bool,
) -> int:
    """
    For this sweep, plot every aggregated metric series that appears as *_mean (with *_std if present).
    Returns number of PNGs generated.
    """
    if not table_rows:
        return 0

    # Discover all mean keys
    mean_keys = sorted({k for rr in table_rows for k in rr.keys() if k.endswith("_mean")})
    made = 0

    for mean_key in mean_keys:
        std_key = mean_key[:-5] + "_std"

        # skip non-numeric / boring series, and skip duplicates we've already plotted
        if mean_key in ("effective_s_mean",):
            continue
        if mean_key in already_plotted:
            continue

        y = [to_float(rr.get(mean_key, float("nan"))) for rr in table_rows]
        if all(math.isnan(v) for v in y):
            continue

        yerr = [to_float(rr.get(std_key, 0.0), 0.0) for rr in table_rows]
        base = baseline_for_metric(mean_key, baseline_vals, baseline_row)

        safe_name = sanitize_filename(mean_key)
        out_png = out_dir / f"sweep_{sanitize_filename(sp)}_{safe_name}.png"
        plot_line_with_err(
            xs,
            y,
            yerr,
            title=f"Sweep {sp}: {mean_key}",
            xlabel=xlabel,
            ylabel=mean_key,
            out_png=out_png,
            baseline_y=base,
        )
        made += 1
        already_plotted.add(mean_key)
        if verbose:
            print(f"[png] {out_png}")

    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_plot_all_metrics", action="store_true", help="Disable plotting every aggregated metric")
    ap.add_argument("--verbose", action="store_true", help="Print each generated file path")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = read_rows(in_csv)

    # Filter ok rows.
    ok: List[Dict[str, str]] = []
    for r in rows:
        if r.get("csv_parse_ok") != "1":
            continue
        if str(r.get("exit_code", "")) not in ("0", "0.0"):
            continue
        ok.append(r)

    if not ok:
        print("No successful rows to plot.")
        return 1

    # Baseline (first baseline row).
    baseline = None
    for r in ok:
        if r.get("plan_kind") == "baseline":
            baseline = r
            break

    # Baseline derived values
    baseline_vals: Dict[str, float] = {}
    if baseline is not None:
        eff = effective_seconds(baseline)
        baseline_vals["dronet_ach_hz_mean"] = safe_div(to_float(baseline.get("dronet_total", "")), eff)
        baseline_vals["mlp_ach_hz_mean"] = safe_div(to_float(baseline.get("mlp_total", "")), eff)
        baseline_vals["mlp_miss_rate_mean"] = safe_div(to_float(baseline.get("mlp_misses", "")), eff)

        req_mlp = to_float(baseline.get("mlp_hz", ""))
        baseline_vals["mlp_miss_ratio_mean"] = safe_div(to_float(baseline.get("mlp_misses", "")), req_mlp * eff)

        req_d = to_float(baseline.get("dronet_sensor_hz", ""))
        kept = safe_div(to_float(baseline.get("dronet_total", "")), req_d * eff)
        baseline_vals["dronet_drop_ratio_mean"] = float("nan") if math.isnan(kept) else (1.0 - kept)

        mf = to_float(baseline.get("m_fresh", ""))
        ms = to_float(baseline.get("m_stale", ""))
        baseline_vals["stale_ratio_mean"] = safe_div(ms, mf + ms)

    # Helper: x-value for a given parameter name.
    def x_value(param: str, r: Dict[str, str]) -> float:
        if param == "core_mask":
            mask = to_int(r.get("core_mask", "0"), 0)
            return float(popcount_u64(mask))
        if param in r and r.get(param, "") not in ("", None):
            return to_float(r.get(param, ""))
        return to_float(r.get("sweep_value", ""))

    # Some known raw metrics (optional direct aggregation)
    key_candidates = [
        "dronet_total",
        "mlp_total",
        "mlp_misses",
        "d_lat_avg_ms",
        "d_lat_p50_ms",
        "d_lat_p90_ms",
        "d_lat_p99_ms",
        "m_lat_avg_ms",
        "m_lat_p50_ms",
        "m_lat_p90_ms",
        "m_lat_p99_ms",
        "m_jitter_avg_ms",
        "m_jitter_p50_ms",
        "m_jitter_p99_ms",
        "d_blocked",
        "m_blocked",
        "m_stale",
        "m_fresh",
    ]

    # Prepare sweeps.
    sweeps: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in ok:
        if r.get("plan_kind") == "sweep":
            sp = r.get("sweep_param", "")
            if sp:
                sweeps[sp].append(r)

    total_png = 0
    total_csv = 0

    # For each sweep_param, aggregate by x.
    for sp, srows in sweeps.items():
        buckets: Dict[float, List[Dict[str, str]]] = defaultdict(list)
        for r in srows:
            xv = x_value(sp, r)
            if math.isnan(xv):
                continue
            buckets[xv].append(r)

        xs = sorted(buckets.keys())
        if not xs:
            continue

        table_rows: List[Dict[str, Any]] = []

        for xv in xs:
            group = buckets[xv]

            effs = [effective_seconds(g) for g in group]
            dronet_ach_hzs = [safe_div(to_float(g.get("dronet_total", "")), effective_seconds(g)) for g in group]
            mlp_ach_hzs = [safe_div(to_float(g.get("mlp_total", "")), effective_seconds(g)) for g in group]
            miss_rates = [safe_div(to_float(g.get("mlp_misses", "")), effective_seconds(g)) for g in group]

            miss_ratios: List[float] = []
            drop_ratios: List[float] = []
            stale_ratios: List[float] = []
            d_block_rates: List[float] = []
            m_block_rates: List[float] = []

            for g in group:
                eff = effective_seconds(g)

                req_mlp = to_float(g.get("mlp_hz", ""))
                miss = to_float(g.get("mlp_misses", ""))
                miss_ratios.append(safe_div(miss, req_mlp * eff))

                req_d = to_float(g.get("dronet_sensor_hz", ""))
                d_total = to_float(g.get("dronet_total", ""))
                kept = safe_div(d_total, req_d * eff)
                drop_ratios.append(float("nan") if math.isnan(kept) else (1.0 - kept))

                mf = to_float(g.get("m_fresh", ""))
                ms = to_float(g.get("m_stale", ""))
                stale_ratios.append(safe_div(ms, mf + ms))

                d_blocked = to_float(g.get("d_blocked", ""))
                m_blocked = to_float(g.get("m_blocked", ""))
                d_block_rates.append(safe_div(d_blocked, eff))
                m_block_rates.append(safe_div(m_blocked, eff))

            agg: Dict[str, Any] = {
                "sweep_param": sp,
                "x": xv,
                "n": len(group),
                "effective_s_mean": mean(effs),
                "effective_s_std": stddev(effs),
            }

            # Achieved throughput (warmup-corrected)
            agg["dronet_ach_hz_mean"] = mean(dronet_ach_hzs)
            agg["dronet_ach_hz_std"] = stddev(dronet_ach_hzs)
            agg["mlp_ach_hz_mean"] = mean(mlp_ach_hzs)
            agg["mlp_ach_hz_std"] = stddev(mlp_ach_hzs)

            # Misses
            agg["mlp_miss_rate_mean"] = mean(miss_rates)
            agg["mlp_miss_rate_std"] = stddev(miss_rates)
            agg["mlp_miss_ratio_mean"] = mean(miss_ratios)
            agg["mlp_miss_ratio_std"] = stddev(miss_ratios)

            # Drops / staleness
            agg["dronet_drop_ratio_mean"] = mean(drop_ratios)
            agg["dronet_drop_ratio_std"] = stddev(drop_ratios)
            agg["stale_ratio_mean"] = mean(stale_ratios)
            agg["stale_ratio_std"] = stddev(stale_ratios)

            # Backpressure rates
            agg["d_blocked_rate_mean"] = mean(d_block_rates)
            agg["d_blocked_rate_std"] = stddev(d_block_rates)
            agg["m_blocked_rate_mean"] = mean(m_block_rates)
            agg["m_blocked_rate_std"] = stddev(m_block_rates)

            # Requested rates (averaged)
            req_mlp_vals = [to_float(g.get("mlp_hz", "")) for g in group]
            req_d_vals = [to_float(g.get("dronet_sensor_hz", "")) for g in group]
            agg["mlp_req_hz_mean"] = mean(req_mlp_vals)
            agg["mlp_req_hz_std"] = stddev(req_mlp_vals)
            agg["dronet_req_hz_mean"] = mean(req_d_vals)
            agg["dronet_req_hz_std"] = stddev(req_d_vals)

            # Direct raw metrics if present
            for key in key_candidates:
                vals = [safe_metric(g, key) for g in group]
                if all(math.isnan(v) for v in vals):
                    continue
                agg[f"{key}_mean"] = mean(vals)
                agg[f"{key}_std"] = stddev(vals)

            table_rows.append(agg)

        # Write aggregated CSV
        out_table = out_dir / f"sweep_{sanitize_filename(sp)}_agg.csv"
        fieldnames = sorted({k for rr in table_rows for k in rr.keys()})
        write_csv(out_table, fieldnames, table_rows)
        total_csv += 1
        if args.verbose:
            print(f"[csv] {out_table}")

        xlabel = "cores" if sp == "core_mask" else sp

        already_plotted: Set[str] = set()

        # A few “main” plots (still generated, but now everything else also gets plotted)
        def add_plot(mean_key: str, std_key: str, title: str, ylabel: str, filename: str, refline: Optional[str] = None) -> None:
            nonlocal total_png
            y = [to_float(rr.get(mean_key, float("nan"))) for rr in table_rows]
            if all(math.isnan(v) for v in y):
                return
            yerr = [to_float(rr.get(std_key, 0.0), 0.0) for rr in table_rows]
            out_png = out_dir / filename
            base = baseline_for_metric(mean_key, baseline_vals, baseline)
            plot_line_with_err(xs, y, yerr, title, xlabel, ylabel, out_png, baseline_y=base, refline=refline)
            already_plotted.add(mean_key)
            total_png += 1
            if args.verbose:
                print(f"[png] {out_png}")

        add_plot("dronet_ach_hz_mean", "dronet_ach_hz_std", f"Sweep {sp}: dronet achieved throughput", "dronet_hz (achieved)", f"sweep_{sanitize_filename(sp)}_dronet_ach_hz.png")
        add_plot("mlp_ach_hz_mean", "mlp_ach_hz_std", f"Sweep {sp}: mlp achieved throughput", "mlp_hz (achieved)", f"sweep_{sanitize_filename(sp)}_mlp_ach_hz.png")

        if sp == "mlp_hz":
            add_plot("mlp_ach_hz_mean", "mlp_ach_hz_std", "MLP keep-up: achieved vs requested", "mlp_hz (achieved)", "keepup_mlp_hz.png", refline="y=x")
        if sp == "dronet_sensor_hz":
            add_plot("dronet_ach_hz_mean", "dronet_ach_hz_std", "Dronet keep-up: achieved vs requested", "dronet_hz (achieved)", "keepup_dronet_sensor_hz.png", refline="y=x")

        add_plot("mlp_miss_rate_mean", "mlp_miss_rate_std", f"Sweep {sp}: mlp miss rate", "misses/sec", f"sweep_{sanitize_filename(sp)}_mlp_miss_rate.png")
        add_plot("mlp_miss_ratio_mean", "mlp_miss_ratio_std", f"Sweep {sp}: mlp miss ratio", "misses / requested_inputs", f"sweep_{sanitize_filename(sp)}_mlp_miss_ratio.png")
        add_plot("stale_ratio_mean", "stale_ratio_std", f"Sweep {sp}: stale ratio", "m_stale / (m_fresh + m_stale)", f"sweep_{sanitize_filename(sp)}_stale_ratio.png")
        add_plot("dronet_drop_ratio_mean", "dronet_drop_ratio_std", f"Sweep {sp}: dronet drop ratio", "1 - kept_fraction", f"sweep_{sanitize_filename(sp)}_dronet_drop_ratio.png")
        add_plot("d_blocked_rate_mean", "d_blocked_rate_std", f"Sweep {sp}: dronet blocked rate", "blocked/sec", f"sweep_{sanitize_filename(sp)}_d_blocked_rate.png")
        add_plot("m_blocked_rate_mean", "m_blocked_rate_std", f"Sweep {sp}: mlp blocked rate", "blocked/sec", f"sweep_{sanitize_filename(sp)}_m_blocked_rate.png")

        # Queueing curves (scatter): throughput vs p99 latency (if both exist)
        if any("d_lat_p99_ms_mean" in rr for rr in table_rows) and any("dronet_ach_hz_mean" in rr for rr in table_rows):
            out_png = out_dir / f"sweep_{sanitize_filename(sp)}_queue_dronet_p99_vs_hz.png"
            plot_scatter(
                [to_float(rr.get("dronet_ach_hz_mean", float("nan"))) for rr in table_rows],
                [to_float(rr.get("d_lat_p99_ms_mean", float("nan"))) for rr in table_rows],
                title=f"{sp}: dronet p99 latency vs achieved throughput",
                xlabel="dronet_hz (achieved)",
                ylabel="d_lat_p99_ms",
                out_png=out_png,
            )
            total_png += 1
            if args.verbose:
                print(f"[png] {out_png}")

        if any("m_lat_p99_ms_mean" in rr for rr in table_rows) and any("mlp_ach_hz_mean" in rr for rr in table_rows):
            out_png = out_dir / f"sweep_{sanitize_filename(sp)}_queue_mlp_p99_vs_hz.png"
            plot_scatter(
                [to_float(rr.get("mlp_ach_hz_mean", float("nan"))) for rr in table_rows],
                [to_float(rr.get("m_lat_p99_ms_mean", float("nan"))) for rr in table_rows],
                title=f"{sp}: mlp p99 latency vs achieved throughput",
                xlabel="mlp_hz (achieved)",
                ylabel="m_lat_p99_ms",
                out_png=out_png,
            )
            total_png += 1
            if args.verbose:
                print(f"[png] {out_png}")

        # NEW: plot *every* aggregated metric (unless disabled)
        if not args.no_plot_all_metrics:
            total_png += auto_plot_all_metrics(
                sp=sp,
                xs=xs,
                table_rows=table_rows,
                out_dir=out_dir,
                xlabel=xlabel,
                baseline_vals=baseline_vals,
                baseline_row=baseline,
                already_plotted=already_plotted,
                verbose=args.verbose,
            )

    # Save a quick summary CSV for corners if present
    corners = [r for r in ok if r.get("plan_kind") in ("min_all", "max_all")]
    if corners:
        corner_out = out_dir / "corners.csv"
        keys = sorted({k for r in corners for k in r.keys()})
        write_csv(corner_out, keys, corners)
        total_csv += 1
        if args.verbose:
            print(f"[csv] {corner_out}")

    # Render grid2d heatmaps if present (selected metrics)
    grid_rows = [r for r in ok if (r.get("plan_kind") or "").startswith("grid")]
    if grid_rows:
        grids: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for r in grid_rows:
            gp = r.get("grid_params", "") or ""
            if gp:
                grids[gp].append(r)

        def metric_value(name: str, r: Dict[str, str]) -> float:
            eff = effective_seconds(r)
            if name == "dronet_ach_hz":
                return safe_div(to_float(r.get("dronet_total", "")), eff)
            if name == "mlp_ach_hz":
                return safe_div(to_float(r.get("mlp_total", "")), eff)
            if name == "d_lat_p99_ms":
                return to_float(r.get("d_lat_p99_ms", ""))
            if name == "m_lat_p99_ms":
                return to_float(r.get("m_lat_p99_ms", ""))
            if name == "mlp_miss_ratio":
                req_mlp = to_float(r.get("mlp_hz", ""))
                miss = to_float(r.get("mlp_misses", ""))
                return safe_div(miss, req_mlp * eff)
            if name == "stale_ratio":
                mf = to_float(r.get("m_fresh", ""))
                ms = to_float(r.get("m_stale", ""))
                return safe_div(ms, mf + ms)
            return float("nan")

        heat_metrics = [
            ("dronet_ach_hz", "dronet_hz (achieved)"),
            ("mlp_ach_hz", "mlp_hz (achieved)"),
            ("d_lat_p99_ms", "dronet p99 (ms)"),
            ("m_lat_p99_ms", "mlp p99 (ms)"),
            ("mlp_miss_ratio", "mlp miss ratio"),
            ("stale_ratio", "stale ratio"),
        ]

        for gp, grows in grids.items():
            if ":" not in gp:
                continue
            xparam, yparam = gp.split(":", 1)

            cell: Dict[Tuple[float, float], List[Dict[str, str]]] = defaultdict(list)
            xset, yset = set(), set()
            for r in grows:
                xv = x_value(xparam, r)
                yv = x_value(yparam, r)
                if math.isnan(xv) or math.isnan(yv):
                    continue
                xset.add(xv)
                yset.add(yv)
                cell[(xv, yv)].append(r)

            xs = sorted(xset)
            ys = sorted(yset)
            if not xs or not ys:
                continue

            for mname, mlabel in heat_metrics:
                z: List[List[float]] = []
                for yv in ys:
                    rowz: List[float] = []
                    for xv in xs:
                        group = cell.get((xv, yv), [])
                        vals = [metric_value(mname, rr) for rr in group]
                        rowz.append(mean(vals))
                    z.append(rowz)

                out_png = out_dir / f"grid_{sanitize_filename(gp.replace(':','_'))}_{sanitize_filename(mname)}.png"
                plot_heatmap(
                    xs=xs,
                    ys=ys,
                    z=z,
                    title=f"Grid {gp}: {mlabel}",
                    xlabel=("cores" if xparam == "core_mask" else xparam),
                    ylabel=("cores" if yparam == "core_mask" else yparam),
                    out_png=out_png,
                )
                total_png += 1
                if args.verbose:
                    print(f"[png] {out_png}")

    print(f"Wrote outputs to: {out_dir}")
    print(f"Generated: {total_png} PNG(s), {total_csv} CSV(s)")
    if not args.no_plot_all_metrics:
        print("Note: auto-plot-all-metrics is ENABLED (disable with --no_plot_all_metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())