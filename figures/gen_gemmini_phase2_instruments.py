#!/usr/bin/env python3
"""Three panels of what phase-2 has actually MEASURED off the FPGA, from artifacts on disk.

Every number is read from a receipt, never transcribed: the movement fit and its provenance come
from the measurement product under out/artifacts/measurements/, and the per-workload intensities are
recomputed here from the canonical command buffers with the same library the campaign uses.

The point of the middle panel is the LICENCE, not the ranking. The ridge is an UPPER bound, because
the fit's slope is a marginal rate over a domain its 128.5-cycle fixed cost dominates. So only the
compute side is decidable, and workloads below the line are drawn as UNKNOWN rather than
memory-bound -- the distinction `arithmetic_intensity` used to collapse.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.perf import lane_cost, offload
from merlin.perf.optimization_ledger import arithmetic_intensity

PALETTE = ["#5e8db4", "#cf8a82", "#8fa674", "#d2a23f", "#9d7fae", "#7c9aa6", "#b08968"]
_BG = "#faf6ef"
REPO = repo_root()
BALANCE = (REPO / "out/artifacts/measurements/gsim_GemminiRocketConfig/gemmini/"
           "movement_balance_v1_20260908T184840Z_939a075/movement_balance.json")
BUFFERS = (REPO / "out/artifacts/perf-bench/gemmini/development_phase2_global_encoding_20260908/"
           "validation/canonical/default")
WORKLOADS = ("resnet50", "smolvla", "lstmnetvit", "tinyllama")


def _style():
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.figsize": (13.4, 4.4), "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 10, "font.family": "serif",
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#b9ad97",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b9ad97", "axes.labelcolor": "#2f2a23", "axes.titlecolor": "#2f2a23",
        "text.color": "#2f2a23", "xtick.color": "#5c5446", "ytick.color": "#5c5446",
        "axes.prop_cycle": cycler(color=PALETTE)})


def collect() -> dict:
    balance = json.loads(BALANCE.read_text())
    rows = []
    for name in WORKLOADS:
        path = BUFFERS / name / "command_buffer.json"
        if not path.is_file():
            continue
        cb = json.loads(path.read_text())
        rep, cost = offload.offload_report(cb), lane_cost.lane_cost(cb)
        verdict = arithmetic_intensity(rep.routed_macs, cost.traffic_bytes,
                                       machine_macs_per_byte=balance["ridge_point"][
                                           "macs_per_byte_upper_bound"])
        rows.append({"model": name, "routed_macs": rep.routed_macs,
                     "on_unit": rep.contractions_on_unit,
                     "off_unit": rep.contractions_off_unit,
                     "traffic_bytes": cost.traffic_bytes,
                     "host_regions": cost.host_lane_region_count,
                     "macs_per_byte": verdict.get("macs_per_byte"),
                     "bound_by": verdict.get("bound_by")})
    return {"balance": balance, "workloads": rows}


def render(data: dict, out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    bal, rows = data["balance"], data["workloads"]
    fit, ridge = bal["balance"], bal["ridge_point"]["macs_per_byte_upper_bound"]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    # -- panel 0: the movement fit, the one measurement that set the direction ------------------
    xs = [s["bytes_moved"] for s in fit["samples"]]
    ys = [s["cycles"] for s in fit["samples"]]
    ax0.scatter(xs, ys, s=42, color=PALETTE[0], zorder=3, label="GSIM samples")
    grid = [0, max(xs) * 1.08]
    slope, base = 1.0 / fit["peak_bytes_per_cycle"], fit["base_latency_cycles"]
    ax0.plot(grid, [base + slope * g for g in grid], color=PALETTE[1], lw=1.8,
             label=f"{fit['peak_bytes_per_cycle']:.3f} B/cycle")
    ax0.axhline(base, color=PALETTE[4], ls=":", lw=1.4,
                label=f"fixed cost {base:.1f} cyc")
    ax0.set_xlabel("bytes moved"); ax0.set_ylabel("cycles")
    ax0.set_title(f"Movement balance, measured\n"
                  f"n={fit['n_samples']} sizes, $r^2$={fit['r_squared']:.5f}")
    ax0.legend(loc="upper left", framealpha=0.9)
    ax0.annotate("the intercept dominates this domain,\nso the slope is a MARGINAL rate",
                 xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8,
                 color="#6b5f4d", style="italic")

    # -- panel 1: intensity vs an UPPER-BOUND ridge; only one side is decidable -----------------
    named = [r for r in rows if r["macs_per_byte"] is not None]
    order = sorted(named, key=lambda r: r["macs_per_byte"])
    ypos = range(len(order))
    colors = [PALETTE[2] if r["bound_by"] == "compute" else PALETTE[6] for r in order]
    ax1.barh(list(ypos), [r["macs_per_byte"] for r in order], color=colors, height=0.6,
             hatch=["" if r["bound_by"] == "compute" else "//" for r in order],
             edgecolor="#8a7d68")
    # Labelled via the legend rather than placed by hand: hand-placed text here collided with
    # whichever bar happened to be longest, and the ordering is data-dependent.
    ax1.axvline(ridge, color=PALETTE[1], lw=1.8,
                label=f"ridge $\\leq$ {ridge:.2f} MACs/B (UPPER bound)")
    # Below the axes, like panel 3: inside the axes it covered either the longest bar or the
    # zero-bar annotation, depending on the data.
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), framealpha=0.92, fontsize=8)
    # A zero-length bar is indistinguishable from a missing one, and this one is a FINDING: the
    # canonical "tinyllama" buffer routes no MACs at all (19 contractions, every one off-unit).
    for y, row in zip(ypos, order):
        if not row["macs_per_byte"]:
            ax1.annotate("0 routed MACs — all-host placement", xy=(0, y), xytext=(8, 0),
                         textcoords="offset points", va="center", fontsize=8,
                         color="#8a6d3b", style="italic")
    ax1.set_yticks(list(ypos)); ax1.set_yticklabels([r["model"] for r in order])
    ax1.set_xlabel("arithmetic intensity (MACs / byte)")
    ax1.set_title("Only the compute side is provable\nhatched = UNKNOWN, not memory-bound")
    ax1.set_xlim(0, max(ridge, max(r["macs_per_byte"] for r in order)) * 1.35)

    # -- panel 2: where the work actually sits -------------------------------------------------
    labels = [r["model"] for r in rows]
    idx = range(len(rows))
    ax2.bar([i - 0.2 for i in idx], [r["on_unit"] for r in rows], width=0.4,
            color=PALETTE[0], label="contractions ON unit")
    ax2.bar([i + 0.2 for i in idx], [r["off_unit"] for r in rows], width=0.4,
            color=PALETTE[1], hatch="//", edgecolor="#8a7d68", label="OFF unit")
    ax2.set_xticks(list(idx)); ax2.set_xticklabels(labels, rotation=12)
    ax2.set_ylabel("contraction count")
    twin = ax2.twinx()
    twin.plot(list(idx), [r["host_regions"] for r in rows], color=PALETTE[3], marker="o",
              lw=1.6, label="host-lane regions")
    twin.set_ylabel("host-lane regions"); twin.set_yscale("log"); twin.grid(False)
    ax2.set_title("Offload completeness\nhost regions on a log axis")
    handles, texts = ax2.get_legend_handles_labels()
    h2, t2 = twin.get_legend_handles_labels()
    ax2.legend(handles + h2, texts + t2, loc="upper center", framealpha=0.9,
               bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)

    commit = (bal.get("provenance", {}).get("merlin", {}) or {}).get("commit", "")[:7]
    fig.suptitle(f"Gemmini phase-2 instruments, measured off the FPGA "
                 f"(engine={bal['engine']}, package={bal['compiler_package']}, "
                 f"merlin@{commit})", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    print(f"wrote {out_stem}.pdf / .png")


def main() -> int:
    data = collect()
    out = Path(__file__).resolve().parent
    (out / "gemmini_phase2_instruments_20260909.json").write_text(
        json.dumps(data, indent=1), encoding="utf-8")
    render(data, out / "gemmini_phase2_instruments")
    for row in data["workloads"]:
        print(f"  {row['model']:11s} MACs/B={row['macs_per_byte']:8.3f} -> {row['bound_by']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
