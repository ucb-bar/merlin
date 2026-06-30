"""LAYER 4 — FIGURES: paper-styled PNGs driven by the artifact's ingested data.

REUSES the palette + ``card``/``callout`` helpers from ``scripts/plot_paper_style.py`` (imported as
a module) so the figures match the committed paper aesthetic, but drives them with THIS artifact's
data (not the hardcoded paths) and saves into the ``compare_<ts>/`` dir. Three views:
  - fig 1 "all four incl. baseline" (absolute wall, log)
  - fig 2 "zoomed speedup contest" (drop baseline; speedup vs baseline)
  - fig 3 "perf + structural util" (wall bars + vfmacc-form / accumulator-resident annotation)

Degrades gracefully: a model with no baseline cell is skipped from the speedup view; gemm-only specs
get the gemm bar instead. matplotlib is optional — absence is reported, not fatal.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .empirical import Measurement


def _load_style():
    """Import merlin.plotting.plot_paper_style for its palette + helpers (no side effects on import)."""
    try:
        from merlin.plotting import plot_paper_style as mod
        return mod
    except Exception:
        return None


def _have_mpl() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


_DEFAULT_PALETTE = {
    "INK": "#2b2b2b", "SALMON": "#cf8b7d", "SAGE": "#9bb08a", "STEEL": "#6f93b0",
    "GOLD": "#e7c25c", "CREAM": "#f5f1e6", "CARD_EC": "#33312b", "V3": "#b8742a",
}


def _palette(style) -> dict:
    pal = dict(_DEFAULT_PALETTE)
    if style is not None:
        for k in pal:
            pal[k] = getattr(style, k, pal[k])
    return pal


def _config_color(name: str, kind: str, pal: dict) -> str:
    if kind == "baseline":
        return pal["SALMON"]
    if name == "xnnpack":
        return pal["STEEL"]
    if name == "openblas":
        return pal["SAGE"]
    return pal["V3"]   # ours


def render(spec, measurements: dict[tuple[str, str], Measurement], ccas: dict[str, Any],
           out_dir: Path) -> list[str]:
    """Render the three views into out_dir. Returns the list of written PNG basenames."""
    out_dir = Path(out_dir)
    if not _have_mpl():
        (out_dir / "FIGURES_SKIPPED.txt").write_text(
            "matplotlib not installed; figures skipped (install merlin[kernels-plots]).\n")
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    style = _load_style()
    pal = _palette(style)
    written: list[str] = []

    model_wls = [w for w in spec.workloads if w.kind == "model"]
    gemm_wls = [w for w in spec.workloads if w.kind == "gemm"]
    wls = model_wls + gemm_wls
    configs = list(spec.configs)

    def val_s(cfg_name, wl_name):
        m = measurements.get((cfg_name, wl_name))
        if m is None or m.status != "measured" or not m.value:
            return None
        return m.value / 1e9   # ns -> s

    # ---- FIG 1: all configs incl. baseline, absolute wall (log) ----
    if wls:
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(wls) + 4), 5.0))
        ax.set_facecolor(pal["CREAM"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        x = np.arange(len(wls))
        nser = len(configs)
        bw = 0.8 / max(nser, 1)
        any_bar = False
        for i, cfg in enumerate(configs):
            ys = [val_s(cfg.name, w.name) or 0 for w in wls]
            if any(y > 0 for y in ys):
                any_bar = True
            ax.bar(x + (i - (nser - 1) / 2) * bw, ys, bw * 0.9,
                   color=_config_color(cfg.name, cfg.kind, pal),
                   edgecolor=pal["CARD_EC"], linewidth=0.8, label=cfg.name, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([w.name for w in wls], fontsize=10)
        ax.set_ylabel("wall (s, log) — lower = faster")
        ax.set_title("(1) All configs incl. baseline", loc="left", color=pal["INK"], pad=8)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", ls=":", alpha=0.35)
        fig.tight_layout()
        if any_bar:
            fig.savefig(out_dir / "fig1_all_configs.png", dpi=150, bbox_inches="tight")
            written.append("fig1_all_configs.png")
        plt.close(fig)

    # ---- FIG 2: zoomed speedup contest (drop baseline; speedup vs baseline) ----
    contenders = [c for c in configs if c.kind != "baseline"]
    speed_wls = [w for w in wls if val_s("baseline", w.name)]
    if contenders and speed_wls:
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(speed_wls) + 4), 5.0))
        ax.set_facecolor(pal["CREAM"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        x = np.arange(len(speed_wls))
        nser = len(contenders)
        bw = 0.8 / max(nser, 1)
        for i, cfg in enumerate(contenders):
            ys = []
            for w in speed_wls:
                base = val_s("baseline", w.name)
                cur = val_s(cfg.name, w.name)
                ys.append(base / cur if (base and cur) else np.nan)
            ax.bar(x + (i - (nser - 1) / 2) * bw, ys, bw * 0.9,
                   color=_config_color(cfg.name, cfg.kind, pal),
                   edgecolor=pal["CARD_EC"], linewidth=0.8, label=cfg.name, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([w.name for w in speed_wls], fontsize=10)
        ax.set_ylabel("speedup vs baseline (x) — higher = faster")
        ax.set_title("(2) Zoomed contest: ours vs experts (no baseline)",
                     loc="left", color=pal["INK"], pad=8)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / "fig2_speedup_contest.png", dpi=150, bbox_inches="tight")
        written.append("fig2_speedup_contest.png")
        plt.close(fig)

    # ---- FIG 3: perf + structural util (wall bars annotated with vfmacc form / acc-resident) ----
    if wls:
        rep = (model_wls or wls)[0]
        fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(configs) + 3), 5.0))
        ax.set_facecolor(pal["CREAM"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        names = [c.name for c in configs]
        ys = [val_s(c.name, rep.name) or 0 for c in configs]
        colors = [_config_color(c.name, c.kind, pal) for c in configs]
        bars = ax.bar(range(len(configs)), ys, color=colors,
                      edgecolor=pal["CARD_EC"], linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"wall on {rep.name} (s) — lower = faster")
        ax.set_title("(3) Perf + structural form (vfmacc / accumulator)",
                     loc="left", color=pal["INK"], pad=8)
        for i, cfg in enumerate(configs):
            cca = ccas.get(cfg.name)
            tag = "scalar/baseline"
            if cca is not None:
                vf = cca.provenance.get("fma_loop_vfmacc_vf")
                vv = cca.provenance.get("fma_loop_vfmacc_vv")
                form = ".vf" if vf else (".vv" if vv else "?")
                resid = "resident" if cca.compute.accumulator_resident else "spilled"
                tag = f"{form}/{resid}"
            ax.text(i, (ys[i] if ys[i] else 0), tag, ha="center", va="bottom",
                    fontsize=7.5, color=pal["INK"])
        ax.grid(True, axis="y", ls=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_perf_util.png", dpi=150, bbox_inches="tight")
        written.append("fig3_perf_util.png")
        plt.close(fig)

    return written
