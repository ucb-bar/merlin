#!/usr/bin/env python3
"""Agentic trajectory v3 — Anthropic-style: activity-phase BACKGROUND bands + cumulative-token lines,
with test-pass MILESTONES and round boundaries as vertical dotted markers (not a second y-axis).

4 panels (one per backend, top->bottom): raw C++ / C+++scaffold / Python-Merlin / Merlin+CIRCT.
  x            = active wall-clock time (round transcript durations concatenated; own scale per arm)
  background   = what the agent is DOING at each moment — thinking / reading / writing / bash / tool-run
                 (spike·verilator·CIRCT) — coloured bands, like the Anthropic turn-type figure
  lines        = CUMULATIVE tokens by type (cache-read / input / output) on a log axis — every type visible
  gold dotted  = test-pass MILESTONE (the suite count rose) — labelled with the new count + cum $ + tokens
  faint dotted = round boundary

Per-event wall stamps don't exist, so within a round the activity sequence is laid evenly across the
round's measured duration (honest given the data). Authoritative tokens/cost/duration come from each
round's transcript result event.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path("/path/to/oscar-merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import (use_merlin_style, style_ax, title, suptitle, emph,
                              BG, INK, GOLD, BLUE, NAVY, SLATE, MAUVE, SAGE)

CB = REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0"
OUT = CB / "reports" / "plots"
ARMS = [
    ("raw_baseline/rb_abc11",            "raw C++ (from scratch)"),
    ("cpp_merlininfra/rbinfra_abc11",    "C++ + Merlin scaffold"),
    ("merlin_assisted/merlin_abc9",      "Merlin — Python tooling"),
    ("merlin_assisted/merlincirct_abc9", "Merlin + CIRCT hints"),
]
# activity-phase background palette (pale tints — colour says WHAT, line carries the metric)
ACT = {
    "think": ("#3b3b5c", "thinking"),
    "read":  ("#8B93A6", "reading"),
    "write": ("#7D886C", "writing code"),
    "bash":  ("#b8a48f", "bash / shell"),
    "tool":  ("#815E5E", "tool run (spike·verilator·CIRCT)"),
}
ACT_ALPHA = 0.30
# cumulative-token line identity
C_CACHE, C_INPUT, C_OUTPUT = SLATE, SAGE, NAVY


def _round_records(run: Path):
    recs = []
    for tp in sorted(glob.glob(str(run / "rounds" / "round_*.transcript.jsonl"))):
        result = None
        acts = []          # ordered activity categories through the round
        nver = ncir = 0
        for l in open(tp):
            try:
                e = json.loads(l)
            except Exception:
                continue
            t = e.get("type")
            if t == "result":
                result = e
            elif t == "assistant":
                for c in (e.get("message", {}).get("content") or []):
                    if not isinstance(c, dict):
                        continue
                    typ = c.get("type")
                    if typ == "thinking":
                        acts.append("think")
                    elif typ == "tool_use":
                        nm = (c.get("name") or "").lower()
                        s = json.dumps(c.get("input", {}))
                        if nm == "read":
                            acts.append("read")
                        elif nm in ("edit", "write", "notebookedit"):
                            acts.append("write")
                        elif nm == "bash":
                            is_ver = ("--sim verilator" in s) or ('"verilator"' in s)
                            is_cir = ("circt" in run.as_posix() and
                                      ("rtl_check" in s or "gen_isa" in s or "rtl_facts" in s or "facts.json" in s))
                            if is_ver:
                                acts.append("tool"); nver += 1
                            elif is_cir:
                                acts.append("tool"); ncir += 1
                            else:
                                acts.append("bash")
                        else:
                            acts.append("bash")
        if result is None:
            continue
        u = result.get("usage", {}) or {}
        recs.append(dict(
            dur=(result.get("duration_ms", 0) or 0) / 1000.0,
            cost=result.get("total_cost_usd", 0) or 0.0,
            tin=int(u.get("input_tokens", 0) or 0),
            tcache=int(u.get("cache_read_input_tokens", 0) or 0) + int(u.get("cache_creation_input_tokens", 0) or 0),
            tout=int(u.get("output_tokens", 0) or 0),
            acts=acts or ["bash"], nver=nver, ncir=ncir))
    return recs


def _passed_per_round(run: Path, nrounds: int):
    verds = sorted(glob.glob(str(run / "qa_history" / "verdict_round_*.json")))
    seq = []
    for v in verds:
        try:
            seq.append(int(json.loads(Path(v).read_text()).get("n_passed") or 0))
        except Exception:
            pass
    l3 = None
    vc = run / "verilator_checkpoints.json"
    if vc.is_file():
        try:
            atts = (json.loads(vc.read_text()).get("attempts") or [])
            if atts:
                l3 = max(int(a.get("n_passed") or 0) for a in atts)
        except Exception:
            pass
    out = []
    for k in range(nrounds):
        if seq:
            idx = min(int(k * len(seq) / max(1, nrounds)), len(seq) - 1)
            out.append(max(seq[: idx + 1]))
        else:
            out.append(0)
    if l3 is not None and out:
        out[-1] = max(out[-1], l3)
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1])
    return out, l3


def main():
    use_merlin_style()
    OUT.mkdir(parents=True, exist_ok=True)
    HALO = [pe.withStroke(linewidth=4.0, foreground=BG)]
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(20, 17))

    for ax, (rel, label) in zip(axes, ARMS):
        run = CB / "runs" / rel
        recs = _round_records(run)
        if not recs:
            ax.text(0.5, 0.5, f"{label}: no data", ha="center"); continue
        n = len(recs)
        starts = np.concatenate([[0.0], np.cumsum([r["dur"] for r in recs])])
        total = float(starts[-1])
        passed, l3 = _passed_per_round(run, n)
        style_ax(ax, grid=None)

        # ---- BACKGROUND: activity-phase bands (sequence laid evenly across each round's duration) ----
        for k, r in enumerate(recs):
            x0, x1 = starts[k], starts[k + 1]
            acts = r["acts"]
            m = len(acts)
            edges = np.linspace(x0, x1, m + 1)
            # merge consecutive same-category runs into single bands (clean, Anthropic-style)
            j = 0
            while j < m:
                jj = j
                while jj + 1 < m and acts[jj + 1] == acts[j]:
                    jj += 1
                col = ACT[acts[j]][0]
                ax.axvspan(edges[j], edges[jj + 1], color=col, alpha=ACT_ALPHA, lw=0, zorder=0)
                j = jj + 1

        # ---- round boundaries: faint dotted ----
        for k in range(1, n):
            ax.axvline(starts[k], color=INK, ls=(0, (1, 2)), lw=1.0, alpha=0.25, zorder=2)

        # ---- cumulative tokens by type (log y), lines on top ----
        xs = [0.0]; cca = [0.0]; cin = [0.0]; cou = [0.0]
        ac = ai = ao = 0
        for k, r in enumerate(recs):
            ac += r["tcache"]; ai += r["tin"]; ao += r["tout"]
            xs.append(starts[k + 1]); cca.append(ac); cin.append(ai); cou.append(ao)
        xs = np.array(xs)
        cca = np.array(cca); cin = np.array(cin); cou = np.array(cou)
        tot = cca + cin + cou
        for arr, col in ((cca, C_CACHE), (cin, C_INPUT), (cou, C_OUTPUT)):
            ax.step(xs, np.clip(arr, 1, None), where="post", color=col, lw=3.4, zorder=6,
                    path_effects=[pe.withStroke(linewidth=5.0, foreground=BG)])
        ax.set_yscale("log")
        ax.set_ylim(1e3, tot[-1] * 6)

        # ---- test-pass MILESTONES: gold dotted verticals + label (count, cum $, cum tokens) ----
        cum_cost = np.cumsum([r["cost"] for r in recs])
        prev = 0
        for k in range(n):
            if passed[k] > prev:
                xm = starts[k + 1]
                ax.axvline(xm, color=GOLD, ls=(0, (4, 3)), lw=2.6, alpha=0.95, zorder=5)
                txt = f"{passed[k]}/20\n${cum_cost[k]:.0f} · {tot[k+1]/1e6:.0f}M"
                ax.annotate(txt, (xm, ax.get_ylim()[1]), xytext=(-8, -8),
                            textcoords="offset points", ha="right", va="top",
                            fontsize=13, fontweight="bold", color="#7a6a40", zorder=8,
                            path_effects=HALO)
                prev = passed[k]

        # ---- cosmetics ----
        ax.set_xlim(0, total * 1.01)
        ax.set_ylabel("cum. tokens (log)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        title(ax, label, fs=21, pad=10)
        ax.text(total * 0.012, tot[-1] * 3.2,
                f"{total/60:.0f} min active   ·   ${cum_cost[-1]:.0f}   ·   {tot[-1]/1e6:.0f}M tokens   ·   "
                f"{n} rounds   ·   final {passed[-1]}/20",
                fontsize=14.5, color=INK, va="top", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d9cfc0", lw=1.0))
        if ax is axes[-1]:
            ax.set_xlabel("active wall-clock time (s)   —   own scale per arm", fontsize=16)

    # shared legend: activity bands + token lines + markers
    handles = [Patch(fc=ACT[k][0], alpha=ACT_ALPHA, label=ACT[k][1]) for k in ("think", "read", "write", "bash", "tool")]
    handles += [
        Line2D([0], [0], color=C_CACHE, lw=3.4, label="cum. cache-read"),
        Line2D([0], [0], color=C_INPUT, lw=3.4, label="cum. input"),
        Line2D([0], [0], color=C_OUTPUT, lw=3.4, label="cum. output"),
        Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="test-pass milestone"),
        Line2D([0], [0], color=INK, lw=1.0, ls=(0, (1, 2)), label="round boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=13.5,
               frameon=True, facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.985))
    suptitle(fig, "Agentic authoring trajectory — activity, tokens & milestones over active time", y=1.0, fs=25)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = OUT / "fig_trajectory_v2.png"
    svg = OUT / "fig_trajectory_v2.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"wrote {png}\nwrote {svg}")


if __name__ == "__main__":
    raise SystemExit(main())
