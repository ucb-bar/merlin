#!/usr/bin/env python3
"""Agentic authoring trajectory — polished FLAVORS (each its own PNG), house style, 300 dpi.

Every flavor now carries the full line set the user asked for: fine per-type cumulative token lines
(cache-read / input / output) + a TOTAL-tokens line + a cumulative-SPEND ($) line, on distinct
palette tones. Test-pass MILESTONES come from the real full-suite progression (e.g. 13→17→20) in the
selfcheck log and are drawn as gold dotted verticals with the count + cum $ + cum tokens; ROUNDS are
marked with faint dotted verticals + rN labels. Tool waits (verilator/CIRCT) are weighted by real
duration so they show as wide bands and the token curves plateau across them.

4 panels, top->bottom: raw C++ / C+++scaffold / Python-Merlin / Merlin+CIRCT; own time scale each.
Per-message wall stamps don't exist; within a round messages are laid by weighted time across the
round's measured duration; round totals are authoritative; milestones map selfcheck wall_offset onto
the active-wall axis proportionally (endpoints align at convergence).
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

REPO = Path("/path/to/merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import (use_merlin_style, style_ax, title, suptitle,
                              BG, INK, GOLD, BLUE, NAVY, SLATE, MAUVE, SAGE)

CB = REPO / "merlin" / "experiments" / "capsule_bench" / "targets" / "gemmini"
OUT = CB / "reports" / "plots"
ARMS = [
    ("raw_baseline/rb_abc11",            "raw C++ (from scratch)"),
    ("cpp_merlininfra/rbinfra_abc11",    "C++ + Merlin scaffold"),
    ("merlin_assisted/merlin_abc9",      "Merlin — Python tooling"),
    ("merlin_assisted/merlincirct_abc9", "Merlin + CIRCT hints"),
]
ACTS = ["think", "read", "write", "bash", "tool"]
ACT_COL = {"think": "#4C4C73", "read": "#6E93B0", "write": "#C2974A", "bash": "#9DB682", "tool": "#B06A6A"}
ACT_LAB = {"think": "thinking", "read": "reading", "write": "writing code",
           "bash": "bash / shell", "tool": "tool wait (verilator·CIRCT)"}
WEIGHT = {"think": 1.0, "read": 1.0, "write": 1.4, "bash": 1.2, "tool": 28.0}
# Opus list $/Mtok — used ONLY as the per-message SHAPE for distributing each round's authoritative
# total_cost_usd across messages (each round is rescaled to its billed cost, so the curve agrees with billing).
PR_IN, PR_OUT, PR_CR, PR_CW = 15.0, 75.0, 1.5, 18.75
# token/spend line tones (palette-following, distinct intensities)
L_CACHE, L_INPUT, L_OUTPUT, L_TOTAL, L_SPEND = SLATE, SAGE, NAVY, INK, "#4B3F6E"
GOLDLAB = "#7a6a40"
HALO = [pe.withStroke(linewidth=4.0, foreground=BG)]
LHALO = [pe.withStroke(linewidth=5.2, foreground=BG)]


# ----------------------------------------------------------------- data
def load_arm(rel):
    run = CB / "runs" / rel
    is_circt = "circt" in rel
    rounds = []
    for tp in sorted(glob.glob(str(run / "rounds" / "round_*.transcript.jsonl"))):
        result = None; msgs = []; nver = ncir = 0
        for l in open(tp):
            try:
                e = json.loads(l)
            except Exception:
                continue
            if e.get("type") == "result":
                result = e
            elif e.get("type") == "assistant":
                m = e.get("message", {}); u = m.get("usage", {}) or {}
                din = int(u.get("input_tokens", 0) or 0)
                dcr = int(u.get("cache_read_input_tokens", 0) or 0)
                dcw = int(u.get("cache_creation_input_tokens", 0) or 0)
                dou = int(u.get("output_tokens", 0) or 0)
                cat = "bash"
                has_think = any(isinstance(c, dict) and c.get("type") == "thinking" for c in (m.get("content") or []))
                tu = next((c for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "tool_use"), None)
                if tu is not None:
                    nm = (tu.get("name") or "").lower(); s = json.dumps(tu.get("input", {}))
                    if nm == "read":
                        cat = "read"
                    elif nm in ("edit", "write", "notebookedit"):
                        cat = "write"
                    elif nm == "bash":
                        if ("--sim verilator" in s) or ('"verilator"' in s):
                            cat = "tool"; nver += 1
                        elif is_circt and ("rtl_check" in s or "gen_isa" in s or "rtl_facts" in s or "facts.json" in s):
                            cat = "tool"; ncir += 1
                        else:
                            cat = "bash"
                    else:
                        cat = "bash"
                elif has_think:
                    cat = "think"
                msgs.append((cat, din, dcr, dcw, dou, WEIGHT[cat]))
        if result is None:
            continue
        u = result.get("usage", {}) or {}
        rounds.append(dict(
            dur=(result.get("duration_ms", 0) or 0) / 1000.0,
            cost=result.get("total_cost_usd", 0) or 0.0,
            T_in=int(u.get("input_tokens", 0) or 0),
            T_ca=int(u.get("cache_read_input_tokens", 0) or 0) + int(u.get("cache_creation_input_tokens", 0) or 0),
            T_ou=int(u.get("output_tokens", 0) or 0),
            msgs=msgs or [("bash", 0, 0, 0, 0, 1.0)], nver=nver, ncir=ncir))
    n = len(rounds)
    starts = np.concatenate([[0.0], np.cumsum([r["dur"] for r in rounds])]) if n else np.array([0.0])
    total = float(starts[-1])
    t = [0.0]; cca = [0.0]; cin = [0.0]; cou = [0.0]; csp = [0.0]; cat_t = []
    acc_ca = acc_in = acc_ou = acc_sp = 0.0
    for ri, r in enumerate(rounds):
        ms = r["msgs"]; m = len(ms)
        s_in = sum(x[1] for x in ms) or 1
        s_ca = sum(x[2] + x[3] for x in ms) or 1            # cache total (read + creation)
        s_ou = sum(x[4] for x in ms) or 1; wsum = sum(x[5] for x in ms) or 1
        # per-message COST SHAPE = token×list-price; rescaled so the round sums to the BILLED total_cost_usd
        costw = [(x[1] * PR_IN + x[2] * PR_CR + x[3] * PR_CW + x[4] * PR_OUT) for x in ms]
        scost = sum(costw) or 1
        edges = starts[ri] + (starts[ri + 1] - starts[ri]) * np.concatenate([[0], np.cumsum([x[5] for x in ms]) / wsum])
        for j, (cat, din, dcr, dcw, dou, w) in enumerate(ms):
            acc_in += din / s_in * r["T_in"]
            acc_ca += (dcr + dcw) / s_ca * r["T_ca"]
            acc_ou += dou / s_ou * r["T_ou"]
            acc_sp += costw[j] / scost * r["cost"]          # granular spend, sums to the billed round cost
            t.append(edges[j + 1]); cin.append(acc_in); cca.append(acc_ca); cou.append(acc_ou); csp.append(acc_sp)
            cat_t.append((edges[j], edges[j + 1], cat))
    arr = lambda a: np.array(a, float)
    d = dict(n=n, starts=starts, total=total, rounds=rounds, t=arr(t),
             cca=arr(cca), cin=arr(cin), cou=arr(cou), csp=arr(csp), cat_t=cat_t,
             cum_cost=(np.cumsum([r["cost"] for r in rounds]) if n else np.array([0.0])))
    d["tot"] = d["cca"] + d["cin"] + d["cou"]
    d["tot_tok"] = float(d["tot"][-1]) if n else 0.0
    d["passed"] = _passed_round(run, n)
    d["milestones"] = _fine_milestones(run, total)
    # ---- convert ALL x-axis quantities to MINUTES ----
    S = 60.0
    d["t"] = d["t"] / S
    d["starts"] = d["starts"] / S
    d["total"] = total / S
    d["cat_t"] = [(a / S, b / S, c) for a, b, c in d["cat_t"]]
    d["milestones"] = [(x / S, c) for x, c in d["milestones"]]
    return d


def _passed_round(run, n):
    seq = []
    for v in sorted(glob.glob(str(run / "qa_history" / "verdict_round_*.json"))):
        try:
            seq.append(int(json.loads(Path(v).read_text()).get("n_passed") or 0))
        except Exception:
            pass
    out = [max(seq[: min(int(k * len(seq) / max(1, n)), len(seq) - 1) + 1]) if seq else 0 for k in range(n)]
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1])
    return out


def _fine_milestones(run, total):
    p = run / "selfcheck_log.jsonl"
    if not p.is_file() or total <= 0:
        return []
    rows = [json.loads(l) for l in open(p) if l.strip()]
    offs = [r.get("wall_offset_s") for r in rows if r.get("wall_offset_s")]
    if not offs:
        return []
    mo = max(offs) or 1
    best = 0; out = []
    for r in rows:
        if str(r.get("capsules")) != "all":
            continue
        nc = r.get("n_capsules") or 0; np_ = r.get("n_passed") or 0
        if nc < 20:
            continue
        if np_ > best:
            best = np_
            out.append((r.get("wall_offset_s", 0) / mo * total, np_))
    return out


# ----------------------------------------------------------------- shared draw
def _share_grid(d, ngrid=420, win=23):
    g = np.linspace(0, d["total"], ngrid) if d["total"] else np.array([0.0, 1.0])
    raw = {a: np.zeros(len(g)) for a in ACTS}
    for (x0, x1, cat) in d["cat_t"]:
        raw[cat][(g >= x0) & (g < x1)] += 1.0
    k = np.hanning(win); k /= k.sum()
    for a in ACTS:
        raw[a] = np.convolve(np.pad(raw[a], win // 2, mode="edge"), k, mode="valid")[:len(g)]
    tot = sum(raw[a] for a in ACTS); tot[tot == 0] = 1
    return g, {a: raw[a] / tot for a in ACTS}


def _coarse_bands(ax, d, alpha=0.30, ngrid=240, win=35):
    g, sh = _share_grid(d, ngrid=ngrid, win=win)
    dom = np.array([max(ACTS, key=lambda a: sh[a][i]) for i in range(len(g))])
    i = 0
    while i < len(g):
        j = i
        while j + 1 < len(g) and dom[j + 1] == dom[i]:
            j += 1
        ax.axvspan(g[i], g[min(j + 1, len(g) - 1)], color=ACT_COL[dom[i]], alpha=alpha, lw=0, zorder=0)
        i = j + 1


RND_C = "#6f675c"   # clearly-defined round divider colour


def _rounds(ax, d, topax=None, fs=12.5, labels=True):
    # round dividers BEHIND the curves; boxed rN labels on the TOP axes so they're never covered
    tx = topax or ax
    for k in range(1, d["n"]):
        ax.axvline(d["starts"][k], color=RND_C, ls=(0, (3, 2)), lw=1.4, alpha=0.55, zorder=2)
    if not labels:
        return
    # round labels: collapse runs of closely-spaced rounds into a single range chip, e.g. "r8 → r14",
    # so dense back-to-back rounds (the short ones near the end) never overprint each other.
    span = max(d["total"], 1)
    xm = [(d["starts"][k] + d["starts"][k + 1]) / 2 for k in range(d["n"])]
    min_sep = 0.05 * span
    i = 0
    while i < d["n"]:
        j = i
        while j + 1 < d["n"] and xm[j + 1] - xm[j] < min_sep:
            j += 1
        lab = f"r{i}" if i == j else f"r{i} → r{j}"
        xc = xm[i] if i == j else (xm[i] + xm[j]) / 2
        tx.text(xc, 0.022, lab, transform=tx.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color="#4f483f", zorder=20,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#cbbfa8", lw=0.9, alpha=0.96))
        i = j + 1


def _milestones(ax, d, topax=None, fs=12.0, labels=True):
    ms = d["milestones"]
    if not ms:
        return
    tx = topax or ax   # draw on the TOP axes so the gold line, marker and label sit IN FRONT of the curves
    for x, _ in ms:
        tx.axvline(x, color=GOLD, ls=(0, (5, 2)), lw=3.2, alpha=1.0, zorder=18)
        tx.plot([x], [1.0], marker="v", ms=13, color=GOLD, mec=INK, mew=1.2,
                transform=tx.get_xaxis_transform(), clip_on=False, zorder=21)
    if not labels:        # keep the gold marker + line, drop the floating $/token boxes
        return
    span = max(d["total"], 1)
    # stack labels onto distinct vertical levels: a label drops to the next free level only when a
    # previously-placed label on that level is within min_gap horizontally → no two ever overlap.
    levels = [0.96, 0.74, 0.52, 0.30]
    min_gap = 0.20 * span
    last_x = [-1e18] * len(levels)
    for x, c in sorted(ms):
        li = next((j for j in range(len(levels)) if x - last_x[j] >= min_gap), len(levels) - 1)
        last_x[li] = x
        # cost + tokens read straight off the displayed cumulative curves at the milestone x (consistent)
        tok = d["tot"][min(np.searchsorted(d["t"], x), len(d["tot"]) - 1)]
        cost = float(np.interp(x, d["t"], d["csp"]))
        ha, dx = ("right", -9) if x > 0.18 * span else ("left", 9)
        tx.annotate(f"{c}/20\n${cost:.1f} · {tok/1e6:.1f}M", (x, levels[li]),
                    xycoords=("data", "axes fraction"), xytext=(dx, -4), textcoords="offset points",
                    ha=ha, va="top", fontsize=fs, fontweight="bold", color=GOLDLAB, zorder=22,
                    bbox=dict(boxstyle="round,pad=0.30", fc="#fdf6e6", ec=GOLD, lw=1.2, alpha=1.0))


def _chip(ax, d, compact=False, y=1.045, fs=12.5):
    # float the summary ABOVE the panel, right-aligned (opposite the title) — never collides with data
    fin = d['passed'][-1] if d['passed'] else '?'
    if compact:
        txt = f"${d['cum_cost'][-1]:.0f} · {d['tot_tok']/1e6:.0f}M · {d['n']}r · {fin}/20 · {d['total']:.0f}min"
    else:
        txt = (f"{d['total']:.0f} min active   ·   ${d['cum_cost'][-1]:.0f}   ·   {d['tot_tok']/1e6:.0f}M tok   ·   "
               f"{d['n']} rounds   ·   final {fin}/20")
    ax.text(1.0, y, txt, transform=ax.transAxes, fontsize=fs, color=INK, va="bottom", ha="right",
            zorder=11, bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d9cfc0", lw=1.0))


def _token_lines(axT, d, log=True):
    """cache/input/output/total cumulative token lines on axT."""
    for arr, col, lw in ((d["cca"], L_CACHE, 3.2), (d["cin"], L_INPUT, 3.2),
                         (d["cou"], L_OUTPUT, 3.2), (d["tot"], L_TOTAL, 3.8)):
        axT.plot(d["t"], np.clip(arr, 1, None), color=col, lw=lw, zorder=8, path_effects=LHALO)
    if log:
        axT.set_yscale("log"); axT.set_ylim(1e3, d["tot_tok"] * 6)
    else:
        axT.set_ylim(0, d["tot_tok"] * 1.18 or 1)


def _spend_axis(ax, d, outward=68, fs=1.0):
    """second right axis with the cumulative-spend line ($)."""
    axS = ax.twinx()
    axS.spines["right"].set_position(("outward", outward))
    axS.spines["top"].set_visible(False)
    axS.spines["right"].set_color(L_SPEND)
    axS.plot(d["t"], d["csp"], color=L_SPEND, lw=3.6, ls=(0, (6, 2)), zorder=9, path_effects=LHALO)
    axS.set_ylim(0, max(d["csp"][-1] * 1.18, 1))
    axS.set_ylabel("cumulative spend ($)", color=L_SPEND, fontsize=14 * fs)
    axS.tick_params(colors=L_SPEND, labelsize=11 * fs)
    return axS


def _basics(ax, d, lab, last, fs=1.0):
    title(ax, lab, fs=20 * fs, pad=10); ax.set_xlim(0, d["total"] * 1.01); ax.tick_params(labelsize=13 * fs)
    if last:
        ax.set_xlabel("active wall-clock time (min)   —   own scale per arm", fontsize=16 * fs)


def _fig(h=17.5, top=0.928, bottom=0.085, hspace=0.42):
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(21, h))
    fig.subplots_adjust(left=0.055, right=0.88, top=top, bottom=bottom, hspace=hspace)
    return fig, axes


def _legend(fig, handles, sub, fname, sup_y=0.972, leg_y=0.012, fs=14, sup_fs=24, dpi=300):
    # single full-width row (spans the figure), bigger, BELOW the x-axis label
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=fs,
               frameon=True, facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, leg_y),
               columnspacing=1.3, handlelength=1.8, borderpad=0.7)
    if sub:
        suptitle(fig, sub, y=sup_y, fs=sup_fs)
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=dpi, facecolor=BG)
    fig.savefig(OUT / f"{fname}.svg", bbox_inches="tight", facecolor=BG)
    print(f"wrote {fname}.png")


def _line_handles(include_spend=True, rate=False):
    pre = "rate " if rate else "cum. "
    h = []
    if not rate:   # cache-read rate is ~flat/uninformative → shown only in the cumulative view
        h.append(Line2D([0], [0], color=L_CACHE, lw=3.2, label=pre + "cache-read"))
    h += [Line2D([0], [0], color=L_INPUT, lw=3.2, label=pre + "input"),
          Line2D([0], [0], color=L_OUTPUT, lw=3.2, label=pre + "output")]
    if not rate:   # 'total' is meaningful for cumulative (envelope); for rate it ≈ envelope, so omit
        h.append(Line2D([0], [0], color=L_TOTAL, lw=3.8, label=pre + "total"))
    if include_spend:
        h.append(Line2D([0], [0], color=L_SPEND, lw=3.6, ls=(0, (6, 2)), label="cumulative spend ($)"))
    h += [Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="test-pass milestone"),
          Line2D([0], [0], color=INK, lw=1.0, ls=(0, (1, 3)), label="round")]
    return h


# ----------------------------------------------------------------- flavors
def fl_A(DATA):
    """A — activity SHARE area (left) + all token lines (log) + spend axis."""
    fig, axes = _fig()
    for ax, (d, lab) in zip(axes, DATA):
        style_ax(ax, grid=None)
        g, sh = _share_grid(d)
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14)
        axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
        _token_lines(axT, d, log=True); axT.set_ylabel("cum. tokens (log)", fontsize=14); axT.tick_params(labelsize=11)
        _spend_axis(ax, d)
        _rounds(ax, d); _milestones(ax, d); _chip(ax, d); _basics(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.6, label=ACT_LAB[a]) for a in ACTS] + _line_handles()
    _legend(fig, h, "Authoring trajectory — activity share + cumulative tokens (by type) + spend", "fig_traj_A_share_spend")


def fl_B(DATA):
    """B — activity SHARE area (left) + token lines emphasised LINEAR (types + total) + spend axis."""
    fig, axes = _fig()
    for ax, (d, lab) in zip(axes, DATA):
        style_ax(ax, grid=None)
        g, sh = _share_grid(d)
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14)
        axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
        _token_lines(axT, d, log=True); axT.set_ylabel("cum. tokens by type (log)", fontsize=14); axT.tick_params(labelsize=11)
        _spend_axis(ax, d)
        _rounds(ax, d); _milestones(ax, d); _chip(ax, d); _basics(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.5, label=ACT_LAB[a]) for a in ACTS] + _line_handles()
    _legend(fig, h, "Authoring trajectory — activity share + cumulative tokens by type + spend", "fig_traj_B_types")


def fl_C(DATA):
    """C — activity bands (background) + token CONSUMPTION RATE lines + spend axis."""
    fig, axes = _fig()
    for ax, (d, lab) in zip(axes, DATA):
        style_ax(ax, grid=None); _coarse_bands(ax, d, alpha=0.30)
        ax.set_yticks([]); ax.set_ylim(0, 1)
        axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
        for arr, col, lw in ((d["cca"], L_CACHE, 3.2), (d["cin"], L_INPUT, 3.2),
                             (d["cou"], L_OUTPUT, 3.2)):  # no 'total' rate: it ≈ cache-read (cache dominates) and hides it
            if len(d["t"]) < 4:
                continue
            tdif = np.clip(np.diff(d["t"], prepend=d["t"][0]), 1e-6, None)
            rate = np.gradient(arr, np.cumsum(tdif))
            w = max(7, len(rate) // 22) | 1; kk = np.hanning(w); kk /= kk.sum()
            rate = np.convolve(np.pad(rate, w // 2, mode="edge"), kk, mode="valid")[:len(rate)]
            axT.plot(d["t"], np.clip(rate, 1, None), color=col, lw=lw, zorder=8, path_effects=LHALO)
        axT.set_yscale("log"); axT.set_ylabel("token rate (tok/min, log)", fontsize=14); axT.tick_params(labelsize=11)
        _spend_axis(ax, d)
        _rounds(ax, d); _milestones(ax, d); _chip(ax, d); _basics(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.30, label=ACT_LAB[a]) for a in ACTS] + _line_handles(rate=True)
    _legend(fig, h, "Authoring trajectory — activity bands + token consumption rate + spend", "fig_traj_C_rate")


def fl_F(DATA):
    """F — activity SHARE area (left) + token-type STACKED area + total line + spend axis."""
    fig, axes = _fig()
    for ax, (d, lab) in zip(axes, DATA):
        style_ax(ax, grid=None)
        g, sh = _share_grid(d)
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14)
        axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
        base = np.zeros_like(d["t"])
        for arr, col in ((d["cca"], L_CACHE), (d["cin"], L_INPUT), (d["cou"], L_OUTPUT)):
            axT.fill_between(d["t"], base, base + arr, color=col, alpha=0.32, lw=0, zorder=3)
            base = base + arr
        axT.plot(d["t"], d["tot"], color=L_TOTAL, lw=3.6, zorder=8, path_effects=LHALO)
        axT.set_ylim(0, d["tot_tok"] * 1.18 or 1); axT.set_ylabel("cum. tokens (stacked by type)", fontsize=14)
        axT.tick_params(labelsize=11)
        _spend_axis(ax, d)
        _rounds(ax, d); _milestones(ax, d); _chip(ax, d); _basics(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.55, label=ACT_LAB[a]) for a in ACTS] + [
        Patch(fc=L_CACHE, alpha=0.32, label="cache-read"), Patch(fc=L_INPUT, alpha=0.32, label="input"),
        Patch(fc=L_OUTPUT, alpha=0.32, label="output"), Line2D([0], [0], color=L_TOTAL, lw=3.6, label="total tokens"),
        Line2D([0], [0], color=L_SPEND, lw=3.6, ls=(0, (6, 2)), label="spend ($)"),
        Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="milestone")]
    _legend(fig, h, "Authoring trajectory — activity share + cumulative tokens (stacked) + spend", "fig_traj_F_dual")


def _rate(arr, t):
    if len(t) < 4:
        return np.zeros_like(arr)
    r = np.gradient(arr, t)
    w = max(7, len(r) // 22) | 1
    k = np.hanning(w); k /= k.sum()
    return np.convolve(np.pad(r, w // 2, mode="edge"), k, mode="valid")[:len(r)]


def _panel_rate(ax, d, lab, last, compact=False, fs=1.0, *,
                activity="stack", band_alpha=0.40, show_spend=True,
                milestone_labels=True, round_labels=True):
    """The A_rate panel. The de-clutter knobs change ONLY what is drawn, never the data:
       activity = "stack" (full-height background) | "strip" (slim bottom ribbon) | "none";
       show_spend toggles the 3rd ($) axis; milestone_labels / round_labels toggle the floating chips.
    Defaults reproduce the kept fig_traj_A_rate exactly."""
    style_ax(ax, grid=None)
    g, sh = _share_grid(d)
    if activity == "stack":
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=band_alpha, zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14 * fs)
    elif activity == "strip":
        frac = 0.16   # confine the activity bands to a slim ribbon along the panel floor
        ax.stackplot(g, *[sh[a] * frac for a in ACTS], colors=[ACT_COL[a] for a in ACTS],
                     alpha=min(band_alpha + 0.18, 0.7), zorder=1)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("")
        ax.text(0.004, frac + 0.015, "activity", transform=ax.get_yaxis_transform(),
                fontsize=10.5 * fs, color=INK, va="bottom", ha="left", alpha=0.8)
    else:  # "none"
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("")
    axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
    # only the input + output token RATES carry signal here; cache-read rate is ~flat and the
    # 'total' (black) line just traces their envelope → both dropped to de-clutter the panel.
    for arr, col, lw in ((d["cin"], L_INPUT, 3.4), (d["cou"], L_OUTPUT, 3.4)):
        axT.plot(d["t"], np.clip(_rate(arr, d["t"]), 1, None), color=col, lw=lw * fs, zorder=8, path_effects=LHALO)
    axT.set_yscale("log"); axT.set_ylabel("token rate (tok/min, log)", fontsize=14 * fs); axT.tick_params(labelsize=11 * fs)
    axS = _spend_axis(ax, d, outward=62, fs=fs) if show_spend else axT
    # draw rounds + milestones on the FRONT axes so labels/markers are IN FRONT of the curves
    _rounds(ax, d, topax=axS, fs=12.5 * fs, labels=round_labels)
    _milestones(ax, d, topax=axS, fs=10.5 * fs, labels=milestone_labels)
    _chip(ax, d, compact=compact, fs=12.5 * fs)
    _basics(ax, d, lab, last, fs=fs)


def fl_A_rate(DATA):
    """A_rate (stacked, tall) — 4 panels vertical. Presentation build: larger text + 400 dpi.
    THE KEPT VERSION — do not change. De-clutter experiments live in fl_A_rate_options()."""
    fig, axes = _fig(top=0.985, bottom=0.15)   # no suptitle → reclaim the top; bigger bottom for xlabel+legend
    for ax, (d, lab) in zip(axes, DATA):
        _panel_rate(ax, d, lab, ax is axes[-1], fs=1.5)
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + _line_handles(rate=True)
    _legend(fig, h, "", "fig_traj_A_rate", fs=20, leg_y=0.008, dpi=400)


def _rate_legend_handles(activity="stack", show_spend=True):
    """Legend handles for an A_rate OPTION figure — only what that option actually draws."""
    h = []
    if activity != "none":
        h += [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS]
    h += [Line2D([0], [0], color=L_INPUT, lw=3.2, label="rate input"),
          Line2D([0], [0], color=L_OUTPUT, lw=3.2, label="rate output")]
    if show_spend:
        h.append(Line2D([0], [0], color=L_SPEND, lw=3.6, ls=(0, (6, 2)), label="cumulative spend ($)"))
    h += [Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="test-pass milestone"),
          Line2D([0], [0], color=INK, lw=1.0, ls=(0, (1, 3)), label="round")]
    return h


def _rate_option(DATA, fname, subtitle, **panel_kw):
    """Render one de-clutter OPTION of fig_traj_A_rate to its own file (kept version untouched)."""
    fig, axes = _fig(top=0.945, bottom=0.15)        # leave a thin strip for the option subtitle
    for ax, (d, lab) in zip(axes, DATA):
        _panel_rate(ax, d, lab, ax is axes[-1], fs=1.5, **panel_kw)
    h = _rate_legend_handles(activity=panel_kw.get("activity", "stack"),
                             show_spend=panel_kw.get("show_spend", True))
    _legend(fig, h, subtitle, fname, fs=20, sup_fs=22, sup_y=0.992, leg_y=0.008, dpi=400)


def fl_A_rate_options(DATA):
    """Five ways to answer 'I like it but it's busy' — each keeps as much of the kept figure as
    possible and removes exactly one source of clutter. One PNG/SVG per option."""
    _rate_option(DATA, "fig_traj_A_rate_opt1_no_floating_chips",
                 "Option 1 — keep everything, drop the floating $/round chips (gold markers + dividers stay)",
                 milestone_labels=False, round_labels=False)
    _rate_option(DATA, "fig_traj_A_rate_opt2_activity_strip",
                 "Option 2 — activity bands shrunk to a slim ribbon so the rate lines breathe",
                 activity="strip")
    _rate_option(DATA, "fig_traj_A_rate_opt3_no_spend_axis",
                 "Option 3 — drop the 3rd ($) axis + spend line (totals already in the corner chip)",
                 show_spend=False)
    _rate_option(DATA, "fig_traj_A_rate_opt4_lines_only",
                 "Option 4 — token-rate lines only on a clean background (activity bands removed)",
                 activity="none")
    _rate_option(DATA, "fig_traj_A_rate_opt5_calm",
                 "Option 5 — calm: muted bands + no floating chips (1 + lighter background)",
                 band_alpha=0.22, milestone_labels=False, round_labels=False)


# ====================================================================================
#  CHOSEN STYLE = opt3 (no $ axis).  Four per-run figures + one combined w/ scale cue.
#  Per-run figures share the SAME panel geometry as the combined panels (left/right margins
#  + axes height), so a single run is instantly recognisable as "that panel".
# ====================================================================================
RUN_SLUGS = ["run1_raw_cpp", "run2_scaffold", "run3_python_merlin", "run4_circt"]
XLAB = "Time (min)"

# combined-figure geometry → reused so per-run panels come out the same shape
_CMB = dict(figh=17.5, top=0.975, bottom=0.165, hspace=0.55, left=0.055, right=0.88, n=4)


def _panel_axes_height_in():
    """Inches-tall of one panel in the combined layout (so singles can match it exactly)."""
    c = _CMB
    avail = (c["top"] - c["bottom"]) * c["figh"]
    return avail / (c["n"] + (c["n"] - 1) * c["hspace"])


def _scale_bar(ax, d, minutes=20, y=-0.185, fs=17, color=BLUE):
    """A fixed-duration ('{minutes} min') ruler drawn just BELOW the time axis (clear of the tick
    numbers). Every panel uses its OWN time scale, so the SAME 20 min renders at a different length
    per run → an at-a-glance cue that the scales differ (shorter bar ⇒ that run took longer).
    Left-aligned at t=0 so the four stacked bars read as a direct comparison."""
    tr = ax.get_xaxis_transform()            # x in data units, y in axes fraction
    x0, x1 = 0.0, float(minutes)
    ax.plot([x0, x1], [y, y], transform=tr, color=color, lw=4.6, zorder=27,
            solid_capstyle="butt", clip_on=False)
    for xx in (x0, x1):                       # end ticks
        ax.plot([xx, xx], [y - 0.032, y + 0.032], transform=tr, color=color, lw=4.6,
                zorder=27, clip_on=False)
    ax.text(x1 + 0.012 * d["total"], y, f"{minutes} min", transform=tr, ha="left", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=28, clip_on=False)


def _rate_single(d, lab, fname):
    """One run on its own, sized so the AXES match a single panel of the combined figure exactly
    (same width via identical fig-width + left/right margins; same height in inches)."""
    ah = _panel_axes_height_in()              # combined panel height (inches)
    figh = ah + 1.0 + 1.15                    # tight bottom: just ticks + xlabel + legend (no bar)
    fig, ax = plt.subplots(figsize=(21, figh))
    top = 1 - 1.0 / figh
    bottom = top - ah / figh
    fig.subplots_adjust(left=_CMB["left"], right=_CMB["right"], top=top, bottom=bottom)
    _panel_rate(ax, d, lab, last=True, fs=1.5, show_spend=False)
    ax.set_xlabel(XLAB, fontsize=16 * 1.5, labelpad=16)   # individuals carry NO 20-min bar
    h = _rate_legend_handles(activity="stack", show_spend=False)
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=17, frameon=True,
               facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, bottom - 0.22),
               columnspacing=1.3, handlelength=1.8, borderpad=0.7)
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=400, facecolor=BG)
    fig.savefig(OUT / f"{fname}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {fname}.png")


def fl_A_rate_arms(DATA):
    """Each run as its OWN high-quality figure (chosen opt3 style), matching the combined panel shape."""
    for (d, lab), slug in zip(DATA, RUN_SLUGS):
        _rate_single(d, lab, f"fig_traj_A_rate_{slug}")


# ------------------------------------------------------------------ progressive BUILD-UP of run 1
#  Layer-by-layer reveal of the raw-C++ panel to ease an audience into reading it. Every frame uses
#  the SAME fixed figure box + margins and is saved WITHOUT tight-cropping, so the frame never moves
#  between slides — only the newly-added layer appears. The AXES match the per-run figures exactly
#  (same width + matched panel height) so the build frames are the same size/shape as run1.


def _panel_rate_staged(ax, d, lab, last, fs, stage):
    """opt3 single-run panel drawn up to `stage`: 0 frame · 1 rounds · 2 milestones+chip ·
    3 token-rate lines · 4 activity-share background (complete)."""
    style_ax(ax, grid=None)
    ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14 * fs)
    if stage >= 4:                                   # activity-share background
        g, sh = _share_grid(d)
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
    axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
    rmax = 1.0
    for arr, col in ((d["cin"], L_INPUT), (d["cou"], L_OUTPUT)):
        r = np.clip(_rate(arr, d["t"]), 1, None)
        rmax = max(rmax, float(r.max()))
        if stage >= 3:                               # token-rate lines (input / output)
            axT.plot(d["t"], r, color=col, lw=3.0 * fs, zorder=8, path_effects=LHALO)
    axT.set_yscale("log"); axT.set_ylim(1, rmax * 1.35)      # fixed range so the frame is identical every stage
    axT.set_ylabel("token rate (tok/min, log)", fontsize=14 * fs); axT.tick_params(labelsize=11 * fs)
    if stage >= 1:                                   # round dividers + rN labels
        _rounds(ax, d, topax=axT, fs=12.5 * fs, labels=True)
    if stage >= 2:                                   # gold test-pass / spend milestones + summary chip
        _milestones(ax, d, topax=axT, fs=10.5 * fs, labels=True)
        _chip(ax, d, fs=12.5 * fs)
    _basics(ax, d, lab, last, fs=fs)
    ax.set_xlabel(XLAB, fontsize=16 * fs, labelpad=16)


def _buildup_handles(stage):
    h = []
    if stage >= 4:
        h += [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS]
    if stage >= 3:
        h += [Line2D([0], [0], color=L_INPUT, lw=3.2, label="rate input"),
              Line2D([0], [0], color=L_OUTPUT, lw=3.2, label="rate output")]
    if stage >= 2:
        h += [Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="test-pass milestone")]
    if stage >= 1:
        h += [Line2D([0], [0], color=INK, lw=1.0, ls=(0, (1, 3)), label="round")]
    return h


def fl_run1_buildup(DATA):
    """Five aligned build-up frames of run 1 (raw C++) for a layer-by-layer reveal — same axes
    size/shape as the per-run figures (fig_traj_A_rate_run1_*)."""
    d, lab = DATA[0]
    ah = _panel_axes_height_in()                          # match the per-run panel height exactly
    figh = ah + 1.0 + 1.6                                 # + top margin (title/chip) + bottom (xlabel/legend)
    top = 1 - 1.0 / figh
    bottom = top - ah / figh
    stages = [(0, "build0_frame"), (1, "build1_rounds"), (2, "build2_milestones"),
              (3, "build3_rates"), (4, "build4_full")]
    for stage, slug in stages:
        fig, ax = plt.subplots(figsize=(21, figh))        # same width + axes box as run1
        fig.subplots_adjust(left=_CMB["left"], right=_CMB["right"], top=top, bottom=bottom)
        _panel_rate_staged(ax, d, lab, last=True, fs=1.5, stage=stage)
        h = _buildup_handles(stage)
        if h:                                             # legend grows with the layers; centred → axes never move
            fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=15, frameon=True,
                       facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.02),
                       columnspacing=1.2, handlelength=1.7, borderpad=0.6)
        fname = f"fig_traj_A_rate_run1_{slug}"
        # NO bbox_inches="tight" → every frame is the SAME canvas with the axes in the SAME place
        fig.savefig(OUT / f"{fname}.png", dpi=400, facecolor=BG)
        fig.savefig(OUT / f"{fname}.svg", facecolor=BG)
        plt.close(fig)
        print(f"wrote {fname}.png")


def _panel_duration(ax, d, lab, grmax, fs=1.0):
    """A de-cluttered opt3 panel for the common-time-scale figure: activity stack + input/output
    rate lines + faint round dividers. NO milestones, NO chip, NO round labels, NO 20-min bar."""
    style_ax(ax, grid=None)
    g, sh = _share_grid(d)
    ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("activity share", fontsize=14 * fs)
    ax.tick_params(axis="y", labelsize=12 * fs)
    ax.set_xlim(0, d["total"]); ax.set_xticklabels([])
    axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
    for arr, col in ((d["cin"], L_INPUT), (d["cou"], L_OUTPUT)):
        axT.plot(d["t"], np.clip(_rate(arr, d["t"]), 1, None), color=col, lw=3.0 * fs, zorder=8,
                 path_effects=LHALO)
    axT.set_yscale("log"); axT.set_ylim(1, grmax * 1.4)
    axT.set_ylabel("tok/min (log)", fontsize=12 * fs); axT.tick_params(labelsize=11 * fs)
    for k in range(1, d["n"]):                            # faint round dividers only (no rN labels)
        ax.axvline(d["starts"][k], color=RND_C, ls=(0, (3, 2)), lw=1.0, alpha=0.35, zorder=2)
    ax.set_title(f"{lab}    ·    {d['total']:.0f} min", loc="left", color=INK,
                 fontsize=15 * fs, fontweight="bold", pad=6)


def fl_combined_duration(DATA):
    """Same FIGURE shape/scale as fig_traj_A_rate_combined_scales (21×17.5, identical panel bands)
    so a slide can morph straight from it — the ONLY difference is each panel's WIDTH is now
    proportional to its real duration. The 20-min bar, gold milestones, round labels and chip are
    dropped to keep it clean."""
    c = _CMB
    totals = [d["total"] for d, _ in DATA]
    maxT = max(totals)
    grmax = 1.0                                           # shared rate range → comparable right axes
    for d, _l in DATA:
        for arr in (d["cin"], d["cou"]):
            grmax = max(grmax, float(np.clip(_rate(arr, d["t"]), 1, None).max()))
    fullW = c["right"] - c["left"]                        # full panel width = combined_scales panel width
    H = (c["top"] - c["bottom"]) / (c["n"] + (c["n"] - 1) * c["hspace"])   # same panel height as scales
    S = c["hspace"] * H
    fig = plt.figure(figsize=(21, c["figh"]))
    for i, (d, lab) in enumerate(DATA):
        b = c["top"] - i * (H + S) - H                   # same vertical band as the scales panels
        w = fullW * (totals[i] / maxT)                   # width ∝ duration (left-aligned)
        ax = fig.add_axes([c["left"], b, w, H])
        _panel_duration(ax, d, lab, grmax, fs=1.5)
    # shared time ruler just below the panels (full common scale, 0..maxT)
    axr = fig.add_axes([c["left"], c["bottom"] - 0.055, fullW, 0.0012])
    axr.set_xlim(0, maxT); axr.set_yticks([])
    for sp in ("left", "right", "top"):
        axr.spines[sp].set_visible(False)
    axr.spines["bottom"].set_color(INK)
    axr.tick_params(labelsize=16)
    axr.set_xlabel("Time (min)   —   common scale; each panel's width = run duration", fontsize=20)
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + [
        Line2D([0], [0], color=L_INPUT, lw=3.2, label="rate input"),
        Line2D([0], [0], color=L_OUTPUT, lw=3.2, label="rate output"),
        Line2D([0], [0], color=RND_C, lw=1.4, ls=(0, (3, 2)), label="round")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=18, frameon=True,
               facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.3, handlelength=1.8, borderpad=0.6)
    fig.savefig(OUT / "fig_traj_A_rate_combined_duration.png", dpi=400, facecolor=BG)
    fig.savefig(OUT / "fig_traj_A_rate_combined_duration.svg", facecolor=BG)
    plt.close(fig)
    print("wrote fig_traj_A_rate_combined_duration.png")


def fl_A_rate_combined_scales(DATA):
    """All four runs together (chosen opt3 style). The 20-min scale bar lives BELOW each time
    axis; the four left-aligned bars have very different lengths → the differing time scales are
    obvious at a glance."""
    c = _CMB
    fig, axes = _fig(h=c["figh"], top=c["top"], bottom=c["bottom"], hspace=c["hspace"])
    for ax, (d, lab) in zip(axes, DATA):
        _panel_rate(ax, d, lab, ax is axes[-1], fs=1.5, show_spend=False)
        _scale_bar(ax, d, minutes=20)
    axes[-1].set_xlabel(XLAB, fontsize=16 * 1.5, labelpad=54)   # sits below the scale bar
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + \
        _rate_legend_handles(activity="none", show_spend=False)
    _legend(fig, h, "", "fig_traj_A_rate_combined_scales", fs=18, leg_y=0.028, dpi=400)


def fl_A_rate_short(DATA):
    """A_rate (stacked, SHORTER panels) — same content, less tall per panel."""
    fig, axes = _fig(h=12.5, top=0.905, bottom=0.135, hspace=0.62)
    for ax, (d, lab) in zip(axes, DATA):
        _panel_rate(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + _line_handles(rate=True)
    _legend(fig, h, "Authoring trajectory — activity share + token consumption RATE + spend",
            "fig_traj_A_rate_short", sup_y=0.962, leg_y=0.02)


def fl_A_rate_2x2(DATA):
    """A_rate (2x2) — wider landscape layout: raw·scaffold over python·circt."""
    fig, axes = plt.subplots(2, 2, figsize=(28, 13))
    fig.subplots_adjust(left=0.045, right=0.93, top=0.90, bottom=0.11, wspace=0.40, hspace=0.30)
    flat = axes.flatten()
    for i, (ax, (d, lab)) in enumerate(zip(flat, DATA)):
        _panel_rate(ax, d, lab, last=(i >= 2), compact=True, fs=0.92)
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + _line_handles(rate=True)
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=14.5,
               frameon=True, facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.022),
               columnspacing=1.3, handlelength=1.8, borderpad=0.7)
    suptitle(fig, "Authoring trajectory — activity share + token consumption RATE + spend", y=0.965, fs=26)
    fig.savefig(OUT / "fig_traj_A_rate_2x2.png", bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(OUT / "fig_traj_A_rate_2x2.svg", bbox_inches="tight", facecolor=BG)
    print("wrote fig_traj_A_rate_2x2.png")


def fl_B_rate(DATA):
    """B_rate — activity SHARE area (left) + STACKED token consumption-rate area (right) + spend axis."""
    fig, axes = _fig()
    for ax, (d, lab) in zip(axes, DATA):
        style_ax(ax, grid=None)
        g, sh = _share_grid(d)
        ax.stackplot(g, *[sh[a] for a in ACTS], colors=[ACT_COL[a] for a in ACTS], alpha=0.40, zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("activity share", fontsize=14)
        axT = ax.twinx(); axT.spines["top"].set_visible(False); axT.spines["right"].set_color(INK)
        base = np.zeros_like(d["t"])
        for arr, col in ((d["cca"], L_CACHE), (d["cin"], L_INPUT), (d["cou"], L_OUTPUT)):
            r = np.clip(_rate(arr, d["t"]), 0, None)
            axT.fill_between(d["t"], base, base + r, color=col, alpha=0.5, lw=0, zorder=3)
            base = base + r
        axT.plot(d["t"], base, color=L_TOTAL, lw=2.6, zorder=8, path_effects=LHALO)
        axT.set_ylim(0, (base.max() * 1.2) or 1); axT.set_ylabel("token rate (tok/min, stacked)", fontsize=14)
        axT.tick_params(labelsize=11)
        _spend_axis(ax, d)
        _rounds(ax, d); _milestones(ax, d); _chip(ax, d); _basics(ax, d, lab, ax is axes[-1])
    h = [Patch(fc=ACT_COL[a], alpha=0.40, label=ACT_LAB[a]) for a in ACTS] + [
        Patch(fc=L_CACHE, alpha=0.5, label="cache-read rate"), Patch(fc=L_INPUT, alpha=0.5, label="input rate"),
        Patch(fc=L_OUTPUT, alpha=0.5, label="output rate"), Line2D([0], [0], color=L_TOTAL, lw=2.6, label="total rate"),
        Line2D([0], [0], color=L_SPEND, lw=3.6, ls=(0, (6, 2)), label="spend ($)"),
        Line2D([0], [0], color=GOLD, lw=2.6, ls=(0, (4, 3)), label="milestone")]
    _legend(fig, h, "Authoring trajectory — activity share + STACKED token rate + spend", "fig_traj_B_rate")


# ====================================================================================
#  LOW-OVERLOAD COMPARISON FIGURES — one metric, all four runs overlaid.
#  Each run gets a distinct COLOUR + LINE STYLE + MARKER so overlapping curves stay
#  readable.  (name, colour, marker, linestyle)
# ====================================================================================
RUN_STYLE = [
    ("raw C++ (from scratch)",   MAUVE, "o", "-"),
    ("C++ & Merlin Infra",       NAVY,  "s", (0, (7, 2.5))),         # deep blue, long dash
    ("Merlin & Python tooling",  SLATE, "D", (0, (1.5, 2.0))),       # bluish grey, dotted
    ("Merlin & CIRCT Tooling",   GOLD,  "^", (0, (8, 2.5, 1.5, 2.5))),  # gold, dash-dot
]
_HALO_TXT = lambda lw=3.4: [pe.withStroke(linewidth=lw, foreground=BG)]   # cream halo → text reads over lines


def _save_cmp(fig, fname, dpi=700):
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=dpi, facecolor=BG)
    fig.savefig(OUT / f"{fname}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {fname}.png")


def _final_tests(d):
    """Final tests-passing = the best the run's self-check reached (matches the gold milestones in
    the detailed plots; e.g. Infra reaches 19/20 at ~203 min)."""
    ms = d.get("milestones") or []
    if ms:
        return int(max(c for _, c in ms))
    return int(d["passed"][-1]) if d.get("passed") else 0


def _tests_steps(d):
    """(time, tests-passing) step series showing the intermediate rises within a run (e.g. 16→17→19),
    using the self-check milestones; the curve steps up to each value at exactly the time it happened."""
    ms = sorted(d.get("milestones") or [])
    if not ms and d.get("passed"):                        # fallback: per-round QA counts
        starts = d["starts"]
        ms = [(float(starts[min(k + 1, len(starts) - 1)]), int(p)) for k, p in enumerate(d["passed"])]
    xs, ys, last = [0.0], [0], 0
    for x, c in ms:
        c = max(int(c), last)                             # enforce non-decreasing
        xs.append(float(x)); ys.append(c); last = c
    xs.append(d["total"]); ys.append(ys[-1])
    return xs, ys


def _cmp_handles():
    return [Line2D([0], [0], color=c, lw=4.4, ls=ls, marker=m, mfc=c, mec=INK, ms=13, label=n)
            for n, c, m, ls in RUN_STYLE]


def _draw_tests(ax, DATA, fs=1.0):
    style_ax(ax)
    xmax = max(d["total"] for d, _ in DATA)
    for (d, _l), (name, col, mk, ls) in zip(DATA, RUN_STYLE):
        xs, ys = _tests_steps(d)
        ax.step(xs, ys, where="post", color=col, lw=4.4, ls=ls, zorder=5, solid_capstyle="round")
        ch = [k for k in range(1, len(ys)) if ys[k] != ys[k - 1]]    # markers only where the count CHANGES
        ax.scatter([xs[k] for k in ch], [ys[k] for k in ch], s=95, color=col, ec=INK, lw=1.2,
                   zorder=6, marker=mk)
        ax.scatter([xs[-1]], [ys[-1]], s=240, color=col, ec=INK, lw=1.8, zorder=7, marker=mk)
        ax.text(xs[-1] + 0.014 * xmax, ys[-1], f"{ys[-1]}/20", color=col, fontsize=21 * fs,
                fontweight="bold", va="center", ha="left", zorder=9, path_effects=_HALO_TXT())
    ax.set_xlim(0, xmax * 1.13); ax.set_ylim(-0.6, 21.2)
    ax.set_yticks([0, 5, 10, 15, 20])                                # integer ticks only — no .5
    ax.set_xlabel("Time (min)", fontsize=23 * fs); ax.set_ylabel("tests passing (of 20)", fontsize=23 * fs)
    ax.tick_params(labelsize=19 * fs)
    title(ax, "Tests passing over time", fs=26 * fs)


def fl_compare_tests_facets(DATA):
    """Cleaner 'tests passing over time': ONE lane per run, shared time axis → zero overlap,
    each run trivially distinguishable, and the vertically-aligned jumps make the velocity
    comparison obvious (CIRCT jumps early, Infra late).  Shows the intermediate rises within a
    run (e.g. 13→17→20), not just the final jump."""
    fig, axes = plt.subplots(len(DATA), 1, figsize=(15.5, 14), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.93, bottom=0.075, hspace=0.55)
    xmax = max(d["total"] for d, _ in DATA)
    for i, (ax, (d, _l), (name, col, mk, ls)) in enumerate(zip(axes, DATA, RUN_STYLE)):
        style_ax(ax)
        xs, ys = _tests_steps(d)
        ax.fill_between(xs, 0, ys, step="post", color=col, alpha=0.18, zorder=2)
        ax.step(xs, ys, where="post", color=col, lw=5.2, zorder=5, solid_capstyle="round")
        ch = [k for k in range(1, len(ys)) if ys[k] != ys[k - 1]]    # marker at each intermediate rise
        ax.scatter([xs[k] for k in ch], [ys[k] for k in ch], s=150, color=col, ec=INK, lw=1.6,
                   zorder=6, marker=mk)
        ax.scatter([xs[-1]], [ys[-1]], s=340, color=col, ec=INK, lw=2.2, zorder=7, marker=mk)
        ax.set_xlim(0, xmax * 1.12); ax.set_ylim(0, 21.8)
        ax.set_yticks([0, 10, 20])                                   # integer ticks only
        ax.tick_params(labelsize=23)                                 # match the spend-figure sizing
        ax.set_ylabel("tests", fontsize=27)
        # run name as a TITLE above the lane (no longer clipping the curve), in the run colour
        ax.set_title(name, loc="left", color=col, fontsize=30, fontweight="bold", pad=10)
        ax.text(xs[-1] + 0.012 * xmax, ys[-1], f"{ys[-1]}/20", color=col, fontsize=27,
                fontweight="bold", va="center", ha="left", path_effects=_HALO_TXT(4.0))
        if i == len(DATA) - 1:
            ax.set_xlabel("Time (min)", fontsize=28)
    suptitle(fig, "Tests passing over time — one lane per run", y=0.99, fs=33)
    _save_cmp(fig, "fig_compare_tests_facets")


def _draw_cost(ax, DATA, fs=1.0):
    style_ax(ax)
    xmax = max(d["total"] for d, _ in DATA)
    ymax = max(float(d["csp"][-1]) for d, _ in DATA)
    for (d, _l), (name, col, mk, ls) in zip(DATA, RUN_STYLE):
        ax.plot(d["t"], d["csp"], color=col, lw=4.4, ls=ls, zorder=5, solid_capstyle="round")
        xe, ye = float(d["t"][-1]), float(d["csp"][-1])
        ax.scatter([xe], [ye], s=240, color=col, ec=INK, lw=1.8, zorder=7, marker=mk)
        # label ABOVE the endpoint (clear space over the fan of curves) + cream halo
        ax.annotate(f"${d['cum_cost'][-1]:.0f} · {d['tot_tok'] / 1e6:.0f}M", (xe, ye),
                    xytext=(0, 18), textcoords="offset points", color=col, fontsize=20 * fs,
                    fontweight="bold", va="bottom", ha="center", zorder=9, path_effects=_HALO_TXT(3.8))
    ax.set_xlim(0, xmax * 1.10); ax.set_ylim(0, ymax * 1.24)
    ax.set_xlabel("Time (min)", fontsize=23 * fs); ax.set_ylabel("cumulative cost (USD)", fontsize=23 * fs)
    ax.tick_params(labelsize=19 * fs)
    title(ax, "Spend over time   (label = total $ · tokens)", fs=26 * fs)


def fl_compare_tests(DATA):
    fig, ax = plt.subplots(figsize=(15, 8.8))
    _draw_tests(ax, DATA, fs=1.18)
    ax.legend(handles=_cmp_handles(), loc="lower right", fontsize=19, framealpha=0.96)
    _save_cmp(fig, "fig_compare_tests_vs_time")


def fl_compare_cost(DATA):
    """Standalone 'Spend over time' (same drawing as the overview's right panel)."""
    fig, ax = plt.subplots(figsize=(15, 8.8))
    _draw_cost(ax, DATA, fs=1.18)
    ax.legend(handles=_cmp_handles(), loc="upper left", fontsize=19, framealpha=0.96)
    _save_cmp(fig, "fig_compare_cost_vs_time")


def fl_compare_efficiency(DATA):
    """The recommended single-glance view: time (x) vs cost (y), bubble size = tests passing.
    The ideal run is a BIG bubble in the bottom-left (fast, cheap, most tests)."""
    fig, ax = plt.subplots(figsize=(14.5, 9.3))
    style_ax(ax, grid="both")
    xmax = max(d["total"] for d, _ in DATA)
    ymax = max(float(d["cum_cost"][-1]) for d, _ in DATA)
    for (d, _l), (name, col, mk, ls) in zip(DATA, RUN_STYLE):
        t = d["total"]; c = float(d["cum_cost"][-1]); tp = _final_tests(d); tok = d["tot_tok"] / 1e6
        s = (max(tp - 14, 1)) ** 2 * 170 + 320           # amplify the 17–20 spread into visible area
        ax.scatter([t], [c], s=s, color=col, ec=INK, lw=2.0, alpha=0.9, zorder=5)
        ax.annotate(f"{name}\n{tp}/20 tests · ${c:.0f} · {tok:.0f}M tok · {t:.0f} min", (t, c),
                    xytext=(16, 18), textcoords="offset points", fontsize=17, fontweight="bold",
                    color=INK, zorder=8, ha="left", va="bottom", path_effects=_HALO_TXT(3.8))
    ax.set_xlim(0, xmax * 1.26); ax.set_ylim(0, ymax * 1.34)
    ax.set_xlabel("time to finish (min)   →   faster is left", fontsize=22)
    ax.set_ylabel("total cost (USD)   →   cheaper is down", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.annotate("best  ↙   cheaper · faster\n(bigger bubble = more tests passing)",
                xy=(0.015, 0.02), xycoords="axes fraction", fontsize=17, color=INK,
                fontweight="bold", ha="left", va="bottom", fontstyle="italic")
    title(ax, "Cost vs time per run — bubble size = tests passing", fs=25)
    _save_cmp(fig, "fig_compare_efficiency")


def fl_compare_overview(DATA):
    """Single-slide overview: progress (left) + spend (right), shared run legend."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(24, 10))
    _draw_tests(axL, DATA, fs=1.12); _draw_cost(axR, DATA, fs=1.12)
    fig.legend(handles=_cmp_handles(), loc="lower center", ncol=4, fontsize=21, frameon=True,
               facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, -0.005),
               columnspacing=2.2, handlelength=2.8)
    suptitle(fig, "Comparing the four runs — progress and spend", y=1.0, fs=28)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _save_cmp(fig, "fig_compare_overview")


def main():
    use_merlin_style()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = [(load_arm(rel), lab) for rel, lab in ARMS]
    for fn in (fl_A, fl_B, fl_C, fl_F, fl_A_rate, fl_B_rate, fl_A_rate_2x2, fl_A_rate_short,
               fl_A_rate_options, fl_A_rate_arms, fl_A_rate_combined_scales, fl_combined_duration,
               fl_run1_buildup,
               fl_compare_tests, fl_compare_cost, fl_compare_efficiency, fl_compare_overview,
               fl_compare_tests_facets):
        fn(DATA)


if __name__ == "__main__":
    raise SystemExit(main())
