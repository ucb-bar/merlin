"""Single-column campaign figure: one stitched run, agent trajectory, test cases, RTL performance, cores.

The continuations of a campaign are drawn as ONE run. The x axis is minutes of active run time: the
concatenation, in launch order, of each attempt's window from its lead agent's start to its last
record, cut where a finished attempt's background grader stopped recording (no grade, grade result
or certification job for 45 min). Records just outside a window are clipped to its edge. Pass status
is carried across the whole timeline and never reset.

Every series is computed from :func:`.records.read_campaign`'s dataset; ``build_series`` returns the
numbers (also written to the notes JSON) and ``draw`` renders them.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------- style

WHITE = "#FFFFFF"
GRIDC = "#E4DED3"
SPINE = "#B7A994"
INK = "#2B2B33"
INK2 = "#6E6658"
NAVY = "#1D3A5C"
ROSE = "#8C6461"
SLATE = "#8D97AB"
TAUPE = "#B9A48E"
SAGE = "#7F8C6A"
PINK = "#E3BFC0"
SAND = "#EBD9BF"
LAV = "#B7B4C8"
PBLUE = "#C9D6E3"
PSAGE = "#D6DCC6"
PTAUPE = "#D9C9B6"
PALE = "#EEE8DE"
CALLOUT = "#FBF6EE"

ACTIVITY = [  # bottom -> top
    ("thinking", "thinking", LAV),
    ("reading", "reading", PBLUE),
    ("writing", "writing code", SAND),
    ("shell", "shell", PSAGE),
    ("helpers", "wait: helpers", PTAUPE),
    ("checks", "wait: checks", PINK),
]
COMPUTE = [  # bottom -> top
    ("l3_sim", "L3 sim", NAVY),
    ("l2_sim", "L2 sim", SLATE),
    ("reference", "L0/L1", SAGE),
    ("compile", "compile", TAUPE),
    ("other", "other", SAND),
]
SERIF = ["P052", "URW Palladio L", "Palatino", "DejaVu Serif"]
SANS = ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]
FS = 6.5
GRADER_GAP_S = 2700.0


# -------------------------------------------------------------------------------------- timeline


def grader_down_start(data: dict, att: dict) -> float | None:
    """Last background-grading record of a finished attempt that kept working >45 min without one."""
    if att.get("live") or not att.get("last_record"):
        return None
    ts = [g["updated"] for g in data["grades"] if g["attempt"] == att["id"] and g.get("updated")]
    ts += [e["t"] for e in data["events"] if e["attempt"] == att["id"] and e.get("t")]
    ts += [j["finished"] for j in data["rtl_jobs"] if j["attempt"] == att["id"] and j.get("finished")]
    last = max(ts, default=None)
    return last if last and att["last_record"] - last > GRADER_GAP_S else None


class Stitch:
    """Real time -> minutes of active run time (agent working, background grader alive)."""

    def __init__(self, data: dict):
        wins = []
        for a in data["attempts"]:
            s, e = a.get("agent_start"), a.get("last_record")
            if not (s and e and e > s):
                continue
            down = grader_down_start(data, a)
            wins.append((s, min(e, down) if down else e, a["id"]))
        wins.sort()
        self.wins = wins
        self.m0, m = [], 0.0
        for s, e, _ in wins:
            self.m0.append(m)
            m += (e - s) / 60.0
        self.total = m
        self._starts = [w[0] for w in wins]
        self._index = {aid: i for i, (_, _, aid) in enumerate(wins)}

    def inside(self, t: float) -> float | None:
        i = bisect.bisect_right(self._starts, t) - 1
        if i >= 0 and t <= self.wins[i][1]:
            return self.m0[i] + (t - self.wins[i][0]) / 60.0
        return None

    def clip(self, t: float, attempt: str) -> float | None:
        i = self._index.get(attempt)
        if i is None:
            return self.inside(t)
        s, e, _ = self.wins[i]
        return self.m0[i] + (min(max(t, s), e) - s) / 60.0


def smooth(y, sigma: float) -> np.ndarray:
    y = np.asarray(y, float)
    if sigma <= 0 or len(y) < 3:
        return y
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(y, r, mode="edge"), k / k.sum(), mode="valid")


# ---------------------------------------------------------------------------------------- series

FAILS = ("fail", "failed", "mismatch", "error", "timeout", "timed_out")


def pass_curve(data: dict, S: Stitch) -> tuple[list[tuple[float, int, int]], dict[str, str]]:
    """Stepped (minute, test cases passing L2-or-better, holding L3) points and the final state.

    A test case counts from its first L2 pass and is promoted at its first L3 certificate; a later
    failing L2 grade drops it, a later failing L3 certificate demotes it to L2. Never reset."""
    evs = []
    for e in data["events"]:
        if e.get("t") and e.get("test_case"):
            m = S.clip(e["t"], e["attempt"])
            if m is not None:
                evs.append((m, e["t"], e))
    evs.sort(key=lambda x: (x[0], x[1]))
    state: dict[str, str] = {}
    out = [(0.0, 0, 0)]
    for m, _, e in evs:
        name, status, prev = e["test_case"], str(e.get("status")), state.get(e["test_case"])
        new = prev
        if e["tier"] == "L2":
            if status in ("pass", "passed"):
                new = prev or "L2"
            elif status in FAILS:
                new = None
        elif status in ("pass", "passed"):
            new = "L3"
        elif status in FAILS and prev == "L3":
            new = "L2"
        if new != prev:
            if new is None:
                state.pop(name, None)
            else:
                state[name] = new
            out.append((m, len(state), sum(1 for v in state.values() if v == "L3")))
    return out, state


def perf_series(data: dict, S: Stitch):
    """Geomean speedup (first / latest RTL cycles per test case measured >= 2 times on the same test-case
    digest; IQR) and median/IQR PE-array utilization of each test case's latest measurement."""
    rows = sorted(((S.clip(r["t"], r["attempt"]), r["t"], r) for r in data.get("perf") or []), key=lambda x: (x[0] or 0, x[1]))
    first, latest, util = {}, {}, {}
    sp, ut = [], []
    for m, _, r in rows:
        if m is None:
            continue
        name = r["test_case"]
        if name not in first or first[name]["case_digest"] != r["case_digest"]:
            first[name] = r
            latest.pop(name, None)
        else:
            latest[name] = r
        if r.get("floor_cycles"):
            util[name] = r["floor_cycles"] / r["cycles"]
        if latest:
            lr = np.log([first[c]["cycles"] / latest[c]["cycles"] for c in latest])
            sp.append((m, float(np.exp(lr.mean())), float(np.exp(np.percentile(lr, 25))), float(np.exp(np.percentile(lr, 75))), len(lr)))
        if util:
            u = np.array(list(util.values())) * 100
            ut.append((m, float(np.median(u)), float(np.percentile(u, 25)), float(np.percentile(u, 75)), len(u)))
    return sp, ut, len(rows)


def activity_series(data: dict, S: Stitch, n: int):
    acc = {k: np.zeros(n) for k, _, _ in ACTIVITY}
    turn = np.zeros(n)

    def add(arr, a, b):
        t = a
        while t < b:
            nxt = min(b, (math.floor(t / 60.0) + 1) * 60.0)
            m = S.inside((t + nxt) / 2)
            if m is not None:
                arr[min(int(m), n - 1)] += nxt - t
            t = nxt

    for th in data["threads"]:
        for a, b in th["turns"]:
            add(turn, a, b)
        for b0, b1, act in th["spans"]:
            if b0 and b1 and b1 > b0 and act in acc:
                add(acc[act], b0, b1)
    used = sum(acc[k] for k in acc if k != "thinking")
    scale = np.where(used > turn, turn / np.maximum(used, 1e-9), 1.0)
    for k in acc:
        if k != "thinking":
            acc[k] = acc[k] * scale
    acc["thinking"] = np.maximum(turn - sum(acc[k] for k in acc if k != "thinking"), 0.0)
    return acc, turn


def token_series(data: dict, S: Stitch, n: int):
    rin, rout = np.zeros(n), np.zeros(n)
    cum = {k: np.zeros(n) for k in ("fresh", "cached", "output")}
    for th in data["threads"]:
        prev = None
        for t, tok in th["tokens"]:
            if prev is not None:
                m = S.inside(t)
                if m is not None:
                    i = min(int(m), n - 1)
                    d = {k: max(0, tok.get(k, 0) - prev.get(k, 0)) for k in cum}
                    rin[i] += d["fresh"] + d["cached"]
                    rout[i] += d["output"]
                    for k in cum:
                        cum[k][i] += d[k]
            prev = tok
    return rin, rout, {k: np.cumsum(v) for k, v in cum.items()}


def compute_series(data: dict, S: Stitch, n: int):
    """Estimated core use by kind, normalised per launch to its measured CPU time.

    Job intervals give the shape; each launch's concurrent-job counts are scaled by one factor f <= 1
    and capped at the reservation so their integral equals the CPU time the launch's accounting
    recorded. CPU the records cannot attribute is 'other', spread evenly over the launch. A launch
    without accounting uses the median factor of the others."""
    cap = float(data.get("reserved_cores") or 20)
    bin_s = data["bin_s"]
    per = {}
    for a in data["attempts"]:
        rows = []
        for k, v in (data["core_bins"].get(a["id"]) or {}).items():
            rows.append((int(k) * bin_s, {kind: x / bin_s for kind, x in v.items()}))
        per[a["id"]] = sorted(rows, key=lambda r: r[0])

    def capped(rows, f):
        return sum(min(cap, f * sum(k.values())) * bin_s for _, k in rows)

    factors = {}
    for a in data["attempts"]:
        acct, rows = a.get("accounting") or {}, per[a["id"]]
        if not rows or not acct.get("cpu_s"):
            continue
        if capped(rows, 1.0) <= acct["cpu_s"]:
            factors[a["id"]] = 1.0
            continue
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if capped(rows, mid) < acct["cpu_s"] else (lo, mid)
        factors[a["id"]] = (lo + hi) / 2
    median_f = float(np.median(list(factors.values()))) if factors else 1.0
    out = {k: np.zeros(n) for k, _, _ in COMPUTE}
    cnt = np.zeros(n)
    notes = {}
    for a in data["attempts"]:
        acct, rows = a.get("accounting") or {}, per[a["id"]]
        f = factors.get(a["id"], median_f)
        rest = 0.0
        if acct.get("cpu_s") and acct.get("runtime_s"):
            rest = max(0.0, acct["cpu_s"] - capped(rows, f)) / acct["runtime_s"]
        notes[a["id"]] = {"factor": f, "unattributed_cores": rest, "measured": bool(acct.get("cpu_s"))}
        filled = {}
        for t, kinds in rows:
            tot = sum(kinds.values())
            g = min(f, cap / tot) if tot > 0 else f
            filled[t] = {k: v * g for k, v in kinds.items()}
        if rest > 0 and a.get("launch"):
            t = math.floor(a["launch"] / bin_s) * bin_s
            while t < a["launch"] + acct["runtime_s"]:
                cur = filled.setdefault(t, {})
                cur["other"] = cur.get("other", 0.0) + min(rest, max(0.0, cap - sum(cur.values())))
                t += bin_s
        for t, kinds in filled.items():
            m = S.inside(t + bin_s / 2)
            if m is None:
                continue
            i = min(int(m), n - 1)
            for k, v in kinds.items():
                if k in out:
                    out[k][i] += v
            cnt[i] += 1
    for k in out:
        out[k] = np.where(cnt > 0, out[k] / np.maximum(cnt, 1), 0.0)
    return out, notes


def fmt_tok(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M" if v >= 1e7 else f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def build_series(data: dict) -> dict[str, Any]:
    S = Stitch(data)
    n = int(math.ceil(S.total)) + 1
    curve, final_state = pass_curve(data, S)
    acts, turn = activity_series(data, S, n)
    rin, rout, cum = token_series(data, S, n)
    comp, comp_notes = compute_series(data, S, n)
    sp, ut, n_perf = perf_series(data, S)
    return {"S": S, "n": n, "curve": curve, "final_state": final_state, "acts": acts, "turn": turn, "rin": rin,
            "rout": rout, "cum": cum, "comp": comp, "comp_notes": comp_notes, "sp": sp, "ut": ut, "n_perf": n_perf}


# ---------------------------------------------------------------------------------------- drawing


def _style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.serif": SERIF, "font.size": FS,
        "axes.facecolor": WHITE, "figure.facecolor": WHITE, "savefig.facecolor": WHITE, "axes.edgecolor": SPINE,
        "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRIDC, lw=0.5, ls=(0, (1, 2.2)))
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)


def _title(fig, ax, text, W, H, x_in=None):
    bb = ax.get_position()
    x = (x_in / W) if x_in is not None else bb.x0 - 0.30 / W
    fig.text(x, bb.y1 + 0.05 / H, text, ha="left", va="bottom", fontsize=8.5, family="serif", fontweight="bold", color=NAVY)


def _callout(ax, x, y, txt, ha, xmax, on=None):
    dx = 0.012 * xmax if ha == "left" else -0.012 * xmax
    (on or ax).text(x + dx, y, txt, transform=ax.transData, ha=ha, va="top", fontsize=FS - 0.3, color=ROSE,
                    fontweight="bold", zorder=7, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.1", fc=CALLOUT, ec=TAUPE, lw=0.5))


def draw(data: dict, ser: dict, out_stem: str, *, formats=("pdf", "svg", "png"), dpi: int = 600) -> dict[str, Any]:
    """Render the figure to ``<out_stem>.<fmt>`` and return the notes (numbers behind it)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _style(plt)
    S, n, curve = ser["S"], ser["n"], ser["curve"]
    XMAX = max(S.total, 1.0)
    xm = np.arange(n) + 0.5
    cases = data["test_cases"]
    N = len(cases) or 1
    cum = ser["cum"]
    tot_tok = cum["fresh"] + cum["cached"] + cum["output"]
    cap = int(data.get("reserved_cores") or 20)
    lab = {"l2": "L2 sim", "l3": "L3 sim", **(data.get("labels") or {})}
    compute = [(k, {"l3_sim": lab["l3"], "l2_sim": lab["l2"]}.get(k, lbl), col) for k, lbl, col in COMPUTE]

    W, L, R = 3.4, 0.46, 0.50
    spec = [("act", 0.64, 0.30), ("agents", 0.17, 0.03), ("pass", 0.66, 0.48), ("speed", 0.46, 0.25),
            ("util", 0.46, 0.06), ("comp", 0.52, 0.25), ("cohort", 0.58, 0.44)]
    H = 0.04 + 0.30 + sum(h + a for _, h, a in spec)
    fig = plt.figure(figsize=(W, H))
    axes, y = {}, H - 0.04
    for name, h, a in spec:
        y -= a + h
        left = 1.02 if name == "cohort" else L
        axes[name] = fig.add_axes([left / W, y / H, (W - left - R) / W, h / H])
    for k in ("act", "agents", "pass", "speed", "util", "comp"):
        _axes_style(axes[k])
        axes[k].set_xlim(0, XMAX)
        if k != "comp":
            axes[k].tick_params(labelbottom=False)

    def tokens_at(m):
        return float(tot_tok[min(max(int(m), 0), n - 1)]) if n else 0.0

    miles = []
    l3_100 = next((c for c in curve if c[2] >= 100), None)
    near80 = next((c for c in curve if c[1] >= round(0.8 * N)), None)
    for c in sorted([x for x in (near80, l3_100) if x], key=lambda c: c[0]):
        if not miles or c[0] - miles[-1][1][0] > 0.18 * XMAX:
            miles.append((f"{c[1]}/{N}", c))

    # (a) agent trajectory
    ax = axes["act"]
    sig = max(2.0, XMAX / 140.0)
    tot = smooth(ser["turn"], sig)
    stack = np.vstack([smooth(ser["acts"][k], sig) / np.maximum(tot, 1e-9) for k, _, _ in ACTIVITY])
    stack = stack / np.maximum(stack.sum(axis=0), 1e-9)
    base = np.zeros(n)
    for (_, _, col), v in zip(ACTIVITY, stack):
        ax.fill_between(xm, base, base + v, color=col, lw=0)
        base = base + v
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_ylabel("share", labelpad=1)
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.plot(xm, np.maximum(smooth(ser["rin"], sig * 1.5), 1), color=SAGE, lw=1.0)
    ax2.plot(xm, np.maximum(smooth(ser["rout"], sig * 1.5), 1), color=NAVY, lw=1.0)
    ax2.set_ylim(2e2, 5e6)
    ax2.set_yticks([1e3, 1e5])
    ax2.set_yticklabels(["1k", "100k"])
    ax2.minorticks_off()
    ax2.tick_params(axis="y", colors=INK2, labelsize=FS, length=2, pad=1.5)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(SPINE)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.set_ylabel("tok/min", color=INK2, labelpad=1)
    for lbl, (m, _n2, n3) in miles:
        for k in ("act", "agents", "pass"):
            axes[k].axvline(m, color=TAUPE, lw=0.7, ls=(0, (3, 2)), zorder=4)
        ax.plot([m], [1.0], marker="v", ms=2.8, color=TAUPE, clip_on=False, zorder=6)
        _callout(ax, m, 0.93, f"{lbl} · L3 {n3}\n{fmt_tok(tokens_at(m))} tok", "left", XMAX, on=ax2)
    _callout(ax, XMAX, 0.93, f"{curve[-1][1]}/{N} · L3 {curve[-1][2]}\n{fmt_tok(tokens_at(XMAX))} tok", "right", XMAX, on=ax2)
    _title(fig, ax, "Agent trajectory", W, H)
    bb = ax.get_position()
    fig.text(bb.x1 + 0.34 / W, bb.y1 + 0.05 / H,
             f"{S.total:.0f} min · {fmt_tok(cum['fresh'][-1])} fresh + {fmt_tok(cum['cached'][-1])} cached in · {fmt_tok(cum['output'][-1])} out",
             ha="right", va="bottom", fontsize=FS - 0.3, color=INK2)

    ax = axes["agents"]
    active = smooth(ser["turn"] / 60.0, 1.0)
    ax.fill_between(xm, active, step="mid", color=SLATE, alpha=0.55, lw=0)
    ax.step(xm, active, where="mid", color=SLATE, lw=0.6)
    peak = max(1, int(math.ceil(active.max()))) if n else 1
    ax.set_ylim(0, peak + 0.3)
    ax.set_yticks([])
    ax.grid(False, axis="y")
    ax.text(0.006 * XMAX, peak + 0.25, f"agents active (0–{peak})", ha="left", va="top", fontsize=FS - 0.3, color=INK2)
    handles = [Patch(color=c, label=lbl) for _, lbl, c in ACTIVITY]
    handles += [Line2D([], [], color=SAGE, lw=1.0, label="input tok/min"), Line2D([], [], color=NAVY, lw=1.0, label="output tok/min")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.13, -0.14), ncol=4, frameon=False, fontsize=FS,
              handlelength=1.0, handleheight=0.75, columnspacing=0.7, handletextpad=0.3, labelspacing=0.15, borderaxespad=0)

    # (b) test cases passed over time
    ax = axes["pass"]
    xs = [c[0] for c in curve] + [XMAX]
    l2 = [c[1] for c in curve] + [curve[-1][1]]
    l3 = [c[2] for c in curve] + [curve[-1][2]]
    ax.fill_between(xs, l2, step="post", color=SLATE, alpha=0.22, lw=0)
    ax.step(xs, l2, where="post", color=SLATE, lw=1.1)
    ax.fill_between(xs, l3, step="post", color=NAVY, alpha=0.20, lw=0)
    ax.step(xs, l3, where="post", color=NAVY, lw=1.1)
    ax.plot([XMAX], [l2[-1]], "o", ms=3.0, color=SLATE, clip_on=False, zorder=5)
    ax.plot([XMAX], [l3[-1]], "s", ms=2.8, color=NAVY, clip_on=False, zorder=5)
    ax.text(XMAX * 1.025, l2[-1], f"{l2[-1]} L2", ha="left", va="center", fontsize=FS, color=SLATE, fontweight="bold")
    ax.text(XMAX * 1.025, l3[-1], f"{l3[-1]} L3", ha="left", va="center", fontsize=FS, color=NAVY, fontweight="bold")
    ax.text(0.97 * XMAX, l2[-1] - 0.07 * N, f"passed on {lab['l2']} (L2)", ha="right", va="top", fontsize=FS, color=SLATE)
    ax.text(0.97 * XMAX, l3[-1] - 0.07 * N, f"certified on {lab['l3']} RTL (L3)", ha="right", va="top", fontsize=FS, color=NAVY)
    regression = None
    window = [(max(c[1] for c in curve[max(0, i - 40):i]) - curve[i][1], i) for i in range(1, len(curve))]
    if window:
        drop, i = max(window)
        if drop >= 5:
            mx, my = curve[i][0], curve[i][1]
            regression = {"minute": mx, "from": my + drop, "to": my}
            ax.annotate("regression", xy=(mx, my + 0.01 * N), xytext=(mx, my + 0.3 * N), ha="center", va="bottom", fontsize=FS,
                        color=ROSE, arrowprops=dict(arrowstyle="-|>", lw=0.5, color=ROSE, mutation_scale=5, shrinkA=0, shrinkB=1))
    ax.set_ylim(0, N * 1.22)
    ax.set_yticks([t for t in (0, 100, 200, 300, 400) if t <= N * 1.05])
    ax.set_ylabel("test cases", labelpad=1)
    _title(fig, ax, f"Test cases passed over time (of {N})", W, H)

    # (c) performance on RTL
    sp, ut = ser["sp"], ser["ut"]
    ax = axes["speed"]
    if sp:
        sx = [p[0] for p in sp] + [XMAX]
        g, q1, q3 = ([p[j] for p in sp] + [sp[-1][j]] for j in (1, 2, 3))
        ax.fill_between(sx, q1, q3, step="post", color=NAVY, alpha=0.12, lw=0)
        ax.step(sx, g, where="post", color=NAVY, lw=1.1)
        ax.text(XMAX * 1.025, g[-1], f"×{g[-1]:.2f}", ha="left", va="center", fontsize=FS, color=NAVY, fontweight="bold")
        ax.text(0.02 * XMAX, 3.1, f"speedup vs first certified run · geomean, n = {sp[-1][4]} test cases", ha="left", va="top",
                fontsize=FS, color=INK2)
    else:
        ax.text(0.5 * XMAX, 1.0, "no test case measured twice on RTL", ha="center", va="center", fontsize=FS, color=INK2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.5, 3.6)
    ax.set_yticks([0.5, 1, 2])
    ax.set_yticklabels(["0.5", "1", "2"])
    ax.minorticks_off()
    ax.axhline(1, color=SPINE, lw=0.6)
    ax.set_ylabel("speedup", labelpad=1)
    _title(fig, ax, f"Performance on {lab['l3']} RTL (measured cycles)", W, H)
    ax = axes["util"]
    if ut:
        ux = [p[0] for p in ut] + [XMAX]
        md, a1, a3 = ([p[j] for p in ut] + [ut[-1][j]] for j in (1, 2, 3))
        ax.fill_between(ux, a1, a3, step="post", color=ROSE, alpha=0.16, lw=0)
        ax.step(ux, md, where="post", color=ROSE, lw=1.1)
        ax.text(XMAX * 1.025, md[-1], f"{md[-1]:.1f}%", ha="left", va="center", fontsize=FS, color=ROSE, fontweight="bold")
        ax.set_ylim(0, max(4.0, max(a3) * 1.45))
        ax.text(0.02 * XMAX, ax.get_ylim()[1] * 0.98, f"PE-array utilization (= roofline efficiency) · median, n = {ut[-1][4]}",
                ha="left", va="top", fontsize=FS, color=INK2)
    else:
        ax.set_ylim(0, 4)
        ax.text(0.5 * XMAX, 2, "no bound roofline", ha="center", va="center", fontsize=FS, color=INK2)
    ax.set_ylabel("util. %", labelpad=1)

    # (d) compute on the reserved cores
    ax = axes["comp"]
    sigc = max(1.5, XMAX / 200.0)
    base = np.zeros(n)
    tops = {}
    for k, _, col in compute:
        v = smooth(ser["comp"][k], sigc)
        ax.fill_between(xm, base, base + v, color=col, lw=0, alpha=0.9 if k != "other" else 1.0)
        tops[k] = (base.copy(), base + v)
        base = base + v
    ax.set_ylim(0, cap)
    ax.set_yticks([0, cap // 2, cap])
    ax.set_ylabel("cores", labelpad=1)
    ax.set_xlabel("Time (min of active run)", labelpad=2, fontsize=FS + 0.5, color=INK)
    _title(fig, ax, f"Compute on the {cap} reserved cores", W, H)
    k_last = max(3, n // 25)
    ys = [[float(((tops[k][0] + tops[k][1]) / 2)[-k_last:].mean()), lbl, col] for k, lbl, col in compute]
    for i in range(1, len(ys)):
        ys[i][0] = max(ys[i][0], ys[i - 1][0] + 0.21 * cap)
    over = ys[-1][0] - 0.975 * cap
    if over > 0:
        for row in ys:
            row[0] -= over
    for yv, lbl, col in ys:
        ax.add_patch(plt.Rectangle((XMAX * 1.02, yv - 0.065 * cap), XMAX * 0.025, 0.13 * cap, color=col, clip_on=False, lw=0))
        ax.text(XMAX * 1.06, yv, lbl, ha="left", va="center", fontsize=FS, color=INK, clip_on=False)

    # (e) final status by cohort
    ax = axes["cohort"]
    cohorts = data.get("cohorts") or []
    totals, f2, f3, l3req = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(bool)
    for name, info in cases.items():
        c = info.get("cohort")
        totals[c] += 1
        l3req[c] = l3req[c] or bool(info.get("l3_required"))
    for name, st in ser["final_state"].items():
        c = cases.get(name, {}).get("cohort")
        f2[c] += 1
        f3[c] += st == "L3"
    for i, co in enumerate(cohorts):
        c, Nc = co["id"], max(totals[co["id"]], 1)
        ax.barh(i, 1.0, color=PALE, height=0.66, lw=0)
        ax.barh(i, f3[c] / Nc, color=NAVY, height=0.66, lw=0)
        ax.barh(i, (f2[c] - f3[c]) / Nc, left=f3[c] / Nc, color=SLATE, height=0.66, lw=0)
        ax.text(-0.03, i, co["label"] + ("" if l3req[c] else "*"), ha="right", va="center", fontsize=FS, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(len(cohorts) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=1.5)
    _title(fig, ax, "Final status by cohort", W, H, x_in=0.16)
    hk = [Patch(color=NAVY, label="L3"), Patch(color=SLATE, label="L2 only"), Patch(color=PALE, label="not passing")]
    ax.legend(handles=hk, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=FS, handlelength=1.0,
              handleheight=0.75, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    if not all(l3req[co["id"]] for co in cohorts):
        ax.text(-0.03, -0.17, "* L3 not required", transform=ax.transAxes, ha="right", va="top", fontsize=FS - 0.3, color=INK2)

    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi if fmt == "png" else None)
    plt.close(fig)

    lat = sorted((r["t1"] - r["t0"]) / 60 for r in data.get("selfcheck_latency") or [])
    gl = sorted((g["updated"] - g["started"]) / 60 for g in data["grades"] if g.get("updated") and g.get("started") and g.get("n_completed"))

    def q(xs, f):
        return xs[int(f * (len(xs) - 1))] if xs else None

    return {
        "campaign": data.get("campaign"),
        "stitched_windows": [{"attempt": aid, "start": s, "end": e} for s, e, aid in S.wins],
        "active_minutes": S.total,
        "milestones": [{"label": lbl, "minute": m, "l2": n2, "l3": n3, "tokens": tokens_at(m)} for lbl, (m, n2, n3) in miles],
        "regression": regression,
        "final": {"test_cases": N, "l2": curve[-1][1], "l3": curve[-1][2]},
        "tokens": {"total": float(tot_tok[-1]) if n else 0.0, **{k: float(v[-1]) if n else 0.0 for k, v in cum.items()}},
        "tokens_per_l3_test_case": (float(tot_tok[-1]) / curve[-1][2]) if curve[-1][2] else None,
        "performance": {
            "rtl_rows_used": ser["n_perf"],
            "excluded": data.get("perf_excluded"),
            "speedup_geomean": sp[-1][1] if sp else None,
            "speedup_iqr": [sp[-1][2], sp[-1][3]] if sp else None,
            "speedup_test_cases": sp[-1][4] if sp else 0,
            "utilization_median_pct": ut[-1][1] if ut else None,
            "utilization_iqr_pct": [ut[-1][2], ut[-1][3]] if ut else None,
            "utilization_test_cases": ut[-1][4] if ut else 0,
        },
        "cohorts": {co["id"]: {"label": co["label"], "l2": f2[co["id"]], "l3": f3[co["id"]], "test_cases": totals[co["id"]],
                               "l3_required": l3req[co["id"]]} for co in cohorts},
        "selfcheck_latency_min": {"n": len(lat), "median": q(lat, 0.5), "p90": q(lat, 0.9)},
        "grade_duration_min": {"n": len(gl), "median": q(gl, 0.5), "p90": q(gl, 0.9)},
        "agent_minutes": {k: float(ser["acts"][k].sum() / 60) for k, _, _ in ACTIVITY},
        "compute_normalisation": ser["comp_notes"],
    }
