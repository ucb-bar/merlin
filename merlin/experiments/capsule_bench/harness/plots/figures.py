#!/usr/bin/env python3
"""Publication figures for the multi-model / multi-harness capsule-bench comparison.

Every number is read from ``report_data.json`` (produced by ``report_data.py``) — nothing is typed into
a figure by hand, so a figure cannot drift from the run that produced it.

House style, deliberately stripped back from the denser DSE presentation look: muted academic palette,
large type, one message per figure, no hatching and no callout clutter. Each figure writes BOTH a vector
SVG (for a paper) and a 300-dpi PNG (for slides).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

# ---------------------------------------------------------------- palette + theme
GREEN   = "#7f9e6b"   # passed
GREEN_L = "#a9c08a"
SALMON  = "#cf7f77"   # wrong hardware encoding (the compute-plane wall)
SALMON_L= "#e5a9a2"
AMBER   = "#dfae4e"   # wrong numerics
SLATE   = "#8e959d"   # never reached a tier (contract plane)
SLATE_L = "#c3c8cd"
BLUE    = "#5b87ab"
BLUE_L  = "#9dbdd4"
PLUM    = "#8f6f9e"
INK     = "#23262a"
MUTED   = "#6b7176"
CREAM   = "#f6f2ea"
RULE    = "#d8d4cc"


def theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.28,
        "font.size": 14, "axes.titlesize": 18, "axes.titleweight": "bold",
        "axes.labelsize": 14.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "legend.frameon": False,
        "axes.edgecolor": RULE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True,
        "grid.color": "#ecebe7", "grid.linewidth": 0.9,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.constrained_layout.use": False,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}.svg + .png")


# ---------------------------------------------------------------- cell selection
# One representative run per (model x harness). Where a harness defect degraded some runs of the same
# pairing, the CLEAN run is chosen and the choice is visible in Figure 7.
# One representative run per (model x harness). Where a harness defect degraded some runs of the same
# pairing, the CLEAN run is chosen and the choice is visible in Figure 7. The two Opus cells are placed
# adjacent on purpose: same model, same bundle, same corpus, different harness -- the natural experiment
# that separates the model from the tooling it was driven through.
CELLS = [
    ("merlincirct_gemarm4_codex2",   "GPT-5.6-sol\nCodex"),
    ("merlincirct_gemarm4_codex3",   "GPT-5.6-sol\nCodex (run 3)"),
    ("merlincirct_ctl_opus_claude",  "Opus 5\nClaude Code"),
    ("merlincirct_hxm_opus_oc5",     "Opus 5\nopencode"),
    ("merlincirct_hxm_glm_claude",   "GLM-5\nClaude Code"),
    ("merlincirct_hxm_glm_codex",    "GLM-5\nCodex"),
    ("merlincirct_hxm2_glm_opencode","GLM-5\nopencode"),
    ("merlincirct_hxm_nemo_codex",   "Nemotron\nCodex"),
    ("merlincirct_hxm3_nemo_opencode","Nemotron\nopencode"),
]

#: Model colour, used wherever a cell is drawn as a line or a point.
MODEL_COLOUR = {"GPT-5.6-sol": GREEN, "Opus 5": BLUE, "GLM-5": SALMON, "Nemotron": SLATE}


def by_run(data: dict) -> dict:
    return {c["run"]: c for c in data["cells"]}


def _pick(data: dict) -> list[tuple[dict, str]]:
    R = by_run(data)
    return [(R[r], lbl) for r, lbl in CELLS if r in R]


# ---------------------------------------------------------------- Figure 1 — tier matrix
def fig_tier_matrix(data: dict, out: Path) -> None:
    """Where each capsule dies, for every model x harness cell. The three regimes read off at a glance."""
    cells = _pick(data)
    caps = [r["capsule"] for r in cells[0][0]["public"]]
    code = {"pass": (GREEN, "passed"), "L2": (SALMON, "wrong HW encoding  (L2 spike)"),
            "L1": (SALMON_L, "L1"), "L0": (AMBER, "wrong numerics  (L0)"),
            "L3": (BLUE_L, "wrong on RTL  (L3)"), "-": (SLATE_L, "no tier ran  (contract plane)")}
    fig, ax = plt.subplots(figsize=(13.6, 9.2))
    for j, (c, _) in enumerate(cells):
        ff = c["first_fail_tier"]
        for i, cap in enumerate(caps):
            v = ff.get(cap, "-")
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.92, facecolor=code.get(v, (SLATE_L,))[0],
                                       edgecolor="white", linewidth=1.6))
    ax.set_xlim(-0.05, len(cells)); ax.set_ylim(-0.1, len(caps))
    ax.set_xticks([j + 0.47 for j in range(len(cells))])
    ax.set_xticklabels([l for _, l in cells], fontsize=12.5, linespacing=1.5)
    ax.set_yticks([i + 0.46 for i in range(len(caps))])
    ax.set_yticklabels([c.split("_", 1)[0] + "  " + c.split("_", 1)[1].replace("_", " ")
                        for c in caps], fontsize=11.5)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Every capsule, every model, where it dies", pad=22)
    # Legend order follows the ladder, and lists only outcomes this figure actually draws — the L3 band
    # in particular IS the story of sol's third run and must never be filtered out of the key.
    present = {v for c, _ in cells for v in c["first_fail_tier"].values()}
    handles = [Patch(facecolor=code[v][0], label=code[v][1])
               for v in ("pass", "L0", "L2", "L3", "-") if v in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles), fontsize=12.5)
    save(fig, out, "fig1_tier_matrix")


# ---------------------------------------------------------------- Figure 2 — ladder funnel
def fig_ladder(data: dict, out: Path) -> None:
    """How far up the oracle ladder each cell gets. L0/L1 only read the command buffer and barely
    discriminate; L2 executes the real instruction encoding and is where the field separates.

    Drawn as slopes rather than grouped bars because the MESSAGE is the drop between rungs — and because
    sol's run 3 holds 20 through L2 and then collapses to 1 at the RTL tier, which a bar cluster hides."""
    cells = _pick(data)
    tiers = ["L0", "L1", "L2", "L3"]
    names = ["L0\nnumerics", "L1\ncb self-\nconsistency", "L2  · spike\nthe pass bar",
             "L3  · RTL\nadvisory"]
    # colour carries the MODEL, dash pattern carries the HARNESS — both axes readable at once
    mcol = MODEL_COLOUR
    hdash = {"Codex": (0, ()), "Codex (run 3)": (0, (1, 1.4)), "opencode": (0, (6, 3)),
             "Claude Code": (0, (2, 2, 7, 2))}
    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    # Opus 5 and GLM-5 x Codex trace EXACTLY the same line (17, 17, 0, 0). That coincidence is the
    # central result, so it is drawn as a deliberate small offset and named, never left as one curve
    # silently painted over another.
    nudge = {"merlincirct_hxm_opus_oc5": 0.42}   # separates two curves that coincide exactly
    for c, lbl in cells:
        model, harness = lbl.split("\n")
        tr = c["tier_reached"]
        ys = [tr.get(t, 0) + nudge.get(c["run"], 0) for t in tiers]
        ax.plot(range(len(tiers)), ys, color=mcol.get(model, SLATE), lw=3.4,
                linestyle=hdash.get(harness, (0, ())), marker="o", markersize=11,
                markeredgecolor="white", markeredgewidth=2.2, zorder=3, solid_capstyle="round")
    ax.axhline(20, color=RULE, lw=1.1, ls=(0, (4, 5)), zorder=1)
    for c, lbl in cells:
        y3 = c["tier_reached"].get("L3", 0)
        if lbl.startswith("GPT-5.6-sol"):
            ax.annotate(f"{lbl.split(chr(10))[1]} \u2192 {y3}/20 on RTL", (3, y3),
                        textcoords="offset points",
                        xytext=(-18, 16) if y3 > 10 else (-6, -34),
                        ha="right", fontsize=13, color=GREEN if y3 > 10 else INK,
                        fontweight="bold")
    ax.text(2.06, 16.0, "Opus 5, native harness", fontsize=13, color=BLUE, fontweight="bold")
    ax.text(0.06, 7.4, "everyone else computes the right\nanswer and encodes it wrong",
            fontsize=13.5, color=SALMON, ha="left", va="center", linespacing=1.5, fontweight="bold")
    ax.text(0.06, 1.35, "never produced a runnable artifact", fontsize=13, color=SLATE)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(names, fontsize=13.5, linespacing=1.7)
    ax.set_xlim(-0.28, 3.42); ax.set_ylim(-1.2, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("capsules passing this tier")
    ax.grid(axis="y", visible=True)
    ax.set_title("Everything is decided at L2, where the encoding first runs", pad=22)
    handles = [plt.Line2D([], [], color=v, lw=3.4, label=k) for k, v in mcol.items()]
    handles += [plt.Line2D([], [], color=MUTED, lw=2.2, linestyle=v, label=k)
                for k, v in hdash.items() if k != "Codex (run 3)"]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=12.5)
    save(fig, out, "fig2_ladder")


# ---------------------------------------------------------------- Figure 3 — failure planes
def fig_planes(data: dict, out: Path) -> None:
    """What KIND of failure. A contract failure never reached the compiler's arithmetic at all."""
    cells = _pick(data)
    order = [("pass", "passed", GREEN), ("compute", "wrong code produced", SALMON),
             ("contract", "never produced a runnable artifact", SLATE),
             ("crash:model", "own tool / own IR crashed", AMBER),
             ("crash:harness", "harness fault", BLUE_L)]
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    ys = range(len(cells))
    for y, (c, lbl) in zip(ys, cells):
        left = 0
        for key, _, col in order:
            v = c["kinds"].get(key, 0)
            if not v:
                continue
            ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.4, height=0.68)
            if v >= 2:
                ax.text(left + v / 2, y, str(v), ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
            left += v
    ax.set_yticks(list(ys))
    ax.set_yticklabels([l.replace("\n", " · ") for _, l in cells], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("public capsules  (20)")
    ax.set_xlim(0, 20.4); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title("Nemotron never reaches the arithmetic; GLM-5 and Opus 5 do", pad=20)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in order],
              ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=12.5)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out, "fig3_failure_planes")


# ---------------------------------------------------------------- Figure 4 — tooling conformance
def fig_tooling(data: dict, out: Path) -> None:
    """Did the model USE the RTL-derived tooling the arm grants it? The one sol run that skipped the
    ISA tools is the one whose compiler failed on held-out capsules."""
    cells = _pick(data)
    checks = [("isa_tools_used", "ran the RTL\nISA tools"), ("cca_used", "enumerated the\nlever set"),
              ("no_regex_ok", "no regex in\nits compiler"), ("full_selfcheck", "self-checked all\ncapsules")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.6), gridspec_kw={"width_ratios": [2.5, 1]})
    for j, (c, lbl) in enumerate(cells):
        ever = c["conformance"]["ever"]
        for i, (k, _) in enumerate(checks):
            v = ever.get(k)
            mark = "\u2713" if v else ("\u2013" if v is None else "\u2717")
            col = GREEN if v else (SLATE_L if v is None else SALMON)
            # The no-regex scan reads every .py in the submission, including one the model VENDORED
            # from merlin's own granted tooling -- and merlin's interface_emit.py is regex-based. A
            # model flagged for merlin's regex is a false positive, not a violation, and gets its own
            # state rather than being scored either way.
            if k == "no_regex_ok" and v is False and c["conformance"].get("regex_only_in_vendored"):
                col, mark = AMBER, "!"
            ax.add_patch(plt.Rectangle((i, j), 0.9, 0.82, facecolor=col,
                                       edgecolor="white", linewidth=2))
            ax.text(i + 0.45, j + 0.41, mark, ha="center", va="center",
                    color="white", fontsize=17, fontweight="bold")
    ax.set_xlim(-0.05, len(checks)); ax.set_ylim(-0.1, len(cells))
    ax.set_xticks([i + 0.45 for i in range(len(checks))])
    ax.set_xticklabels([n for _, n in checks], fontsize=12.5, linespacing=1.5)
    ax.set_yticks([j + 0.41 for j in range(len(cells))])
    ax.set_yticklabels([l.replace("\n", " · ") for _, l in cells], fontsize=12.5)
    ax.invert_yaxis(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which granted tooling each model actually ran", pad=18, fontsize=17)

    hid = [c["hidden_passed"] for c, _ in cells]
    ax2.barh(range(len(cells)), hid, color=[GREEN if h == 5 else (AMBER if h else SLATE_L) for h in hid],
             height=0.62, edgecolor="white", linewidth=1.2)
    for j, h in enumerate(hid):
        ax2.text(h + 0.16, j, f"{h}/5", va="center", fontsize=13, color=INK)
    ax2.set_yticks(range(len(cells))); ax2.set_yticklabels([])
    ax2.invert_yaxis(); ax2.set_xlim(0, 6.4); ax2.set_xticks([0, 5])
    ax2.set_xlabel("held-out capsules passed")
    ax2.set_title("Generalization", pad=18, fontsize=17)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig4_tooling")


# ---------------------------------------------------------------- Figure 5 — architecture
def fig_architecture(data: dict, out: Path) -> None:
    """What was actually SHIPPED, by the role each module plays.

    A raw file count is the wrong measure and would invert the result: the Nemotron x Codex submission
    ships eight files, more than the winning run, but seven of them are ``parse_fixed``, ``parse_fixed2``,
    ``emit_improved``, ``debug_tensors`` and ad-hoc test scripts — the trail of a model thrashing, frozen
    into the deliverable. Classified by role, the split is clean: every passing run built the dialect +
    lowering structure this arm exists to ask for; the failing runs put everything in one file and dropped
    the generated encoder beside it."""
    cells = _pick(data)
    roles = [("structure", "dialect + lowering passes it wrote", GREEN),
             ("generated", "RTL-generated encoder", BLUE_L),
             ("vendored", "merlin's own file, copied", PLUM),
             ("monolith", "one catch-all file", SLATE),
             ("debris", "abandoned iterations left in the submission", AMBER)]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ys = list(range(len(cells)))
    for y, (c, lbl) in zip(ys, cells):
        left = 0
        for key, _, col in roles:
            v = len(c["submission"]["kinds"].get(key, []))
            if not v:
                continue
            ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.6, height=0.66)
            if v >= 2:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white",
                        fontsize=13, fontweight="bold")
            left += v
        ax.text(left + 0.22, y, f"{c['submission']['n_lines']:,} lines", va="center",
                fontsize=12, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{l.replace(chr(10), ' · ')}" for _, l in cells], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("modules in the frozen submission")
    ax.set_xlim(0, 13.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_title("Every run that passed authored a dialect. No run that failed did.", pad=20)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in roles],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=12)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=True)
    save(fig, out, "fig5_architecture")


# ---------------------------------------------------------------- Figure 6 — diagnostic fidelity
def fig_diagnostics(data: dict, out: Path) -> None:
    """What the agent was actually SHOWN when a capsule failed.

    Two independent damage modes, both measured on FAILING capsules only (a passing capsule has no
    diagnostic to damage): the digit scrubber rewrote every numeral, so ``tensor<16x16xi8>`` arrived as
    ``tensor<#x#xi#>`` and ``rc=0`` as ``rc=#``; and a whole class of tool failures carried no stderr at
    all, reaching the model as a ~26-character non-message. Both are ways of telling a model it is wrong
    without telling it what is wrong."""
    cells = _pick(data)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    labs, red, emp, tot = [], [], [], []
    for c, lbl in cells:
        d = c["diagnostics"]
        n = d["verdicts"]
        labs.append(lbl.replace("\n", " \u00b7 "))
        tot.append(n)
        # split the union into "numerals destroyed" and "no content at all" so neither mode hides
        e = d["near_empty"]
        labs_r = d["unusable"] - e
        red.append(100 * max(labs_r, 0) / n if n else 0)
        emp.append(100 * e / n if n else 0)
    ys = list(range(len(labs)))
    ax.barh(ys, red, color=SALMON, height=0.64, edgecolor="white", linewidth=1.2,
            label="numerals destroyed  (tensor<#x#xi#>)")
    ax.barh(ys, emp, left=red, color=AMBER, height=0.64, edgecolor="white", linewidth=1.2,
            label="no content at all  (empty stderr)")
    for y, (a, b, n) in enumerate(zip(red, emp, tot)):
        v = a + b
        txt = f"{v:.0f}%" + (f"   of {n} failing capsules" if n else "   converged \u2014 never needed it")
        ax.text(min(v + 1.8, 102), y, txt, va="center", fontsize=12.5, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=13)
    ax.invert_yaxis(); ax.set_xlim(0, 152); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of the model's own error messages it could not read")
    ax.set_title("The feedback channel was broken only for models that needed it", pad=20)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=True)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=12.5)
    save(fig, out, "fig6_diagnostics")


# ---------------------------------------------------------------- Figure 7 — confound map
def fig_confounds(data: dict, out: Path) -> None:
    """Which cells are clean enough to carry a claim, and what contaminated the rest. Nothing in the
    report may rest on a cell this figure marks red."""
    R = by_run(data)
    rows = [(r, l) for r, l in CELLS if r in R]
    issues = [("harness config delivered", lambda c: c["harness_health"]["config_delivered"] is not False),
              ("no bridge-dropped rounds", lambda c: c["harness_health"]["bridge_toolconfig_400s"] == 0),
              ("every round priced", lambda c: c["harness_health"]["rounds_with_no_usage_record"] == 0),
              ("no harness-fault capsules", lambda c: c["kinds"].get("crash:harness", 0) == 0),
              ("readable diagnostics", lambda c: (c["diagnostics"]["verdicts"] == 0 or
                                                  c["diagnostics"]["redacted"] / c["diagnostics"]["verdicts"] < 0.05))]
    fig, ax = plt.subplots(figsize=(13.4, 6.8))
    for j, (run, lbl) in enumerate(rows):
        c = R[run]
        for i, (_, test) in enumerate(issues):
            ok = bool(test(c))
            ax.add_patch(plt.Rectangle((i, j), 0.9, 0.82, facecolor=GREEN if ok else SALMON,
                                       edgecolor="white", linewidth=2))
            ax.text(i + 0.45, j + 0.41, "✓" if ok else "✗", ha="center", va="center",
                    color="white", fontsize=17, fontweight="bold")
    ax.set_xlim(-0.05, len(issues)); ax.set_ylim(-0.1, len(rows))
    ax.set_xticks([i + 0.45 for i in range(len(issues))])
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n, _ in issues], fontsize=12.5, linespacing=1.6)
    ax.set_yticks([j + 0.41 for j in range(len(rows))])
    ax.set_yticklabels([l.replace("\n", " · ") for _, l in rows], fontsize=12.5)
    ax.invert_yaxis(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which cells can carry a claim", pad=20)
    save(fig, out, "fig7_confounds")


# ---------------------------------------------------------------- Figure 8 — efficiency
def fig_efficiency(data: dict, out: Path) -> None:
    """Work done per unit of result, with the caching asymmetry made visible: the bridged runs get no
    prompt cache at all, so their token and dollar figures are not comparable to a native run."""
    cells = _pick(data)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.6))
    labs = [l.replace("\n", " · ") for _, l in cells]
    inp = [(c["tokens"].get("tokens_input") or 0) / 1e6 for c, _ in cells]
    cac = [(c["tokens"].get("tokens_cached") or 0) / 1e6 for c, _ in cells]
    ys = range(len(labs))
    ax.barh(list(ys), inp, color=SALMON, height=0.36, label="fresh input", edgecolor="white")
    ax.barh([y + 0.38 for y in ys], cac, color=BLUE_L, height=0.36, label="served from cache",
            edgecolor="white")
    ax.set_yticks([y + 0.19 for y in ys]); ax.set_yticklabels(labs, fontsize=12)
    ax.invert_yaxis(); ax.set_xlabel("million tokens")
    ax.set_title("Prompt cache: native vs bridged", pad=16, fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=True)

    tools = [c["tokens"].get("tool_calls") or 0 for c, _ in cells]
    ax2.barh(list(ys), tools, color=[GREEN if c["n_passed"] == 20 else SLATE for c, _ in cells],
             height=0.62, edgecolor="white", linewidth=1.2)
    for y, v in enumerate(tools):
        ax2.text(v + 12, y, str(v), va="center", fontsize=12.5, color=INK)
    ax2.set_yticks(list(ys)); ax2.set_yticklabels([])
    ax2.invert_yaxis(); ax2.set_xlabel("tool calls")
    ax2.set_title("Actions spent", pad=16, fontsize=16)
    ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", visible=True)
    fig.subplots_adjust(wspace=0.06)
    save(fig, out, "fig8_efficiency")


# ---------------------------------------------------------------- Figure 9 — the natural experiment
def fig_natural_experiment(data: dict, out: Path) -> None:
    """The one comparison that separates the model from the harness it was driven through.

    Same model, same bundle, same corpus, same sandbox, same FIRST round -- only the agentic harness
    differs. All three cells compute identically at the command-buffer tiers (17 of 20 at both L0 and
    L1), so the divergence is entirely in the hardware encoding. Opus 5 clears it on its native harness
    and does not clear it on a foreign one; GLM-5 does not clear it on the harness where Opus does."""
    R = by_run(data)
    cells = [("merlincirct_hxm_opus_oc5", "Opus 5\nopencode", BLUE_L),
             ("merlincirct_ctl_opus_claude", "Opus 5\nClaude Code", BLUE),
             ("merlincirct_hxm_glm_claude", "GLM-5\nClaude Code", SALMON)]
    tiers = ["L0", "L1", "L2", "L3"]
    names = ["L0\nnumerics", "L1\nconsistency", "L2 · spike\nthe pass bar", "L3 · RTL\ncycle-accurate"]
    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    n = len(cells)
    w = 0.74 / n
    for j, (run, lbl, col) in enumerate(cells):
        c = R.get(run)
        if not c:
            continue
        tr = c["tier_reached"]
        vals = [tr.get(t, 0) for t in tiers]
        xs = [i + j * w - 0.37 + w / 2 for i in range(len(tiers))]
        ax.bar(xs, vals, width=w * 0.9, color=col, edgecolor="white", linewidth=1.4,
               label=lbl.replace("\n", " · "), zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.55, str(v), ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.axvline(1.5, color=RULE, lw=1.4, ls=(0, (4, 4)), zorder=1)
    ax.text(1.56, 21.4, "the command buffer is identical  \u2192  everything diverges here",
            fontsize=13, color=MUTED, ha="left")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(names, fontsize=13.5, linespacing=1.7)
    ax.set_ylim(0, 23.5); ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("capsules passing this tier")
    ax.grid(axis="y", visible=True)
    ax.set_title("Same model, same task, different harness", pad=22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=13)
    save(fig, out, "fig9_natural_experiment")


def main() -> None:
    src = Path(sys.argv[1]); out = Path(sys.argv[2])
    data = json.loads(src.read_text())
    theme()
    print("figures:")
    for fn in (fig_tier_matrix, fig_ladder, fig_planes, fig_tooling,
               fig_architecture, fig_diagnostics, fig_confounds, fig_efficiency,
               fig_natural_experiment):
        fn(data, out)


if __name__ == "__main__":
    main()
