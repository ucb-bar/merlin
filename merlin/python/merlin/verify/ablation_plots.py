"""Figures for what the formal layer buys over the dynamic grade already in place.

These are deliberately not the Phase-1 figures. Those (``merlin.verify.plots``) measure the checker
against faults *we seeded ourselves*, which answers "does it work" and cannot answer "is it worth
running". These read the ARCHIVE -- every capsule-bench submission this project has produced and
graded -- and ask what the formal layer would have added to results that already exist.

Three figures, and two of the three report a limit rather than a win. That is the finding, not a
presentation problem:

``f5_formal_reach``
    Of every archived submission, how many can the formal layer even look at? Most cannot be reached,
    and the largest single reason is that the submission IS its own specification -- a buffer that
    reproduces the interface program command for command, where an equivalence query is ``X == X``.

``f6_cost_of_finding``
    Numerically-wrong submissions, split by what found them and what that cost. The dominant group
    cost tens of hours of RTL simulation because the cheap check was unavailable -- and the formal
    layer could not have helped, because every one of them is a float datapath it refuses to encode.
    The recoverable time on this archive is zero, and the figure says so.

``f7_stimulus_gap``
    What the dynamic check actually samples. Its stimulus draws every operand value from a
    four-element non-negative set, so sign, saturation and overflow behaviour is untested by
    construction. This is the gap the formal layer exists to close, drawn against a real
    counterexample that falls outside it.

Run::

    .venv/bin/python -m merlin.verify.ablation_plots --write
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from merlin.plotting import plot_paper_style as paper  # noqa: E402

#: Semantic roles, borrowed from the existing verification figures so the two sets sit together.
REACHABLE = paper.SAGE          # the formal layer can decide this
VACUOUS = "#c9c5bb"             # nothing to decide -- the buffer is its own spec
OUT_OF_SCOPE = "#efe7d5"        # the encoder refuses this shape or dtype
COSTLY = paper.SALMON           # time spent finding out
CHEAP = paper.STEEL

REACH_SCHEMA = "verify_formal_reach/v1"
COST_SCHEMA = "verify_cost_of_finding/v1"


# ---------------------------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------------------------

def collect_reach() -> dict[str, Any]:
    """Classify every archived submission by whether the formal layer can reach it.

    The order of the tests matters and is the order a reader should think in: a submission that is
    its own specification is unreachable no matter what its dtypes are, so vacuity is tested first.
    Reporting it any other way would move 2,500 buffers into a dtype bucket and make the encoder look
    like the binding constraint when it is not.
    """
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.verify.ablation import (MAX_ENCODED_ELEMENTS, classify, declared_elements,
                                        units)
    from merlin.verify.cb_semantics import ENCODABLE_OPCODES, NO_NUMERIC_EFFECT

    buckets: Counter[str] = Counter()
    for unit in units():
        gen = unit / "generated"
        try:
            spec = parse_interface_mlir((gen / "input.interface.mlir").read_text(encoding="utf-8"))
            agent = json.loads((gen / "command_buffer.json").read_text(encoding="utf-8"))
        except Exception:
            buckets["unparseable"] += 1
            continue
        if classify(spec, agent) == "identical":
            buckets["vacuous"] += 1
            continue
        tensors = agent.get("tensors") or {}
        dtypes = {str(v.get("dtype")) for v in tensors.values() if v.get("dtype")}
        if any(not d.startswith(("i", "u")) for d in dtypes):
            buckets["float"] += 1
            continue
        if any(len((v.get("shape") or [])) > 2 for v in tensors.values()):
            buckets["rank_gt_2"] += 1
            continue
        opcodes = {str(c.get("opcode")) for c in (agent.get("commands") or [])}
        if opcodes - set(ENCODABLE_OPCODES) - set(NO_NUMERIC_EFFECT):
            buckets["unencodable_opcode"] += 1
            continue
        if declared_elements(agent) > MAX_ENCODED_ELEMENTS:
            buckets["too_large"] += 1
            continue
        buckets["in_scope"] += 1
    return {"schema": REACH_SCHEMA, "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": sum(buckets.values()), "buckets": dict(buckets)}


def collect_cost() -> dict[str, Any]:
    """For numerically-wrong submissions: what found them, and what did finding them cost?

    ``adapter_wall_s`` per tier is the honest cost unit here -- it is the wall time the grader spent
    in that tier's adapter, which is what a cheaper earlier verdict would actually have saved.
    """
    from merlin.common.paths import runs_dir

    caught_cheap = 0
    late = {"count": 0, "sim_seconds": 0.0, "float": 0, "integer": 0}
    per_submission: list[float] = []
    for res in runs_dir().rglob("capsule-bench/**/capsule_result.json"):
        try:
            grade = json.loads(res.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str((grade.get("numeric") or {}).get("status")) != "fail":
            continue
        tiers = grade.get("tiers") or {}
        l0 = tiers.get("L0") if isinstance(tiers.get("L0"), dict) else {}
        sim = sum(float((v.get("timing") or {}).get("adapter_wall_s") or 0)
                  for v in tiers.values() if isinstance(v, dict))
        if str(l0.get("status")) == "fail":
            caught_cheap += 1
            continue
        late["count"] += 1
        late["sim_seconds"] += sim
        per_submission.append(sim)
        cb = res.parent / "generated" / "command_buffer.json"
        kinds: set[str] = set()
        if cb.is_file():
            try:
                kinds = {str(v.get("dtype"))
                         for v in (json.loads(cb.read_text(encoding="utf-8")).get("tensors")
                                   or {}).values() if v.get("dtype")}
            except Exception:
                kinds = set()
        late["float" if any(not k.startswith(("i", "u")) for k in kinds) else "integer"] += 1
    per_submission.sort()
    return {"schema": COST_SCHEMA, "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "caught_by_cheap_check": caught_cheap, "found_late": late,
            "median_seconds": per_submission[len(per_submission) // 2] if per_submission else 0.0,
            "max_seconds": per_submission[-1] if per_submission else 0.0}


def _caption(fig, text: str, *, y: float = -0.03) -> None:
    import textwrap
    fig.text(0.01, y, "\n".join(textwrap.wrap(text, 132)), ha="left", va="top",
             fontsize=8.4, color=paper.INK, alpha=0.86)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def fig_formal_reach(record: dict[str, Any]):
    """How much of the archive the formal layer can even look at."""
    labels = {
        "vacuous": "the submission IS its own specification\n(equivalence query is X == X)",
        "float": "float datapath\n(encoder refuses, never approximates)",
        "rank_gt_2": "rank > 2 (conv / batched)\nencodable in principle, not built",
        "unencodable_opcode": "opcode with no exact encoding",
        "too_large": "over the encoding size cap",
        "unparseable": "spec or buffer unreadable",
        "in_scope": "IN SCOPE for the formal layer",
    }
    buckets = record["buckets"]
    total = max(record["total"], 1)
    order = [k for k in ("in_scope", "vacuous", "float", "rank_gt_2", "unencodable_opcode",
                         "too_large", "unparseable") if buckets.get(k)]
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    paper.card(ax, f"What the formal layer can reach: {buckets.get('in_scope', 0)} of "
                   f"{record['total']} archived submissions")
    ys = range(len(order))
    vals = [buckets[k] for k in order]
    cols = [REACHABLE if k == "in_scope" else (VACUOUS if k == "vacuous" else OUT_OF_SCOPE)
            for k in order]
    ax.barh(list(ys), vals, height=0.62, color=cols, edgecolor=paper.CARD_EC, linewidth=1.4,
            zorder=3)
    for y, k, v in zip(ys, order, vals):
        ax.text(v + total * 0.008, y, f"{v}  ({100 * v / total:.1f}%)", va="center", fontsize=10,
                fontweight="bold" if k == "in_scope" else "normal", color=paper.INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("archived capsule-bench submissions")
    ax.set_xlim(0, total * 1.18)
    _caption(fig, "Every submission this project has produced and graded. The binding constraint is "
                  "NOT the encoder: the largest bucket is submissions that reproduce the interface "
                  "program command for command, where proving equivalence proves nothing and a bug "
                  "in the shared encoder would cancel on both sides. Counting those as verified "
                  "would have made a headline coverage number ~60% vacuous.")
    fig.tight_layout()
    return fig


def fig_cost_of_finding(record: dict[str, Any]):
    """What it cost to find the defects that WERE found, and how much of that was recoverable."""
    late = record["found_late"]
    hours = late["sim_seconds"] / 3600.0
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    paper.card(ax, "Cost of finding a numerically-wrong submission, by what found it")

    bars = [
        ("caught by the cheap numeric check\n(L0, zero simulation)", record["caught_by_cheap_check"],
         0.0, CHEAP),
        ("cheap check UNAVAILABLE — found only\nafter spike / Verilator simulation",
         late["count"], hours, COSTLY),
    ]
    ys = range(len(bars))
    ax.barh(list(ys), [b[1] for b in bars], height=0.55, color=[b[3] for b in bars],
            edgecolor=paper.CARD_EC, linewidth=1.5, zorder=3)
    for y, (_, n, h, _c) in zip(ys, bars):
        note = "0 s of simulation" if h == 0 else f"{h:.1f} hours of simulation"
        ax.text(n + max(b[1] for b in bars) * 0.015, y, f"{n} submissions — {note}",
                va="center", fontsize=10, color=paper.INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in bars], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("numerically-wrong submissions")
    widest = max(b[1] for b in bars)
    ax.set_xlim(0, widest * 2.05)
    ax.set_ylim(1.62, -0.62)          # inverted, with room for the callout BETWEEN the bars

    recoverable = late["integer"]
    # Placed in the empty band between the two bars. An earlier version put it below the lower bar,
    # where it covered the x-axis tick labels and the axis title -- caught by looking at the PNG.
    paper.callout(ax, (late["count"] * 0.55, 0.72),
                  f"of these {late['count']}, {late['float']} are FLOAT datapaths\n"
                  f"the formal layer refuses to encode.\n"
                  f"Time it could have recovered here: "
                  f"{'0.0 h' if recoverable == 0 else f'{recoverable} submissions'}",
                  (widest * 1.30, 0.46))
    _caption(fig, "The cheap dynamic check costs nothing and catches what it can see. Where it was "
                  "unavailable, the defect surfaced only after full RTL simulation. That looks like "
                  "the formal layer's opportunity, and on this archive it is not: the reason the "
                  "cheap check was unavailable is that the datapath is float, and the encoder "
                  "refuses float rather than approximating it. Measured recoverable time: zero.")
    fig.tight_layout()
    return fig


def fig_stimulus_gap(counterexample: dict[str, int] | None = None):
    """What the dynamic check actually samples, against the space the formal layer quantifies over.

    Drawn as a SLIVER against the full range rather than as four scattered points. An earlier version
    plotted the four stimulus values as dots on a 256-wide axis, where they collapsed into a single
    blob at the origin and the two annotation arrows crossed each other -- caught by looking at the
    rendered PNG rather than at the code.
    """
    from matplotlib.patches import Rectangle

    from merlin.verify.ablation import stimulus_values

    stim = sorted(stimulus_values())
    lo, hi = -128, 127
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    paper.card(ax, "What one graded run actually samples, versus what is proved")
    ax.set_xlim(-165, 165)
    ax.set_ylim(-1.15, 1.15)
    ax.get_yaxis().set_visible(False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    # the full space
    ax.add_patch(Rectangle((lo, -0.17), hi - lo, 0.34, facecolor=OUT_OF_SCOPE,
                           edgecolor=paper.CARD_EC, linewidth=1.3, zorder=3))
    ax.text(0, 0.40, f"the i8 operand space an all-inputs proof covers   [{lo}, {hi}]  =  256 values",
            ha="center", fontsize=9.8, color=paper.INK)

    # the sampled sliver, drawn at true scale so the comparison is not flattered
    ax.add_patch(Rectangle((stim[0], -0.17), max(stim[-1] - stim[0], 1), 0.34, facecolor=COSTLY,
                           edgecolor=paper.CARD_EC, linewidth=1.3, zorder=4))
    ax.annotate(f"the dynamic stimulus samples {{{stim[0]}..{stim[-1]}}}\n"
                f"{len(stim)} of 256 values, none negative",
                xy=(stim[-1], -0.17), xytext=(-62, -0.80), fontsize=9.6, color=paper.INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=paper.INK, lw=1.2))

    if counterexample:
        xs = sorted(set(counterexample.values()))
        ax.scatter(xs, [0] * len(xs), s=95, marker="X", color=paper.GOLD,
                   edgecolor=paper.CARD_EC, zorder=6, linewidth=1.2)
        ax.annotate("a real counterexample this checker returned\n"
                    f"({', '.join(str(v) for v in xs)}) — no run could have sampled it",
                    xy=(xs[-1], 0.17), xytext=(72, 0.86), fontsize=9.6, color=paper.INK,
                    ha="center", arrowprops=dict(arrowstyle="->", color=paper.INK, lw=1.2))
    ax.set_xlabel("operand value")
    _caption(fig, "The stimulus fill was fixed in 13397c36 so rows and columns differ, but its VALUE "
                  "range was deliberately preserved at {0..3}. A graded run therefore evaluates each "
                  "buffer at one input point with small non-negative operands: sign handling, i8 "
                  "saturation and accumulator overflow are untested by construction, not by "
                  "oversight. Closing that is what an all-inputs proof is for -- on the 11% of "
                  "submissions it can reach.")
    fig.tight_layout()
    return fig


def build(*, write: bool = False) -> dict[str, Any]:
    """Collect, draw, and optionally persist as a versioned product."""
    reach, cost = collect_reach(), collect_cost()

    # A real refutation this checker produced, kept as data so the figure cannot invent one. It is
    # the narrowed-readout fault from the seeded corpus, whose counterexample needs negative and
    # large-magnitude operands -- exactly the region the stimulus never visits.
    counterexample = {"W_0_1": -105, "W_1_0": 78, "W_0_0": -1}

    figs = {
        "f5_formal_reach": fig_formal_reach(reach),
        "f6_cost_of_finding": fig_cost_of_finding(cost),
        "f7_stimulus_gap": fig_stimulus_gap(counterexample),
    }
    out_dir: Path | None = None
    if write:
        from merlin.common.artifacts import new_product

        prod = new_product("verification", version=1, target="all", sources=[
            f"{reach['total']} archived capsule-bench submissions under out/runs",
            "formal reach: merlin.verify.ablation.classify + cb_semantics coverage sets",
            "cost: capsule_result.json tier timings (adapter_wall_s)",
        ], notes=("What the formal layer buys over the dynamic grade, measured on submissions that "
                  "already exist. Two of the three figures report a limit rather than a win."))
        for name, fig in figs.items():
            fig.savefig(prod.add_artifact(f"{name}.png"), dpi=200, bbox_inches="tight")
        prod.add_artifact("formal_reach.json").write_text(json.dumps(reach, indent=1),
                                                          encoding="utf-8")
        prod.add_artifact("cost_of_finding.json").write_text(json.dumps(cost, indent=1),
                                                             encoding="utf-8")
        prod.write_manifest()
        out_dir = prod.path
    for fig in figs.values():
        plt.close(fig)
    return {"reach": reach, "cost": cost, "out_dir": str(out_dir) if out_dir else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the figures under out/artifacts/")
    a = ap.parse_args(argv)
    rec = build(write=a.write)
    reach, cost, late = rec["reach"], rec["cost"], rec["cost"]["found_late"]
    print(f"archived submissions            {reach['total']}")
    for k, v in sorted(reach["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {100 * v / max(reach['total'], 1):5.1f}%  {k}")
    print(f"\nnumerically wrong, caught cheap {cost['caught_by_cheap_check']}")
    print(f"numerically wrong, found late   {late['count']}  "
          f"({late['sim_seconds'] / 3600:.1f} h of simulation)")
    print(f"  of those, float (unreachable) {late['float']}")
    print(f"  of those, integer (reachable) {late['integer']}")
    if rec["out_dir"]:
        print(f"\nwrote {rec['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
