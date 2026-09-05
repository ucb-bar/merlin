"""The four figures that report the verification layer, each drawn from a JSON record.

Why these four, and why in this shape:

* **F1 detection matrix** is the headline, because the claim "three layers, and none subsumes
  another" is exactly a claim about *which layer catches which defect*. A bar chart of test counts
  would not distinguish a layer that earns its cost from one that duplicates the layer below it. If
  the claim were false the figure would show it immediately: a column that is detected-everywhere is
  a redundant layer, and a row that is missed-everywhere is a defect class nothing here covers.
  The RTL column is drawn as a hatched *not measured* band rather than left out, because a layer we
  did not run must be visible as unmeasured -- an absent column reads as "no defects there".
* **F2 cost-to-detect** answers the other half: a layer is worth its place only if what it catches
  justifies what it charges. Log scale, because the layers are three orders of magnitude apart and a
  linear axis would render the cheap layer as a zero-width bar -- i.e. would hide the actual result.
* **F3 obligation coverage** is the honest-denominator figure. Its point is the *omitted* stack: a
  suite that only plots what it checks is the failure mode this whole layer exists to prevent. Every
  omitted obligation carries its recorded reason into the caption.
* **F4 formal scaling** bounds the formal layer. It is log-log because a solver cost curve is only
  interpretable as a slope; the mesh-tile point is annotated because "one real hardware tile is
  inside the tractable region" is the load-bearing claim, and it is annotated from the target's
  DERIVED mesh edge, never from a typed-in shape.

Nothing in the drawing code knows a measurement. Each figure takes a record dict, and every number
and every target name it prints is read out of that record at plot time; the collectors below are
what produce the records, by running the real harnesses. A number typed into a plot is a number that
cannot go stale, which is precisely why it must not be here.

CLI::

    python -m merlin.verify.plots                 # collect fresh, write a versioned product
    python -m merlin.verify.plots --from <dir>    # re-draw from records already collected
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# The repo's paper style, imported rather than re-declared: a second palette is a second thing to
# keep in sync, and the figures are meant to sit beside the existing ones without looking foreign.
from merlin.plotting import plot_paper_style as paper  # noqa: E402

DETECTION_SCHEMA = "verify_detection_matrix/v1"
COVERAGE_SCHEMA = "verify_obligation_coverage/v1"
SCALING_SCHEMA = "verify_formal_scaling/v1"

DETECTION_FILE = "detection_matrix.json"
COVERAGE_FILE = "obligation_coverage.json"
SCALING_FILE = "formal_scaling.json"

#: Semantic colours, taken from the paper palette. Detection is the good outcome, so it gets the
#: sage; a miss is a pale card tone rather than an alarm colour, because a miss is expected of a
#: layer that does not target that defect class; unmeasured is grey AND hatched, so it can never be
#: mistaken for a measured result even in greyscale print.
DETECTED = paper.SAGE
MISSED = "#efe7d5"
UNMEASURED = "#c9c5bb"
UNMEASURED_HATCH = "///"
#: A layer that RAN but could not decide (solver timeout, missing tool). Deliberately hatched like
#: the unmeasured column rather than shaded like a miss: in greyscale a pale "miss" and a pale
#: "abstain" are the same cell, and that difference is the whole honesty of the figure.
ABSTAINED = "#e2d9c4"
ABSTAINED_HATCH = "xx"

#: The shapes the formal layer is timed at. These are the experiment's INPUTS (a sweep plan), not
#: results -- the solve times and verdicts all come back from the solver. The ladder is cubes plus
#: two rectangular points, so the curve cannot be read as an artefact of square shapes alone.
DEFAULT_SCALING_SHAPES: tuple[tuple[int, int, int], ...] = (
    (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (16, 32, 16), (32, 32, 32), (64, 16, 64),
)


# ---------------------------------------------------------------------------------------------
# collectors -- these run the real harnesses and emit the records the figures read
# ---------------------------------------------------------------------------------------------

def collect_detection(*, m: int = 4, k: int = 4, n: int = 4, reuse: int = 2,
                      timeout_ms: int = 60_000) -> dict[str, Any]:
    """Run every seeded fault past every layer. Returns the ``verify_detection_matrix/v1`` record."""
    from .evaluate import run_matrix

    return run_matrix(m=m, k=k, n=n, reuse=reuse, timeout_ms=timeout_ms)


def _capsule_targets() -> list[str]:
    """Every target with a capsule store, via the sanctioned corpus locator.

    Discovered, never listed in code: a target name written down here would be exactly the overfit
    the repo's gates exist to catch, and a corpus path written down here would be a second copy of a
    location that already has one owner.
    """
    from merlin.targetgen.corpora import capsule_store_targets

    return capsule_store_targets()


def collect_coverage(targets: list[str] | None = None) -> dict[str, Any]:
    """Emit the per-target obligation ledger, plus each target's mesh edge and whether it is derived.

    The mesh edge is read structurally off the compiled check set (``facts["mesh_dim"]``) rather than
    parsed out of the human-readable ``shape_source`` string, so F4's mesh-tile annotation rests on
    the same derivation the suite itself uses.
    """
    from merlin.targetgen.lit_check_compiler import compile_checks
    from merlin.targetgen.lit_suite import emit

    names = targets if targets is not None else _capsule_targets()
    records: list[dict[str, Any]] = []
    for name in names:
        cov = dict(emit(name))
        fact = (compile_checks(name).facts or {}).get("mesh_dim") or {}
        cov["mesh_edge"] = {
            "value": int(fact["value"]) if fact.get("derived") and fact.get("value") else None,
            "derived": bool(fact.get("derived")),
        }
        records.append(cov)
    return {
        "schema": COVERAGE_SCHEMA,
        "targets": records,
        "declared_total": sum(int(r["obligations_declared"]) for r in records),
        "emitted_total": sum(int(r["emitted"]) for r in records),
        "omitted_total": sum(int(r["omitted"]) for r in records),
        # The prior state, recorded once so the figure can cite it instead of asserting it. It is a
        # count of CONSUMERS of `compiler_obligations`, which was zero before this layer existed --
        # see docs/design/compiler_verification.md section 1, "Finding 2".
        "baseline": {
            "emitted": 0,
            "source": "compiler_obligations had no consumer before this layer "
                      "(docs/design/compiler_verification.md, Finding 2)",
        },
    }


def collect_scaling(shapes: tuple[tuple[int, int, int], ...] = DEFAULT_SCALING_SHAPES,
                    *, reuse: int = 2, timeout_ms: int = 300_000,
                    coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    """Time the formal layer at each shape. Every point records its verdict, not just its seconds.

    A timing curve over unknown verdicts would be meaningless -- a solver that gave up fast is not a
    solver that is fast -- so ``status``/``verified`` travel with each point and the figure marks any
    non-``unsat`` point differently.
    """
    from .refine import validate_workload

    cov = coverage if coverage is not None else collect_coverage()
    edges: dict[str, int] = {}
    for rec in cov["targets"]:
        edge = (rec.get("mesh_edge") or {})
        if edge.get("derived") and edge.get("value"):
            edges[rec["target"]] = int(edge["value"])

    points: list[dict[str, Any]] = []
    for m, k, n in shapes:
        t0 = time.time()
        res = validate_workload(m=m, k=k, n=n, reuse=reuse, timeout_ms=timeout_ms)
        seconds = time.time() - t0
        # A cube whose edge equals some target's DERIVED mesh edge is one hardware tile for that
        # target; that is the only thing that makes a point on this curve mean something physical.
        tiles = sorted(t for t, e in edges.items() if m == k == n == e)
        points.append({
            "m": m, "k": k, "n": n, "product": m * k * n,
            "seconds": seconds, "status": res.status, "verified": bool(res.verified),
            "n_outputs": res.n_outputs, "mesh_tile_for": tiles,
        })
    return {
        "schema": SCALING_SCHEMA,
        "reuse": reuse,
        "timeout_ms": timeout_ms,
        "derived_mesh_edges": edges,
        "points": points,
    }


# ---------------------------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------------------------

def _fmt_seconds(value: float) -> str:
    """Sub-second costs read as milliseconds; anything else as seconds. Three orders of magnitude
    separate the layers, and '0.00 s' next to '3.29 s' loses the whole point of the comparison."""
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


def _caption(fig, text: str, *, y: float = -0.02, width: int = 150) -> None:
    fig.text(0.5, y, textwrap.fill(text, width), ha="center", va="top", fontsize=8.6,
             color=paper.INK)


def _by_fault_layer(record: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(d["fault"], d["layer"]): d for d in record["detections"]}


# ---------------------------------------------------------------------------------------------
# F1 -- detection matrix
# ---------------------------------------------------------------------------------------------

def fig_detection_matrix(record: dict[str, Any]):
    """Layers x fault classes, each cell annotated with the wall time that layer charged.

    Drawn as explicit rectangles rather than an ``imshow``: an unmeasured cell has to be *hatched*,
    not merely a third colour, so that a reader in greyscale cannot read "not measured" as "missed".
    """
    layers = list(record["layers"])
    unmeasured = dict(record.get("layers_not_measured") or {})
    columns = layers + list(unmeasured)
    faults = list(record["faults"])
    cells = _by_fault_layer(record)

    fig, ax = plt.subplots(figsize=(2.35 * len(columns) + 4.2, 0.86 * len(faults) + 3.0))
    _bound = record.get("timeout_ms")
    paper.card(ax, f"Which layer catches which defect  "
                   f"(shape {record['shape']['m']}x{record['shape']['k']}x{record['shape']['n']}, "
                   f"reuse {record['shape']['reuse']}"
                   + (f", solver bound {_bound / 1000:.0f}s)" if _bound else ")"))

    for row, fault in enumerate(faults):
        y = len(faults) - 1 - row
        for col, layer in enumerate(columns):
            hit = cells.get((fault["name"], layer))
            if hit is None:
                # An unmeasured column is one band, labelled once in the middle: six repetitions of
                # "not measured" would read as six findings rather than one absent layer.
                face, hatch, label = UNMEASURED, UNMEASURED_HATCH, ""
            elif hit["detected"]:
                face, hatch, label = DETECTED, None, f"DETECTED\n{_fmt_seconds(hit['seconds'])}"
            elif hit.get("outcome") == "abstained":
                # NOT a miss. The layer ran out of budget; drawing it as a miss would convert a
                # timeout into a coverage claim about the layer below it.
                face, hatch, label = (ABSTAINED, ABSTAINED_HATCH,
                                      f"ABSTAIN\n{_fmt_seconds(hit['seconds'])}")
            else:
                face, hatch, label = MISSED, None, f"miss\n{_fmt_seconds(hit['seconds'])}"
            ax.add_patch(Rectangle((col, y), 1.0, 1.0, facecolor=face, edgecolor=paper.CARD_EC,
                                   linewidth=1.4, hatch=hatch, zorder=3))
            if label:
                # A hatched cell needs an opaque plate behind its text, or the crosshatch runs
                # straight through the label and the one cell the reader most needs to read
                # becomes the one cell they cannot.
                bbox = (dict(boxstyle="round,pad=0.28", fc="white", ec=paper.CARD_EC, lw=1.0)
                        if hatch else None)
                ax.text(col + 0.5, y + 0.5, label, ha="center", va="center", fontsize=9.0,
                        fontweight="bold" if hit["detected"] else "normal",
                        color=paper.INK, zorder=4, bbox=bbox)
    for name in unmeasured:
        ax.text(columns.index(name) + 0.5, len(faults) / 2.0, "NOT MEASURED", ha="center",
                va="center", fontsize=11, fontweight="bold", color=paper.INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=paper.CARD_EC, lw=1.0))

    ax.set_xlim(0, len(columns))
    ax.set_ylim(-1.15, len(faults))
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels([c if c in layers else f"{c.upper()}\n(not measured)" for c in columns],
                       fontsize=10.5, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(faults)) + 0.5)
    ax.set_yticklabels([f["name"] for f in reversed(faults)], fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # The whole argument for keeping the cheap layer is a fault only IT catches. Find those from the
    # data; if the layers were redundant this callout would have nothing to point at.
    # A fault counts as caught-alone ONLY when every other layer actually DECIDED and found nothing.
    # If any layer abstained, the exclusivity is an artifact of that layer's budget, not a coverage
    # fact, and crediting it would let three solver timeouts inflate the cheap layer's headline.
    solo: dict[str, list[str]] = {}
    for fault in faults:
        outcomes = {l: (cells.get((fault["name"], l)) or {}) for l in layers}
        if any(o.get("outcome") == "abstained" for o in outcomes.values()):
            continue
        caught = [l for l, o in outcomes.items() if o.get("detected")]
        if len(caught) == 1:
            solo.setdefault(caught[0], []).append(fault["name"])
    if solo:
        layer, names = max(solo.items(), key=lambda kv: len(kv[1]))
        col = columns.index(layer)
        rows = sorted(len(faults) - 1 - [f["name"] for f in faults].index(nm) for nm in names)
        paper.callout(
            ax, (col + 0.5, rows[0]),
            f"{len(names)} fault{'s' if len(names) != 1 else ''} caught by the {layer} layer ALONE\n"
            f"({', '.join(names)})\nthe cheap layer is not redundant",
            (col + 1.35, -0.62))

    handles = [Patch(facecolor=DETECTED, edgecolor=paper.CARD_EC, label="detected"),
               Patch(facecolor=MISSED, edgecolor=paper.CARD_EC, label="missed")]
    if any(c.get("outcome") == "abstained" for c in cells.values()):
        handles.append(Patch(facecolor=ABSTAINED, edgecolor=paper.CARD_EC, hatch=ABSTAINED_HATCH,
                             label="abstained (no verdict)"))
    handles.append(Patch(facecolor=UNMEASURED, edgecolor=paper.CARD_EC,
                         hatch=UNMEASURED_HATCH, label="not measured"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=len(handles),
              fontsize=9.5, frameon=False)

    fp = [d for d in record.get("false_positives", []) if d["detected"]]
    fp_text = (f"{len(fp)} layer(s) FLAGGED the unmutated program -- the matrix above is not usable"
               if fp else
               "no layer flagged the unmutated program, so every DETECTED cell above is a true positive")
    unmeasured_text = "  ".join(f"{name.upper()}: {why}" for name, why in unmeasured.items())
    bound = record.get("timeout_ms")
    bound_text = (f" Solver bound {bound / 1000:.0f} s per formal attempt."
                  if bound else
                  " Solver bound NOT RECORDED (pre-v2 record): abstentions are indistinguishable "
                  "from misses in this data.")
    ab = [c for c in cells.values() if c.get("outcome") == "abstained"]
    ab_text = (f" {len(ab)} attempt(s) ABSTAINED (no verdict within the bound); an abstention is not "
               f"evidence that the layer would have missed the fault."
               if ab else "")
    _caption(fig, f"Figure F1: {len(faults)} seeded faults x {len(layers)} measured layers, "
                  f"one run each; cell time is that layer's wall cost for that attempt. "
                  f"Control row: {fp_text}.{bound_text}{ab_text} {unmeasured_text}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# F2 -- cost to detect
# ---------------------------------------------------------------------------------------------

def fig_cost_to_detect(record: dict[str, Any]):
    """What each layer charges per attempt, on a log axis, with the static-vs-formal ratio called out.

    Cost is averaged over ALL attempts, not just successful detections: a layer is paid for every
    time it runs, so charging it only for its wins would flatter the expensive layer.
    """
    layers = list(record["layers"])
    unmeasured = dict(record.get("layers_not_measured") or {})
    cells = _by_fault_layer(record)
    n_faults = len(record["faults"])

    stats = []
    for layer in layers:
        times = [d["seconds"] for (_, l), d in cells.items() if l == layer]
        hits = sum(1 for (_, l), d in cells.items() if l == layer and d["detected"])
        abst = sum(1 for (_, l), d in cells.items()
                   if l == layer and d.get("outcome") == "abstained")
        stats.append({"layer": layer, "mean": float(np.mean(times)), "lo": min(times),
                      "hi": max(times), "hits": hits, "abstained": abst})

    rows = stats + [{"layer": name, "mean": None, "why": why} for name, why in unmeasured.items()]
    fig, ax = plt.subplots(figsize=(11.0, 0.95 * len(rows) + 3.0))
    _sh, _b = record["shape"], record.get("timeout_ms")
    paper.card(ax, f"What each layer charges to catch what it catches  "
                   f"(shape {_sh['m']}x{_sh['k']}x{_sh['n']}"
                   + (f", solver bound {_b / 1000:.0f}s)" if _b else ")"))

    y = np.arange(len(rows))[::-1]
    measured = [s["mean"] for s in stats]
    left = min(measured) / 6.0
    right = max(measured) * 45.0
    for yi, row in zip(y, rows):
        if row["mean"] is None:
            ax.barh(yi, right - left, left=left, height=0.6, color=UNMEASURED,
                    edgecolor=paper.CARD_EC, linewidth=1.5, hatch=UNMEASURED_HATCH, zorder=3)
            ax.text(float(np.sqrt(left * right)), yi, "NOT MEASURED -- " + row["why"], va="center",
                    ha="center", fontsize=9.4, fontweight="bold", color=paper.INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=paper.CARD_EC, lw=1.0))
            continue
        ax.barh(yi, row["mean"] - left, left=left, height=0.6, color=paper.GOLD,
                edgecolor=paper.CARD_EC, linewidth=1.6, zorder=3)
        ax.plot([row["lo"], row["hi"]], [yi, yi], color=paper.CARD_EC, lw=1.4, zorder=5)
        # The label clears the min-max rule, not just the bar: an attempt slower than the mean would
        # otherwise draw its whisker straight through the number.
        # An abstention count belongs beside the hit count: "catches 0/6" next to a 40 s mean is a
        # different statement depending on whether the layer decided and missed, or never answered.
        extra = f", {row['abstained']} abstained" if row.get("abstained") else ""
        ax.text(max(row["mean"], row["hi"]) * 1.25, yi,
                f"{_fmt_seconds(row['mean'])} mean   (catches {row['hits']}/{n_faults}{extra})",
                va="center", ha="left", fontsize=10.2, fontweight="bold", color=paper.INK)

    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_yticks(y)
    ax.set_yticklabels([r["layer"] if r["mean"] is not None else r["layer"].upper() for r in rows],
                       fontsize=11)
    ax.set_xlabel("wall seconds per attempt (log scale) -- lower is cheaper")
    ax.set_ylim(-0.7, len(rows) - 0.3)

    # The headline ratio: the two layers that both catch things, priced against each other.
    cheap = min(stats, key=lambda s: s["mean"])
    dear = max(stats, key=lambda s: s["mean"])
    if cheap["layer"] != dear["layer"]:
        yi = y[[r["layer"] for r in rows].index(dear["layer"])]
        paper.callout(
            ax, (dear["mean"], yi + 0.32),
            f"at {_sh['m']}x{_sh['k']}x{_sh['n']}, {dear['layer']} costs "
            f"{dear['mean'] / cheap['mean']:,.0f}x the {cheap['layer']} layer\n"
            f"({_fmt_seconds(dear['mean'])} vs {_fmt_seconds(cheap['mean'])})\n"
            f"so the cheap layer runs on every commit\nand the dear one at bounded shapes\n"
            f"THIS RATIO IS SHAPE-DEPENDENT -- it grows sharply with M.N.K",
            (right / 12.0, yi + 0.95))

    _abs_total = sum(st.get("abstained", 0) for st in stats)
    _abs_note = (f" {_abs_total} attempt(s) ABSTAINED within the bound and are included in the mean "
                 f"at their full elapsed cost, but are not counted as catches."
                 if _abs_total else "")
    _caption(fig, f"Figure F2: mean wall cost per detection attempt across the seeded fault corpus "
                  f"at shape {_sh['m']}x{_sh['k']}x{_sh['n']}; the black rule spans min-max over "
                  f"attempts. Log axis, because the layers are orders of magnitude apart and a "
                  f"linear axis would draw the cheapest layer as nothing. The hatched row is a layer "
                  f"this harness did not run: it is shown at full width with no value, never as a "
                  f"zero. The layer ratio is a property of THIS shape, not of the layers.{_abs_note}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# F3 -- obligation coverage
# ---------------------------------------------------------------------------------------------

def fig_obligation_coverage(record: dict[str, Any]):
    """Per target: declared obligations split into checked and omitted-with-a-recorded-reason.

    The omitted stack is the point of the figure. A coverage plot that showed only the checked bars
    would report the same green whether the denominator were 11 or 110.
    """
    targets = list(record["targets"])
    order = sorted(targets, key=lambda r: (-int(r["obligations_declared"]), r["target"]))
    names = [r["target"] for r in order]
    checked = [int(r["emitted"]) for r in order]
    omitted = [int(r["omitted"]) for r in order]

    # A floor on the width: with one or two targets the bars are narrow but the callout and the
    # reason caption are not, and a figure narrower than its own annotations lays out badly.
    fig, ax = plt.subplots(figsize=(max(1.55 * len(order) + 4.5, 11.0), 6.4))
    paper.card(ax, "Declared compiler obligations: checked vs omitted-with-reason")

    x = np.arange(len(order))
    ax.bar(x, checked, width=0.62, color=paper.SAGE, edgecolor=paper.CARD_EC, linewidth=1.5,
           zorder=3, label="checked by the static suite")
    ax.bar(x, omitted, width=0.62, bottom=checked, color=paper.SALMON, edgecolor=paper.CARD_EC,
           linewidth=1.5, hatch="//", zorder=3, label="omitted, reason recorded")
    for xi, (c, o) in zip(x, zip(checked, omitted)):
        if c:
            ax.text(xi, c / 2, str(c), ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=paper.INK, zorder=4)
        if o:
            ax.text(xi, c + o / 2, str(o), ha="center", va="center", fontsize=10.5, color=paper.INK,
                    zorder=4, bbox=dict(boxstyle="circle,pad=0.22", fc="white",
                                        ec=paper.CARD_EC, lw=0.8))
        ax.text(xi, c + o + 0.08, f"{c + o} declared", ha="center", va="bottom", fontsize=9.0,
                color=paper.INK)

    top = max(int(r["obligations_declared"]) for r in order)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("obligations declared by the capability manifest")
    ax.set_ylim(0, top + 2.6)
    ax.legend(fontsize=9.5, loc="upper right", frameon=False)

    base = record.get("baseline") or {}
    paper.callout(
        ax, (x[0], checked[0] + omitted[0] + 0.12),
        f"{record['emitted_total']} of {record['declared_total']} declared obligations are checked.\n"
        + textwrap.fill(f"Before this layer the count was {base.get('emitted')}: "
                        f"{base.get('source', '')}", 62),
        # Kept inside the bar span: with few targets an offset callout would sit outside the
        # axes entirely and drag the layout with it.
        (float(np.clip(x[0] + 1.55, x[0], x[-1])), top + 1.75))

    # Every omitted obligation carries its own reason; grouping by reason keeps the caption honest
    # without repeating the same sentence six times.
    grouped: dict[str, list[str]] = {}
    for rec in order:
        for om in rec.get("omission_reasons", []):
            grouped.setdefault(om["reason"], []).append(f"{rec['target']}/{om['obligation']}")
    reasons = "  ".join(f"[{', '.join(who)}] {why}" for why, who in grouped.items())
    _caption(fig, f"Figure F3: obligations come from each target's capability manifest, so the "
                  f"denominator is derived, not chosen. Omission reasons -- {reasons}", y=-0.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# F4 -- formal-layer scaling
# ---------------------------------------------------------------------------------------------

def fig_formal_scaling(record: dict[str, Any]):
    """Solve time against problem size, log-log, with the derived mesh-tile point annotated.

    Log-log because the only interpretable summary of a solver cost curve is its slope; the fitted
    slope is computed from these points, not asserted. A point whose verdict was not ``unsat`` is
    drawn hollow: a fast ``unknown`` is a solver giving up, and plotting it as a win would invert
    the meaning of the whole figure.
    """
    points = sorted(record["points"], key=lambda p: p["product"])
    xs = np.array([p["product"] for p in points], dtype=float)
    ys = np.array([p["seconds"] for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    paper.card(ax, "Formal layer: solve time vs problem size")

    ax.plot(xs, ys, color=paper.STEEL, lw=1.8, zorder=2)
    for p in points:
        verified = bool(p["verified"])
        ax.scatter([p["product"]], [p["seconds"]], s=110, zorder=4,
                   facecolor=paper.SAGE if verified else "white",
                   edgecolor=paper.CARD_EC, linewidth=1.6,
                   marker="o" if verified else "X")
        ax.annotate(f"{p['m']}x{p['k']}x{p['n']}\n{_fmt_seconds(p['seconds'])}",
                    (p["product"], p["seconds"]), textcoords="offset points", xytext=(0, -30),
                    ha="center", fontsize=8.6, color=paper.INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Headroom below: each point's label hangs under its marker, and the cheapest point's label
    # would otherwise be clipped into the axis - a clipped number is a number nobody can check.
    ax.set_ylim(min(ys) * 0.22, max(ys) * 4.5)
    ax.set_xlim(min(xs) * 0.5, max(xs) * 4.0)
    ax.set_xlabel("problem size  M x K x N (elements, log scale)")
    ax.set_ylabel("solver wall seconds (log scale)")
    ax.grid(True, which="both", color="#d8d2c2", lw=0.6, zorder=1)

    # Slope of the log-log fit -- the summary a reader would otherwise eyeball wrongly.
    slope = float(np.polyfit(np.log10(xs), np.log10(ys), 1)[0]) if len(xs) > 1 else float("nan")

    # EVERY derived mesh tile is annotated, not just the first. More than one target in this tree
    # has a derivable mesh edge, and drawing only the first silently turns a multi-target result
    # into a single-target anecdote. Which targets appear is read from the record, never named here.
    tiles = [p for p in points if p["mesh_tile_for"]]
    edges = record.get("derived_mesh_edges") or {}
    for i, tile in enumerate(tiles):
        who = ", ".join(tile["mesh_tile_for"])
        edge = edges.get(tile["mesh_tile_for"][0])
        # The curve runs bottom-left to top-right, so the two EMPTY corners are upper-left and
        # lower-right. Alternate between them: stacking both callouts on the same side puts the
        # second one on top of the first one's point label.
        pos = ((tile["product"] / 9.0, tile["seconds"] * 9.0) if i % 2 == 0
               else (tile["product"] * 1.9, tile["seconds"] * 0.20))
        paper.callout(
            ax, (tile["product"], tile["seconds"]),
            f"{tile['m']}x{tile['k']}x{tile['n']} = one mesh tile for {who}\n"
            f"(edge {edge}, derived from that target's RTL facts)\n"
            f"{tile['status']} in {_fmt_seconds(tile['seconds'])}",
            pos)

    verdicts = sorted({p["status"] for p in points})
    unverified = [p for p in points if not p["verified"]]
    _caption(fig, f"Figure F4: translation validation of the interface plane at {len(points)} "
                  f"concrete shapes (reuse {record['reuse']}, solver timeout "
                  f"{record['timeout_ms'] / 1000:.0f} s). Verdicts observed: {', '.join(verdicts)}; "
                  f"{len(points) - len(unverified)}/{len(points)} verified. Empirical log-log slope "
                  f"{slope:.2f}. Filled markers are verified; a hollow X would be a shape the solver "
                  f"did not settle, which is not a data point about speed. EVERY point here is a "
                  f"CORRECT program, so this curve prices VERIFICATION (proving unsat). It says "
                  f"nothing about refutation cost (finding a counterexample), which is the direction "
                  f"fault detection depends on and which grows far faster -- see F1's solver bound.")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

FIGURES: tuple[tuple[str, str, str], ...] = (
    ("f1_detection_matrix", DETECTION_FILE, "fig_detection_matrix"),
    ("f2_cost_to_detect", DETECTION_FILE, "fig_cost_to_detect"),
    ("f3_obligation_coverage", COVERAGE_FILE, "fig_obligation_coverage"),
    ("f4_formal_scaling", SCALING_FILE, "fig_formal_scaling"),
)


def load_records(source: Path) -> dict[str, dict[str, Any]]:
    """Read the three records out of a directory. A missing record is an error, not an empty figure."""
    out: dict[str, dict[str, Any]] = {}
    for name in (DETECTION_FILE, COVERAGE_FILE, SCALING_FILE):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; collect the records before drawing")
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def draw_all(records: dict[str, dict[str, Any]], out_dir: Path, *,
             formats: tuple[str, ...] = ("png", "svg")) -> list[Path]:
    """Draw every figure from already-loaded records. Returns the paths written."""
    written: list[Path] = []
    for stem, source, fn_name in FIGURES:
        fig = globals()[fn_name](records[source])
        for ext in formats:
            path = out_dir / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=160)
            written.append(path)
        plt.close(fig)
    return written


def internal_note(records: dict[str, Any]) -> str:
    """Render the candid internal note FROM the records, so it cannot drift away from them.

    The previous note was hand-written, and an audit found its headline mesh-tile figure ("1.56 s")
    matched no committed record in the tree -- it was 2x optimistic against the freshest one. A note
    that is typed is a note that goes stale silently; this one is derived, so a number in it is a
    number some JSON in the same directory contains.
    """
    det = records.get(DETECTION_FILE, {})
    cov = records.get(COVERAGE_FILE, {})
    sca = records.get(SCALING_FILE, {})
    sh = det.get("shape") or {}
    shape = f"{sh.get('m')}x{sh.get('k')}x{sh.get('n')}"
    bound = det.get("timeout_ms")
    cells = _by_fault_layer(det)
    layers = list(det.get("layers", []))
    names = [f["name"] for f in det.get("faults", [])]

    def _outcome(fault, layer):
        c = cells.get((fault, layer)) or {}
        return c.get("outcome", "detected" if c.get("detected") else "clean")

    solo = [f for f in names
            if not any(_outcome(f, l) == "abstained" for l in layers)
            and [l for l in layers if _outcome(f, l) == "detected"] == ["static"]]
    abst = [(f, l) for f in names for l in layers if _outcome(f, l) == "abstained"]

    rows = []
    for f in names:
        cs = "  ".join(f"{l}={_outcome(f, l)}" for l in layers)
        rows.append(f"  {f:26s} {cs}")

    tiles = [pt for pt in sca.get("points", []) if pt.get("mesh_tile_for")]
    tile_lines = [f"  {pt['m']}x{pt['k']}x{pt['n']} ({', '.join(pt['mesh_tile_for'])}): "
                  f"{pt['status']} in {pt['seconds']:.2f} s" for pt in tiles]

    return "\n".join([
        "# Verification layers — internal note",
        "",
        "GENERATED from the JSON records in this same directory by `merlin.verify.plots`.",
        "Do not hand-edit: every number below is read from a record beside it, because the previous",
        "hand-written version of this note carried a headline figure that matched no record at all.",
        "",
        "## What was measured",
        "",
        f"- detection matrix at **{shape}** (reuse {sh.get('reuse')}), solver bound "
        f"**{bound} ms** per formal attempt" if bound else
        "- detection matrix: solver bound NOT RECORDED (pre-v2 record)",
        f"- record schema: `{det.get('schema', 'UNKNOWN')}`",
        f"- {len(names)} seeded faults x {len(layers)} measured layers",
        "",
        "## Per-fault outcome",
        "",
        *rows,
        "",
        "`clean` = the layer ran and found nothing. `abstained` = it could not decide (solver timeout,",
        "or the encoder refused the program) and is NOT evidence of absence.",
        "",
        f"- caught by the static layer ALONE: **{len(solo)}** ({', '.join(solo) if solo else 'none'})",
        f"- abstentions: **{len(abst)}**"
        + (" (" + ", ".join(f"{l} on {f}" for f, l in abst) + ")" if abst else ""),
        "",
        "## Verification cost at a derived mesh tile",
        "",
        "These are CORRECT programs, so this prices verification (`unsat`), not refutation (`sat`).",
        "Refutation at a mesh tile has been measured to exceed a 60 s bound; nothing here measures it.",
        "",
        *(tile_lines or ["  (no derived mesh tile in the scaling sweep)"]),
        "",
        "## Obligation coverage",
        "",
        f"- **{cov.get('emitted_total')} of {cov.get('declared_total')}** declared obligations are "
        f"checked; {cov.get('omitted_total')} omitted with recorded reasons.",
        f"- baseline before this work: **{(cov.get('baseline') or {}).get('emitted')}** "
        f"({(cov.get('baseline') or {}).get('source')})",
        "",
        "## Standing limitations",
        "",
        "- The formally validated pass is in the PROTOTYPE catalog, so the obligation gate refuses to",
        "  credit its verdict to production; the production passes read 0 / 4 verified.",
        "- Detection claims for the formal layer hold at the small shapes where refutation terminates.",
        "  At a real mesh tile the layer verifies in seconds but refutes nothing within 60 s.",
        "- The RTL tiers are not measured here and are recorded as such, never estimated.",
        "",
    ])


def build(*, source: Path | None = None, m: int = 4, k: int = 4, n: int = 4, reuse: int = 2,
          shapes: tuple[tuple[int, int, int], ...] = DEFAULT_SCALING_SHAPES) -> Path:
    """Collect (or re-read) the records, draw the four figures, and write the versioned product."""
    from merlin.common.artifacts import new_product

    # A product whose `sources` is empty cannot be audited once somebody cites its numbers, so both
    # paths describe what they actually read or measured -- including the shape and the solver bound,
    # without which a detection count means nothing.
    if source is not None:
        records = load_records(source)
        notes = f"figures re-drawn from records in {source}"
        det = records.get(DETECTION_FILE, {})
        sh = det.get("shape") or {}
        sources = [f"records re-read from {source}"] + [
            f"{name}: schema {rec.get('schema', 'UNKNOWN')}" for name, rec in sorted(records.items())
        ] + [
            f"detection shape: {sh.get('m')}x{sh.get('k')}x{sh.get('n')} reuse {sh.get('reuse')}",
            f"detection solver bound: {det.get('timeout_ms', 'NOT RECORDED')} ms",
        ]
    else:
        coverage = collect_coverage()
        detection = collect_detection(m=m, k=k, n=n, reuse=reuse)
        records = {
            DETECTION_FILE: detection,
            COVERAGE_FILE: coverage,
            SCALING_FILE: collect_scaling(shapes, reuse=reuse, coverage=coverage),
        }
        notes = "records collected fresh by merlin.verify.plots"
        sources = [
            "records measured by merlin.verify.plots (fresh run)",
            f"detection shape: {m}x{k}x{n} reuse {reuse}",
            f"detection solver bound: {detection.get('timeout_ms', 'NOT RECORDED')} ms",
            f"fault corpus: {len(detection.get('faults', []))} seeded faults",
            f"scaling shapes: {list(shapes)}",
        ]

    product = new_product("verification", version=1, notes=notes, sources=sources)
    for name, rec in records.items():
        product.add_artifact(name).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for path in draw_all(records, product.path):
        product.add_artifact(path.name)
    # The candid note ships INSIDE the product, so `latest` always has one and it can never be
    # pinned to a run whose records were never written -- which is how the previous note ended up
    # orphaned in a superseded directory.
    product.add_artifact("README.md").write_text(internal_note(records), encoding="utf-8")
    product.write_manifest()
    return product.path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", type=Path,
                    help="a directory holding the three JSON records; skips re-measuring")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--reuse", type=int, default=2)
    args = ap.parse_args(argv)

    path = build(source=args.source, m=args.m, k=args.k, n=args.n, reuse=args.reuse)
    print(f"wrote {path}")
    for stem, _, _ in FIGURES:
        print(f"  {path / (stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
