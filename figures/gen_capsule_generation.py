#!/usr/bin/env python3
"""Figure: capability-derived capsule generation (single-column).

Shows the derivation that produces a capsule corpus, with the real counts rather than a schematic:
pinned target specs + RTL-derived capabilities + workload requirements define a compiler-independent
coverage space; representative obligations are materialized as ISA / layer / model-slice / model
capsules; each capsule exposes its program while the numerical and structural answer surfaces stay
hidden; obligations that could not be materialized stay VISIBLE instead of shrinking the denominator;
public capsules drive authoring while capability-derived hidden capsules grade the frozen compiler.

DATA IS DERIVED, NEVER TYPED. Unlike the board-measurement figures beside this one, whose JSON comes
from a sealed run, this figure's data is a repo-local census -- so a hardcoded number would go stale
silently and nothing would notice. ``--derive`` rebuilds the sidecar JSON from the live corpus,
``--check`` fails on drift, and the default path plots the tracked JSON.

Palette note, deliberate: this figure uses ``paper_plot_style`` (the Okabe-Ito/serif set the other
figures in this directory use) rather than the repo-wide ``merlin_plotstyle`` cream/navy identity,
because it ships in the same paper as those figures and a second palette inside one paper is worse
than reusing a plainer one. The DIAGRAM primitives below are adapted from
``merlin/experiments/gemmini_perf_bench/scripts/gen_circt_diagram.py``; they take their colours as
arguments, so they carry no palette of their own.

    python figures/gen_capsule_generation.py --derive     # rebuild the JSON from the corpus
    python figures/gen_capsule_generation.py --check      # non-zero if the JSON is stale
    python figures/gen_capsule_generation.py              # render the PDF + PNG
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                       # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))                             # sibling paper_plot_style
sys.path.insert(0, str(REPO / "merlin" / "python"))       # the merlin package, for derivation

from paper_plot_style import COLORS                       # noqa: E402,F401  (rcParams applied on import)

STEM = "capsule_generation"

# ---------------------------------------------------------------------------------------------
# palette -- the same Okabe-Ito set `paper_plot_style` draws from, assigned to THIS figure's roles.
# Kept local rather than added to the shared module: these names are about capsule kinds, and the
# shared COLORS keys are about baselines.
# ---------------------------------------------------------------------------------------------
INK = "#222222"
FACE = "#FCFCFC"
KIND_COLORS = {
    "isa": "#56B4E9",           # sky blue
    "layer": "#0072B2",         # blue
    "model_slice": "#009E73",   # bluish green
    "model": "#E69F00",         # orange
}
KIND_LABELS = {"isa": "ISA", "layer": "layer", "model_slice": "model-slice", "model": "model"}
GAP = "#D55E00"                 # vermillion -- obligations that could not be materialized
HIDDEN = "#777777"              # the held-out lane
SH = [pe.withSimplePatchShadow(offset=(1.4, -1.4), shadow_rgbFace=(0.2, 0.2, 0.2),
                               alpha=0.16, rho=1.0)]

#: The eight coverage axes, in report order. Kept here so the figure and the JSON agree on both the
#: set and the spelling; a new axis shows up as a missing key rather than as a silently short list.
AXES = ("cells", "composition", "memory_mapping", "shape_geometry",
        "host_only", "host_lane", "epilogue", "conv_geometry")


# =============================================================================================
# derivation
# =============================================================================================
def _targets() -> list[str]:
    """Every target with a conformance spec. Discovered, so a new target needs no edit here."""
    d = REPO / "merlin" / "contract" / "capsules" / "conformance"
    return sorted(p.stem for p in d.glob("*.yaml"))


def _repo_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001 -- provenance, never fatal
        return "unknown"


def derive() -> dict:
    """Census the live corpus. Every number in the figure comes from here and nowhere else."""
    import yaml
    from merlin.targetgen import conformance as CF
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    manifest = yaml.safe_load(
        (REPO / "merlin" / "contract" / "capsules" / "MANIFEST.yaml").read_text(encoding="utf-8"))

    per_target: dict[str, dict] = {}
    for target in _targets():
        desc = descriptor_for(target)
        if desc is None:
            continue
        te = load_target_experiment(desc)
        graded = list(te.graded_roots())
        try:
            hidden_roots = list(te.hidden_roots())
        except Exception:                                 # noqa: BLE001 -- absent holdouts are normal
            hidden_roots = []

        kinds: dict[str, int] = {k: 0 for k in KIND_COLORS}
        labels: dict[str, int] = {"public": 0, "dev": 0, "hidden": 0}
        for root in graded + hidden_roots:
            for p in Path(root).glob("*/capsule.yaml"):
                cap = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                k, lab = str(cap.get("kind")), str(cap.get("label"))
                if k in kinds:
                    kinds[k] += 1
                if lab in labels:
                    labels[lab] += 1
        if not sum(kinds.values()):
            continue                                       # a target with no corpus is not plotted

        # Coverage. `tile_dim` is MANDATORY: without it the cover spells cells `family/dtype` while
        # the requirement spells them `family/dtype/alignment`, the intersection is empty, and every
        # required cell reports uncovered. `exclude` matters in the other direction -- a capsule
        # excluded from grading must not be allowed to represent its cell.
        coverage: dict[str, list | None] = {}
        spec_path = REPO / "merlin" / "contract" / "capsules" / "conformance" / f"{target}.yaml"
        if spec_path.is_file():
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            tile = (spec.get("boundaries") or {}).get("tile_edge")
            exclude = set(getattr(te, "graded_exclude", ()) or ())
            gap = CF.uncovered(spec, graded, labels={"public", "dev"}, tile_dim=tile, exclude=exclude)
            required = {c["cell"] for c in (spec.get("cells") or [])}
            unc = {u["cell"] if isinstance(u, dict) else u for u in (gap.get("uncovered") or [])}
            coverage["cells"] = [len(required) - len(unc), len(required)]
            for axis in AXES[1:]:
                block = gap.get(axis) or {}
                c, r = block.get("n_covered"), block.get("n_required")
                coverage[axis] = None if c is None or r is None else [c, r]

        not_built = ((manifest.get("roster_generation") or {}).get(target) or {}).get("not_built") or []
        forbid = ((manifest.get("lane_generation") or {}).get(target) or {}).get(
            "forbid_not_provable") or []
        per_target[target] = {
            "kinds": kinds, "labels": labels, "coverage": coverage,
            "not_built": len(not_built), "forbid_not_provable": len(forbid),
        }

    # Unmaterializable obligations, as COUNTS + REASONS. Names are deliberately not stored: a capsule
    # name is the one thing a held-out set must never leak, and the figure needs neither.
    reasons: dict[str, int] = {}
    for block, key in (("roster_generation", "not_built"), ("lane_generation", "forbid_not_provable")):
        for rows in (manifest.get(block) or {}).values():
            for row in (rows or {}).get(key) or []:
                reasons[str(row.get("reason"))[:120]] = reasons.get(str(row.get("reason"))[:120], 0) + 1

    totals = {k: sum(t["kinds"][k] for t in per_target.values()) for k in KIND_COLORS}
    return {
        "schema": "merlin.figure.capsule_generation/v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_sha": _repo_sha(),
        "targets": sorted(per_target),
        "per_target": per_target,
        "totals_by_kind": totals,
        "totals_by_label": {lab: sum(t["labels"][lab] for t in per_target.values())
                            for lab in ("public", "dev", "hidden")},
        "unmaterializable": {
            "total": sum(t["not_built"] + t["forbid_not_provable"] for t in per_target.values()),
            "reasons": reasons,
        },
        "held_out_manifest": manifest.get("held_out"),
    }


def validate(data: dict) -> None:
    """Refuse a dataset that cannot be true. A figure that invents a number is worse than no figure."""
    if not data.get("per_target"):
        raise ValueError("no target produced a corpus census; the figure would be empty")
    for target, row in data["per_target"].items():
        kinds, labels = sum(row["kinds"].values()), sum(row["labels"].values())
        if kinds != labels:
            raise ValueError(
                f"{target}: kinds sum to {kinds} but labels sum to {labels}; a capsule has been "
                f"double-counted or dropped, and every proportion in the figure would be wrong")
        for axis, pair in (row.get("coverage") or {}).items():
            if pair is not None and pair[0] > pair[1]:
                raise ValueError(f"{target}/{axis}: covered {pair[0]} exceeds required {pair[1]}")
    # The gap band is the figure's argument. If it renders empty the figure asserts the opposite of
    # its caption, so an empty one has to be a deliberate, visible state rather than a silent zero.
    if data["unmaterializable"]["total"] == 0 and data["unmaterializable"]["reasons"]:
        raise ValueError("unmaterializable total is 0 but reasons were recorded; the two disagree")


# =============================================================================================
# drawing primitives (adapted from gen_circt_diagram.py; colours are arguments, not a palette)
# =============================================================================================
def _tc(hexc: str) -> str:
    r, g, b = (int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"


def chip(ax, cx, cy, w, h, text, fc, *, fs=6.4, weight="bold", ec=INK, lw=0.9, hatch=None):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.004,rounding_size=0.9",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4, hatch=hatch)
    b.set_path_effects(SH)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", color=_tc(fc), fontsize=fs,
            fontweight=weight, zorder=5, linespacing=1.05)


def arrow(ax, p0, p1, *, rad=0.0, color=INK, lw=1.1, dotted=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3, shrinkA=1.5, shrinkB=1.5,
                                 linestyle=("dotted" if dotted else "solid")))


def band(ax, y, label):
    """A faint left-margin band letter, so the caption can refer to (a)-(d)."""
    ax.text(-8.4, y, label, ha="left", va="center", fontsize=7, color="#888888",
            fontweight="bold", zorder=6)


# =============================================================================================
# the figure
# =============================================================================================
def render(data: dict, out_stem: Path) -> None:
    per = data["per_target"]
    targets = [t for t in data["targets"] if t in per]
    fig, ax = plt.subplots(figsize=(3.4, 5.3))
    ax.set_xlim(-9, 104)
    ax.set_ylim(-5, 100)
    ax.axis("off")

    # ---------------- (a) the derivation ----------------
    band(ax, 96, "a")
    srcs = [("pinned target\nspecification", 16.5), ("RTL-derived\ncapabilities", 50), ("workload\nrequirements", 83.5)]
    for text, cx in srcs:
        chip(ax, cx, 95, 31, 7.5, text, FACE, fs=5.8, weight="normal")
    for _, cx in srcs:
        arrow(ax, (cx, 91.2), (50, 86.4))
    chip(ax, 50, 82.5, 92, 7.2, "compiler-independent coverage space", "#EDEDED", fs=6.6)
    n_cells = sum((per[t]["coverage"].get("cells") or [0, 0])[1] for t in targets)
    ax.text(50, 77.4, f"{len(AXES)} axes · {n_cells} required cells · {len(targets)} targets",
            ha="center", va="center", fontsize=5.6, color="#555555", style="italic")

    # ---------------- (b) materialization ----------------
    band(ax, 70, "b")
    arrow(ax, (50, 76.0), (50, 72.4))
    ax.text(50, 70.2, "materialized as representative obligations", ha="center", va="center",
            fontsize=5.8, color="#555555", style="italic")
    tot = data["totals_by_kind"]
    grand = sum(tot.values()) or 1
    x = 4.0
    for kind in ("isa", "layer", "model_slice", "model"):
        w = 92.0 * tot[kind] / grand
        chip(ax, x + w / 2, 64.6, w, 7.0, "", KIND_COLORS[kind], fs=5.6)
        ax.text(x + w / 2, 64.6, str(tot[kind]), ha="center", va="center",
                fontsize=6.0, color=_tc(KIND_COLORS[kind]), fontweight="bold", zorder=6)
        if w > 14:
            ax.text(x + w / 2, 59.6, KIND_LABELS[kind], ha="center", va="center",
                    fontsize=5.4, color=INK)
        else:                       # too narrow to label under: point at it from the right margin
            ax.text(x + w + 1.6, 59.6, KIND_LABELS[kind], ha="left", va="center",
                    fontsize=5.4, color=INK)
            arrow(ax, (x + w + 1.2, 60.4), (x + w / 2, 62.2), rad=0.25, lw=0.7, color="#888888")
        x += w

    # ---------------- (c) what a capsule exposes ----------------
    band(ax, 50, "c")
    arrow(ax, (50, 57.0), (50, 53.6))
    chip(ax, 27.5, 49.0, 45, 8.4, "VISIBLE\ncapsule.yaml · interface MLIR", FACE, fs=5.3, weight="normal")
    chip(ax, 74.5, 49.0, 45, 8.4, "", "#E8E8E8", fs=5.3, weight="normal", hatch="/////")
    ax.text(74.5, 49.0, "HIDDEN\nnumeric + structural answers", ha="center", va="center",
            fontsize=5.3, color=INK, zorder=7, linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="none", alpha=0.94))
    ax.text(50, 42.6, "the compiler sees the program, never the answer",
            ha="center", va="center", fontsize=5.6, color="#555555", style="italic")

    # ---------------- (d) per target ----------------
    band(ax, 33, "d")
    arrow(ax, (50, 40.2), (50, 36.6))
    top, bar_h, step = 33.0, 4.4, 7.4
    max_total = max(sum(per[t]["kinds"].values()) + per[t]["not_built"]
                    + per[t]["forbid_not_provable"] for t in targets) or 1
    span = 61.0
    for i, t in enumerate(targets):
        y = top - i * step
        ax.text(-4.0, y, t, ha="left", va="center", fontsize=5.6, color=INK)
        left = 19.5
        for kind in ("isa", "layer", "model_slice", "model"):
            w = span * per[t]["kinds"][kind] / max_total
            if w <= 0:
                continue
            ax.add_patch(FancyBboxPatch((left, y - bar_h / 2), w, bar_h,
                                        boxstyle="square,pad=0", facecolor=KIND_COLORS[kind],
                                        edgecolor="white", linewidth=0.4, zorder=4))
            left += w
        gapn = per[t]["not_built"] + per[t]["forbid_not_provable"]
        if gapn:
            w = max(span * gapn / max_total, 1.6)         # a floor, so one gap is never invisible
            ax.add_patch(FancyBboxPatch((left, y - bar_h / 2), w, bar_h, boxstyle="square,pad=0",
                                        facecolor="white", edgecolor=GAP, linewidth=0.9,
                                        hatch="xxx", zorder=5))
            ax.text(left + w + 1.2, y, f"{gapn} unmaterialized", ha="left", va="center",
                    fontsize=4.9, color=GAP, fontweight="bold")
            left += w
        ax.text(88.0, y, f"{per[t]['labels']['hidden']} hidden", ha="left", va="center",
                fontsize=4.9, color=HIDDEN)

    # ---------------- the two lanes ----------------
    ylane = top - (len(targets) - 1) * step - 6.6
    arrow(ax, (34, ylane + 3.4), (34, ylane), color=KIND_COLORS["layer"])
    arrow(ax, (78, ylane + 3.4), (78, ylane), color=HIDDEN, dotted=True)
    chip(ax, 34, ylane - 3.0, 44, 5.6,
         f"public ({data['totals_by_label']['public']}) → authoring",
         FACE, fs=5.2, weight="normal", ec=KIND_COLORS["layer"])
    chip(ax, 78, ylane - 3.0, 38, 5.6,
         f"hidden ({data['totals_by_label']['hidden']}) → frozen",
         "#EFEFEF", fs=5.2, weight="normal", ec=HIDDEN)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.995, bottom=0.005)
    for ext in ("pdf", "png"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {out}")
    plt.close(fig)


# =============================================================================================
def _json_path() -> Path:
    existing = sorted(HERE.glob(f"{STEM}_*.json"))
    if existing:
        return existing[-1]
    return HERE / f"{STEM}_{datetime.now(timezone.utc):%Y%m%d}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derive", action="store_true", help="re-derive the sidecar JSON and write it")
    ap.add_argument("--check", action="store_true", help="re-derive and diff; non-zero on drift")
    a = ap.parse_args(argv)

    path = _json_path()
    if a.derive:
        data = derive()
        validate(data)
        path = HERE / f"{STEM}_{datetime.now(timezone.utc):%Y%m%d}.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return 0

    if a.check:
        if not path.is_file():
            print(f"no sidecar JSON at {path}; run --derive")
            return 1
        old = json.loads(path.read_text(encoding="utf-8"))
        new = derive()
        drift = [k for k in ("per_target", "totals_by_kind", "totals_by_label", "unmaterializable")
                 if old.get(k) != new.get(k)]
        if drift:
            print(f"STALE: {path.name} disagrees with the corpus on {drift}; re-run --derive")
            return 1
        print(f"{path.name}: current")
        return 0

    if not path.is_file():
        print(f"no sidecar JSON at {path}; run --derive first")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    render(data, HERE / STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
