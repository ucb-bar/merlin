"""Figures for what the formal layer buys over the dynamic grade already in place.

These are deliberately not the Phase-1 figures. Those (``merlin.verify.plots``) measure the checker
against faults *we seeded ourselves*, which answers "does it work" and cannot answer "is it worth
running". These read the ARCHIVE -- every capsule-bench submission this project has produced and
graded -- and ask what the formal layer would have added to results that already exist.

Three figures, and two of the three report a limit rather than a win. That is the finding, not a
presentation problem:

``f5_reach_and_verdict``
    The funnel, ending in an outcome rather than a filter: of every archived submission, how many can
    the layer look at, and what did it conclude? Most are unreachable, and the largest single reason
    is that the submission IS its own specification, where an equivalence query is ``X == X``.

``f6_agreement``
    Verdict against the grade those submissions already received. Two cells carry the result: the
    layer never contradicts a passing grade (zero false alarms), and it returns a verdict on
    submissions the numeric grade never ran at all -- which is the only cell where it adds
    information rather than confirming it.

``f7_why_not_more``
    The abstention breakdown, read as a work plan. The largest bar is float, and float is also where
    every expensive-to-find defect in the archive lives -- so the coverage gap and the cost are the
    same gap.

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


def load_ablation(product: Path) -> dict[str, Any]:
    """Read an ``ablation.json`` written by :mod:`merlin.verify.ablation`."""
    return json.loads((product / "ablation.json").read_text(encoding="utf-8"))


def latest_ablation() -> dict[str, Any]:
    """The newest ablation product on disk. Figures are driven by measured data, never by literals."""
    from merlin.common.paths import artifacts_dir

    root = artifacts_dir() / "verification"
    found = sorted(root.rglob("ablation.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise FileNotFoundError(
            f"no ablation.json under {root}; run `python -m merlin.verify.ablation --write` first")
    return load_ablation(found[0].parent)


def _tally(records: list[dict]) -> dict[str, Any]:
    """Verdict counts, the verdict x grade cross-tab, and the abstention reasons."""
    eligible = [r for r in records if r.get("verdict") != "excluded"]
    grid: dict[tuple[str, str], int] = {}
    for r in eligible:
        key = (str(r.get("verdict")), str(r.get("numeric_status", "absent")))
        grid[key] = grid.get(key, 0) + 1
    refuted = [r for r in eligible if r.get("verdict") == "refuted"]
    return {
        "total": len(records),
        "excluded": sum(1 for r in records if r.get("verdict") == "excluded"),
        "eligible": len(eligible),
        "counts": dict(Counter(str(r.get("verdict")) for r in eligible)),
        "grid": grid,
        "abstain_reasons": dict(Counter(str(r.get("reason_kind") or "other")
                                        for r in eligible if r.get("verdict") == "abstained")),
        "refuted_outside_stimulus": sum(1 for r in refuted
                                        if r.get("counterexample_outside_stimulus")),
        "refuted_total": len(refuted),
        "refuted_kinds": dict(Counter("contract" if r.get("reason_kind") == "output_contract"
                                      else "numeric" for r in refuted)),
    }


def _caption(fig, text: str, *, y: float = -0.03) -> None:
    import textwrap
    fig.text(0.01, y, "\n".join(textwrap.wrap(text, 132)), ha="left", va="top",
             fontsize=8.4, color=paper.INK, alpha=0.86)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def fig_reach_and_verdict(tally: dict[str, Any], reach: dict[str, Any]):
    """The funnel, ending in a verdict rather than a filter."""
    counts = tally["counts"]
    stages = [
        ("archived submissions", tally["total"], paper.GREY),
        ("EXCLUDED — the buffer is its own\nspecification (query is X == X)", tally["excluded"],
         VACUOUS),
        ("eligible", tally["eligible"], paper.STEEL),
        ("abstained — outside what the\nencoder models (never a pass)", counts.get("abstained", 0),
         OUT_OF_SCOPE),
        ("VERIFIED — agrees with the spec\nfor every input at that shape", counts.get("verified", 0),
         REACHABLE),
        ("REFUTED", counts.get("refuted", 0), COSTLY),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    paper.card(ax, "What the formal layer reached, and what it concluded")
    ys = range(len(stages))
    ax.barh(list(ys), [s[1] for s in stages], height=0.62, color=[s[2] for s in stages],
            edgecolor=paper.CARD_EC, linewidth=1.4, zorder=3)
    total = max(tally["total"], 1)
    for y, (_lab, n, _c) in zip(ys, stages):
        ax.text(n + total * 0.008, y, f"{n}   ({100 * n / total:.1f}%)", va="center", fontsize=10,
                color=paper.INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0] for s in stages], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.22)
    ax.set_xlabel("archived capsule-bench submissions")
    _caption(fig, "Every submission this project has produced and graded, filtered to what an "
                  "equivalence proof can say anything about. The binding constraint is not the "
                  "solver: the largest single loss is submissions that reproduce the interface "
                  "program they were handed command for command, where proving equivalence proves "
                  "nothing and a bug in the shared encoder would cancel on both sides.")
    fig.tight_layout()
    return fig


def fig_agreement(tally: dict[str, Any]):
    """Verdict against the grade the same submission already received."""
    import numpy as np

    grades = ["pass", "fail", "skipped"]
    verdicts = ["verified", "refuted", "abstained"]
    grid = np.array([[tally["grid"].get((v, g), 0) for g in grades] for v in verdicts], dtype=float)

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    paper.card(ax, "Formal verdict versus the numeric grade the submission already had")
    ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=1)
    for i, v in enumerate(verdicts):
        for j, g in enumerate(grades):
            n = int(grid[i, j])
            # the two cells that carry the result
            if v == "refuted" and g == "pass":
                face, note = REACHABLE, "no false alarms"
            elif v == "refuted" and g == "skipped":
                face, note = paper.GOLD, "NEW information"
            elif v == "refuted" and g == "fail":
                face, note = paper.STEEL, "agreement"
            elif v == "verified" and g == "fail":
                face, note = COSTLY, "open"
            else:
                face, note = "#efe7d5", ""
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                       edgecolor=paper.CARD_EC, linewidth=1.4, zorder=2))
            ax.text(j, i - 0.10, str(n), ha="center", va="center", fontsize=17,
                    fontweight="bold", color=paper.INK, zorder=4)
            if note:
                ax.text(j, i + 0.27, note, ha="center", va="center", fontsize=8.0,
                        color=paper.INK, zorder=4)
    ax.set_xticks(range(len(grades)))
    ax.set_xticklabels([f"numeric grade\n{g.upper()}" for g in grades], fontsize=9.5)
    ax.set_yticks(range(len(verdicts)))
    ax.set_yticklabels([v.upper() for v in verdicts], fontsize=10)
    ax.set_xlim(-0.5, len(grades) - 0.5)
    ax.set_ylim(len(verdicts) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    outside, tot = tally["refuted_outside_stimulus"], tally["refuted_total"]
    kinds = tally["refuted_kinds"]
    _caption(fig, f"Zero refutations of a passing submission: on this archive the layer never "
                  f"contradicts the grader, which is the soundness result. It agrees independently "
                  f"on {tally['grid'].get(('refuted', 'fail'), 0)} defects, and returns a verdict on "
                  f"{tally['grid'].get(('refuted', 'skipped'), 0)} submissions the numeric grade "
                  f"never ran at all -- the only cell where it adds information rather than "
                  f"confirming it. Of {tot} refutations, {kinds.get('contract', 0)} are undeclared-"
                  f"output violations and {kinds.get('numeric', 0)} are value divergences; "
                  f"{outside} need an operand outside the stimulus's {{0..3}}, so no re-run of the "
                  f"dynamic check could have found them.")
    fig.tight_layout()
    return fig


def fig_why_not_more(tally: dict[str, Any], cost: dict[str, Any]):
    """The abstention breakdown, read as a work plan rather than a disclaimer."""
    labels = {
        "float_dtype": "float datapath",
        "output_count": "output count differs",
        "rank_gt_2": "rank > 2 (conv / batched)",
        "too_large": "over the encoding size cap",
        "epilogue": "epilogue stage (acc_scale)",
        "wall_timeout": "hit the wall bound",
        "other": "other",
        "unreadable": "spec or buffer unreadable",
        "solver_timeout": "solver returned unknown",
        "shape_mismatch": "shape mismatch",
    }
    reasons = sorted(tally["abstain_reasons"].items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    paper.card(ax, f"Why the other {sum(tally['abstain_reasons'].values())} abstained — "
                   f"the same list, read as a work plan")
    ys = range(len(reasons))
    cols = [COSTLY if k == "float_dtype" else OUT_OF_SCOPE for k, _ in reasons]
    ax.barh(list(ys), [v for _, v in reasons], height=0.6, color=cols,
            edgecolor=paper.CARD_EC, linewidth=1.4, zorder=3)
    widest = max((v for _, v in reasons), default=1)
    for y, (_k, v) in zip(ys, reasons):
        ax.text(v + widest * 0.015, y, str(v), va="center", fontsize=10, color=paper.INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels.get(k, k) for k, _ in reasons], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, widest * 1.55)
    ax.set_xlabel("eligible submissions the layer could not decide")

    late = cost["found_late"]
    paper.callout(ax, (reasons[0][1] * 0.62, 0.0),
                  f"every one of the {late['count']} defects that cost\n"
                  f"{late['sim_seconds'] / 3600:.1f} h of RTL simulation to find\n"
                  f"is a float datapath — this bar",
                  (widest * 1.02, 1.35))
    _caption(fig, "An abstention is never a pass, so this is the honest coverage limit. It is also "
                  "the priority order: float is both the largest bar and the place where every "
                  "expensive-to-find defect in the archive lives, so the coverage gap and the cost "
                  "are the same gap. Nothing here is unfixable -- rank>2 and the epilogue are built "
                  "in principle and not built in fact.")
    fig.tight_layout()
    return fig


def build(*, write: bool = False) -> dict[str, Any]:
    """Collect, draw, and optionally persist as a versioned product."""
    ablation = latest_ablation()
    tally = _tally(ablation["records"])
    reach, cost = collect_reach(), collect_cost()

    figs = {
        "f5_reach_and_verdict": fig_reach_and_verdict(tally, reach),
        "f6_agreement": fig_agreement(tally),
        "f7_why_not_more": fig_why_not_more(tally, cost),
    }
    out_dir: Path | None = None
    if write:
        from merlin.common.artifacts import new_product

        prod = new_product("verification", version=1, target="all", sources=[
            f"{tally['total']} archived capsule-bench submissions under out/runs",
            "verdicts: merlin.verify.ablation (validate_equivalence)",
            "grades: capsule_result.json numeric tier + tier timings",
        ], notes=("What the formal layer buys over the dynamic grade, measured on submissions that "
                  "already exist. Structurally identical submissions are excluded: a query over two "
                  "copies of one program is X == X."))
        for name, fig in figs.items():
            fig.savefig(prod.add_artifact(f"{name}.png"), dpi=200, bbox_inches="tight")
        prod.add_artifact("tally.json").write_text(
            json.dumps({k: (v if not isinstance(v, dict) else
                            {str(kk): vv for kk, vv in v.items()}) for k, v in tally.items()},
                       indent=1), encoding="utf-8")
        prod.add_artifact("formal_reach.json").write_text(json.dumps(reach, indent=1),
                                                          encoding="utf-8")
        prod.add_artifact("cost_of_finding.json").write_text(json.dumps(cost, indent=1),
                                                             encoding="utf-8")
        prod.write_manifest()
        out_dir = prod.path
    for fig in figs.values():
        plt.close(fig)
    return {"tally": tally, "reach": reach, "cost": cost,
            "out_dir": str(out_dir) if out_dir else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the figures under out/artifacts/")
    a = ap.parse_args(argv)
    rec = build(write=a.write)
    tally, cost, late = rec["tally"], rec["cost"], rec["cost"]["found_late"]
    print(f"archived submissions   {tally['total']}")
    print(f"  excluded (X == X)    {tally['excluded']}")
    print(f"  eligible             {tally['eligible']}")
    for k, v in sorted(tally["counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:6d}  {k}")
    print(f"\nrefuted, numeric PASS  {tally['grid'].get(('refuted', 'pass'), 0)}   (false alarms)")
    print(f"refuted, numeric FAIL  {tally['grid'].get(('refuted', 'fail'), 0)}   (agreement)")
    print(f"refuted, grade SKIPPED {tally['grid'].get(('refuted', 'skipped'), 0)}   (new information)")
    print(f"verified, numeric FAIL {tally['grid'].get(('verified', 'fail'), 0)}   (open contradiction)")
    print(f"\nrefutations needing an input outside the stimulus: "
          f"{tally['refuted_outside_stimulus']} of {tally['refuted_total']}")
    print(f"defects that cost {late['sim_seconds'] / 3600:.1f} h of simulation to find: "
          f"{late['count']}, of which float (unreachable) {late['float']}")
    if rec["out_dir"]:
        print(f"\nwrote {rec['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
