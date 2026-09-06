#!/usr/bin/env python3
"""Write the report's prose from ``run_facts.json``, so no figure and no sentence can disagree.

Every number below is read from the facts file at render time. Nothing is typed in.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from merlin.common.paths import artifacts_dir      # noqa: E402

ARM_LABEL = {"arm1": "arm 1 · raw C++", "arm2": "arm 2 · C++ & Merlin infra",
             "arm3": "arm 3 · Merlin Python tooling", "arm4": "arm 4 · Merlin & CIRCT tooling",
             "eqsat": "e-graph seam", "UNKNOWN": "unclassified"}


def _money(f):
    if f.get("cost_usd") is not None:
        return f"${f['cost_usd']:,.0f}"
    if f.get("notional_usd") is not None:
        return f"~${f['notional_usd']:,.0f}"
    return "unpriced"


def write(facts: list[dict], out: Path) -> Path:
    sel = [f for f in facts if f.get("selected")]
    p1 = [f for f in facts if f["phase"] == "phase1"]
    graded = [f for f in facts if f.get("passed") is not None and f.get("capsules")]
    L = []
    A = L.append

    A("# Agentic compiler-generation benchmark — run report\n")
    A("Generated from `run_facts.json`. Every number here is read from that file at render time; "
      "none is typed in, so no sentence can drift from the run that produced it.\n")

    A("## What is on disk\n")
    A(f"- **{len(facts)} run directories** indexed across the configured roots "
      f"({len(p1)} functional, {len(facts) - len(p1)} performance).")
    A(f"- **{len(graded)}** produced a graded verdict on a sized corpus.")
    A(f"- **{len(sel)}** are selected for this report — the best run in each target × arm cell, plus "
      f"every member of a like-for-like ladder.\n")

    tot_tok = sum(f.get("total_tokens") or 0 for f in graded)
    tot_h = sum((f.get("active_wall_s") or 0) for f in graded) / 3600.0
    met = sum(f["cost_usd"] for f in graded if f.get("cost_usd") is not None)
    notional = sum(f["notional_usd"] for f in graded if f.get("notional_usd") is not None)
    unpriced = sum(1 for f in graded if f["cost_kind"] == "unpriced")
    calls = sum(f.get("tool_calls") or 0 for f in graded)
    A(f"Across the graded runs: **{tot_h:,.0f} active hours**, **{tot_tok / 1e9:.2f} B tokens**, "
      f"**{calls:,} tool calls**, **${met:,.0f} metered** and **~${notional:,.0f} notional** "
      f"(a subscription seat is not billed per token, so the two are never added). "
      f"{unpriced} graded run(s) carry no dollar figure at all.\n")

    # The read/write split only exists on runs that recorded it. Runs that logged only the summed
    # `tokens_cached` have it folded into reads, so a global cache-write share would read as "we
    # never write cache" when it means "most runs could not tell us". Split the population.
    split = [f for f in graded
             if f.get("availability", {}).get("token_split", {}).get("kind") == "measured"]
    ci = sum(f.get("cache_read_tokens") or 0 for f in graded)
    inp = sum(f.get("input_tokens") or 0 for f in graded)
    outp = sum(f.get("output_tokens") or 0 for f in graded)
    tot = max(ci + inp + outp + sum(f.get("cache_creation_tokens") or 0 for f in graded), 1)
    A(f"Token mix over all graded runs: **{100 * ci / tot:.0f}% cached input**, "
      f"{100 * inp / tot:.0f}% fresh input, {100 * outp / tot:.1f}% output. That ratio is the "
      f"economics of the programme — the agent re-reads a large fixed context every turn.\n")
    if split:
        sr = sum(f.get("cache_read_tokens") or 0 for f in split)
        sw = sum(f.get("cache_creation_tokens") or 0 for f in split)
        A(f"Cache reads and cache writes are billed roughly an order of magnitude apart, and only "
          f"**{len(split)} of {len(graded)}** graded runs recorded them separately; the rest logged "
          f"only their sum, which cannot be undone afterwards. Over the runs that did record it, "
          f"writes are **{100 * sw / max(sr + sw, 1):.1f}%** of cached input. The figures mark the "
          f"runs where the split is unavailable rather than assuming it is zero.\n")

    A("## Runs per target and arm\n")
    cells = Counter((f["target"], f["arm"]) for f in graded if f["phase"] == "phase1")
    targets = sorted({t for t, _ in cells})
    arms = ["arm1", "arm2", "arm3", "arm4"]
    A("| target | " + " | ".join(ARM_LABEL[a] for a in arms) + " |")
    A("|---|" + "---|" * len(arms))
    for t in targets:
        A(f"| {t} | " + " | ".join(str(cells.get((t, a), 0)) for a in arms) + " |")
    A("")

    A("## The ladders — the only like-for-like comparisons\n")
    ladders = defaultdict(list)
    for f in sel:
        if f.get("ladder"):
            ladders[f["ladder"]].append(f)
    if not ladders:
        A("_No tag-matched ladder was found._\n")
    for key in sorted(ladders, key=lambda k: -max(x["capsules"] or 0 for x in ladders[k])):
        members = sorted(ladders[key], key=lambda f: f["arm"])
        corpora = sorted({f["capsules"] for f in members})
        target, _, tag = key.split("/")
        A(f"**{target} · `{tag}`** — {corpora[0] if len(corpora) == 1 else corpora} capsules, "
          f"model `{members[0]['model']}`.\n")
        A("| arm | passed | active | tokens | cost | tool calls |")
        A("|---|---|---|---|---|---|")
        for f in members:
            hours = (f.get("active_wall_s") or 0) / 3600.0
            A(f"| {ARM_LABEL[f['arm']]} | {f['passed']}/{f['capsules']} | "
              f"{hours:.2f} h" + ("" if hours else " *(no clock recorded)*") + " | "
              f"{(f.get('total_tokens') or 0) / 1e6:.1f} M | {_money(f)} | {f.get('tool_calls') or 0} |")
        A("")

    A("## Where the time goes\n")
    conc = [f for f in facts
            if f.get("availability", {}).get("concurrency", {}).get("kind") in ("measured", "derived")
            and (f.get("span_wall_s") or 0) > 60]
    by_phase = defaultdict(list)
    for f in conc:
        by_phase[f["phase"]].append(f)
    for phase in sorted(by_phase):
        rows = by_phase[phase]
        over = [f for f in rows if f["overlap_share"] > 0]
        A(f"- **{phase}**: {len(over)} of {len(rows)} run(s) show two or more tool calls in flight at "
          f"once" + (f"; median overlap among those is "
                     f"{statistics.median([f['overlap_share'] for f in over]):.0%} of the wall, peak "
                     f"{max(f['max_concurrent'] for f in over)} simultaneous calls." if over else "."))
    A("")
    A("Overlap is only reported where it survives a flush check: the readings are recomputed with "
      "spans too short to have a trustworthy duration removed, and a run whose answer moves is "
      "refused rather than published. An overlap that depends on those spans was measuring how the "
      "harness read its log, not what the agent did.\n")

    A("## What a grade costs, per capsule\n")
    tiers = defaultdict(list)
    for f in facts:
        for k, s in (f.get("tier_cost") or {}).items():
            tier, status = k.split("/", 1)
            if status == "pass" and s.get("median_active_s") is not None:
                tiers[(f["phase"], tier)].append(s["median_active_s"])
    A("| lane | tier | runs | median seconds per passing capsule |")
    A("|---|---|---|---|")
    for (phase, tier) in sorted(tiers):
        vals = tiers[(phase, tier)]
        A(f"| {phase} | {tier} | {len(vals)} | {statistics.median(vals):.3g} |")
    A("")
    A("Passing capsules only. A failing capsule aborts in hundredths of a second, so a median over "
      "both populations describes the pass rate rather than the cost. A carried certificate records "
      "no duration by design and is not counted.\n")

    p2 = [f for f in facts if f["phase"] == "phase2" and f.get("broker_actions")]
    if p2:
        A("## Phase 1 versus phase 2\n")
        actions = sorted({a for f in p2 for a in f["broker_actions"]})
        secs = Counter()
        callsn = Counter()
        for f in p2:
            for a, d in (f.get("broker_totals") or {}).items():
                secs[a] += d["seconds"]
                callsn[a] += d["calls"]
        A(f"Phase 1 gives the agent a shell and a simulator it drives itself. Phase 2 gives it a "
          f"closed brokered action set — **{len(actions)} actions**, recorded per run rather than "
          f"assumed, because the set is derived from the frozen candidate's own manifest.\n")
        A("| action | calls | minutes |")
        A("|---|---|---|")
        for a, sec in secs.most_common(8):
            A(f"| `{a}` | {callsn[a]} | {sec / 60:.1f} |")
        A("")

    A("## What the data cannot say\n")
    reasons = Counter()
    for f in sel:
        for name, st in (f.get("availability") or {}).items():
            if st.get("kind") == "unavailable":
                reasons[st.get("reason", "")[:150]] += 1
    for reason, n in reasons.most_common(8):
        A(f"- **{n} selected run(s)** — {reason}")
    A("")
    conflicts = [f for f in facts if f.get("arm_conflict")]
    if conflicts:
        A(f"{len(conflicts)} run(s) carry a bundle and a run-id that disagree about which arm they "
          f"are. The bundle decides, because an arm *is* its grant set, but the disagreement is "
          f"recorded rather than resolved silently:")
        for f in conflicts:
            A(f"- `{f['target']}/{f['run_id']}` — {f['arm_conflict']} (reported as {f['arm']})")
        A("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", type=Path, default=artifacts_dir() / "agentic-report" / "run_facts.json")
    ap.add_argument("--out", type=Path, default=artifacts_dir() / "agentic-report" / "report.md")
    a = ap.parse_args(argv)
    path = write(json.loads(a.facts.read_text()), a.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
