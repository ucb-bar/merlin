#!/usr/bin/env python3
"""Generate the performance campaign's analysis: cost, attempts, outcome, and the gaps.

The library half (:mod:`merlin.agentreport.phase2_campaign`) touches no path. This script is the
caller: it walks the run roots, reads the ledger, collects the per-workload receipts, and writes
both the machine-readable analysis and the document.

    gen_phase2_analysis.py --target gemmini --out <dir>

Roots and the ledger location are arguments because they are launcher decisions. Nothing is
defaulted silently: a root that holds no telemetry is reported as such, since an empty cost table
reads as a campaign that spent nothing rather than one whose receipts are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.agentreport.phase2 import read_phase2
from merlin.agentreport.phase2_campaign import build_analysis
from merlin.common.paths import artifacts_dir, runs_dir
from merlin.perf.optimization_ledger import read_ledger


def _stage_dirs(roots):
    """Directories that carry phase-2 telemetry: a tool ledger, or a broker control tree."""
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if ((child / "agent" / "tools.jsonl").is_file()
                    or (child / "control").is_dir() or (child / "global_control").is_dir()):
                yield child


def _outcome_rows(target: str):
    """Per-workload outcome, read from each bundle's own receipts. Absent files are reported."""
    bench = artifacts_dir() / "perf-bench" / target
    rows = []
    for bundle in sorted(bench.glob("*_bundle_*")):
        row = {"bundle": bundle.name}
        for name, key in (("build_plan.json", "plan"), ("link_receipt.json", "link")):
            path = bundle / name
            if not path.is_file():
                row[key] = None
                row[f"{key}_absent_reason"] = f"{name} was not written for this bundle"
                continue
            try:
                row[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                row[key] = None
                row[f"{key}_absent_reason"] = f"{name} could not be read: {exc}"
        frame = bundle / "compiler" / "kernel.stack_frame.json"
        if frame.is_file():
            try:
                row["stack_frame"] = json.loads(frame.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                row["stack_frame"] = None
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ledger", type=Path, default=None,
                        help="the campaign's optimization ledger (JSON)")
    parser.add_argument("--root", type=Path, action="append", default=None,
                        help="a run root to walk; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    roots = args.root or [runs_dir() / args.target / "perf-bench" / "agent_stages",
                          artifacts_dir() / "perf-bench" / args.target]
    ledger_path = args.ledger
    if ledger_path is None:
        candidates = sorted((artifacts_dir() / "perf-bench" / args.target).glob(
            "*_ledger_*/optimization_ledger.json"))
        if not candidates:
            print("no optimization ledger found; pass --ledger", file=sys.stderr)
            return 2
        ledger_path = candidates[-1]

    stages = list(_stage_dirs(roots))
    calls, spans, points = [], 0, 0
    for stage in stages:
        facts = read_phase2(stage)
        calls.extend(facts.broker_calls)
        spans += len(facts.spanset.spans)
        points += facts.n_point_events

    analysis = build_analysis(
        target=args.target, stages=len(stages), tool_spans=spans, point_events=points,
        broker_calls=calls, ledger=read_ledger(ledger_path, target=args.target),
        outcomes=_outcome_rows(args.target))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    document = analysis.to_dict()
    document["inputs"] = {"roots": [str(r) for r in roots], "ledger": str(ledger_path),
                          "stages": [str(s) for s in stages]}
    (out / "campaign_analysis.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")

    budget = analysis.budget
    print(f"stages with telemetry     {analysis.stages}")
    print(f"tool spans                {analysis.tool_spans:,}  "
          f"(+{analysis.point_events:,} point events excluded)")
    print(f"brokered calls            {budget.calls:,}  "
          f"wall {budget.wall_seconds/3600:.2f} h")
    print(f"  refused                 {budget.refused:,} calls "
          f"({100*budget.refused/max(budget.calls,1):.1f}%), "
          f"{budget.refused_seconds/3600:.2f} h "
          f"({100*budget.refused_seconds/max(budget.wall_seconds,1):.1f}% of wall)")
    for tier, row in budget.by_tier().items():
        print(f"  tier {tier:8s}         {int(row['calls']):5d} calls  "
              f"{row['wall_seconds']/3600:6.2f} h  {int(row['refused']):4d} refused")
    print(f"attempts                  {analysis.attempts.total}  "
          f"{analysis.attempts.by_verdict}")
    print(f"  scopes reached          {analysis.attempts.scopes_reached} of "
          f"{len(analysis.attempts.by_scope)}")
    print(f"  distinct instruments    {len(analysis.attempts.instruments)}")
    print(f"  integrity problems      {len(analysis.attempts.integrity_problems)}")
    print(f"availability score        {document['availability_score']:.3f}")
    for name, status in sorted(document["availability"].items()):
        if status["kind"] == "unavailable":
            print(f"  GAP {name}: {status['reason'][:110]}")
    print(f"\nwrote {out / 'campaign_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
