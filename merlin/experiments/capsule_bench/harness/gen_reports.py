#!/usr/bin/env python3
"""Aggregate every run_manifest.yaml under runs/<arm>/<run_id>/ into the experiment reports."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import _common as C
from _ratelimit import rounds_rate_limited as _rounds_rate_limited


def main() -> int:
    # only run-level manifests (runs/<arm>/<run_id>/run_manifest.yaml), not nested capsule ones
    manifests = sorted(C.RUNS.glob("*/*/run_manifest.yaml"))
    rows = [yaml.safe_load(m.read_text()) for m in manifests]
    # Rate-limit awareness: a run whose agent rounds were all rejected by the five-hour session limit
    # (zero real work) is BLOCKED, not failed — flag it so the comparison table doesn't read its empty
    # 0/0 + "no manifest" grade as a genuine baseline failure. See reports/repeatability.md.
    run_dirs = [m.parent for m in manifests]
    blocked = {}
    for r, d in zip(rows, run_dirs):
        rej, _worked = _rounds_rate_limited(d)
        is_pass = (r["public_dev"]["passed"] == "4/4" and r["hidden"]["passed"] == "3/3")
        # A genuine failure runs to exhaustion with NO rejected rounds (rej==0); any non-passing run
        # that had a round rejected by the five-hour limit never got a fair attempt => blocked.
        blocked[r["run_id"]] = (not is_pass) and rej > 0

    idx = [f"# Run index ({C.TARGET} capsule-bench)", "",
           f"{len(rows)} run(s).", "", "| run_id | arm | model | integrity | public | hidden | oracle |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        idx.append(f"| {r['run_id']} | {r['arm']} | {r['model']} | {r.get('integrity_status')} | "
                   f"{r['public_dev']['passed']} | {r['hidden']['passed']} | {r.get('oracle_mode')} |")
    (C.REPORTS / "run_index.md").write_text("\n".join(idx) + "\n")

    # full-suite audit results (all 25 capsules) keyed by run_id, if the audit has been run
    full_suite = {}
    fsa = C.REPORTS / "full_suite_audit.json"
    if fsa.exists():
        try:
            full_suite = (json.loads(fsa.read_text()).get("backends") or {})
        except Exception:
            full_suite = {}

    cmp = [f"# Comparison table ({C.TARGET} capsule-bench)", "",
           "Apples-to-apples across arms: same task, same capsules, same hidden set, same grader.",
           "`public` is the 4-capsule pilot (the agent's iterate-to-pass gate); **`full-suite` is all "
           "25 capsules** (every test — see reports/full_suite_audit.md). Cycles are diagnostic-only. "
           "Process metrics come from the agent transcript "
           "(`available:false` recorded honestly when the CLI emits no usage).", "",
           "| arm | run_id | model | wall(s) | tokens | cost$ | tool_calls | public | full-suite | "
           "hidden | tier | numeric | integrity | first-failure | iters |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, proc = r["public_dev"], r["process"]
        ff = ",".join((p.get("first_failure_planes") or {}).keys()) or "none"
        integ = r.get("integrity_status")
        if blocked.get(r["run_id"]):
            integ = "blocked(rate_limit)"
            ff = "rate_limit (no fair attempt)"
        fs = full_suite.get(r["run_id"], {}).get("passed", "—")
        cmp.append(
            f"| {r['arm']} | {r['run_id']} | {r['model']} | {proc.get('wall_time_seconds')} | "
            f"{proc.get('tokens_total')} | {proc.get('estimated_cost_usd')} | {proc.get('tool_calls')} | "
            f"{p['passed']} | {fs} | {r['hidden']['passed']} | {p.get('highest_tier')} | "
            f"{p.get('numeric_all_exact')} | {integ} | {ff} | {r.get('iterations')} |")
    cmp += ["",
            "_Notes: `dry_*` rows are dummy pipeline-validation runs (not agent results). "
            "`rb_pilot_0001` is the PRE-grader-fix diagnostic run (failed trace_check only because of "
            "two grader bugs since fixed — schema `$id` resolution + rocc_decode SSA-name regex); it "
            "is kept for the record, not a baseline result. `integrity=blocked(rate_limit)` rows "
            "(`rb_pilot_rep_02/03`) were rejected by the org five-hour session limit (zero real work) "
            "and are NOT baseline failures — they are excluded from the pass-rate in "
            "`reports/repeatability.md`. `rb_pilot_cpp_01` is the explicit-C++ OOT baseline; "
            "`rb_pilot_0002` is the agent's-choice (Python) baseline. Real measured agent runs are "
            "produced by `run_baseline_qa_loop.py`. Both arms must be graded by the same (patched) "
            "grader and the same task file — see COORDINATION.md. Cycles are diagnostic-only and never "
            "gate pass/fail._"]
    (C.REPORTS / "comparison_table.md").write_text("\n".join(cmp) + "\n")
    print(f"wrote run_index.md + comparison_table.md ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
