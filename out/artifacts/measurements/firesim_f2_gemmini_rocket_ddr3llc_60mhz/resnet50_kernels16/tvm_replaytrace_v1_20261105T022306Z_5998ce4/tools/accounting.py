#!/usr/bin/env python3
"""Tuning accounting: attempted/unique/built/successful trials, timeouts, occupied slot-seconds per campaign."""
import glob, json, os, sys
from pathlib import Path

W = Path("/work/exp/tvmfs/work")
rows, tot = [], {}
for kind in ("resnet", "kernel"):
    occ_total = 0.0
    for camp in sorted(p for p in (W / ("tune-" + kind)).iterdir() if p.is_dir()):
        t = [json.loads(l) for l in (camp / "trials.jsonl").read_text().splitlines() if l.strip()]
        sketches = {(r["task"], r["sketch"], json.dumps(r["trace"], sort_keys=True) if not isinstance(r["trace"], str) else r["trace"]) for r in t}
        built = [r for r in t if r.get("build_error") is None]
        ok = [r for r in t if r.get("cycles")]
        timeouts = [r for r in t if "timeout" in str(r.get("run_error", "")).lower() or "watchdog" in str(r.get("run_error", "")).lower()]
        # occupancy: every admitted job (3 baselines + all candidates, incl. failures) under the campaign tree
        occ = 0.0
        njobs = 0
        for f in camp.rglob("occupancy.json"):
            d = json.loads(f.read_text())
            njobs += 1
            occ += (d.get("finished_at", 0) - d.get("admitted_at", 0))
        base = json.loads((camp / "baselines.json").read_text())
        rec = json.loads((camp / "firesim-campaign.json").read_text())
        bc = {b["task"]: b["cycles"] for b in base}
        best = {}
        for r in ok:
            best[r["task"]] = min(best.get(r["task"], 10**30), r["cycles"])
        rows.append(dict(kind=kind, campaign=camp.name, budget=rec["proposal_budget"], attempted=len(t), unique=len(sketches),
                         built=len(built), successful=len(ok), timeouts=len(timeouts),
                         build_errors=len([r for r in t if r.get("build_error")]), run_errors=len([r for r in t if r.get("run_error")]),
                         admitted_jobs=njobs, occupied_slot_seconds=round(occ, 1), baseline_cycles=bc, best_cycles=best))
        occ_total += occ
    tot[kind] = round(occ_total, 1)
out = dict(per_campaign=rows, occupied_slot_seconds_total=tot,
           occupied_slot_hours_total={k: round(v / 3600, 3) for k, v in tot.items()},
           caps_slot_hours=dict(resnet=4, kernel=24))
Path(sys.argv[1] if len(sys.argv) > 1 else "/work/exp/tvmfs/f2/results/tuning-accounting.json").write_text(json.dumps(out, indent=1))
for r in rows:
    print(r["kind"], r["campaign"], "att", r["attempted"], "uniq", r["unique"], "built", r["built"], "ok", r["successful"], "to", r["timeouts"],
          "jobs", r["admitted_jobs"], "occ_s", r["occupied_slot_seconds"])
print(out["occupied_slot_hours_total"])
