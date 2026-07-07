#!/usr/bin/env python3
"""Repeatability sweep: run the measured baseline QA loop N times (fresh agent each) and aggregate
the distribution (pass-rate, rounds-to-converge, wall, cost, tokens, tool-calls) so the baseline is a
DISTRIBUTION, not n=1. LLM agents are stochastic; one converged run is a single sample.

Each repeat is an independent run_id (rb_pilot_rep_<k>) via run_baseline_qa_loop.py with identical
flags. Runs are SEQUENTIAL (the oracle/verilator is heavy; parallel runs would contend). Writes
reports/repeatability.md + repeatability.json.

Usage:
  run_repeatability.py --n 3 [--arm raw_baseline] [--model claude-opus-4-8] [--max-rounds 8]
                       [--start-index 1] [--prefix rb_pilot_rep]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

import _common as C

DRIVER = C.EXP / "scripts" / "run_baseline_qa_loop.py"


def _load(run_dir: Path) -> dict | None:
    mf = run_dir / "run_manifest.yaml"
    qs = run_dir / "qa_loop_summary.yaml"
    if not mf.exists():
        return None
    m = yaml.safe_load(mf.read_text())
    q = yaml.safe_load(qs.read_text()) if qs.exists() else {}
    return {"run_id": m["run_id"],
            "public": m["public_dev"]["passed"], "hidden": m["hidden"]["passed"],
            "tier": m["public_dev"]["highest_tier"],
            "numeric_all_exact": m["public_dev"]["numeric_all_exact"],
            "integrity": m.get("integrity_status"),
            "converged": q.get("converged"), "n_rounds": q.get("n_rounds"),
            "wall_s": (m.get("process") or {}).get("wall_time_seconds"),
            "cost_usd": (m.get("process") or {}).get("estimated_cost_usd"),
            "tokens_total": (m.get("process") or {}).get("tokens_total"),
            "tool_calls": (m.get("process") or {}).get("tool_calls"),
            "answer_access_clean": all(r.get("answer_access_clean", True) for r in q.get("rounds", []))
                                   and ((q.get("finalize") or {}).get("answer_access_clean", True))}


def _agg(vals: list) -> dict:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {}
    return {"min": min(vals), "max": max(vals), "mean": round(statistics.mean(vals), 2),
            "median": statistics.median(vals),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--arm", default="raw_baseline")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=2700)
    ap.add_argument("--qa-timeout", type=int, default=1200)
    ap.add_argument("--sandbox", default="none")
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--prefix", default="rb_pilot_rep")
    # Rate-limit awareness: each run sleeps until a five-hour window resets and retries, so the sweep
    # spans windows unattended. --resume lets an interrupted run continue rather than be skipped.
    ap.add_argument("--max-rate-limit-waits", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--account-config-dir", default="",
                    help="CLAUDE_CONFIG_DIR for a second subscription account (separate org budget)")
    a = ap.parse_args(argv)

    results = []
    for k in range(a.start_index, a.start_index + a.n):
        run_id = f"{a.prefix}_{k:02d}"
        run_dir = C.RUNS / a.arm / run_id
        print(f"\n========== REPEAT {k} -> {run_id} ==========", flush=True)
        # Sweep-level resume: a run with a written run_manifest.yaml is fully graded — skip it (don't
        # re-spend on a finished run); otherwise (re)launch with --resume so a mid-flight run continues
        # from its qa_loop_state checkpoint rather than restarting from round 0.
        if (run_dir / "run_manifest.yaml").exists():
            print(f"  -> already complete (run_manifest.yaml present) — skipping", flush=True)
        else:
            cmd = [sys.executable, str(DRIVER), "--run-id", run_id, "--arm", a.arm,
                   "--model", a.model, "--effort", a.effort, "--max-rounds", str(a.max_rounds),
                   "--round-timeout", str(a.round_timeout), "--qa-timeout", str(a.qa_timeout),
                   "--sandbox", a.sandbox, "--max-rate-limit-waits", str(a.max_rate_limit_waits)]
            if a.account_config_dir:
                cmd += ["--account-config-dir", a.account_config_dir]
            if a.resume or run_dir.exists():
                cmd.append("--resume")  # an existing-but-unfinished run_dir must resume, not be refused
            subprocess.run(cmd, cwd=str(C.EXP / "scripts"))
        r = _load(C.RUNS / a.arm / run_id)
        if r:
            results.append(r)
            print(f"  -> public={r['public']} hidden={r['hidden']} tier={r['tier']} "
                  f"rounds={r['n_rounds']} wall={r['wall_s']}s cost=${r['cost_usd']} "
                  f"clean={r['answer_access_clean']}", flush=True)

    n_full = sum(1 for r in results if r["public"] == "4/4" and r["hidden"] == "3/3")
    agg = {"public_4of4_AND_hidden_3of3": f"{n_full}/{len(results)}",
           "all_integrity_clean": all(r["integrity"] == "clean" for r in results),
           "all_answer_access_clean": all(r["answer_access_clean"] for r in results),
           "all_numeric_exact": all(r["numeric_all_exact"] for r in results),
           "rounds_to_converge": _agg([r["n_rounds"] for r in results]),
           "wall_seconds": _agg([r["wall_s"] for r in results]),
           "cost_usd": _agg([r["cost_usd"] for r in results]),
           "tool_calls": _agg([r["tool_calls"] for r in results]),
           "tokens_total": _agg([r["tokens_total"] for r in results])}
    out = {"arm": a.arm, "model": a.model, "n": len(results), "runs": results, "aggregate": agg}
    (C.REPORTS / "repeatability.json").write_text(json.dumps(out, indent=2))

    md = [f"# Repeatability — {a.arm} ({a.model}), n={len(results)}", "",
          f"- **Full pass (public 4/4 AND hidden 3/3): {agg['public_4of4_AND_hidden_3of3']}**",
          f"- integrity clean (all): {agg['all_integrity_clean']}; "
          f"answer-access clean (all): {agg['all_answer_access_clean']}; "
          f"numeric exact (all): {agg['all_numeric_exact']}",
          f"- rounds-to-converge: {agg['rounds_to_converge']}",
          f"- wall(s): {agg['wall_seconds']}", f"- cost$: {agg['cost_usd']}",
          f"- tool_calls: {agg['tool_calls']}", "",
          "| run_id | public | hidden | tier | rounds | wall(s) | cost$ | tools | clean |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['run_id']} | {r['public']} | {r['hidden']} | {r['tier']} | {r['n_rounds']} | "
                  f"{r['wall_s']} | {r['cost_usd']} | {r['tool_calls']} | {r['answer_access_clean']} |")
    (C.REPORTS / "repeatability.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote reports/repeatability.{{md,json}} — full pass {agg['public_4of4_AND_hidden_3of3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
